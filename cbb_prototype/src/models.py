"""Steps 3 + 4 - models and honest walk-forward validation.

Walk-forward = expanding window, never shuffled:
    fold 1: train on games before Jan 1,  test Jan
    fold 2: train on games before Feb 1,  test Feb
    fold 3: train on games before Mar 1,  test Mar-Apr (incl. conference/NCAA tournaments)
Every reported number is on those held-out test games only. Hyper-parameters are fixed, conservative
defaults - they were NOT tuned on the test folds.
"""
import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from .config import C_AQUA, C_BLUE, C_GRID, C_INK, C_MUTED, C_ORANGE, OUT, PROC, SEED, update_results  # noqa: E402
from .features import FEATURES, LABELS  # noqa: E402

FOLDS = [("2024-01-01", "2024-02-01"), ("2024-02-01", "2024-03-01"), ("2024-03-01", "2024-05-01")]
TARGETS = ["margin", "total", "h1_margin", "h1_total"]
ML_MODELS = {"A": "Baseline A (always home)", "B": "Baseline B (rating formula)",
             "tree": "Decision tree (depth 3)", "gbdt_raw": "LightGBM (uncalibrated)",
             "gbdt": "LightGBM (Platt-calibrated)"}
N_BOOT = 2000


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_MUTED)


def _clf():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=SEED, verbose=-1, n_jobs=1, deterministic=True, force_row_wise=True)


def _reg():
    return lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                             random_state=SEED, verbose=-1, n_jobs=1, deterministic=True, force_row_wise=True)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def walk_forward(feat: pd.DataFrame):
    """Fit on each expanding training window and predict the next unseen month."""
    feat = feat.sort_values(["date", "game_id"]).reset_index(drop=True)
    parts, artifacts = [], []
    for k, (start, end) in enumerate(FOLDS, 1):
        tr = feat[feat["date"] < start]
        te = feat[(feat["date"] >= start) & (feat["date"] < end)].copy()
        te["fold"] = k

        # --- win probability: LightGBM on first 75% of training (by date), Platt scaling on the last 25% ---
        cut = int(len(tr) * 0.75)
        fit, cal = tr.iloc[:cut], tr.iloc[cut:]
        clf = _clf().fit(fit[FEATURES], fit["home_win"])
        platt = LogisticRegression(C=1e6).fit(_logit(clf.predict_proba(cal[FEATURES])[:, 1]), cal["home_win"])
        te["p_gbdt_raw"] = clf.predict_proba(te[FEATURES])[:, 1]
        te["p_gbdt"] = platt.predict_proba(_logit(te["p_gbdt_raw"].values))[:, 1]

        # --- small interpretable tree ---
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED)
        tree.fit(tr[FEATURES], tr["home_win"])
        te["p_tree"] = tree.predict_proba(te[FEATURES])[:, 1]

        # --- baselines ---
        te["p_A"] = te["lg_homewin"]                       # constant: as-of home win rate
        sigma = float((tr["margin"] - tr["b_margin"]).std())  # only free parameter of Baseline B
        te["p_B"] = norm.cdf(te["b_margin"] / sigma)

        # --- regressors (margin, total, 1st-half margin, 1st-half total) ---
        regs = {}
        for t in TARGETS:
            sub = tr.dropna(subset=[t])
            regs[t] = _reg().fit(sub[FEATURES], sub[t])
            te[f"pred_{t}_GBDT"] = regs[t].predict(te[FEATURES])
            te[f"pred_{t}_A"] = te[f"a_{t}"]
            te[f"pred_{t}_B"] = te[f"b_{t}"]

        parts.append(te)
        artifacts.append({"fold": k, "start": start, "end": end, "n_train": len(tr), "n_test": len(te),
                          "clf": clf, "platt": platt, "tree": tree, "regs": regs, "sigma_B": sigma})
    return pd.concat(parts, ignore_index=True), artifacts


# ----------------------------------------------------------------------------- metrics
def _p(p):
    return np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)


def ml_metrics(df: pd.DataFrame) -> dict:
    y = df["home_win"].values
    out = {}
    for key, name in ML_MODELS.items():
        p = df[f"p_{key}"].values
        pick_home = np.ones(len(df), bool) if key == "A" else p > 0.5
        out[key] = {"name": name, "accuracy": float((pick_home == (y == 1)).mean()),
                    "log_loss": float(log_loss(y, _p(p), labels=[0, 1])),
                    "brier": float(brier_score_loss(y, _p(p)))}
    return out


def mae_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for t in TARGETS:
        d = df.dropna(subset=[t])
        out[t] = {m: float((d[t] - d[f"pred_{t}_{m}"]).abs().mean()) for m in ("A", "B", "GBDT")}
        out[t]["n"] = len(d)
    return out


def paired_bootstrap(df: pd.DataFrame, seed=SEED) -> dict:
    """95% intervals for GBDT minus Baseline B, resampling test games (paired).
    Games are treated as independent, which is only approximate (teams repeat, same-day games correlate)."""
    rng = np.random.default_rng(seed)
    y = df["home_win"].values
    out = {}

    def ci(diff, label, lower_is_better):
        idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
        means = diff[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[label] = {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
                      "excludes_zero": bool(lo > 0 or hi < 0), "lower_is_better": lower_is_better}

    ll = lambda p: -(y * np.log(_p(p)) + (1 - y) * np.log(1 - _p(p)))  # noqa: E731
    ci(ll(df["p_gbdt"]) - ll(df["p_B"]), "log_loss_gbdt_minus_B", True)
    ci(((df["p_gbdt"] > 0.5) == (y == 1)).astype(float).values - ((df["p_B"] > 0.5) == (y == 1)).astype(float).values,
       "accuracy_gbdt_minus_B", False)
    for t in TARGETS:
        d = df.dropna(subset=[t])
        ci((d[t] - d[f"pred_{t}_GBDT"]).abs().values - (d[t] - d[f"pred_{t}_B"]).abs().values,
           f"mae_{t}_gbdt_minus_B", True)
    # accuracy interval for the main model alone
    acc = ((df["p_gbdt"] > 0.5) == (y == 1)).astype(float).values
    idx = rng.integers(0, len(acc), size=(N_BOOT, len(acc)))
    out["accuracy_gbdt_alone"] = {"value": float(acc.mean()), "ci95": [float(x) for x in np.percentile(acc[idx].mean(1), [2.5, 97.5])]}
    return out


def calibration_table(p, y, bins=10):
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ece = float((t["n"] * (t["p"] - t["y"]).abs()).sum() / t["n"].sum())
    return t.reset_index(drop=True), ece


# ----------------------------------------------------------------------------- explanations
def tree_rules(tree, names=FEATURES) -> list:
    """Turn a fitted decision tree into plain-English IF/THEN lines."""
    t, lines = tree.tree_, []

    def describe(bounds):
        """bounds: {feature: [lo, hi]} -> merged, readable conditions (no repeated splits on one feature)."""
        parts = []
        for f, (lo, hi) in bounds.items():
            unit = "" if f == "neutral" else " (home minus away)"
            if lo is not None and hi is not None:
                parts.append(f"{LABELS[f]}{unit} is between {lo:.2f} and {hi:.2f}")
            elif hi is not None:
                parts.append(f"{LABELS[f]}{unit} is at most {hi:.2f}")
            else:
                parts.append(f"{LABELS[f]}{unit} is above {lo:.2f}")
        return " AND ".join(parts) if parts else "always"

    def rec(node, bounds):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            lines.append(f"IF {describe(bounds)} THEN home team wins "
                         f"{100 * v[1] / v.sum():.0f}% of the time (n={int(t.n_node_samples[node])} training games)")
            return
        f, th = names[t.feature[node]], t.threshold[node]
        lo, hi = bounds.get(f, [None, None])
        left = {**bounds, f: [lo, th if hi is None else min(hi, th)]}    # feature <= threshold
        right = {**bounds, f: [th if lo is None else max(lo, th), hi]}   # feature >  threshold
        rec(t.children_left[node], left)
        rec(t.children_right[node], right)

    rec(0, {})
    return lines


# ----------------------------------------------------------------------------- charts
def chart_calibration(df, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=1, label="Perfect calibration")
    for col, label, c in (("p_gbdt_raw", "LightGBM (uncalibrated)", C_ORANGE), ("p_gbdt", "LightGBM (Platt-calibrated)", C_BLUE)):
        t, ece = calibration_table(df[col].values, df["home_win"].values)
        ax.plot(t["p"], t["y"], "o-", color=c, lw=2, ms=6, mec="#fcfcfb", mew=1.5, label=f"{label}, ECE {ece:.3f}")
    ax.set_xlabel("Predicted home-win probability", color=C_MUTED)
    ax.set_ylabel("Observed home-win rate", color=C_MUTED)
    ax.set_title("Calibration on held-out games (10 equal-count bins)", loc="left", color=C_INK, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_importance(imp: pd.Series, path):
    top = imp.sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.barh([LABELS[i] for i in top.index], top.values, color=C_BLUE, height=0.6)
    ax.set_xlabel("Share of total split gain (%)", color=C_MUTED)
    ax.set_title("Win-probability model: top 10 features (final fold)", loc="left", color=C_INK, fontsize=11)
    style_axes(ax); ax.grid(axis="x", color=C_GRID); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_mae(mae: dict, path):
    labels = {"margin": "Full-game margin", "total": "Full-game total", "h1_margin": "1st-half margin", "h1_total": "1st-half total"}
    x = np.arange(len(TARGETS)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for j, (m, name, c) in enumerate((("A", "Baseline A", C_ORANGE), ("B", "Baseline B", C_AQUA), ("GBDT", "LightGBM", C_BLUE))):
        vals = [mae[t][m] for t in TARGETS]
        bars = ax.bar(x + (j - 1) * w, vals, w * 0.92, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels([labels[t] for t in TARGETS], color=C_INK)
    ax.set_ylabel("Mean absolute error (points)", color=C_MUTED)
    ax.set_title("Held-out error by target (lower is better)", loc="left", color=C_INK, fontsize=11)
    style_axes(ax); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------------- orchestration
def run():
    feat = pd.read_parquet(PROC / "features.parquet")
    preds, arts = walk_forward(feat)
    preds.to_parquet(PROC / "predictions.parquet")
    joblib.dump({"folds": arts, "features": FEATURES}, PROC / "artifacts.joblib")

    ml, mae, boot = ml_metrics(preds), mae_metrics(preds), paired_bootstrap(preds)
    per_fold = []
    for k in sorted(preds["fold"].unique()):
        d = preds[preds["fold"] == k]
        m, e = ml_metrics(d), mae_metrics(d)
        per_fold.append({"fold": int(k), "n": len(d), "start": FOLDS[k - 1][0], "end": FOLDS[k - 1][1],
                         "acc_gbdt": m["gbdt"]["accuracy"], "acc_B": m["B"]["accuracy"], "acc_A": m["A"]["accuracy"],
                         "ll_gbdt": m["gbdt"]["log_loss"], "ll_B": m["B"]["log_loss"],
                         "mae_margin_gbdt": e["margin"]["GBDT"], "mae_margin_B": e["margin"]["B"]})

    # feature importance + plain-English tree rules from the final fold
    last = arts[-1]
    gain = pd.Series(last["clf"].booster_.feature_importance(importance_type="gain"), index=FEATURES)
    imp = 100 * gain / gain.sum()
    gain_m = pd.Series(last["regs"]["margin"].booster_.feature_importance(importance_type="gain"), index=FEATURES)
    imp_m = 100 * gain_m / gain_m.sum()
    rules = tree_rules(last["tree"])
    (OUT / "decision_tree_rules.txt").write_text(
        f"Decision tree (max_depth=3) fitted on {last['n_train']} games before {last['start']}.\n"
        "'Edge' = home team minus away team; efficiency in points per 100 possessions.\n\n" + "\n".join(rules) + "\n")

    _, ece_raw = calibration_table(preds["p_gbdt_raw"].values, preds["home_win"].values)
    _, ece_cal = calibration_table(preds["p_gbdt"].values, preds["home_win"].values)
    chart_calibration(preds, OUT / "calibration.png")
    chart_importance(imp, OUT / "feature_importance.png")
    chart_mae(mae, OUT / "mae_comparison.png")

    update_results("metrics", {
        "n_test_games": len(preds), "test_start": preds["date"].min(), "test_end": preds["date"].max(),
        "folds": [{"fold": a["fold"], "train_before": a["start"], "test_until": a["end"],
                   "n_train": a["n_train"], "n_test": a["n_test"]} for a in arts],
        "home_win_rate_test": float(preds["home_win"].mean()),
        "ml": ml, "mae": mae, "bootstrap": boot, "per_fold": per_fold,
        "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal,
        "top_features_win_model": {LABELS[i]: float(v) for i, v in imp.sort_values(ascending=False).head(8).items()},
        "top_features_margin_model": {LABELS[i]: float(v) for i, v in imp_m.sort_values(ascending=False).head(8).items()},
        "tree_rules": rules,
        "hyperparameters": "fixed, conservative LightGBM settings (num_leaves=8, lr=0.03, 200-250 trees); not tuned on test folds",
    })
    return preds, arts


if __name__ == "__main__":
    run()
