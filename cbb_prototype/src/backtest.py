"""Step 5 - ROI backtest.

CASE A (real closing lines supplied at data/raw/lines.csv): ATS / over-under hit rates + flat 1-unit ROI.
          The edge threshold is chosen on a VALIDATION fold (Jan) and evaluated on the later folds only.
CASE B (no real lines - what happened in this run): ROI is NOT reported as real. The only simulation is
          against a labelled PROXY line (Baseline B's own number) and is stamped "PROXY - NOT REAL MARKET RESULTS".
          It measures disagreement with a simple formula, not an edge against a sportsbook.

Bets are flat 1 unit at -110 (win +0.909, loss -1, push 0). Break-even win rate = 52.38%.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import C_BLUE, C_INK, C_MUTED, C_ORANGE, OUT, PROC, SEED, update_results  # noqa: E402
from .data import load_lines  # noqa: E402
from .models import N_BOOT, style_axes  # noqa: E402

PAYOUT = 100 / 110
BREAKEVEN = 110 / 210
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 4.0]
PRIMARY = 2.0  # fixed a priori (the brief's example) - NOT chosen by looking at results
NO_LINES_MSG = ("ROI could not be validated because free historical betting lines were not available; "
                "with the client's API access to closing lines, this is the first thing I would test.")


def simulate(df, pred_col, line_col, actual_col, thr) -> pd.DataFrame:
    """Bet home/over (+1) when pred - line >= thr, away/under (-1) when <= -thr. Returns one row per bet."""
    edge = df[pred_col] - df[line_col]
    side = np.where(edge >= thr, 1, np.where(edge <= -thr, -1, 0))
    d = df[side != 0].copy()
    d["side"] = side[side != 0]
    d["result"] = np.sign(d[actual_col] - d[line_col]) * d["side"]  # +1 win, -1 loss, 0 push
    d["profit"] = np.where(d["result"] > 0, PAYOUT, np.where(d["result"] < 0, -1.0, 0.0))
    return d.sort_values(["date", "game_id"])


def summarize(bets: pd.DataFrame, boot=False, seed=SEED) -> dict:
    dec = bets[bets["result"] != 0]
    n = len(dec)
    if n == 0:
        return {"n_bets": 0, "pushes": int(len(bets))}
    cum = np.concatenate([[0.0], bets["profit"].cumsum().values])
    out = {"n_bets": n, "pushes": int(len(bets) - n), "win_rate": float((dec["result"] > 0).mean()),
           "units": float(dec["profit"].sum()), "roi_pct": float(100 * dec["profit"].mean()),
           "max_drawdown_units": float((np.maximum.accumulate(cum) - cum).max())}
    if boot and n >= 20:
        rng = np.random.default_rng(seed)
        prof = dec["profit"].values
        idx = rng.integers(0, n, size=(N_BOOT, n))
        roi = 100 * prof[idx].mean(1)
        wr = (prof[idx] > 0).mean(1)
        out["win_rate_ci95"] = [float(x) for x in np.percentile(wr, [2.5, 97.5])]
        out["roi_ci95_pct"] = [float(x) for x in np.percentile(roi, [2.5, 97.5])]
        out["indistinguishable_from_breakeven"] = bool(out["roi_ci95_pct"][0] <= 0 <= out["roi_ci95_pct"][1])
    return out


def threshold_table(df, pred_col, line_col, actual_col) -> dict:
    return {str(t): summarize(simulate(df, pred_col, line_col, actual_col, t), boot=(t == PRIMARY)) for t in THRESHOLDS}


def chart_proxy(series: dict, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    for (label, bets), c in zip(series.items(), (C_BLUE, C_ORANGE)):
        dec = bets[bets["result"] != 0]
        ax.plot(dec["date"], dec["profit"].cumsum(), color=c, lw=2, label=f"{label} ({len(dec)} bets)")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_ylabel("Cumulative units (flat 1u at -110)", color=C_MUTED)
    ax.set_title("PROXY - NOT REAL MARKET RESULTS\nLightGBM vs. Baseline B's number, edge >= 2 pts", loc="left", color=C_INK, fontsize=11)
    style_axes(ax); ax.legend(frameon=False, fontsize=8, loc="best")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def real_lines_backtest(preds: pd.DataFrame, lines: pd.DataFrame) -> dict:
    """Case A. lines needs columns game_id, spread_home (negative = home favoured), total."""
    d = preds.merge(lines[["game_id", "spread_home", "total"]].rename(columns={"total": "line_total"}), on="game_id")
    d["line_margin"] = -d["spread_home"]
    valid, test = d[d["fold"] == 1], d[d["fold"] > 1]
    out = {"case": "A", "n_games_with_lines": len(d), "markets": {}}
    for mkt, pred, line, act in (("spread", "pred_margin_GBDT", "line_margin", "margin"),
                                 ("total", "pred_total_GBDT", "line_total", "total")):
        # threshold picked on validation fold only (need >= 30 bets there)
        val = {t: summarize(simulate(valid, pred, line, act, t)) for t in THRESHOLDS}
        ok = {t: s for t, s in val.items() if s.get("n_bets", 0) >= 30}
        chosen = max(ok, key=lambda t: ok[t]["roi_pct"]) if ok else PRIMARY
        out["markets"][mkt] = {"chosen_threshold_on_validation": chosen,
                               "hit_rate_all_games": summarize(simulate(test, pred, line, act, 0.0), boot=True),
                               "test_at_chosen": summarize(simulate(test, pred, line, act, chosen), boot=True),
                               "test_all_thresholds": threshold_table(test, pred, line, act)}
    return out


def run():
    preds = pd.read_parquet(PROC / "predictions.parquet")
    lines = load_lines()
    if lines is not None:
        update_results("backtest", real_lines_backtest(preds, lines))
        return

    # ---- Case B: proxy only ----
    p = preds.copy()
    p["proxy_line_margin"] = (p["pred_margin_B"] * 2).round() / 2  # Baseline B's number, snapped to half points
    p["proxy_line_total"] = (p["pred_total_B"] * 2).round() / 2
    res = {"case": "B", "message": NO_LINES_MSG, "label": "PROXY - NOT REAL MARKET RESULTS",
           "proxy_definition": "Line = Baseline B's own prediction (rounded to 0.5). Bet LightGBM's side when it differs "
                               "from that line by >= threshold points; 1 unit at -110. Measures disagreement with a "
                               "simple formula, not edge against a sportsbook.",
           "primary_threshold_pts": PRIMARY, "break_even_win_rate": BREAKEVEN,
           "threshold_note": "Threshold fixed a priori at 2.0 (not tuned); all thresholds shown, none selected.",
           "spread": threshold_table(p, "pred_margin_GBDT", "proxy_line_margin", "margin"),
           "total": threshold_table(p, "pred_total_GBDT", "proxy_line_total", "total"),
           "ats_ou_hit_rates": "not computed - requires real lines"}
    chart_proxy({"Spread": simulate(p, "pred_margin_GBDT", "proxy_line_margin", "margin", PRIMARY),
                 "Total": simulate(p, "pred_total_GBDT", "proxy_line_total", "total", PRIMARY)},
                OUT / "proxy_cumulative_profit.png")
    update_results("backtest", res)


if __name__ == "__main__":
    run()
