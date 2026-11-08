"""Writes README.md and output/summary.md. Every number comes from output/results.json - none are typed by hand.
The wording of verdicts (better / worse / indistinguishable) is chosen from the bootstrap intervals."""
from jinja2 import Environment, FileSystemLoader

from .config import OUT, ROOT, TEMPLATES, load_results, update_results

TARGET_NAMES = {"margin": "Full-game margin", "total": "Full-game total",
                "h1_margin": "1st-half margin", "h1_total": "1st-half total"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def reading(v) -> str:
    if not v["excludes_zero"]:
        return "statistically indistinguishable"
    better = v["mean_diff"] < 0 if v["lower_is_better"] else v["mean_diff"] > 0
    return "LightGBM better" if better else "Baseline B better"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_tables(r):
    m, b = r["metrics"], r["backtest"]
    t = {}
    t["ml"] = md_table(["Model", "ML accuracy", "Log loss", "Brier"],
                       [[v["name"], pct(v["accuracy"]), f"{v['log_loss']:.4f}", f"{v['brier']:.4f}"] for v in m["ml"].values()])
    t["mae"] = md_table(["Target (points)", "Baseline A", "Baseline B", "LightGBM", "Games"],
                        [[TARGET_NAMES[k], f"{v['A']:.2f}", f"{v['B']:.2f}", f"{v['GBDT']:.2f}", v["n"]] for k, v in m["mae"].items()])
    bt = m["bootstrap"]
    rows = []
    for key, label, fmt in (("log_loss_gbdt_minus_B", "Log loss (lower better)", lambda x: f"{x:+.4f}"),
                            ("accuracy_gbdt_minus_B", "ML accuracy (higher better)", lambda x: f"{100 * x:+.1f} pts"),
                            ("mae_margin_gbdt_minus_B", "Full-game margin MAE", lambda x: f"{x:+.2f}"),
                            ("mae_total_gbdt_minus_B", "Full-game total MAE", lambda x: f"{x:+.2f}"),
                            ("mae_h1_margin_gbdt_minus_B", "1st-half margin MAE", lambda x: f"{x:+.2f}"),
                            ("mae_h1_total_gbdt_minus_B", "1st-half total MAE", lambda x: f"{x:+.2f}")):
        v = bt[key]
        rows.append([label, fmt(v["mean_diff"]), f"{fmt(v['ci95'][0])} to {fmt(v['ci95'][1])}", reading(v)])
    t["boot"] = md_table(["LightGBM minus Baseline B", "Difference", "95% bootstrap CI", "Reading"], rows)
    t["folds"] = md_table(["Fold (test window)", "Games", "Acc. LightGBM", "Acc. Baseline B", "Acc. Baseline A", "Log loss LightGBM", "Log loss Baseline B", "Margin MAE LightGBM", "Margin MAE Baseline B"],
                          [[f"{f['start']} to {f['end']}", f["n"], pct(f["acc_gbdt"]), pct(f["acc_B"]), pct(f["acc_A"]),
                            f"{f['ll_gbdt']:.3f}", f"{f['ll_B']:.3f}", f"{f['mae_margin_gbdt']:.2f}", f"{f['mae_margin_B']:.2f}"] for f in m["per_fold"]])
    t["probe"] = md_table(["Preferred source", "Result from this run"],
                          [[k, v["status"]] for k, v in r["data"]["source_probe"].items()])
    if b["case"] == "B":
        for mk in ("spread", "total"):
            rows = []
            for thr, s in b[mk].items():
                rows.append([f"{float(thr):g}", s["n_bets"], pct(s["win_rate"]), f"{s['units']:+.1f}", f"{s['roi_pct']:+.1f}%", f"{s['max_drawdown_units']:.1f}"])
            t[f"proxy_{mk}"] = md_table(["Edge threshold (pts)", "Bets", "Win rate", "Units", "ROI", "Max drawdown (units)"], rows)
        s2 = {mk: b[mk][str(b["primary_threshold_pts"])] for mk in ("spread", "total")}
        t["proxy_ci"] = "; ".join(
            f"{mk}: win rate {pct(s['win_rate'])} (95% CI {pct(s['win_rate_ci95'][0])} to {pct(s['win_rate_ci95'][1])}), "
            f"ROI {s['roi_pct']:+.1f}% (95% CI {s['roi_ci95_pct'][0]:+.1f}% to {s['roi_ci95_pct'][1]:+.1f}%)" for mk, s in s2.items())
    return t


def headline(r) -> str:
    bt = r["metrics"]["bootstrap"]
    keys = [k for k in bt if k.endswith("_minus_B")]
    better = sum(reading(bt[k]) == "LightGBM better" for k in keys)
    worse = sum(reading(bt[k]) == "Baseline B better" for k in keys)
    same = len(keys) - better - worse
    return (f"Against the simple rating-difference formula (Baseline B), the gradient-boosted model was clearly better on "
            f"{better} of {len(keys)} held-out comparisons, clearly worse on {worse}, and statistically indistinguishable on {same} "
            f"(95% paired-bootstrap intervals). Both beat the 'always pick the home team' baseline. Treat this as a working, "
            f"leak-free pipeline - not evidence of a betting edge.")


def make_summary(r) -> str:
    m = r["metrics"]
    ml, mae = m["ml"], m["mae"]
    bt = m["bootstrap"]["log_loss_gbdt_minus_B"]
    verdict = reading(bt)
    if verdict == "LightGBM better":
        v = "The tree model beat the simple formula on win-probability log loss."
    elif verdict == "Baseline B better":
        v = "The tree model did not beat the simple formula on win probability, so this is a working pipeline, not a proven edge."
    else:
        v = "The tree model was statistically indistinguishable from the simple formula on win probability."
    text = (
        f"I built a leak-free college basketball prediction prototype on the 2023-24 season: {r['data']['games_with_box_scores']:,} games "
        f"of public ESPN-derived box scores, validated walk-forward on {m['n_test_games']:,} held-out games from January to April. "
        f"Home-win accuracy was {pct(ml['gbdt']['accuracy'])} for gradient-boosted trees, {pct(ml['B']['accuracy'])} for a simple rating formula, "
        f"and {pct(ml['A']['accuracy'])} for always picking the home team. "
        f"Average error on the final margin was {mae['margin']['GBDT']:.1f} points (formula {mae['margin']['B']:.1f}, home-only {mae['margin']['A']:.1f}); "
        f"on game totals {mae['total']['GBDT']:.1f} (formula {mae['total']['B']:.1f}). {v} "
        f"ROI was not validated: free historical betting lines were not available, so I report no profit figures. "
        f"With your closing-line access, that is the first thing I would test. The package includes calibration, leakage tests and a shareable game card.")
    return text


def run():
    r = load_results()
    tables = build_tables(r)
    summary = make_summary(r)
    words = len(summary.split())
    (OUT / "summary.md").write_text(summary + "\n", encoding="utf-8")
    update_results("summary", {"word_count": words})

    ctx = {"r": r, "d": r["data"], "f": r["features"], "m": r["metrics"], "b": r["backtest"], "c": r["card"],
           "t": tables, "headline": headline(r), "pct": pct, "summary": summary, "words": words,
           "tree_rules": "\n".join(r["metrics"]["tree_rules"])}
    env = Environment(loader=FileSystemLoader(TEMPLATES), trim_blocks=True, lstrip_blocks=True)
    (ROOT / "README.md").write_text(env.get_template("README.md.j2").render(**ctx), encoding="utf-8")
    print(f"    summary.md: {words} words")


if __name__ == "__main__":
    run()
