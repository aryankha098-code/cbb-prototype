"""Builds and executes notebooks/analysis.ipynb. Markdown cells explain; every number is printed by a code cell."""
import nbformat as nbf
from nbclient import NotebookClient

from .config import ROOT

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
R = json.loads((ROOT / "output" / "results.json").read_text())
"""

CELLS = [
    ("md", "# CBB prediction prototype - analysis walkthrough\n\n**SAMPLE - PROTOTYPE** on one past season. This notebook only *reads* the outputs of `python run_all.py`; "
           "every number below is printed by code from `output/results.json` and the processed tables."),
    ("code", SETUP),
    ("md", "## 1. Data\nThe preferred sources (Barttorvik, cbbdata, ESPN) are probed first. The source actually used is shown below. No betting lines were invented."),
    ("code", """print(R["data"]["source_used"])
display(pd.DataFrame(R["data"]["source_probe"]).T[["status"]])
print({k: v for k, v in R["data"].items() if k.startswith("games") or k in ("date_min", "date_max", "neutral_site_games", "real_betting_lines_available")})
games = pd.read_parquet(ROOT / "data/processed/games.parquet")
games[["date","home_name","away_name","neutral","home_score","away_score","h1_home","h1_away","poss"]].head()"""),
    ("md", "## 2. Features: as-of-date only\nEach game's features use only games **before** its date. The check below is the same one asserted in `tests/test_leakage.py` "
           "(which also proves invariance to truncated and corrupted future data)."),
    ("code", """feats = pd.read_parquet(ROOT / "data/processed/features.parquet")
print("rows:", len(feats), "| every row's latest data date < game date:", bool((feats.hist_max_date < feats.date).all()))
feats[["date","home_name","away_name","d_srs","d_net_eff","d_form10_margin","exp_tempo","b_margin","margin"]].head()"""),
    ("md", "## 3. Walk-forward results (held-out games only)\nModels are refit on an expanding window and scored on the next unseen month. Baseline A = always home / home-court margin; Baseline B = efficiency-margin formula."),
    ("code", """m = R["metrics"]
print("held-out games:", m["n_test_games"], m["test_start"][:10], "to", m["test_end"][:10])
display(pd.DataFrame(m["ml"]).T.set_index("name").astype(float).round(4))
display(pd.DataFrame(m["mae"]).T.round(2))"""),
    ("md", "### Is LightGBM really different from the simple formula?\nPaired bootstrap over games (difference = LightGBM minus Baseline B). An interval that includes 0 means statistically indistinguishable."),
    ("code", """b = pd.DataFrame({k: v for k, v in m["bootstrap"].items() if k.endswith("_minus_B")}).T
b["ci_low"] = b["ci95"].apply(lambda x: x[0]); b["ci_high"] = b["ci95"].apply(lambda x: x[1])
display(b[["mean_diff","ci_low","ci_high","excludes_zero","lower_is_better"]].round(4))
display(pd.DataFrame(m["per_fold"]).round(3))"""),
    ("md", "### Calibration, error by target and feature importance"),
    ("code", """for f in ("calibration.png", "mae_comparison.png", "feature_importance.png"):
    display(Image(filename=str(ROOT / "output" / f), width=560))"""),
    ("md", "### Interpretable decision tree (depth 3)\nPlain-English rules from the final fold's tree."),
    ("code", 'print("\\n\\n".join(m["tree_rules"]))'),
    ("md", "## 4. ROI backtest\nWithout real closing lines no ROI can be validated. The only simulation is against a *proxy* line (Baseline B's own number) and is **not market performance**."),
    ("code", """bt = R["backtest"]
print(bt["case"], "-", bt.get("message", ""))
print(bt.get("label", ""))
if bt["case"] == "B":
    for mk in ("spread", "total"):
        print("\\nPROXY -", mk)
        display(pd.DataFrame(bt[mk]).T[["n_bets","win_rate","units","roi_pct","max_drawdown_units"]].astype(float).round(3))
    display(Image(filename=str(ROOT / "output" / "proxy_cumulative_profit.png"), width=560))"""),
    ("md", "## 5. Game card\nOne real held-out game, generated from pre-game information only."),
    ("code", """print(R["card"]["game"], R["card"]["date"][:10], "| selection rule:", R["card"]["selection_rule"])
display(Image(filename=str(ROOT / "output" / "game_card.png"), width=520))"""),
    ("md", "## 6. Limitations\nOne season; no injuries/lineups/line movement/real market lines; fixed untuned hyper-parameters; ESPN-derived box scores rather than Barttorvik; bootstrap treats games as independent. See the README for the full list and the extension plan."),
]


def build_and_run():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "analysis.ipynb"
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    build_and_run()
