"""Run the whole prototype:  pip install -r requirements.txt && python run_all.py

Stages: data -> features -> models -> backtest -> game card -> README/summary -> tests -> notebook.
Optional flags: --skip-notebook, --skip-tests
"""
import argparse
import subprocess
import sys
import time

import numpy as np

from src import backtest, card, data, features, models, report
from src.config import PROC, ROOT, SEED, reset_results, update_results

np.random.seed(SEED)


def stage(n, title):
    print(f"\n=== [{n}] {title} ===", flush=True)
    return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-notebook", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    reset_results()

    t = stage(1, "Data (cached in data/raw)")
    games = data.build_games()
    print(f"    {len(games):,} games, {games['date'].min():%Y-%m-%d} to {games['date'].max():%Y-%m-%d}; "
          f"first-half scores for {games['h1_home'].notna().sum():,}. [{time.time() - t:.0f}s]")

    t = stage(2, "As-of-date features (no future data)")
    feats = features.build_features(games)
    feats.to_parquet(PROC / "features.parquet")
    update_results("features", {"games_in_feature_table": len(feats), "min_prior_games": features.MIN_GAMES,
                                "n_model_features": len(features.FEATURES), "date_min": feats["date"].min(),
                                "date_max": feats["date"].max()})
    print(f"    {len(feats):,} games with >= {features.MIN_GAMES} prior games for both teams. [{time.time() - t:.0f}s]")

    t = stage(3, "Models + walk-forward validation")
    preds, _ = models.run()
    print(f"    {len(preds):,} held-out games scored. [{time.time() - t:.0f}s]")

    t = stage(4, "Backtest (Case A if data/raw/lines.csv exists, else labelled proxy only)")
    backtest.run()
    print(f"    done. [{time.time() - t:.0f}s]")

    t = stage(5, "Game card (HTML + 1080x1350 PNG)")
    card.run()
    print(f"    done. [{time.time() - t:.0f}s]")

    t = stage(6, "README + summary (numbers read from output/results.json)")
    report.run()
    print(f"    done. [{time.time() - t:.0f}s]")

    if not args.skip_tests:
        t = stage(7, "Tests (leakage + metric sanity)")
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
        update_results("tests", {"all_passed": rc == 0})
        print(f"    pytest exit code {rc}. [{time.time() - t:.0f}s]")
        if rc != 0:
            sys.exit(rc)

    if not args.skip_notebook:
        t = stage(8, "Notebook (executed)")
        from src import notebook
        notebook.build_and_run()
        print(f"    notebooks/analysis.ipynb written and executed. [{time.time() - t:.0f}s]")

    print("\nAll done. See README.md, output/summary.md, output/game_card.png")


if __name__ == "__main__":
    main()
