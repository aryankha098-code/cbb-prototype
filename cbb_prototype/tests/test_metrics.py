"""Basic metric sanity tests: hand-computed cases, plus plausibility checks on the real run's numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.backtest import BREAKEVEN, PAYOUT, real_lines_backtest, simulate, summarize  # noqa: E402
from src.models import ml_metrics, tree_rules  # noqa: E402


def test_simulate_hand_computed():
    df = pd.DataFrame({"game_id": [1, 2, 3, 4], "date": pd.to_datetime(["2024-01-01"] * 4),
                       "pred": [10.0, 10.0, 0.0, 5.0], "line": [5.0, 5.0, 5.0, 5.0], "actual": [8.0, 2.0, 0.0, 5.0]})
    bets = simulate(df, "pred", "line", "actual", thr=2.0)  # games 1,2 (home side) and 3 (away side); game 4 no edge
    assert list(bets["game_id"]) == [1, 2, 3]
    assert list(bets["result"]) == [1, -1, 1]  # 8>5 win, 2<5 loss, away side wins since 0<5
    s = summarize(bets)
    assert s["n_bets"] == 3 and s["win_rate"] == pytest.approx(2 / 3)
    assert s["units"] == pytest.approx(2 * PAYOUT - 1)
    assert s["max_drawdown_units"] == pytest.approx(1.0)
    assert BREAKEVEN == pytest.approx(0.5238, abs=1e-4)


def test_push_is_not_a_bet():
    df = pd.DataFrame({"game_id": [1], "date": pd.to_datetime(["2024-01-01"]), "pred": [9.0], "line": [5.0], "actual": [5.0]})
    s = summarize(simulate(df, "pred", "line", "actual", 2.0))
    assert s["n_bets"] == 0 and s["pushes"] == 1


def test_perfect_probabilities_score_perfectly():
    y = np.array([1, 0, 1, 1, 0])
    df = pd.DataFrame({"home_win": y})
    for k in ("A", "B", "tree", "gbdt_raw", "gbdt"):
        df[f"p_{k}"] = np.where(y == 1, 0.999, 0.001)
    m = ml_metrics(df)
    assert m["gbdt"]["accuracy"] == 1.0 and m["gbdt"]["log_loss"] < 0.01
    assert m["A"]["accuracy"] == pytest.approx(y.mean())  # "always home" accuracy = home win rate


def test_tree_rules_merge_repeated_splits():
    from sklearn.tree import DecisionTreeClassifier
    from src.features import FEATURES
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
    y = (X["d_srs"] + 0.3 * rng.normal(size=500) > 0).astype(int)
    rules = tree_rules(DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y))
    assert len(rules) == 8 and all(r.startswith("IF ") and "THEN home team wins" in r for r in rules)


def test_real_run_numbers_are_plausible():
    p = ROOT / "output" / "results.json"
    if not p.exists():
        pytest.skip("run `python run_all.py` first")
    m = json.loads(p.read_text())["metrics"]
    for k, v in m["ml"].items():
        assert 0.5 < v["accuracy"] < 0.9, k
        assert 0.3 < v["log_loss"] < 0.8, k
    assert m["ml"]["A"]["accuracy"] == pytest.approx(m["home_win_rate_test"], abs=1e-9)
    for t, v in m["mae"].items():
        assert all(v[k] > 0 for k in ("A", "B", "GBDT")), t
    # a sane model should beat the naive "home court only" baseline on margin error
    assert m["mae"]["margin"]["GBDT"] < m["mae"]["margin"]["A"]
    assert m["ml"]["gbdt"]["log_loss"] < m["ml"]["A"]["log_loss"]


def test_predictions_are_out_of_sample_and_chronological():
    p = ROOT / "data" / "processed" / "predictions.parquet"
    if not p.exists():
        pytest.skip("run `python run_all.py` first")
    preds = pd.read_parquet(p)
    assert preds["date"].min() >= pd.Timestamp("2024-01-01")  # nothing from the initial training window is scored
    assert preds["p_gbdt"].between(0, 1).all()
    assert (preds.groupby("fold")["date"].min().diff().dropna() > pd.Timedelta(0)).all()


def test_case_a_hook_runs_on_synthetic_lines():
    """SYNTHETIC lines, used only to exercise the code path for when real lines are supplied. Never saved."""
    p = ROOT / "data" / "processed" / "predictions.parquet"
    if not p.exists():
        pytest.skip("run `python run_all.py` first")
    preds = pd.read_parquet(p)
    rng = np.random.default_rng(1)
    lines = pd.DataFrame({"game_id": preds["game_id"],
                          "spread_home": -(preds["margin"] * 0.3 + rng.normal(0, 8, len(preds))).round() ,
                          "total": (preds["total"] * 0.3 + 100 + rng.normal(0, 6, len(preds))).round()})
    out = real_lines_backtest(preds, lines)
    assert out["case"] == "A" and set(out["markets"]) == {"spread", "total"}
    assert out["markets"]["spread"]["chosen_threshold_on_validation"] in (0.5, 1.0, 2.0, 3.0, 4.0)
