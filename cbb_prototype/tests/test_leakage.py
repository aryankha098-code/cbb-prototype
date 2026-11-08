"""Proof that features use only past data.

Test 1: the latest date used for each row's features is strictly before the game date.
Test 2 (the strong one): rebuild the features from data TRUNCATED at a cutoff date. If any feature had
        peeked at the future, features for earlier games would change. They must be identical.
Test 3: corrupt every score on/after the cutoff and rebuild - features before the cutoff must not move.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_games  # noqa: E402
from src.features import FEATURES, build_features  # noqa: E402

CUTOFF = pd.Timestamp("2024-02-01")


def _compare(a: pd.DataFrame, b: pd.DataFrame):
    a, b = a.set_index("game_id").sort_index(), b.set_index("game_id").sort_index()
    assert list(a.index) == list(b.index), "different set of games survived the filters"
    cols = FEATURES + ["b_margin", "b_total", "a_margin", "lg_hca", "d_srs", "h_n_games", "a_n_games"]
    np.testing.assert_allclose(a[cols].values.astype(float), b[cols].values.astype(float), rtol=1e-9, atol=1e-9)


def test_max_history_date_before_game_date():
    feats = build_features(load_games())
    assert len(feats) > 1000
    assert (feats["hist_max_date"] < feats["date"]).all(), "a feature used data from the game date or later"


def test_truncation_invariance():
    games = load_games()
    full = build_features(games)
    trunc = build_features(games[games["date"] < CUTOFF])
    _compare(full[full["date"] < CUTOFF], trunc)


def test_future_scores_do_not_move_past_features():
    games = load_games()
    full = build_features(games)
    bad = games.copy()
    fut = bad["date"] >= CUTOFF
    for c in ("home_score", "away_score", "fgm_h", "fgm_a", "fga_h", "fga_a", "tov_h", "tov_a", "poss"):
        bad.loc[fut, c] = bad.loc[fut, c] * 3 + 17  # nonsense future data
    corrupted = build_features(bad)
    _compare(full[full["date"] < CUTOFF], corrupted[corrupted["date"] < CUTOFF])
