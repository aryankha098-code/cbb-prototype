"""Step 2 - as-of-date feature engineering.

=====================  NO-LEAKAGE RULE (read this first)  =====================
Every feature for a game on date D is computed ONLY from games with date STRICTLY BEFORE D.
  * team stats: cumulative sums over the team's earlier games (np.searchsorted(..., side="left"))
  * opponent-adjusted rating (SRS): ridge regression fitted on games before D only
  * league constants (home-court advantage, mean total, ...): cumulative over games before D only
Nothing here ever reads a season-end rating. `hist_max_date` records the latest date actually used
for each row, and tests/test_leakage.py proves (a) hist_max_date < game date and (b) rebuilding the
features from data truncated at a cutoff date reproduces the same features for every earlier game.
================================================================================
"""
import numpy as np
import pandas as pd

MIN_GAMES = 8          # both teams need at least this many prior games
SRS_RIDGE = 5.0        # shrinkage (in "games") for the opponent-adjusted rating; not tuned
SRS_MARGIN_CAP = 25.0  # cap blowout margins when fitting the rating

SUM_COLS = ["pts", "opp_pts", "poss", "mf", "fgm", "tpm", "fga", "fta", "orb", "tov", "drb",
            "o_fgm", "o_tpm", "o_fga", "o_fta", "o_orb", "o_tov", "o_drb"]

# per-team metrics that become h_/a_/d_ columns
METRICS = ["off_eff", "def_eff", "net_eff", "tempo", "efg", "tov", "orb", "ftr",
           "d_efg", "d_tov", "d_orb", "d_ftr",
           "form5_win", "form5_margin", "form10_win", "form10_margin", "rest", "srs"]


def _long_table(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team), sorted by team then date."""
    parts = []
    for me, op, tid, oid, ms, os_ in (("h", "a", "home_id", "away_id", "home_score", "away_score"),
                                      ("a", "h", "away_id", "home_id", "away_score", "home_score")):
        d = pd.DataFrame({"team_id": g[tid].values, "date": g["date"].values,
                          "pts": g[ms].values, "opp_pts": g[os_].values,
                          "poss": g["poss"].values, "mf": g["minutes"].values / 40.0})
        for c in ("fgm", "tpm", "fga", "fta", "orb", "tov", "drb"):
            d[c] = g[f"{c}_{me}"].values
            d["o_" + c] = g[f"{c}_{op}"].values
        d["margin"] = d["pts"] - d["opp_pts"]
        d["win"] = (d["margin"] > 0).astype(float)
        parts.append(d)
    return pd.concat(parts).sort_values(["team_id", "date"], kind="stable").reset_index(drop=True)


class _TeamHist:
    """A team's game log with cumulative sums, so 'stats before date D' is an O(1) lookup."""

    def __init__(self, df: pd.DataFrame):
        self.dates = df["date"].values.astype("datetime64[D]")
        self.cum = {c: np.concatenate([[0.0], np.cumsum(df[c].values.astype(float))]) for c in SUM_COLS}
        self.margin = df["margin"].values.astype(float)
        self.win = df["win"].values.astype(float)

    def count_before(self, d) -> int:
        return int(np.searchsorted(self.dates, d, side="left"))

    def asof(self, d):
        """Features from games strictly before date d. Returns (features, last_prior_date) or None."""
        k = int(np.searchsorted(self.dates, d, side="left"))  # side="left": same-day games excluded
        if k < MIN_GAMES:
            return None
        S = lambda c: self.cum[c][k]  # noqa: E731
        f = {
            "n_games": k,
            "off_eff": 100 * S("pts") / S("poss"),
            "def_eff": 100 * S("opp_pts") / S("poss"),
            "tempo": S("poss") / S("mf"),
            "efg": (S("fgm") + 0.5 * S("tpm")) / S("fga"),
            "tov": S("tov") / (S("fga") + 0.475 * S("fta") + S("tov")),
            "orb": S("orb") / (S("orb") + S("o_drb")),
            "ftr": S("fta") / S("fga"),
            "d_efg": (S("o_fgm") + 0.5 * S("o_tpm")) / S("o_fga"),
            "d_tov": S("o_tov") / (S("o_fga") + 0.475 * S("o_fta") + S("o_tov")),
            "d_orb": S("o_orb") / (S("o_orb") + S("drb")),
            "d_ftr": S("o_fta") / S("o_fga"),
        }
        f["net_eff"] = f["off_eff"] - f["def_eff"]
        for w in (5, 10):
            lo = max(0, k - w)
            f[f"form{w}_win"] = self.win[lo:k].mean()
            f[f"form{w}_margin"] = self.margin[lo:k].mean()
        last = self.dates[k - 1]
        f["rest"] = min(int((d - last) / np.timedelta64(1, "D")), 14)
        return f, last


def _league_cumulatives(g: pd.DataFrame) -> dict:
    """Cumulative league-wide sums by game order, for as-of league constants."""
    nn = (~g["neutral"]).astype(float).values
    margin = (g["home_score"] - g["away_score"]).values.astype(float)
    h1ok = g["h1_home"].notna().values.astype(float)
    h1m = np.nan_to_num((g["h1_home"] - g["h1_away"]).values.astype(float))
    h1t = np.nan_to_num((g["h1_home"] + g["h1_away"]).values.astype(float))
    total = (g["home_score"] + g["away_score"]).values.astype(float)
    cs = lambda x: np.concatenate([[0.0], np.cumsum(x)])  # noqa: E731
    return {
        "dates": g["date"].values.astype("datetime64[D]"),
        "n": cs(np.ones(len(g))), "pts": cs(total), "poss": cs(g["poss"].values),
        "tempo": cs((g["poss"] / (g["minutes"] / 40)).values), "nn": cs(nn), "hca": cs(margin * nn),
        "hwin": cs((margin > 0).astype(float)), "h1n": cs(h1ok), "h1hca": cs(h1m * nn * h1ok),
        "h1nn": cs(nn * h1ok), "h1tot": cs(h1t * h1ok), "tot_h1games": cs(total * h1ok),
    }


MIN_LEAGUE_GAMES = 100  # need this many eligible past games before league constants are trusted


def _league_asof(L: dict, d):
    k = int(np.searchsorted(L["dates"], d, side="left"))
    S = lambda c: L[c][k]  # noqa: E731
    if S("n") < MIN_LEAGUE_GAMES:
        return None
    return {
        "lg_eff": 100 * S("pts") / (2 * S("poss")), "lg_tempo": S("tempo") / S("n"),
        "lg_total": S("pts") / S("n"), "lg_hca": S("hca") / max(S("nn"), 1), "lg_homewin": S("hwin") / S("n"),
        "lg_h1_hca": S("h1hca") / max(S("h1nn"), 1), "lg_h1_total": S("h1tot") / max(S("h1n"), 1),
        "lg_total_h1games": S("tot_h1games") / max(S("h1n"), 1),
    }


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with as-of features, targets and baseline inputs (see module docstring)."""
    g = games.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    lt = _long_table(g)
    hist = {tid: _TeamHist(df) for tid, df in lt.groupby("team_id")}

    # League constants (home-court advantage, mean total, ...) are computed over ELIGIBLE past games only:
    # games where both teams already had >= MIN_GAMES prior games. This keeps early-season mismatches
    # against non-D1 opponents out of the constants. A game's eligibility depends only on data before
    # that game, so this is still leak-free.
    gd = g["date"].values.astype("datetime64[D]")
    eligible = np.array([hist[h].count_before(d) >= MIN_GAMES and hist[a].count_before(d) >= MIN_GAMES
                         for h, a, d in zip(g["home_id"], g["away_id"], gd)])
    L = _league_cumulatives(g[eligible].reset_index(drop=True))

    # ---- opponent-adjusted rating (SRS) fitted incrementally on games before each date ----
    teams = sorted(set(g["home_id"]) | set(g["away_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    H = T  # column of the home-court term
    XtX, Xty = np.zeros((T + 1, T + 1)), np.zeros(T + 1)
    ridge = np.full(T + 1, SRS_RIDGE)
    ridge[H] = 1.0
    hi, ai = g["home_id"].map(idx).values, g["away_id"].map(idx).values
    marg = np.clip((g["home_score"] - g["away_score"]).values.astype(float), -SRS_MARGIN_CAP, SRS_MARGIN_CAP)
    c_home = (~g["neutral"]).astype(float).values
    gdates = g["date"].values.astype("datetime64[D]")

    rows, added = [], 0
    for d in np.unique(gdates):
        while added < len(g) and gdates[added] < d:  # add only games strictly before d
            i, j, c, m = hi[added], ai[added], c_home[added], marg[added]
            XtX[i, i] += 1; XtX[j, j] += 1; XtX[i, j] -= 1; XtX[j, i] -= 1
            XtX[i, H] += c; XtX[H, i] += c; XtX[j, H] -= c; XtX[H, j] -= c; XtX[H, H] += c * c
            Xty[i] += m; Xty[j] -= m; Xty[H] += c * m
            added += 1
        srs_max_date = gdates[added - 1] if added else None
        r = np.linalg.solve(XtX + np.diag(ridge), Xty) if added else np.zeros(T + 1)
        lg = _league_asof(L, d)
        if lg is None:
            continue  # too few eligible past games to estimate league constants

        for gi in np.flatnonzero(gdates == d):
            row = g.iloc[gi]
            fh = hist[row["home_id"]].asof(d)
            fa = hist[row["away_id"]].asof(d)
            if fh is None or fa is None or srs_max_date is None:
                continue  # fewer than MIN_GAMES prior games for a team
            (h, h_last), (a, a_last) = fh, fa
            h["srs"], a["srs"] = r[hi[gi]], r[ai[gi]]
            rec = {"game_id": row["game_id"], "date": row["date"], "home_id": row["home_id"],
                   "away_id": row["away_id"], "home_name": row["home_name"], "away_name": row["away_name"],
                   "neutral": int(row["neutral"]), "season_type": row["season_type"],
                   "hist_max_date": pd.Timestamp(max(h_last, a_last, srs_max_date)),
                   "h_n_games": h["n_games"], "a_n_games": a["n_games"], **lg}
            for m in METRICS:
                rec[f"h_{m}"], rec[f"a_{m}"], rec[f"d_{m}"] = h[m], a[m], h[m] - a[m]
            # targets (known only after the game; never used as features)
            rec["margin"] = float(row["home_score"] - row["away_score"])
            rec["home_win"] = int(rec["margin"] > 0)
            rec["total"] = float(row["home_score"] + row["away_score"])
            rec["h1_margin"] = row["h1_home"] - row["h1_away"]
            rec["h1_total"] = row["h1_home"] + row["h1_away"]
            rows.append(rec)

    f = pd.DataFrame(rows)

    # matchup features
    f["exp_tempo"] = (f["h_tempo"] + f["a_tempo"]) / 2
    f["sum_off_eff"] = f["h_off_eff"] + f["a_off_eff"]
    f["sum_def_eff"] = f["h_def_eff"] + f["a_def_eff"]
    hc = 1 - f["neutral"]

    # Baseline A: home-court only / league averages (all as-of)
    f["a_margin"] = f["lg_hca"] * hc
    f["a_total"] = f["lg_total"]
    f["a_h1_margin"] = f["lg_h1_hca"] * hc
    f["a_h1_total"] = f["lg_h1_total"]
    # Baseline B: efficiency-margin difference x expected tempo + home court
    core = f["exp_tempo"] * (f["h_net_eff"] - f["a_net_eff"]) / 100
    f["b_margin"] = core + f["lg_hca"] * hc
    f["b_total"] = f["exp_tempo"] * ((f["h_off_eff"] + f["a_def_eff"] - f["lg_eff"]) +
                                     (f["a_off_eff"] + f["h_def_eff"] - f["lg_eff"])) / 100
    f["b_h1_margin"] = 0.5 * core + f["lg_h1_hca"] * hc  # assumption: 1H margin ~ half the full-game rating margin
    f["b_h1_total"] = f["b_total"] * f["lg_h1_total"] / f["lg_total_h1games"]
    return f


FEATURES = [f"d_{m}" for m in METRICS] + ["neutral", "exp_tempo", "sum_off_eff", "sum_def_eff"]

LABELS = {  # plain-English names used for tree rules, importance chart and card drivers
    "d_off_eff": "Offensive efficiency edge", "d_def_eff": "Defensive efficiency edge (higher = worse D)",
    "d_net_eff": "Efficiency-margin edge", "d_tempo": "Tempo difference",
    "d_efg": "Shooting (eFG%) edge", "d_tov": "Turnover rate difference", "d_orb": "Offensive-rebound rate edge",
    "d_ftr": "Free-throw rate edge", "d_d_efg": "Opponent eFG% allowed difference",
    "d_d_tov": "Turnovers forced difference", "d_d_orb": "Opp. offensive-rebound rate allowed difference",
    "d_d_ftr": "Opp. free-throw rate allowed difference",
    "d_form5_win": "Last-5 win-rate edge", "d_form5_margin": "Last-5 average-margin edge",
    "d_form10_win": "Last-10 win-rate edge", "d_form10_margin": "Last-10 average-margin edge",
    "d_rest": "Rest-days edge", "d_srs": "Opponent-adjusted rating edge",
    "neutral": "Neutral-site game", "exp_tempo": "Expected tempo",
    "sum_off_eff": "Combined offensive efficiency", "sum_def_eff": "Combined defensive efficiency",
}

if __name__ == "__main__":
    from .config import PROC, update_results
    from .data import load_games

    feats = build_features(load_games())
    feats.to_parquet(PROC / "features.parquet")
    update_results("features", {"games_in_feature_table": len(feats), "min_prior_games": MIN_GAMES,
                                "n_model_features": len(FEATURES), "date_min": feats["date"].min(),
                                "date_max": feats["date"].max()})
    print(feats.shape)
