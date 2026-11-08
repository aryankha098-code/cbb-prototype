"""Step 1 - data acquisition.

Source order requested: (1) Barttorvik, (a) cbbdata API, (b) public GitHub/Kaggle dataset, (c) ESPN scoreboard.
The pipeline PROBES (1), (a) and (c) and records the outcome in data/raw/source_probe.json, then uses
(b): the ESPN-derived game schedule + team box scores published in the public GitHub repo
sportsdataverse/hoopR-mbb-data. In the environment this was built in, Barttorvik/cbbdata/ESPN were
blocked (see the probe file), so (b) is what the results come from.

Politeness: every download is cached in data/raw, uses a generic User-Agent and sleeps between requests.
"""
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from .config import PROC, RAW, SEASON, update_results

UA = {"User-Agent": "cbb-prototype/0.1 (research sample)"}
DELAY_S = 1.5

HOOPR = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
SCHEDULE_URL = f"{HOOPR}/schedules/parquet/mbb_schedule_{SEASON}.parquet"
TEAMBOX_URL = f"{HOOPR}/team_box/parquet/team_box_{SEASON}.parquet"

# Sources we try first. We only record reachability - we do not ship an unverified parser for them.
PROBES = {
    "barttorvik": f"https://barttorvik.com/{SEASON}_super_sked.csv",
    "cbbdata_api": "https://cbbdata.aweatherman.com",
    "espn_scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates=20240110&groups=50&limit=400",
}

BOX_COLS = {
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "tpm", "three_point_field_goals_attempted": "tpa",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb", "turnovers": "tov",
}


def fetch(url: str, dest) -> None:
    """Download once, then reuse the cached copy."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    time.sleep(DELAY_S)
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def probe_sources() -> dict:
    """Record whether the preferred sources are reachable (cached so we do not re-hit them)."""
    path = RAW / "source_probe.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for name, url in PROBES.items():
        time.sleep(DELAY_S)
        try:
            r = requests.get(url, headers=UA, timeout=15, stream=True)
            out[name] = {"url": url, "status": f"HTTP {r.status_code}"}
            r.close()
        except Exception as e:  # blocked / DNS / proxy refusal
            out[name] = {"url": url, "status": "unreachable", "detail": str(e)[:160]}
    path.write_text(json.dumps(out, indent=2))
    return out


def parse_linescores(series: pd.Series) -> pd.Series:
    """hoopR stores linescores as a string like "[{'value': 36.0} {'value': 39.0}]"."""
    pat = re.compile(r"'value':\s*([0-9.]+)")
    return series.apply(lambda s: [float(x) for x in pat.findall(s)] if isinstance(s, str) else [])


def build_games() -> pd.DataFrame:
    """One row per completed game: scores, first-half scores, neutral flag and both teams' box stats."""
    sched_path, box_path = RAW / f"mbb_schedule_{SEASON}.parquet", RAW / f"team_box_{SEASON}.parquet"
    fetch(SCHEDULE_URL, sched_path)
    fetch(TEAMBOX_URL, box_path)
    sched, box = pd.read_parquet(sched_path), pd.read_parquet(box_path)

    s = sched[sched["status_type_completed"] == True].drop_duplicates("game_id").copy()  # noqa: E712
    n_completed = len(s)

    # First-half score = first linescore value. Only trusted if the linescores add up to the final score.
    hl, al = parse_linescores(s["home_linescores"]), parse_linescores(s["away_linescores"])
    s["n_periods"] = hl.apply(len)
    ok = (hl.apply(sum) == s["home_score"]) & (al.apply(sum) == s["away_score"]) & (s["n_periods"] >= 2)
    s["h1_home"] = hl.apply(lambda x: x[0] if x else np.nan).where(ok)
    s["h1_away"] = al.apply(lambda x: x[0] if x else np.nan).where(ok)

    b = box[["game_id", "team_home_away"] + list(BOX_COLS)].rename(columns=BOX_COLS)
    h = b[b["team_home_away"] == "home"].drop(columns="team_home_away")
    a = b[b["team_home_away"] == "away"].drop(columns="team_home_away")
    h = h.rename(columns={c: f"{c}_h" for c in h.columns if c != "game_id"})
    a = a.rename(columns={c: f"{c}_a" for c in a.columns if c != "game_id"})

    keep = {
        "game_id": "game_id", "game_date": "date", "neutral_site": "neutral", "season_type": "season_type",
        "home_id": "home_id", "away_id": "away_id", "home_location": "home_name", "away_location": "away_name",
        "home_score": "home_score", "away_score": "away_score", "n_periods": "n_periods",
        "h1_home": "h1_home", "h1_away": "h1_away", "venue_full_name": "venue", "notes_headline": "notes",
    }
    g = s[list(keep)].rename(columns=keep)
    g = g.merge(h, on="game_id").merge(a, on="game_id")  # inner joins drop games without box scores
    g["date"] = pd.to_datetime(g["date"])
    g["neutral"] = g["neutral"].astype(bool)
    for c in ("home_id", "away_id", "game_id"):
        g[c] = pd.to_numeric(g[c]).astype("int64")

    # Possessions (standard estimate), averaged over both teams; minutes adjust for overtime.
    poss_h = g["fga_h"] - g["orb_h"] + g["tov_h"] + 0.475 * g["fta_h"]
    poss_a = g["fga_a"] - g["orb_a"] + g["tov_a"] + 0.475 * g["fta_a"]
    g["poss"] = (poss_h + poss_a) / 2
    g["minutes"] = 40 + 5 * (g["n_periods"] - 2)

    g = g.dropna(subset=["home_score", "away_score", "poss"]).query("poss > 20")
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    g.to_parquet(PROC / "games.parquet")

    update_results("data", {
        "source_used": "hoopR-mbb-data (ESPN-derived box scores) from public GitHub, season 2023-24",
        "source_urls": [SCHEDULE_URL, TEAMBOX_URL],
        "source_probe": probe_sources(),
        "games_completed_in_schedule": n_completed,
        "games_with_box_scores": len(g),
        "games_with_first_half_scores": int(g["h1_home"].notna().sum()),
        "date_min": g["date"].min().date().isoformat(),
        "date_max": g["date"].max().date().isoformat(),
        "neutral_site_games": int(g["neutral"].sum()),
        "real_betting_lines_available": (RAW / "lines.csv").exists(),
        "lines_note": "No free bulk historical closing lines for 2023-24 were found; none were invented.",
    })
    return g


def load_games() -> pd.DataFrame:
    p = PROC / "games.parquet"
    return pd.read_parquet(p) if p.exists() else build_games()


def load_lines():
    """Optional: real closing lines dropped in by the user at data/raw/lines.csv with columns
    game_id, spread_home (negative = home favoured), total. Returns None if absent."""
    p = RAW / "lines.csv"
    return pd.read_csv(p) if p.exists() else None


if __name__ == "__main__":
    games = build_games()
    print(games.shape)
