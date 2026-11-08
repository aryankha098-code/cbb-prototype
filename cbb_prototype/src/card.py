"""Step 6 - one sample game card (HTML + 1080x1350 PNG) from a real held-out game, pre-game info only.

Game choice rule (fixed, not result-based): among held-out regular-season, non-neutral games, take the one with
the highest combined pre-game opponent-adjusted rating (home + away) - i.e. the best matchup on paper.
Everything on the card is read from predictions/feature tables built only from games before that date.
"""
import shutil

import joblib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .config import OUT, PROC, TEMPLATES, update_results
from .data import load_games
from .features import FEATURES, LABELS

DISCLAIMER = ("Quick sample built on one past season. Not betting advice. "
              "Results are backtested and do not guarantee future performance.")


def fmt_metric(base: str, v: float) -> str:
    if base in ("efg", "tov", "orb", "ftr", "d_efg", "d_tov", "d_orb", "d_ftr"):
        return f"{100 * v:.1f}%"
    if base.endswith("_win"):
        return f"{100 * v:.0f}%"
    if base.endswith("_margin") or base == "srs":
        return f"{v:+.1f}"
    if base == "rest":
        return f"{v:.0f}d"
    return f"{v:.1f}"


def _half(x: float) -> float:
    return round(x * 2) / 2


def fmt_spread(m: float, home: str, away: str) -> str:
    r = _half(abs(m))
    return "Pick'em" if r == 0 else f"{home if m > 0 else away} -{r:.1f}"


def short(name: str, n=15) -> str:
    return name if len(name) <= n else name[: n - 1] + "."


def pick_game(preds: pd.DataFrame) -> pd.Series:
    d = preds[(preds["season_type"] == 2) & (preds["neutral"] == 0)].copy()
    d["combined"] = d["h_srs"] + d["a_srs"]
    return d.sort_values(["combined", "game_id"], ascending=[False, True]).iloc[0]


def team_log(games: pd.DataFrame, team_id: int, before) -> list:
    g = games[((games["home_id"] == team_id) | (games["away_id"] == team_id)) & (games["date"] < before)]
    rows = []
    for _, x in g.sort_values(["date", "game_id"], ascending=False).head(10).iterrows():
        home = x["home_id"] == team_id
        pts, opp = (x["home_score"], x["away_score"]) if home else (x["away_score"], x["home_score"])
        rows.append({"res": "W" if pts > opp else "L", "score": f"{int(pts)}-{int(opp)}",
                     "opp": short(x["away_name"] if home else x["home_name"]),
                     "loc": "N" if x["neutral"] else ("H" if home else "A")})
    return rows


def head_to_head(games, r) -> str:
    h = games[(games["date"] < r["date"]) & (
        ((games["home_id"] == r["home_id"]) & (games["away_id"] == r["away_id"])) |
        ((games["home_id"] == r["away_id"]) & (games["away_id"] == r["home_id"])))]
    if h.empty:
        return "Head-to-head: no earlier meeting this season."
    parts = []
    for _, x in h.sort_values("date").iterrows():
        hs, as_ = (x["home_score"], x["away_score"]) if x["home_id"] == r["home_id"] else (x["away_score"], x["home_score"])
        parts.append(f"{x['date']:%b %d}: {r['home_name']} {'won' if hs > as_ else 'lost'} {int(hs)}-{int(as_)}")
    return "Head-to-head this season: " + "; ".join(parts)


STAT_ROWS = [("off_eff", "Offensive efficiency", True), ("def_eff", "Defensive efficiency", False),
             ("tempo", "Tempo (poss/40)", None), ("efg", "eFG%", True), ("tov", "Turnover rate", False),
             ("orb", "Off. rebound rate", True), ("ftr", "Free-throw rate", True)]


def stat_rows(r) -> list:
    out = []
    for base, label, higher_better in STAT_ROWS:
        hv, av = r[f"h_{base}"], r[f"a_{base}"]
        mx = max(hv, av)
        h_best = a_best = False
        if higher_better is not None and not np.isclose(hv, av):
            h_best = (hv > av) == higher_better
            a_best = not h_best
        out.append({"label": label, "hv": fmt_metric(base, hv), "av": fmt_metric(base, av),
                    "hw": round(230 * hv / mx), "aw": round(230 * av / mx),
                    "h_best": h_best, "a_best": a_best, "neutral": higher_better is None})
    return out


UNIT_HINT = {"exp_tempo": " possessions per 40 min", "sum_off_eff": " (both teams' points scored per 100 poss.)",
             "sum_def_eff": " (both teams' points allowed per 100 poss.)"}


def drivers(r, clf, pick_team: str, pick_is_home: bool) -> list:
    """Top-3 contributors (by size) to the raw, uncalibrated win log-odds for this game, in plain English.
    Each is labelled as supporting or working against the pick, so a close call is not dressed up."""
    contrib = clf.predict(r[FEATURES].to_frame().T.astype(float), pred_contrib=True)[0][:-1]  # last = bias
    out = []
    for i in np.argsort(-np.abs(contrib))[:3]:
        f, c = FEATURES[i], contrib[i]
        supports = (c > 0) == pick_is_home
        if f.startswith("d_"):
            base = f[2:]
            detail = (f"{short(r['home_name'])} {fmt_metric(base, r['h_' + base])} vs "
                      f"{short(r['away_name'])} {fmt_metric(base, r['a_' + base])}")
        elif f == "neutral":
            detail = "neutral site" if r["neutral"] else "home-court game"
        else:
            detail = f"{r[f]:.1f}{UNIT_HINT.get(f, '')}"
        out.append({"title": LABELS[f] + ":", "supports": bool(supports),
                    "detail": f"{detail}. This {'supports' if supports else 'works against'} the {pick_team} pick."})
    return out


def render_png(html_path, png_path) -> int:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--no-sandbox"])
        except Exception:  # Playwright's own Chromium not installed -> use a system Chrome/Chromium
            exe = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
            if not exe:
                raise RuntimeError("No browser found. Run: playwright install chromium")
            browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1080, "height": 1350})
        page.goto(f"file://{html_path}")
        page.wait_for_load_state("load")
        # layout check: the card must not need more than 1350px
        overflow = page.evaluate("document.querySelector('.body').scrollHeight - document.querySelector('.body').clientHeight")
        page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": 1080, "height": 1350})
        browser.close()
    return int(overflow)


def run():
    preds = pd.read_parquet(PROC / "predictions.parquet")
    games = load_games()
    arts = joblib.load(PROC / "artifacts.joblib")["folds"]
    r = pick_game(preds)
    clf = arts[int(r["fold"]) - 1]["clf"]  # model trained only on games before this game's test month

    home, away = r["home_name"], r["away_name"]
    p_home = float(r["p_gbdt"])
    ml_team, ml_p = (home, p_home) if p_home >= 0.5 else (away, 1 - p_home)
    picks = {"ml_team": ml_team, "ml_prob": f"{100 * ml_p:.0f}%", "home_pct": round(100 * p_home), "away_pct": round(100 * (1 - p_home)),
             "spread_full": fmt_spread(r["pred_margin_GBDT"], home, away), "total_full": f"{_half(r['pred_total_GBDT']):.1f}",
             "spread_h1": fmt_spread(r["pred_h1_margin_GBDT"], home, away), "total_h1": f"{_half(r['pred_h1_total_GBDT']):.1f}"}
    card = {"home": home, "away": away, "neutral": bool(r["neutral"]), "date_str": f"{r['date']:%A, %B %d, %Y}",
            "venue_line": f"Home game for {home}" if not r["neutral"] else "Neutral site",
            "picks": picks,
            "last10": {"home": {"name": home, "rows": team_log(games, int(r["home_id"]), r["date"])},
                       "away": {"name": away, "rows": team_log(games, int(r["away_id"]), r["date"])}},
            "h2h": head_to_head(games, r), "stats": stat_rows(r),
            "drivers": drivers(r, clf, ml_team, p_home >= 0.5),
            "confidence_note": (f"Low-confidence pick ({100 * ml_p:.0f}%): treat as a lean, not a strong call."
                                if ml_p < 0.60 else "")}

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("card.html.j2").render(c=card)
    html_path, png_path = OUT / "game_card.html", OUT / "game_card.png"
    html_path.write_text(html, encoding="utf-8")
    overflow = render_png(html_path.resolve(), png_path)

    actual = int(r["margin"])
    update_results("card", {
        "selection_rule": "highest combined pre-game opponent-adjusted rating among held-out, regular-season, non-neutral games",
        "game": f"{away} at {home}", "date": r["date"], "model_fold": int(r["fold"]), "picks": picks,
        "pregame_home_win_prob": p_home, "layout_overflow_px": overflow,
        "outcome_not_shown_on_card": {"final_home_margin": actual, "ml_pick_correct": bool((p_home > 0.5) == (actual > 0))},
        "drivers": card["drivers"], "disclaimer": DISCLAIMER})
    return card


if __name__ == "__main__":
    run()
