# College basketball (CBB) game-prediction prototype

> **SAMPLE - PROTOTYPE.** A quick, honest sample of my approach on **one past season (2023-24)**. It is not the final product and not betting advice.
> Every number in this README, in `output/summary.md` and on the game card is generated from the code run (`output/results.json`).

## What this is

A small, reproducible pipeline that predicts, for college basketball games:

- **Moneyline** - home win probability (calibrated)
- **Spread** - predicted home margin
- **Total (O/U)** - predicted combined points
- the same **margin and total for the 1st half**

plus a **game card** (HTML + shareable 1080x1350 PNG) showing the picks, last-10 form, head-to-head, a stats comparison and the model's top drivers.

**Headline, honestly:** Against the simple rating-difference formula (Baseline B), the gradient-boosted model was clearly better on 1 of 6 held-out comparisons, clearly worse on 3, and statistically indistinguishable on 2 (95% paired-bootstrap intervals). Both beat the 'always pick the home team' baseline. Treat this as a working, leak-free pipeline - not evidence of a betting edge.

## Data source actually used

**hoopR-mbb-data (ESPN-derived box scores) from public GitHub, season 2023-24** - the ESPN-derived game schedule and team box scores published in the public GitHub repo `sportsdataverse/hoopR-mbb-data`. Raw files are cached in `data/raw/`.

I tried the preferred sources first. Results from this run (`data/raw/source_probe.json`):

| Preferred source | Result from this run |
|---|---|
| barttorvik | unreachable |
| cbbdata_api | unreachable |
| espn_scoreboard | unreachable |

Barttorvik, the cbbdata API and ESPN's scoreboard were all blocked in the environment this was built in, so the GitHub dataset (fallback b in the brief) is what the results come from. Consequences: ratings are **computed by me from box scores** (no Barttorvik adjusted efficiency), and possessions are the standard box-score estimate.

- Games: 6240 completed games with box scores (2023-11-06 to 2024-04-08); first-half scores available for 6240 (taken from period scores and checked to add up to the final score); 744 neutral-site games.
- **Betting lines: none.** No free bulk historical closing lines for 2023-24 were found; none were invented.

## How to run

```bash
pip install -r requirements.txt
python run_all.py            # data -> features -> models -> backtest -> card -> README -> tests -> notebook
```

- Python 3.11+ (built and run on 3.13). Seeds are fixed (`SEED = 42`); raw downloads are cached, with a delay between requests.
- The PNG is rendered with Playwright. If its own Chromium is not installed, the code falls back to a system Chrome/Chromium; otherwise run `playwright install chromium`.
- Flags: `--skip-notebook`, `--skip-tests`. Tests alone: `python -m pytest -q`.
- **Have real closing lines?** Put `data/raw/lines.csv` with columns `game_id, spread_home (negative = home favoured), total` and re-run: the backtest switches to Case A (ATS / over-under hit rates and ROI, threshold picked on the January validation fold and evaluated on later months). That code path is only exercised by a unit test on synthetic data in this prototype.

## Method in brief

1. **No leakage.** Features for a game on date D use only games with date **before** D: cumulative team stats, an opponent-adjusted rating (ridge regression fitted on earlier games only) and league constants. Season-end ratings are never used. Games where either team has fewer than 8 prior games are skipped, leaving 4196 games (2023-12-10 to 2024-04-08).
2. **Proof.** `tests/test_leakage.py` checks that the latest date used per row is before the game date, that rebuilding features from data truncated at a cutoff reproduces the same features for earlier games, and that corrupting every future score changes nothing in the past.
3. **Models.** Baseline A (always home / home-court margin), Baseline B (efficiency-margin difference x expected tempo + home court), LightGBM (win probability, margin, total, 1H margin, 1H total) and one depth-3 decision tree. Win probabilities are Platt-calibrated on a time-ordered slice of the training window. Hyper-parameters are fixed, conservative defaults and were not tuned on test data.
4. **Validation.** Expanding-window walk-forward, never shuffled: train before Jan 1 / test Jan; train before Feb 1 / test Feb; train before Mar 1 / test Mar-Apr. All results below are on those 3587 held-out games only.

## Results (held-out games only)

### Moneyline (home-win probability)

| Model | ML accuracy | Log loss | Brier |
|---|---|---|---|
| Baseline A (always home) | 60.2% | 0.6760 | 0.2413 |
| Baseline B (rating formula) | 68.1% | 0.5941 | 0.2049 |
| Decision tree (depth 3) | 66.3% | 0.6126 | 0.2107 |
| LightGBM (uncalibrated) | 66.5% | 0.6086 | 0.2099 |
| LightGBM (Platt-calibrated) | 66.0% | 0.6085 | 0.2110 |

Calibration error (ECE, 10 equal-count bins): 0.057 uncalibrated, 0.051 after Platt scaling.

### Spread / total / 1st half (mean absolute error, points; lower is better)

| Target (points) | Baseline A | Baseline B | LightGBM | Games |
|---|---|---|---|---|
| Full-game margin | 10.71 | 9.57 | 9.43 | 3587 |
| Full-game total | 14.63 | 13.50 | 13.49 | 3587 |
| 1st-half margin | 7.84 | 7.30 | 7.42 | 3587 |
| 1st-half total | 8.42 | 7.94 | 8.03 | 3587 |

### Is the difference real? LightGBM minus Baseline B (paired bootstrap, 2000 resamples)

| LightGBM minus Baseline B | Difference | 95% bootstrap CI | Reading |
|---|---|---|---|
| Log loss (lower better) | +0.0144 | +0.0057 to +0.0228 | Baseline B better |
| ML accuracy (higher better) | -2.1 pts | -3.4 pts to -0.8 pts | Baseline B better |
| Full-game margin MAE | -0.14 | -0.26 to -0.02 | LightGBM better |
| Full-game total MAE | -0.01 | -0.17 to +0.16 | statistically indistinguishable |
| 1st-half margin MAE | +0.12 | +0.05 to +0.20 | Baseline B better |
| 1st-half total MAE | +0.10 | -0.00 to +0.19 | statistically indistinguishable |

Games are resampled as if independent, which is only approximate (teams repeat and same-day games are correlated), so treat the intervals as indicative.

### By fold

| Fold (test window) | Games | Acc. LightGBM | Acc. Baseline B | Acc. Baseline A | Log loss LightGBM | Log loss Baseline B | Margin MAE LightGBM | Margin MAE Baseline B |
|---|---|---|---|---|---|---|---|---|
| 2024-01-01 to 2024-02-01 | 1386 | 64.7% | 67.7% | 60.0% | 0.632 | 0.603 | 9.65 | 9.71 |
| 2024-02-01 to 2024-03-01 | 1395 | 67.0% | 70.0% | 60.1% | 0.586 | 0.579 | 9.41 | 9.50 |
| 2024-03-01 to 2024-05-01 | 806 | 66.5% | 65.5% | 60.8% | 0.608 | 0.604 | 9.08 | 9.46 |

### Charts

![Calibration](output/calibration.png)
![Error by target](output/mae_comparison.png)
![Feature importance](output/feature_importance.png)

Top drivers of the win-probability model (final fold, share of split gain): Opponent-adjusted rating edge 27%, Efficiency-margin edge 16%, Offensive efficiency edge 8%, Last-10 average-margin edge 5%, Combined defensive efficiency 4%, Last-5 average-margin edge 4%, Combined offensive efficiency 4%, Last-5 win-rate edge 3%.

### Plain-English decision tree (depth 3, fitted on games before Mar 1)

```text
IF Opponent-adjusted rating edge (home minus away) is at most -9.58 THEN home team wins 12% of the time (n=125 training games)
IF Opponent-adjusted rating edge (home minus away) is between -9.58 and -5.92 THEN home team wins 30% of the time (n=314 training games)
IF Opponent-adjusted rating edge (home minus away) is between -5.92 and -0.21 AND Efficiency-margin edge (home minus away) is at most -3.26 THEN home team wins 47% of the time (n=800 training games)
IF Opponent-adjusted rating edge (home minus away) is between -5.92 and -0.21 AND Efficiency-margin edge (home minus away) is above -3.26 THEN home team wins 61% of the time (n=320 training games)
IF Opponent-adjusted rating edge (home minus away) is between -0.21 and 3.80 AND Last-10 average-margin edge (home minus away) is at most 5.22 THEN home team wins 65% of the time (n=664 training games)
IF Opponent-adjusted rating edge (home minus away) is between -0.21 and 3.80 AND Last-10 average-margin edge (home minus away) is above 5.22 THEN home team wins 77% of the time (n=266 training games)
IF Opponent-adjusted rating edge (home minus away) is between 3.80 and 6.83 THEN home team wins 82% of the time (n=508 training games)
IF Opponent-adjusted rating edge (home minus away) is above 6.83 THEN home team wins 94% of the time (n=393 training games)
```

Tree held-out accuracy 66.3% / log loss 0.6126 (all folds, each fold refit).

## ROI backtest

**Case B - no real lines. No real ROI is reported.**

> ROI could not be validated because free historical betting lines were not available; with the client's API access to closing lines, this is the first thing I would test.

ATS and over/under hit rates also require real lines, so they are not computed.

For transparency only, I simulated flat 1-unit bets at -110 against a **proxy line = Baseline B's own number**. This is labelled **PROXY - NOT REAL MARKET RESULTS** and it does **not** measure edge against a sportsbook: Baseline B is a raw, unsmoothed formula, so a smoother model disagreeing with it says nothing about beating a market line that already prices in injuries, lineups and sharp money. Do not quote these figures as betting performance. The threshold was fixed in advance at 2.0 points; all thresholds are shown and none was selected. Break-even win rate at -110 is 52.38%.

**Spread (proxy)**

| Edge threshold (pts) | Bets | Win rate | Units | ROI | Max drawdown (units) |
|---|---|---|---|---|---|
| 0.5 | 3098 | 57.2% | +286.8 | +9.3% | 26.3 |
| 1 | 2731 | 58.0% | +291.1 | +10.7% | 18.1 |
| 2 | 1954 | 59.5% | +264.4 | +13.5% | 24.8 |
| 3 | 1405 | 61.4% | +242.5 | +17.3% | 13.8 |
| 4 | 946 | 61.8% | +170.8 | +18.1% | 9.0 |

**Total (proxy)**

| Edge threshold (pts) | Bets | Win rate | Units | ROI | Max drawdown (units) |
|---|---|---|---|---|---|
| 0.5 | 3284 | 56.0% | +224.9 | +6.8% | 19.6 |
| 1 | 3032 | 56.7% | +249.7 | +8.2% | 16.5 |
| 2 | 2565 | 57.5% | +252.8 | +9.9% | 16.0 |
| 3 | 2096 | 58.5% | +246.5 | +11.8% | 14.7 |
| 4 | 1689 | 59.3% | +222.0 | +13.1% | 11.9 |

Bootstrap intervals at the 2.0-point threshold - spread: win rate 59.5% (95% CI 57.2% to 61.6%), ROI +13.5% (95% CI +9.2% to +17.6%); total: win rate 57.5% (95% CI 55.7% to 59.4%), ROI +9.9% (95% CI +6.4% to +13.4%).

![Proxy cumulative profit](output/proxy_cumulative_profit.png)

## Game card

Built from a real held-out game using only pre-game information. Selection rule (fixed, not result-based): highest combined pre-game opponent-adjusted rating among held-out, regular-season, non-neutral games. Game: **Auburn at Tennessee**, 2024-02-28, scored by the model trained only on earlier games.

![Game card](output/game_card.png)

For transparency, not shown on the card: the final home margin was 8 and the moneyline pick was correct. One game proves nothing; the card is a demo of the UI. Its "drivers" list the largest contributions to the uncalibrated model and can include factors that cut against the pick.

`output/game_card.html` is self-contained (inline CSS, no CDN) so it renders identically offline.

## Limitations

- **One season, small samples.** 3587 held-out games; one fold has only 806. Intervals are wide for anything fine-grained.
- **Did not beat a simple formula everywhere.** See the paired-bootstrap table above; the main model's value here is the process and pipeline, not a demonstrated edge.
- **No injuries or lineups, no line movement, no real market lines.** These are the biggest gaps versus a sportsbook.
- **Overfitting risk.** 22 features and a few thousand training games per fold. Hyper-parameters were fixed, not tuned; with tuning, take care to tune on validation data only.
- **Data caveats.** ESPN-derived box scores (not Barttorvik); estimated possessions; early-season games against non-D1 teams are noisy, so as-of league constants are noisier in December (they are computed over eligible games only); ESPN's "home" label on neutral-site games is arbitrary, so home-court is zeroed for them.
- **First-half baselines are rough.** Baseline B's 1st-half numbers are scaled from its full-game numbers (assumption), not fitted.
- **Proxy ROI is not a result.** See above.
- **Bootstrap treats games as independent.**

## How I'd extend this for your project

- **Lines:** pull historical closing spreads/totals/moneylines via your API access (e.g. the CFBData college-basketball API) and run the Case A backtest first - thresholds chosen on a validation window, ROI reported with intervals.
- **Better ratings:** add Evan Miya-style (lineup/possession-adjusted) ratings as as-of features.
- **Lineups and injuries:** minutes-weighted, lineup-based features and availability flags.
- **1st-half models:** fit them on their own features (early-game pace, 1H shooting) instead of reusing full-game ones; add a 1H moneyline.
- **More data:** several seasons, rolling-season training, proper hyper-parameter tuning on validation windows, probability calibration per fold.
- **Deployment:** Supabase (storage + scheduled jobs) -> API on Render -> card UI on Vercel; nightly pull, predict and card render.
- **Monitoring:** weekly tracking of calibration, log loss, MAE and (with real lines) CLV/ROI, plus a data-drift check, with alerts when metrics degrade.

## Project layout

```text
src/            data.py  features.py  models.py  backtest.py  card.py  report.py  notebook.py  config.py
notebooks/      analysis.ipynb  (executed walkthrough)
templates/      card.html.j2  README.md.j2
tests/          test_leakage.py  test_metrics.py
data/raw        cached downloads        data/processed   games / features / predictions
output/         game_card.png  game_card.html  charts  results.json  summary.md  decision_tree_rules.txt
```

## Summary for a proposal (131 words, generated from this run)

> I built a leak-free college basketball prediction prototype on the 2023-24 season: 6,240 games of public ESPN-derived box scores, validated walk-forward on 3,587 held-out games from January to April. Home-win accuracy was 66.0% for gradient-boosted trees, 68.1% for a simple rating formula, and 60.2% for always picking the home team. Average error on the final margin was 9.4 points (formula 9.6, home-only 10.7); on game totals 13.5 (formula 13.5). The tree model did not beat the simple formula on win probability, so this is a working pipeline, not a proven edge. ROI was not validated: free historical betting lines were not available, so I report no profit figures. With your closing-line access, that is the first thing I would test. The package includes calibration, leakage tests and a shareable game card.