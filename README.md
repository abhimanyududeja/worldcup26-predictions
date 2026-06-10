# 2026 FIFA World Cup Prediction System
### ML model vs. my football intuition - both locked before kickoff

Two predictions for the 2026 FIFA World Cup (June 11 – July 19, 2026), both committed with a timestamp **before the first match was played**:

1. **An ML model** built from 49,378 historical international matches (Elo + bivariate Poisson scoring) blended with current Transfermarkt squad market values.
2. **My personal bracket** as a football fan - including bold calls the model wouldn't make.

As the tournament unfolds, both predictions are scored against reality. The point isn't to "beat" the model - it's to compare what a calibrated statistical system and a human with football knowledge each get right and wrong.

The model picks **Spain** as champion. I picked **France**. We agree on most R32 teams and diverge sharply at the semi-finals.

---

## Why I built this

I'm a football fan first. My previous ML projects (FundTank, a brain tumor classifier) taught me techniques, but I didn't have skin in the game on whether the model was actually right. The 2026 World Cup was different - it's something I genuinely care about, and it gave me a real test: lock predictions before kickoff and see how the model holds up under the tournament.

The biggest thing I learned was how much calibration and baseline comparisons actually matter. They're not the fun part of ML, and it's tempting to skip them. But without them, you can ship a fancy-looking model that's worse than just picking the higher-ranked team every time, and not even know it. Every model in this project is compared against baselines (always-predict-class-frequencies and pure-Elo without my adjustments), with test log loss reported honestly.

If I had another month, the next thing I'd add is player-level features. The model treats Brazil with peak Neymar the same as Brazil without him. Squad market value gets part of the way there - it dropped Cristiano Ronaldo to a low value because he's 41 - but real player-level data (injuries, fitness, current form) is the obvious next step.

---

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build the model (run in order)
python src/01_fetch_data.py
python src/02_elo.py
python src/02b_diagnose_elo.py
python src/03_verify_fixtures.py
python src/04_match_model.py
python src/05_score_model.py
python src/06_simulate.py
python src/06b_simulate_adjusted.py
python src/08_lock_dual_bracket.py

# After matches happen
python src/09_score_predictions.py
```

---

## Methodology

### Elo Ratings
Standard Elo, computed over all 49,378 matches from 1872 to today. Tuning:
- K-factor varies by tournament importance: friendlies 20, qualifiers 30, continental 50, World Cup 60
- Home advantage: +65 Elo (skipped when match is at a neutral venue)
- Goal-difference multiplier scaling with margin (matches the World Football Elo Ratings convention)

### Match Probability Model
Multinomial logistic regression on adjusted Elo difference. Trained on 25,316 matches from 2000 onward. **Test log loss: 0.890** vs. baseline 1.043 (14.7% improvement).

### Score Model
Two Poisson regressions (home goals, away goals) with features `[elo_diff, elo_avg]`, scaled by /400 to keep the optimizer stable. **Test log loss: 0.887** - slightly better than the standalone match model because score modeling carries more information than 3-way classification.

### Squad Value Adjustment
Elo alone underrates teams whose squads are stronger than their recent results (Brazil, Portugal, Germany) and overrates teams whose qualifying campaigns inflated their Elo (Colombia, Ecuador). I pull current Transfermarkt squad values, z-score the log of squad value across the 48 teams, and add `75 × z` Elo points. Result: Portugal +98 Elo, Brazil +92, Germany +97; Australia -84, Iran -93.

### Tournament Simulation
10,000 Monte Carlo runs of the bracket. Group stage uses the actual fixture list and FIFA tiebreakers (points, GD, GF). 8 best 3rd-place teams advance per the 48-team format. Knockouts use a simplified Elo-seeded bracket (documented limitation, see below). Penalty shootouts are Elo-weighted coin flips, heavily softened.

---

## Backtest validation

To validate the methodology on out-of-sample tournament data, the same Elo + bivariate Poisson core was retrained on data strictly before each tournament's start date (no lookahead bias) and run against the 2018 and 2022 World Cups.

### 2022 World Cup (Qatar)

- Model's top pick: Brazil (25.9% to win)
- Argentina (actual winner) ranked **#2** at 15.5%
- France (actual runner-up) ranked **#3** at 8.3%
- Both finalists in the top 5
- Log loss on 64 actual matches: **1.027** vs baseline 1.099 (**6.6% improvement**)
- Outcome accuracy: 56.2%

### 2018 World Cup (Russia)

- Model's top pick: Brazil (25.3% to win)
- France (actual winner) ranked **#4** at 8.9%
- Croatia (actual runner-up) ranked **#13** (real miss)
- Log loss on 64 actual matches: **0.978** vs baseline 1.099 (**11.0% improvement**)
- Outcome accuracy: 54.7%

### What the backtest validates and what it reveals

The methodology beats baseline by 6-11% on real out-of-sample tournament data, and identifies the actual finalists in the top 5 for both tournaments tested. That validates the core approach.

It also reveals a clear failure mode: **Brazil was the model's #1 pick in both 2018 and 2022, at 25%+ each time. Brazil won neither.** This is CONMEBOL Elo inflation: South American teams' Elo gets pumped by repeated qualifying matches against other high-Elo CONMEBOL teams.

The 2026 model's squad value adjustment partially corrects this: Brazil drops from a hypothetical #1 backtest pick (pure Elo) to #5 at 6.3% (2026 model with squad value). That validates the squad value adjustment as a meaningful methodological improvement, not just a heuristic.

The Croatia 2018 miss (rank #13) is a real failure mode. Almost no pre-tournament model predicted Croatia's final run, but it's a limitation worth surfacing.

### Backtest limitations

- Squad value adjustment NOT included in the backtest (no historical squad value data available). The backtest validates the Elo + Poisson core, not the full 2026 pipeline.
- Knockout bracket uses simplified Elo-seeded pairing, matching the actual 32-team WC format used in 2018 and 2022.
- Penalty shootouts modeled as Elo-weighted coin flips.

Reproduce with: `python src/10_backtest.py`

---

## Calibration analysis

The model's predicted probabilities were tested against actual outcomes from the 2018 and 2022 World Cup matches. For each match, three binary prediction events (home win, draw, away win) were extracted, giving 384 total prediction events. These were binned by predicted probability and compared to actual frequency.

| Predicted bin | Avg predicted | Actual freq | N |
|---|---|---|---|
| 0-10% | 7.6% | 28.6% | 7 |
| 10-20% | 15.2% | 15.4% | 52 |
| 20-30% | 25.9% | 23.2% | 177 |
| 30-40% | 35.2% | 43.2% | 37 |
| 40-50% | 45.4% | 41.3% | 46 |
| 50-60% | 54.7% | 60.7% | 28 |
| 60-70% | 63.9% | 69.0% | 29 |
| 70-80% | 73.5% | 71.4% | 7 |
| 80-90% | 85.2% | 0.0% | 1 |

**Expected Calibration Error (ECE): 4.00%** (0% is perfectly calibrated; lower is better).

The model is well-calibrated in the 10-50% range (the bins with the most samples). It is slightly underconfident in the 50-70% range, predicting 55-65% when the outcome actually happens 60-70% of the time. The extreme bins (0-10%, 80-90%) have too few samples to draw firm conclusions.

Reproduce with: `python src/11_calibration.py`

---

## Model vs. market

Comparison of the ML model's predictions against current FanDuel sportsbook odds (snapshot from June 2, 2026), with the bookmaker's 10.4% vig removed via normalization.

The market is a strong baseline because it aggregates the views of many informed bettors with skin in the game. If the model and market agree, the model is not doing something obviously wrong. If they disagree, the disagreement is itself a finding.

### Where market and model agree

The top tier (Spain, France, England, Brazil, Argentina, Portugal) appears at the top of both rankings. Where they differ is in the relative ordering and confidence within that tier.

### Three notable disagreements

**Portugal: market 8.2%, model 3.5%.** The market sees Portugal as a top-5 contender. The model has them 8th. This independently validates the Portugal-underrated call. The Elo-based model under-weights Portugal because of a weak qualifying group and slow decay from their EURO 2016 peak. Sophisticated bettors price Portugal much higher.

**Ecuador and Colombia: model is too bullish.** Model has Ecuador at 4.4% and Colombia at 4.5%. Market has them at 1.1% and 2.2%. That is the CONMEBOL inflation pattern visible in the 2018 and 2022 backtests, where Brazil was the model's #1 pick both times despite never winning. The 2026 squad value adjustment helps but does not fully fix this.

**Spain and France: market has them tied** (15.8% vs 15.1%); model has them far apart (Spain 21.6% vs France 10.7%). The market's view (co-favorites) aligns with my fan bracket pick (France over Spain) more than the model does.

Reproduce with: `python src/12_bookmaker_compare.py`

---

## Top 10 Champion Probabilities (Locked)

| Rank | Team | P(win) |
|---|---|---|
| 1 | Spain | 21.6% |
| 2 | France | 10.7% |
| 3 | Argentina | 10.2% |
| 4 | England | 8.5% |
| 5 | Brazil | 6.3% |
| 6 | Germany | 4.7% |
| 7 | Colombia | 4.5% |
| 8 | Ecuador | 4.4% |
| 9 | Netherlands | 4.2% |
| 10 | Norway | 3.5% |

---

## ML vs. Fan: Where We Differ

| Stage | Agreement |
|---|---|
| R32 field | 29/32 teams in common |
| R16 field | 14/16 teams in common |
| QF field | 5/8 teams in common |
| SF field | **1/4 teams in common** |
| Finalists | 1/2 teams in common (France) |
| Champion | DISAGREE: ML says Spain, I say France |

The model and I largely agree through R16. The picture diverges sharply at SF - the model has Spain/Brazil/Germany/Ecuador, I have France/England/Netherlands/Brazil.

---

## Honest Limitations

Real flaws that affect prediction quality, listed so readers can judge:

1. **Simplified knockout bracket.** The actual 2026 R32 has specific group-position slot assignments. My simulator uses Elo-seeded pairing (1v32, 2v31, …). This affects individual team paths but not aggregate progression probabilities much.

2. **No aggressive recency weighting.** A 2018 match still influences ratings even though squads have completely turned over. Elo decays old matches naturally, but probably not aggressively enough.

3. **Missing factors the model can't see:** injuries, manager quality, tactical setup, team chemistry, travel fatigue, tactical matchups.

4. **CONMEBOL Elo inflation.** South American teams play many tough qualifying matches against each other, inflating their Elo relative to UEFA teams that play weaker opposition. Squad-value adjustment partially corrects but doesn't fully fix.

5. **Estimates in squad value data.** ~10 of 48 teams (Curaçao, Haiti, Iraq, etc.) don't have authoritative published squad values. Those are best-effort estimates marked in `squad_values.csv`.

6. **Score model home/away asymmetry.** Trained on data where home/away is meaningful, applied to neutral WC matches. Small bias introduced.

7. **Penalty shootouts modeled crudely** as Elo-weighted coin flips.

---

## Future Work

- Player-level features (injuries, form, fitness, FIFA EA-style ratings)
- Recency-weighted Elo (exponential decay on matches older than 2-3 years)
- Blend the model with market odds as a hybrid prediction (we compare them now; blending is the next step)
- Dixon-Coles correction for low-score bias in the Poisson model
- Actual 2026 bracket structure (with the 495-combination 3rd-place rules)
- Continued calibration analysis as 2026 matches happen (pre-tournament calibration from backtests is already in place)

---

## Results (updated as tournament unfolds)

*To be filled in after each match day. After the tournament, I'll add a "What happened" section reflecting honestly on which picks worked and which failed.*

---

## Tech Stack

Python, pandas, scikit-learn, scipy. Historical data from [github.com/martj42/international_results](https://github.com/martj42/international_results). Squad values aggregated from Transfermarkt via Sportingpedia (June 2026).

---

## Files

- `src/01_*` to `src/09_*` - pipeline scripts in run order
- `data/raw/results.csv` - historical international matches
- `data/raw/fixtures_2026.csv` - 72 group-stage fixtures
- `data/raw/squad_values.csv` - Transfermarkt squad values for 48 teams
- `predictions/dual_bracket_*.json` - the locked, immutable predictions
