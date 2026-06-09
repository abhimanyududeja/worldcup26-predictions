# 2026 FIFA World Cup Prediction System
### ML model vs. my football intuition — both locked before kickoff

Two predictions for the 2026 FIFA World Cup (June 11 – July 19, 2026), both committed with a timestamp **before the first match was played**:

1. **An ML model** built from 49,378 historical international matches (Elo + bivariate Poisson scoring) blended with current Transfermarkt squad market values.
2. **My personal bracket** as a football fan — including bold calls the model wouldn't make.

As the tournament unfolds, both predictions are scored against reality. The point isn't to "beat" the model — it's to compare what a calibrated statistical system and a human with football knowledge each get right and wrong.

The model picks **Spain** as champion. I picked **France**. We agree on most R32 teams and diverge sharply at the semi-finals.

---

## Why I built this

I'm a football fan first. My previous ML projects (FundTank, a brain tumor classifier) taught me techniques, but I didn't have skin in the game on whether the model was actually right. The 2026 World Cup was different — it's something I genuinely care about, and it gave me a real test: lock predictions before kickoff and see how the model holds up under the tournament.

The biggest thing I learned was how much calibration and baseline comparisons actually matter. They're not the fun part of ML, and it's tempting to skip them. But without them, you can ship a fancy-looking model that's worse than just picking the higher-ranked team every time, and not even know it. Every model in this project is compared against baselines (always-predict-class-frequencies and pure-Elo without my adjustments), with test log loss reported honestly.

If I had another month, the next thing I'd add is player-level features. The model treats Brazil with peak Neymar the same as Brazil without him. Squad market value gets part of the way there — it dropped Cristiano Ronaldo to a low value because he's 41 — but real player-level data (injuries, fitness, current form) is the obvious next step.

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
Two Poisson regressions (home goals, away goals) with features `[elo_diff, elo_avg]`, scaled by /400 to keep the optimizer stable. **Test log loss: 0.887** — slightly better than the standalone match model because score modeling carries more information than 3-way classification.

### Squad Value Adjustment
Elo alone underrates teams whose squads are stronger than their recent results (Brazil, Portugal, Germany) and overrates teams whose qualifying campaigns inflated their Elo (Colombia, Ecuador). I pull current Transfermarkt squad values, z-score the log of squad value across the 48 teams, and add `75 × z` Elo points. Result: Portugal +98 Elo, Brazil +92, Germany +97; Australia -84, Iran -93.

### Tournament Simulation
10,000 Monte Carlo runs of the bracket. Group stage uses the actual fixture list and FIFA tiebreakers (points, GD, GF). 8 best 3rd-place teams advance per the 48-team format. Knockouts use a simplified Elo-seeded bracket (documented limitation, see below). Penalty shootouts are Elo-weighted coin flips, heavily softened.

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

The model and I largely agree through R16. The picture diverges sharply at SF — the model has Spain/Brazil/Germany/Ecuador, I have France/England/Netherlands/Brazil.

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
- Bookmaker odds as a second ground truth, blended with the model
- Dixon-Coles correction for low-score bias in the Poisson model
- Actual 2026 bracket structure (with the 495-combination 3rd-place rules)
- Calibration plot from post-tournament binning

---

## Results (updated as tournament unfolds)

*To be filled in after each match day. After the tournament, I'll add a "What happened" section reflecting honestly on which picks worked and which failed.*

---

## Tech Stack

Python, pandas, scikit-learn, scipy. Historical data from [github.com/martj42/international_results](https://github.com/martj42/international_results). Squad values aggregated from Transfermarkt via Sportingpedia (June 2026).

---

## Files

- `src/01_*` to `src/09_*` — pipeline scripts in run order
- `data/raw/results.csv` — historical international matches
- `data/raw/fixtures_2026.csv` — 72 group-stage fixtures
- `data/raw/squad_values.csv` — Transfermarkt squad values for 48 teams
- `predictions/dual_bracket_*.json` — the locked, immutable predictions
