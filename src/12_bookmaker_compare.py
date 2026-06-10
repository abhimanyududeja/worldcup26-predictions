"""
src/12_bookmaker_compare.py

Compare ML model predictions to current bookmaker odds.
Source: FanDuel Sportsbook via FOX Sports, snapshot June 2, 2026.

Convert American odds -> implied probabilities -> normalize to remove
the bookmaker's vig (overround typically 5-15%). The vig-free probabilities
represent the market's consensus probability for each team to win the WC.

The market is brutally efficient, so the model is unlikely to "beat" it
in expectation. The point is to surface where they disagree most.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"

# FanDuel Sportsbook (via FOX Sports), June 2 2026.
ODDS_RAW = {
    "Spain": 475, "France": 500, "England": 650, "Brazil": 850,
    "Argentina": 900, "Portugal": 1000, "Germany": 1400,
    "Netherlands": 2200, "Norway": 3500, "Belgium": 3500,
    "Colombia": 4000, "Morocco": 5000, "Uruguay": 5000,
    "United States": 6000, "Switzerland": 6500, "Japan": 6500,
    "Mexico": 8000, "Croatia": 8000, "Ecuador": 8000,
    "Senegal": 9000, "Sweden": 10000,
}

def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

with open(PROC / "predictions_adjusted.json") as f:
    detailed = json.load(f)
model_probs = {t["team"]: t["p_win"] for t in detailed["tournament_simulation"]}

raw_implied = {team: american_to_implied(odds) for team, odds in ODDS_RAW.items()}
overround = sum(raw_implied.values())
print(f"Bookmaker overround (vig): {(overround - 1) * 100:.1f}%")
print(f"(Sum of raw implied probabilities; >100% is the bookie's margin)\n")
vig_free = {team: p / overround for team, p in raw_implied.items()}

comparison = []
for team in ODDS_RAW:
    market_p = vig_free[team]
    model_p = model_probs.get(team, 0)
    comparison.append({
        "team": team,
        "american_odds": ODDS_RAW[team],
        "market_pct": market_p * 100,
        "model_pct": model_p * 100,
        "diff_pct": (model_p - market_p) * 100,
    })
comparison.sort(key=lambda x: -x["market_pct"])

print(f"{'Team':<22s} {'Market':>10s} {'Model':>10s} {'Diff':>10s}")
print("-" * 56)
for r in comparison:
    diff_str = f"{r['diff_pct']:+.1f}%"
    print(f"{r['team']:<22s} {r['market_pct']:>8.2f}%  {r['model_pct']:>8.2f}%  {diff_str:>10s}")

# Biggest disagreements
print(f"\nBiggest model OVER market (model more bullish):")
overs = sorted(comparison, key=lambda x: -x["diff_pct"])[:5]
for r in overs:
    print(f"  {r['team']:<22s} model {r['model_pct']:.1f}%, market {r['market_pct']:.1f}% (+{r['diff_pct']:.1f})")

print(f"\nBiggest market OVER model (market more bullish):")
unders = sorted(comparison, key=lambda x: x["diff_pct"])[:5]
for r in unders:
    print(f"  {r['team']:<22s} market {r['market_pct']:.1f}%, model {r['model_pct']:.1f}% ({r['diff_pct']:.1f})")

with open(PROC / "bookmaker_comparison.json", "w") as f:
    json.dump({
        "source": "FanDuel Sportsbook via FOX Sports",
        "as_of": "2026-06-02",
        "overround": overround,
        "comparison": comparison,
    }, f, indent=2)
print(f"\nSaved to data/processed/bookmaker_comparison.json")
