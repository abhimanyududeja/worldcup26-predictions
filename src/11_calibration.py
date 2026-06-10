"""
src/11_calibration.py

Compute model calibration from backtest per-match predictions.

A well-calibrated model: when it predicts 70% probability, the outcome
happens 70% of the time. We treat each match as 3 binary prediction
events (home win, draw, away win), bin by predicted probability, and
compare predicted vs actual frequency.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"

with open(PROC / "backtest_results.json") as f:
    backtest = json.load(f)

events = []
for year in ["2022", "2018"]:
    for m in backtest[year]["per_match"]:
        actual = m["actual"]
        events.append({"predicted": m["p_home_win"], "actual": 1 if actual == "H" else 0})
        events.append({"predicted": m["p_draw"], "actual": 1 if actual == "D" else 0})
        events.append({"predicted": m["p_away_win"], "actual": 1 if actual == "A" else 0})

print(f"Total binary prediction events: {len(events)}")

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
calibration = []
print(f"\n{'Bin':>15s} {'Predicted':>12s} {'Actual':>10s} {'N':>6s}")
for i in range(len(bins) - 1):
    low, high = bins[i], bins[i+1]
    in_bin = [e for e in events if low <= e["predicted"] < high]
    if not in_bin:
        continue
    avg_predicted = float(np.mean([e["predicted"] for e in in_bin]))
    avg_actual = float(np.mean([e["actual"] for e in in_bin]))
    n = len(in_bin)
    calibration.append({
        "bin_low": low, "bin_high": high,
        "avg_predicted": avg_predicted, "avg_actual": avg_actual, "n": n,
    })
    print(f"  {low*100:>3.0f}-{high*100:>3.0f}%   {avg_predicted*100:>9.1f}%  {avg_actual*100:>8.1f}%  {n:>5d}")

total_n = sum(c["n"] for c in calibration)
ece = sum(c["n"] * abs(c["avg_predicted"] - c["avg_actual"]) for c in calibration) / total_n
print(f"\nExpected Calibration Error: {ece*100:.2f}%")
print("(0% = perfect, lower is better)")

with open(PROC / "calibration.json", "w") as f:
    json.dump({
        "calibration": calibration,
        "n_total": len(events),
        "expected_calibration_error": ece,
    }, f, indent=2)
print(f"\nSaved to data/processed/calibration.json")
