"""
13_results_to_json.py - Score the locked predictions against the finished
tournament and write data/processed/results_2026.json for the dashboard.

Group-stage scores come from data/raw/wc2026_results.csv. Knockout results are
cached to data/raw/wc2026_knockouts.csv on first run so the dashboard never
needs a network call.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SRC_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# ---- knockout results (cached) ------------------------------------------
ko_path = RAW / "wc2026_knockouts.csv"
if ko_path.exists():
    ko = pd.read_csv(ko_path)
else:
    allm = pd.read_csv(SRC_URL)
    ko = allm[(allm.tournament == "FIFA World Cup") & (allm.date >= "2026-06-25")][
        ["date", "home_team", "away_team", "home_score", "away_score"]
    ].sort_values("date").reset_index(drop=True)
    ko.to_csv(ko_path, index=False)
    print(f"cached {len(ko)} knockout matches -> {ko_path.name}")

final = ko.iloc[-1]
champion = final.home_team if final.home_score > final.away_score else final.away_team
runner_up = final.away_team if final.home_score > final.away_score else final.home_team

# ---- locked predictions --------------------------------------------------
lock = json.load(open(sorted((ROOT / "predictions").glob("dual_bracket_*.json"))[-1]))
ml_pick = lock["ml_model"]["champion"]
fan_pick = lock["fan_bracket"]["champion"]

# ---- group-stage scoring (mirrors 09_score_predictions.py) ---------------
res = pd.read_csv(RAW / "wc2026_results.csv").dropna(subset=["home_score", "away_score"])
preds = {p["match_id"]: p for p in json.load(open(PROC / "predictions_adjusted.json"))["match_predictions"]}

lls, correct, exact = [], 0, 0
for _, r in res.iterrows():
    p = preds.get(int(r.match_id))
    if not p:
        continue
    hs, as_ = int(r.home_score), int(r.away_score)
    actual = "home_win" if hs > as_ else ("away_win" if hs < as_ else "draw")
    probs = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
    labels = ["home_win", "draw", "away_win"]
    lls.append(-np.log(max(probs[labels.index(actual)], 1e-10)))
    if labels[int(np.argmax(probs))] == actual:
        correct += 1
    if f"{hs}-{as_}" == p.get("most_likely_score"):
        exact += 1

n = len(lls)
out = {
    "champion": champion,
    "runner_up": runner_up,
    "final_score": f"{int(final.home_score)}-{int(final.away_score)}",
    "final_date": str(final.date),
    "ml_pick": ml_pick,
    "fan_pick": fan_pick,
    "ml_correct": bool(ml_pick == champion),
    "fan_correct": bool(fan_pick == champion),
    "group_matches": int(n),
    "outcome_accuracy": round(correct / n, 4),
    "exact_scores": round(exact / n, 4),
    "live_log_loss": round(float(np.mean(lls)), 4),
    "baseline_log_loss": 1.043,
    "test_log_loss": 0.890,
    "knockouts": [
        {"date": str(r.date), "home": r.home_team, "away": r.away_team,
         "score": f"{int(r.home_score)}-{int(r.away_score)}"}
        for _, r in ko.tail(8).iterrows()
    ],
}
PROC.mkdir(parents=True, exist_ok=True)
json.dump(out, open(PROC / "results_2026.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "knockouts"}, indent=2))
