import json
from pathlib import Path
from datetime import datetime, timezone

DATA = Path(__file__).parent.parent / "data" / "processed"
PRED_DIR = Path(__file__).parent.parent / "predictions"
PRED_DIR.mkdir(exist_ok=True)

with open(DATA / "predictions_adjusted.json") as f:
    preds = json.load(f)
sim = preds["tournament_simulation"]

def top_n_by(prob_key, n):
    return [r["team"] for r in sorted(sim, key=lambda x: x[prob_key], reverse=True)[:n]]

model_bracket = {
    "r32_teams": top_n_by("p_r32", 32),
    "r16_teams": top_n_by("p_r16", 16),
    "qf_teams": top_n_by("p_qf", 8),
    "sf_teams": top_n_by("p_sf", 4),
    "finalists": top_n_by("p_final", 2),
    "champion": top_n_by("p_win", 1)[0],
}

fan_bracket = {
    "group_winners": {
        "A": "Mexico", "B": "Switzerland", "C": "Brazil", "D": "Turkey",
        "E": "Germany", "F": "Netherlands", "G": "Belgium", "H": "Spain",
        "I": "France", "J": "Argentina", "K": "Portugal", "L": "Ghana"
    },
    "group_runners_up": {
        "A": "Czech Republic", "B": "Canada", "C": "Morocco", "D": "Australia",
        "E": "Ecuador", "F": "Japan", "G": "Iran", "H": "Uruguay",
        "I": "Norway", "J": "Algeria", "K": "Colombia", "L": "England"
    },
    "third_place_advancers": [
        "South Korea", "United States", "Croatia", "Senegal",
        "Sweden", "Egypt", "Ivory Coast", "Austria"
    ],
    "r32_teams": [
        "Mexico", "Switzerland", "Brazil", "Turkey", "Germany", "Netherlands",
        "Belgium", "Spain", "France", "Argentina", "Portugal", "Ghana",
        "Czech Republic", "Canada", "Morocco", "Australia", "Ecuador", "Japan",
        "Iran", "Uruguay", "Norway", "Algeria", "Colombia", "England",
        "South Korea", "United States", "Croatia", "Senegal",
        "Sweden", "Egypt", "Ivory Coast", "Austria"
    ],
    "r16_teams": ["Mexico","Morocco","Switzerland","Brazil","Turkey","Japan",
                  "Germany","Netherlands","Belgium","Norway","Spain","France",
                  "Argentina","Colombia","Portugal","England"],
    "qf_teams": ["Mexico","Brazil","Japan","Netherlands","Norway","France",
                 "Argentina","England"],
    "sf_teams": ["Brazil","Netherlands","France","England"],
    "finalists": ["Brazil","France"],
    "champion": "France",
}

ts = datetime.now(timezone.utc)
ts_str = ts.strftime("%Y%m%d_%H%M%S")

lock = {
    "locked_at_utc": ts.isoformat(),
    "tournament": "2026 FIFA World Cup",
    "ml_model": {**model_bracket, "description": "Elo + squad value, 10000 Monte Carlo simulations"},
    "ml_full_probs": sim,
    "fan_bracket": {**fan_bracket, "name": "Abhimanyu"},
    "scoring_method": "team appearances at each stage vs reality",
}

out_path = PRED_DIR / f"dual_bracket_{ts_str}.json"
with open(out_path, "w") as f:
    json.dump(lock, f, indent=2)

ml_champ = model_bracket['champion']
fan_champ = fan_bracket['champion']
champ_agree = ml_champ == fan_champ
champ_msg = "AGREE" if champ_agree else f"DISAGREE (ML: {ml_champ}, You: {fan_champ})"

print(f"LOCKED at {ts.isoformat()}")
print(f"File: {out_path.name}\n")
print("=" * 72)
print(f"{'Stage':12s} {'ML Pick':28s} | {'Your Pick':28s}")
print("=" * 72)
print(f"{'Champion':12s} {ml_champ:28s} | {fan_champ:28s}")
print(f"{'Finalists':12s} {' / '.join(model_bracket['finalists']):28s} | {' / '.join(fan_bracket['finalists']):28s}")
print(f"\n{'-' * 72}")
print(f"SF teams")
print(f"  ML:   {', '.join(sorted(model_bracket['sf_teams']))}")
print(f"  You:  {', '.join(sorted(fan_bracket['sf_teams']))}")
print(f"\nQF teams")
print(f"  ML:   {', '.join(sorted(model_bracket['qf_teams']))}")
print(f"  You:  {', '.join(sorted(fan_bracket['qf_teams']))}")
print(f"\n{'-' * 72}")
print(f"AGREEMENT BETWEEN ML AND YOUR PICKS:")
for stage, key in [("R32 field", "r32_teams"), ("R16 field", "r16_teams"),
                   ("QF field", "qf_teams"), ("SF field", "sf_teams"),
                   ("Finalists", "finalists")]:
    overlap = set(model_bracket[key]) & set(fan_bracket[key])
    n = len(model_bracket[key])
    print(f"  {stage:12s} {len(overlap)}/{n} teams in common")
print(f"  Champion     {champ_msg}")
print(f"\n{'=' * 72}")
print(f"This file is the commitment. Both brackets are now frozen.")
print(f"As matches happen, we score both against reality.")
