import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
PRED = ROOT / "predictions"

locked_files = sorted(PRED.glob("dual_bracket_*.json"))
if not locked_files:
    raise FileNotFoundError("No locked predictions found.")
with open(locked_files[-1]) as f:
    lock = json.load(f)
print(f"Loaded: {locked_files[-1].name}")
print(f"Locked at: {lock['locked_at_utc']}\n")

results = pd.read_csv(RAW / "results.csv")
played = results.dropna(subset=["home_score", "away_score"]).copy()
if len(played) > 0:
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

print(f"Matches played so far: {len(played)}/72")

if len(played) == 0:
    print("\nNo matches played yet. The tournament starts tomorrow!")
    print("After each match, edit data/raw/results.csv to add scores and re-run this script.\n")
    print("Locked predictions summary:")
    print(f"  ML champion:   {lock['ml_model']['champion']}")
    print(f"  Your champion: {lock['fan_bracket']['champion']}")
    exit()

with open(DATA / "predictions_adjusted.json") as f:
    ml_preds = json.load(f)
match_preds = {p["match_id"]: p for p in ml_preds["match_predictions"]}

print("\n" + "=" * 78)
print("PER-MATCH ML MODEL ACCURACY")
print("=" * 78)
log_losses = []
correct_outcome = 0
correct_score = 0
for _, r in played.iterrows():
    mid = int(r["match_id"])
    pred = match_preds.get(mid)
    if not pred: continue
    if r["home_score"] > r["away_score"]:
        actual, p_actual = "home_win", pred["p_home_win"]
    elif r["home_score"] < r["away_score"]:
        actual, p_actual = "away_win", pred["p_away_win"]
    else:
        actual, p_actual = "draw", pred["p_draw"]
    log_losses.append(-np.log(max(p_actual, 1e-10)))
    probs = [pred["p_home_win"], pred["p_draw"], pred["p_away_win"]]
    labels = ["home_win", "draw", "away_win"]
    pred_outcome = labels[int(np.argmax(probs))]
    if pred_outcome == actual: correct_outcome += 1
    actual_score = f"{r['home_score']}-{r['away_score']}"
    if actual_score == pred["most_likely_score"]: correct_score += 1
    flag = "OK" if pred_outcome == actual else "  "
    print(f"  M{mid:>3d}: {r['home_team']:>22s} {actual_score} {r['away_team']:<22s}  pred {pred['most_likely_score']}  P(actual)={p_actual:.2f} {flag}")

print(f"\n  ML outcome accuracy:    {correct_outcome}/{len(played)} = {correct_outcome/len(played):.1%}")
print(f"  ML exact score correct: {correct_score}/{len(played)} = {correct_score/len(played):.1%}")
print(f"  ML mean log loss:       {np.mean(log_losses):.4f}  (baseline ~1.04, our test set was 0.89)")

print("\n" + "=" * 78)
print("CURRENT GROUP STANDINGS")
print("=" * 78)
groups = sorted(played["group"].unique())
group_standings = {}
for g in groups:
    g_played = played[played["group"] == g]
    teams = set(g_played["home_team"]) | set(g_played["away_team"])
    st = {t: {"pts":0,"gf":0,"ga":0,"pl":0} for t in teams}
    for _, m in g_played.iterrows():
        h, a, hs, as_ = m["home_team"], m["away_team"], m["home_score"], m["away_score"]
        st[h]["pl"]+=1; st[a]["pl"]+=1
        st[h]["gf"]+=hs; st[h]["ga"]+=as_
        st[a]["gf"]+=as_; st[a]["ga"]+=hs
        if hs>as_: st[h]["pts"]+=3
        elif hs<as_: st[a]["pts"]+=3
        else: st[h]["pts"]+=1; st[a]["pts"]+=1
    for t in st: st[t]["gd"] = st[t]["gf"] - st[t]["ga"]
    ranked = sorted(st.items(), key=lambda x:(x[1]["pts"],x[1]["gd"],x[1]["gf"]), reverse=True)
    group_standings[g] = ranked
    print(f"\nGroup {g}:")
    for i, (team, s) in enumerate(ranked):
        ml_picked = team in lock["ml_model"]["r32_teams"]
        fan_picked = team in lock["fan_bracket"]["r32_teams"]
        markers = []
        if ml_picked: markers.append("ML")
        if fan_picked: markers.append("YOU")
        mtag = "[" + ",".join(markers) + "]" if markers else ""
        print(f"  {i+1}. {team:24s} P{s['pl']} pts={s['pts']:2d} gd={s['gd']:+d} gf={s['gf']} {mtag}")

if len(played) == 72:
    actual_advancing = set()
    thirds = []
    for g, ranked in group_standings.items():
        actual_advancing.add(ranked[0][0])
        actual_advancing.add(ranked[1][0])
        if len(ranked) >= 3:
            t = ranked[2][0]
            thirds.append((t, ranked[2][1]["pts"], ranked[2][1]["gd"], ranked[2][1]["gf"]))
    thirds.sort(key=lambda x:(x[1],x[2],x[3]), reverse=True)
    for t in thirds[:8]: actual_advancing.add(t[0])
    print("\n" + "=" * 78)
    print("R32 FIELD: ACTUAL vs. PREDICTED")
    print("=" * 78)
    ml_r32 = set(lock["ml_model"]["r32_teams"])
    fan_r32 = set(lock["fan_bracket"]["r32_teams"])
    print(f"  ML correctly picked:  {len(ml_r32 & actual_advancing)}/32 advancing teams")
    print(f"  You correctly picked: {len(fan_r32 & actual_advancing)}/32 advancing teams")

print("\n" + "=" * 78)
print("Re-run this script after each match day.")
print("=" * 78)
