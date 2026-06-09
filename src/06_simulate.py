import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson as scipy_poisson

DATA = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"
N_SIMS = 10000

elo = pd.read_csv(DATA / "current_elo.csv").set_index("team")["elo"].to_dict()
fixtures = pd.read_csv(RAW / "fixtures_2026.csv")
with open(DATA / "score_model.json") as f:
    sm = json.load(f)

SCALE = sm["feature_scale"]
HOME_ADV = sm["home_advantage_elo"]
MAX_GOALS = sm["max_goals"]


def get_lambdas(team_a, team_b, neutral=True):
    elo_a = elo.get(team_a, 1500)
    elo_b = elo.get(team_b, 1500)
    elo_diff = (elo_a + (0 if neutral else HOME_ADV) - elo_b) / SCALE
    elo_avg = ((elo_a + elo_b) / 2 - 1500) / SCALE
    lam_a = np.exp(sm["home_intercept"] + sm["home_coef"][0] * elo_diff + sm["home_coef"][1] * elo_avg)
    lam_b = np.exp(sm["away_intercept"] + sm["away_coef"][0] * elo_diff + sm["away_coef"][1] * elo_avg)
    return lam_a, lam_b


def simulate_match(team_a, team_b, must_have_winner=False, rng=None):
    lam_a, lam_b = get_lambdas(team_a, team_b)
    score_a = min(rng.poisson(lam_a), MAX_GOALS)
    score_b = min(rng.poisson(lam_b), MAX_GOALS)
    if must_have_winner and score_a == score_b:
        elo_a = elo.get(team_a, 1500)
        elo_b = elo.get(team_b, 1500)
        p = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        p = 0.5 + 0.3 * (p - 0.5)
        winner = team_a if rng.random() < p else team_b
        return score_a, score_b, winner
    if score_a > score_b:
        return score_a, score_b, team_a
    if score_b > score_a:
        return score_a, score_b, team_b
    return score_a, score_b, None


def simulate_group(group_fixtures, rng):
    teams = set(group_fixtures["home_team"]) | set(group_fixtures["away_team"])
    standings = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for _, m in group_fixtures.iterrows():
        h, a = m["home_team"], m["away_team"]
        s_h, s_a, _ = simulate_match(h, a, rng=rng)
        standings[h]["gf"] += s_h; standings[h]["ga"] += s_a
        standings[a]["gf"] += s_a; standings[a]["ga"] += s_h
        if s_h > s_a: standings[h]["pts"] += 3
        elif s_h < s_a: standings[a]["pts"] += 3
        else: standings[h]["pts"] += 1; standings[a]["pts"] += 1
    for t in standings:
        standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]
    ranked = sorted(standings.items(),
                    key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], rng.random()),
                    reverse=True)
    return [t[0] for t in ranked], standings


def simulate_tournament(rng):
    groups = list("ABCDEFGHIJKL")
    group_results = {}
    thirds = []
    for g in groups:
        ranked, standings = simulate_group(fixtures[fixtures["group"] == g], rng)
        group_results[g] = ranked
        thirds.append((ranked[2], standings[ranked[2]]["pts"],
                       standings[ranked[2]]["gd"], standings[ranked[2]]["gf"]))
    thirds.sort(key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)
    top8_thirds = [t[0] for t in thirds[:8]]

    stage = {}
    advancing = []
    for g in groups:
        advancing += [group_results[g][0], group_results[g][1]]
        stage[group_results[g][0]] = "R32"
        stage[group_results[g][1]] = "R32"
        stage[group_results[g][2]] = "group_stage"
        stage[group_results[g][3]] = "group_stage"
    for t in top8_thirds:
        advancing.append(t)
        stage[t] = "R32"

    advancing.sort(key=lambda t: elo.get(t, 1500), reverse=True)
    pairs = [(advancing[i], advancing[31-i]) for i in range(16)]

    def play_round(pairs):
        return [simulate_match(a, b, must_have_winner=True, rng=rng)[2] for a, b in pairs]

    r16 = play_round(pairs)
    for t in r16: stage[t] = "R16"
    qf = play_round([(r16[i], r16[i+1]) for i in range(0, 16, 2)])
    for t in qf: stage[t] = "QF"
    sf = play_round([(qf[i], qf[i+1]) for i in range(0, 8, 2)])
    for t in sf: stage[t] = "SF"
    final = play_round([(sf[i], sf[i+1]) for i in range(0, 4, 2)])
    for t in final: stage[t] = "Final"
    _, _, champion = simulate_match(final[0], final[1], must_have_winner=True, rng=rng)
    stage[champion] = "Champion"
    return stage


def predict_match(team_a, team_b):
    lam_a, lam_b = get_lambdas(team_a, team_b)
    p_h = scipy_poisson.pmf(np.arange(MAX_GOALS + 1), lam_a)
    p_a_arr = scipy_poisson.pmf(np.arange(MAX_GOALS + 1), lam_b)
    M = np.outer(p_h, p_a_arr)
    p_win = np.tril(M, -1).sum()
    p_draw = np.diag(M).sum()
    p_loss = np.triu(M, 1).sum()
    total = p_win + p_draw + p_loss
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {
        "p_home_win": float(p_win/total),
        "p_draw": float(p_draw/total),
        "p_away_win": float(p_loss/total),
        "expected_home_goals": float(lam_a),
        "expected_away_goals": float(lam_b),
        "most_likely_score": f"{i}-{j}",
    }


print("Computing per-match predictions for 72 group matches...")
match_preds = []
for _, m in fixtures.iterrows():
    pred = predict_match(m["home_team"], m["away_team"])
    match_preds.append({
        "match_id": int(m["match_id"]), "date": m["date"], "group": m["group"],
        "home_team": m["home_team"], "away_team": m["away_team"], "venue": m["venue"],
        **pred,
    })

print(f"\nRunning {N_SIMS} tournament simulations...")
rng = np.random.RandomState(42)
all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
stages_list = ["group_stage", "R32", "R16", "QF", "SF", "Final", "Champion"]
counts = {t: {s: 0 for s in stages_list} for t in all_teams}

for i in range(N_SIMS):
    if i and i % 2000 == 0:
        print(f"  {i}/{N_SIMS}")
    sr = simulate_tournament(rng)
    for t, s in sr.items():
        counts[t][s] += 1

results = []
for t in all_teams:
    c = counts[t]
    p_r32 = sum(c[s] for s in ["R32","R16","QF","SF","Final","Champion"]) / N_SIMS
    p_r16 = sum(c[s] for s in ["R16","QF","SF","Final","Champion"]) / N_SIMS
    p_qf  = sum(c[s] for s in ["QF","SF","Final","Champion"]) / N_SIMS
    p_sf  = sum(c[s] for s in ["SF","Final","Champion"]) / N_SIMS
    p_f   = sum(c[s] for s in ["Final","Champion"]) / N_SIMS
    p_w   = c["Champion"] / N_SIMS
    results.append({"team": t, "elo": elo.get(t,1500), "p_r32": p_r32, "p_r16": p_r16,
                    "p_qf": p_qf, "p_sf": p_sf, "p_final": p_f, "p_win": p_w})

results.sort(key=lambda r: r["p_win"], reverse=True)
print(f"\n{'Team':22s} {'Elo':>6s} {'R32':>6s} {'R16':>6s} {'QF':>6s} {'SF':>6s} {'Final':>7s} {'Win':>6s}")
print("-" * 75)
for r in results:
    print(f"{r['team']:22s} {r['elo']:6.0f} {r['p_r32']:>6.3f} {r['p_r16']:>6.3f} "
          f"{r['p_qf']:>6.3f} {r['p_sf']:>6.3f} {r['p_final']:>7.3f} {r['p_win']:>6.3f}")

out = {"match_predictions": match_preds, "tournament_simulation": results, "n_sims": N_SIMS}
with open(DATA / "predictions.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {DATA / 'predictions.json'}")
