import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson as scipy_poisson

# Import bracket logic from same directory
sys.path.insert(0, str(Path(__file__).parent))
from bracket_logic import (
    assign_third_place_to_slots,
    resolve_r32_matchups,
    play_knockouts,
)

DATA = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"
N_SIMS = 10000
BONUS_PER_SD = 75.0

elo_orig = pd.read_csv(DATA / "current_elo.csv").set_index("team")["elo"].to_dict()
fixtures = pd.read_csv(RAW / "fixtures_2026.csv")
sv = pd.read_csv(RAW / "squad_values.csv")
with open(DATA / "score_model.json") as f:
    sm = json.load(f)

SCALE = sm["feature_scale"]
HOME_ADV = sm["home_advantage_elo"]
MAX_GOALS = sm["max_goals"]

sv["log_sv"] = np.log(sv["squad_value_m_eur"])
mean_log = sv["log_sv"].mean()
std_log = sv["log_sv"].std()
sv["z"] = (sv["log_sv"] - mean_log) / std_log
sv["elo_bonus"] = sv["z"] * BONUS_PER_SD
sv["elo_orig"] = sv["team"].map(elo_orig)
sv["elo_adj"] = sv["elo_orig"] + sv["elo_bonus"]

print("=" * 80)
print(f"{'Team':22s} {'SquadM':>9s} {'EloOrig':>8s} {'Bonus':>7s} {'EloAdj':>8s}")
print("=" * 80)
for _, r in sv.sort_values("elo_adj", ascending=False).iterrows():
    print(f"{r['team']:22s} {r['squad_value_m_eur']:>9.1f} {r['elo_orig']:>8.0f} "
          f"{r['elo_bonus']:>+7.0f} {r['elo_adj']:>8.0f}")

elo = dict(zip(sv["team"], sv["elo_adj"]))


def get_lambdas(team_a, team_b):
    ea, eb = elo.get(team_a, 1500), elo.get(team_b, 1500)
    elo_diff = (ea - eb) / SCALE
    elo_avg = ((ea + eb) / 2 - 1500) / SCALE
    lam_a = np.exp(sm["home_intercept"] + sm["home_coef"][0]*elo_diff + sm["home_coef"][1]*elo_avg)
    lam_b = np.exp(sm["away_intercept"] + sm["away_coef"][0]*elo_diff + sm["away_coef"][1]*elo_avg)
    return lam_a, lam_b


def simulate_match(ta, tb, must_have_winner=False, rng=None):
    lam_a, lam_b = get_lambdas(ta, tb)
    sa = min(rng.poisson(lam_a), MAX_GOALS)
    sb = min(rng.poisson(lam_b), MAX_GOALS)
    if must_have_winner and sa == sb:
        ea, eb = elo.get(ta, 1500), elo.get(tb, 1500)
        p = 1/(1+10**((eb-ea)/400)); p = 0.5 + 0.3*(p-0.5)
        return sa, sb, (ta if rng.random()<p else tb)
    if sa > sb: return sa, sb, ta
    if sb > sa: return sa, sb, tb
    return sa, sb, None


def simulate_group(gf, rng):
    teams = set(gf["home_team"]) | set(gf["away_team"])
    st = {t: {"pts":0,"gf":0,"ga":0} for t in teams}
    for _, m in gf.iterrows():
        h, a = m["home_team"], m["away_team"]
        sh, sa, _ = simulate_match(h, a, rng=rng)
        st[h]["gf"]+=sh; st[h]["ga"]+=sa; st[a]["gf"]+=sa; st[a]["ga"]+=sh
        if sh>sa: st[h]["pts"]+=3
        elif sh<sa: st[a]["pts"]+=3
        else: st[h]["pts"]+=1; st[a]["pts"]+=1
    for t in st: st[t]["gd"]=st[t]["gf"]-st[t]["ga"]
    r = sorted(st.items(), key=lambda x:(x[1]["pts"],x[1]["gd"],x[1]["gf"],rng.random()), reverse=True)
    return [t[0] for t in r], st


# Stage code -> label (for output JSON / dashboard)
_STAGE_LABEL = {1: "R32", 2: "R16", 3: "QF", 4: "SF", 5: "Final", 6: "Champion"}


def simulate_tournament(rng):
    """v2: knockouts follow the official 2026 FIFA bracket structure."""
    groups = list("ABCDEFGHIJKL")
    gr, st_by_g = {}, {}
    for g in groups:
        ranked, st = simulate_group(fixtures[fixtures["group"]==g], rng)
        gr[g] = ranked
        st_by_g[g] = st

    # 8 best 3rd-place teams (FIFA tiebreakers: points, GD, GF)
    thirds_pool = [
        (g, gr[g][2],
         st_by_g[g][gr[g][2]]["pts"],
         st_by_g[g][gr[g][2]]["gd"],
         st_by_g[g][gr[g][2]]["gf"])
        for g in groups
    ]
    thirds_pool.sort(key=lambda x: (x[2], x[3], x[4], rng.random()), reverse=True)
    qualifying_thirds = thirds_pool[:8]

    # Build the per-slot inputs for bracket_logic
    group_winners = {g: gr[g][0] for g in groups}
    group_runners_up = {g: gr[g][1] for g in groups}
    qualifying_groups = [t[0] for t in qualifying_thirds]
    thirds_by_group = {t[0]: t[1] for t in qualifying_thirds}

    # Initialize stage labels
    stage = {}
    for g in groups:
        stage[gr[g][0]] = "R32"
        stage[gr[g][1]] = "R32"
        stage[gr[g][2]] = "group_stage"  # 3rd: may be upgraded to R32 below
        stage[gr[g][3]] = "group_stage"
    for _, team, *_ in qualifying_thirds:
        stage[team] = "R32"

    # FIFA 3rd-place slot assignment + concrete R32 matchups
    third_assign = assign_third_place_to_slots(qualifying_groups)
    matchups = resolve_r32_matchups(group_winners, group_runners_up,
                                    thirds_by_group, third_assign)

    # Play through the knockouts using simulate_match as the winner oracle
    def match_winner_fn(a, b):
        _, _, w = simulate_match(a, b, must_have_winner=True, rng=rng)
        return w

    winners, deepest = play_knockouts(matchups, match_winner_fn)

    # Upgrade stage labels for teams that advanced past R32
    for team, depth in deepest.items():
        stage[team] = _STAGE_LABEL[depth]

    return stage


def predict_match(ta, tb):
    lam_a, lam_b = get_lambdas(ta, tb)
    ph = scipy_poisson.pmf(np.arange(MAX_GOALS+1), lam_a)
    pa = scipy_poisson.pmf(np.arange(MAX_GOALS+1), lam_b)
    M = np.outer(ph, pa)
    pw = np.tril(M,-1).sum(); pd_ = np.diag(M).sum(); pl = np.triu(M,1).sum()
    tot = pw+pd_+pl
    i,j = np.unravel_index(M.argmax(), M.shape)
    return {"p_home_win":float(pw/tot),"p_draw":float(pd_/tot),"p_away_win":float(pl/tot),
            "expected_home_goals":float(lam_a),"expected_away_goals":float(lam_b),
            "most_likely_score":f"{i}-{j}"}


print("\nComputing per-match predictions...")
match_preds = []
for _, m in fixtures.iterrows():
    p = predict_match(m["home_team"], m["away_team"])
    match_preds.append({"match_id":int(m["match_id"]),"date":m["date"],"group":m["group"],
                        "home_team":m["home_team"],"away_team":m["away_team"],"venue":m["venue"], **p})

print(f"Running {N_SIMS} simulations (v2 bracket-aware)...")
rng = np.random.RandomState(42)
all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
sl = ["group_stage","R32","R16","QF","SF","Final","Champion"]
counts = {t:{s:0 for s in sl} for t in all_teams}
for i in range(N_SIMS):
    if i and i%2000==0: print(f"  {i}/{N_SIMS}")
    sr = simulate_tournament(rng)
    for t,s in sr.items(): counts[t][s]+=1

results = []
for t in all_teams:
    c = counts[t]
    results.append({"team":t,"elo_orig":elo_orig.get(t,1500),"elo_adj":elo.get(t,1500),
        "p_r32":sum(c[s] for s in ["R32","R16","QF","SF","Final","Champion"])/N_SIMS,
        "p_r16":sum(c[s] for s in ["R16","QF","SF","Final","Champion"])/N_SIMS,
        "p_qf":sum(c[s] for s in ["QF","SF","Final","Champion"])/N_SIMS,
        "p_sf":sum(c[s] for s in ["SF","Final","Champion"])/N_SIMS,
        "p_final":sum(c[s] for s in ["Final","Champion"])/N_SIMS,
        "p_win":c["Champion"]/N_SIMS})

results.sort(key=lambda r:r["p_win"], reverse=True)
print(f"\n{'Team':22s} {'EloA':>5s} {'R32':>6s} {'R16':>6s} {'QF':>6s} {'SF':>6s} {'Fin':>6s} {'Win':>6s}")
print("-"*72)
for r in results:
    print(f"{r['team']:22s} {r['elo_adj']:5.0f} {r['p_r32']:>6.3f} {r['p_r16']:>6.3f} "
          f"{r['p_qf']:>6.3f} {r['p_sf']:>6.3f} {r['p_final']:>6.3f} {r['p_win']:>6.3f}")

out = {"match_predictions":match_preds,"tournament_simulation":results,"n_sims":N_SIMS,
       "bonus_per_sd":BONUS_PER_SD,"model_version":"elo+squad_value+fifa_bracket"}
with open(DATA / "predictions_adjusted.json","w") as f: json.dump(out,f,indent=2)
print(f"\nSaved to predictions_adjusted.json")
