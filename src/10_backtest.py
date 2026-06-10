"""
src/10_backtest.py

Backtest the model on past World Cups (2022 and 2018) to validate
the Elo + bivariate Poisson methodology on out-of-sample tournament data.

Process per tournament:
  1. Filter historical matches to those strictly before the WC start date.
  2. Recompute Elo from scratch on the filtered set (no lookahead).
  3. Retrain match and score models on the filtered set's running Elo features.
  4. For each ACTUAL tournament match, predict probabilities + compute log loss.
  5. Run Monte Carlo simulations of the 32-team tournament format.
  6. Report top picks and where the actual finalists ranked.

Limitations:
  - Squad value adjustment NOT included (no historical squad value data).
    Backtest validates the Elo + Poisson core, not the full 2026 pipeline.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson as scipy_poisson

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

TOURNAMENTS = {
    2022: {
        "name": "2022 FIFA World Cup (Qatar)",
        "start": "2022-11-20", "end": "2022-12-18",
        "winner": "Argentina", "runner_up": "France",
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
    2018: {
        "name": "2018 FIFA World Cup (Russia)",
        "start": "2018-06-14", "end": "2018-07-15",
        "winner": "France", "runner_up": "Croatia",
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
}

HOME_ADV = 65
MAX_GOALS = 8
CONTINENTAL_KEYS = ("UEFA Euro", "Copa America", "African Cup of Nations",
                    "AFC Asian Cup", "CONCACAF", "Gold Cup")

def k_for(t):
    if not isinstance(t, str): return 20
    if "FIFA World Cup" in t and "qualification" not in t.lower(): return 60
    if "qualification" in t.lower(): return 30
    if any(k in t for k in CONTINENTAL_KEYS): return 50
    return 20


def compute_elo(matches_df, record_features=False):
    elo = defaultdict(lambda: 1500.0)
    df = matches_df.sort_values("date").reset_index(drop=True)
    features = []
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m.get("home_score"), m.get("away_score")
        if pd.isna(hs) or pd.isna(as_): continue
        hs, as_ = int(hs), int(as_)
        neutral = bool(m.get("neutral", False))
        ha = 0 if neutral else HOME_ADV
        elo_h_adj = elo[h] + ha
        elo_a_adj = elo[a]
        if record_features:
            features.append({
                "date": m["date"],
                "elo_diff": elo_h_adj - elo_a_adj,
                "elo_avg": (elo_h_adj + elo_a_adj) / 2,
                "home_goals": hs, "away_goals": as_,
            })
        k = k_for(m.get("tournament", "Friendly"))
        exp_h = 1 / (1 + 10**((elo_a_adj - elo_h_adj) / 400))
        actual_h = 1 if hs > as_ else (0 if hs < as_ else 0.5)
        gd = abs(hs - as_)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else 1.75 if gd == 3 else 1.75 + (gd-3)*0.0625
        change = k * mult * (actual_h - exp_h)
        elo[h] += change
        elo[a] -= change
    if record_features:
        return dict(elo), pd.DataFrame(features)
    return dict(elo)


def train_models(features_df):
    df = features_df.copy()
    df["diff_s"] = df["elo_diff"] / 400
    df["avg_s"] = df["elo_avg"] / 400
    X = df[["diff_s", "avg_s"]].values
    hm = PoissonRegressor(alpha=0.01, max_iter=1000).fit(X, df["home_goals"])
    am = PoissonRegressor(alpha=0.01, max_iter=1000).fit(X, df["away_goals"])
    return {
        "home_intercept": float(hm.intercept_), "home_coefs": hm.coef_.tolist(),
        "away_intercept": float(am.intercept_), "away_coefs": am.coef_.tolist(),
        "n_train": len(df),
    }


def predict_match(home, away, elo, models, neutral=True):
    elo_h = elo.get(home, 1500.0) + (0 if neutral else HOME_ADV)
    elo_a = elo.get(away, 1500.0)
    diff = (elo_h - elo_a) / 400
    avg = (elo_h + elo_a) / 2 / 400
    lam_h = np.exp(models["home_intercept"] + models["home_coefs"][0]*diff + models["home_coefs"][1]*avg)
    lam_a = np.exp(models["away_intercept"] + models["away_coefs"][0]*diff + models["away_coefs"][1]*avg)
    ph = scipy_poisson.pmf(np.arange(MAX_GOALS+1), lam_h)
    pa = scipy_poisson.pmf(np.arange(MAX_GOALS+1), lam_a)
    M = np.outer(ph, pa)
    pw = np.tril(M, -1).sum(); pd_ = np.diag(M).sum(); pl = np.triu(M, 1).sum()
    tot = pw + pd_ + pl
    return {"p_home_win": pw/tot, "p_draw": pd_/tot, "p_away_win": pl/tot,
            "lam_h": float(lam_h), "lam_a": float(lam_a)}


def simulate_tournament_32(groups, elo, models, rng):
    group_results, stage = {}, {}
    for g, teams in groups.items():
        st = {t: {"pts":0,"gf":0,"ga":0} for t in teams}
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                ta, tb = teams[i], teams[j]
                p = predict_match(ta, tb, elo, models, neutral=True)
                ha = min(rng.poisson(p["lam_h"]), MAX_GOALS)
                ab = min(rng.poisson(p["lam_a"]), MAX_GOALS)
                st[ta]["gf"]+=ha; st[ta]["ga"]+=ab
                st[tb]["gf"]+=ab; st[tb]["ga"]+=ha
                if ha>ab: st[ta]["pts"]+=3
                elif ab>ha: st[tb]["pts"]+=3
                else: st[ta]["pts"]+=1; st[tb]["pts"]+=1
        for t in st: st[t]["gd"] = st[t]["gf"] - st[t]["ga"]
        ranked = sorted(st.items(), key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], rng.random()), reverse=True)
        group_results[g] = [r[0] for r in ranked]
        for i, t in enumerate([r[0] for r in ranked]):
            stage[t] = "R16" if i < 2 else "group_stage"
    g_order = ['A','B','C','D','E','F','G','H']
    r16_pairs = []
    for i in range(0, 8, 2):
        g1, g2 = g_order[i], g_order[i+1]
        r16_pairs.append((group_results[g1][0], group_results[g2][1]))
        r16_pairs.append((group_results[g2][0], group_results[g1][1]))
    def ko(ta, tb):
        p = predict_match(ta, tb, elo, models, neutral=True)
        ha = min(rng.poisson(p["lam_h"]), MAX_GOALS)
        ab = min(rng.poisson(p["lam_a"]), MAX_GOALS)
        if ha == ab:
            ea, eb = elo.get(ta, 1500), elo.get(tb, 1500)
            p_a = 1/(1+10**((eb-ea)/400))
            p_a = 0.5 + 0.3*(p_a-0.5)
            return ta if rng.random() < p_a else tb
        return ta if ha > ab else tb
    r16w = [ko(a, b) for a, b in r16_pairs]
    for t in r16w: stage[t] = "QF"
    qfw = [ko(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
    for t in qfw: stage[t] = "SF"
    sfw = [ko(qfw[i], qfw[i+1]) for i in range(0, 4, 2)]
    for t in sfw: stage[t] = "Final"
    champion = ko(sfw[0], sfw[1])
    stage[champion] = "Champion"
    return stage


def run_backtest(year, matches):
    cfg = TOURNAMENTS[year]
    print(f"\n{'='*72}\nBACKTEST: {cfg['name']}")
    print(f"Actual winner: {cfg['winner']} | Actual runner-up: {cfg['runner_up']}")
    print(f"{'='*72}\n")
    cutoff = pd.to_datetime(cfg["start"])
    pre = matches[matches["date"] < cutoff].copy()
    wc = matches[(matches["date"] >= pd.to_datetime(cfg["start"])) &
                 (matches["date"] <= pd.to_datetime(cfg["end"])) &
                 (matches["tournament"] == "FIFA World Cup")].copy()
    print(f"Pre-tournament training data: {len(pre):,} matches before {cfg['start']}")
    print(f"Actual {year} WC matches found: {len(wc)}")
    if len(wc) == 0:
        print(f"ERROR: No {year} WC matches found in dataset")
        return None
    print("Computing Elo on pre-tournament data...")
    final_elo, features_df = compute_elo(pre, record_features=True)
    train_set = features_df[features_df["date"].dt.year >= 2000].drop(columns=["date"])
    print(f"Training Poisson model on {len(train_set):,} matches from 2000-{year}")
    models = train_models(train_set)
    print(f"  home: int={models['home_intercept']:.3f}, coefs={[round(c,3) for c in models['home_coefs']]}")
    print(f"  away: int={models['away_intercept']:.3f}, coefs={[round(c,3) for c in models['away_coefs']]}")
    all_teams = [t for g in cfg["groups"].values() for t in g]
    team_elos = sorted([(t, final_elo.get(t, 1500.0)) for t in all_teams], key=lambda x: -x[1])
    print(f"\nTop 10 pre-WC Elo (of 32 participants):")
    for i, (t, e) in enumerate(team_elos[:10]):
        mark = " <- actual winner" if t == cfg["winner"] else (" <- actual runner-up" if t == cfg["runner_up"] else "")
        print(f"  {i+1:2d}. {t:24s} {int(e):4d}{mark}")
    losses, correct = [], 0
    for _, m in wc.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m["home_score"], m["away_score"]
        if pd.isna(hs) or pd.isna(as_): continue
        hs, as_ = int(hs), int(as_)
        p = predict_match(h, a, final_elo, models, neutral=True)
        if hs > as_: actual_p, actual_l = p["p_home_win"], "H"
        elif hs < as_: actual_p, actual_l = p["p_away_win"], "A"
        else: actual_p, actual_l = p["p_draw"], "D"
        losses.append(-np.log(max(actual_p, 1e-10)))
        probs = [p["p_home_win"], p["p_draw"], p["p_away_win"]]
        pred_l = ["H", "D", "A"][int(np.argmax(probs))]
        if pred_l == actual_l: correct += 1
    mean_ll = float(np.mean(losses))
    baseline_ll = float(-np.log(1/3))
    print(f"\nLog loss on {len(losses)} actual matches: {mean_ll:.4f}")
    print(f"Baseline (uniform 1/3):                {baseline_ll:.4f}")
    print(f"Improvement over baseline:             {(baseline_ll-mean_ll)/baseline_ll*100:.1f}%")
    print(f"Outcome accuracy (most likely):        {correct}/{len(losses)} = {correct/len(losses):.1%}")
    print(f"\nRunning Monte Carlo (5,000 simulations)...")
    n_sims = 5000
    rng = np.random.RandomState(42)
    counts = defaultdict(lambda: defaultdict(int))
    so = ["group_stage", "R16", "QF", "SF", "Final", "Champion"]
    for i in range(n_sims):
        if i and i % 1000 == 0: print(f"  {i}/{n_sims}")
        res = simulate_tournament_32(cfg["groups"], final_elo, models, rng)
        for t, s in res.items():
            try:
                idx = so.index(s)
                for j in range(1, idx+1): counts[t][so[j]] += 1
            except ValueError: pass
    results = []
    for t in all_teams:
        c = counts[t]
        results.append({
            "team": t, "elo": float(final_elo.get(t, 1500)),
            "p_r16": c["R16"]/n_sims, "p_qf": c["QF"]/n_sims,
            "p_sf": c["SF"]/n_sims, "p_final": c["Final"]/n_sims,
            "p_win": c["Champion"]/n_sims,
        })
    results.sort(key=lambda r: -r["p_win"])
    print(f"\nMODEL'S TOP 10 PICKS FOR {year}:")
    for i, r in enumerate(results[:10]):
        mark = ""
        if r["team"] == cfg["winner"]: mark = " <- actual winner"
        elif r["team"] == cfg["runner_up"]: mark = " <- actual runner-up"
        print(f"  {i+1:2d}. {r['team']:24s} {r['p_win']*100:5.1f}% to win{mark}")
    champ_rank = next(i+1 for i, r in enumerate(results) if r["team"] == cfg["winner"])
    ru_rank = next(i+1 for i, r in enumerate(results) if r["team"] == cfg["runner_up"])
    print(f"\nActual champion {cfg['winner']} ranked #{champ_rank} (of 32)")
    print(f"Actual runner-up {cfg['runner_up']} ranked #{ru_rank} (of 32)")
    return {
        "year": year, "name": cfg["name"],
        "winner": cfg["winner"], "runner_up": cfg["runner_up"],
        "n_pre_matches": len(pre), "n_wc_matches": len(losses),
        "log_loss": mean_ll, "baseline_log_loss": baseline_ll,
        "improvement_pct": (baseline_ll-mean_ll)/baseline_ll*100,
        "outcome_accuracy": correct/len(losses),
        "winner_rank": int(champ_rank), "runner_up_rank": int(ru_rank),
        "top10": results[:10], "all_picks": results,
    }


def main():
    print("Loading historical match data...")
    matches = pd.read_csv(RAW / "historical_matches.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    print(f"Total: {len(matches):,} matches")
    out = {}
    for year in [2022, 2018]:
        result = run_backtest(year, matches)
        if result: out[str(year)] = result
    with open(PROC / "backtest_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nSaved to data/processed/backtest_results.json")


if __name__ == "__main__":
    main()
