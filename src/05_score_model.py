from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.stats import poisson

DATA = Path(__file__).parent.parent / "data" / "processed"
HOME_ADV = 65
MAX_GOALS = 10
SCALE = 400.0

matches = pd.read_csv(DATA / "matches_with_elo.csv")
matches["date"] = pd.to_datetime(matches["date"])

modern = matches[matches["date"] >= "2000-01-01"].copy()
print(f"Training data: {len(modern)} matches")

modern["elo_diff_s"] = (
    modern["home_elo_pre"] + (~modern["neutral"].astype(bool)) * HOME_ADV
    - modern["away_elo_pre"]
) / SCALE
modern["elo_avg_s"] = ((modern["home_elo_pre"] + modern["away_elo_pre"]) / 2 - 1500) / SCALE

X = modern[["elo_diff_s", "elo_avg_s"]].values
y_home = modern["home_score"].values.astype(float)
y_away = modern["away_score"].values.astype(float)

print(f"Feature ranges: elo_diff_s in [{X[:,0].min():.2f}, {X[:,0].max():.2f}]")
print(f"                elo_avg_s in [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
print(f"Avg home goals: {y_home.mean():.3f} | Avg away goals: {y_away.mean():.3f}")

X_tr, X_te, yh_tr, yh_te, ya_tr, ya_te = train_test_split(
    X, y_home, y_away, test_size=0.2, random_state=42
)

home_model = PoissonRegressor(alpha=0.001, max_iter=5000)
home_model.fit(X_tr, yh_tr)
away_model = PoissonRegressor(alpha=0.001, max_iter=5000)
away_model.fit(X_tr, ya_tr)

print(f"\nHome goal model: intercept={home_model.intercept_:.4f}, coefs={home_model.coef_}")
print(f"Away goal model: intercept={away_model.intercept_:.4f}, coefs={away_model.coef_}")

def score_matrix(lam_h, lam_a, max_g=MAX_GOALS):
    p_h = poisson.pmf(np.arange(max_g + 1), lam_h)
    p_a = poisson.pmf(np.arange(max_g + 1), lam_a)
    return np.outer(p_h, p_a)

def outcome_probs(lam_h, lam_a):
    M = score_matrix(lam_h, lam_a)
    p_home = np.tril(M, -1).sum()
    p_draw = np.diag(M).sum()
    p_away = np.triu(M, 1).sum()
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

lam_h_te = home_model.predict(X_te)
lam_a_te = away_model.predict(X_te)
probs_te = np.array([outcome_probs(lh, la) for lh, la in zip(lam_h_te, lam_a_te)])

y_outcome = np.where(yh_te > ya_te, 0, np.where(yh_te < ya_te, 2, 1))
ll = log_loss(y_outcome, probs_te, labels=[0, 1, 2])

avg = np.bincount(y_outcome, minlength=3) / len(y_outcome)
ll_avg = log_loss(y_outcome, np.tile(avg, (len(y_outcome), 1)), labels=[0, 1, 2])

print(f"\nLog loss comparison:")
print(f"  Baseline (class frequencies): {ll_avg:.4f}")
print(f"  Score model:                  {ll:.4f}")
print(f"  (Logistic match model:        0.8895 for comparison)")

print(f"\nExample predictions (elo_avg = ~2000, top international match):")
for d in [-400, -200, -100, 0, 100, 200, 400]:
    d_s = d / SCALE
    a_s = 500 / SCALE
    lh = home_model.predict([[d_s, a_s]])[0]
    la = away_model.predict([[d_s, a_s]])[0]
    p_h, p_d, p_a = outcome_probs(lh, la)
    M = score_matrix(lh, la)
    i, j = np.unravel_index(M.argmax(), M.shape)
    print(f"  diff {d:+5d}:  λ_h={lh:.2f} λ_a={la:.2f}  W/D/L = {p_h:.3f}/{p_d:.3f}/{p_a:.3f}  most likely: {i}-{j}")

out = {
    "home_intercept": float(home_model.intercept_),
    "home_coef": home_model.coef_.tolist(),
    "away_intercept": float(away_model.intercept_),
    "away_coef": away_model.coef_.tolist(),
    "feature_names": ["elo_diff", "elo_avg_c"],
    "feature_scale": SCALE,
    "home_advantage_elo": HOME_ADV,
    "max_goals": MAX_GOALS,
    "test_log_loss": ll,
    "baseline_log_loss": ll_avg,
}
with open(DATA / "score_model.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nModel saved to score_model.json")
