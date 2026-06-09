from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

DATA = Path(__file__).parent.parent / "data" / "processed"
HOME_ADV = 65

matches = pd.read_csv(DATA / "matches_with_elo.csv")
matches["date"] = pd.to_datetime(matches["date"])

modern = matches[matches["date"] >= "2000-01-01"].copy()
print(f"Training data: {len(modern)} matches from 2000-01-01 to {modern['date'].max().date()}")

modern["elo_diff"] = (
    modern["home_elo_pre"] + (~modern["neutral"].astype(bool)) * HOME_ADV
    - modern["away_elo_pre"]
)

def classify(row):
    if row["home_score"] > row["away_score"]:
        return 0
    if row["home_score"] < row["away_score"]:
        return 2
    return 1

modern["outcome"] = modern.apply(classify, axis=1)

X = modern[["elo_diff"]].values
y = modern["outcome"].values

labels = ["home_win", "draw", "away_win"]
freqs = np.bincount(y) / len(y)
print(f"\nOutcome distribution in training data:")
for lbl, f in zip(labels, freqs):
    print(f"  {lbl:10s}: {f:.3f}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_tr, y_tr)

p_te = model.predict_proba(X_te)
ll = log_loss(y_te, p_te, labels=[0, 1, 2])

avg = np.bincount(y_tr, minlength=3) / len(y_tr)
ll_avg = log_loss(y_te, np.tile(avg, (len(y_te), 1)), labels=[0, 1, 2])

def elo_baseline(elo_diff):
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    return np.column_stack([p_home * 0.7, np.full_like(p_home, 0.25), (1 - p_home) * 0.7])

p_elo = elo_baseline(X_te.flatten())
p_elo = p_elo / p_elo.sum(axis=1, keepdims=True)
ll_elo = log_loss(y_te, p_elo, labels=[0, 1, 2])

print(f"\nLog loss (lower is better):")
print(f"  Average-frequency baseline: {ll_avg:.4f}")
print(f"  Pure-Elo baseline:          {ll_elo:.4f}")
print(f"  Our logistic model:         {ll:.4f}")
print(f"  Improvement over avg:       {((ll_avg - ll) / ll_avg * 100):.1f}%")

print(f"\nExample predictions  (elo_diff: home_win / draw / away_win)")
for d in [-400, -200, -100, -50, 0, 50, 100, 200, 400]:
    p = model.predict_proba([[d]])[0]
    print(f"  diff {d:+5d}:   {p[0]:.3f}  /  {p[1]:.3f}  /  {p[2]:.3f}")

coefs = {
    "intercept": model.intercept_.tolist(),
    "coef": model.coef_.tolist(),
    "classes": model.classes_.tolist(),
    "home_advantage_elo": HOME_ADV,
    "test_log_loss": ll,
    "baseline_log_loss": ll_avg,
}
with open(DATA / "match_model.json", "w") as f:
    json.dump(coefs, f, indent=2)
print(f"\nModel saved to {DATA / 'match_model.json'}")
