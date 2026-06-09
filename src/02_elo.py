from pathlib import Path
from collections import defaultdict
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

INITIAL_RATING = 1500
HOME_ADVANTAGE = 65

TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Championship": 45,
    "Gold Cup": 45,
    "Confederations Cup": 50,
    "Friendly": 20,
}
DEFAULT_K = 30


def get_k_factor(tournament):
    if pd.isna(tournament):
        return DEFAULT_K
    for key, k in TOURNAMENT_K.items():
        if key.lower() in str(tournament).lower():
            return k
    return DEFAULT_K


def goal_diff_multiplier(home_score, away_score):
    gd = abs(home_score - away_score)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_history(matches):
    ratings = defaultdict(lambda: float(INITIAL_RATING))
    rows = []
    for _, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        h_score, a_score = m["home_score"], m["away_score"]
        neutral = bool(m.get("neutral", False))
        r_home = ratings[home]
        r_away = ratings[away]
        eff_home = r_home + (0 if neutral else HOME_ADVANTAGE)
        e_home = expected_score(eff_home, r_away)
        if h_score > a_score:
            actual = 1.0
        elif h_score < a_score:
            actual = 0.0
        else:
            actual = 0.5
        k = get_k_factor(m.get("tournament", ""))
        mult = goal_diff_multiplier(h_score, a_score)
        delta = k * mult * (actual - e_home)
        rows.append({
            "date": m["date"], "home_team": home, "away_team": away,
            "home_score": h_score, "away_score": a_score, "neutral": neutral,
            "tournament": m.get("tournament", ""),
            "home_elo_pre": r_home, "away_elo_pre": r_away,
            "k_factor": k, "gd_mult": mult,
        })
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return dict(ratings), pd.DataFrame(rows)


def main():
    matches = pd.read_csv(RAW_DIR / "results.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    print(f"Loaded raw: {len(matches)} rows, {matches['date'].min().date()} to {matches['date'].max().date()}")

    today = pd.Timestamp.now().normalize()
    before = len(matches)
    matches = matches[matches["date"] < today]
    matches = matches.dropna(subset=["home_score", "away_score"])
    print(f"After cleaning: {len(matches)} rows (dropped {before - len(matches)} future/missing-score)")
    print(f"Effective range: {matches['date'].min().date()} to {matches['date'].max().date()}")

    matches = matches.sort_values("date").reset_index(drop=True)

    print("Computing Elo history...")
    final_ratings, enriched = compute_elo_history(matches)

    enriched.to_csv(PROCESSED_DIR / "matches_with_elo.csv", index=False)
    ratings_df = (pd.DataFrame(list(final_ratings.items()), columns=["team", "elo"])
                  .sort_values("elo", ascending=False).reset_index(drop=True))
    ratings_df.to_csv(PROCESSED_DIR / "current_elo.csv", index=False)

    print("\n=== Top 20 Elo (raw, unfiltered) ===")
    print(ratings_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
