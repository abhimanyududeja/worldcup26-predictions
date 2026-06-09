from pathlib import Path
import pandas as pd

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

elo = pd.read_csv(PROCESSED / "current_elo.csv")
matches = pd.read_csv(PROCESSED / "matches_with_elo.csv")
matches["date"] = pd.to_datetime(matches["date"])

home_counts = matches.groupby("home_team").size().rename("home_n")
away_counts = matches.groupby("away_team").size().rename("away_n")
home_last = matches.groupby("home_team")["date"].max().rename("home_last")
away_last = matches.groupby("away_team")["date"].max().rename("away_last")

team_stats = pd.concat([home_counts, away_counts, home_last, away_last], axis=1).fillna({"home_n": 0, "away_n": 0})
team_stats["total"] = team_stats["home_n"] + team_stats["away_n"]
team_stats["last_played"] = team_stats[["home_last", "away_last"]].max(axis=1)
team_stats = team_stats[["total", "last_played"]].reset_index().rename(columns={"index": "team"})

elo = elo.merge(team_stats, on="team", how="left")
elo["total"] = elo["total"].fillna(0).astype(int)

print("=" * 70)
print("UNFILTERED TOP 20")
print("=" * 70)
top = elo.head(20).copy()
top["last_played"] = top["last_played"].dt.date
print(top.to_string(index=False))

print("\n" + "=" * 70)
print("WHERE ARE THE OBVIOUS TOP TEAMS?")
print("=" * 70)
big = ["Brazil", "Argentina", "France", "Spain", "Germany", "Portugal",
       "Netherlands", "England", "Belgium", "Italy", "Croatia", "Morocco",
       "Uruguay", "Colombia", "Mexico", "United States", "Japan"]
for t in big:
    row = elo[elo["team"] == t]
    if row.empty:
        print(f"  {t:18s}  NOT FOUND")
        continue
    r = row.iloc[0]
    rank = elo.index[elo["team"] == t][0] + 1
    last = r["last_played"].date() if pd.notna(r["last_played"]) else "never"
    print(f"  #{rank:4d}  {t:18s}  Elo={r['elo']:7.1f}  matches={r['total']:5d}  last={last}")

print("\n" + "=" * 70)
print("FILTERED TOP 20: active in last 2 years AND 50+ matches")
print("=" * 70)
cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
filt = elo[(elo["last_played"] >= cutoff) & (elo["total"] >= 50)].head(20).copy()
filt["last_played"] = filt["last_played"].dt.date
print(filt.to_string(index=False))
