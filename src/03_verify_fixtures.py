from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
fixtures = pd.read_csv(DATA / "raw" / "fixtures_2026.csv")
elo = pd.read_csv(DATA / "processed" / "current_elo.csv")

print(f"Loaded {len(fixtures)} fixtures")
print(f"Loaded Elo for {len(elo)} teams\n")

fixture_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
elo_teams = set(elo["team"])

missing = sorted(fixture_teams - elo_teams)
if missing:
    print(f"PROBLEM: {len(missing)} team(s) in fixtures NOT found in Elo data:")
    for t in missing:
        candidates = sorted([e for e in elo_teams if t.lower()[:4] in e.lower() or e.lower()[:4] in t.lower()])[:5]
        print(f"  '{t}'  -- closest Elo names: {candidates}")
else:
    print("All fixture team names matched to Elo data. Good.")

print(f"\nFixture team count: {len(fixture_teams)} (should be 48)")
print(f"Stage counts:")
print(fixtures["stage"].value_counts().to_string())
print(f"\nGroup counts (should all be 6):")
print(fixtures["group"].value_counts().sort_index().to_string())
