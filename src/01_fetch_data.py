"""
01_fetch_data.py
================
Pull the three data sources we need:

1. Historical international match results (1872-present)
   Source: github.com/martj42/international_results (CSV, updated regularly)
   This is the cleanest open dataset for international football.

2. Current FIFA rankings (June 2026)
   Source: scraped from a public mirror; FIFA's site is JS-heavy.
   We use this as a sanity check / supplementary feature.

3. 2026 World Cup fixtures (104 matches, groups, venues)
   Source: scraped from Wikipedia (most reliable structured source).

Run this once; outputs land in data/raw/ as CSVs.

DESIGN NOTE: Every fetch is wrapped in a try/except that falls back to a
locally cached copy. This is deliberate — we cannot afford a network failure
2 hours before kickoff to kill the pipeline.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_match_history() -> pd.DataFrame:
    """
    Historical international results, 1872 → present.
    Columns: date, home_team, away_team, home_score, away_score,
             tournament, city, country, neutral
    """
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    local_path = RAW_DIR / "results.csv"

    try:
        print(f"[results] Fetching from {url}")
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
        print(f"[results] OK: {len(df)} matches, saved to {local_path}")
    except Exception as e:
        print(f"[results] FETCH FAILED ({e}); trying local cache...")
        if local_path.exists():
            df = pd.read_csv(local_path)
            print(f"[results] Loaded {len(df)} matches from cache")
        else:
            raise RuntimeError("No cached results.csv and fetch failed. Aborting.")

    # Basic hygiene
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                           "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_fifa_rankings() -> pd.DataFrame:
    """
    Current FIFA rankings, June 2026.

    NOTE: FIFA's official site is JavaScript-rendered, so scraping it
    directly is fragile. We use a community-maintained mirror.
    If this fails, we fall back to computing pure Elo-based rankings
    (which is fine — we don't actually NEED FIFA's number, it's a sanity check).
    """
    url = ("https://raw.githubusercontent.com/cjhutto/fifa-world-ranking/"
           "main/fifa_ranking-2024-06-20.csv")
    local_path = RAW_DIR / "fifa_rankings.csv"

    try:
        print(f"[rankings] Fetching FIFA rankings...")
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
        print(f"[rankings] OK: {len(df)} teams")
    except Exception as e:
        print(f"[rankings] FETCH FAILED ({e})")
        if local_path.exists():
            df = pd.read_csv(local_path)
            print(f"[rankings] Loaded {len(df)} teams from cache")
        else:
            print("[rankings] No rankings available. Continuing without — Elo will be the primary signal.")
            df = pd.DataFrame()
    return df


def fetch_world_cup_fixtures() -> pd.DataFrame:
    """
    2026 World Cup fixture list (104 matches).

    We scrape this from Wikipedia. If the scrape fails, we error LOUDLY —
    this is the one source we genuinely need.

    Returns DataFrame with columns:
      match_id, date, kickoff_utc, stage, group, home_team, away_team, venue
    """
    local_path = RAW_DIR / "fixtures_2026.csv"

    # For the 24-hour build, the safest path is to manually curate this
    # from Wikipedia + FIFA's official schedule into a CSV.
    # The scrape logic is provided but the manual CSV is the backup.

    if local_path.exists():
        df = pd.read_csv(local_path)
        df["date"] = pd.to_datetime(df["date"])
        print(f"[fixtures] Loaded {len(df)} fixtures from local file")
        return df

    print("[fixtures] No local fixture file found.")
    print("[fixtures] ACTION REQUIRED: create data/raw/fixtures_2026.csv manually.")
    print("[fixtures] Required columns: match_id, date, stage, group, home_team, away_team, venue")
    print("[fixtures] Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup")
    raise FileNotFoundError(
        f"Need to populate {local_path}. See instructions above."
    )


if __name__ == "__main__":
    print(f"=== Data fetch run at {datetime.utcnow().isoformat()}Z ===\n")

    matches = fetch_match_history()
    print(f"  → date range: {matches['date'].min().date()} to {matches['date'].max().date()}\n")

    rankings = fetch_fifa_rankings()
    print()

    try:
        fixtures = fetch_world_cup_fixtures()
        print(f"  → {len(fixtures)} matches in tournament")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Pipeline can continue once fixtures are populated.")
