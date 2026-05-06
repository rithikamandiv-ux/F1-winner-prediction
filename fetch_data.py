import argparse
import time
from pathlib import Path

import fastf1
import pandas as pd


RAW_DATA_DIR = Path("data_raw")
CACHE_DIR = Path("cache")


def parse_years(years_arg: str) -> list[int]:
    years = []
    for token in years_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", maxsplit=1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(token))
    return sorted(set(years))


def fetch_season(year: int, sleep_seconds: float) -> pd.DataFrame:
    print(f"\nProcessing {year} season...")
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0]

    season_races = []
    for event in schedule.itertuples():
        race_name = event.EventName
        print(f"Fetching {race_name}...")
        try:
            race_session = fastf1.get_session(year, race_name, "R")
            race_session.load(
                laps=False,
                telemetry=False,
                weather=False,
                messages=False,
                livedata=False,
            )
            race_df = race_session.results

            qualifying_session = fastf1.get_session(year, race_name, "Q")
            qualifying_session.load(
                laps=False,
                telemetry=False,
                weather=False,
                messages=False,
            )
            qualifying_df = qualifying_session.results[["Abbreviation", "Position"]].rename(
                columns={"Position": "q_pos"}
            )

            merged = race_df.merge(qualifying_df, on="Abbreviation", how="left")
            merged["race_id"] = f"{year}-{race_name}"
            merged["is_winner"] = (merged["Position"] == 1).astype(int)
            season_races.append(merged)
        except Exception as error:
            print(f"Skipping {race_name}: {error}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not season_races:
        return pd.DataFrame()

    return pd.concat(season_races, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Formula 1 race and qualifying results.")
    parser.add_argument(
        "--years",
        default="2020-2026",
        help="Years to fetch. Examples: 2020-2026 or 2020,2021,2026",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=15.0,
        help="Delay between race fetches to avoid API throttling.",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    all_seasons = []
    for year in years:
        try:
            year_df = fetch_season(year, sleep_seconds=args.sleep_seconds)
        except Exception as error:
            print(f"Could not load schedule for {year}: {error}")
            continue

        if year_df.empty:
            print(f"No race data collected for {year}.")
            continue

        output_path = RAW_DATA_DIR / f"f1_{year}.csv"
        year_df.to_csv(output_path, index=False)
        all_seasons.append(year_df)
        print(f"Saved season data: {output_path}")

    if not all_seasons:
        print("\nNo race data was collected. No combined CSV was created.")
        return

    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_df.to_csv("f1_2020_2026.csv", index=False)
    print("\nDataset created successfully.")
    print("Shape:", combined_df.shape)


if __name__ == "__main__":
    main()
