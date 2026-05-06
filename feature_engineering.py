import pandas as pd

INPUT_FILE = "f1_processed.csv"
OUTPUT_FILE = "f1_features.csv"

track_type_map = {
    "Monaco Grand Prix": "street",
    "Singapore Grand Prix": "street",
    "Azerbaijan Grand Prix": "street",
    "Miami Grand Prix": "hybrid",
    "Australian Grand Prix": "hybrid",
    "British Grand Prix": "permanent",
    "Italian Grand Prix": "permanent",
    "Japanese Grand Prix": "permanent",
    "Belgian Grand Prix": "permanent",
    "Bahrain Grand Prix": "permanent",
    "Saudi Arabian Grand Prix": "street",
    "Abu Dhabi Grand Prix": "permanent",
    "Spanish Grand Prix": "permanent",
    "Dutch Grand Prix": "permanent",
    "Hungarian Grand Prix": "permanent",
    "Canadian Grand Prix": "hybrid",
    "United States Grand Prix": "permanent",
    "Mexican Grand Prix": "permanent",
    "Brazilian Grand Prix": "permanent",
    "Las Vegas Grand Prix": "street",
    "Qatar Grand Prix": "permanent",
    "Austrian Grand Prix": "permanent",
    "Chinese Grand Prix": "permanent",
    "Emilia Romagna Grand Prix": "permanent",
}

wet_races = {
    "2020-Turkish Grand Prix",
    "2021-Belgian Grand Prix",
    "2021-Russian Grand Prix",
    "2021-Hungarian Grand Prix",
    "2022-Japanese Grand Prix",
    "2022-Singapore Grand Prix",
    "2023-Monaco Grand Prix",
    "2023-Dutch Grand Prix",
    "2023-Japanese Grand Prix",
    "2024-Sao Paulo Grand Prix",
}

strategy_risk_map = {
    "street": 2,
    "hybrid": 1,
    "permanent": 0,
}


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df["year"] = df["race_id"].str[:4].astype(int)

    race_order = df[["year", "race_id"]].drop_duplicates().reset_index(drop=True)
    race_order["race_number"] = race_order.groupby("year").cumcount() + 1
    df = df.merge(race_order, on=["year", "race_id"], how="left")

    df["race_name"] = df["race_id"].str[5:]
    df["track_type"] = df["race_name"].map(track_type_map).fillna("permanent")
    df["wet_race"] = df["race_id"].isin(wet_races).astype(int)
    df["strategy_risk"] = df["track_type"].map(strategy_risk_map).fillna(0).astype(int)

    # Preserve time order so each feature only uses prior races.
    df = df.sort_values(["year", "race_number", "DriverId"])

    df["driver_avg_finish"] = df.groupby("DriverId")["Position"].transform(
        lambda x: x.shift().expanding().mean()
    )
    df["team_avg_finish"] = df.groupby("TeamName")["Position"].transform(
        lambda x: x.shift().expanding().mean()
    )

    overall_mean_position = df["Position"].mean()
    df["team_avg_finish"] = df["team_avg_finish"].fillna(overall_mean_position)
    df["driver_avg_finish"] = df["driver_avg_finish"].fillna(overall_mean_position)

    df["driver_recent_form"] = df.groupby("DriverId")["Position"].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean()
    )
    df["driver_recent_form"] = df["driver_recent_form"].fillna(overall_mean_position)

    df["is_dnf"] = (
        ~df["Status"].str.contains("Finished|\\+", case=False, na=False)
    ).astype(int)
    df["dnf_rate"] = df.groupby("DriverId")["is_dnf"].transform(
        lambda x: x.shift().expanding().mean()
    )
    df["dnf_rate"] = df["dnf_rate"].fillna(0)

    df.to_csv(OUTPUT_FILE, index=False)
    print(
        df[
            [
                "race_id",
                "Abbreviation",
                "TeamName",
                "Position",
                "q_pos",
                "track_type",
                "driver_avg_finish",
                "team_avg_finish",
                "driver_recent_form",
                "dnf_rate",
                "wet_race",
                "strategy_risk",
            ]
        ].head(20)
    )
    print("\nFeature engineering complete.")
    print("Saved as f1_features.csv")
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()