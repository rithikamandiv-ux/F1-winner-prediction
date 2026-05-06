import pandas as pd

INPUT_FILE = "f1_combined.csv"
OUTPUT_FILE = "f1_processed.csv"

COLUMNS_TO_KEEP = [
    "race_id",
    "DriverNumber",
    "Abbreviation",
    "DriverId",
    "TeamName",
    "TeamId",
    "Position",
    "GridPosition",
    "q_pos",
    "Status",
    "Points",
    "Laps",
    "is_winner",
]


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df = df[COLUMNS_TO_KEEP]

    df = df.dropna(subset=["Position", "GridPosition", "q_pos"])
    df["Position"] = df["Position"].astype(int)
    df["GridPosition"] = df["GridPosition"].astype(int)
    df["q_pos"] = df["q_pos"].astype(int)

    print(df.info())
    print(df.head())
    print(df.isnull().sum())

    df.to_csv(OUTPUT_FILE, index=False)
    print("Processed dataset saved.")
    print(df.shape)


if __name__ == "__main__":
    main()