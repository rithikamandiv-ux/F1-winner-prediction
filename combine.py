from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data_raw")
OUTPUT_FILE = Path("f1_combined.csv")


def main() -> None:
    csv_files = sorted(RAW_DATA_DIR.glob("f1_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No raw season files found in data_raw/.")

    frames = [pd.read_csv(file_path) for file_path in csv_files]
    combined_df = pd.concat(frames, ignore_index=True)

    print("Combined shape:", combined_df.shape)
    print("\nINFO:\n")
    print(combined_df.info())
    print("\nHEAD:\n")
    print(combined_df.head())
    print("\nMISSING VALUES:\n")
    print(combined_df.isnull().sum())

    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved combined dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()