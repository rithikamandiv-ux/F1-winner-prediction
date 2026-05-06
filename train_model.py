import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "f1_features.csv"
FEATURES = [
    "DriverId",
    "TeamName",
    "GridPosition",
    "q_pos",
    "driver_avg_finish",
    "team_avg_finish",
    "driver_recent_form",
    "track_type",
    "dnf_rate",
    "wet_race",
    "strategy_risk",
]


def encode_with_unknown(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> None:
    encoder = LabelEncoder()
    train_df[column] = encoder.fit_transform(train_df[column])

    known_labels = set(encoder.classes_)
    test_df[column] = test_df[column].apply(
        lambda value: value if value in known_labels else "Unknown"
    )

    if "Unknown" not in encoder.classes_:
        encoder.classes_ = pd.Index(list(encoder.classes_) + ["Unknown"])

    test_df[column] = encoder.transform(test_df[column])


def main() -> None:
    raw_df = pd.read_csv(INPUT_FILE)
    raw_df["year"] = raw_df["race_id"].str[:4].astype(int)

    train_df = raw_df[raw_df["year"].between(2020, 2025)].copy()
    test_df = raw_df[raw_df["year"] == 2026].copy()
    test_df_original = test_df.copy()

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    if test_df.empty:
        print("No 2026 data found yet. Fetch and preprocess 2026 data first.")
        raise SystemExit(1)
    if train_df.empty:
        print("No 2020-2025 data found. Build training data first.")
        raise SystemExit(1)

    for column in ["DriverId", "TeamName", "track_type"]:
        encode_with_unknown(train_df, test_df, column)

    x_train = train_df[FEATURES]
    y_train = train_df["is_winner"]
    x_test = test_df[FEATURES]
    y_test = test_df["is_winner"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nPredicted winners by 2026 race:\n")
    win_probs = model.predict_proba(x_test)[:, 1]
    results = test_df_original[
        ["race_id", "Abbreviation", "TeamName", "GridPosition", "q_pos", "Position", "is_winner"]
    ].copy()
    results["predicted_win_probability"] = win_probs

    for race_id in results["race_id"].unique():
        race_results = results[results["race_id"] == race_id].copy()
        race_results = race_results.sort_values("predicted_win_probability", ascending=False)

        predicted_winner = race_results.iloc[0]["Abbreviation"]
        actual_rows = race_results[race_results["is_winner"] == 1]
        actual_winner = actual_rows.iloc[0]["Abbreviation"] if not actual_rows.empty else "N/A"

        print(race_id)
        print(f"Predicted winner: {predicted_winner}")
        print(f"Actual winner: {actual_winner}")
        print(
            race_results[
                ["Abbreviation", "TeamName", "q_pos", "predicted_win_probability", "is_winner"]
            ].head(5)
        )
        print("-" * 60)

        # End of race prediction loop

    results.to_csv("predictions_2026.csv", index=False)
    print("\nSaved predictions to predictions_2026.csv")


if __name__ == "__main__":
    main()