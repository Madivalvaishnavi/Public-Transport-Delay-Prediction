
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "transport_delays.csv"
OUTPUT_DIR = BASE_DIR / "processed_data"

TARGET_COLUMN = "DelayMinutes"

CATEGORICAL_COLUMNS = [
    "RouteID",
    "StopID",
    "VehicleID",
    "TrafficLevel",
]

DROP_COLUMNS = ["Timestamp"]


def main():
    # Check that the generated dataset exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run 'python generate_data.py' first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert timestamp to datetime and sort chronologically
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )
    df = df.dropna(subset=["Timestamp", TARGET_COLUMN])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Remove columns that should not be direct model inputs
    df = df.drop(columns=DROP_COLUMNS)

    # Handle missing values
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("Unknown").astype(str)

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    # Convert categorical values into numeric columns
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        dtype=int
    )

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split the data chronologically to better simulate
    # predicting future transport delays.
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Not enough records to create train and test sets."
        )

    # Save processed data
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # Save feature-column order for future predictions
    with (OUTPUT_DIR / "feature_columns.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("\n".join(X.columns))

    print("\nPreprocessing completed successfully!")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of input features:", X.shape[1])
    print("Target column:", TARGET_COLUMN)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
