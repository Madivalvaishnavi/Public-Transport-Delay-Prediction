
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

X_TEST_FILE = PROCESSED_DIR / "X_test.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

MODEL_NAMES = [
    "linear_regression",
    "random_forest",
    "gradient_boosting",
]


def main():
    # Check that testing data exists
    if not X_TEST_FILE.exists() or not Y_TEST_FILE.exists():
        raise FileNotFoundError(
            "Test data not found. Run python preprocess.py first."
        )

    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{name}.joblib"

        if not model_path.exists():
            print(f"Skipping {name}: trained model not found.")
            continue

        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test, predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE_Minutes": round(mae, 3),
            "RMSE_Minutes": round(rmse, 3),
            "R2_Score": round(r2, 4),
        })

    if not results:
        raise FileNotFoundError(
            "No trained models found. Run python train_model.py first."
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "MAE_Minutes"
    ).reset_index(drop=True)

    results_file = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(results_file, index=False)

    print("\nMODEL EVALUATION RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("\nLower MAE and RMSE are better.")
    print("R2 closer to 1 generally indicates a better fit.")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
