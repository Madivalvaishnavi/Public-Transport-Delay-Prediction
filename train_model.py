
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load processed training and testing data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Save trained model
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

        results.append({
            "Model": name,
            "MAE_Minutes": round(mae, 3),
            "RMSE_Minutes": round(rmse, 3),
            "R2_Score": round(r2, 4),
        })

        print(f"MAE: {mae:.3f} minutes")
        print(f"RMSE: {rmse:.3f} minutes")
        print(f"R2 score: {r2:.4f}")
        print(f"Saved model: {model_path}")

    results_df = pd.DataFrame(results).sort_values("MAE_Minutes")
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nAll models trained successfully!")
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
