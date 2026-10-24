
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_NAME = "gradient_boosting"


def main():
    # Load testing data and trained model
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    # Save actual and predicted values
    comparison = pd.DataFrame({
        "ActualDelayMinutes": y_test.to_numpy(),
        "PredictedDelayMinutes": predictions
    })

    comparison.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    # Graph 1: Actual versus predicted delays
    plt.figure(figsize=(12, 5))
    plt.plot(
        comparison["ActualDelayMinutes"].iloc[:100].to_numpy(),
        label="Actual Delay"
    )
    plt.plot(
        comparison["PredictedDelayMinutes"].iloc[:100].to_numpy(),
        label="Predicted Delay"
    )
    plt.title("Actual vs Predicted Transport Delays")
    plt.xlabel("Test Record")
    plt.ylabel("Delay (minutes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=150
    )
    plt.close()

    # Graph 2: Actual versus predicted scatter plot
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.6)

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.title("Actual vs Predicted Delay Values")
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "prediction_scatter.png",
        dpi=150
    )
    plt.close()

    # Graph 3: Distribution of prediction errors
    errors = y_test.to_numpy() - predictions

    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=30, edgecolor="black")
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Actual - Predicted (minutes)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "prediction_errors.png",
        dpi=150
    )
    plt.close()

    print("Visualization completed successfully!")
    print("Model:", MODEL_NAME)
    print("Test records:", len(y_test))
    print(
        "MAE:",
        round(mean_absolute_error(y_test, predictions), 3),
        "minutes"
    )
    print("Saved predictions and 3 graphs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
