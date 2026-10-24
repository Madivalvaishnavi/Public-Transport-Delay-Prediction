
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "transport_delays.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    df = pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # Aggregate observations into one chronological 15-minute series.
    # Each timestamp in the generated dataset has one record.
    series = (
        df.set_index("Timestamp")["DelayMinutes"]
        .resample("15min")
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    split = int(len(series) * 0.80)
    train = series.iloc[:split]
    test = series.iloc[split:]

    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print("\nFitting ARIMA(2, 1, 2)...")

    model = ARIMA(train, order=(2, 1, 2))
    fitted_model = model.fit()

    predictions = fitted_model.forecast(steps=len(test))
    predictions.index = test.index

    mae = mean_absolute_error(test, predictions)
    rmse = mean_squared_error(test, predictions) ** 0.5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame({
        "Timestamp": test.index,
        "ActualDelayMinutes": test.values,
        "PredictedDelayMinutes": predictions.values,
    })

    comparison.to_csv(
        OUTPUT_DIR / "arima_predictions.csv",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(predictions.index, predictions.values, label="ARIMA forecast")
    plt.title("ARIMA: Actual vs Predicted Transport Delays")
    plt.xlabel("Timestamp")
    plt.ylabel("Delay (minutes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arima_forecast.png", dpi=150)
    plt.close()

    print("\nARIMA EVALUATION")
    print("MAE:", round(mae, 3), "minutes")
    print("RMSE:", round(rmse, 3), "minutes")
    print("Forecast saved:", OUTPUT_DIR / "arima_predictions.csv")
    print("Graph saved:", OUTPUT_DIR / "arima_forecast.png")


if __name__ == "__main__":
    main()
