
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "transport_delays.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    df = pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # Aggregate 15-minute observations into hourly averages
    series = (
        df.set_index("Timestamp")["DelayMinutes"]
        .resample("1h")
        .mean()
        .dropna()
    )

    split = int(len(series) * 0.80)
    train = series.iloc[:split]
    test = series.iloc[split:]

    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print("Fitting SARIMA model...")

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False, maxiter=100)
    predictions = fitted_model.forecast(steps=len(test))
    predictions.index = test.index

    mae = mean_absolute_error(test, predictions)
    rmse = mean_squared_error(test, predictions) ** 0.5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "Timestamp": test.index,
        "ActualDelayMinutes": test.to_numpy(),
        "PredictedDelayMinutes": predictions.to_numpy()
    })

    results.to_csv(
        OUTPUT_DIR / "sarima_predictions.csv",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test.to_numpy(), label="Actual")
    plt.plot(
        predictions.index,
        predictions.to_numpy(),
        label="SARIMA forecast"
    )
    plt.title("SARIMA: Actual vs Predicted Transport Delays")
    plt.xlabel("Timestamp")
    plt.ylabel("Average delay (minutes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sarima_forecast.png", dpi=150)
    plt.close()

    print("\nSARIMA EVALUATION")
    print("MAE:", round(mae, 3), "minutes")
    print("RMSE:", round(rmse, 3), "minutes")
    print("Saved predictions:", OUTPUT_DIR / "sarima_predictions.csv")
    print("Saved graph:", OUTPUT_DIR / "sarima_forecast.png")


if __name__ == "__main__":
    main()
