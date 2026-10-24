
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "transport_delays.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python generate_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and sort the data
    df = pd.read_csv(INPUT_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # Aggregate all routes and stops into 5-minute bins.
    # Because source records are 15 minutes apart, many 5-minute
    # intervals will have no observations. Those intervals are
    # interpolated for this demonstration analysis.
    series = (
        df.set_index("Timestamp")["DelayMinutes"]
        .resample("5min")
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    series.to_csv(OUTPUT_DIR / "delay_5min_series.csv",
                  header=["DelayMinutes"])

    print("TIME-SERIES ANALYSIS")
    print("=" * 45)
    print("Original records:", len(df))
    print("5-minute intervals:", len(series))
    print("Missing values after interpolation:", series.isna().sum())

    # Augmented Dickey-Fuller stationarity test
    adf_result = adfuller(series.dropna())

    print("\nADF STATIONARITY TEST")
    print("ADF statistic:", round(adf_result[0], 4))
    print("p-value:", round(adf_result[1], 6))

    if adf_result[1] < 0.05:
        print("Result: Evidence of stationarity at the 5% level.")
    else:
        print("Result: Insufficient evidence of stationarity.")
        print("Differencing or further analysis may be needed.")

    # Save a time-series plot
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values)
    plt.title("Transport Delay Over Time (5-Minute Series)")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Delay (minutes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_series_plot.png", dpi=150)
    plt.close()

    # Seasonal decomposition
    # The data is sampled every 5 minutes. A daily period is
    # 24 * 60 / 5 = 288 intervals.
    if len(series) >= 2 * 288:
        decomposition = seasonal_decompose(
            series,
            model="additive",
            period=288
        )

        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "seasonal_decomposition.png",
            dpi=150
        )
        plt.close(fig)

        print("Seasonal decomposition graph saved.")
    else:
        print("Not enough observations for daily decomposition.")

    print("\nTime-series analysis completed!")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
