
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Keep the generated dataset reproducible
random.seed(42)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "transport_delays.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
TOTAL_RECORDS = 5000
START_DATE = datetime(2026, 1, 1, 5, 0)

routes = ["R101", "R102", "R103", "R104", "R105"]
stops = ["S01", "S02", "S03", "S04", "S05", "S06"]
vehicles = ["BUS01", "BUS02", "BUS03", "BUS04", "BUS05"]

traffic_levels = ["Low", "Medium", "High"]
traffic_weights = [0.25, 0.45, 0.30]

columns = [
    "Timestamp",
    "RouteID",
    "StopID",
    "VehicleID",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "Rain_mm",
    "TrafficLevel",
    "PassengerLoad",
    "PreviousDelayMinutes",
    "DelayMinutes",
]

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for i in range(TOTAL_RECORDS):
        timestamp = START_DATE + timedelta(minutes=15 * i)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = int(day_of_week >= 5)

        route = random.choice(routes)
        stop = random.choice(stops)
        vehicle = random.choice(vehicles)

        rain = round(random.choices(
            [0, random.uniform(0.1, 3), random.uniform(3, 15)],
            weights=[0.60, 0.25, 0.15]
        )[0], 2)

        traffic = random.choices(
            traffic_levels,
            weights=traffic_weights
        )[0]

        passenger_load = random.randint(10, 100)
        previous_delay = random.randint(0, 20)

        # Synthetic delay-generation logic.
        # This is simulated data, not real bus tracking data.
        delay = 2.0
        delay += {"Low": 0, "Medium": 4, "High": 9}[traffic]
        delay += rain * 0.35
        delay += max(0, passenger_load - 60) * 0.06
        delay += previous_delay * 0.25

        # Simulate heavier traffic during peak hours.
        if hour in [7, 8, 9, 17, 18, 19]:
            delay += 4

        if is_weekend:
            delay -= 1

        # Add random variation and keep delay non-negative.
        delay += random.gauss(0, 2.5)
        delay = max(0, round(delay, 2))

        writer.writerow({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "RouteID": route,
            "StopID": stop,
            "VehicleID": vehicle,
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "IsWeekend": is_weekend,
            "Rain_mm": rain,
            "TrafficLevel": traffic,
            "PassengerLoad": passenger_load,
            "PreviousDelayMinutes": previous_delay,
            "DelayMinutes": delay,
        })

print("Dataset generated successfully!")
print(f"Total records: {TOTAL_RECORDS}")
print(f"Saved to: {OUTPUT_FILE}")
print("Note: This is synthetic demonstration data.")
