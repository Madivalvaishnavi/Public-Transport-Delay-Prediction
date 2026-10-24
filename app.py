
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Paths and page setup
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="TransitDelay AI",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #8a94a6;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load saved models and files
# --------------------------------------------------
@st.cache_data
def load_feature_columns():
    path = PROCESSED_DIR / "feature_columns.txt"
    if not path.exists():
        raise FileNotFoundError(
            "processed_data/feature_columns.txt is missing. "
            "Run python preprocess.py first."
        )
    return path.read_text(encoding="utf-8").splitlines()


@st.cache_resource
def load_model(name):
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_options(columns, prefix, fallback):
    """Read available category values from one-hot feature names."""
    options = [
        col[len(prefix):]
        for col in columns
        if col.startswith(prefix)
    ]
    return options if options else fallback


st.markdown(
    '<div class="main-title">🚌 TransitDelay AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Public Transport Delay Prediction and Model Analytics'
    '</div>',
    unsafe_allow_html=True,
)

# Check essential files before displaying the dashboard.
try:
    feature_columns = load_feature_columns()
except Exception as exc:
    st.error(str(exc))
    st.stop()

model_names = {
    "Gradient Boosting": "gradient_boosting",
    "Random Forest": "random_forest",
    "Linear Regression": "linear_regression",
}

available_models = {
    label: filename
    for label, filename in model_names.items()
    if (MODEL_DIR / f"{filename}.joblib").exists()
}

if not available_models:
    st.error("No trained models found in the models folder.")
    st.stop()

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Delay Prediction",
        "Model Performance",
        "Forecast Graphs",
        "Project Information",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine-learning and time-series demonstration")
st.sidebar.caption("Dataset: synthetic demonstration data")


# --------------------------------------------------
# Page 1: Interactive delay prediction
# --------------------------------------------------
if page == "Delay Prediction":
    st.header("Predict a Transport Delay")
    st.write(
        "Enter the transport conditions below. "
        "The selected trained model estimates the delay in minutes."
    )

    route_options = get_options(
        feature_columns, "RouteID_", ["R101", "R102", "R103", "R104", "R105"]
    )
    stop_options = get_options(
        feature_columns, "StopID_", ["S01", "S02", "S03", "S04", "S05", "S06"]
    )
    vehicle_options = get_options(
        feature_columns,
        "VehicleID_",
        ["BUS01", "BUS02", "BUS03", "BUS04", "BUS05"],
    )
    traffic_options = get_options(
        feature_columns, "TrafficLevel_", ["Low", "Medium", "High"]
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            route = st.selectbox("Route", route_options)
            stop = st.selectbox("Stop", stop_options)
            vehicle = st.selectbox("Vehicle", vehicle_options)
            traffic = st.selectbox("Traffic level", traffic_options)

            hour = st.slider("Hour of day", 0, 23, 8)
            day_of_week = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda day: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][day],
            )

        with right:
            rain = st.number_input(
                "Rainfall (mm)", min_value=0.0, max_value=100.0,
                value=0.0, step=0.1
            )
            passenger_load = st.slider(
                "Passenger load", min_value=10, max_value=100, value=50
            )
            previous_delay = st.number_input(
                "Previous delay (minutes)",
                min_value=0.0, max_value=300.0,
                value=5.0, step=1.0
            )
            model_label = st.selectbox(
                "Prediction model", list(available_models.keys())
            )

        submitted = st.form_submit_button(
            "Predict Delay", type="primary", use_container_width=True
        )

    if submitted:
        # Initialize exactly the columns and order used during training.
        input_row = {column: 0 for column in feature_columns}

        numeric_values = {
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "IsWeekend": int(day_of_week >= 5),
            "Rain_mm": rain,
            "PassengerLoad": passenger_load,
            "PreviousDelayMinutes": previous_delay,
        }

        for column, value in numeric_values.items():
            if column in input_row:
                input_row[column] = value

        # Activate the matching one-hot encoded category columns.
        selected_categories = {
            "RouteID": route,
            "StopID": stop,
            "VehicleID": vehicle,
            "TrafficLevel": traffic,
        }

        for category, value in selected_categories.items():
            dummy_column = f"{category}_{value}"
            if dummy_column in input_row:
                input_row[dummy_column] = 1

        input_df = pd.DataFrame([input_row], columns=feature_columns)

        try:
            model = load_model(available_models[model_label])
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, prediction)

            st.subheader("Prediction Result")
            metric_col, status_col = st.columns([1, 2])

            with metric_col:
                st.metric("Predicted delay", f"{prediction:.2f} min")

            with status_col:
                if prediction < 3:
                    st.success("Low predicted delay")
                elif prediction < 7:
                    st.warning("Moderate predicted delay")
                else:
                    st.error("High predicted delay")

            st.info(
                "This is an estimate from a model trained on synthetic data. "
                "It is not a live bus-tracking result or a guarantee of "
                "actual arrival time."
            )

            st.markdown("**Input summary**")
            st.dataframe(
                pd.DataFrame(
                    [{"Field": key, "Value": value}
                     for key, value in {
                         "Route": route,
                         "Stop": stop,
                         "Vehicle": vehicle,
                         "Traffic": traffic,
                         "Hour": hour,
                         "Day of week": day_of_week,
                         "Rainfall (mm)": rain,
                         "Passenger load": passenger_load,
                         "Previous delay (minutes)": previous_delay,
                         "Model": model_label,
                     }.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# Page 2: Model performance
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison = load_csv(str(comparison_path))

    if comparison.empty:
        st.warning(
            "Model comparison file not found. "
            "Run python evaluate_model.py first."
        )
    else:
        st.dataframe(comparison, hide_index=True, use_container_width=True)

        if "Model" in comparison.columns and "MAE_Minutes" in comparison.columns:
            chart_data = comparison.set_index("Model")[["MAE_Minutes"]]
            st.subheader("Mean Absolute Error")
            st.bar_chart(chart_data)

        if "Model" in comparison.columns and "RMSE_Minutes" in comparison.columns:
            st.subheader("Root Mean Squared Error")
            st.bar_chart(comparison.set_index("Model")[["RMSE_Minutes"]])

        if "Model" in comparison.columns and "R2_Score" in comparison.columns:
            st.subheader("R² Score")
            st.bar_chart(comparison.set_index("Model")[["R2_Score"]])

        st.download_button(
            "Download model comparison CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# Page 3: Existing forecast and prediction graphs
# --------------------------------------------------
elif page == "Forecast Graphs":
    st.header("Prediction and Forecast Visualizations")

    graph_files = [
        ("Actual vs Predicted Delays", "actual_vs_predicted.png"),
        ("Prediction Scatter Plot", "prediction_scatter.png"),
        ("Prediction Error Distribution", "prediction_errors.png"),
        ("Time-Series Delay Plot", "time_series_plot.png"),
        ("Seasonal Decomposition", "seasonal_decomposition.png"),
        ("ARIMA Forecast", "arima_forecast.png"),
        ("SARIMA Forecast", "sarima_forecast.png"),
    ]

    for title, filename in graph_files:
        path = OUTPUT_DIR / filename
        if path.exists():
            with st.expander(title, expanded=(filename == "actual_vs_predicted.png")):
                st.image(str(path), use_container_width=True)
        else:
            st.caption(f"{title}: graph not available yet ({filename})")

    for title, filename in [
        ("Machine-learning predictions", "predictions.csv"),
        ("ARIMA predictions", "arima_predictions.csv"),
        ("SARIMA predictions", "sarima_predictions.csv"),
    ]:
        path = OUTPUT_DIR / filename
        if path.exists():
            with st.expander(f"View {title}"):
                data = load_csv(str(path))
                st.dataframe(data.head(100), use_container_width=True)
                st.download_button(
                    f"Download {filename}",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                )


# --------------------------------------------------
# Page 4: Project information
# --------------------------------------------------
else:
    st.header("About This Project")

    st.markdown(
        """
        **Project:** Public Transport Delay Prediction

        **Goal:** Estimate transport delays from transport conditions and
        compare machine-learning and statistical forecasting approaches.

        **Machine-learning models**
        - Linear Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor

        **Time-series analysis**
        - Resampling and interpolation
        - Augmented Dickey-Fuller stationarity test
        - Seasonal decomposition
        - ARIMA and SARIMA forecast experiments

        **Evaluation metrics**
        - MAE: mean absolute prediction error
        - RMSE: error metric that penalizes larger errors
        - R²: goodness-of-fit measure

        **Dataset limitation:** The current dataset is synthetically generated.
        The dashboard demonstrates a modelling workflow; it does not use
        live vehicle GPS, traffic feeds, or actual public transport records.
        """
    )

    st.subheader("Project files")
    for path in [
        BASE_DIR / "generate_data.py",
        BASE_DIR / "preprocess.py",
        BASE_DIR / "train_model.py",
        BASE_DIR / "evaluate_model.py",
        BASE_DIR / "time_series_analysis.py",
        BASE_DIR / "arima_model.py",
        BASE_DIR / "sarima_model.py",
    ]:
        st.write(("✅ " if path.exists() else "⚠️ ") + path.name)

st.markdown("---")
st.caption("TransitDelay AI · Educational forecasting demonstration")
