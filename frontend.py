
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PROJECT_DIR = __import__("pathlib").Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "test_FD001.txt"

st.title("⚙️ Predictive Maintenance Dashboard")
st.caption("Remaining Useful Life prediction using a trained 50-cycle LSTM")

st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Test Engines", "100")
col2.metric("Test MAE", "16.95 cycles")
col3.metric("Test RMSE", "23.67 cycles")

st.caption(
    "Metrics from evaluation on the NASA C-MAPSS FD001 test set."
)

st.divider()


columns = [
    "unit", "cycle",
    "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

feature_cols = [
    "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

@st.cache_data
def load_test_data():
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
    df.columns = columns
    return df

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.success("Backend connected")
    else:
        st.error("Backend returned an unexpected status.")
except requests.RequestException:
    st.error("Backend unavailable. Start FastAPI before predicting.")
    st.stop()

try:
    test_df = load_test_data()
except Exception as error:
    st.error(f"Could not load the NASA test dataset: {error}")
    st.stop()

st.divider()
st.subheader("Engine RUL Prediction")

engine_ids = sorted(test_df["unit"].unique())
selected_engine = st.selectbox("Select engine", engine_ids)

engine_data = (
    test_df[test_df["unit"] == selected_engine]
    .sort_values("cycle")
)

st.write(f"Observed cycles: {len(engine_data)}")
st.write(f"Latest observed cycle: {int(engine_data['cycle'].max())}")

if st.button("Predict Remaining Useful Life", type="primary"):
    features = engine_data[feature_cols].to_numpy(dtype=float)

    # Prepare the final 50 observed cycles
    if len(features) < 50:
        padding = np.repeat(features[:1], 50 - len(features), axis=0)
        features = np.vstack([padding, features])
    else:
        features = features[-50:]

    payload = {"sensor_data": features.tolist()}

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            predicted_rul = result["predicted_rul_cycles"]

            st.metric(
                label="Predicted Remaining Useful Life",
                value=f"{predicted_rul:.2f} cycles",
            )
            st.caption(
                "Model estimate based on the engine's last 50 observed cycles."
            )
        else:
            st.error(f"Prediction failed ({response.status_code}): {response.text}")

    except requests.RequestException as error:
        st.error(f"Could not reach prediction API: {error}")


# Evaluation visualizations
st.divider()
st.subheader("Model Performance Analysis")

RESULTS_PATH = PROJECT_DIR / "api_evaluation_results.csv"

if RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH)

    # Actual vs Predicted RUL
    st.markdown("### Actual vs Predicted RUL")

    fig1, ax1 = plt.subplots()
    ax1.scatter(
        results_df["actual_rul"],
        results_df["predicted_rul"],
    )

    lower = min(
        results_df["actual_rul"].min(),
        results_df["predicted_rul"].min(),
    )
    upper = max(
        results_df["actual_rul"].max(),
        results_df["predicted_rul"].max(),
    )

    ax1.plot([lower, upper], [lower, upper], linestyle="--")
    ax1.set_xlabel("Actual RUL (cycles)")
    ax1.set_ylabel("Predicted RUL (cycles)")
    ax1.set_title("Actual vs Predicted")
    st.pyplot(fig1)
    plt.close(fig1)

    # Prediction error distribution
    results_df["error"] = (
        results_df["predicted_rul"] - results_df["actual_rul"]
    )

    st.markdown("### Prediction Error Distribution")

    fig2, ax2 = plt.subplots()
    ax2.hist(results_df["error"], bins=20)
    ax2.axvline(0, linestyle="--")
    ax2.set_xlabel("Prediction Error (cycles)")
    ax2.set_ylabel("Number of Engines")
    ax2.set_title("Prediction Error Distribution")
    st.pyplot(fig2)
    plt.close(fig2)

else:
    st.warning(
        "Evaluation results not found. Run evaluate_api.py "
        "to generate api_evaluation_results.csv."
    )
