
import json
import urllib.request

import numpy as np
import pandas as pd

# Load the NASA C-MAPSS test dataset
columns = [
    "unit", "cycle",
    "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

test_df = pd.read_csv(
    "data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None,
)
test_df.columns = columns

# Use the first engine and its final 50 observed cycles
engine = test_df[test_df["unit"] == 1].sort_values("cycle")

feature_cols = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

features = engine[feature_cols].to_numpy(dtype=float)

# Pad shorter sequences if necessary
if len(features) < 50:
    padding = np.repeat(features[:1], 50 - len(features), axis=0)
    features = np.vstack([padding, features])
else:
    features = features[-50:]

# Send the request to FastAPI
payload = json.dumps({
    "sensor_data": features.tolist()
}).encode("utf-8")

request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))
        
    print("Status:", response.status)
    print("Prediction:", result)

    # Load the actual RUL values
    rul_df = pd.read_csv(
        "data/raw/RUL_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    actual_rul = float(rul_df.iloc[0, 0])
    predicted_rul = result["predicted_rul_cycles"]

    absolute_error = abs(predicted_rul - actual_rul)

    print("Actual RUL:", actual_rul, "cycles")
    print("Predicted RUL:", predicted_rul, "cycles")
    print("Absolute error:", round(absolute_error, 2), "cycles")

except Exception as error:
    print("Request or evaluation failed:",error)