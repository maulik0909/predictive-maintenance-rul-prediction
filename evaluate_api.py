
import json
import urllib.request

import numpy as np
import pandas as pd

BASE_DIR = "data/raw"

columns = [
    "unit", "cycle",
    "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

feature_cols = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    f"{BASE_DIR}/test_FD001.txt",
    sep=r"\s+",
    header=None,
)
test_df.columns = columns

actual_rul = pd.read_csv(
    f"{BASE_DIR}/RUL_FD001.txt",
    sep=r"\s+",
    header=None,
).iloc[:, 0].to_numpy(dtype=float)

results = []

for unit_id in sorted(test_df["unit"].unique()):
    engine = test_df[test_df["unit"] == unit_id].sort_values("cycle")
    features = engine[feature_cols].to_numpy(dtype=float)

    if len(features) < 50:
        padding = np.repeat(features[:1], 50 - len(features), axis=0)
        features = np.vstack([padding, features])
    else:
        features = features[-50:]

    payload = json.dumps({
        "sensor_data": features.tolist()
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        prediction = json.loads(response.read().decode("utf-8"))
        predicted_rul = prediction["predicted_rul_cycles"]

    results.append({
        "engine": unit_id,
        "actual_rul": actual_rul[unit_id - 1],
        "predicted_rul": predicted_rul,
    })

df = pd.DataFrame(results)
df["absolute_error"] = abs(df["actual_rul"] - df["predicted_rul"])

mae = df["absolute_error"].mean()
rmse = np.sqrt(((df["actual_rul"] - df["predicted_rul"]) ** 2).mean())

print("\nEvaluation across test engines")
print("Number of engines:", len(df))
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")

print("\nFirst five results:")
print(df.head().to_string(index=False))

df.to_csv("api_evaluation_results.csv", index=False)
print("\nSaved results to api_evaluation_results.csv")
