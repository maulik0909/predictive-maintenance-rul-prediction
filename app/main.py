
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Predictive Maintenance API",
    description="Remaining Useful Life (RUL) prediction",
    version="1.0.0",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "rul_lstm_50.keras"
SCALER_PATH = PROJECT_DIR / "models" / "rul_scaler.pkl"

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("Scaler feature names:", getattr(scaler, "feature_names_in_", None))
print("Scaler mean shape:", scaler.mean_.shape)
print("Scaler scale shape:", scaler.scale_.shape)
print("Scaler feature count:", scaler.n_features_in_)
print("Model input shape:", model.input_shape)

SEQUENCE_LENGTH = 50
N_FEATURES = 24


class PredictionRequest(BaseModel):
    sensor_data: list[list[float]] = Field(
        ...,
        description="50 chronological cycles, each containing 24 features",
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_loaded": True,
        "sequence_length": SEQUENCE_LENGTH,
        "number_of_features": N_FEATURES,
    }


@app.post("/predict")
def predict_rul(request: PredictionRequest):
    data = np.asarray(request.sensor_data, dtype=np.float32)

    if data.shape != (SEQUENCE_LENGTH, N_FEATURES):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="sensor_data must contain exactly 50 cycles and 24 features per cycle",
        )

    scaled_data = scaler.transform(data)
    sequence = scaled_data.reshape(1, SEQUENCE_LENGTH, N_FEATURES)
    
    prediction = float(model.predict(sequence, verbose=0)[0][0])

    return {
        "predicted_rul_cycles": round(max(0.0, prediction), 2)
    }
