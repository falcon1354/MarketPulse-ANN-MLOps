from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

# ── Canonical feature list ──────────────────────────────────────────────────
# MUST match FEATURE_COLS in dataset_builder.py. Order matters for the scaler.
FEATURE_COLS = ["sentiment_score", "price"]
WINDOW_SIZE  = 10

# ── Load model & scaler ─────────────────────────────────────────────────────
model  = tf.keras.models.load_model("models/lstm_model.keras")
scaler = joblib.load("data/scaler.pkl")

# Sanity-check: scaler must have been fitted on the same number of features
assert scaler.n_features_in_ == len(FEATURE_COLS), (
    f"Scaler expects {scaler.n_features_in_} features but FEATURE_COLS has "
    f"{len(FEATURE_COLS)}. Regenerate scaler by re-running dataset_builder.py."
)


# ── Request schema ──────────────────────────────────────────────────────────
class InputData(BaseModel):
    sentiment_score: list[float]
    price: list[float]


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "MarketPulse API Running",
        "features": FEATURE_COLS,
        "window_size": WINDOW_SIZE,
    }


@app.post("/predict")
def predict(data: InputData):
    try:
        sentiment = data.sentiment_score
        price     = data.price

        # Validate window size
        if len(sentiment) != WINDOW_SIZE or len(price) != WINDOW_SIZE:
            return {"error": f"Each feature must contain exactly {WINDOW_SIZE} values."}

        # Build feature matrix: shape (WINDOW_SIZE, len(FEATURE_COLS)) = (10, 2)
        # Column order must match FEATURE_COLS = ["sentiment_score", "price"]
        features = np.column_stack([sentiment, price])          # (10, 2)

        # Scale using the same scaler fitted in dataset_builder.py
        features_scaled = scaler.transform(features)            # (10, 2)

        # Add batch dimension → (1, 10, 2)
        X = np.expand_dims(features_scaled, axis=0)

        # Predict
        pred   = float(model.predict(X)[0][0])
        result = int(pred > 0.5)

        return {
            "prediction": result,
            "confidence": round(pred, 4),
            "label": "UP" if result == 1 else "DOWN",
        }

    except Exception as e:
        return {"error": str(e)}