import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


# -----------------------------
# LOAD PROCESSED DATA
# -----------------------------
def load_data(path="data/processed_data.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna()
    return df


# -----------------------------
# CREATE SEQUENCES (RNN/LSTM/GRU)
# -----------------------------
def create_sequences(data, window_size=10):
    X, y = [], []
    features = data[["sentiment_score", "price", "price_change"]].values
    labels   = data["label"].values

    for i in range(len(data) - window_size):
        X.append(features[i : i + window_size])
        y.append(labels[i + window_size])

    return np.array(X), np.array(y)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def prepare_dataset():
    df = load_data()
    print("Dataset loaded:", df.shape)
    print("Full label distribution:", df["label"].value_counts().to_dict())
    
    # Add price_change feature (grouped by ticker to avoid jumps between stocks)
    df["price_change"] = df.groupby("ticker")["price"].diff()
    df = df.dropna()

    # ── Step 1: Chronological split BEFORE any scaling ──────────────────────
    # shuffle=False is mandatory for time-series — future must not leak into past
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(df_train)} | Test rows: {len(df_test)}")

    # ── Step 2: Scale (fit ONLY on train to prevent data leakage) ───────────
    feature_cols = ["sentiment_score", "price", "price_change"]
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    os.makedirs("data", exist_ok=True)
    joblib.dump(scaler, "data/scaler.pkl")
    print("Scaler saved to data/scaler.pkl")

    # ── Step 3: Create sequences from each split separately ─────────────────
    window_size = 10
    X_train, y_train = create_sequences(df_train, window_size)
    X_test,  y_test  = create_sequences(df_test,  window_size)

    print("\nSequences created:")
    print("  X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("  X_test :", X_test.shape,  "| y_test :", y_test.shape)
    print("  Train label dist:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("  Test  label dist:", dict(zip(*np.unique(y_test,  return_counts=True))))

    # ── Save ────────────────────────────────────────────────────────────────
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy",  X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy",  y_test)

    print("\nDataset ready for ANN/RNN models!")


if __name__ == "__main__":
    prepare_dataset()