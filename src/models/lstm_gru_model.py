import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("file:./mlruns")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    X_train = np.load("data/X_train.npy")
    X_test  = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test  = np.load("data/y_test.npy")
    return X_train, X_test, y_train, y_test


# -----------------------------
# CLASS WEIGHTS
# -----------------------------
def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


# -----------------------------
# RNN MODEL
# -----------------------------
def build_rnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(64, activation="tanh", return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, activation="tanh"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# LSTM MODEL
# -----------------------------
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# GRU MODEL
# -----------------------------
def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# TRAIN & EVALUATE
# -----------------------------
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test, class_weights):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        
        print(f"\nTraining {name} model...")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1,
        )

        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"\n{name} Results:")
        print("Accuracy:", round(acc, 4))
        print("F1 Score:", round(f1, 4))
        print(classification_report(y_test, y_pred, zero_division=0))

    return acc, f1


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("MarketPulse_ANN_Project")

    class_weights = get_class_weights(y_train)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 1. RNN
    rnn_model = build_rnn(input_shape)
    rnn_acc, rnn_f1 = train_and_evaluate(rnn_model, "RNN", X_train, X_test, y_train, y_test, class_weights)
    rnn_model.save("models/rnn_model.keras")
    print("[SAVED] RNN model -> models/rnn_model.keras")

    # 2. LSTM
    lstm_model = build_lstm(input_shape)
    lstm_acc, lstm_f1 = train_and_evaluate(lstm_model, "LSTM", X_train, X_test, y_train, y_test, class_weights)
    lstm_model.save("models/lstm_model.keras")
    print("[SAVED] LSTM model -> models/lstm_model.keras")

    # 3. GRU
    gru_model = build_gru(input_shape)
    gru_acc, gru_f1 = train_and_evaluate(gru_model, "GRU", X_train, X_test, y_train, y_test, class_weights)
    gru_model.save("models/gru_model.keras")
    print("[SAVED] GRU model -> models/gru_model.keras")

    print("\nFINAL COMPARISON")
    print(f"RNN  -> Accuracy: {rnn_acc:.4f} | F1: {rnn_f1:.4f}")
    print(f"LSTM -> Accuracy: {lstm_acc:.4f} | F1: {lstm_f1:.4f}")
    print(f"GRU  -> Accuracy: {gru_acc:.4f} | F1: {gru_f1:.4f}")