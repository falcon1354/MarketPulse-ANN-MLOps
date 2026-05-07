import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    return X_train, X_test, y_train, y_test


# -----------------------------
# BUILD IMPROVED RNN MODEL
# -----------------------------
def build_rnn(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------
# TRAIN + EVALUATE
# -----------------------------
def train_model():
    X_train, X_test, y_train, y_test = load_data()

    print("Training data shape:", X_train.shape)

    # 🔍 Check class distribution
    print("Train labels distribution:", np.unique(y_train, return_counts=True))

    # ⚖️ Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    print("Class weights:", class_weights)

    # Build model
    model = build_rnn((X_train.shape[1], X_train.shape[2]))

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=8,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # 🔍 Debug predictions
    print("\nPredictions distribution:", np.unique(y_pred, return_counts=True))
    print("Actual distribution:", np.unique(y_test, return_counts=True))

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n✅ FINAL RESULTS")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    return model, history


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_model()