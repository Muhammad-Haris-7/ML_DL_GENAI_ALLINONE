import io
import base64

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def fig_to_base64(fig):
    buf = io.BytesIO()

    fig.savefig(buf, format="png", bbox_inches="tight")

    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)

    return encoded


def run_dl_pipeline(csv_path, target_column):

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found.")

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # Handle missing values
    for col in X.columns:

        if is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

        else:
            mode_vals = X[col].mode()

            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "missing"

            X[col] = X[col].fillna(fill_val)

    # Encode categorical columns
    for col in X.columns:

        if not is_numeric_dtype(X[col]):
            encoder = LabelEncoder()

            X[col] = encoder.fit_transform(X[col].astype(str))

    # Encode target
    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(y.astype(str))

    num_classes = len(np.unique(y_encoded))

    # Feature Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    # Multi-class support
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Neural Network
    model = Sequential()

    model.add(Dense(64, activation="relu",
                    input_shape=(X_train.shape[1],)))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=20,
        batch_size=32,
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, predicted_classes)

    # Accuracy Graph
    fig1, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(history.history["accuracy"],
             label="Training Accuracy")

    ax1.plot(history.history["val_accuracy"],
             label="Validation Accuracy")

    ax1.set_title("Neural Network Accuracy")

    ax1.set_xlabel("Epoch")

    ax1.set_ylabel("Accuracy")

    ax1.legend()

    accuracy_chart = fig_to_base64(fig1)

    # Loss Graph
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    ax2.plot(history.history["loss"],
             label="Training Loss")

    ax2.plot(history.history["val_loss"],
             label="Validation Loss")

    ax2.set_title("Neural Network Loss")

    ax2.set_xlabel("Epoch")

    ax2.set_ylabel("Loss")

    ax2.legend()

    loss_chart = fig_to_base64(fig2)

    return {

        "accuracy": round(accuracy * 100, 2),

        "rows": len(df),

        "features": X.shape[1],

        "target_column": target_column,

        "accuracy_chart": accuracy_chart,

        "loss_chart": loss_chart

    }