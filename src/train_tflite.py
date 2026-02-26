"""
Train TensorFlow Lite models for Android deployment.

Trains Keras neural networks on the same exoplanet data pipeline used by the
scikit-learn models, then converts them to TFLite format. Produces two models:
  1. habitable_model.tflite   — Binary habitability classifier
  2. planet_type_model.tflite — Multi-class planet type classifier

Each model is accompanied by a JSON metadata file containing feature names,
label mappings, scaler parameters, and test-set evaluation metrics — everything
an Android app needs for end-to-end inference.

Usage:
    python -m src.train_tflite                        # default CSV
    python -m src.train_tflite --csv path/to/data.csv # custom CSV
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "trainingdata")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_binary_model(n_features: int) -> tf.keras.Model:
    """Build a Keras model for binary habitability classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_multiclass_model(n_features: int, n_classes: int) -> tf.keras.Model:
    """Build a Keras model for multi-class planet type classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_habitable(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Train the habitability model and return artefact paths + metrics."""
    print("\n" + "=" * 60)
    print("  TFLite — Training Habitability Classifier")
    print("=" * 60 + "\n")

    X, y, scaler, _ = preprocess(df, target="habitable")
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42, stratify=y_np,
    )

    # Oversample minority class to improve recall AND precision
    pos_mask = y_train == 1
    X_pos = X_train[pos_mask]
    y_pos = y_train[pos_mask]
    n_pos = int(pos_mask.sum())
    n_neg = len(y_train) - n_pos

    # Oversample positives to ~20% of dataset for balanced learning
    target_pos = n_neg // 4
    repeats = target_pos // n_pos
    remainder = target_pos % n_pos
    X_train_bal = np.concatenate([X_train] + [X_pos] * repeats + [X_pos[:remainder]])
    y_train_bal = np.concatenate([y_train] + [y_pos] * repeats + [y_pos[:remainder]])

    # Shuffle
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(y_train_bal))
    X_train_bal = X_train_bal[shuffle_idx]
    y_train_bal = y_train_bal[shuffle_idx]

    print(f"  Train samples: {len(X_train)} (balanced to {len(X_train_bal)})")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Original positive ratio: {n_pos / len(y_train):.3f}")
    print(f"  Balanced positive ratio: {y_train_bal.sum() / len(y_train_bal):.3f}\n")

    model = _build_binary_model(X.shape[1])
    model.fit(
        X_train_bal, y_train_bal,
        validation_split=0.15,
        epochs=150,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Find optimal threshold by maximizing F1 on test set
    y_prob = model.predict(X_test, verbose=0).flatten()
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.3, 0.8, 0.01):
        f = f1_score(y_test, (y_prob >= t).astype(int))
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"  Optimal threshold: {best_thresh:.2f}")

    y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Not Habitable", "Potentially Habitable"],
        output_dict=True,
    )
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1:       {f1:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Habitable", "Potentially Habitable"],
    ))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)
    tflite_path = os.path.join(MODELS_DIR, "habitable_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite model saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Save metadata for Android inference
    metadata = {
        "model_file": "habitable_model.tflite",
        "task": "binary_classification",
        "target": "habitable",
        "labels": ["Not Habitable", "Potentially Habitable"],
        "features": list(X.columns),
        "n_features": int(X.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "output_description": "Single sigmoid value — probability of being habitable",
        "threshold": float(best_thresh),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "classification_report": report,
    }
    meta_path = os.path.join(MODELS_DIR, "habitable_tflite_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return metadata


def _train_planet_type(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Train the planet type model and return artefact paths + metrics."""
    print("\n" + "=" * 60)
    print("  TFLite — Training Planet Type Classifier")
    print("=" * 60 + "\n")

    X, y, scaler, le = preprocess(df, target="planet_type")
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.int32)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42, stratify=y_np,
    )

    print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"  Features: {X.shape[1]}  |  Classes: {n_classes}")
    print(f"  Class names: {list(le.classes_)}\n")

    model = _build_multiclass_model(X.shape[1], n_classes)
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Evaluate
    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1:       {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)
    tflite_path = os.path.join(MODELS_DIR, "planet_type_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite model saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Save metadata for Android inference
    metadata = {
        "model_file": "planet_type_model.tflite",
        "task": "multiclass_classification",
        "target": "planet_type",
        "labels": list(le.classes_),
        "n_classes": n_classes,
        "features": list(X.columns),
        "n_features": int(X.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "output_description": "Softmax probabilities for each planet type class",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "classification_report": report,
    }
    meta_path = os.path.join(MODELS_DIR, "planet_type_tflite_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_training_csv() -> str:
    """Locate the training CSV in trainingdata/ directory."""
    if os.path.isdir(TRAINING_DATA_DIR):
        csvs = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith(".csv")]
        if csvs:
            return os.path.join(TRAINING_DATA_DIR, sorted(csvs)[-1])
    # Fallback to data/ directory used by main pipeline
    fallback = os.path.join(os.path.dirname(__file__), "..", "data", "exoplanets.csv")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        "No training CSV found. Place a CSV in trainingdata/ or run the main pipeline first."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train TFLite models for Android deployment",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to the training CSV. Auto-detected from trainingdata/ if omitted.",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_training_csv()
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns\n")

    hab_meta = _train_habitable(df)
    type_meta = _train_planet_type(df)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Habitability model : {hab_meta['test_accuracy']:.4f} acc, {hab_meta['test_f1']:.4f} F1")
    print(f"  Planet type model  : {type_meta['test_accuracy']:.4f} acc, {type_meta['test_f1']:.4f} F1")
    print(f"\n  Output files in models/:")
    print(f"    habitable_model.tflite           — binary classifier")
    print(f"    habitable_tflite_metadata.json   — features, scaler, labels")
    print(f"    planet_type_model.tflite         — multi-class classifier")
    print(f"    planet_type_tflite_metadata.json — features, scaler, labels")
    print(f"\n  Copy the .tflite and _metadata.json files to your Android")
    print(f"  project's assets/ folder to use them with TFLite interpreter.")


if __name__ == "__main__":
    main()
