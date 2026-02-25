"""
Train ML models on NASA exoplanet data.

Supports two tasks:
  1. Habitability classification (binary) — predict whether a planet is
     potentially habitable based on equilibrium temperature, radius, and
     insolation flux.
  2. Planet type classification (multi-class) — categorize planets into
     Rocky, Super-Earth, Sub-Neptune, Neptune-like, or Gas Giant.

Models trained: Random Forest, Gradient Boosting, and Logistic Regression
(or their equivalents). The best model by cross-validated F1 score is saved.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.preprocess import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_candidates(target: str) -> dict:
    """Return candidate models suitable for the target task."""
    if target == "habitable":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        }
    else:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
        }


def train_and_evaluate(
    df: pd.DataFrame,
    target: str = "habitable",
    test_size: float = 0.2,
) -> dict:
    """Train candidate models, evaluate, and save the best one.

    Args:
        df: Raw exoplanet DataFrame.
        target: "habitable" or "planet_type".
        test_size: Fraction of data to hold out for testing.

    Returns:
        Dictionary with evaluation metrics and model metadata.
    """
    print(f"\n{'='*60}")
    print(f"  Training task: {target}")
    print(f"{'='*60}\n")

    # Preprocess
    X, y, scaler, le = preprocess(df, target=target)
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    print(f"Target distribution:\n{y.value_counts().to_string()}\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Cross-validated model selection
    candidates = get_candidates(target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "f1_weighted" if target == "planet_type" else "f1"

    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
        }
        print(f"  {name:25s}  CV F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Select best model
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_f1"])
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}\n")

    # Refit on full training set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Evaluation on held-out test set
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(
        y_test, y_pred, average="weighted" if target == "planet_type" else "binary"
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_ if le else ["Not Habitable", "Potentially Habitable"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("Test set results:")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=(
                le.classes_ if le else ["Not Habitable", "Potentially Habitable"]
            ),
        )
    )
    print(f"Confusion Matrix:\n{cm}\n")

    # Feature importances (for tree-based models)
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        feat_imp = sorted(
            zip(X.columns, imp), key=lambda x: x[1], reverse=True
        )
        print("Top 10 Feature Importances:")
        for fname, fval in feat_imp[:10]:
            print(f"  {fname:30s}  {fval:.4f}")
            importances[fname] = float(fval)
        print()

    # Save model artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{target}_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{target}_scaler.joblib")
    meta_path = os.path.join(MODELS_DIR, f"{target}_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    if le:
        le_path = os.path.join(MODELS_DIR, f"{target}_label_encoder.joblib")
        joblib.dump(le, le_path)

    metadata = {
        "target": target,
        "best_model": best_name,
        "features": list(X.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_results": cv_results,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "classification_report": report,
        "feature_importances": importances,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Metadata saved to {meta_path}")

    return metadata
