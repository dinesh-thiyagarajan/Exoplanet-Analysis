"""
Load trained models and make predictions on exoplanet data.

Provides functions to:
  - Load a saved model and predict on new data
  - Find the most promising habitable planet candidates
  - Generate summary statistics and analysis
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.preprocess import engineer_features, NUMERIC_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model(target: str = "habitable"):
    """Load a trained model, scaler, and metadata.

    Returns:
        (model, scaler, label_encoder_or_None, metadata)
    """
    model = joblib.load(os.path.join(MODELS_DIR, f"{target}_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{target}_scaler.joblib"))

    le = None
    le_path = os.path.join(MODELS_DIR, f"{target}_label_encoder.joblib")
    if os.path.exists(le_path):
        le = joblib.load(le_path)

    with open(os.path.join(MODELS_DIR, f"{target}_metadata.json")) as f:
        metadata = json.load(f)

    return model, scaler, le, metadata


def predict(df: pd.DataFrame, target: str = "habitable") -> pd.DataFrame:
    """Run predictions on a DataFrame of exoplanets.

    Args:
        df: Raw exoplanet DataFrame (same schema as training data).
        target: "habitable" or "planet_type".

    Returns:
        DataFrame with original columns plus prediction and probability columns.
    """
    model, scaler, le, metadata = load_model(target)
    features = metadata["features"]

    df_eng = engineer_features(df)
    available = [c for c in features if c in df_eng.columns]
    X = df_eng[available].copy()
    X = X.fillna(X.median())
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    preds = model.predict(X_scaled)
    result = df.copy()

    if target == "habitable":
        result["habitability_prediction"] = preds
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)
            result["habitability_probability"] = proba[:, 1]
    elif target == "planet_type" and le is not None:
        result["planet_type_prediction"] = le.inverse_transform(preds)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)
            for i, cls in enumerate(le.classes_):
                result[f"prob_{cls}"] = proba[:, i]

    return result


def find_habitable_candidates(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Find the most promising potentially habitable exoplanet candidates.

    Returns the top_n planets ranked by habitability probability.
    """
    result = predict(df, target="habitable")
    candidates = (
        result[result["habitability_probability"] > 0.5]
        .sort_values("habitability_probability", ascending=False)
        .head(top_n)
    )

    display_cols = [
        "pl_name",
        "hostname",
        "habitability_probability",
        "pl_eqt",
        "pl_rade",
        "pl_bmasse",
        "pl_insol",
        "pl_orbper",
        "pl_orbsmax",
        "sy_dist",
    ]
    available_cols = [c for c in display_cols if c in candidates.columns]
    return candidates[available_cols]


def dataset_summary(df: pd.DataFrame) -> dict:
    """Generate a high-level summary of the exoplanet dataset."""
    summary = {
        "total_planets": len(df),
        "discovery_methods": df["discoverymethod"].value_counts().to_dict()
        if "discoverymethod" in df.columns
        else {},
        "year_range": (
            int(df["disc_year"].min()),
            int(df["disc_year"].max()),
        )
        if "disc_year" in df.columns
        else None,
        "radius_stats": df["pl_rade"].describe().to_dict()
        if "pl_rade" in df.columns
        else {},
        "mass_stats": df["pl_bmasse"].describe().to_dict()
        if "pl_bmasse" in df.columns
        else {},
        "temperature_stats": df["pl_eqt"].describe().to_dict()
        if "pl_eqt" in df.columns
        else {},
    }
    return summary
