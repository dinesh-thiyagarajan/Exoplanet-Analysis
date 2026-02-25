"""
Data preprocessing and feature engineering for exoplanet ML models.

Handles missing values, creates derived features, and prepares data
for training classification and regression models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --- Habitability label heuristic ---------------------------------------------------
# We define a simple habitability score based on well-known criteria:
#   1. Equilibrium temperature between 180 K and 310 K  (liquid water range)
#   2. Planet radius between 0.5 and 2.5 Earth radii    (rocky planet range)
#   3. Insolation flux between 0.2 and 2.0 Earth flux   (temperate zone)
#
# Planets meeting ALL three criteria are labeled "potentially habitable".
# This is a simplified heuristic for ML training purposes, not a scientific claim.


def label_habitability(df: pd.DataFrame) -> pd.Series:
    """Create a binary habitability label based on physical criteria."""
    temp_ok = df["pl_eqt"].between(180, 310)
    radius_ok = df["pl_rade"].between(0.5, 2.5)
    insol_ok = df["pl_insol"].between(0.2, 2.0)

    label = (temp_ok & radius_ok & insol_ok).astype(int)
    return label


# --- Planet type classification -----------------------------------------------------

def classify_planet_type(df: pd.DataFrame) -> pd.Series:
    """Classify planets into broad categories by radius (Earth radii)."""
    conditions = [
        df["pl_rade"] < 1.25,
        df["pl_rade"].between(1.25, 2.0),
        df["pl_rade"].between(2.0, 6.0),
        df["pl_rade"].between(6.0, 15.0),
        df["pl_rade"] >= 15.0,
    ]
    choices = ["Rocky", "Super-Earth", "Sub-Neptune", "Neptune-like", "Gas Giant"]
    return pd.Series(
        np.select(conditions, choices, default="Unknown"),
        index=df.index,
        name="planet_type",
    )


# --- Feature engineering ------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features useful for ML models."""
    out = df.copy()

    # Stellar luminosity from log to linear (if available)
    if "st_lum" in out.columns:
        out["st_lum_linear"] = 10.0 ** out["st_lum"]

    # Ratio features
    out["radius_mass_ratio"] = out["pl_rade"] / out["pl_bmasse"].replace(0, np.nan)
    out["planet_star_radius_ratio"] = out["pl_rade"] / (
        out["st_rad"].replace(0, np.nan) * 109.076  # Solar radii -> Earth radii
    )
    out["flux_temp_ratio"] = out["pl_insol"] / out["pl_eqt"].replace(0, np.nan)

    # Log-transformed features (useful for skewed distributions)
    for col in ["pl_orbper", "pl_bmasse", "pl_rade", "sy_dist"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    return out


# --- Full preprocessing pipeline ----------------------------------------------------

NUMERIC_FEATURES = [
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_bmasse",
    "pl_orbeccen",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_lum",
    "st_logg",
    "st_met",
    "sy_dist",
    # Engineered features
    "st_lum_linear",
    "radius_mass_ratio",
    "planet_star_radius_ratio",
    "flux_temp_ratio",
    "log_pl_orbper",
    "log_pl_bmasse",
    "log_pl_rade",
    "log_sy_dist",
]


def preprocess(
    df: pd.DataFrame,
    target: str = "habitable",
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, LabelEncoder | None]:
    """Full preprocessing: feature engineering, encoding, imputation, scaling.

    Args:
        df: Raw exoplanet DataFrame.
        target: Which target column to produce.
                "habitable" -> binary habitability label
                "planet_type" -> multi-class planet type
        scaler: Pre-fitted scaler (for inference). If None, a new one is fit.

    Returns:
        X: Feature matrix (scaled).
        y: Target labels.
        scaler: Fitted StandardScaler.
        le: Fitted LabelEncoder (only for planet_type target, else None).
    """
    df = engineer_features(df)

    # Create target
    le = None
    if target == "habitable":
        y = label_habitability(df)
    elif target == "planet_type":
        y_raw = classify_planet_type(df)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=df.index, name="planet_type")
    else:
        raise ValueError(f"Unknown target: {target}")

    # Select numeric features that exist in this DataFrame
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[available].copy()

    # Drop rows where the target itself has missing inputs
    if target == "habitable":
        key_cols = ["pl_eqt", "pl_rade", "pl_insol"]
        valid_mask = df[key_cols].notna().all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

    # Impute remaining missing values with column medians
    X = X.fillna(X.median())

    # Scale
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )
    else:
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )

    return X_scaled, y, scaler, le
