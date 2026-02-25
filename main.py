"""
Exoplanet Analysis — Main entry point.

Fetches data from NASA Exoplanet Archive, trains ML models, and identifies
potentially habitable exoplanet candidates.

Usage:
    python main.py                  # Full pipeline: fetch, train, predict
    python main.py --skip-fetch     # Use cached data, retrain models
    python main.py --predict-only   # Use cached data + trained models
"""

import argparse
import os
import sys
import pandas as pd

from src.fetch_data import fetch_exoplanet_data
from src.generate_sample_data import generate_sample_data
from src.train import train_and_evaluate
from src.predict import find_habitable_candidates, predict, dataset_summary

DATA_PATH = os.path.join("data", "exoplanets.csv")


def main():
    parser = argparse.ArgumentParser(description="Exoplanet Analysis ML Pipeline")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data download; use existing CSV in data/",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training; load saved models and predict",
    )
    args = parser.parse_args()

    # ---- Step 1: Fetch data ----
    if args.predict_only or args.skip_fetch:
        if not os.path.exists(DATA_PATH):
            print(f"Error: {DATA_PATH} not found. Run without --skip-fetch first.")
            sys.exit(1)
        print(f"Loading cached data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            df = fetch_exoplanet_data(save=True)
        except Exception as e:
            print(f"Could not fetch from NASA API: {e}")
            print("Falling back to synthetic sample data for demonstration.\n")
            df = generate_sample_data(n_planets=5000)

    # ---- Dataset summary ----
    summary = dataset_summary(df)
    print(f"\nDataset Summary:")
    print(f"  Total confirmed exoplanets: {summary['total_planets']}")
    if summary["year_range"]:
        print(f"  Discovery year range: {summary['year_range'][0]} - {summary['year_range'][1]}")
    print(f"  Discovery methods:")
    for method, count in sorted(
        summary["discovery_methods"].items(), key=lambda x: -x[1]
    )[:5]:
        print(f"    {method:30s} {count:>5d}")

    # ---- Step 2: Train models ----
    if not args.predict_only:
        print("\n--- Training Habitability Classifier ---")
        hab_meta = train_and_evaluate(df, target="habitable")

        print("\n--- Training Planet Type Classifier ---")
        type_meta = train_and_evaluate(df, target="planet_type")

    # ---- Step 3: Predict & find candidates ----
    print("\n" + "=" * 60)
    print("  Top Potentially Habitable Exoplanet Candidates")
    print("=" * 60 + "\n")

    candidates = find_habitable_candidates(df, top_n=20)
    if len(candidates) == 0:
        print("No candidates found with habitability probability > 0.5")
    else:
        print(candidates.to_string(index=False))

    # Planet type predictions summary
    print("\n" + "=" * 60)
    print("  Planet Type Distribution (Model Predictions)")
    print("=" * 60 + "\n")

    typed = predict(df, target="planet_type")
    print(typed["planet_type_prediction"].value_counts().to_string())

    print("\nDone! Models and metadata saved in models/ directory.")


if __name__ == "__main__":
    main()
