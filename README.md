# Exoplanet Analysis — ML Pipeline

Machine learning pipeline that analyzes exoplanet data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). Trains classification models for **habitability prediction** and **planet type categorization**.

## Features

- **Data fetching** from the NASA Exoplanet Archive TAP API (5,000+ confirmed exoplanets)
- **Feature engineering** — derived features like radius/mass ratio, log transforms, flux ratios
- **Habitability classifier** — binary classification predicting whether a planet is potentially habitable based on equilibrium temperature, radius, and insolation flux
- **Planet type classifier** — multi-class classification into Rocky, Super-Earth, Sub-Neptune, Neptune-like, and Gas Giant
- **Model selection** — compares Random Forest, Gradient Boosting, and Logistic Regression via cross-validation; saves the best model
- **Candidate ranking** — ranks exoplanets by habitability probability

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
# Full pipeline: fetch data → train models → predict
python main.py

# Skip data download, use cached CSV
python main.py --skip-fetch

# Load already-trained models, just run predictions
python main.py --predict-only
```

## Project Structure

```
├── main.py                     # Entry point — runs full pipeline
├── requirements.txt            # Python dependencies
├── src/
│   ├── fetch_data.py           # Downloads data from NASA Exoplanet Archive
│   ├── generate_sample_data.py # Synthetic data fallback for offline use
│   ├── preprocess.py           # Feature engineering & preprocessing
│   ├── train.py                # Model training & evaluation
│   └── predict.py              # Prediction & candidate ranking
├── data/                       # Downloaded/generated CSV data (git-ignored)
└── models/                     # Trained model artifacts (git-ignored)
```

## Models

| Task | Best Model | Test F1 | Features |
|------|-----------|---------|----------|
| Habitability | Gradient Boosting | 0.94 | 22 |
| Planet Type | Random Forest | 0.99 | 22 |

Top features for habitability: equilibrium temperature, planet radius, insolation flux.

## Habitability Criteria

A planet is labeled "potentially habitable" if it meets all three conditions:
1. Equilibrium temperature: 180–310 K (liquid water range)
2. Planet radius: 0.5–2.5 Earth radii (rocky planet range)
3. Insolation flux: 0.2–2.0 Earth flux (temperate zone)

## Data Source

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Planetary Systems (PS) table, default parameter sets for confirmed exoplanets.
