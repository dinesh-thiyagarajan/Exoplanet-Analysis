"""
Fetch exoplanet data from the NASA Exoplanet Archive using the TAP API.

Downloads confirmed exoplanet data including orbital parameters, stellar
properties, and discovery metadata.
"""

import os
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# NASA Exoplanet Archive TAP API endpoint
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns to fetch from the Planetary Systems (ps) table
COLUMNS = [
    "pl_name",           # Planet name
    "hostname",          # Host star name
    "discoverymethod",   # Discovery method
    "disc_year",         # Discovery year
    "pl_orbper",         # Orbital period (days)
    "pl_orbsmax",        # Semi-major axis (AU)
    "pl_rade",           # Planet radius (Earth radii)
    "pl_bmasse",         # Planet mass (Earth masses)
    "pl_orbeccen",       # Orbital eccentricity
    "pl_eqt",            # Equilibrium temperature (K)
    "pl_insol",          # Insolation flux (Earth flux)
    "st_teff",           # Stellar effective temperature (K)
    "st_rad",            # Stellar radius (Solar radii)
    "st_mass",           # Stellar mass (Solar masses)
    "st_lum",            # Stellar luminosity (log Solar)
    "st_logg",           # Stellar surface gravity (log cm/s^2)
    "st_met",            # Stellar metallicity (dex)
    "sy_dist",           # Distance (parsecs)
    "default_flag",      # 1 = default parameter set for this planet
]

QUERY = f"""
SELECT {', '.join(COLUMNS)}
FROM ps
WHERE default_flag = 1
"""


def fetch_exoplanet_data(save: bool = True) -> pd.DataFrame:
    """Download confirmed exoplanet data from NASA Exoplanet Archive.

    Args:
        save: If True, save the data as a CSV in the data/ directory.

    Returns:
        DataFrame with one row per confirmed exoplanet (default parameter set).
    """
    print("Fetching data from NASA Exoplanet Archive...")
    params = {
        "query": QUERY.strip(),
        "format": "csv",
    }
    response = requests.get(TAP_URL, params=params, timeout=120)
    response.raise_for_status()

    # Parse CSV response
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(df)} confirmed exoplanets with {len(df.columns)} columns.")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "exoplanets.csv")
        df.to_csv(path, index=False)
        print(f"Saved to {path}")

    return df


if __name__ == "__main__":
    fetch_exoplanet_data()
