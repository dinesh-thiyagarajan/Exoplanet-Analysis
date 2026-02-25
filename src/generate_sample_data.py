"""
Generate a realistic synthetic exoplanet dataset for development and testing.

The distributions are modeled after the real NASA Exoplanet Archive data.
This is used as a fallback when the live API is not accessible.

When running locally, use `python main.py` to fetch real data instead.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RNG = np.random.default_rng(42)


def generate_sample_data(n_planets: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic exoplanet data.

    Planet parameters follow approximate distributions observed in the
    real NASA Exoplanet Archive catalog.
    """
    print(f"Generating synthetic dataset with {n_planets} exoplanets...")

    # --- Discovery methods (approximate real proportions) ---
    methods = RNG.choice(
        ["Transit", "Radial Velocity", "Microlensing", "Imaging", "Transit Timing Variations"],
        size=n_planets,
        p=[0.75, 0.17, 0.04, 0.02, 0.02],
    )

    # --- Discovery year ---
    disc_year = RNG.choice(
        range(1995, 2025),
        size=n_planets,
        p=_year_weights(),
    )

    # --- Planet radius (Earth radii) — bimodal: small planets + gas giants ---
    pl_rade = np.concatenate([
        RNG.lognormal(mean=np.log(1.5), sigma=0.5, size=int(n_planets * 0.55)),   # small
        RNG.lognormal(mean=np.log(11.0), sigma=0.3, size=int(n_planets * 0.30)),   # gas giant
        RNG.lognormal(mean=np.log(4.0), sigma=0.4, size=n_planets - int(n_planets * 0.55) - int(n_planets * 0.30)),  # sub-Neptune
    ])
    RNG.shuffle(pl_rade)
    pl_rade = np.clip(pl_rade, 0.3, 30.0)

    # --- Planet mass (Earth masses) — correlated with radius ---
    pl_bmasse = np.where(
        pl_rade < 2.0,
        pl_rade ** 3.5 * RNG.lognormal(0, 0.3, n_planets),      # rocky: M ~ R^3.5
        pl_rade ** 2.0 * RNG.lognormal(0, 0.4, n_planets) * 3,  # gaseous: M ~ R^2
    )
    pl_bmasse = np.clip(pl_bmasse, 0.01, 13000)

    # --- Orbital period (days) ---
    pl_orbper = RNG.lognormal(mean=np.log(20), sigma=1.8, size=n_planets)
    pl_orbper = np.clip(pl_orbper, 0.1, 100000)

    # --- Semi-major axis (AU) — Kepler's 3rd law approximation ---
    st_mass = RNG.normal(loc=1.0, scale=0.3, size=n_planets)
    st_mass = np.clip(st_mass, 0.08, 5.0)
    pl_orbsmax = ((pl_orbper / 365.25) ** 2 * st_mass) ** (1 / 3)

    # --- Orbital eccentricity ---
    pl_orbeccen = RNG.beta(a=1.2, b=5.0, size=n_planets)
    pl_orbeccen = np.clip(pl_orbeccen, 0, 0.99)

    # --- Stellar properties ---
    st_teff = RNG.normal(loc=5500, scale=800, size=n_planets)
    st_teff = np.clip(st_teff, 2500, 12000)

    st_rad = (st_mass ** 0.8) * RNG.normal(1.0, 0.1, n_planets)
    st_rad = np.clip(st_rad, 0.1, 10.0)

    st_lum = np.log10(st_rad ** 2 * (st_teff / 5778) ** 4) + RNG.normal(0, 0.05, n_planets)

    st_logg = np.log10(st_mass / st_rad ** 2 * 27400) + RNG.normal(0, 0.1, n_planets)
    st_logg = np.clip(st_logg, 2.0, 6.0)

    st_met = RNG.normal(loc=0.0, scale=0.25, size=n_planets)
    st_met = np.clip(st_met, -1.0, 0.6)

    # --- Equilibrium temperature (K) ---
    st_lum_linear = 10.0 ** st_lum
    pl_eqt = 278 * (st_lum_linear ** 0.25) / (pl_orbsmax ** 0.5)
    pl_eqt += RNG.normal(0, 20, n_planets)
    pl_eqt = np.clip(pl_eqt, 50, 6000)

    # --- Insolation flux (Earth flux) ---
    pl_insol = st_lum_linear / (pl_orbsmax ** 2)
    pl_insol = np.clip(pl_insol, 0.001, 10000)

    # --- Distance (parsecs) ---
    sy_dist = RNG.lognormal(mean=np.log(500), sigma=1.0, size=n_planets)
    sy_dist = np.clip(sy_dist, 1.0, 10000)

    # --- Host star names ---
    prefixes = ["Kepler-", "TOI-", "K2-", "HD ", "GJ ", "WASP-", "HAT-P-", "TrES-", "CoRoT-", "TRAPPIST-"]
    hostnames = [
        RNG.choice(prefixes) + str(RNG.integers(1, 9999))
        for _ in range(n_planets)
    ]
    pl_names = [h + " b" for h in hostnames]

    # --- Inject missing values (realistic: ~10-30% for some columns) ---
    def add_nans(arr, frac):
        mask = RNG.random(len(arr)) < frac
        arr = arr.astype(float)
        arr[mask] = np.nan
        return arr

    df = pd.DataFrame({
        "pl_name": pl_names,
        "hostname": hostnames,
        "discoverymethod": methods,
        "disc_year": disc_year,
        "pl_orbper": add_nans(pl_orbper, 0.02),
        "pl_orbsmax": add_nans(pl_orbsmax, 0.05),
        "pl_rade": add_nans(pl_rade, 0.08),
        "pl_bmasse": add_nans(pl_bmasse, 0.25),
        "pl_orbeccen": add_nans(pl_orbeccen, 0.15),
        "pl_eqt": add_nans(pl_eqt, 0.12),
        "pl_insol": add_nans(pl_insol, 0.10),
        "st_teff": add_nans(st_teff, 0.03),
        "st_rad": add_nans(st_rad, 0.05),
        "st_mass": add_nans(st_mass, 0.05),
        "st_lum": add_nans(st_lum, 0.15),
        "st_logg": add_nans(st_logg, 0.08),
        "st_met": add_nans(st_met, 0.20),
        "sy_dist": add_nans(sy_dist, 0.03),
        "default_flag": 1,
    })

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "exoplanets.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} synthetic exoplanets to {path}")

    return df


def _year_weights():
    """Approximate discovery-year distribution peaking in recent years."""
    years = list(range(1995, 2025))
    weights = [1] * 5 + [3] * 5 + [8] * 5 + [20] * 5 + [40] * 5 + [60] * 5
    total = sum(weights)
    return [w / total for w in weights]


if __name__ == "__main__":
    generate_sample_data()
