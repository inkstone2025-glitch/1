# data.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd

# ===================================================================
# Country-level reference data — loaded from data/country_data.csv
# Columns: country_name, enroll_mean, enroll_var, max_sites
# ===================================================================
_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "country_data.csv"

_country_df = pd.read_csv(_CSV_PATH)
_country_df["max_sites"] = _country_df["max_sites"].astype(int)

COUNTRY_DATA: dict = _country_df.to_dict(orient="list")

# Pre-built lookup: country_name -> max_sites
MAX_SITES_BY_COUNTRY: dict[str, int] = dict(
    zip(_country_df["country_name"], _country_df["max_sites"])
)


def get_max_sites(country_list: list[str]) -> dict[str, int]:
    """Return {country: max_sites} for the given countries. Raises if a country is unknown."""
    result = {}
    for c in country_list:
        if c not in MAX_SITES_BY_COUNTRY:
            raise ValueError(
                f"Unknown country '{c}'. Available: {list(MAX_SITES_BY_COUNTRY.keys())}"
            )
        result[c] = MAX_SITES_BY_COUNTRY[c]
    return result


def prepare_simulation_data(
    sites_per_country: dict[str, int],
    total_sample_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    country_df = pd.DataFrame(COUNTRY_DATA)

    full_list_of_countries = COUNTRY_DATA['country_name']
    n_countries = len(full_list_of_countries)
    max_sites_per_country = 20

    site_data_template = pd.DataFrame({
        'country_name': [country for country in full_list_of_countries for _ in range(max_sites_per_country)],
        'NoOfSites': list(range(1, max_sites_per_country + 1)) * n_countries,
        'init_lower': [0] * n_countries * max_sites_per_country,
        'init_upper': [182] * n_countries * max_sites_per_country,
        'enroll_lower': [None] * n_countries * max_sites_per_country,
        'enroll_upper': [None] * n_countries * max_sites_per_country,
        'enroll_symmetric': [False] * n_countries * max_sites_per_country,
        'origin': [0] * n_countries * max_sites_per_country,
        'origin_shift': [0] * n_countries * max_sites_per_country,
        'init_elapsed': [None] * n_countries * max_sites_per_country,
        'active_days': [None] * n_countries * max_sites_per_country,
        'active_num': [None] * n_countries * max_sites_per_country,
        'discontinued': [False] * n_countries * max_sites_per_country,
    })

    # Filter per-country using each country's site count
    list_of_countries = list(sites_per_country.keys())
    frames = []
    for country, n_sites in sites_per_country.items():
        mask = (
            (site_data_template['country_name'] == country) &
            (site_data_template['NoOfSites'] <= n_sites)
        )
        frames.append(site_data_template[mask])
    site_data = pd.concat(frames, ignore_index=True).copy()

    site_data['site_name'] = site_data['country_name'] + '-' + site_data['NoOfSites'].astype(str)
    site_data = site_data.merge(country_df, on=['country_name'], how='left')

    country_targets = pd.DataFrame({
        'country_name': list_of_countries,
        'target': [0] * len(list_of_countries),
        'over_enrollment_target': [1e+06] * len(list_of_countries),
        'region': ['row'] * len(list_of_countries),
        'desired_target': [0] * len(list_of_countries),
        'region_target': [0] * len(list_of_countries),
    })
    # Allocate total target_N to countries based on their enroll_mean
    if not country_targets.empty:
        means = (
            country_df.set_index("country_name")
            .loc[list_of_countries, "enroll_mean"]
            .astype(float)
            .to_numpy()
        )

        # Convert to weights (fallback to uniform if something weird happens)
        means = np.clip(means,0.0,None)
        if means.sum() <=0:
            weights = np.ones_like(means) / len(means)
        else:
            weights = means / means.sum()

        # Allocate integer desired_target so sum = total_sample_size
        raw = weights * float(total_sample_size)
        base = np.floor(raw).astype(float)
        remainder = int(total_sample_size) - int(base.sum())

        # Give leftover subjects to countries with largest fractional parts
        frac = raw - base
        if remainder > 0:
            idx = np.argsort(-frac)[:remainder]
            base[idx] +=1

        country_targets["desired_target"] = base.tolist()

    return site_data, country_targets
