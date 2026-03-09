# data.py

from __future__ import annotations
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd

# ===================================================================
# Country-level reference data (enrollment rates + max available sites)
# ===================================================================
COUNTRY_DATA = {
    'country_name': ["Argentina", "Austria", "Australia", "Belgium", "Bulgaria", "Chile", "China",
                     "Czech Republic", "Estonia", "Spain", "Finland", "United Kingdom", "Greece",
                     "Hong Kong", "Croatia", "Hungary", "India", "Italy", "South Korea", "Mexico",
                     "Malaysia", "Netherlands", "New Zealand", "Peru", "Philippines", "Pakistan",
                     "Poland", "Portugal", "Russia", "Sweden", "Slovenia", "Slovakia", "United States",
                     "South Africa"],
    'enroll_mean': [0.119772907292732, 0.192334329301287, 0.0618951894087672, 0.0286274875512264,
                    0.119838446575526, 0.0806564602327388, 0.0791928399316703, 0.0473969393018412,
                    0.222907907834042, 0.140656438344257, 0.228451851974456, 0.240697308137417,
                    0.0446349424012019, 0.0341052241463141, 0.0287313432835821, 0.223994825296515,
                    0.815384615384615, 0.0864162712152835, 0.119278535368533, 0.292446636045215,
                    0.0253909884412825, 0.218052663079997, 0.348039215686274, 0.549295774647887,
                    0.031464518705898, 0.286351674641148, 0.156966230122541, 0.0371447028423773,
                    0.105837056590878, 0.0583303664614245, 0.0355189241427243, 0.128137225097023,
                    0.261467907927595, 0.0867855922595078],
    'enroll_var': [0.0567082762702618, 0.111002141963495, 0.0313009416199871, 6.57009731286169e-05,
                   0.0352066820442973, 0.0147741339401526, 0.000860370177161715, 0.00311255898353705,
                   0.148589410569972, 0.251672584664282, 0.186371171190654, 0.174378365461528,
                   0.000367013470584561, 0.000171997099524204, 0.00230693547189426, 0.084195557331705,
                   0.170414201183432, 0.0329968169373322, 0.0656415681621831, 0.332556808949861,
                   0.000360250892786568, 0.150428177274775, 0.203046905036524, 0.406268597500496,
                   0.00022036023023808, 0.228234740314248, 0.0889324754210265, 0.000175478570331644,
                   0.0106998573962253, 0.00836066031579909, 0.000209162555319519, 0.0838120196358846,
                   0.223706111169478, 0.046715154428215],
    # Max available sites per country (placeholder values — replace with real data)
    'max_sites': [12, 8, 10, 6, 7, 9, 15, 5, 6, 11, 7, 14, 5, 4, 4, 8, 20, 10, 9, 12,
                  5, 8, 6, 8, 6, 10, 11, 5, 13, 6, 4, 7, 18, 8],
}

# Pre-built lookup: country_name -> max_sites
MAX_SITES_BY_COUNTRY: dict[str, int] = dict(
    zip(COUNTRY_DATA['country_name'], COUNTRY_DATA['max_sites'])
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

    full_list_of_countries = data['country_name']
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
