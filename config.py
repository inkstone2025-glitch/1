# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import os
import requests

# ===================================================================
# Databricks API config
# ===================================================================

DATABRICKS_HOST  = os.environ.get(
    "DATABRICKS_HOST",
    "https://adb-1970865590743686.6.azuredatabricks.net",
).rstrip("/")

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

SURVIVAL_EP = os.environ.get("SURVIVAL_EP", "Survival___API")
POWER_EP    = os.environ.get("POWER_EP", "power_calculation_api")
RECRUIT_EP  = os.environ.get("RECRUIT_EP", "intelhub_recruitment_api")
COST_EP     = os.environ.get("COST_EP", "Costing_model_api")

DEFAULT_CONNECT_TIMEOUT = int(os.environ.get("DEFAULT_CONNECT_TIMEOUT", "5"))
DEFAULT_READ_TIMEOUT    = int(os.environ.get("DEFAULT_READ_TIMEOUT", "120"))
MAX_RETRIES             = int(os.environ.get("MAX_RETRIES", "4"))
BACKOFF_BASE_S          = float(os.environ.get("BACKOFF_BASE_S", "0.6"))

session = requests.Session()
if DATABRICKS_TOKEN:
    session.headers.update({
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type":  "application/json",
    })
session.trust_env = True

# ===================================================================
# User defined parameters
# ===================================================================

# 0. Cost model parameters
COST_BASE = {
    "t_a": "Neurosciences",                  # fixed variable, user defined
    "indication": "Alzheimer Disease",       # fixed variable, user defined
    "phase": "PHASE III",                    # fixed variable, user defined
    "product_type": "Pharma",                # fixed variable, user defined
    "n_visits": 10,                          # fixed variable, user defined
    "imaging": "No",                         # fixed variable, user defined
    "adhoc_study_cost_reason": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",    # random text for now to satisfy the input requirement.
    "precision_medicine_per_patient" : 0
}


# 1. Survival parameters (AD)
SURV_BASE = {
    "marg_probs": {"age": 0.4, "biomarker": 0.5, "time_diag": 0.4},
    "init_prob": 0.8,
    "beta": {"age": 0.05, "biomarker": 0.08, "time_diag": 0.06},
    "intercepts": [-2.64, -9.89],
    "kappa": [0.44, 0.068],
    "tau": 8.30,
    "t0": 1.0,
    "surv_treat_at_t0": 0.5,
    "trial_dur": 2.0,
}


# 2. User-defined country list for this trial
# Names must match the country data in data.py
# Max available sites per country are defined in data.py (COUNTRY_DATA)
SELECTED_COUNTRIES: list[str] = [
    "United States",
    "India",
    "United Kingdom",
    "Peru",
    "Mexico",
    "Pakistan",
    "Hungary",
    "Estonia",
]


# 3. BO setting
@dataclass
class BOConfig:
    seed: int = 123             # Random seed for the entire BO run.
    N_INIT: int = 25            # Number of inital random design evaluated before BO starts. Increased for higher-dim space. 15-30
    N_ITERS: int = 100          # Number of BO iterations after initial design. Each one evaluates Q_BATCH new designs.
    Q_BATCH: int = 2            # Number of candidates designs evaluated per BO interation. =1: sequential BO (sample and stable); >1 : parallel BO (fast wall-clock, complex)
    RAW_SAMP: int = 512         # Number of random samples used to seed acuisition optimization. Larger = better global exploration but more computation. 128-1024
    RESTARTS: int = 20          # Number of local optimization restarts when maximizing the acquisition. Larger = reduce risk of getting stuck in local optima. 5-20
    NUM_MC_SAMPLES: int = 128   # Number of Monte Carlo samples used to estimate qNEHVI. Larger = more accurate hypervolume estiamtes. 64-256