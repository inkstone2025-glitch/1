#pipeline.py

from __future__ import annotations

import math
import json
from typing import Any, Dict, List, Tuple, TypedDict

import requests
import pandas as pd
from langgraph.graph import StateGraph, END

from .api import (
    call_endpoint,
    payload_survival,
    parse_survival,
    payload_power,
    parse_power,
    build_N_list,
)

from .config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    SURVIVAL_EP,
    POWER_EP,
    RECRUIT_EP,
)

from .data import prepare_simulation_data

# ===================================================================
# 5. Recruitment API wrapper + data prep
# ===================================================================
def invoke_recruitment_api_local(
    site_data: pd.DataFrame,
    country_targets: pd.DataFrame,
    *,
    endpoint: str,
    host: str = None,
    token: str = None,
    num_sim: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    host  = (host or DATABRICKS_HOST).rstrip("/")
    token = token or DATABRICKS_TOKEN
    if not token:
        raise ValueError("DATABRICKS_TOKEN is required.")

    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # dataframes -> JSON
    site_json_str    = site_data.to_json(orient="split", index=False)
    country_json_str = country_targets.to_json(orient="split", index=False)

    def json_to_python_literal(s: str) -> str:
        return s.replace("null", "None").replace("true", "True").replace("false", "False")

    site_literal    = json_to_python_literal(site_json_str)
    country_literal = json_to_python_literal(country_json_str)

    payload_dict = {
        "inputs": {
            "site_data":       site_literal,
            "country_targets": country_literal,
        },
        "params": {
            "num_sim": int(num_sim),
        },
    }
    data_json = json.dumps(payload_dict)

    r = requests.post(url, headers=headers, data=data_json, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(
            f"Recruitment {endpoint} -> {r.status_code}: {r.text[:1500]}\n"
            f"Payload sent: {data_json[:800]}"
        )

    result = r.json()
    pred = result.get("predictions", {})
    study   = pred.get("study", {})
    country = pred.get("country", {})

    projections_df = pd.DataFrame(data=study.get("data", []),   columns=study.get("columns", []))
    country_df     = pd.DataFrame(data=country.get("data", []), columns=country.get("columns", []))
    return projections_df, country_df, result


# ===================================================================
# 6. LangGraph state and nodes
# ===================================================================
class PipeState(TypedDict, total=False):
    # user inputs
    survival_params: Dict[str, Any]
    recruit_sites_per_country: Dict[str, int]  # e.g. {"USA": 5, "India": 12}
    target_N: int  # BO-chosen sample size
    power_N: List[int]

    # survival outputs
    surv_df: pd.DataFrame
    surv_scalars: Dict[str, Any]

    # power outputs
    power_df: pd.DataFrame

    # recruitment inputs
    site_data: pd.DataFrame
    country_targets: pd.DataFrame

    # recruitment outputs
    recruit_proj_df: pd.DataFrame
    recruit_country_df: pd.DataFrame
    recruit_raw: Dict[str, Any]

    # summary
    summary: Dict[str, Any]


def survival_node(state: PipeState) -> PipeState:
    resp = call_endpoint(SURVIVAL_EP, payload_survival(state["survival_params"]), read_timeout=180)
    out  = parse_survival(resp)
    return out  # surv_df, surv_scalars


def power_node(state: PipeState) -> PipeState:
    scal = state["surv_scalars"]
    HR   = scal["HR"]
    pev  = scal["pev_contr_at_final"]

    if "target_N" in state:
        center_N = int(state["target_N"])
        N_list = build_N_list(center_N)
    else:
        center_N = int(math.ceil(float(scal["total_n_at_90_pos"])))
        N_list = [center_N]

    resp = call_endpoint(POWER_EP, payload_power(N_list, HR, pev), read_timeout=60)
    pdf  = parse_power(resp, N_list)
    return {"power_df": pdf, "power_N": N_list}


def prepare_node(state: PipeState) -> PipeState:
    # Use BO-chosen target_N if present; otherwise use survival-based N
    if "target_N" in state:
        total_sample_size = int(state["target_N"])
    else:
        total_sample_size = int(math.ceil(float(state["surv_scalars"]["total_n_at_90_pos"])))

    site_data, country_targets = prepare_simulation_data(
        sites_per_country = state["recruit_sites_per_country"],
        total_sample_size = total_sample_size,
    )
    return {"site_data": site_data, "country_targets": country_targets}


def recruit_node(state: PipeState) -> PipeState:
    proj_df, country_df, raw = invoke_recruitment_api_local(
        site_data=state["site_data"],
        country_targets=state["country_targets"],
        endpoint=RECRUIT_EP,
        num_sim=50,
    )
    return {"recruit_proj_df": proj_df, "recruit_country_df": country_df, "recruit_raw": raw}


def gather_node(state: PipeState) -> PipeState:
    s = state["surv_scalars"]
    summary = {
        "N_total_90pos": int(math.ceil(float(s["total_n_at_90_pos"]))),
        "HR":            float(s["HR"]),
        "pev_final":     float(s["pev_contr_at_final"]),
        "median_surv":   s.get("median_survival_time"),
        "power_rows":    len(state.get("power_df", [])),
        "recruit_rows":  len(state.get("recruit_proj_df", [])),
        "target_N":      state.get("target_N"),
    }
    return {"summary": summary}


def build_graph():
    g = StateGraph(PipeState)
    g.add_node("survival", survival_node)
    g.add_node("power",    power_node)
    g.add_node("prepare",  prepare_node)
    g.add_node("recruit",  recruit_node)
    g.add_node("gather",   gather_node)

    g.set_entry_point("survival")
    g.add_edge("survival", "power")
    g.add_edge("survival", "prepare")
    g.add_edge("prepare",  "recruit")
    g.add_edge("power",    "gather")
    g.add_edge("recruit",  "gather")
    g.add_edge("gather",   END)
    return g.compile()