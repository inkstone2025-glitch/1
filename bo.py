# bo.py
from __future__ import annotations

import random
import warnings
from typing import Any, Dict, List, Tuple
from IPython.display import display
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.exceptions.warnings import InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from .api import extract_power_at, call_endpoint, payload_cost, parse_cost
from .pipeline import build_graph
from .config import SURV_BASE, COST_BASE, COST_EP, MUST_INCLUDE_COUNTRIES, OPTIONAL_COUNTRIES
from .data import get_max_sites


# ===================================================================
# 9. Objective helpers
# ===================================================================
def get_time_to_target(final: Dict[str, Any], N_target: int) -> float:
    """Time (days) to reach N_target using median projection."""
    try:
        proj_df = final["recruit_proj_df"]
        TIME_COL = "time"
        ENROLLED_COL = "proj_enroll_0.5"
        if not all(col in proj_df.columns for col in [TIME_COL, ENROLLED_COL]):
            raise KeyError(f"Missing columns {TIME_COL}/{ENROLLED_COL} in recruit_proj_df.")

        successful = proj_df[proj_df[ENROLLED_COL] >= N_target]
        if successful.empty:
            return float(proj_df[TIME_COL].max() + 1000.0)
        return float(successful[TIME_COL].iloc[0])
    except Exception as e:
        print(f"[WARN] get_time_to_target failed: {e}")
        return float(1e9)


# ===================================================================
# 13. Multi-objective GP model
# ===================================================================
def fit_mo_model(train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor) -> ModelListGP:
    '''
    Fit one GP per objective (ModelListGP), with input/output normalization.
    bounds: tensor of size (2, d) just like BO_BOUNDS
    '''
    d = train_X.shape[-1]
    m_list = []
    for i in range(train_Y.shape[-1]):
        # Input [-, :] for all the lowest/highest values per dimension (from your bounds)
        model = SingleTaskGP(
            train_X,
            train_Y[..., i:i+1],
            input_transform=Normalize(d, bounds=bounds),
            outcome_transform=Standardize(m=1)
        )

        # Force minimum noise level for numerical stability (important for hypervolume improvement)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        m_list.append(model)
    return ModelListGP(*m_list)


def _df_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    return {"columns": list(df.columns), "data": df.to_numpy().tolist()}


def run_bo(
    *,
    app,
    SURV_BASE: Dict[str, Any],
    seed,
    N_INIT,
    N_ITERS,
    Q_BATCH,
    RAW_SAMP,
    RESTARTS,
    NUM_MC_SAMPLES,
    output_log_path: str | Path,
    resume_from_file: str | Path | None,
):
    """
    Runs BO and returns:
      - bo_log_df (with stored curve payloads)
      - train_X, train_Y (torch)
    """

    # Ensure paths are path objects
    output_log_path = Path(output_log_path)
    if resume_from_file:
        resume_from_file = Path(resume_from_file)

    tkwargs = {"dtype": torch.double, "device": "cpu"}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    decisions_all=[]
    finals_all = []
    bo_records = []

    # Two-tier country setup
    must_list     = list(MUST_INCLUDE_COUNTRIES)
    optional_list = list(OPTIONAL_COUNTRIES)
    all_countries = must_list + optional_list
    country_max_sites = get_max_sites(all_countries)
    n_must     = len(must_list)
    n_optional = len(optional_list)

    # Build app & get baseline N from survival (for target_N bounds)
    baseline_output = app.invoke({
        "survival_params": SURV_BASE,
        "recruit_sites_per_country": {must_list[0]: 1},
        "target_N": 1000,
    })
    baseline_N90 = baseline_output["summary"]["N_total_90pos"]

    # Decision variable ranges:
    #   must-include: [1, max_sites_i]  (always present)
    #   optional:     [0, max_sites_i]  (0 = exclude, ≥1 = include)
    #   target_N:     [N_min, N_max]
    N_TARGET_MIN = max(50, int(0.5 * baseline_N90))
    N_TARGET_MAX = int(1.5 * baseline_N90)

    lower_bounds = (
        [1.0] * n_must
        + [0.0] * n_optional
        + [float(N_TARGET_MIN)]
    )
    upper_bounds = (
        [float(country_max_sites[c]) for c in must_list]
        + [float(country_max_sites[c]) for c in optional_list]
        + [float(N_TARGET_MAX)]
    )

    BO_BOUNDS = torch.tensor(
        [lower_bounds, upper_bounds],
        **tkwargs,
    )

    def x_to_decisions(*, x_row: torch.Tensor) -> dict:
        sites_per_country = {}

        # Must-include countries: clamp to [1, max_sites]
        for i, country in enumerate(must_list):
            n = int(round(float(x_row[..., i].item())))
            sites_per_country[country] = max(1, min(country_max_sites[country], n))

        # Optional countries: 0 = exclude, ≥1 = include
        optional_selected = []
        for j, country in enumerate(optional_list):
            n = int(round(float(x_row[..., n_must + j].item())))
            n = max(0, min(country_max_sites[country], n))
            if n >= 1:
                sites_per_country[country] = n
                optional_selected.append(country)

        # Last dimension is target_N
        target_N = int(round(float(x_row[..., n_must + n_optional].item())))
        target_N = max(N_TARGET_MIN, min(N_TARGET_MAX, target_N))

        total_sites = sum(sites_per_country.values())

        return {
            "sites_per_country": sites_per_country,
            "countries": list(sites_per_country.keys()),
            "optional_countries_selected": optional_selected,
            "target_N": target_N,
            "total_sites": total_sites,
        }

    def evaluate_design(
        *,
        app,
        x_row: torch.Tensor,
        surv_params: Dict[str, Any],
        tkwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate a single candidate x_row.

        Returns:
        y        : torch.Tensor shape (3,) = [-duration_days, -cost_scaled, power]
        decisions: dict (includes sites_per_country, target_N, etc.)
        final    : dict returned by app.invoke
        """

        # 1) Decode candidate -> trial design decisions
        decisions = x_to_decisions(x_row=x_row)

        # 2) Run pipeline
        final = app.invoke(
            {
                "survival_params":          surv_params,
                "recruit_sites_per_country": decisions["sites_per_country"],
                "target_N":                 int(decisions["target_N"]),
            }
        )

        # 3) Compute objectives
        duration_days = float(get_time_to_target(final, int(decisions["target_N"])))
        duration_months = duration_days / 30.44

        # Prepare cost endpoint payload and call endpoint
        cost_payload = payload_cost(
            n_subjects=int(decisions["target_N"]),
            n_sites=int(decisions["total_sites"]),
            duration_months=duration_months,
            context=COST_BASE,
        )
        cost_resp = call_endpoint(COST_EP, cost_payload)
        total_cost = parse_cost(cost_resp)

        power_val = float(extract_power_at(int(decisions["target_N"]), final["power_df"]))

        # 4) Convert to maximization space
        cost_scaled = total_cost / 1e6
        y = torch.tensor([-duration_days, -cost_scaled, power_val], **tkwargs)

        return y, decisions, final

    def evaluate_batch(
        *,
        app,
        X: torch.Tensor,
        surv_params: Dict[str, Any],
        tkwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluate a batch of candidates and return:
        - Y tensor (N, 3)
        - decisions_all (N dicts)
        - finals_all (N dicts)
        """
        ys: List[torch.Tensor] = []
        decisions_all: List[Dict[str, Any]] = []
        finals_all: List[Dict[str, Any]] = []

        q = X.shape[0]
        for j in range(q):
            y, decisions, final = evaluate_design(
                app=app,
                x_row=X[j : j + 1],
                surv_params=surv_params,
                tkwargs=tkwargs,
            )

            ys.append(y)
            decisions_all.append(decisions)
            finals_all.append(final)

        return torch.stack(ys, dim=0), decisions_all, finals_all

    def generate_initial_design(n_pts=10) -> torch.Tensor:
        xs = []
        for _ in range(n_pts):
            row = []
            for country in must_list:
                row.append(random.uniform(1.0, float(country_max_sites[country])))
            for country in optional_list:
                row.append(random.uniform(0.0, float(country_max_sites[country])))
            row.append(random.uniform(float(N_TARGET_MIN), float(N_TARGET_MAX)))
            xs.append(row)
        return torch.tensor(xs, **tkwargs)

    bo_records: List[Dict[str, Any]] = []

    def _record_eval(
        *,
        eval_id: int,
        phase: str,
        x_row: torch.Tensor,
        y_row: torch.Tensor,
        decisions: dict,
        final: dict,
    ) -> None:
        """
        Append one evaluation row into bo_records.
        """
        y_row_flat = y_row.view(-1)

        duration_days = float(-y_row_flat[0].item())
        cost_total    = float(-y_row_flat[1].item() * 1e6)
        power         = float(y_row_flat[2].item())
        # ---- Per country target allocation
        alloc_df = final["country_targets"][["country_name","desired_target"]].copy()
        alloc_df["desired_target"] = alloc_df["desired_target"].astype(int)

        target_allocation = dict(
            zip(alloc_df["country_name"], alloc_df["desired_target"])
        )

        bo_records.append(
            {
                "eval_id": int(eval_id),
                "phase": phase,
                # Raw tensors for resuming
                "x_row": x_row.cpu().numpy().tolist(),
                "y_row": y_row.cpu().numpy().tolist(),
                # decisions
                "sites_per_country": dict(decisions["sites_per_country"]),
                "countries": list(decisions["countries"]),
                "optional_countries_selected": list(decisions["optional_countries_selected"]),
                "num_countries": len(decisions["countries"]),
                "n_sites_total": int(decisions["total_sites"]),
                "target_N": int(decisions["target_N"]),
                "target_allocation": target_allocation,

                # objectives
                "duration_days": duration_days,
                "cost_total": cost_total,
                "power": power,

                # curves
                "recruit_proj_payload": _df_to_payload(final["recruit_proj_df"]),
                "surv_payload": _df_to_payload(final["surv_df"]),
                "power_payload": _df_to_payload(final["power_df"]),
            }
        )

    # Resume logic
    start_fresh = True
    if resume_from_file and os.path.isfile(resume_from_file):
        print(f"Resuming from file: {resume_from_file}")
        try:
            old_df = pd.read_pickle(resume_from_file)

            # Validate required columns
            if "x_row" in old_df.columns and "y_row" in old_df.columns:
                # Check dimensionality compatibility
                sample_x = np.array(old_df["x_row"].iloc[0])
                expected_dim = BO_BOUNDS.shape[1]
                if len(sample_x) != expected_dim:
                    print(f"Dimension mismatch: saved={len(sample_x)}, expected={expected_dim}. Starting fresh.")
                else:
                    # restore the global record list
                    bo_records.clear()
                    bo_records.extend(old_df.to_dict(orient="records"))

                    # resume train_X and train_Y from the file
                    train_X = torch.tensor(np.stack(old_df["x_row"].tolist()), **tkwargs)
                    train_Y = torch.tensor(np.stack(old_df["y_row"].tolist()), **tkwargs)

                    print(f"Loaded {len(train_X)} previous designs.")

                    warnings.filterwarnings("ignore", category=InputDataWarning)
                    model = fit_mo_model(train_X, train_Y, bounds=BO_BOUNDS)
                    warnings.filterwarnings("default", category=InputDataWarning)

                    start_fresh = False
            else:
                print(f"File {resume_from_file} is missing required columns. Starting fresh.")
        except Exception as e:
            print(f"Failed to load from {resume_from_file}: {e}. Starting fresh.")

    # fresh start: generate initial design and evaluate
    if start_fresh:
        print("\n=== Initial random design ===")
        warnings.filterwarnings("ignore", category=InputDataWarning)
        train_X = generate_initial_design(N_INIT)
        train_Y, decisions_all, finals_all = evaluate_batch(
            app=app,
            X=train_X,
            surv_params=SURV_BASE,
            tkwargs=tkwargs,
        )

        for i in range(train_X.shape[0]):
            _record_eval(eval_id=i,
                        phase="init",
                        x_row=train_X[i],
                        y_row=train_Y[i],
                        decisions=decisions_all[i],
                        final=finals_all[i],
                        )

    model = fit_mo_model(train_X, train_Y, bounds=BO_BOUNDS)
    warnings.filterwarnings("default", category=InputDataWarning)

    # Adaptive reference point: worst observed value minus 10% of each objective's range.
    # This guarantees every observed point dominates the reference point.
    y_min = train_Y.min(dim=0).values
    y_range = train_Y.max(dim=0).values - y_min
    y_range = torch.clamp(y_range, min=1e-6)  # avoid zero range
    ref_point = y_min - 0.1 * y_range

    bo_log_df = pd.DataFrame(bo_records)

    # --- Resume logic for BO loop
    n_bo_evals = sum(1 for record in bo_records if record["phase"] == "bo")
    n_batches_done = n_bo_evals // Q_BATCH
    n_remining_batches = N_ITERS - n_batches_done

    print(f"\n=== BO Status ===")
    print(f"Total BO iterations planned: {N_ITERS}")
    print(f"BO iterations completed: {n_batches_done}")
    print(f"BO iterations remaining: {n_remining_batches}")
    print(f"Decision space: {n_must} must-include + {n_optional} optional + target_N = {n_must + n_optional + 1}D")

    if n_remining_batches <= 0:
        print("No BO iterations left to run. Exiting.")
        return pd.DataFrame(bo_records), train_X, train_Y
    print(f"Running BO for {n_remining_batches} more iterations...\n")

    # --- BO loop
    for it in range(n_remining_batches):
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_MC_SAMPLES]))

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            sampler=sampler,
        )

        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=BO_BOUNDS,
            q=Q_BATCH,
            num_restarts=RESTARTS,
            raw_samples=RAW_SAMP,
        )

        eval_id_start = len(bo_records)

        # Evaluate new candidate(s)
        y_new, decisions_new, finals_new = evaluate_batch(
            app=app,
            X=candidates,
            surv_params=SURV_BASE,
            tkwargs=tkwargs,
        )

        # Log the batch
        for j in range(candidates.shape[0]):
            eval_id = eval_id_start + j
            _record_eval(
                eval_id=eval_id,
                phase="bo",
                x_row=candidates[j],
                y_row=y_new[j],
                decisions=decisions_new[j],
                final=finals_new[j],
            )

        # Append to training data
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)
        decisions_all.extend(decisions_new)

        # Refit model
        warnings.filterwarnings("ignore", category=InputDataWarning)
        model = fit_mo_model(train_X, train_Y,bounds=BO_BOUNDS)
        warnings.filterwarnings("default", category=InputDataWarning)

        # Calculating hypervolume values
        from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
        with torch.no_grad():
            mask = is_non_dominated(train_Y)
            pareto_Y = train_Y[mask]

            partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
            hv = partitioning.compute_hypervolume().item()
            print(f"[Iter {it+1:03d}] evals={len(bo_records)} pareto={mask.sum().item()} hv={hv:.3f}")

            batch_size = len(candidates)
            if len(bo_records) >= batch_size:
                for i in range(1, batch_size + 1):
                     bo_records[-i]["hypervolume"] = float(hv)

        try:
            bo_log_df = pd.DataFrame(bo_records)
            bo_log_df["countries_str"] = bo_log_df["countries"].apply(lambda xs: ", ".join(xs))

            temp_log_path = output_log_path.with_suffix(".tmp.pkl")
            bo_log_df.to_pickle(temp_log_path)
            os.replace(temp_log_path, output_log_path)

            print(f"--> Saved checkpoint BO log to: {output_log_path.resolve()}")
        except Exception as e:
            print(f"[ERROR] Failed to save BO log to {output_log_path}: {e}")


    aux = dict(
        tkwargs=tkwargs,
        BO_BOUNDS=BO_BOUNDS,
        ref_point=ref_point,
    )
    return bo_log_df, train_X, train_Y
