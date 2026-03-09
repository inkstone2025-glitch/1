# dashboard.py
# scripts/dashboard.py
# Run: .\.venv\Scripts\python.exe -m streamlit run .\scripts\dashboard.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly for interactive 2D/3D
import plotly.express as px
import plotly.graph_objects as go

# Pareto utility
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

# Matplotlib for spider + curves
import matplotlib.pyplot as plt

# click to show
from streamlit_plotly_events import plotly_events
import plotly.express as px

# -----------------------------
# Helpers
# -----------------------------
DEFAULT_PKL = Path("outputs") / "bo_log_df_2026_AD-2.pkl"


def _payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert stored payload {"columns":[...], "data":[...]} -> DataFrame."""
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dashboard resilient to schema drift across runs.
    """
    df = df.copy()

    # Core columns you almost certainly have
    for col in ["eval_id", "duration_days", "cost_total", "power", "countries"]:
        if col not in df.columns:
            df[col] = np.nan

    # Helpful derived columns
    if "countries_str" not in df.columns:
        df["countries_str"] = df["countries"].apply(
            lambda xs: ", ".join(xs) if isinstance(xs, (list, tuple)) else str(xs)
        )

    # Countries count (effective)
    if "num_countries_eff" not in df.columns:
        df["num_countries_eff"] = df["countries"].apply(
            lambda xs: len(xs) if isinstance(xs, (list, tuple)) else np.nan
        )

    # Total sites used (directly from n_sites_total which is sum of per-country sites)
    if "actual_sites_used" not in df.columns:
        if "n_sites_total" in df.columns:
            df["actual_sites_used"] = df["n_sites_total"]
        else:
            df["actual_sites_used"] = np.nan

    # Optional: stored per-country target allocation
    if "target_allocation" not in df.columns:
        df["target_allocation"] = None

    return df


def compute_pareto_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_pareto based on objectives:
      - minimize duration_days
      - minimize cost_total
      - maximize power
    We transform into maximization space for is_non_dominated:
      y = [-duration, -cost, +power]
    """
    df = df.copy()

    # Drop rows with missing objectives
    mask_ok = df[["duration_days", "cost_total", "power"]].notna().all(axis=1)
    df["is_pareto"] = False
    if mask_ok.sum() == 0:
        return df

    Y = np.column_stack([
        -df.loc[mask_ok, "duration_days"].astype(float).to_numpy(),
        -df.loc[mask_ok, "cost_total"].astype(float).to_numpy(),
         df.loc[mask_ok, "power"].astype(float).to_numpy(),
    ])
    pareto = is_non_dominated(torch.tensor(Y, dtype=torch.double)).numpy()
    df.loc[mask_ok, "is_pareto"] = pareto
    return df


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_trials"] = int(len(df))
    out["n_pareto"] = int(df["is_pareto"].sum()) if "is_pareto" in df.columns else 0

    # Objective stats
    for col in ["duration_days", "cost_total", "power"]:
        if col in df.columns and df[col].notna().any():
            out[f"{col}_min"] = float(df[col].min())
            out[f"{col}_max"] = float(df[col].max())
            out[f"{col}_mean"] = float(df[col].mean())
        else:
            out[f"{col}_min"] = None
            out[f"{col}_max"] = None
            out[f"{col}_mean"] = None

    return out


def pick_recommendations(df: pd.DataFrame) -> Dict[str, Optional[pd.Series]]:
    """
    Simple, explainable recommendations:
      - best_balanced (rank-based)
      - fastest_pareto
      - cheapest_pareto
      - highest_power_pareto
    """
    rec: Dict[str, Optional[pd.Series]] = {
        "best_balanced": None,
        "fastest_pareto": None,
        "cheapest_pareto": None,
        "highest_power_pareto": None,
    }
    if df.empty:
        return rec

    # Work on rows with objective values
    d = df.dropna(subset=["duration_days", "cost_total", "power"]).copy()
    if d.empty:
        return rec

    # Balanced score: prefer high power, low duration, low cost
    d["rank_power"] = d["power"].rank(pct=True)
    d["rank_duration"] = d["duration_days"].rank(pct=True)  # higher rank = larger duration
    d["rank_cost"] = d["cost_total"].rank(pct=True)
    d["balance_score"] = 2.0* d["rank_power"] - d["rank_duration"] - d["rank_cost"]

    rec["best_balanced"] = d.loc[d["balance_score"].idxmax()]

    pareto = d[d.get("is_pareto", False) == True]
    if not pareto.empty:
        rec["fastest_pareto"] = pareto.loc[pareto["duration_days"].idxmin()]
        rec["cheapest_pareto"] = pareto.loc[pareto["cost_total"].idxmin()]
        rec["highest_power_pareto"] = pareto.loc[pareto["power"].idxmax()]

    return rec


def build_allocation_table(row: pd.Series) -> pd.DataFrame:
    """
    Build a per-country detail table:
      country_name | num_sites | num_patients

    - num_patients is from target_allocation (if available)
    - num_sites uses recruit_n_sites (sites per country) if available, else 1
    """
    countries = row.get("countries", [])
    if not isinstance(countries, (list, tuple)):
        countries = []

    # Per-country sites from the new log schema
    sites_alloc = row.get("sites_per_country", {})
    if not isinstance(sites_alloc, dict):
        sites_alloc = {}

    alloc = row.get("target_allocation", None)

    # alloc might be dict or None/NaN
    alloc_dict: Dict[str, int] = {}
    if isinstance(alloc, dict):
        alloc_dict = {str(k): int(v) for k, v in alloc.items()}
    elif isinstance(alloc, str):
        try:
            import json
            tmp = json.loads(alloc)
            if isinstance(tmp, dict):
                alloc_dict = {str(k): int(v) for k, v in tmp.items()}
        except Exception:
            alloc_dict = {}

    rows = []
    for c in countries:
        rows.append({
            "country_name": c,
            "num_sites": int(sites_alloc.get(c, 0)),
            "num_patients": int(alloc_dict.get(c, 0)) if alloc_dict else 0,
        })

    df = pd.DataFrame(rows)
    if not df.empty and alloc_dict:
        df.loc[:, "num_patients"] = df["num_patients"].astype(int)
    return df


def spider_fig_from_row(row: pd.Series, global_df: pd.DataFrame) -> plt.Figure:
    """
    Spider plot for Duration/Cost/Power with normalization vs global_df ranges.
    """
    duration_days = float(row["duration_days"])
    total_cost = float(row["cost_total"])
    power_val = float(row["power"])

    dur_min, dur_max = float(global_df["duration_days"].min()), float(global_df["duration_days"].max())
    cost_min, cost_max = float(global_df["cost_total"].min()), float(global_df["cost_total"].max())
    pow_min, pow_max = float(global_df["power"].min()), float(global_df["power"].max())

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    dur_norm = (duration_days - dur_min) / max(dur_max - dur_min, 1e-9)
    cost_norm = (total_cost - cost_min) / max(cost_max - cost_min, 1e-9)
    pow_norm = (power_val - pow_min) / max(pow_max - pow_min, 1e-9)

    scores = np.array([1.0 - clamp01(dur_norm), 1.0 - clamp01(cost_norm), clamp01(pow_norm)], dtype=float)

    labels = ["Trial Duration", "Total Cost", "Statistical Power"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    vals = np.concatenate([scores, scores[:1]])

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.plot(angles, vals, linewidth=2.5)
    ax.fill(angles, vals, alpha=0.25)

    raw_labels = [f"{duration_days:.0f} days", f"${total_cost/1e6:.2f}M", f"{power_val:.3f}"]
    for ang, txt in zip(angles[:-1], raw_labels):
        ax.text(ang, 0.95, txt, ha="center", va="center", fontsize=10, fontweight="bold")

    title_bits = [f"Eval {int(row['eval_id'])}"]
    if "n_sites_total" in row and pd.notna(row.get("n_sites_total")):
        title_bits.append(f"Sites={int(row['n_sites_total'])}")
    if "num_countries_eff" in row and pd.notna(row.get("num_countries_eff")):
        title_bits.append(f"Countries={int(row['num_countries_eff'])}")
    if "target_N" in row and pd.notna(row.get("target_N")):
        title_bits.append(f"N={int(row['target_N'])}")
    ax.set_title(" | ".join(title_bits), pad=18)

    plt.tight_layout()
    return fig


def recruitment_figs_from_payload(row: pd.Series) -> Tuple[plt.Figure, plt.Figure]:
    recruit_proj_df = _payload_to_df(row["recruit_proj_payload"])

    # (1) Cumulative prob plot (if column exists)
    fig1, ax1 = plt.subplots()
    if "enrollment_prob" in recruit_proj_df.columns:
        ax1.plot(recruit_proj_df["time"], recruit_proj_df["enrollment_prob"])
        ax1.set_ylabel("Cumulative Prob. % LSR reached")
        ax1.set_title("Projected subject recruitment rate")
    else:
        ax1.plot(recruit_proj_df["time"], recruit_proj_df["proj_enroll_0.5"])
        ax1.set_ylabel("Median enrolled")
        ax1.set_title("Recruitment (median)")

    ax1.set_xlabel("Days since enrollment start")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # (2) Quantile band plot
    fig2, ax2 = plt.subplots()
    proj = recruit_proj_df.iloc[::7, :].copy() if len(recruit_proj_df) > 14 else recruit_proj_df.copy()

    ax2.plot(proj["time"], proj["proj_enroll_0.5"], label="Median")

    if {"proj_enroll_0.25", "proj_enroll_0.75"}.issubset(proj.columns):
        ax2.fill_between(proj["time"], proj["proj_enroll_0.25"], proj["proj_enroll_0.75"], alpha=0.15, label="IQR")
    if {"proj_enroll_0.025", "proj_enroll_0.975"}.issubset(proj.columns):
        ax2.fill_between(proj["time"], proj["proj_enroll_0.025"], proj["proj_enroll_0.975"], alpha=0.08, label="95%")

    ax2.set_xlabel("Days since enrollment start")
    ax2.set_ylabel("Number of subjects")
    ax2.set_title("Subject Recruitment Projection")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best")

    plt.tight_layout()
    return fig1, fig2


def survival_fig_from_payload(row: pd.Series) -> plt.Figure:
    surv_df = _payload_to_df(row["surv_payload"])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.step(surv_df["time"], surv_df["S"], where="post")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Event-free survival")
    ax.set_title("Population average event-free survival curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def load_bo_log(pkl_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    df = _ensure_columns(df)
    df = compute_pareto_flags(df)
    return df


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Trial BO Dashboard", layout="wide")
    logo_path = Path("assets/GSK_logo.svg")
    if logo_path.exists():
        st.image(str(logo_path),width=180)
    st.title("Trial Design Bayesian Optimization Dashboard")
    st.caption("All views are driven from outputs/bo_log_df.pkl")

    # Sidebar: load path + basic controls
    st.sidebar.header("Data")
    pkl_path_str = st.sidebar.text_input("bo_log_df.pkl path", value=str(DEFAULT_PKL))
    pkl_path = Path(pkl_path_str)

    if not pkl_path.exists():
        st.error(f"Pickle not found: {pkl_path.resolve()}")
        st.stop()

    bo = load_bo_log(pkl_path)

    # Summary panel
    stats = summarize(bo)
    recs = pick_recommendations(bo)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total designs", stats["n_trials"])
    colB.metric("Pareto designs", stats["n_pareto"])
    if stats["duration_days_min"] is not None:
        colC.metric("Best duration (days)", f"{stats['duration_days_min']:.0f}")
    if stats["power_max"] is not None:
        colD.metric("Best power", f"{stats['power_max']:.3f}")

    # Auto-summary text
    st.subheader("Executive Summary")
    summary_text = (
        f"Out of **{stats['n_trials']}** evaluated trial designs, the run identified "
        f"**{stats['n_pareto']}** non-dominated (Pareto-optimal) designs. "
        f"Observed ranges: trial duration **{stats['duration_days_min']:.0f}–{stats['duration_days_max']:.0f}** days, "
        f"cost **\\${stats['cost_total_min']/1e6:.2f}M–\\${stats['cost_total_max']/1e6:.2f}M**, "
        f"statistical power **{stats['power_min']:.3f}–{stats['power_max']:.3f}**."
    )
    st.write(summary_text)

    # Recommendations
    st.subheader("Recommendations")
    c1, c2, c3, c4 = st.columns(4)

    def _rec_card(slot, label: str, row: Optional[pd.Series]):
        if row is None:
            slot.info(f"{label}\n\n(No candidate)")
            return
        slot.success(
            f"**{label}**\n\n"
            f"- Eval ID: **{int(row['eval_id'])}**\n"
            f"- Duration: **{float(row['duration_days']):.0f}** days\n"
            f"- Cost: **${float(row['cost_total'])/1e6:.2f}M**\n"
            f"- Power: **{float(row['power']):.3f}**"
        )

    _rec_card(c1, "Best balanced", recs["best_balanced"])
    _rec_card(c2, "Fastest Pareto", recs["fastest_pareto"])
    _rec_card(c3, "Cheapest Pareto", recs["cheapest_pareto"])
    _rec_card(c4, "Highest power Pareto", recs["highest_power_pareto"])

    st.divider()

    # Filters
    st.sidebar.header("Filters")
    d = bo.dropna(subset=["duration_days", "cost_total", "power"]).copy()

    # Slider ranges
    dur_min, dur_max = float(d["duration_days"].min()), float(d["duration_days"].max())
    cost_minM, cost_maxM = float(d["cost_total"].min()/1e6), float(d["cost_total"].max()/1e6)
    pow_min, pow_max = float(d["power"].min()), float(d["power"].max())

    dur_rng = st.sidebar.slider("Duration (days)", dur_min, dur_max, (dur_min, dur_max))
    cost_rng = st.sidebar.slider("Cost ($M)", cost_minM, cost_maxM, (cost_minM, cost_maxM))
    pow_rng = st.sidebar.slider("Power", pow_min, pow_max, (pow_min, pow_max))
    only_pareto = st.sidebar.checkbox("Show Pareto only", value=False)

    filtered = d[
        (d["duration_days"] >= dur_rng[0]) & (d["duration_days"] <= dur_rng[1]) &
        (d["cost_total"] >= cost_rng[0]*1e6) & (d["cost_total"] <= cost_rng[1]*1e6) &
        (d["power"] >= pow_rng[0]) & (d["power"] <= pow_rng[1])
    ].copy()
    if only_pareto:
        filtered = filtered[filtered["is_pareto"] == True]

    # Trade-off explorer
    st.subheader("Trade-off Explorer (2D)")
    x_axis = st.selectbox("X axis", ["duration_days", "cost_total", "power"], index=0)
    y_axis = st.selectbox("Y axis", ["duration_days", "cost_total", "power"], index=2)
    color_by = st.selectbox("Color", ["is_pareto", "power", "duration_days", "cost_total", "phase"], index=0)

    fig2d = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color=color_by if color_by in filtered.columns else None,
        hover_data=["eval_id", "num_countries_eff", "actual_sites_used", "target_N"],
        title="2D trade-off view",
    )
    st.plotly_chart(fig2d, width='stretch')

    st.subheader("Trade-off Explorer (3D)")
    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=filtered["duration_days"],
                y=filtered["cost_total"],
                z=filtered["power"],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.75,
                    color=filtered["power"],
                    colorscale="Viridis",
                    colorbar=dict(title="Power"),
                ),
                text=filtered["eval_id"].astype(int).astype(str),
                hovertemplate=(
                    "<b>Eval ID:</b> %{text}<br>"
                    "Duration: %{x:.1f} days<br>"
                    "Cost: %{y:,.0f} $<br>"
                    "Power: %{z:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Duration (days)",
            yaxis_title="Cost ($)",
            zaxis_title="Power",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="3D Duration vs Cost vs Power",
    )
    st.plotly_chart(fig3d, width='stretch')

    st.divider()

    # Checking the convergence (Hidden by default)
    # ================================================================
    with st.expander("Algorithm Diagnostics: Optimization Convergence", expanded=False):
        st.write(" This chart tracks the 'Hypervolume' of the Pareto front over iterations, which is a common metric for multi-objective optimization performance. A rising hypervolume indicates that the algorithm is finding better trade-offs between objectives. A flattening line indicates the model has successfully converged on the best possible trial designs")
        if "hypervolume" in bo.columns:
            hv_df = bo[bo["phase"] == "bo"].copy()
            if not hv_df.empty:
                fig_hv = px.line(
                    hv_df,
                    x="eval_id",
                    y="hypervolume",
                    markers=True,
                    labels={"eval_id": "Evaluation ID", "hypervolume": "Hypervolume Score"},
                )
                fig_hv.update_traces(line_color="#2ca02c", marker=dict(size=6))
                fig_hv.update_layout(xaxis_title="Evaluation ID", yaxis_title="Hypervolume (Higher is better)",
                                     margin=dict(l=0, r=0, t=0, b=0))
                
                st.plotly_chart(fig_hv, use_container_width=True)
                st.caption(f"**Current Best Hypervolume:** {hv_df['hypervolume'].max():.3f}")
            else:
                st.info("No hypervolume data available in the log.")
        else:
            st.info("Hypervolume column not found in the log data.")
    # ==================================================================
    
    # Ranked candiates table
    st.subheader("Trial Design Shortlist")
    sort_col = st.selectbox("Sort by", ["duration_days", "cost_total", "power"], index=0)
    ascending = True if sort_col in ("duration_days", "cost_total") else False
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    
# Define which columns to show and what to rename them to
    column_mapping = {
        "eval_id": "Eval ID",
        "is_pareto": "Pareto Optimal?",
        "duration_days": "Duration (Days)",
        "cost_total": "Total Cost ($)",
        "power": "Power",
        "target_N": "Target Subjects (N)",
        "num_countries_eff": "Active Countries",
        "actual_sites_used": "Total Sites Used",
        "countries_str": "Countries Selected",
        "phase": "Algorithm Phase" 
    }
    
    # Filter to only the columns that actually exist in the dataframe
    existing_cols = [c for c in column_mapping.keys() if c in filtered.columns]
    
    # Create a clean dataframe for display and download
    display_df = filtered[existing_cols].rename(columns=column_mapping)
    
    # Format the cost column to look like currency (optional, but highly recommended)
    if "Total Cost ($)" in display_df.columns:
        display_df["Total Cost ($)"] = display_df["Total Cost ($)"].apply(lambda x: f"${x:,.0f}")
        
    # Format the duration to standard numbers
    if "Duration (Days)" in display_df.columns:
        display_df["Duration (Days)"] = display_df["Duration (Days)"].apply(lambda x: f"{x:,.0f}")

    # Display the table
    st.dataframe(display_df, use_container_width=True, height=320)

    # Download button (Now uses the clean names!)
    st.download_button(
        "Download Trial Scenarios (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="trial_design_shortlist.csv",
        mime="text/csv",
    )
    st.divider()
    # ========================================
    
    # Drill-down to spider/curves
    st.subheader("Drill-down: View One Design (Spider plot + Curves)")

    # pick eval_id from filtered set for convenience
    eval_ids = filtered["eval_id"].astype(int).tolist() if not filtered.empty else bo["eval_id"].astype(int).tolist()
    selected_eval = st.selectbox("Select eval_id", eval_ids)

    row = bo.loc[bo["eval_id"].astype(int) == int(selected_eval)]
    if row.empty:
        st.warning("Selected eval_id not found in bo_log_df.")
        st.stop()

    r = row.iloc[0]

    # Header line
    total_patients = int(r["target_N"]) if pd.notna(r.get("target_N")) else 0
    st.write(
        f"**Eval {int(r['eval_id'])}** | "
        f"Countries: **{int(r['num_countries_eff']) if pd.notna(r.get('num_countries_eff')) else 'NA'}** | "
        f"Total Sites: **{int(r['n_sites_total']) if pd.notna(r.get('n_sites_total')) else 'NA'}** | "
        f"Patients: **{total_patients}** | "
        f"Duration: **{float(r['duration_days']):.0f} days** | "
        f"Cost: **${float(r['cost_total'])/1e6:.2f}M** | "
        f"Power: **{float(r['power']):.3f}**"
    )

    # Per-country table
    alloc_df = build_allocation_table(r)
    if not alloc_df.empty:
        st.dataframe(alloc_df, width='stretch', hide_index=True)

        # small sanity note if allocation exists
        if alloc_df["num_patients"].sum() > 0:
            st.caption(f"Allocation sum: {int(alloc_df['num_patients'].sum())}")

    # Render spider + curves
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("### Spider plot")
        fig_spider = spider_fig_from_row(r, bo.dropna(subset=["duration_days", "cost_total", "power"]))
        st.pyplot(fig_spider, clear_figure=True)

    with row1_col2:
        st.markdown("### AD Survival curve")
        if "surv_payload" in r and isinstance(r["surv_payload"], dict):
            fig_surv = survival_fig_from_payload(r)
            st.pyplot(fig_surv, clear_figure=True)
        else:
            st.info("No surv_payload stored for this eval.")

    with row2_col1:
        st.markdown("### Recruitment-1")
        if "recruit_proj_payload" in r and isinstance(r["recruit_proj_payload"], dict):
            fig_r1, fig_r2 = recruitment_figs_from_payload(r)
            st.pyplot(fig_r1, clear_figure=True)
        else:
            st.info("No recruit_proj_payload stored for this eval.")
    
    with row2_col2:
        st.markdown("### Recruitment-2")
        if "recruit_proj_payload" in r and isinstance(r["recruit_proj_payload"], dict):
            fig_r1, fig_r2 = recruitment_figs_from_payload(r)
            st.pyplot(fig_r2, clear_figure=True)
        else:
            st.info("No recruit_proj_payload stored for this eval.")


if __name__ == "__main__":
    main()