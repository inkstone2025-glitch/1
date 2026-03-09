# plot.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Payload helpers
# ============================================================
def _payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload["data"], columns=payload["columns"])

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


# ============================================================
# (A) Eval-ID based “detail plots” (NO re-run)
# ============================================================
def plot_spider_from_eval_id(
    eval_id: int,
    bo_log_df: pd.DataFrame,
) -> None:
    """
    Uses stored curve payloads in bo_log_df to plot:
      - Spider plot (Duration/Cost/Power)
      - Recruitment projection (2 plots)
      - Survival curve
    WITHOUT re-running simulation.
    """
    if "eval_id" not in bo_log_df.columns:
        raise ValueError("bo_log_df must contain an 'eval_id' column.")

    row = bo_log_df.loc[bo_log_df["eval_id"] == eval_id]
    if row.empty:
        raise ValueError(f"Eval ID {eval_id} not found.")
    r = row.iloc[0].to_dict()

    # Validate payload fields exist
    for k in ["recruit_proj_payload", "surv_payload", "power_payload"]:
        if k not in r or r[k] is None:
            raise KeyError(
                f"Missing '{k}' for eval_id={eval_id}. "
                "Make sure your BO logging stored curve payloads for every eval."
            )

    recruit_proj_df = _payload_to_df(r["recruit_proj_payload"])
    surv_df         = _payload_to_df(r["surv_payload"])
    power_df        = _payload_to_df(r["power_payload"])  # not plotted but kept for completeness

    duration_days = float(r["duration_days"])
    total_cost    = float(r["cost_total"])
    power_val     = float(r["power"])

    # -------- Print trial design details --------
    countries = list(r.get("countries", []))
    num_countries = len(countries)

    n_sites_total = _safe_int(r.get("n_sites_total", 0), 0)
    target_N = _safe_int(r.get("target_N", 0), 0)
    #recruit_n_site = _safe_int(r.get("recruit_n_site", 1), 1)

    alloc = r.get("target_allocation", None)
    if not isinstance(alloc, dict) or len(alloc) == 0:
        raise KeyError(
            "Missing 'target_allocation' in bo_log_df for this eval_id."
            "Re-run BO after logging it in record_eval()."
        )

    sites_alloc = r.get("sites_per_country", {})
    if not isinstance(sites_alloc, dict):
        sites_alloc = {}

    # Build per country table
    rows = []
    for c in countries:
        rows.append(
            {
                "country_name": c,
                "num_sites": _safe_int(sites_alloc.get(c, 0), 0),
                "num_patients": _safe_int(alloc.get(c, 0), 0),
            }
        )
    design_df = pd.DataFrame(rows)

    total_patients_alloc = int(design_df["num_patients"].sum()) if not design_df.empty else 0
    total_patients = total_patients_alloc if total_patients_alloc > 0 else target_N

    # Line 1 summary
    print(
        f"[Eval {int(r['eval_id'])}]"
        f"Countries={num_countries}|"
        f"Site_total={n_sites_total}|"
        f"Patients={total_patients}|"
        f"Duration={duration_days: .1f} days |"
        f"Cost=${total_cost/1e6: .2f}M |"
        f"Power={power_val:.3f}"
    )

    print("\nPer-country design:")
    print(design_df.to_string(index=False))

    if total_patients_alloc != target_N and target_N >0:
        print(f"\n[WARN] Allocation sum({total_patients_alloc}) != target_N ({target_N}).")

    # -------- Spider normalization ranges from the full BO log --------
    dur_min, dur_max = float(bo_log_df["duration_days"].min()), float(bo_log_df["duration_days"].max())
    cost_min, cost_max = float(bo_log_df["cost_total"].min()), float(bo_log_df["cost_total"].max())
    pow_min, pow_max = float(bo_log_df["power"].min()), float(bo_log_df["power"].max())

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    dur_norm  = (duration_days - dur_min) / max(dur_max - dur_min, 1e-9)
    cost_norm = (total_cost - cost_min) / max(cost_max - cost_min, 1e-9)
    pow_norm  = (power_val - pow_min) / max(pow_max - pow_min, 1e-9)

    # Scores: higher is better
    scores = np.array(
        [1.0 - clamp01(dur_norm), 1.0 - clamp01(cost_norm), clamp01(pow_norm)],
        dtype=float,
    )

    labels = ["Trial Duration", "Cost", "Statistical Power"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    vals   = np.concatenate([scores, scores[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.plot(angles, vals, linewidth=2.5)
    ax.fill(angles, vals, alpha=0.25)

    raw_labels = [f"{duration_days:.0f} days", f"${total_cost/1e6:.2f}M", f"{power_val:.3f}"]
    for ang, txt in zip(angles[:-1], raw_labels):
        ax.text(ang, 0.95, txt, ha="center", va="center", fontsize=10, fontweight="bold")

    title = (
        f"Eval {int(r['eval_id'])} | "
        f"Sites={r.get('n_sites_total','?')} | "
        f"Countries={r.get('num_countries','?')} | "
        f"N={r.get('target_N','?')}"
    )
    ax.set_title(title, pad=18)
    plt.tight_layout()
    plt.show()

    # -------- Recruitment plots (your provided code) --------
    fig, ax = plt.subplots()
    ax.plot(recruit_proj_df["time"], recruit_proj_df["enrollment_prob"])
    plt.xlabel("Days since enrollment start")
    plt.ylabel("Cumulative Prob. % LSR reached")
    plt.title("Projected subject recruitment rate")
    plt.show()

    fig, ax = plt.subplots()
    projections_7 = recruit_proj_df.iloc[::7, :]
    ax.plot(projections_7["time"], projections_7["proj_enroll_0.5"])
    ax.fill_between(
        projections_7["time"],
        projections_7["proj_enroll_0.25"],
        projections_7["proj_enroll_0.75"],
        color="b",
        alpha=0.1,
    )
    ax.fill_between(
        projections_7["time"],
        projections_7["proj_enroll_0.025"],
        projections_7["proj_enroll_0.975"],
        color="b",
        alpha=0.05,
    )
    plt.xlabel("Days since enrollment start")
    plt.ylabel("Number of subjects")
    plt.title("Subject Recruitment Projection")
    plt.show()

    # -------- Survival curve (your provided code) --------
    plt.figure(figsize=(6, 4))
    plt.step(surv_df["time"], surv_df["S"], where="post")
    plt.ylim(0, 1.05)
    plt.xlabel("Time (years)")
    plt.ylabel("Event-free survival %")
    plt.title("Population average event free survival curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ============================================================
# (B) Objective arrays from BO log (no train_Y needed)
# ============================================================
def objectives_from_log(
    bo_log_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (dur_all, cost_all, power_all) as numpy arrays from bo_log_df.
    """
    for col in ["duration_days", "cost_total", "power"]:
        if col not in bo_log_df.columns:
            raise ValueError(f"bo_log_df must contain '{col}'.")

    dur_all   = bo_log_df["duration_days"].to_numpy(dtype=float)
    cost_all  = bo_log_df["cost_total"].to_numpy(dtype=float)
    power_all = bo_log_df["power"].to_numpy(dtype=float)
    return dur_all, cost_all, power_all


# ============================================================
# (C) 4-panel objective-space plot (from bo_log_df)
# ============================================================
def plot_objective_space_4panel_from_log(
    bo_log_df: pd.DataFrame,
    *,
    show_pareto: bool = False,
    pareto_mask: Optional[np.ndarray] = None,
    cost_scale: float = 1.0,
) -> None:
    """
    4-panel plot using bo_log_df only.
    If you already computed a Pareto mask elsewhere, pass it in as pareto_mask.
    Otherwise, set show_pareto=False (default).

    cost_scale:
      - 1.0 -> plot cost in $
      - 1e-7 -> plot in "10M $" units (cost / 10,000,000)
    """
    dur_all, cost_all, power_all = objectives_from_log(bo_log_df)

    cost_all_plot = cost_all * cost_scale
    if cost_scale == 1.0:
        cost_label = "Total Cost ($)"
    elif np.isclose(cost_scale, 1e-7):
        cost_label = "Total Cost (10M $)"
    else:
        cost_label = f"Total Cost (scaled ×{cost_scale:g})"

    # correlation + regression helpers
    def poly_regression(x: np.ndarray, y: np.ndarray, degree: int = 2):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        coeff = np.polyfit(x, y, degree)
        poly  = np.poly1d(coeff)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ys = poly(xs)
        return xs, ys

    def get_corr_text(x: np.ndarray, y: np.ndarray) -> str:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        try:
            from scipy import stats
            pear = stats.pearsonr(x, y)
            return f"Pearson r = {pear.statistic:.4f}, p = {pear.pvalue:.2e}"
        except Exception:
            r = float(np.corrcoef(x, y)[0, 1])
            return f"Pearson r = {r:.4f} (p-value unavailable; install scipy)"

    fig = plt.figure(figsize=(14, 10))

    # (1) COST vs DURATION
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(cost_all_plot, dur_all, s=50, marker="o")
    xs, ys = poly_regression(cost_all_plot, dur_all, degree=2)
    ax1.plot(xs, ys, color="red", linewidth=2)
    ax1.text(0.05, 0.95, get_corr_text(cost_all_plot, dur_all),
             transform=ax1.transAxes, fontsize=12, va="top")
    ax1.set_xlabel(cost_label)
    ax1.set_ylabel("Trial Duration (days)")
    ax1.set_title("Total Cost vs. Trial Duration")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # (2) COST vs POWER
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(power_all, cost_all_plot, s=50, marker="o")
    xs, ys = poly_regression(power_all, cost_all_plot, degree=3)
    ax2.plot(xs, ys, color="red", linewidth=2)
    ax2.text(0.05, 0.95, get_corr_text(power_all, cost_all_plot),
             transform=ax2.transAxes, fontsize=12, va="top")
    ax2.set_xlabel("Statistical Power")
    ax2.set_ylabel(cost_label)
    ax2.set_title("Total Cost vs. Statistical Power")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # (3) POWER vs DURATION
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(power_all, dur_all, s=50, marker="o")
    xs, ys = poly_regression(power_all, dur_all, degree=2)
    ax3.plot(xs, ys, color="red", linewidth=2)
    ax3.text(0.05, 0.95, get_corr_text(power_all, dur_all),
             transform=ax3.transAxes, fontsize=12, va="top")
    ax3.set_xlabel("Statistical Power")
    ax3.set_ylabel("Trial Duration (days)")
    ax3.set_title("Statistical Power vs. Trial Duration")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # (4) 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(dur_all, cost_all_plot, power_all, s=60, marker="o")

    try:
        from matplotlib import cm
        surf = ax4.plot_trisurf(
            dur_all,
            cost_all_plot,
            power_all,
            cmap=cm.viridis,
            alpha=0.5,
            linewidth=0.2,
            antialiased=True,
        )
        fig.colorbar(surf, ax=ax4, shrink=0.6, aspect=10, pad=0.10, label="Power")
    except Exception as e:
        print(f"[WARN] Could not draw trisurf surface: {e}")

    ax4.set_xlabel("Trial Duration (days)")
    ax4.set_ylabel(cost_label)
    ax4.set_zlabel("Statistical Power")
    ax4.set_title("3D Objective Space")

    plt.tight_layout()
    plt.show()


# ============================================================
# (D) Interactive 3D plot (from bo_log_df)
# ============================================================
def plot_interactive_3d_from_log(
    bo_log_df: pd.DataFrame,
    *,
    title: str = "3D Design Space (interactive): Duration vs Cost vs Power",
    include_countries: bool = True,
) -> pd.DataFrame:
    """
    Interactive Plotly plot using only bo_log_df.
    Returns a safe copy used for plotting (so you can inspect it).
    """
    import plotly.graph_objects as go

    plot_df = bo_log_df.copy()

    # Ensure required fields exist
    for col in ["duration_days", "cost_total", "power", "n_sites_total", "num_countries", "target_N"]:
        if col not in plot_df.columns:
            raise ValueError(f"bo_log_df must contain '{col}' for interactive plotting.")

    if "eval_id" not in plot_df.columns:
        plot_df["eval_id"] = np.arange(len(plot_df), dtype=int)

    if "phase" not in plot_df.columns:
        plot_df["phase"] = "eval"

    if include_countries:
        if "countries_str" not in plot_df.columns:
            if "countries" in plot_df.columns:
                plot_df["countries_str"] = plot_df["countries"].apply(
                    lambda xs: ", ".join(xs) if isinstance(xs, (list, tuple)) else str(xs)
                )
            else:
                plot_df["countries_str"] = ""
    else:
        plot_df["countries_str"] = ""

    dur_all   = plot_df["duration_days"].to_numpy(dtype=float)
    cost_all  = plot_df["cost_total"].to_numpy(dtype=float)
    power_all = plot_df["power"].to_numpy(dtype=float)

    customdata = np.stack(
        [
            plot_df["eval_id"].to_numpy(dtype=int),                 # [0]
            plot_df["phase"].astype(str).to_numpy(),                # [1]
            plot_df["n_sites_total"].to_numpy(dtype=int),           # [2]
            plot_df["num_countries"].to_numpy(dtype=int),         # [3]
            plot_df["target_N"].to_numpy(dtype=int),                # [4]
            plot_df["countries_str"].astype(str).to_numpy(),        # [5]
        ],
        axis=-1,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=dur_all,
            y=cost_all,
            z=power_all,
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.75,
                color=power_all,
                colorscale="Viridis",
                colorbar=dict(title="Power"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Eval ID:</b> %{customdata[0]}<br>"
                "<b>Phase:</b> %{customdata[1]}<br>"
                "<b>Sites:</b> %{customdata[2]}<br>"
                "<b>#Countries:</b> %{customdata[3]}<br>"
                "<b>Target N:</b> %{customdata[4]}<br>"
                "<b>Countries:</b> %{customdata[5]}<br><br>"
                "Duration: %{x:.1f} days<br>"
                "Cost: %{y:,.0f} $<br>"
                "Power: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Duration (days)",
            yaxis_title="Total cost ($)",
            zaxis_title="Statistical power",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.show()

    return plot_df