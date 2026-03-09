# run_bo.py

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repot root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# debug for old tokens
if "DATABRICKS_TOKEN" in os.environ:
    print(f"token loaded: starts with'{os.environ['DATABRICKS_TOKEN'][:4]}...' and ends with '{os.environ['DATABRICKS_TOKEN'][-4:]}'.")

# import bo modules

from trial_bo.pipeline import build_graph
from trial_bo.bo import run_bo
from trial_bo.config import SURV_BASE, BOConfig

def main() -> None:
    # make sure output directory exists
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Path to save new results
    output_log_path = out_dir / "bo_log_df_2026_AD-3.pkl"

    # define the old BO log file to resume from (if any)
    resume_from_file = out_dir / "bo_log_df_2026_AD-3.pkl"

    # build LangGraph app
    app = build_graph()

    cfg = BOConfig()

    # run BO
    bo_log_df, train_X, train_Y = run_bo(
        app = app,
        SURV_BASE = SURV_BASE,
        seed=cfg.seed,
        N_INIT=cfg.N_INIT,
        N_ITERS=cfg.N_ITERS,
        Q_BATCH=cfg.Q_BATCH,
        RAW_SAMP=cfg.RAW_SAMP,
        RESTARTS=cfg.RESTARTS,
        NUM_MC_SAMPLES=cfg.NUM_MC_SAMPLES,
        resume_from_file=resume_from_file,
        output_log_path=output_log_path,
    )

    print("\nDone.")

if __name__ == "__main__":
    main()
