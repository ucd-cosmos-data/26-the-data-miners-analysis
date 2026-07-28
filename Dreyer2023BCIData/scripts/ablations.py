"""Pre-specified SCTM sensitivity analyses.

This script deliberately labels a sensorimotor-idle pseudo-state as circular:
it is included to quantify the inflation that occurs if a state proxy is too
close to the sensorimotor representation, not as evidence for the main claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import fit_subject_sctm  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _run(label: str, frame: pd.DataFrame, covs: np.ndarray) -> list[dict]:
    results, cursor = [], 0
    for subject in frame["subject"].drop_duplicates():
        part = frame[frame["subject"] == subject]
        n = len(part)
        out = fit_subject_sctm(covs[cursor:cursor+n], part, n_perm=100, seed=hash((subject, label)) % 2**32)
        out.update({"subject": subject, "ablation": label})
        results.append(out)
        cursor += n
    return results


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    base = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"]).merge(split, on="subject")
    base = base[base["split"] == "confirm"].reset_index(drop=True)
    all_features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    covs = np.load(DATA_PROCESSED / "mi_covariances.npy")
    index = all_features.index[all_features["subject"].isin(base["subject"].unique())].to_numpy()
    covs = covs[index]

    output = []
    output += _run("primary_reject_excluded", base, covs)
    include_rejected = base.copy()
    include_rejected["reject_flag"] = False
    output += _run("include_artifact_flagged_trials", include_rejected, covs)
    open_loop = base[base["run_index"].isin([1, 2])].copy()
    # Filter covariances with exactly the same boolean mask.
    open_mask = base["run_index"].isin([1, 2]).to_numpy()
    output += _run("open_loop_R1_R2", open_loop, covs[open_mask])
    circular = base.copy()
    circular["z_state"] = circular["sensorimotor_idle_mu_logpower"]
    output += _run("circular_sensorimotor_idle_proxy", circular, covs)
    result = pd.DataFrame(output)
    result.to_csv(RESULTS / "tables" / "sctm_ablations_confirm.csv", index=False)
    print(result.groupby("ablation")["differential_partial_r2"].agg(["count", "mean", "std"]).to_string())


if __name__ == "__main__":
    main()
