"""Execute the fixed d=1/d=2/d=3 state-latent sensitivity series."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import cross_state_gap, fit_subject_sctm  # noqa: E402
from src.models.state_latent import fit_subject_state_components, lag1_permutation_pvalue  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=250)
    args = parser.parse_args()
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    subjects = pd.read_parquet(DATA_PROCESSED / "subjects.parquet")
    original = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    covariances = np.load(DATA_PROCESSED / "mi_covariances.npy")
    output_rows, loading_rows, validation_rows, trajectories = [], [], [], []

    for dimension in (1, 2, 3):
        state_parts, loads = [], []
        for subject, frame in features.groupby("subject", sort=False):
            state, loading = fit_subject_state_components(frame.reset_index(drop=True), dimension)
            state["latent_dimension"] = dimension
            loading["latent_dimension"] = dimension
            state_parts.append(state)
            loads.append(loading)
        state = pd.concat(state_parts, ignore_index=True)
        loads = pd.concat(loads, ignore_index=True)
        root = DATA_PROCESSED / "ablations" / f"latent_dimension_{dimension}"
        root.mkdir(parents=True, exist_ok=True)
        state.to_parquet(root / "state_latent.parquet", index=False)
        loads.to_parquet(root / "state_loadings.parquet", index=False)
        trajectories.append(state)
        loading_rows.append(loads)

        for component in range(1, dimension + 1):
            column = f"z_state_{component}"
            current = state[["subject", "run_index", "trial", column]].rename(columns={column: "z_state"})
            aligned = features.merge(current, on=["subject", "run_index", "trial"], validate="one_to_one")
            aligned = aligned.merge(split[["subject", "split"]], on="subject", validate="many_to_one")
            confirm = aligned[aligned["split"] == "confirm"]
            component_outcomes = []
            for subject in confirm["subject"].drop_duplicates():
                frame = confirm[confirm["subject"] == subject]
                cov = covariances[frame.index.to_numpy()]
                sctm = fit_subject_sctm(cov, frame, n_perm=args.permutations, seed=_seed(f"{dimension}:{component}:{subject}"))
                cross = cross_state_gap(cov, frame)
                component_outcomes.append(
                    {
                        "subject": subject,
                        "dataset": subject[0],
                        "latent_dimension": dimension,
                        "component": component,
                        "validity_label": "fixed_dimension_sensitivity_ablation",
                        **{f"sctm_{key}": value for key, value in sctm.items()},
                        **{f"cross_{key}": value for key, value in cross.items()},
                    }
                )
            output_rows.extend(component_outcomes)
            for subject, frame in state.groupby("subject", sort=False):
                values = frame[column].to_numpy()
                lag1, p = lag1_permutation_pvalue(values, n_perm=1000, seed=_seed(f"lag:{dimension}:{component}:{subject}"))
                run_mean = frame.groupby("run_index")[column].mean()
                original_subject = original[original["subject"] == subject]["z_state"].to_numpy()
                rho_to_primary = spearmanr(values, original_subject).statistic
                validation_rows.append(
                    {
                        "subject": subject,
                        "latent_dimension": dimension,
                        "component": component,
                        "lag1_autocorrelation": lag1,
                        "lag1_shuffle_p": p,
                        "spearman_to_primary_d1": rho_to_primary,
                        "first_last_run_change": float(run_mean.iloc[-1] - run_mean.iloc[0]),
                    }
                )
    results = pd.DataFrame(output_rows)
    validation = pd.DataFrame(validation_rows)
    loads = pd.concat(loading_rows, ignore_index=True)
    results.to_csv(RESULTS / "tables" / "latent_dimension_ablation.csv", index=False)
    validation.to_csv(RESULTS / "tables" / "latent_dimension_validation.csv", index=False)
    loads.to_csv(RESULTS / "tables" / "latent_dimension_loadings.csv", index=False)
    summary = (
        results[results["sctm_status"] == "ok"]
        .groupby(["latent_dimension", "component"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            differential_partial_r2=("sctm_differential_partial_r2", "mean"),
            common_norm=("sctm_common_state_norm", "mean"),
            cross_state_gap=("cross_cross_state_gap", "mean"),
        )
    )
    summary.to_csv(RESULTS / "tables" / "latent_dimension_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
