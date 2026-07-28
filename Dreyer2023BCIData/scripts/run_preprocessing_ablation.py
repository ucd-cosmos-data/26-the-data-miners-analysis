"""Evaluate isolated no-EOG and MI-window-state preprocessing branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import cross_state_gap, fit_subject_sctm  # noqa: E402
from src.models.state_latent import fit_subject_state, lag1_permutation_pvalue  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["no_eog_regression", "mi_window_circularity"], required=True)
    parser.add_argument("--permutations", type=int, default=250)
    args = parser.parse_args()
    root = DATA_PROCESSED / "ablations" / args.branch
    features = pd.read_parquet(root / "state_features.parquet")
    covariances = np.load(root / "mi_covariances.npy")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    primary = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    state_parts, loadings, diagnostics = [], [], []
    for subject, frame in features.groupby("subject", sort=False):
        state, loading = fit_subject_state(frame.reset_index(drop=True))
        state_parts.append(state)
        loadings.append(loading)
        lag1, lag_p = lag1_permutation_pvalue(state["z_state"].to_numpy(), n_perm=1000, seed=_seed(f"{args.branch}:{subject}"))
        primary_subject = primary[primary["subject"] == subject]["z_state"].to_numpy()
        diagnostics.append(
            {
                "subject": subject,
                "branch": args.branch,
                "lag1_autocorrelation": lag1,
                "lag1_shuffle_p": lag_p,
                "correlation_to_primary_state": spearmanr(state["z_state"], primary_subject).statistic,
            }
        )
    state = pd.concat(state_parts, ignore_index=True)
    state.to_parquet(root / "state_latent.parquet", index=False)
    pd.concat(loadings, ignore_index=True).to_parquet(root / "state_loadings.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(root / "state_diagnostics.csv", index=False)
    aligned = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"], validate="one_to_one")
    aligned = aligned.merge(split[["subject", "split"]], on="subject", validate="many_to_one")
    confirm = aligned[aligned["split"] == "confirm"]
    covariances = covariances[features.index[features["subject"].isin(confirm["subject"].unique())].to_numpy()]
    rows, cursor = [], 0
    for subject in confirm["subject"].drop_duplicates():
        frame = confirm[confirm["subject"] == subject]
        cov = covariances[cursor : cursor + len(frame)]
        cursor += len(frame)
        sctm = fit_subject_sctm(cov, frame, n_perm=args.permutations, seed=_seed(f"{args.branch}:{subject}:sctm"))
        cross = cross_state_gap(cov, frame)
        rows.append(
            {
                "branch": args.branch,
                "validity_label": "circularity_ablation_not_valid_for_primary_inference" if args.branch == "mi_window_circularity" else "preprocessing_sensitivity_ablation",
                "subject": subject,
                "dataset": subject[0],
                "rejection_rate": float(frame["reject_flag"].mean()),
                **{f"sctm_{key}": value for key, value in sctm.items()},
                **{f"cross_{key}": value for key, value in cross.items()},
                "permutations": args.permutations,
                "seed": _seed(f"{args.branch}:{subject}:sctm"),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(root / "evaluation.csv", index=False)
    output = RESULTS / "tables" / ("mi_window_state_circularity_ablation.csv" if args.branch == "mi_window_circularity" else "eog_preprocessing_ablation.csv")
    result.to_csv(output, index=False)
    metadata = {
        "branch": args.branch,
        "config_hash": hashlib.sha256(json.dumps({"branch": args.branch, "permutations": args.permutations}, sort_keys=True).encode()).hexdigest()[:16],
        "n_subjects": int(result["subject"].nunique()),
        "n_trials": int(len(features)),
        "validity_label": result["validity_label"].iloc[0],
    }
    (root / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
