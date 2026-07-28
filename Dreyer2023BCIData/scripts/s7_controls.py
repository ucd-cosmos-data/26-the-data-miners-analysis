"""S7 null controls for the within-run cross-state generalisation gap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import cross_state_gap_features, tangent_features  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    base = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    full = base.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"]).merge(split, on="subject")
    full = full[full["split"] == "confirm"].reset_index(drop=True)
    all_features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    covs_all = np.load(DATA_PROCESSED / "mi_covariances.npy")
    idx = all_features.index[all_features["subject"].isin(full["subject"].unique())].to_numpy()
    covs = covs_all[idx]
    rng = np.random.default_rng(20260727)
    rows, cursor = [], 0
    # Preserve covariance order; see matching note in s6_sctm.py.
    for subject in full["subject"].drop_duplicates():
        frame = full[full["subject"] == subject]
        n = len(frame)
        subject_covs = covs[cursor:cursor+n]
        cursor += n
        tangent = tangent_features(subject_covs)
        observed = cross_state_gap_features(tangent, frame)
        rows.append({"subject": subject, "control": "observed_state", "repeat": 0, **observed})
        for repeat in range(args.repeats):
            shuffled = frame.copy()
            shuffled["z_state"] = shuffled.groupby("run_index")["z_state"].transform(lambda s: rng.permutation(s.to_numpy()))
            rows.append({"subject": subject, "control": "within_run_state_permutation", "repeat": repeat, **cross_state_gap_features(tangent, shuffled)})
            time_control = frame.copy()
            time_control["z_state"] = time_control.groupby("run_index")["trial"].transform(lambda s: s - s.mean())
            rows.append({"subject": subject, "control": "within_run_trial_index", "repeat": repeat, **cross_state_gap_features(tangent, time_control)})
            random_control = frame.copy()
            random_control["z_state"] = random_control.groupby("run_index")["trial"].transform(lambda s: rng.normal(size=len(s)))
            rows.append({"subject": subject, "control": "random_within_run_split", "repeat": repeat, **cross_state_gap_features(tangent, random_control)})
        print(subject, flush=True)
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "tables" / "cross_state_controls_confirm.csv", index=False)
    print(output.groupby("control")["cross_state_gap"].agg(["count", "mean", "std"]).to_string())


if __name__ == "__main__":
    main()
