"""S4: derive pre-cue state features and Ledoit-Wolf MI covariances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extract import ledoit_wolf_covariances, state_features  # noqa: E402
from src.paths import DATA_PROCESSED, SFREQ_TARGET, all_subjects  # noqa: E402


def one(subject: str) -> tuple[pd.DataFrame, np.ndarray]:
    source = DATA_PROCESSED / "epochs" / f"{subject}_epochs.npz"
    table_file = DATA_PROCESSED / "epochs" / f"{subject}_trials.parquet"
    if not source.exists() or not table_file.exists():
        raise FileNotFoundError(f"{subject}: missing preprocessing artifacts")
    arrays = np.load(source)
    table = pd.read_parquet(table_file).reset_index(drop=True)
    features = state_features(arrays["state_eeg"], arrays["state_eog"], arrays["emg"], float(arrays["sfreq"]))
    if len(features) != len(table):
        raise ValueError(f"{subject}: feature/table length mismatch")
    covs = ledoit_wolf_covariances(arrays["mi"])
    return pd.concat([table, features], axis=1), covs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        frames, arrays = [], []
        for shard in range(args.n_shards):
            table_path = DATA_PROCESSED / f"state_features_part{shard}.parquet"
            cov_path = DATA_PROCESSED / f"mi_covariances_part{shard}.npy"
            if not table_path.exists() or not cov_path.exists():
                raise FileNotFoundError(f"missing feature shard {shard}")
            frames.append(pd.read_parquet(table_path))
            arrays.append(np.load(cov_path))
        features, covs = pd.concat(frames, ignore_index=True), np.concatenate(arrays)
        features.to_parquet(DATA_PROCESSED / "state_features.parquet", index=False)
        np.save(DATA_PROCESSED / "mi_covariances.npy", covs)
        print({"n_trials": len(features), "covariance_shape": covs.shape, "combined_shards": args.n_shards})
        return
    subjects = args.subjects or all_subjects()
    subjects = [s for i, s in enumerate(subjects) if i % args.n_shards == args.shard_index]
    tables, covariances, failed = [], [], []
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {subject}", flush=True)
        try:
            features, covs = one(subject)
            tables.append(features)
            covariances.append(covs)
        except Exception as exc:
            failed.append({"subject": subject, "error": repr(exc)})
    if not tables:
        raise RuntimeError("No preprocessed subjects available")
    features = pd.concat(tables, ignore_index=True)
    covs = np.concatenate(covariances)
    suffix = f"_part{args.shard_index}" if args.n_shards > 1 else ""
    features.to_parquet(DATA_PROCESSED / f"state_features{suffix}.parquet", index=False)
    np.save(DATA_PROCESSED / f"mi_covariances{suffix}.npy", covs)
    pd.DataFrame(failed).to_csv(DATA_PROCESSED / f"feature_failures{suffix}.csv", index=False)
    print({"n_trials": len(features), "covariance_shape": covs.shape, "failed": len(failed)})


if __name__ == "__main__":
    main()
