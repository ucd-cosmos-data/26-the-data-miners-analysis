"""Extract features/covariances for isolated preprocessing ablation branches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extract import ledoit_wolf_covariances, state_features  # noqa: E402
from src.paths import DATA_PROCESSED, all_subjects  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["no_eog_regression", "mi_window_circularity"], required=True)
    args = parser.parse_args()
    root = DATA_PROCESSED / "ablations" / args.branch
    epochs = root / "epochs"
    rows, covariances, failures = [], [], []
    for subject in all_subjects():
        array_file = epochs / f"{subject}_epochs.npz"
        table_file = epochs / f"{subject}_trials.parquet"
        if not array_file.exists() or not table_file.exists():
            failures.append({"subject": subject, "reason": "missing_epochs"})
            continue
        try:
            arrays = np.load(array_file)
            table = pd.read_parquet(table_file).reset_index(drop=True)
            feature = state_features(arrays["state_eeg"], arrays["state_eog"], arrays["emg"], float(arrays["sfreq"]))
            if len(feature) != len(table):
                raise ValueError("feature/table length mismatch")
            rows.append(pd.concat([table, feature], axis=1))
            covariances.append(ledoit_wolf_covariances(arrays["mi"]))
        except Exception as exc:
            failures.append({"subject": subject, "reason": repr(exc)})
    if not rows:
        raise RuntimeError(f"{args.branch}: no usable subject artifacts")
    features = pd.concat(rows, ignore_index=True)
    covs = np.concatenate(covariances)
    root.mkdir(parents=True, exist_ok=True)
    features.to_parquet(root / "state_features.parquet", index=False)
    np.save(root / "mi_covariances.npy", covs)
    pd.DataFrame(failures).to_csv(root / "feature_failures.csv", index=False)
    metadata = {
        "branch": args.branch,
        "validity_label": "circularity_ablation_not_valid_for_primary_inference" if args.branch == "mi_window_circularity" else "preprocessing_ablation",
        "n_trials": int(len(features)),
        "n_subjects": int(features["subject"].nunique()),
        "n_failures": int(len(failures)),
    }
    (root / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
