"""Create the immutable exploratory/confirmatory subject split.

The split is determined from the participant list only, not EEG or outcomes.
Its CSV and JSON digest are checked by downstream scripts; regenerating it
without --force is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_PROCESSED, DOCS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = DATA_PROCESSED / "subject_split.csv"
    if output.exists() and not args.force:
        print(f"{output} already exists; refusing to overwrite")
        return

    subjects = pd.read_parquet(DATA_PROCESSED / "subjects.parquet")[["subject", "dataset"]]
    rng = np.random.default_rng(args.seed)
    subsets = []
    for dataset, group in subjects.groupby("dataset", sort=True):
        ids = group["subject"].to_numpy().copy()
        rng.shuffle(ids)
        # 30/87, allocated by largest remainder across dataset sizes:
        n_explore = {"A": 21, "B": 7, "C": 2}[dataset]
        subsets.extend(
            {"subject": subject, "dataset": dataset, "split": "explore" if i < n_explore else "confirm"}
            for i, subject in enumerate(ids)
        )
    split = pd.DataFrame(subsets).sort_values(["dataset", "subject"]).reset_index(drop=True)
    split.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "seed": args.seed,
        "n_explore": int((split["split"] == "explore").sum()),
        "n_confirm": int((split["split"] == "confirm").sum()),
        "allocation": split.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_dict(),
        "sha256": digest,
        "rule": "stratified random assignment from subject ID and dataset only",
    }
    (DOCS / "subject_split.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
