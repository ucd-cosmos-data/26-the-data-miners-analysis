"""S2: create dual-stream, per-subject epochs from the raw GDF recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_INTERIM, DATA_PROCESSED, MI_RUNS, MI_WINDOW, SFREQ_TARGET, all_subjects, gdf_path  # noqa: E402
from src.preprocessing.epochs import preprocess_run  # noqa: E402


def process_subject(
    subject: str,
    *,
    output_dir: Path,
    mi_eog_regression: bool,
    circular_state_window: bool,
    matched_erd_stream: bool = False,
    overwrite: bool = False,
) -> dict:
    output = output_dir / f"{subject}_epochs.npz"
    metadata_path = output_dir / f"{subject}_trials.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and metadata_path.exists() and not overwrite:
        return {"subject": subject, "status": "cached"}

    flags = pd.read_csv(DATA_INTERIM / "comment_flags.csv")
    all_mi, all_state, all_eog, all_emg, all_tables = [], [], [], [], []
    bad_by_run: dict[str, list[str]] = {}
    errors: list[str] = []
    for suffix in MI_RUNS:
        path = gdf_path(subject, suffix)
        if not path.exists():
            errors.append(f"missing:{suffix}")
            continue
        run_index = int(suffix[1])
        comment_bad = flags[
            (flags["subject"] == subject) & (flags["suffix"] == suffix) & (flags["channel"] != "")
        ]["channel"].tolist()
        try:
            run = preprocess_run(
                str(path),
                subject,
                run_index,
                comment_bad,
                mi_eog_regression=mi_eog_regression,
                state_window=MI_WINDOW if circular_state_window else (-3.0, -0.5),
                matched_erd_stream=matched_erd_stream,
            )
        except Exception as exc:
            errors.append(f"{suffix}:{type(exc).__name__}:{exc}")
            continue
        all_mi.append(run.mi)
        all_state.append(run.state_eeg)
        all_eog.append(run.state_eog)
        all_emg.append(run.emg)
        all_tables.append(run.trial_table)
        bad_by_run[suffix] = run.bad_channels

    if not all_tables:
        raise RuntimeError(f"{subject}: no runs could be preprocessed: {errors}")
    table = pd.concat(all_tables, ignore_index=True)
    np.savez_compressed(
        output,
        mi=np.concatenate(all_mi),
        state_eeg=np.concatenate(all_state),
        state_eog=np.concatenate(all_eog),
        emg=np.concatenate(all_emg),
        sfreq=SFREQ_TARGET,
    )
    table.to_parquet(metadata_path, index=False)
    return {
        "subject": subject,
        "status": "done",
        "n_trials": len(table),
        "n_rejected": int(table["reject_flag"].sum()),
        "bad_channels": json.dumps(bad_by_run),
        "errors": "; ".join(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--branch",
        choices=["primary", "no_eog_regression", "mi_window_circularity", "matched_erd"],
        default="primary",
        help="Primary uses pre-cue state + EOG regression. Non-primary branches are ablations.",
    )
    args = parser.parse_args()
    output_dir = (
        DATA_PROCESSED / "epochs"
        if args.branch == "primary"
        else DATA_PROCESSED / "ablations" / args.branch / "epochs"
    )
    subjects = args.subjects or all_subjects()
    subjects = [s for i, s in enumerate(subjects) if i % args.n_shards == args.shard_index]
    records = []
    for i, subject in enumerate(subjects, start=1):
        print(f"[{i}/{len(subjects)}] {subject}", flush=True)
        try:
            records.append(
                process_subject(
                    subject,
                    output_dir=output_dir,
                    mi_eog_regression=args.branch != "no_eog_regression",
                    circular_state_window=args.branch == "mi_window_circularity",
                    matched_erd_stream=args.branch == "matched_erd",
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:
            records.append({"subject": subject, "status": "failed", "errors": repr(exc)})
    report = pd.DataFrame(records)
    suffix = f"_shard{args.shard_index}" if args.n_shards > 1 else ""
    report.to_csv(DATA_INTERIM / f"preprocessing_report_{args.branch}{suffix}.csv", index=False)
    print(report["status"].value_counts().to_dict())
    if "n_trials" in report:
        print("processed trials:", int(report["n_trials"].fillna(0).sum()))


if __name__ == "__main__":
    main()
