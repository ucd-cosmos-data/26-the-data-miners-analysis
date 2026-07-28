"""S0: build the file manifest and QC report for the raw GDF tree.

Reads every GDF header and event table, verifies the trial structure implied
by the OpenViBE Graz scenario, and records anomalies (missing runs, stray
files, unexpected event counts) so that later stages can fail loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdf_io import (  # noqa: E402
    FILE_CHANNEL_ORDER,
    read_events,
    read_header,
    sha256_file,
)
from src.paths import (  # noqa: E402
    DATA_INTERIM,
    DATASETS,
    EXPECTED_SUFFIXES,
    RUN_KINDS,
    SIGNALS,
    all_subjects,
    gdf_path,
    subject_dir,
)

EXPECTED_CHANNELS = list(FILE_CHANNEL_ORDER)
KEY_CODES = {
    "trial_start": 768,
    "cue_left": 769,
    "cue_right": 770,
    "feedback": 781,
    "cross": 786,
    "trial_end": 800,
    "beep": 33282,
}


def _describe(subject: str, suffix: str, with_hash: bool) -> dict:
    path = gdf_path(subject, suffix)
    row = {
        "subject": subject,
        "dataset": subject[0],
        "suffix": suffix,
        "kind": RUN_KINDS[suffix],
        "run_index": int(suffix[1]) if suffix.startswith("R") else np.nan,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return row

    header = read_header(path)
    events = read_events(path, header)
    codes, counts = np.unique(events[:, 1], return_counts=True)
    tally = dict(zip(codes.tolist(), counts.tolist()))

    row.update(
        {
            "size_bytes": path.stat().st_size,
            "version": header.version,
            "n_channels": header.n_channels,
            "sfreq": header.sfreq,
            "n_samples": header.n_samples,
            "duration_s": round(header.duration_s, 3),
            "n_events": header.n_events,
            "channels_as_expected": header.ch_names == EXPECTED_CHANNELS,
        }
    )
    for name, code in KEY_CODES.items():
        row[f"n_{name}"] = tally.get(code, 0)

    cues = events[np.isin(events[:, 1], [769, 770])]
    row["n_trials"] = len(cues)
    if len(cues) > 1:
        gaps = np.diff(cues[:, 0]) / header.sfreq
        row["cue_gap_min_s"] = round(float(gaps.min()), 3)
        row["cue_gap_max_s"] = round(float(gaps.max()), 3)
        row["cue_gap_median_s"] = round(float(np.median(gaps)), 3)
    if with_hash:
        row["sha256"] = sha256_file(path)
    return row


def _timing_check(subject: str, suffix: str) -> dict | None:
    """Verify cue->feedback->trial-end offsets against the scenario spec."""
    path = gdf_path(subject, suffix)
    if not path.exists() or RUN_KINDS[suffix] == "baseline":
        return None
    header = read_header(path)
    events = read_events(path, header)
    fs = header.sfreq
    cues = events[np.isin(events[:, 1], [769, 770])][:, 0]
    fb = events[events[:, 1] == 781][:, 0]
    end = events[events[:, 1] == 800][:, 0]
    cross = events[events[:, 1] == 786][:, 0]
    n = min(len(cues), len(fb), len(end), len(cross))
    if n == 0:
        return None
    return {
        "subject": subject,
        "suffix": suffix,
        "cue_to_feedback_s": round(float(np.median((fb[:n] - cues[:n]) / fs)), 4),
        "cue_to_end_s": round(float(np.median((end[:n] - cues[:n]) / fs)), 4),
        "cross_to_cue_s": round(float(np.median((cues[:n] - cross[:n]) / fs)), 4),
    }


def find_file_anomalies() -> pd.DataFrame:
    rows = []
    for dataset, folder in DATASETS.items():
        for sub_path in sorted((SIGNALS / folder).iterdir()):
            if not sub_path.is_dir():
                continue
            subject = sub_path.name
            names = {p.name for p in sub_path.iterdir()}
            for suffix in EXPECTED_SUFFIXES:
                if f"{subject}_{suffix}.gdf" not in names:
                    rows.append(
                        {
                            "subject": subject,
                            "issue": "missing_gdf",
                            "detail": suffix,
                        }
                    )
            for name in sorted(names):
                if name.endswith(".gdf"):
                    if name not in {f"{subject}_{s}.gdf" for s in EXPECTED_SUFFIXES}:
                        rows.append(
                            {"subject": subject, "issue": "stray_gdf", "detail": name}
                        )
                elif subject not in name:
                    rows.append(
                        {"subject": subject, "issue": "misnamed_aux", "detail": name}
                    )
    return pd.DataFrame(rows)


def check_baseline_duplication(subjects: list[str], n: int = 12) -> pd.DataFrame:
    """All baseline files share a byte size; confirm they are not duplicates."""
    rows = []
    for subject in subjects[:n]:
        for suffix in ("OE_baseline", "CE_baseline"):
            path = gdf_path(subject, suffix)
            if path.exists():
                rows.append(
                    {
                        "subject": subject,
                        "suffix": suffix,
                        "sha256": sha256_file(path),
                    }
                )
    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated("sha256", keep=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hash", action="store_true", help="skip sha256")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    subjects = all_subjects()
    print(f"subjects: {len(subjects)}")

    tasks = [(s, suf) for s in subjects for suf in EXPECTED_SUFFIXES]
    rows = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(_describe)(s, suf, not args.no_hash) for s, suf in tasks
    )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DATA_INTERIM / "manifest.csv", index=False)

    timing = Parallel(n_jobs=args.jobs)(
        delayed(_timing_check)(s, suf) for s, suf in tasks
    )
    timing_df = pd.DataFrame([t for t in timing if t is not None])
    timing_df.to_csv(DATA_INTERIM / "timing_check.csv", index=False)

    anomalies = find_file_anomalies()
    anomalies.to_csv(DATA_INTERIM / "file_anomalies.csv", index=False)

    dupes = check_baseline_duplication(subjects)
    dupes.to_csv(DATA_INTERIM / "baseline_hashes.csv", index=False)

    present = manifest[manifest["exists"]]
    lines = [
        "# S0 QC report",
        "",
        f"- subjects: {len(subjects)}",
        f"- expected GDF files: {len(tasks)}",
        f"- present: {len(present)}  missing: {(~manifest['exists']).sum()}",
        f"- total size: {present['size_bytes'].sum() / 1e9:.2f} GB",
        f"- sfreq values: {sorted(present['sfreq'].unique().tolist())}",
        f"- channel counts: {sorted(present['n_channels'].unique().tolist())}",
        f"- channel order as expected: {present['channels_as_expected'].all()}",
        "",
        "## Missing files",
        "",
    ]
    miss = manifest[~manifest["exists"]]
    lines += (
        [f"- {r.subject} {r.suffix}" for r in miss.itertuples()]
        if len(miss)
        else ["- none"]
    )

    mi = present[present["kind"] != "baseline"]
    bad_trials = mi[mi["n_trials"] != 40]
    lines += [
        "",
        "## Trial structure (non-baseline runs)",
        "",
        f"- runs with exactly 40 cues: {(mi['n_trials'] == 40).sum()} / {len(mi)}",
        f"- class balance 20/20 everywhere: "
        f"{((mi['n_cue_left'] == 20) & (mi['n_cue_right'] == 20)).all()}",
    ]
    if len(bad_trials):
        lines += [f"- ANOMALY {r.subject} {r.suffix}: {r.n_trials} cues" for r in bad_trials.itertuples()]

    if len(timing_df):
        lines += [
            "",
            "## Timing (median seconds relative to cue)",
            "",
            f"- cue to feedback: {timing_df['cue_to_feedback_s'].median():.3f} "
            f"(spec 1.250)",
            f"- cue to trial end: {timing_df['cue_to_end_s'].median():.3f} "
            f"(spec 5.000)",
            f"- cross to cue: {timing_df['cross_to_cue_s'].median():.3f} (spec 3.000)",
            f"- runs deviating >10 ms from spec: "
            f"{(timing_df['cue_to_end_s'].sub(5.0).abs() > 0.01).sum()}",
        ]

    lines += [
        "",
        "## File anomalies",
        "",
    ]
    lines += (
        [f"- {r.subject}: {r.issue} ({r.detail})" for r in anomalies.itertuples()]
        if len(anomalies)
        else ["- none"]
    )

    lines += [
        "",
        "## Baseline duplication check",
        "",
        f"- files hashed: {len(dupes)}",
        f"- byte-identical duplicates found: {bool(dupes['is_duplicate'].any())}",
        "",
    ]

    (DATA_INTERIM / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"\nwrote {DATA_INTERIM / 'manifest.csv'}")


if __name__ == "__main__":
    main()
