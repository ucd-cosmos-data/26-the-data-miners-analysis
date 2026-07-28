"""S1: emit harmonised subject/run tables, comment flags and a data dictionary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.comments import build_comment_flags  # noqa: E402
from src.data.metadata import (  # noqa: E402
    PRE_LABEL_HARMONISATION,
    build_runs,
    load_subjects,
)
from src.paths import DATA_INTERIM, DATA_PROCESSED, DOCS  # noqa: E402


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.lower().eq("true")


def main() -> None:
    subjects = load_subjects()
    manifest = pd.read_csv(DATA_INTERIM / "manifest.csv")
    manifest["exists"] = _bool_col(manifest, "exists")

    runs = build_runs(subjects, manifest)
    flags = build_comment_flags(subjects)

    subjects.to_parquet(DATA_PROCESSED / "subjects.parquet", index=False)
    runs.to_parquet(DATA_PROCESSED / "runs.parquet", index=False)
    flags.to_csv(DATA_INTERIM / "comment_flags.csv", index=False)

    n_flagged_ch = flags[flags["channel"] != ""].groupby("subject")["channel"].nunique()
    events = flags[flags["session_events"] != ""]

    lines = [
        "# Data dictionary and S1 harmonisation notes",
        "",
        "## Provenance",
        "",
        "Source: `data/raw/BCI Database/Perfomances.csv`, three stacked blocks "
        "(datasets A, B, C) with semicolon separators and comma decimals.",
        "",
        "## Harmonisation decisions",
        "",
        "1. **Dataset B pre-session labels renamed.** B labels three columns "
        + ", ".join(f"`{k}` -> `{v}`" for k, v in PRE_LABEL_HARMONISATION.items())
        + ". They occupy identical column positions, derive from the same "
        "questionnaire document, and share value granularity (multiples of 4, "
        "10 and 20, implying 5-, 2- and 1-item Likert composites). Rows carry "
        "`pre_labels_harmonised = True`. **This is an inference, not a "
        "documented fact.**",
        "",
        "2. **Sleep hours.** Dataset A uses comma decimals; dataset B mixes "
        "decimal hours with hours-and-minutes. A value with exactly two "
        "decimals forming a valid minute count is read as minutes and flagged "
        "in `pre_sleep_last_night_ambiguous`.",
        "",
        "3. **Motivation delta is not interpretable.** `pre_motivation` has "
        "1-item granularity and `post_motivation` has 7-item granularity; they "
        "correlate at r = 0.10 against 0.49 and 0.50 for mood and "
        "mindfulness. The column is named "
        "`delta_motivation_uninterpretable` so it cannot be used by accident.",
        "",
        "4. **Item-level responses are unavailable.** Only composites are "
        "distributed, so composites cannot be recomputed.",
        "",
        "## Tables",
        "",
        f"- `data/processed/subjects.parquet`: {len(subjects)} rows x "
        f"{subjects.shape[1]} columns, one per participant.",
        f"- `data/processed/runs.parquet`: {len(runs)} rows, one per "
        "(subject, motor-imagery run).",
        f"- `data/interim/comment_flags.csv`: {len(flags)} rows expanded from "
        "the experimenter COMMENTS column.",
        "",
        "## Key columns",
        "",
        "- `tacc_run3..6`: online accuracy in percent, feedback runs only. "
        "Calibration runs R1/R2 have no online score.",
        "- `pre_*` / `post_*`: 0-100 rescaled questionnaire composites.",
        "- `delta_mood`, `delta_mindfulness`: comparable pre-post changes.",
        "- `pf16_*`: 16PF5 personality factors. `LEARNING_STYLE`: index of "
        "learning style. `mental_rotation_*`: spatial ability.",
        "",
        "## Missingness (non-text columns with any missing value)",
        "",
    ]

    miss = subjects.isna().sum()
    miss = miss[(miss > 0) & (~miss.index.str.contains("text|comments"))]
    lines += [
        f"- `{c}`: {int(n)} of {len(subjects)}" for c, n in miss.sort_values(ascending=False).items()
    ] or ["- none"]

    lines += [
        "",
        "## Experimenter comment flags",
        "",
        f"- subjects with at least one flagged channel: {len(n_flagged_ch)}",
        f"- total (subject, run, channel) flags: {(flags['channel'] != '').sum()}",
        f"- subjects with noted session events: {events['subject'].nunique()}",
        "",
    ]
    for tag, grp in events.groupby("session_events"):
        lines.append(f"  - `{tag}`: {sorted(grp['subject'].unique())}")

    (DOCS / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"subjects {subjects.shape}, runs {runs.shape}, flags {flags.shape}")
    print(f"datasets: {subjects['dataset'].value_counts().to_dict()}")
    print(f"harmonised pre labels: {int(subjects['pre_labels_harmonised'].sum())}")
    amb = int(subjects["pre_sleep_last_night_ambiguous"].sum())
    print(f"ambiguous sleep values: {amb}")
    print(f"tacc present: {runs['tacc'].notna().sum()} of {len(runs)} runs")
    print(f"comment channel flags: {(flags['channel'] != '').sum()}")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
