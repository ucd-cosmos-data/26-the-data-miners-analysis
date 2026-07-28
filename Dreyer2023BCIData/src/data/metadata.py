"""S1: parse and harmonise Perfomances.csv.

The file holds three stacked blocks (datasets A, B, C) with slightly different
headers, semicolon separators, comma decimals, and several free-text fields
that contain embedded newlines. Dataset B labels three pre-session composites
differently from A and C; see PRE_LABEL_HARMONISATION for the evidence that
they are the same variables.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from src.paths import PERFORMANCES_CSV

# Dataset B names the same three columns differently. They occupy identical
# positions, share the questionnaire document, and show the same value
# granularity (multiples of 4, 10 and 20, implying 5-, 2- and 1-item Likert
# composites), so they are treated as the same variables and flagged.
PRE_LABEL_HARMONISATION = {
    "PRE_Nervousness": "PRE_Mood",
    "PRE_Awakening": "PRE_Mindfulness",
    "PRE_Concentration": "PRE_Motivation",
}

_HEADER_ALIASES = {
    "COMMENTAIRES": "COMMENTS",
    "Symptoms_TXT": "Symptoms",
}

_RENAME = {
    "SUJ_ID": "subject",
    "SUJ_gender": "subject_gender",
    "EXP_gender": "experimenter_gender",
    "COMMENTS": "comments",
    "Birth_year": "birth_year",
    "Vision": "vision",
    "Vision_assistance": "vision_assistance",
    "Symptoms": "symptoms",
    "Level of study": "level_of_study",
    "Level_knowledge neuro": "neuro_knowledge",
    "Meditation practice": "meditation_practice",
    "Laterality answered": "laterality",
    "Manual activity": "manual_activity",
    "Manual activity TXT": "manual_activity_text",
    "score": "mental_rotation_score",
    "time_1": "mental_rotation_time1",
    "time_2": "mental_rotation_time2",
    "PRE_Mood": "pre_mood",
    "PRE_Mindfulness": "pre_mindfulness",
    "PRE_Motivation": "pre_motivation",
    "PRE_Hours_sleep_last_night": "pre_sleep_last_night_raw",
    "PRE_Usual_sleep": "pre_sleep_usual_raw",
    "PRE_Level_of_alertness": "pre_alertness",
    "PRE_Stimulant_doses_12h": "pre_stimulant_12h",
    "PRE_Stimulant_doses_2h": "pre_stimulant_2h",
    "PRE_Stim_normal": "pre_stimulant_normal",
    "PRE_Tabacco": "pre_tobacco",
    "PRE_Tabacco_normal": "pre_tobacco_normal",
    "PRE_Alcohol": "pre_alcohol",
    "PRE_Last_meal": "pre_last_meal",
    "PRE_Last_pills": "pre_last_pills",
    "PRE_Pills_TXT": "pre_pills_text",
    "POST_Mood": "post_mood",
    "POST_Mindfulness": "post_mindfulness",
    "POST_Motivation": "post_motivation",
    "POST_Cognitive load": "post_cognitive_load",
    "POST_Agentivity": "post_agentivity",
    "POST_Expectations_filled": "post_expectations",
    "Interrogation": "interrogation",
}

LEARNING_STYLE = [
    "active", "reflexive", "sensory", "intuitive",
    "visual", "verbal", "sequential", "global",
]
PF16 = [
    "A", "B", "C_", "E", "F", "G", "H", "I", "L", "M", "N", "O",
    "Q1", "Q2", "Q3", "Q4", "IM", "EX", "AX", "TM", "IN", "SC",
]

NUMERIC_COLS = (
    [
        "subject_gender", "experimenter_gender", "birth_year", "vision",
        "vision_assistance", "level_of_study", "neuro_knowledge",
        "meditation_practice", "laterality", "manual_activity",
        "mental_rotation_score", "mental_rotation_time1",
        "mental_rotation_time2",
        "pre_mood", "pre_mindfulness", "pre_motivation", "pre_alertness",
        "pre_stimulant_12h", "pre_stimulant_2h", "pre_stimulant_normal",
        "pre_tobacco", "pre_tobacco_normal", "pre_alcohol", "pre_last_meal",
        "pre_last_pills",
        "post_mood", "post_mindfulness", "post_motivation",
        "post_cognitive_load", "post_agentivity", "post_expectations",
        "interrogation",
    ]
    + LEARNING_STYLE
    + [f"pf16_{c}" for c in PF16]
)

_TIME_LIKE = re.compile(r"^\s*(\d{1,2})\.([0-5]\d)\s*$")


def _to_num(series: pd.Series) -> pd.Series:
    """Coerce a string column that may use comma or dot decimals."""
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_sleep_hours(value: object) -> tuple[float, bool]:
    """Return (hours, ambiguous).

    Dataset A uses comma decimals ("7,5"). Dataset B mixes decimal hours
    ("8.66", "7.5") with hours-and-minutes ("8.30" meaning 8 h 30). A value
    with exactly two decimals whose last pair is a valid minute count is read
    as minutes and flagged, since "8.5" would be the decimal spelling.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, False
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "nan":
        return np.nan, False
    match = _TIME_LIKE.match(text)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        return hours + minutes / 60.0, True
    try:
        return float(text), False
    except ValueError:
        return np.nan, False


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """Return (dataset_letter, csv_text) for each stacked block."""
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if ln.startswith("DATA ")]
    headers = [i for i, ln in enumerate(lines) if ln.startswith("SUJ_ID")]
    if len(headers) != 3:
        raise ValueError(f"expected 3 header rows, found {len(headers)}")

    blocks = []
    for k, head in enumerate(headers):
        stop = starts[k + 1] if k + 1 < len(starts) else len(lines)
        letter = lines[starts[k]].strip()[5]
        blocks.append((letter, "\n".join(lines[head:stop])))
    return blocks


def load_subjects() -> pd.DataFrame:
    text = PERFORMANCES_CSV.read_text(encoding="latin1")
    frames = []
    for letter, block in _split_blocks(text):
        df = pd.read_csv(
            io.StringIO(block), sep=";", dtype=str, engine="python",
            on_bad_lines="skip",
        )
        df = df.rename(columns=_HEADER_ALIASES)
        harmonised = [c for c in PRE_LABEL_HARMONISATION if c in df.columns]
        df = df.rename(columns=PRE_LABEL_HARMONISATION)
        df["dataset"] = letter
        df["pre_labels_harmonised"] = bool(harmonised)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["SUJ_ID"].astype(str).str.match(r"^[ABC]\d+$", na=False)]

    df = raw.rename(columns=_RENAME)
    df = df.rename(columns={c: f"pf16_{c}" for c in PF16 if c in df.columns})

    perf_cols = []
    for k in (3, 4, 5, 6):
        col = f"Perf_RUN_{k}"
        df[f"tacc_run{k}"] = _to_num(df[col])
        perf_cols.append(f"tacc_run{k}")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _to_num(df[col])

    for src, dst in (
        ("pre_sleep_last_night_raw", "pre_sleep_last_night"),
        ("pre_sleep_usual_raw", "pre_sleep_usual"),
    ):
        parsed = df[src].map(parse_sleep_hours)
        df[dst] = [p[0] for p in parsed]
        df[f"{dst}_ambiguous"] = [p[1] for p in parsed]

    # Only Mood and Mindfulness use matching item sets pre and post; the
    # Motivation composites correlate at r = 0.10 and are not comparable.
    df["delta_mood"] = df["post_mood"] - df["pre_mood"]
    df["delta_mindfulness"] = df["post_mindfulness"] - df["pre_mindfulness"]
    df["delta_motivation_uninterpretable"] = (
        df["post_motivation"] - df["pre_motivation"]
    )

    df["tacc_mean"] = df[perf_cols].mean(axis=1)
    df["tacc_sd"] = df[perf_cols].std(axis=1)
    df["tacc_range"] = df[perf_cols].max(axis=1) - df[perf_cols].min(axis=1)
    df["subject_num"] = df["subject"].str[1:].astype(int)

    keep = (
        ["subject", "subject_num", "dataset", "pre_labels_harmonised"]
        + [c for c in _RENAME.values() if c in df.columns and c != "subject"]
        + [c for c in df.columns if c.startswith("pf16_")]
        + LEARNING_STYLE
        + [
            "pre_sleep_last_night", "pre_sleep_last_night_ambiguous",
            "pre_sleep_usual", "pre_sleep_usual_ambiguous",
            "delta_mood", "delta_mindfulness",
            "delta_motivation_uninterpretable",
            "tacc_run3", "tacc_run4", "tacc_run5", "tacc_run6",
            "tacc_mean", "tacc_sd", "tacc_range",
        ]
    )
    seen: set[str] = set()
    ordered = [c for c in keep if c in df.columns and not (c in seen or seen.add(c))]
    return df[ordered].sort_values(["dataset", "subject_num"]).reset_index(drop=True)


def build_runs(subjects: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, MI run) with online accuracy where available."""
    mi = manifest[manifest["kind"].isin(["calibration", "online"])].copy()
    mi["run_index"] = mi["run_index"].astype(int)

    rows = []
    tacc = subjects.set_index("subject")
    for r in mi.itertuples():
        acc = np.nan
        if r.run_index >= 3 and r.subject in tacc.index:
            acc = tacc.loc[r.subject, f"tacc_run{r.run_index}"]
        rows.append(
            {
                "subject": r.subject,
                "dataset": r.subject[0],
                "run_index": r.run_index,
                "suffix": r.suffix,
                "kind": r.kind,
                "exists": bool(r.exists),
                "n_trials": getattr(r, "n_trials", np.nan),
                "duration_s": getattr(r, "duration_s", np.nan),
                "tacc": acc,
                "has_feedback": r.kind == "online",
            }
        )
    runs = pd.DataFrame(rows)
    runs["n_correct"] = np.round(runs["tacc"] / 100.0 * runs["n_trials"])
    return runs.sort_values(["dataset", "subject", "run_index"]).reset_index(drop=True)
