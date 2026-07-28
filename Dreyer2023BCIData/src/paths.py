"""Canonical project paths and dataset constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "BCI Database"
SIGNALS = DATA_RAW / "Signals"
PERFORMANCES_CSV = DATA_RAW / "Perfomances.csv"
OV_SCENARIOS = DATA_RAW / "OV_Scenarios"

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RUNS = RESULTS / "runs"
DOCS = PROJECT_ROOT / "docs"

for _d in (DATA_INTERIM, DATA_PROCESSED, FIGURES, TABLES, RUNS, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = {"A": "DATA A", "B": "DATA B", "C": "DATA C"}

RUN_KINDS = {
    "CE_baseline": "baseline",
    "OE_baseline": "baseline",
    "R1_acquisition": "calibration",
    "R2_acquisition": "calibration",
    "R3_onlineT": "online",
    "R4_onlineT": "online",
    "R5_onlineT": "online",
    "R6_onlineT": "online",
}
EXPECTED_SUFFIXES = list(RUN_KINDS)
MI_RUNS = [s for s, k in RUN_KINDS.items() if k != "baseline"]
ONLINE_RUNS = [s for s, k in RUN_KINDS.items() if k == "online"]

# Timing of one trial relative to cue onset, from the OpenViBE Graz scenario.
T_CROSS = -3.0
T_BEEP = -1.0
T_CUE = 0.0
T_FEEDBACK_START = 1.25
T_FEEDBACK_END = 5.0

# Analysis windows (seconds relative to cue onset).
MI_WINDOW = (0.5, 4.5)
STATE_WINDOW = (-3.0, -0.5)

SFREQ_RAW = 512.0
SFREQ_TARGET = 128.0
MI_BAND = (8.0, 30.0)
STATE_LOWPASS = 45.0

N_TRIALS_PER_RUN = 40


def subject_dir(subject: str) -> Path:
    return SIGNALS / DATASETS[subject[0]] / subject


def gdf_path(subject: str, suffix: str) -> Path:
    return subject_dir(subject) / f"{subject}_{suffix}.gdf"


def all_subjects() -> list[str]:
    """Subject IDs in dataset then numeric order."""
    subjects: list[str] = []
    for key, folder in DATASETS.items():
        found = [p.name for p in (SIGNALS / folder).iterdir() if p.is_dir()]
        subjects.extend(sorted(found, key=lambda s: int(s[1:])))
    return subjects
