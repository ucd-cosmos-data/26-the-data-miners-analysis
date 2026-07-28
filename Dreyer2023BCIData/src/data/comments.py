"""Parse the free-text experimenter COMMENTS column into structured QC flags.

The experimenters recorded channel problems and session events per run, for
example "R6 : dozed off. Fz noisy from R2 to R6" or "BL : CP5 noisy". These
notes are the only human record of electrode quality, so they seed the
bad-channel list that automatic detection later extends.
"""

from __future__ import annotations

import re

import pandas as pd

from src.data.gdf_io import EEG_CHANNELS
from src.paths import EXPECTED_SUFFIXES

# Longest-first so that "FC5" is matched before "F5"-style prefixes.
_CHAN_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(EEG_CHANNELS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_RUN_PATTERN = re.compile(r"\bR([1-6])\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(r"from\s+R([1-6])\s+to\s+R([1-6])", re.IGNORECASE)
_BASELINE_PATTERN = re.compile(r"\b(BL|baseline)\b", re.IGNORECASE)

_NOISE_WORDS = ("nois", "npos", "artifact", "artefact", "bad", "hair", "scratch")
_EVENT_WORDS = {
    "dozed": "drowsiness",
    "asleep": "drowsiness",
    "sleep": "drowsiness",
    "telephone": "interruption",
    "phone": "interruption",
    "interup": "interruption",
    "interrup": "interruption",
    "came in": "interruption",
    "outside noise": "environment",
    "moved": "movement",
}

_CANONICAL = {c.lower(): c for c in EEG_CHANNELS}


def _runs_mentioned(text: str) -> list[str]:
    runs: set[str] = set()
    for lo, hi in _RANGE_PATTERN.findall(text):
        for n in range(int(lo), int(hi) + 1):
            runs.add(str(n))
    for n in _RUN_PATTERN.findall(text):
        runs.add(n)
    suffixes = [s for s in EXPECTED_SUFFIXES if s.startswith("R") and s[1] in runs]
    if _BASELINE_PATTERN.search(text):
        suffixes += ["CE_baseline", "OE_baseline"]
    return suffixes


def parse_comment(subject: str, text: str | float) -> list[dict]:
    """Return one row per (channel, run) flagged by a comment."""
    if not isinstance(text, str) or not text.strip():
        return []

    low = text.lower()
    has_noise = any(w in low for w in _NOISE_WORDS)
    events = sorted({tag for word, tag in _EVENT_WORDS.items() if word in low})
    channels = sorted({_CANONICAL[m.lower()] for m in _CHAN_PATTERN.findall(text)})
    runs = _runs_mentioned(text) or EXPECTED_SUFFIXES

    rows = []
    for ch in channels:
        for suffix in runs:
            rows.append(
                {
                    "subject": subject,
                    "suffix": suffix,
                    "channel": ch,
                    "has_noise_keyword": has_noise,
                    "session_events": ";".join(events),
                    "comment": text.strip(),
                }
            )
    if not channels and (has_noise or events):
        for suffix in runs:
            rows.append(
                {
                    "subject": subject,
                    "suffix": suffix,
                    "channel": "",
                    "has_noise_keyword": has_noise,
                    "session_events": ";".join(events),
                    "comment": text.strip(),
                }
            )
    return rows


def build_comment_flags(subjects: pd.DataFrame) -> pd.DataFrame:
    """Expand the COMMENTS column of the subject table into a flag table."""
    rows: list[dict] = []
    for subject, text in zip(subjects["subject"], subjects["comments"]):
        rows.extend(parse_comment(subject, text))
    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "suffix",
                "channel",
                "has_noise_keyword",
                "session_events",
                "comment",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates()
