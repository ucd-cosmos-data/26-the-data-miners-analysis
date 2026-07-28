"""Regression tests for raw-data assumptions that would otherwise leak silently."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdf_io import CUE_CODES, FILE_CHANNEL_ORDER, read_events, read_header  # noqa: E402
from src.paths import gdf_path  # noqa: E402


def test_a1_event_protocol() -> None:
    path = gdf_path("A1", "R3_onlineT")
    header = read_header(path)
    events = read_events(path, header)
    cues = events[np.isin(events[:, 1], list(CUE_CODES))]
    assert header.sfreq == 512
    assert header.ch_names == FILE_CHANNEL_ORDER
    assert len(cues) == 40
    assert (cues[:, 1] == 769).sum() == 20
    assert (cues[:, 1] == 770).sum() == 20


def test_a40_truncation_is_explicit() -> None:
    path = gdf_path("A40", "R3_onlineT")
    header = read_header(path)
    cues = read_events(path, header)
    cues = cues[np.isin(cues[:, 1], list(CUE_CODES))]
    assert len(cues) == 32
