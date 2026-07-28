"""Replay the subject-specific OpenViBE online CSP/LDA decoder.

The OpenViBE graph is:
    band-pass -> reject EOG/EMG -> CSP(27 -> 6) -> square -> 1 s moving
    average (62.5 ms hop) -> log(1+x) -> native two-class LDA.

The raw recordings do not contain the online classifier output.  This module
reconstructs it while retaining both the continuous LDA margin and the
per-window output, which avoids collapsing 40 trials into one noisy accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.data.gdf_io import CUE_CODES, EEG_CHANNELS, read_gdf
from src.paths import SFREQ_RAW, T_FEEDBACK_END, T_FEEDBACK_START, gdf_path, subject_dir


@dataclass(frozen=True)
class OnlineDecoder:
    """Subject-specific filter/CSP/LDA parameters saved by OpenViBE."""

    low_hz: float
    high_hz: float
    csp: np.ndarray  # (6, 27)
    weights: np.ndarray  # (2, 6)
    bias: np.ndarray  # (2,)


def _numbers(text: str) -> np.ndarray:
    return np.fromstring(text.strip(), sep=" ")


def _setting_values(path: Path) -> list[str]:
    return re.findall(r"<SettingValue>(.*?)</SettingValue>", path.read_text(encoding="latin1"), re.S)


def _first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"no {label} among: {[str(p) for p in paths]}")


def load_decoder(subject: str) -> OnlineDecoder:
    """Read the subject's band, CSP matrix, and native LDA XML files."""
    folder = subject_dir(subject)
    band_file = _first_existing(
        [
            folder / f"frequency-band-selected_{subject}.xml",
            folder / f"frequency-band-selected-{subject}.xml",
            *sorted(folder.glob("frequency-band-selected*.xml")),
        ],
        "frequency-band XML",
    )
    values = _setting_values(band_file)
    if len(values) < 5:
        raise ValueError(f"{band_file}: incomplete band settings")
    low_hz, high_hz = float(values[3]), float(values[4])

    csp_file = _first_existing(
        [
            folder / f"csp-spatial-filter_{subject}.xml",
            folder / f"csp-spatial-filter-{subject}.xml",
            *sorted(folder.glob("csp-spatial-filter*.xml")),
        ],
        "CSP XML",
    )
    csp_values = _setting_values(csp_file)
    if not csp_values:
        raise ValueError(f"{csp_file}: no CSP coefficients")
    raw_csp = _numbers(csp_values[0])
    if raw_csp.size != 6 * 27:
        raise ValueError(f"{csp_file}: expected 162 CSP coefficients, got {raw_csp.size}")
    csp = raw_csp.reshape(6, 27)

    lda_file = _first_existing(
        [
            folder / f"classifier_{subject}.xml",
            folder / f"classifier-{subject}.xml",
            folder / f"mi-bci-classifier_{subject}.xml",
            *sorted(folder.glob("*classifier*.xml")),
        ],
        "classifier XML",
    )
    text = lda_file.read_text(encoding="latin1")
    weights = [_numbers(v) for v in re.findall(r"<Weights>(.*?)</Weights>", text, re.S)]
    bias = [float(v.strip()) for v in re.findall(r"<Bias>(.*?)</Bias>", text, re.S)]
    if len(weights) != 2 or any(w.size != 6 for w in weights) or len(bias) != 2:
        raise ValueError(f"{lda_file}: malformed two-class LDA")
    return OnlineDecoder(low_hz, high_hz, csp, np.stack(weights), np.asarray(bias))


def _window_starts(start: int, stop: int, width: int, hop: int) -> np.ndarray:
    """Starts for complete moving windows in [start, stop)."""
    latest = stop - width
    if latest < start:
        return np.empty(0, dtype=int)
    return np.arange(start, latest + 1, hop, dtype=int)


def replay_run(subject: str, suffix: str) -> tuple[list[dict], dict]:
    """Replay one run and return trial outcomes plus implementation diagnostics."""
    path = gdf_path(subject, suffix)
    data, header, events = read_gdf(path, picks=EEG_CHANNELS, dtype=np.float64)
    if header.sfreq != SFREQ_RAW:
        raise ValueError(f"{path}: unexpected sample rate {header.sfreq}")
    decoder = load_decoder(subject)

    sos = butter(5, [decoder.low_hz, decoder.high_hz], btype="bandpass", fs=header.sfreq, output="sos")
    filtered = sosfiltfilt(sos, data, axis=1)
    csp = decoder.csp @ filtered
    power = csp * csp

    cue_events = events[np.isin(events[:, 1], list(CUE_CODES))]
    width = int(round(header.sfreq))
    hop = int(round(header.sfreq / 16))  # 62.5 ms = 32 samples
    rows: list[dict] = []

    for trial, (cue, code) in enumerate(cue_events, start=1):
        start = int(cue + round(T_FEEDBACK_START * header.sfreq))
        stop = int(cue + round(T_FEEDBACK_END * header.sfreq))
        starts = _window_starts(start, stop, width, hop)
        if not len(starts):
            continue
        # Moving mean of squared CSP output; adding one matches OpenViBE's
        # Simple DSP `log(1+x)` rather than the frequently assumed log-power.
        feats = np.stack([power[:, s : s + width].mean(axis=1) for s in starts])
        feats = np.log1p(feats)
        scores = feats @ decoder.weights.T + decoder.bias
        # Native LDA emits the class with highest score. Difference is a
        # signed continuous decision value: right minus left.
        margins = scores[:, 1] - scores[:, 0]
        mean_margin = float(np.mean(margins))
        pred = "right" if mean_margin > 0 else "left"
        target = CUE_CODES[int(code)]
        rows.append(
            {
                "subject": subject,
                "dataset": subject[0],
                "run_index": int(suffix[1]),
                "suffix": suffix,
                "trial": trial,
                "cue_sample": int(cue),
                "target": target,
                "target_code": int(code),
                "n_windows": int(len(starts)),
                "margin": mean_margin,
                "margin_sd": float(np.std(margins, ddof=1)) if len(margins) > 1 else 0.0,
                "predicted": pred,
                "correct": int(pred == target),
            }
        )

    diagnostics = {
        "subject": subject,
        "suffix": suffix,
        "band_low_hz": decoder.low_hz,
        "band_high_hz": decoder.high_hz,
        "n_trials": len(rows),
        "window_width_samples": width,
        "window_hop_samples": hop,
    }
    return rows, diagnostics
