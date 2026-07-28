"""Dual-stream preprocessing for MI and state analysis.

The two streams have deliberately different operations:
* MI stream: EOG-regressed, 8--30 Hz, no average reference, [0.5, 4.5] s.
* state stream: common-average reference, 45 Hz low pass, EOG retained,
  [-3.0, -0.5] s.

The state stream never uses the MI window, avoiding the circular inference
that would result from defining state with the signal tested for warping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from src.data.gdf_io import CUE_CODES, EEG_CHANNELS, EMG_CHANNELS, EOG_CHANNELS, read_gdf
from src.paths import MI_BAND, MI_WINDOW, SFREQ_TARGET, STATE_LOWPASS, STATE_WINDOW


@dataclass
class ProcessedRun:
    mi: np.ndarray
    state_eeg: np.ndarray
    state_eog: np.ndarray
    emg: np.ndarray
    trial_table: pd.DataFrame
    bad_channels: list[str]


def _resample_decimate(data: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased decimation through scipy's zero-phase polyphase method."""
    from scipy.signal import resample_poly

    return resample_poly(data, up=1, down=factor, axis=-1).astype(np.float32)


def _robust_bad_channels(eeg: np.ndarray, sfreq: float) -> list[int]:
    """Conservative automated bad-channel detector.

    It only identifies flatlines and extreme robust outliers.  An electrode
    that merely has a different alpha amplitude must remain in the data.
    """
    variance = np.var(eeg, axis=1)
    med = np.median(np.log(variance + np.finfo(float).tiny))
    mad = np.median(np.abs(np.log(variance + np.finfo(float).tiny) - med)) + 1e-12
    z_var = 0.6745 * (np.log(variance + np.finfo(float).tiny) - med) / mad

    sos = butter(4, [35, min(90, sfreq / 2 - 1)], btype="bandpass", fs=sfreq, output="sos")
    high = sosfiltfilt(sos, eeg, axis=1)
    high_ratio = np.var(high, axis=1) / (variance + 1e-12)
    h_med = np.median(high_ratio)
    h_mad = np.median(np.abs(high_ratio - h_med)) + 1e-12
    z_high = 0.6745 * (high_ratio - h_med) / h_mad

    # A high-frequency ratio can flag almost an entire sensorimotor montage
    # during a genuinely noisy run; interpolation is not credible in that
    # case. Restrict automatic replacement to unambiguous flatlines or an
    # extreme variance outlier, and retain the high-frequency diagnostic for
    # later QC rather than dropping a whole online run.
    candidates = np.flatnonzero((np.abs(z_var) > 8) | (variance < 1e-6)).tolist()
    return candidates[:3]


def _regress_eog(eeg: np.ndarray, eog: np.ndarray) -> np.ndarray:
    """Residualise EOG from EEG with an intercept, preserving EEG units."""
    design = np.vstack([np.ones(eog.shape[1]), eog]).T
    beta, *_ = np.linalg.lstsq(design, eeg.T, rcond=None)
    # Retain intercept but remove the three EOG components only.
    fitted_eog = (design[:, 1:] @ beta[1:]).T
    return eeg - fitted_eog


def _interpolate_bad(eeg: np.ndarray, bad: list[int]) -> np.ndarray:
    """Spatially conservative fallback interpolation.

    The protocol has a dense sensorimotor grid but not a full-head montage.
    A stable median of valid channels is safer than extrapolating from an
    assumed geometry.  The exact channels are logged and all inferential
    analyses include a no-interpolation sensitivity analysis.
    """
    if not bad:
        return eeg
    good = np.setdiff1d(np.arange(eeg.shape[0]), np.asarray(bad, dtype=int))
    if len(good) < 10:
        raise ValueError("fewer than 10 valid EEG channels; cannot interpolate")
    out = eeg.copy()
    out[bad] = np.median(eeg[good], axis=0)
    return out


def _epoch(data: np.ndarray, cues: np.ndarray, window: tuple[float, float], sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    start_offset = int(round(window[0] * sfreq))
    stop_offset = int(round(window[1] * sfreq))
    width = stop_offset - start_offset
    valid: list[int] = []
    epochs: list[np.ndarray] = []
    for i, cue in enumerate(cues):
        start, stop = int(cue + start_offset), int(cue + stop_offset)
        if start >= 0 and stop <= data.shape[-1]:
            epochs.append(data[:, start:stop])
            valid.append(i)
    if not epochs:
        return np.empty((0, data.shape[0], width), dtype=np.float32), np.asarray(valid, dtype=int)
    return np.stack(epochs).astype(np.float32), np.asarray(valid, dtype=int)


def _trial_reject_flags(mi: np.ndarray, state_eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mark (but never drop) trials with robust amplitude artifacts."""
    ptp = np.ptp(mi, axis=-1).max(axis=1)
    state_ptp = np.ptp(state_eeg, axis=-1).max(axis=1)
    score = np.log(np.maximum(ptp, 1e-6)) + np.log(np.maximum(state_ptp, 1e-6))
    med = np.median(score)
    mad = np.median(np.abs(score - med)) + 1e-12
    z = 0.6745 * (score - med) / mad
    return z > 5.0, z


def preprocess_run(
    path: str,
    subject: str,
    run_index: int,
    comment_bad_channels: list[str] | None = None,
    *,
    mi_eog_regression: bool = True,
    state_window: tuple[float, float] = STATE_WINDOW,
    matched_erd_stream: bool = False,
) -> ProcessedRun:
    """Read and transform one GDF into aligned MI/state trial arrays."""
    picks = EEG_CHANNELS + EOG_CHANNELS + EMG_CHANNELS
    raw, header, events = read_gdf(path, picks=picks, dtype=np.float64)
    if header.sfreq % SFREQ_TARGET:
        raise ValueError(f"{path}: {header.sfreq} not divisible by {SFREQ_TARGET}")

    eeg = raw[: len(EEG_CHANNELS)]
    eog = raw[len(EEG_CHANNELS) : len(EEG_CHANNELS) + len(EOG_CHANNELS)]
    emg = raw[-len(EMG_CHANNELS) :]
    automatic_bad = _robust_bad_channels(eeg, header.sfreq)
    comment_bad = [
        EEG_CHANNELS.index(ch) for ch in (comment_bad_channels or []) if ch in EEG_CHANNELS
    ]
    bad = sorted(set(automatic_bad) | set(comment_bad))
    eeg_clean = _interpolate_bad(eeg, bad)

    # MI stream: residualise eye activity before the mu/beta filter.  No CAR:
    # covariance methods retain full rank and are reference-invariant.
    mi_signal = _regress_eog(eeg_clean, eog) if mi_eog_regression else eeg_clean
    mi_sos = butter(4, MI_BAND, btype="bandpass", fs=header.sfreq, output="sos")
    mi_signal = sosfiltfilt(mi_sos, mi_signal, axis=1)

    if matched_erd_stream:
        # Matched physiological stream: identical EOG regression + 8-30 Hz
        # filtering for both the pre-cue baseline and the MI window.
        state_eeg = mi_signal.copy()
        state_eog = sosfiltfilt(mi_sos, eog, axis=1)
    else:
        # State stream: CAR is appropriate because its spectral topographies are
        # interpreted explicitly. EOG is kept as state/nuisance measurement.
        state_eeg = eeg_clean - eeg_clean.mean(axis=0, keepdims=True)
        state_sos = butter(4, STATE_LOWPASS, btype="lowpass", fs=header.sfreq, output="sos")
        state_eeg = sosfiltfilt(state_sos, state_eeg, axis=1)
        state_eog = sosfiltfilt(state_sos, eog, axis=1)
    emg_sos = butter(4, [20, 45], btype="bandpass", fs=header.sfreq, output="sos")
    emg = sosfiltfilt(emg_sos, emg, axis=1)

    factor = int(round(header.sfreq / SFREQ_TARGET))
    mi_signal, state_eeg, state_eog, emg = [
        _resample_decimate(x, factor) for x in (mi_signal, state_eeg, state_eog, emg)
    ]
    cues = events[np.isin(events[:, 1], list(CUE_CODES))].copy()
    cues[:, 0] //= factor
    mi, mi_ix = _epoch(mi_signal, cues[:, 0], MI_WINDOW, SFREQ_TARGET)
    state, state_ix = _epoch(state_eeg, cues[:, 0], state_window, SFREQ_TARGET)
    eog_epoch, eog_ix = _epoch(state_eog, cues[:, 0], state_window, SFREQ_TARGET)
    emg_epoch, emg_ix = _epoch(emg, cues[:, 0], MI_WINDOW, SFREQ_TARGET)
    common = np.intersect1d(np.intersect1d(mi_ix, state_ix), np.intersect1d(eog_ix, emg_ix))
    # Input trial order is preserved by the GDF event list.
    mi = mi[np.searchsorted(mi_ix, common)]
    state = state[np.searchsorted(state_ix, common)]
    eog_epoch = eog_epoch[np.searchsorted(eog_ix, common)]
    emg_epoch = emg_epoch[np.searchsorted(emg_ix, common)]
    reject, artifact_z = _trial_reject_flags(mi, state)

    source = cues[common]
    table = pd.DataFrame(
        {
            "subject": subject,
            "dataset": subject[0],
            "run_index": run_index,
            "trial": np.arange(1, len(common) + 1),
            "source_trial": common + 1,
            "cue_sample_128hz": source[:, 0],
            "target_code": source[:, 1],
            "target": [CUE_CODES[int(c)] for c in source[:, 1]],
            "reject_flag": reject,
            "artifact_robust_z": artifact_z,
        }
    )
    return ProcessedRun(mi, state, eog_epoch, emg_epoch, table, [EEG_CHANNELS[i] for i in bad])
