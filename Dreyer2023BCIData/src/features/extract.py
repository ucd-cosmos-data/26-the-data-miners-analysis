"""Physiology-informed pre-cue state features and MI covariances."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.covariance import LedoitWolf

from src.data.gdf_io import EEG_CHANNELS


def _indices(names: list[str]) -> list[int]:
    return [EEG_CHANNELS.index(name) for name in names]


POSTERIOR = _indices(["P3", "Pz", "P4", "CP1", "CP3", "CP5", "CP2", "CP4", "CP6"])
FRONTAL = _indices(["Fz", "FCz"])
SENSORIMOTOR = _indices(["C3", "C4", "CP3", "CP4"])


def _bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    band = (freqs >= low) & (freqs <= high)
    return np.trapezoid(psd[..., band], freqs[band], axis=-1)


def _aperiodic_fit(freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust log-log slope/intercept excluding canonical oscillatory bands."""
    keep = (freqs >= 2) & (freqs <= 40) & ~(((freqs >= 4) & (freqs <= 14)) | ((freqs >= 18) & (freqs <= 30)))
    x = np.log10(freqs[keep])
    # OLS is deterministic and adequate as a feature; it is intentionally not
    # a claim that every individual spectrum has a perfect 1/f form.
    X = np.column_stack([np.ones_like(x), x])
    y = np.log10(psd[:, keep] + np.finfo(float).tiny)
    coef = np.linalg.pinv(X) @ y.T
    return coef[0], coef[1]


def state_features(
    state_eeg: np.ndarray,
    state_eog: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
) -> pd.DataFrame:
    """Return one state-feature row per pre-cue trial."""
    # state_eeg: (trials, channels, samples)
    freqs, psd = welch(state_eeg, fs=sfreq, nperseg=state_eeg.shape[-1], axis=-1)
    posterior_alpha = np.log10(_bandpower(freqs, psd[:, POSTERIOR].mean(axis=1), 8, 12) + 1e-12)
    frontal_theta = np.log10(_bandpower(freqs, psd[:, FRONTAL].mean(axis=1), 4, 7) + 1e-12)
    sm_mu = np.log10(_bandpower(freqs, psd[:, SENSORIMOTOR].mean(axis=1), 8, 13) + 1e-12)
    sm_beta = np.log10(_bandpower(freqs, psd[:, SENSORIMOTOR].mean(axis=1), 13, 30) + 1e-12)
    offset, exponent = _aperiodic_fit(freqs, psd.mean(axis=1))

    peak_region = (freqs >= 7) & (freqs <= 14)
    alpha_psd = psd[:, POSTERIOR].mean(axis=1)[:, peak_region]
    alpha_freqs = freqs[peak_region]
    alpha_peak = alpha_freqs[np.argmax(alpha_psd, axis=1)]

    eog_freq, eog_psd = welch(state_eog, fs=sfreq, nperseg=state_eog.shape[-1], axis=-1)
    slow_eye = np.log10(_bandpower(eog_freq, eog_psd.mean(axis=1), 0.1, 1.0) + 1e-12)
    # Robust high-amplitude transient proxy: EOG peak-to-peak in pre-cue epoch.
    blink_amplitude = np.ptp(state_eog, axis=-1).mean(axis=1)
    # A count proxy is more robust than peak counting on 2.5 s snippets.
    blink_z = (state_eog.mean(axis=1) - state_eog.mean(axis=(1, 2), keepdims=True).squeeze()[:, None])
    blink_events = np.array([
        len(find_peaks(np.abs(x), prominence=max(np.std(x) * 1.5, 1e-9))[0])
        for x in blink_z
    ])
    emg_rms = np.sqrt(np.mean(emg * emg, axis=(1, 2)))

    return pd.DataFrame(
        {
            "posterior_alpha_logpower": posterior_alpha,
            "frontal_midline_theta_logpower": frontal_theta,
            "aperiodic_offset": offset,
            "aperiodic_exponent": exponent,
            "alpha_peak_hz": alpha_peak,
            "slow_eye_logpower": slow_eye,
            "blink_amplitude": blink_amplitude,
            "blink_event_count": blink_events,
            "emg_rms_20_45hz": emg_rms,
            # Explicitly separate these from the primary state latent.
            "sensorimotor_idle_mu_logpower": sm_mu,
            "sensorimotor_idle_beta_logpower": sm_beta,
        }
    )


def ledoit_wolf_covariances(mi: np.ndarray) -> np.ndarray:
    """SPD covariance per MI epoch, suitable for Riemannian tangent mapping."""
    covs = np.empty((len(mi), mi.shape[1], mi.shape[1]), dtype=np.float32)
    for i, epoch in enumerate(mi):
        covs[i] = LedoitWolf().fit(epoch.T).covariance_.astype(np.float32)
    return covs
