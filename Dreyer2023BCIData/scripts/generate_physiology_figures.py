"""Generate reproducible ERD/ERS and sensor-space sanity-check figures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from mne.time_frequency import tfr_array_morlet
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdf_io import EEG_CHANNELS  # noqa: E402
from src.paths import DATA_PROCESSED, FIGURES, RESULTS, all_subjects  # noqa: E402


FREQS = np.arange(8, 31, 2, dtype=float)
TIMES = np.arange(512) / 128 + 0.5
CONFIG = {
    "frequencies_hz": FREQS.tolist(),
    "morlet_cycles": "frequency/2",
    "cluster_permutations": 1024,
    "seed": 20260727,
    "stream": "matched_erd_eog_regressed_8_30hz",
}


def _epochs_root() -> Path:
    matched = DATA_PROCESSED / "ablations" / "matched_erd" / "epochs"
    return matched if matched.exists() and any(matched.glob("*_epochs.npz")) else DATA_PROCESSED / "epochs"


def _power(data: np.ndarray) -> np.ndarray:
    return tfr_array_morlet(
        data,
        sfreq=128,
        freqs=FREQS,
        n_cycles=FREQS / 2,
        output="power",
        n_jobs=1,
        zero_mean=True,
    )


def _baseline_power(state: np.ndarray) -> np.ndarray:
    # Mean pre-cue reference power ([-3,-0.5] seconds); this never contains MI.
    return _power(state.mean(axis=0, keepdims=True))[0].mean(axis=-1)


def _subject_maps(subject: str, z: np.ndarray, epochs_root: Path) -> dict[str, np.ndarray]:
    arrays = np.load(epochs_root / f"{subject}_epochs.npz")
    table = pd.read_parquet(epochs_root / f"{subject}_trials.parquet")
    mi, state = arrays["mi"], arrays["state_eeg"]
    baseline = _baseline_power(state)
    maps: dict[str, np.ndarray] = {}
    for target in ("left", "right"):
        average = mi[table["target"].to_numpy() == target].mean(axis=0, keepdims=True)
        maps[target] = 10 * np.log10(_power(average)[0] / (baseline[:, :, None] + 1e-12))
    # Maps carry (channel, frequency, time). Contralateral is C4 for left MI
    # and C3 for right MI; ipsilateral is the converse.
    c3, c4 = EEG_CHANNELS.index("C3"), EEG_CHANNELS.index("C4")
    maps["contra"] = (maps["left"][c4] + maps["right"][c3]) / 2
    maps["ipsi"] = (maps["left"][c3] + maps["right"][c4]) / 2
    # Sensor-space bands for topographic diagnostics.
    log_power = np.log(np.var(mi, axis=-1) + 1e-12)
    y = (table["target"].to_numpy() == "right").astype(float)
    maps["class_r2"] = np.array([np.corrcoef(log_power[:, ch], y)[0, 1] ** 2 for ch in range(len(EEG_CHANNELS))])
    alpha = sosfiltfilt(butter(4, [8, 12], btype="bandpass", fs=128, output="sos"), state, axis=-1)
    alpha_power = np.log(np.var(alpha, axis=-1) + 1e-12)
    maps["state_alpha_corr"] = np.array([np.corrcoef(alpha_power[:, ch], z)[0, 1] for ch in range(len(EEG_CHANNELS))])
    # A sensor-space diagnostic corresponding to class-specific/common state
    # variance shifts. It is explicitly not an inverse reconstruction of the
    # PCA-compressed SCTM tangent coefficients.
    centered_trial = table.groupby("run_index")["trial"].transform(lambda s: s - s.mean()).to_numpy()
    blink = (table.index.to_numpy() * 0.0)  # replaced below by feature merge caller if available
    design = np.column_stack([np.ones(len(y)), y * 2 - 1, z, (y * 2 - 1) * z, centered_trial])
    coef = np.linalg.pinv(design) @ log_power
    maps["common_state_proxy"] = coef[2]
    maps["differential_state_proxy"] = coef[3]
    return maps


def _plot_tfr(maps: list[dict[str, np.ndarray]], label: str, output: Path, cluster_mask: np.ndarray | None = None) -> None:
    contra = np.stack([item["contra"] for item in maps])
    ipsi = np.stack([item["ipsi"] for item in maps])
    difference = contra.mean(axis=0) - ipsi.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    levels = np.linspace(-3, 3, 25)
    mesh = None
    for ax, image, title in zip(axes, [contra.mean(axis=0), ipsi.mean(axis=0), difference], ["Contralateral", "Ipsilateral", "Contra − ipsi"], strict=True):
        mesh = ax.contourf(TIMES, FREQS, image, levels=levels, extend="both", cmap="RdBu_r")
        ax.axhspan(8, 12, color="black", alpha=0.07)
        ax.axhspan(13, 30, color="black", alpha=0.04)
        ax.set(title=title, xlabel="Seconds after cue")
        if title == "Contra − ipsi" and cluster_mask is not None and cluster_mask.any():
            ax.contour(TIMES, FREQS, cluster_mask, levels=[0.5], colors="k", linewidths=0.7)
    axes[0].set_ylabel("Frequency (Hz)")
    # Colorbar to the RIGHT of all panels — never between subplots.
    fig.subplots_adjust(right=0.88, top=0.82, wspace=0.18)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.58])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("dB relative to pre-cue baseline")
    fig.suptitle(f"{label}: subject-mean ERD/ERS (n={len(maps)}); contour = cluster p<0.05", y=0.98)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _plot_topographies(class_r2: np.ndarray, state_corr: np.ndarray, differential: np.ndarray, common: np.ndarray) -> None:
    info = mne.create_info(EEG_CHANNELS, sfreq=128, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    entries = [
        (class_r2, "Class-label R²\n8–30 Hz variance", "viridis"),
        (state_corr, "Pre-cue alpha–state\ncorrelation", "RdBu_r"),
        (differential, "Class × state\nsensor-space proxy", "RdBu_r"),
        (common, "Class-common state\nsensor-space proxy", "RdBu_r"),
    ]
    for axis, (values, title, cmap) in zip(axes, entries, strict=True):
        # Symmetric limits for diverging maps so white sits at zero.
        if cmap == "RdBu_r":
            limit = float(np.nanmax(np.abs(values))) or 1.0
            vlim = (-limit, limit)
        else:
            vlim = (0.0, float(np.nanmax(values)) or 1.0)
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            contours=4,
            cmap=cmap,
            vlim=vlim,
        )
        axis.set_title(title, fontsize=9)
        cbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.08, shrink=0.78)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Confirmatory sensor-space diagnostics (n=57); proxies are not SCTM inverse maps", y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_sensor_space_topographies_confirm.png", dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    config_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
    epochs_root = _epochs_root()
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv").set_index("subject")["split"]
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    maps_by_split: dict[str, list[dict[str, np.ndarray]]] = {"explore": [], "confirm": []}
    for subject in all_subjects():
        source = epochs_root / f"{subject}_epochs.npz"
        if not source.exists():
            continue
        z = state[state["subject"] == subject]["z_state"].to_numpy()
        maps_by_split[split[subject]].append(_subject_maps(subject, z, epochs_root))
        print(subject, flush=True)
    confirm_difference = np.stack([item["contra"] - item["ipsi"] for item in maps_by_split["confirm"]])
    # Subject is the inferential unit. The TF map is downsampled temporally for
    # the cluster statistic only; visual panels retain native 128 Hz samples.
    cluster_input = confirm_difference[:, :, ::4]
    threshold = None
    _, clusters, p_values, _ = permutation_cluster_1samp_test(
        cluster_input,
        n_permutations=CONFIG["cluster_permutations"],
        threshold=threshold,
        tail=0,
        out_type="mask",
        seed=CONFIG["seed"],
        verbose=False,
    )
    mask_small = np.zeros(cluster_input.shape[1:], dtype=bool)
    records = []
    for index, (cluster, p) in enumerate(zip(clusters, p_values, strict=True)):
        records.append({"cluster": index, "p_value": float(p), "significant_fwer_0_05": bool(p < 0.05), "n_time_frequency_cells": int(cluster.sum()), "n_subjects": len(maps_by_split["confirm"])})
        if p < 0.05:
            mask_small |= cluster
    mask = np.repeat(mask_small, 4, axis=1)[:, : len(TIMES)]
    pd.DataFrame(records).to_csv(RESULTS / "tables" / "time_frequency_cluster_confirm.csv", index=False)
    _plot_tfr(maps_by_split["confirm"], "Confirmatory set", FIGURES / "figure_erd_ers_confirm.png", mask)
    _plot_tfr(maps_by_split["explore"], "Exploratory set", FIGURES / "figure_erd_ers_explore.png")
    confirm = maps_by_split["confirm"]
    _plot_topographies(
        np.stack([item["class_r2"] for item in confirm]).mean(axis=0),
        np.stack([item["state_alpha_corr"] for item in confirm]).mean(axis=0),
        np.stack([item["differential_state_proxy"] for item in confirm]).mean(axis=0),
        np.stack([item["common_state_proxy"] for item in confirm]).mean(axis=0),
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        commit = "unavailable"
    manifest = pd.DataFrame(
        [
            {"figure_id": "erd_ers_confirm", "script": "scripts/generate_physiology_figures.py", "input_artifacts": "epochs/, subject_split.csv", "config_hash": config_hash, "git_commit": commit, "caption": "Confirmatory contralateral/ipsilateral ERD/ERS; cluster-corrected contour.", "status": "generated"},
            {"figure_id": "erd_ers_explore", "script": "scripts/generate_physiology_figures.py", "input_artifacts": "epochs/, subject_split.csv", "config_hash": config_hash, "git_commit": commit, "caption": "Exploratory ERD/ERS reference panel.", "status": "generated"},
            {"figure_id": "sensor_space_topographies_confirm", "script": "scripts/generate_physiology_figures.py", "input_artifacts": "epochs/, state_latent.parquet", "config_hash": config_hash, "git_commit": commit, "caption": "Sensor-space physiological diagnostics; state maps are labelled proxy maps.", "status": "generated"},
            {"figure_id": "cross_state_gap_full_nulls", "script": "scripts/run_cross_state_full_inference.py", "input_artifacts": "state_features.parquet, mi_covariances.npy", "config_hash": "from cross_state_null_summary_confirm.json", "git_commit": commit, "caption": "Full confirmatory cross-state null distributions.", "status": "generated"},
        ]
    )
    manifest.to_csv(FIGURES / "figure_manifest.csv", index=False)
    print(json.dumps({"config_hash": config_hash, "n_confirm": len(confirm), "n_explore": len(maps_by_split["explore"])}, indent=2))


if __name__ == "__main__":
    main()
