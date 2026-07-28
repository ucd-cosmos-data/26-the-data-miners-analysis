"""Exploratory manifold and spectral EDA figures (not confirmatory endpoints).

Produces:
  - Isomap embedding of pre-cue state features
  - t-SNE embedding of pre-cue state features
  - Laplacian Eigenmap (SpectralEmbedding) of pre-cue state features
  - Fourier / Welch frequency spectra from MI epochs (C3/C4)

All panels are labelled exploratory and use the explore split only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdf_io import EEG_CHANNELS  # noqa: E402
from src.models.state_latent import PRIMARY_STATE_FEATURES  # noqa: E402
from src.paths import DATA_PROCESSED, FIGURES, RESULTS, all_subjects  # noqa: E402
from src.visualization.style import apply_style, save_figure  # noqa: E402


SEED = 20260727
MAX_POINTS = 2500


def _explore_frame() -> pd.DataFrame:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    frame = features.merge(
        state[["subject", "run_index", "trial", "z_state"]],
        on=["subject", "run_index", "trial"],
        validate="one_to_one",
    ).merge(split[["subject", "split"]], on="subject", validate="many_to_one")
    frame = frame[frame["split"] == "explore"].copy()
    frame = frame.loc[~frame["reject_flag"].astype(bool)].reset_index(drop=True)
    return frame


def _feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    cols = [c for c in PRIMARY_STATE_FEATURES if c in frame.columns]
    matrix = frame[cols].to_numpy(dtype=float)
    keep = np.isfinite(matrix).all(axis=1)
    frame = frame.loc[keep].reset_index(drop=True)
    matrix = matrix[keep]
    # Clip extreme tails so neighbor graphs stay connected for Isomap/Laplacian.
    lo = np.percentile(matrix, 1, axis=0)
    hi = np.percentile(matrix, 99, axis=0)
    matrix = np.clip(matrix, lo, hi)
    if len(frame) > MAX_POINTS:
        rng = np.random.default_rng(SEED)
        # Stratify roughly by subject so one person cannot dominate.
        take = []
        per = max(8, MAX_POINTS // max(frame["subject"].nunique(), 1))
        for _, part in frame.groupby("subject", sort=False):
            idx = part.index.to_numpy()
            chosen = rng.choice(idx, size=min(len(idx), per), replace=False)
            take.append(chosen)
        take = np.concatenate(take)
        if len(take) > MAX_POINTS:
            take = rng.choice(take, size=MAX_POINTS, replace=False)
        take = np.sort(take)
        frame = frame.iloc[take].reset_index(drop=True)
        matrix = matrix[take]
    scaled = StandardScaler().fit_transform(matrix)
    return scaled, frame


def _scatter_embedding(xy: np.ndarray, frame: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=False, sharey=False)
    # Color by continuous state.
    sc0 = axes[0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=frame["z_state"].to_numpy(),
        cmap="coolwarm",
        s=10,
        alpha=0.75,
        linewidths=0,
    )
    cbar0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Pre-cue z-state")
    axes[0].set(title="Colored by state latent", xlabel="Component 1", ylabel="Component 2")

    # Color by MI class.
    target = frame["target"].to_numpy()
    colors = np.where(target == "right", "#F58518", "#4C78A8")
    axes[1].scatter(xy[:, 0], xy[:, 1], c=colors, s=10, alpha=0.75, linewidths=0)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C78A8", markersize=7, label="Left MI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F58518", markersize=7, label="Right MI"),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
    axes[1].set(title="Colored by MI class", xlabel="Component 1", ylabel="Component 2")
    fig.suptitle(f"{title} (exploratory split; not confirmatory)", y=1.02)
    save_figure(fig, path, rect=(0, 0.06, 1, 0.96))


def _plot_manifolds(scaled: np.ndarray, frame: pd.DataFrame) -> dict:
    n_neighbors = min(120, max(30, len(scaled) // 12))
    outputs = {"n_neighbors": int(n_neighbors)}

    iso = Isomap(n_neighbors=n_neighbors, n_components=2, metric="euclidean")
    xy_iso = iso.fit_transform(scaled)
    path_iso = FIGURES / "eda_isomap_state_features.png"
    _scatter_embedding(xy_iso, frame, "Isomap of pre-cue state features", path_iso)
    outputs["isomap"] = str(path_iso)

    tsne = TSNE(
        n_components=2,
        perplexity=min(40, max(5, len(scaled) // 20)),
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        metric="euclidean",
    )
    xy_tsne = tsne.fit_transform(scaled)
    path_tsne = FIGURES / "eda_tsne_state_features.png"
    _scatter_embedding(xy_tsne, frame, "t-SNE of pre-cue state features", path_tsne)
    outputs["tsne"] = str(path_tsne)

    lap = SpectralEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=SEED,
        affinity="nearest_neighbors",
    )
    xy_lap = lap.fit_transform(scaled)
    path_lap = FIGURES / "eda_laplacian_eigenmap_state_features.png"
    _scatter_embedding(xy_lap, frame, "Laplacian Eigenmap of pre-cue state features", path_lap)
    outputs["laplacian_eigenmap"] = str(path_lap)
    return outputs


def _welch_mean(epochs: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    # epochs: (n_trials, n_channels, n_times)
    nperseg = min(256, epochs.shape[-1])
    freqs, power = welch(epochs, fs=sfreq, axis=-1, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, power.mean(axis=0)


def _plot_fft_spectra(explore_subjects: list[str]) -> dict:
    epochs_root = DATA_PROCESSED / "epochs"
    c3, c4 = EEG_CHANNELS.index("C3"), EEG_CHANNELS.index("C4")
    left_c3, left_c4, right_c3, right_c4 = [], [], [], []
    freqs = None
    used = 0
    for subject in explore_subjects:
        path = epochs_root / f"{subject}_epochs.npz"
        trials_path = epochs_root / f"{subject}_trials.parquet"
        if not path.exists() or not trials_path.exists():
            continue
        arrays = np.load(path)
        table = pd.read_parquet(trials_path)
        mi = arrays["mi"]
        sfreq = float(arrays["sfreq"]) if "sfreq" in arrays.files else 128.0
        valid = ~table["reject_flag"].astype(bool).to_numpy() if "reject_flag" in table.columns else np.ones(len(table), dtype=bool)
        mi = mi[valid]
        target = table.loc[valid, "target"].to_numpy()
        if (target == "left").sum() < 10 or (target == "right").sum() < 10:
            continue
        freqs, p_left = _welch_mean(mi[target == "left"], sfreq)
        _, p_right = _welch_mean(mi[target == "right"], sfreq)
        left_c3.append(p_left[c3])
        left_c4.append(p_left[c4])
        right_c3.append(p_right[c3])
        right_c4.append(p_right[c4])
        used += 1

    left_c3_m = np.mean(np.stack(left_c3), axis=0)
    left_c4_m = np.mean(np.stack(left_c4), axis=0)
    right_c3_m = np.mean(np.stack(right_c3), axis=0)
    right_c4_m = np.mean(np.stack(right_c4), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    axes[0].plot(freqs, 10 * np.log10(left_c3_m + 1e-18), color="#4C78A8", linewidth=2, label="C3 (left MI)")
    axes[0].plot(freqs, 10 * np.log10(left_c4_m + 1e-18), color="#F58518", linewidth=2, label="C4 (left MI)")
    axes[0].axvspan(8, 12, color="gray", alpha=0.12)
    axes[0].axvspan(13, 30, color="gray", alpha=0.06)
    axes[0].set(title="Left motor imagery", xlabel="Frequency (Hz)", ylabel="Power (dB)")
    axes[0].set_xlim(1, 45)

    axes[1].plot(freqs, 10 * np.log10(right_c3_m + 1e-18), color="#4C78A8", linewidth=2, label="C3 (right MI)")
    axes[1].plot(freqs, 10 * np.log10(right_c4_m + 1e-18), color="#F58518", linewidth=2, label="C4 (right MI)")
    axes[1].axvspan(8, 12, color="gray", alpha=0.12)
    axes[1].axvspan(13, 30, color="gray", alpha=0.06)
    axes[1].set(title="Right motor imagery", xlabel="Frequency (Hz)")
    axes[1].set_xlim(1, 45)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=2, label="C3"),
        Line2D([0], [0], color="#F58518", linewidth=2, label="C4"),
        Patch(facecolor="gray", alpha=0.25, label="Alpha 8–12 Hz"),
        Patch(facecolor="gray", alpha=0.10, label="Beta 13–30 Hz"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=True)
    fig.suptitle(f"Welch / Fourier spectra at C3–C4 (exploratory; n={used} subjects)", y=1.02)
    path = FIGURES / "eda_fft_frequency_spectra.png"
    save_figure(fig, path, rect=(0, 0.08, 1, 0.96))

    # Also save a single-panel FFT overview for quick viewing.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(freqs, 10 * np.log10((left_c3_m + right_c4_m) / 2 + 1e-18), color="#4C78A8", linewidth=2.2, label="Contralateral mean (C4|left, C3|right)")
    ax.plot(freqs, 10 * np.log10((left_c4_m + right_c3_m) / 2 + 1e-18), color="#F58518", linewidth=2.2, label="Ipsilateral mean (C3|left, C4|right)")
    ax.axvspan(8, 12, color="gray", alpha=0.12)
    ax.axvspan(13, 30, color="gray", alpha=0.06)
    ax.set(xlabel="Frequency (Hz)", ylabel="Power (dB)", title="Exploratory MI Fourier spectra (contra vs ipsi)", xlim=(1, 45))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    path2 = FIGURES / "eda_fft_contra_ipsi_overview.png"
    save_figure(fig, path2, rect=(0, 0.12, 1, 1))

    table = pd.DataFrame(
        {
            "frequency_hz": freqs,
            "left_c3_power": left_c3_m,
            "left_c4_power": left_c4_m,
            "right_c3_power": right_c3_m,
            "right_c4_power": right_c4_m,
        }
    )
    table.to_csv(RESULTS / "tables" / "eda_fft_spectra_explore.csv", index=False)
    return {"fft_panels": str(path), "fft_overview": str(path2), "n_subjects": used}


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = _explore_frame()
    scaled, sampled = _feature_matrix(frame)
    manifold = _plot_manifolds(scaled, sampled)
    explore_subjects = sorted(frame["subject"].unique().tolist())
    spectra = _plot_fft_spectra(explore_subjects)
    summary = {
        "validity_label": "exploratory_eda_only",
        "n_explore_trials_available": int(len(frame)),
        "n_points_embedded": int(len(sampled)),
        "feature_set": [c for c in PRIMARY_STATE_FEATURES if c in frame.columns],
        "seed": SEED,
        **manifold,
        **spectra,
    }
    (RESULTS / "tables" / "eda_manifold_fft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
