"""Regenerate publication figures with cleaned legends/colorbars/layouts.

Rebuilds cheap figures from saved tables. Physiology TFR/topography plots are
repaired in ``generate_physiology_figures.py`` and should be re-run separately
if their source maps need recomputation; this script also offers a fast
layout-only path when intermediate caches exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import FIGURES, RESULTS  # noqa: E402
from src.visualization.style import apply_style, save_figure  # noqa: E402


def _plot_cross_state_nulls() -> None:
    null = pd.read_csv(RESULTS / "tables" / "cross_state_confirm_full_permutation.csv")
    summary = json.loads((RESULTS / "tables" / "cross_state_null_summary_confirm.json").read_text(encoding="utf-8"))
    observed = float(summary["observed_group_mean_gap"])
    time_control = float(summary["time_control_group_mean"])
    perm = null["state_permutation_group_mean_gap"].to_numpy()
    random = null["random_variable_group_mean_gap"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    lo = min(perm.min(), random.min(), observed, time_control) - 0.005
    hi = max(perm.max(), random.max(), observed, time_control) + 0.005
    bins = np.linspace(lo, hi, 36)
    ax.hist(perm, bins=bins, alpha=0.55, color="#4C78A8", label="Within-run state permutation", edgecolor="white", linewidth=0.4)
    ax.hist(random, bins=bins, alpha=0.45, color="#F58518", label="Random within-run variable", edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color="black", linewidth=2.0, label=f"Observed mean = {observed:.4f}")
    ax.axvline(time_control, color="#333333", linestyle="--", linewidth=1.6, label=f"Trial-index control = {time_control:.4f}")
    ax.set(
        xlabel="Subject-mean cross-state AUC gap",
        ylabel="Null replicates (count)",
        title=f"Confirmatory cross-state inference (n={summary['n_subjects_observed']} subjects)",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    save_figure(fig, FIGURES / "cross_state_gap_full_nulls.png", rect=(0, 0.08, 1, 1))


def _plot_ablation_forest() -> None:
    summary = pd.read_csv(RESULTS / "tables" / "ablation_summary_full.csv")
    # Exclude non-comparable rows that reuse the differential column for other R² terms.
    exclude = {"time_over_template", "nuisance_over_time", "template_plus_time_and_nuisance_controls"}
    summary = summary[~summary["ablation"].isin(exclude)].copy()
    summary = summary.dropna(subset=["differential_partial_r2_mean"])
    summary = summary.sort_values("differential_partial_r2_mean").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(summary))
    colors = ["#B00020" if "circularity" in str(v) else "#1B6CA8" for v in summary["validity_label"]]
    ax.hlines(y, 0, summary["differential_partial_r2_mean"], color=colors, linewidth=2.2)
    ax.scatter(summary["differential_partial_r2_mean"], y, c=colors, s=42, zorder=3, edgecolors="white", linewidths=0.5)
    primary = 0.005861
    ax.axvline(primary, color="black", linestyle="--", linewidth=1.2, label="Primary confirm mean")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["ablation"])
    ax.set_xlabel("Mean confirmatory differential partial R²")
    ax.set_title("Ablation forest (sensitivity/circularity; not a new confirmatory endpoint)")
    ax.legend(loc="lower right", frameon=True)
    # Color key for circularity vs sensitivity.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#1B6CA8", marker="o", linestyle="None", label="Sensitivity / fixed series"),
        Line2D([0], [0], color="#B00020", marker="o", linestyle="None", label="Circularity (not primary)"),
        Line2D([0], [0], color="black", linestyle="--", label="Primary confirm mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    save_figure(fig, FIGURES / "ablation_forest_plot.png")


def _plot_deep_baselines() -> None:
    curves = pd.read_csv(RESULTS / "tables" / "deep_baseline_training_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    colors = {"EEGNet": "#4C78A8", "ShallowConvNet": "#F58518"}
    for architecture, part in curves.groupby("architecture"):
        grouped = part.groupby("epoch")[["train_loss", "validation_loss"]].mean()
        color = colors.get(architecture, None)
        axes[0].plot(grouped.index, grouped["train_loss"], label=architecture, color=color, linewidth=2)
        axes[1].plot(grouped.index, grouped["validation_loss"], label=architecture, color=color, linewidth=2)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    for ax in axes:
        ax.legend(loc="upper right", frameon=True)
    fig.suptitle("Deep decoding baselines (GroupKFold by subject)", y=1.02)
    save_figure(fig, FIGURES / "deep_baseline_training_curves.png")


def _plot_state_loadings() -> None:
    loading = pd.read_csv(RESULTS / "tables" / "state_loading_summary.csv")
    labels = [name.replace("_", "\n") for name in loading["feature"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(loading))
    ax.bar(
        x,
        loading["mean"],
        yerr=loading["std"] / np.sqrt(loading["count"].clip(lower=1)),
        capsize=3,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="Mean within-subject loading", title="Pre-cue state-latent loadings")
    save_figure(fig, FIGURES / "state_latent_loadings.png")


def _plot_eda_trajectories() -> None:
    # Rebuild from processed tables if present; otherwise skip gracefully.
    split = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "subject_split.csv")
    runs = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "runs.parquet")
    features = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "state_features.parquet")
    state = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "state_latent.parquet")

    online = runs[(runs["kind"] == "online") & runs["tacc"].notna()].merge(split, on="subject")
    online = online[online["split"] == "explore"]
    summary = online.groupby("run_index")["tacc"].agg(["mean", "sem", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(
        summary["run_index"],
        summary["mean"],
        yerr=1.96 * summary["sem"],
        marker="o",
        capsize=4,
        color="#4C78A8",
        linewidth=2,
        label="Explore-set mean ± 95% CI",
    )
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="Chance (50%)")
    ax.set(xlabel="Online run", ylabel="Recorded TAcc (%)", xticks=[3, 4, 5, 6], ylim=(40, 80))
    ax.set_title("Exploratory online accuracy trajectory")
    ax.legend(loc="lower right", frameon=True)
    save_figure(fig, FIGURES / "eda_online_accuracy_trajectory.png")

    frame = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"])
    frame = frame.merge(split[["subject", "split"]], on="subject")
    frame = frame[frame["split"] == "explore"].copy()
    state_curve = frame.groupby("trial")["z_state"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(state_curve["trial"], state_curve["mean"], color="#4C78A8", linewidth=2, label="Mean z-state")
    ax.fill_between(
        state_curve["trial"],
        state_curve["mean"] - 1.96 * state_curve["sem"],
        state_curve["mean"] + 1.96 * state_curve["sem"],
        alpha=0.22,
        color="#4C78A8",
        label="95% CI",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Zero")
    ax.set(xlabel="Trial number within run", ylabel="Pre-cue state latent (z)", title="Exploratory within-run state trajectory")
    ax.legend(loc="upper right", frameon=True)
    save_figure(fig, FIGURES / "eda_state_within_run_trajectory.png")


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_cross_state_nulls()
    _plot_ablation_forest()
    _plot_deep_baselines()
    _plot_state_loadings()
    _plot_eda_trajectories()
    print({"status": "ok", "figures_dir": str(FIGURES)})


if __name__ == "__main__":
    main()
