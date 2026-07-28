"""Physiology and negative-control checks for the learned state latent."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_PROCESSED, FIGURES, RESULTS  # noqa: E402


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    loadings = pd.read_parquet(DATA_PROCESSED / "state_loadings.parquet")
    frame = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"])

    # Group distribution of within-subject loadings.
    loading_summary = loadings.groupby("feature")["loading"].agg(["mean", "std", "count"]).reset_index()
    loading_summary.to_csv(RESULTS / "tables" / "state_loading_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(loading_summary["feature"], loading_summary["mean"], yerr=loading_summary["std"] / np.sqrt(loading_summary["count"]), capsize=2)
    ax.tick_params(axis="x", rotation=40)
    ax.set(ylabel="Mean within-subject loading", title="Pre-cue state-latent loadings")
    fig.tight_layout()
    fig.savefig(FIGURES / "state_latent_loadings.png", dpi=180)
    plt.close(fig)

    # Negative control 1: state must not encode the randomized class sequence.
    x = frame[["z_state"]].to_numpy()
    y = (frame["target"] == "right").astype(int).to_numpy()
    groups = frame["subject"].to_numpy()
    cv = GroupKFold(n_splits=5)
    class_auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), x, y, groups=groups, cv=cv, scoring="roc_auc")

    # Negative control 2: EMG alone must not reconstruct state.
    emg_r2 = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=10)), frame[["emg_rms_20_45hz"]], frame["z_state"], groups=groups, cv=cv, scoring="r2")
    controls = pd.DataFrame(
        [
            {"control": "z_state_predicts_class_auc", "mean": class_auc.mean(), "sd": class_auc.std()},
            {"control": "EMG_predicts_z_state_r2", "mean": emg_r2.mean(), "sd": emg_r2.std()},
        ]
    )
    controls.to_csv(RESULTS / "tables" / "state_negative_controls.csv", index=False)
    print(controls.to_string(index=False))


if __name__ == "__main__":
    main()
