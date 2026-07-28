"""Exploratory-only descriptive figures and confound scan."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.state_latent import PRIMARY_STATE_FEATURES  # noqa: E402
from src.paths import DATA_PROCESSED, FIGURES, RESULTS  # noqa: E402


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    runs = pd.read_parquet(DATA_PROCESSED / "runs.parquet")
    frame = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"])
    frame = frame.merge(split[["subject", "split"]], on="subject")
    frame = frame[frame["split"] == "explore"].copy()

    # Variance decomposition: how much of each pre-cue feature is within person?
    rows = []
    for feature in PRIMARY_STATE_FEATURES:
        x = frame[[feature, "subject"]].dropna()
        total = x[feature].var(ddof=1)
        between = x.groupby("subject")[feature].mean().var(ddof=1)
        within = x.groupby("subject")[feature].transform(lambda s: s - s.mean()).var(ddof=1)
        rows.append({"feature": feature, "total_variance": total, "between_variance": between, "within_variance": within, "within_fraction": within / total if total else np.nan})
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / "eda_state_variance.csv", index=False)

    # Confound scan, preserving the claim that state cannot just be clock or artifact.
    confounds = ["trial", "run_index", "blink_amplitude", "emg_rms_20_45hz", "artifact_robust_z"]
    scans = []
    for subject, sub in frame.groupby("subject"):
        for feature in confounds:
            rho, p = spearmanr(sub["z_state"], sub[feature], nan_policy="omit")
            scans.append({"subject": subject, "confound": feature, "spearman_rho": rho, "p_value": p})
    pd.DataFrame(scans).to_csv(RESULTS / "tables" / "eda_state_confound_scan.csv", index=False)

    # Run online accuracy trajectories with the recorded finite-trial denominator.
    online = runs[(runs["kind"] == "online") & runs["tacc"].notna()].merge(split, on="subject")
    online = online[online["split"] == "explore"]
    summary = online.groupby("run_index")["tacc"].agg(["mean", "sem", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(summary["run_index"], summary["mean"], yerr=1.96 * summary["sem"], marker="o", capsize=3)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Online run", ylabel="Recorded TAcc (%)", xticks=[3, 4, 5, 6], ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_online_accuracy_trajectory.png", dpi=180)
    plt.close(fig)

    # State trajectory over normalized within-run trial position.
    state_curve = frame.assign(position=frame.groupby(["subject", "run_index"])["trial"].transform(lambda s: (s - 1) / max(len(s) - 1, 1))).groupby("trial")["z_state"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(state_curve["trial"], state_curve["mean"])
    ax.fill_between(state_curve["trial"], state_curve["mean"] - 1.96 * state_curve["sem"], state_curve["mean"] + 1.96 * state_curve["sem"], alpha=.2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(xlabel="Trial number within run", ylabel="Pre-cue state latent (z)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_state_within_run_trajectory.png", dpi=180)
    plt.close(fig)

    print({"n_exploratory_subjects": int(frame["subject"].nunique()), "n_exploratory_trials": len(frame)})


if __name__ == "__main__":
    main()
