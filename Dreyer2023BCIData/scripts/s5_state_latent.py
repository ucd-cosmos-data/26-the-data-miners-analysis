"""S5: fit the pre-cue state latent and run its preregistered validation."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.state_latent import fit_subject_state, lag1_permutation_pvalue  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    states, loadings, validation = [], [], []
    for subject, frame in features.groupby("subject", sort=True):
        state, loading = fit_subject_state(frame.reset_index(drop=True))
        states.append(state)
        loadings.append(loading)
        seed = int.from_bytes(hashlib.sha256(f"20260727:{subject}".encode()).digest()[:4], "little")
        ac, p = lag1_permutation_pvalue(state["z_state"].to_numpy(), n_perm=1000, seed=seed)
        validation.append({"subject": subject, "lag1_autocorrelation": ac, "lag1_permutation_p": p})
    state = pd.concat(states, ignore_index=True)
    loadings = pd.concat(loadings, ignore_index=True)
    validation = pd.DataFrame(validation)
    state.to_parquet(DATA_PROCESSED / "state_latent.parquet", index=False)
    loadings.to_parquet(DATA_PROCESSED / "state_loadings.parquet", index=False)

    # Self-report anchor: first-to-last-run latent shift, paired with the
    # only comparable pre/post composites (Mood and Mindfulness).
    run_mean = state.groupby(["subject", "run_index"], as_index=False)["z_state"].mean()
    summary = run_mean.groupby("subject")["z_state"].agg(["first", "last"]).reset_index()
    summary["state_change_first_to_last"] = summary["last"] - summary["first"]
    subjects = pd.read_parquet(DATA_PROCESSED / "subjects.parquet")
    anchor = summary.merge(subjects[["subject", "delta_mood", "delta_mindfulness"]], on="subject")
    anchor_rows = []
    for column in ["delta_mood", "delta_mindfulness"]:
        valid = anchor[["state_change_first_to_last", column]].dropna()
        rho, p = spearmanr(valid["state_change_first_to_last"], valid[column])
        anchor_rows.append({"self_report": column, "n": len(valid), "spearman_rho": rho, "p_value": p})

    validation.to_csv(RESULTS / "tables" / "state_autocorrelation_validation.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(RESULTS / "tables" / "state_self_report_anchor.csv", index=False)
    summary_out = {
        "n_subjects": int(state["subject"].nunique()),
        "median_lag1": float(validation["lag1_autocorrelation"].median()),
        "n_lag1_p_lt_0_05": int((validation["lag1_permutation_p"] < 0.05).sum()),
        "cp2_pass": bool((validation["lag1_permutation_p"] < 0.05).sum() >= (len(validation) / 2)),
    }
    pd.DataFrame([summary_out]).to_json(RESULTS / "tables" / "state_latent_summary.json", orient="records", indent=2)
    print(pd.Series(summary_out).to_string())
    print(pd.DataFrame(anchor_rows).to_string(index=False))


if __name__ == "__main__":
    main()
