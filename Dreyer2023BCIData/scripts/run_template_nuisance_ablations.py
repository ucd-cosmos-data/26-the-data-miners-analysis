"""Complete remaining fixed SCTM template / nuisance ablation grid."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import design_matrix, tangent_features  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _rss(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> float:
    model = Ridge(alpha=alpha, fit_intercept=False).fit(x, y)
    residual = y - model.predict(x)
    return float(np.sum(residual * residual))


def _partial_r2(full_rss: float, reduced_rss: float) -> float:
    return float(max(0.0, 1.0 - full_rss / max(reduced_rss, 1e-12)))


def evaluate_subject(covs: np.ndarray, frame: pd.DataFrame) -> dict:
    valid = ~frame["reject_flag"].astype(bool).to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    covs = covs[valid]
    if len(frame) < 60 or frame["target"].nunique() < 2:
        return {"status": "insufficient", "n_trials": int(len(frame))}
    target = tangent_features(covs)
    y = np.where(frame["target"].to_numpy() == "right", 1.0, -1.0)
    state = (frame["z_state"].to_numpy() - frame["z_state"].mean()) / (frame["z_state"].std(ddof=1) + 1e-8)
    trial_time = frame.groupby("run_index")["trial"].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-8)).to_numpy()
    run_time = (frame["run_index"].to_numpy() - frame["run_index"].mean()) / (frame["run_index"].std(ddof=1) + 1e-8)
    blink = (frame["blink_amplitude"].to_numpy() - frame["blink_amplitude"].mean()) / (frame["blink_amplitude"].std(ddof=1) + 1e-8)
    emg = (frame["emg_rms_20_45hz"].to_numpy() - frame["emg_rms_20_45hz"].mean()) / (frame["emg_rms_20_45hz"].std(ddof=1) + 1e-8)
    intercept = np.ones(len(frame))

    designs = {
        "fixed_template_only": np.column_stack([intercept, y]),
        "template_plus_time_trend": np.column_stack([intercept, y, trial_time, run_time]),
        "template_plus_time_and_nuisance": np.column_stack([intercept, y, trial_time, run_time, blink, emg]),
        "remove_state": np.column_stack([intercept, y, trial_time, run_time, blink, emg]),
        "remove_differential_term": design_matrix(frame, include_interaction=False),
        "remove_nuisance_covariates": np.column_stack([intercept, y, state, y * state, trial_time, run_time]),
        "state_on_template_time_nuisance": design_matrix(frame, include_interaction=False),
        "h1_differential_full": design_matrix(frame, include_interaction=True),
        "h1b_common_state_only": design_matrix(frame, include_interaction=False),
    }
    rss = {name: _rss(mat, target) for name, mat in designs.items()}
    return {
        "status": "ok",
        "n_trials": int(len(frame)),
        "rss_fixed_template_only": rss["fixed_template_only"],
        "rss_template_plus_time_trend": rss["template_plus_time_trend"],
        "rss_template_plus_time_and_nuisance": rss["template_plus_time_and_nuisance"],
        "rss_remove_state": rss["remove_state"],
        "rss_remove_differential_term": rss["remove_differential_term"],
        "rss_remove_nuisance_covariates": rss["remove_nuisance_covariates"],
        "rss_state_on_template_time_nuisance": rss["state_on_template_time_nuisance"],
        "rss_h1_differential_full": rss["h1_differential_full"],
        "rss_h1b_common_state_only": rss["h1b_common_state_only"],
        "partial_r2_time_over_template": _partial_r2(rss["template_plus_time_trend"], rss["fixed_template_only"]),
        "partial_r2_nuisance_over_time": _partial_r2(rss["template_plus_time_and_nuisance"], rss["template_plus_time_trend"]),
        "partial_r2_common_state_over_controls": _partial_r2(rss["h1b_common_state_only"], rss["template_plus_time_and_nuisance"]),
        "partial_r2_differential_over_common": _partial_r2(rss["h1_differential_full"], rss["h1b_common_state_only"]),
        "partial_r2_state_terms_over_no_state": _partial_r2(rss["h1_differential_full"], rss["remove_state"]),
        "partial_r2_nuisance_contribution": _partial_r2(rss["h1_differential_full"], rss["remove_nuisance_covariates"]),
    }


def main() -> None:
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    base = features.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"], validate="one_to_one")
    base = base.merge(split[["subject", "split"]], on="subject", validate="many_to_one")
    confirm = base[base["split"] == "confirm"].reset_index(drop=True)
    covs = np.load(DATA_PROCESSED / "mi_covariances.npy")[
        features.index[features["subject"].isin(confirm["subject"].unique())].to_numpy()
    ]
    rows, cursor = [], 0
    for subject in confirm["subject"].drop_duplicates():
        part = confirm[confirm["subject"] == subject]
        n = len(part)
        out = evaluate_subject(covs[cursor : cursor + n], part)
        out.update(
            {
                "subject": subject,
                "dataset": subject[0],
                "validity_label": "fixed_ablation_grid_sensitivity",
                "seed": int.from_bytes(hashlib.sha256(f"ablation-grid:{subject}".encode()).digest()[:4], "little"),
            }
        )
        rows.append(out)
        cursor += n
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "tables" / "template_nuisance_ablation_confirm.csv", index=False)
    ok = result[result["status"] == "ok"]
    summary = {
        "validity_label": "fixed_ablation_grid_sensitivity",
        "n_subjects": int(len(ok)),
        "mean_partial_r2_differential_over_common": float(ok["partial_r2_differential_over_common"].mean()),
        "mean_partial_r2_common_state_over_controls": float(ok["partial_r2_common_state_over_controls"].mean()),
        "mean_partial_r2_time_over_template": float(ok["partial_r2_time_over_template"].mean()),
        "mean_partial_r2_nuisance_over_time": float(ok["partial_r2_nuisance_over_time"].mean()),
        "config_hash": hashlib.sha256(json.dumps({"grid": "template_nuisance"}, sort_keys=True).encode()).hexdigest()[:16],
    }
    (RESULTS / "tables" / "template_nuisance_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
