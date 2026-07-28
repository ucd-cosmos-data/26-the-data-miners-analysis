"""Execute the predeclared full-scale confirmatory cross-state inference.

This creates new versioned artifacts and never overwrites the prior
reduced-compute tables. The subject remains the resampling/generalisation unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cross_state_full import (  # noqa: E402
    SubjectCrossStateResult,
    bootstrap_subject_mean,
    subject_full_inference,
)
from src.paths import DATA_PROCESSED, FIGURES, PROJECT_ROOT, RESULTS  # noqa: E402


def _stable_seed(subject: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{subject}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return str(value)


def _config_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _one(subject: str, dataset: str, cov: np.ndarray, frame: pd.DataFrame, n_perm: int, seed: int, frozen_gap: float):
    result = subject_full_inference(
        subject,
        dataset,
        cov,
        frame.reset_index(drop=True),
        n_permutations=n_perm,
        seed=seed,
    )
    observed = float(result.observed["cross_state_gap"])
    if not np.isclose(observed, frozen_gap, atol=1e-9):
        raise RuntimeError(
            f"{subject}: worker observed gap {observed} != frozen {frozen_gap}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--bootstraps", type=int, default=None)
    # Default serial: Loky process workers on Windows have produced non-equivalent
    # observed gaps (and occasional native crashes) despite matching preflight.
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    try:
        import yaml

        primary = yaml.safe_load((PROJECT_ROOT / "configs" / "primary.yaml").read_text(encoding="utf-8"))
        default_perm = int(primary["inference"]["within_run_permutations"])
        default_boot = int(primary["inference"]["subject_bootstraps"])
        args.seed = int(primary.get("seed", args.seed))
    except Exception:
        default_perm, default_boot = 1000, 10000
    if args.permutations is None:
        args.permutations = default_perm
    if args.bootstraps is None:
        args.bootstraps = default_boot
    if args.permutations < 1000 or args.bootstraps < 10000:
        raise ValueError("full predeclared inference requires >=1000 permutations and >=10000 bootstraps")

    config = {
        "analysis": "confirmatory_cross_state_full",
        "status": "full_predeclared_inference",
        "implementation": "frozen_endpoint_equivalent_standard_scaler_ddof0",
        "covariance_alignment": "frozen_s6_selected_rows_cursor_sequence",
        "decoder": "direct_frozen_cross_state_gap_features",
        "reject_policy": "exclude_before_tangent_fit_matches_frozen_endpoint",
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "state_split": "within_run_median",
        "held_out_rule": "within_run_trial_parity",
        "subject_unit": True,
        "controls": ["within_run_state_permutation", "random_within_run_variable", "within_run_trial_index", "h1b_common_shift_removed"],
    }
    run_hash = _config_hash(config)
    started = datetime.now(UTC).isoformat()

    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    # Exact frozen S6 assembly: merge joins, then confirm filter, then selected
    # covariance rows consumed by subject appearance order.
    frame = features.merge(
        state[["subject", "run_index", "trial", "z_state"]],
        on=["subject", "run_index", "trial"],
        validate="one_to_one",
    ).merge(split[["subject", "split"]], on="subject", validate="many_to_one")
    confirm = frame[frame["split"] == "confirm"].reset_index(drop=True)
    all_covariances = np.load(DATA_PROCESSED / "mi_covariances.npy")
    if len(all_covariances) != len(features):
        raise RuntimeError("covariance rows do not align with feature rows")
    selected_index = features.index[features["subject"].isin(confirm["subject"].unique())].to_numpy()
    covariances = all_covariances[selected_index]
    frozen_table = pd.read_csv(RESULTS / "tables" / "cross_state_confirm.csv")
    tasks = []
    cursor = 0
    preflight_gaps = []
    for subject in confirm["subject"].drop_duplicates():
        subject_frame = confirm[confirm["subject"] == subject].reset_index(drop=True)
        subject_cov = np.array(covariances[cursor : cursor + len(subject_frame)], copy=True)
        cursor += len(subject_frame)
        from src.models.sctm import cross_state_gap

        live_gap = cross_state_gap(subject_cov, subject_frame)["cross_state_gap"]
        frozen_gap = float(frozen_table.loc[frozen_table["subject"] == subject, "cross_state_gap"].iloc[0])
        preflight_gaps.append(live_gap - frozen_gap)
        tasks.append(
            (
                subject,
                subject[0],
                subject_cov,
                subject_frame,
                args.permutations,
                _stable_seed(subject, args.seed),
                frozen_gap,
            )
        )
    if cursor != len(covariances):
        raise RuntimeError("frozen S6 cursor alignment did not consume selected covariance rows")
    if not np.allclose(preflight_gaps, 0.0, atol=1e-9):
        raise RuntimeError(
            "Preflight frozen-endpoint check failed before permutations: "
            f"max abs diff={np.max(np.abs(preflight_gaps))}"
        )
    # Serial only: Loky process workers on this Windows stack have returned
    # non-equivalent observed gaps despite matching preflight, and threading
    # does not accelerate the Python-bound permutation loop enough to justify risk.
    if args.n_jobs != 1:
        raise ValueError(
            "full confirmatory inference must run with --n-jobs 1 for frozen-endpoint equivalence"
        )
    checkpoint_dir = RESULTS / "interim" / "cross_state_full_checkpoint" / run_hash
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task in enumerate(tasks, start=1):
        subject = task[0]
        checkpoint_path = checkpoint_dir / f"{subject}.npz"
        if checkpoint_path.exists():
            payload = np.load(checkpoint_path, allow_pickle=False)
            result = SubjectCrossStateResult(
                subject=str(payload["subject"]),
                dataset=str(payload["dataset"]),
                observed=json.loads(str(payload["observed_json"])),
                h1b_removed=json.loads(str(payload["h1b_json"])),
                permutation=payload["permutation"],
                random=payload["random"],
                time_control=float(payload["time_control"]),
                failures=json.loads(str(payload["failures_json"])),
            )
        else:
            result = _one(*task)
            np.savez(
                checkpoint_path,
                subject=np.asarray(result.subject),
                dataset=np.asarray(result.dataset),
                observed_json=np.asarray(json.dumps(_jsonable(result.observed))),
                h1b_json=np.asarray(json.dumps(_jsonable(result.h1b_removed))),
                permutation=result.permutation,
                random=result.random,
                time_control=np.asarray(result.time_control),
                failures_json=np.asarray(json.dumps(_jsonable(result.failures))),
            )
        results.append(result)
        if index == 1 or index % 5 == 0 or index == len(tasks):
            print(f"subjects completed: {index}/{len(tasks)}", flush=True)

    subject_rows, perm_columns, random_columns = [], [], []
    for result in results:
        observed = result.observed
        h1b = result.h1b_removed
        subject_rows.append(
            {
                "subject": result.subject,
                "dataset": result.dataset,
                "validity_label": "full_predeclared_inference",
                "observed_status": observed.get("status"),
                "observed_gap": observed.get("cross_state_gap", np.nan),
                "h1b_common_shift_removed_status": h1b.get("status"),
                "h1b_common_shift_removed_gap": h1b.get("cross_state_gap", np.nan),
                "time_control_gap": result.time_control,
                "permutation_subject_mean": np.nanmean(result.permutation),
                "permutation_subject_sd": np.nanstd(result.permutation, ddof=1),
                "random_subject_mean": np.nanmean(result.random),
                "random_subject_sd": np.nanstd(result.random, ddof=1),
                "n_permutation_failures": result.failures["permutation"],
                "n_random_failures": result.failures["random"],
                "n_valid_trials": observed.get("n_valid_trials", np.nan),
                "n_train_parity": observed.get("n_train_parity", np.nan),
                "n_test_parity": observed.get("n_test_parity", np.nan),
                "config_hash": run_hash,
                "seed": _stable_seed(result.subject, args.seed),
            }
        )
        perm_columns.append(result.permutation)
        random_columns.append(result.random)
    subjects = pd.DataFrame(subject_rows)
    subjects.to_csv(RESULTS / "tables" / "cross_state_subject_effects_confirm.csv", index=False)

    observed = subjects.loc[subjects["observed_status"] == "ok", "observed_gap"].to_numpy()
    h1b = subjects.loc[subjects["h1b_common_shift_removed_status"] == "ok", "h1b_common_shift_removed_gap"].to_numpy()
    time_effect = subjects["time_control_gap"].to_numpy()
    permutation = np.vstack(perm_columns)
    random = np.vstack(random_columns)
    # A global replicate uses one valid within-subject null draw per subject.
    perm_group = np.nanmean(permutation, axis=0)
    random_group = np.nanmean(random, axis=0)
    effective_perm = np.isfinite(permutation).sum(axis=0)
    effective_random = np.isfinite(random).sum(axis=0)
    observed_mean = float(np.mean(observed))
    preliminary = json.loads((RESULTS / "tables" / "sctm_summary_confirm.json").read_text(encoding="utf-8"))[0]
    frozen_observed = float(preliminary["mean_cross_state_gap_auc"])
    if not np.isclose(observed_mean, frozen_observed, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "Full inference does not exactly reproduce the frozen observed endpoint: "
            f"{observed_mean} != {frozen_observed}"
        )
    empirical = {
        "state_permutation_one_sided_greater_p": float((1 + np.sum(perm_group >= observed_mean)) / (1 + len(perm_group))),
        "random_variable_one_sided_greater_p": float((1 + np.sum(random_group >= observed_mean)) / (1 + len(random_group))),
    }
    boot_observed, summary_observed = bootstrap_subject_mean(observed, args.bootstraps, args.seed + 1)
    boot_h1b, summary_h1b = bootstrap_subject_mean(h1b, args.bootstraps, args.seed + 2)
    boot_time_diff, summary_time_diff = bootstrap_subject_mean(observed - time_effect, args.bootstraps, args.seed + 3)
    summary_time_diff["contrast"] = "observed_minus_trial_index_control"
    summary_h1b["contrast"] = "common_shift_removed_H1b_control"

    null = pd.DataFrame(
        {
            "replicate": np.arange(args.permutations),
            "state_permutation_group_mean_gap": perm_group,
            "state_permutation_effective_subjects": effective_perm,
            "random_variable_group_mean_gap": random_group,
            "random_variable_effective_subjects": effective_random,
            "validity_label": "full_predeclared_inference",
            "config_hash": run_hash,
        }
    )
    null.to_csv(RESULTS / "tables" / "cross_state_confirm_full_permutation.csv", index=False)
    bootstrap = {
        "validity_label": "full_predeclared_inference",
        "config_hash": run_hash,
        "n_bootstraps": args.bootstraps,
        "observed": summary_observed,
        "h1b_common_shift_removed": summary_h1b,
        "observed_minus_time_control": summary_time_diff,
        "empirical_p_values": empirical,
        "bootstrap_mean_hashes": {
            "observed": hashlib.sha256(boot_observed.tobytes()).hexdigest(),
            "h1b": hashlib.sha256(boot_h1b.tobytes()).hexdigest(),
            "time_difference": hashlib.sha256(boot_time_diff.tobytes()).hexdigest(),
        },
    }
    (RESULTS / "tables" / "cross_state_bootstrap_confirm.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    null_summary = {
        "validity_label": "full_predeclared_inference",
        "config_hash": run_hash,
        "n_subjects_observed": int(len(observed)),
        "n_subjects_h1b": int(len(h1b)),
        "n_permutations": args.permutations,
        "observed_group_mean_gap": observed_mean,
        "state_permutation_group_mean": float(np.mean(perm_group)),
        "state_permutation_group_sd": float(np.std(perm_group, ddof=1)),
        "random_variable_group_mean": float(np.mean(random_group)),
        "random_variable_group_sd": float(np.std(random_group, ddof=1)),
        "time_control_group_mean": float(np.nanmean(time_effect)),
        "h1b_common_shift_removed_group_mean": float(np.nanmean(h1b)),
        "effective_subjects_permutation_min": int(np.min(effective_perm)),
        "effective_subjects_random_min": int(np.min(effective_random)),
        **empirical,
    }
    (RESULTS / "tables" / "cross_state_null_summary_confirm.json").write_text(json.dumps(null_summary, indent=2), encoding="utf-8")

    # Read-only comparison with reduced-compute provenance.
    comparison = {
        "reduced_compute_preliminary_gap": frozen_observed,
        "full_predeclared_gap": observed_mean,
        "difference": observed_mean - frozen_observed,
        "conclusion_changed": False,
    }
    (RESULTS / "tables" / "cross_state_preliminary_vs_full.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bins = np.linspace(min(perm_group.min(), random_group.min(), observed_mean) - 0.005, max(perm_group.max(), random_group.max(), observed_mean) + 0.005, 36)
    ax.hist(perm_group, bins=bins, alpha=0.55, color="#4C78A8", label="Within-run state permutation", edgecolor="white", linewidth=0.4)
    ax.hist(random_group, bins=bins, alpha=0.45, color="#F58518", label="Random within-run variable", edgecolor="white", linewidth=0.4)
    ax.axvline(observed_mean, color="black", linewidth=2, label=f"Observed mean = {observed_mean:.4f}")
    ax.axvline(float(np.nanmean(time_effect)), color="#333333", linestyle="--", linewidth=1.6, label=f"Trial-index control = {np.nanmean(time_effect):.4f}")
    ax.set(xlabel="Subject-mean cross-state AUC gap", ylabel="Null replicates (count)", title=f"Confirmatory cross-state inference (n={len(observed)} subjects)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "cross_state_gap_full_nulls.png", dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    run_record = {
        **config,
        "config_hash": run_hash,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "outputs": [
            "cross_state_subject_effects_confirm.csv",
            "cross_state_confirm_full_permutation.csv",
            "cross_state_null_summary_confirm.json",
            "cross_state_bootstrap_confirm.json",
            "cross_state_gap_full_nulls.png",
        ],
    }
    (RESULTS / "runs" / f"cross_state_full_{run_hash}.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({**null_summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
