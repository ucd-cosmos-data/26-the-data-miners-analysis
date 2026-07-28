"""Full subject-level inference for the frozen confirmatory cross-state gap.

The unit of generalisation is the subject. Trial-level quantities are used
only to build one gap per subject; confidence intervals and null tests are
then calculated across subjects. All state permutations occur *within run*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.sctm import cross_state_gap_features, tangent_features


@dataclass
class SubjectCrossStateResult:
    subject: str
    dataset: str
    observed: dict[str, float | int | str]
    h1b_removed: dict[str, float | int | str]
    permutation: np.ndarray
    random: np.ndarray
    time_control: float
    failures: dict[str, int]


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware AUC without sklearn estimator overhead in permutation loops."""
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return np.nan
    ranks = rankdata(score, method="average")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ridge_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exact frozen decoder: train-only StandardScaler + RidgeClassifier."""
    classifier = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    classifier.fit(x_train, y_train)
    return classifier.decision_function(x_test)


def _common_shift_residual(
    x: np.ndarray,
    z: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Remove an across-class linear state shift using only training-parity rows."""
    design = np.column_stack([np.ones(fit_mask.sum()), z[fit_mask]])
    coef = np.linalg.pinv(design) @ x[fit_mask]
    all_design = np.column_stack([np.ones(len(z)), z])
    # Retain the intercept and remove only the common state slope.
    return x - all_design[:, 1:2] @ coef[1:2]


def cross_state_gap_from_tangent(
    x: np.ndarray,
    frame: pd.DataFrame,
    z_state: np.ndarray,
    *,
    remove_common_shift: bool = False,
) -> dict[str, float | int | str]:
    """Frozen within-run low/high, parity-held-out bidirectional AUC gap."""
    valid = ~frame["reject_flag"].astype(bool).to_numpy()
    x = x[valid]
    frame = frame.loc[valid].reset_index(drop=True)
    z_state = np.asarray(z_state)[valid]
    if len(frame) < 60:
        return {"status": "insufficient_trials", "n_valid_trials": int(len(frame))}

    state_high = np.zeros(len(frame), dtype=bool)
    for _, index in frame.groupby("run_index", sort=False).groups.items():
        values = z_state[np.asarray(list(index), dtype=int)]
        state_high[np.asarray(list(index), dtype=int)] = values >= np.median(values)
    y = (frame["target"].to_numpy() == "right").astype(int)
    held_out = (frame["trial"].to_numpy() % 2) == 0
    x_eval = x
    if remove_common_shift:
        x_eval = _common_shift_residual(x, z_state, ~held_out)

    out: dict[str, float | int | str] = {
        "status": "ok",
        "n_valid_trials": int(len(frame)),
        "n_train_parity": int((~held_out).sum()),
        "n_test_parity": int(held_out.sum()),
    }
    gaps: list[float] = []
    for name, train_high in (("low_to_high", False), ("high_to_low", True)):
        train = (~held_out) & (state_high == train_high)
        same = held_out & (state_high == train_high)
        cross = held_out & (state_high != train_high)
        if min(train.sum(), same.sum(), cross.sum()) < 10 or np.unique(y[train]).size < 2:
            return {
                "status": "invalid_fold",
                "n_valid_trials": int(len(frame)),
                "n_train": int(train.sum()),
                "n_same": int(same.sum()),
                "n_cross": int(cross.sum()),
            }
        same_auc = _auc(y[same], _ridge_scores(x_eval[train], y[train], x_eval[same]))
        cross_auc = _auc(y[cross], _ridge_scores(x_eval[train], y[train], x_eval[cross]))
        if not np.isfinite(same_auc) or not np.isfinite(cross_auc):
            return {"status": "undefined_auc", "n_valid_trials": int(len(frame))}
        out[f"{name}_within_auc"] = same_auc
        out[f"{name}_cross_auc"] = cross_auc
        out[f"{name}_gap"] = same_auc - cross_auc
        gaps.append(same_auc - cross_auc)
    out["cross_state_gap"] = float(np.mean(gaps))
    return out


def _permute_within_run(frame: pd.DataFrame, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = np.asarray(values, dtype=float).copy()
    for _, index in frame.groupby("run_index", sort=False).groups.items():
        ix = np.asarray(list(index), dtype=int)
        permuted[ix] = rng.permutation(permuted[ix])
    return permuted


def _random_within_run(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(len(frame), dtype=float)
    for _, index in frame.groupby("run_index", sort=False).groups.items():
        ix = np.asarray(list(index), dtype=int)
        values[ix] = rng.normal(size=len(ix))
    return values


def subject_full_inference(
    subject: str,
    dataset: str,
    covariances: np.ndarray,
    frame: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> SubjectCrossStateResult:
    """Run all frozen null constructions for one confirmation subject."""
    frame = frame.reset_index(drop=True)
    # Historical frozen endpoint: drop rejected trials before tangent fit.
    valid = ~frame["reject_flag"].astype(bool).to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    covariances = covariances[valid]
    frame = frame.copy()
    frame["reject_flag"] = False
    x = tangent_features(covariances)
    z = frame["z_state"].to_numpy(dtype=float)
    # Call the frozen endpoint directly for the primary observed statistic and
    # every state-split control. The local helper remains available for the
    # explicit H1b residualisation control only.
    observed = cross_state_gap_features(x, frame)
    observed.update(
        {
            "n_valid_trials": int(len(frame)),
            "n_train_parity": int(((frame["trial"].to_numpy() % 2) != 0).sum()),
            "n_test_parity": int((frame["trial"].to_numpy() % 2 == 0).sum()),
        }
    )
    h1b = cross_state_gap_from_tangent(x, frame, z, remove_common_shift=True)
    time_z = frame.groupby("run_index", sort=False)["trial"].transform(lambda s: s - s.mean()).to_numpy(dtype=float)
    time_frame = frame.copy()
    time_frame["z_state"] = time_z
    time = cross_state_gap_features(x, time_frame)
    rng = np.random.default_rng(seed)
    permutation = np.full(n_permutations, np.nan)
    random = np.full(n_permutations, np.nan)
    failures = {"permutation": 0, "random": 0}
    for index in range(n_permutations):
        perm_frame = frame.copy()
        perm_frame["z_state"] = _permute_within_run(frame, z, rng)
        rand_frame = frame.copy()
        rand_frame["z_state"] = _random_within_run(frame, rng)
        perm = cross_state_gap_features(x, perm_frame)
        rand = cross_state_gap_features(x, rand_frame)
        if perm.get("status") == "ok":
            permutation[index] = float(perm["cross_state_gap"])
        else:
            failures["permutation"] += 1
        if rand.get("status") == "ok":
            random[index] = float(rand["cross_state_gap"])
        else:
            failures["random"] += 1
    return SubjectCrossStateResult(
        subject=subject,
        dataset=dataset,
        observed=observed,
        h1b_removed=h1b,
        permutation=permutation,
        random=random,
        time_control=float(time.get("cross_state_gap", np.nan)),
        failures=failures,
    )


def bootstrap_subject_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    """Bootstrap one subject-level statistic, resampling subjects only."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot = values[indices].mean(axis=1)
    return boot, {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_subjects": int(len(values)),
    }
