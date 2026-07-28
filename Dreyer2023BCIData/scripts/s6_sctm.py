"""S6/S7: confirmatory SCTM differential-warping and cross-state analyses."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sctm import cross_state_gap, fit_subject_sctm  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _bootstrap_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=250)
    parser.add_argument("--split", choices=["explore", "confirm", "all"], default="confirm")
    args = parser.parse_args()
    frame = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    frame = frame.merge(state[["subject", "run_index", "trial", "z_state"]], on=["subject", "run_index", "trial"], validate="one_to_one")
    frame = frame.merge(split[["subject", "split"]], on="subject", validate="many_to_one")
    if args.split != "all":
        frame = frame[frame["split"] == args.split].reset_index(drop=True)

    covs = np.load(DATA_PROCESSED / "mi_covariances.npy")
    # Covariance rows were built in the same subject order as features.
    if len(covs) != len(pd.read_parquet(DATA_PROCESSED / "state_features.parquet")):
        raise ValueError("covariance/feature rows are not aligned")
    full_features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    selected_index = full_features.index[full_features["subject"].isin(frame["subject"].unique())].to_numpy()
    covs = covs[selected_index]
    # Both frame and selected covariance rows now preserve original order.
    rows, gaps = [], []
    cursor = 0
    # Preserve the covariance construction order (dataset then numeric subject
    # ID); lexical group sorting would put A10 before A2 and misalign arrays.
    for subject in frame["subject"].drop_duplicates():
        sub = frame[frame["subject"] == subject]
        n = len(sub)
        subject_covs = covs[cursor : cursor + n]
        cursor += n
        seed = int.from_bytes(hashlib.sha256(f"20260727:{subject}".encode()).digest()[:4], "little")
        result = fit_subject_sctm(subject_covs, sub, n_perm=args.permutations, seed=seed)
        result.update({"subject": subject, "dataset": subject[0], "split": args.split})
        rows.append(result)
        gap = cross_state_gap(subject_covs, sub)
        gap.update({"subject": subject, "dataset": subject[0], "split": args.split})
        gaps.append(gap)
        print(subject, result.get("status"), gap.get("status"), flush=True)
    results = pd.DataFrame(rows)
    gap_df = pd.DataFrame(gaps)
    results.to_csv(RESULTS / "tables" / f"sctm_{args.split}.csv", index=False)
    gap_df.to_csv(RESULTS / "tables" / f"cross_state_{args.split}.csv", index=False)

    ok = results[results["status"] == "ok"]
    gap_ok = gap_df[gap_df["status"] == "ok"]
    summary = {"n_subjects": int(len(ok)), "split": args.split}
    if len(ok):
        mean, lo, hi = _bootstrap_mean(ok["differential_partial_r2"].to_numpy())
        summary.update({"mean_differential_partial_r2": mean, "partial_r2_ci_low": lo, "partial_r2_ci_high": hi, "n_subject_p_lt_0_05": int((ok["permutation_p"] < 0.05).sum())})
    if len(gap_ok):
        mean, lo, hi = _bootstrap_mean(gap_ok["cross_state_gap"].to_numpy())
        summary.update({"mean_cross_state_gap_auc": mean, "gap_ci_low": lo, "gap_ci_high": hi})
    pd.DataFrame([summary]).to_json(RESULTS / "tables" / f"sctm_summary_{args.split}.json", orient="records", indent=2)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
