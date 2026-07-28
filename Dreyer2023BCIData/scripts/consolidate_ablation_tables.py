"""Consolidate executed ablation subject tables into summary artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import RESULTS  # noqa: E402


def main() -> None:
    tables = RESULTS / "tables"
    rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []

    old = tables / "sctm_ablations_confirm.csv"
    if old.exists():
        df = pd.read_csv(old)
        long_parts.append(df)
        for ablation, part in df.groupby("ablation"):
            ok = part[part["status"] == "ok"] if "status" in part else part
            rows.append(
                {
                    "ablation": ablation,
                    "validity_label": (
                        "circularity_ablation_not_valid_for_primary_inference"
                        if "circular" in str(ablation)
                        else "sensitivity_ablation"
                    ),
                    "n_subjects": int(ok["subject"].nunique()) if "subject" in ok else int(len(ok)),
                    "differential_partial_r2_mean": float(ok["differential_partial_r2"].mean()) if len(ok) else np.nan,
                    "common_state_norm_mean": float(ok["common_state_norm"].mean()) if "common_state_norm" in ok and len(ok) else np.nan,
                    "cross_state_gap_mean": np.nan,
                    "source": "sctm_ablations_confirm.csv",
                }
            )

    eog = tables / "eog_preprocessing_ablation.csv"
    if eog.exists():
        df = pd.read_csv(eog)
        long_parts.append(df.rename(columns={"branch": "ablation"}))
        ok = df[df["sctm_status"] == "ok"]
        rows.append(
            {
                "ablation": "no_eog_regression",
                "validity_label": df["validity_label"].iloc[0],
                "n_subjects": int(ok["subject"].nunique()),
                "differential_partial_r2_mean": float(ok["sctm_differential_partial_r2"].mean()),
                "common_state_norm_mean": float(ok["sctm_common_state_norm"].mean()),
                "cross_state_gap_mean": float(ok["cross_cross_state_gap"].mean()),
                "source": "eog_preprocessing_ablation.csv",
            }
        )

    circ = tables / "mi_window_state_circularity_ablation.csv"
    if circ.exists():
        df = pd.read_csv(circ)
        long_parts.append(df.rename(columns={"branch": "ablation"}))
        ok = df[df["sctm_status"] == "ok"]
        rows.append(
            {
                "ablation": "mi_window_circularity",
                "validity_label": df["validity_label"].iloc[0],
                "n_subjects": int(ok["subject"].nunique()),
                "differential_partial_r2_mean": float(ok["sctm_differential_partial_r2"].mean()),
                "common_state_norm_mean": float(ok["sctm_common_state_norm"].mean()),
                "cross_state_gap_mean": float(ok["cross_cross_state_gap"].mean()),
                "source": "mi_window_state_circularity_ablation.csv",
            }
        )

    lat_summary = tables / "latent_dimension_ablation_summary.csv"
    lat = tables / "latent_dimension_ablation.csv"
    if lat.exists():
        df = pd.read_csv(lat)
        df = df.copy()
        df["ablation"] = [
            f"latent_d{int(r.latent_dimension)}_component{int(r.component)}" for r in df.itertuples(index=False)
        ]
        long_parts.append(df)
    if lat_summary.exists():
        df = pd.read_csv(lat_summary)
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "ablation": f"latent_d{int(row.latent_dimension)}_component{int(row.component)}",
                    "validity_label": "fixed_dimension_sensitivity_ablation",
                    "n_subjects": int(row.n_subjects),
                    "differential_partial_r2_mean": float(row.differential_partial_r2),
                    "common_state_norm_mean": float(row.common_norm),
                    "cross_state_gap_mean": float(row.cross_state_gap),
                    "source": "latent_dimension_ablation_summary.csv",
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "ablation_summary_full.csv", index=False)
    if long_parts:
        long = pd.concat(long_parts, ignore_index=True, sort=False)
        long.to_csv(tables / "ablation_results_full.csv", index=False)
        print({"summary_rows": len(summary), "long_rows": len(long)})
    else:
        print({"summary_rows": len(summary), "long_rows": 0})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
