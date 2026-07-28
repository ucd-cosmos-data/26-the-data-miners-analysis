"""S3: replay all saved online decoders and validate against Perf_RUN scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402
from src.replay.online_decoder import replay_run  # noqa: E402


def _one(subject: str, suffix: str) -> tuple[list[dict], dict]:
    try:
        return replay_run(subject, suffix)
    except Exception as exc:  # logged so a malformed auxiliary XML cannot hide
        return [], {"subject": subject, "suffix": suffix, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--subjects", nargs="*", help="optional subset")
    args = parser.parse_args()

    runs = pd.read_parquet(DATA_PROCESSED / "runs.parquet")
    online = runs[(runs["kind"] == "online") & runs["exists"]].copy()
    if args.subjects:
        online = online[online["subject"].isin(args.subjects)]

    outcome = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(_one)(r.subject, r.suffix) for r in online.itertuples()
    )
    rows = [row for run_rows, _ in outcome for row in run_rows]
    diagnostics = [diag for _, diag in outcome]
    trials = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    if trials.empty:
        raise RuntimeError("decoder replay produced no trial outcomes")

    trial_out = DATA_PROCESSED / "trial_outcomes.parquet"
    trials.to_parquet(trial_out, index=False)
    diag.to_csv(DATA_PROCESSED / "decoder_replay_diagnostics.csv", index=False)

    replay = (
        trials.groupby(["subject", "run_index"], as_index=False)
        .agg(n_trials=("correct", "size"), replay_tacc=("correct", lambda s: 100 * s.mean()))
    )
    observed = runs[runs["kind"] == "online"][
        ["subject", "run_index", "tacc", "n_trials"]
    ]
    validation = replay.merge(observed, on=["subject", "run_index"], how="left", suffixes=("_replay", "_manifest"))
    validation["abs_error"] = (validation["replay_tacc"] - validation["tacc"]).abs()
    validation["within_2p5"] = validation["abs_error"] <= 2.5
    validation.to_csv(RESULTS / "tables" / "decoder_replay_validation.csv", index=False)

    scored = validation[validation["tacc"].notna()]
    summary = {
        "n_runs_replayed": len(validation),
        "n_scored_runs": len(scored),
        "n_complete_within_2p5": int(scored["within_2p5"].sum()),
        "prop_within_2p5": float(scored["within_2p5"].mean()),
        "median_abs_error": float(scored["abs_error"].median()),
        "mean_abs_error": float(scored["abs_error"].mean()),
        "cp1_pass": bool(scored["within_2p5"].mean() >= 0.80),
    }
    pd.DataFrame([summary]).to_json(RESULTS / "tables" / "decoder_replay_summary.json", orient="records", indent=2)
    print(pd.Series(summary).to_string())
    errors = diag[diag.get("error", pd.Series(index=diag.index, dtype=object)).notna()] if "error" in diag else diag.iloc[0:0]
    if len(errors):
        print("\nReplay errors:")
        print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
