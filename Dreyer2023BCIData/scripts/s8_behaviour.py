"""S8: trial-level and run-level behavioural validation models."""

from __future__ import annotations

import sys
from pathlib import Path
import json

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def main() -> None:
    replay_summary = json.loads((RESULTS / "tables" / "decoder_replay_summary.json").read_text(encoding="utf-8"))[0]
    features = pd.read_parquet(DATA_PROCESSED / "state_features.parquet")
    state = pd.read_parquet(DATA_PROCESSED / "state_latent.parquet")
    if not replay_summary["cp1_pass"]:
        # The saved XML replay does not recover the published online TAcc
        # sufficiently well, so its trial margin cannot be claimed as an
        # externally validated behavioural outcome. Fall back to the recorded
        # run-level score and make its noise-floor limitation explicit.
        runs = pd.read_parquet(DATA_PROCESSED / "runs.parquet")
        state_run = state.groupby(["subject", "run_index"], as_index=False)["z_state"].mean()
        run = runs[(runs["kind"] == "online") & runs["tacc"].notna()].merge(state_run, on=["subject", "run_index"])
        run["run_centered"] = run.groupby("subject")["run_index"].transform(lambda x: x - x.mean())
        run["z_state"] = (run["z_state"] - run["z_state"].mean()) / (run["z_state"].std(ddof=1) + 1e-8)
        run["accuracy_proportion"] = run["tacc"] / 100
        glm = smf.glm(
            "accuracy_proportion ~ z_state + run_centered + C(subject)",
            data=run,
            family=sm.families.Binomial(),
            freq_weights=run["n_trials"],
        ).fit(cov_type="cluster", cov_kwds={"groups": run["subject"]})
        (RESULTS / "tables" / "run_tacc_binomial_glm.txt").write_text(glm.summary().as_text(), encoding="utf-8")
        pd.DataFrame(
            {"term": glm.params.index, "estimate_log_odds": glm.params.values, "p_value": glm.pvalues.values}
        ).to_csv(RESULTS / "tables" / "run_tacc_binomial_glm.csv", index=False)
        (RESULTS / "tables" / "behavioural_endpoint_note.md").write_text(
            "Decoder replay CP1 failed (published online TAcc not recovered within 2.5 points "
            "for 80% of runs). Trial-level replay margins are therefore not used for inference. "
            "The retained external behavioural analysis is run-level recorded TAcc, which is "
            "explicitly secondary because 40 trials/run gives a substantial binomial noise floor.\n",
            encoding="utf-8",
        )
        print("CP1 failed: wrote run-level recorded-TAcc model only")
        return
    trials = pd.read_parquet(DATA_PROCESSED / "trial_outcomes.parquet")
    frame = trials.merge(
        features[["subject", "run_index", "trial", "blink_amplitude", "emg_rms_20_45hz", "reject_flag"]],
        on=["subject", "run_index", "trial"],
        how="inner",
    ).merge(
        state[["subject", "run_index", "trial", "z_state"]],
        on=["subject", "run_index", "trial"],
        validate="one_to_one",
    )
    frame["trial_centered"] = frame.groupby(["subject", "run_index"])["trial"].transform(lambda x: x - x.mean())
    frame["run_centered"] = frame.groupby("subject")["run_index"].transform(lambda x: x - x.mean())
    for column in ["z_state", "trial_centered", "run_centered", "blink_amplitude", "emg_rms_20_45hz"]:
        frame[column] = (frame[column] - frame[column].mean()) / (frame[column].std(ddof=1) + 1e-8)
    # Random slopes make the population coefficient a within-subject effect.
    mixed = smf.mixedlm(
        "margin ~ z_state + trial_centered + run_centered + blink_amplitude + emg_rms_20_45hz + reject_flag",
        data=frame,
        groups=frame["subject"],
        re_formula="~z_state",
    ).fit(method="lbfgs", maxiter=300, reml=False)
    (RESULTS / "tables" / "trial_margin_mixedlm.txt").write_text(mixed.summary().as_text(), encoding="utf-8")
    pd.DataFrame(
        {
            "term": mixed.params.index,
            "estimate": mixed.params.values,
            "p_value": mixed.pvalues.reindex(mixed.params.index).values,
        }
    ).to_csv(RESULTS / "tables" / "trial_margin_mixedlm.csv", index=False)

    # Binomial mixed model on replayed trial correctness. This is an explicit
    # secondary model because real online feedback is closed-loop.
    frame["correct"] = frame["correct"].astype(int)
    model = BinomialBayesMixedGLM.from_formula(
        "correct ~ z_state + trial_centered + run_centered + blink_amplitude + emg_rms_20_45hz + reject_flag",
        {"subject_intercept": "0 + C(subject)"},
        frame,
    )
    bayes = model.fit_vb()
    terms = model.exog_names
    pd.DataFrame(
        {
            "term": terms,
            "posterior_mean_log_odds": bayes.params[: len(terms)],
            "posterior_sd": bayes.fe_sd,
        }
    ).to_csv(RESULTS / "tables" / "trial_correct_binomial_glmm.csv", index=False)
    print("wrote behavioural mixed-model tables")


if __name__ == "__main__":
    main()
