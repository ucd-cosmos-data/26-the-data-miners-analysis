# Limitations

- Self-report state is measured only pre/post, so it is an external anchor and
  cannot establish moment-to-moment ground truth.
- The state variable is an EEG-derived proxy, not a direct measure of
  alertness, mood, fatigue, or stimulant state.
- Time-on-task remains an important confound: the within-run trial-index
  control (full-inference group mean 0.0190) is larger than the observed
  cross-state gap (0.0044).
- Feedback runs R3–R6 are closed-loop, so reverse causality is possible.
- CP1 decoder replay failure excludes replayed trial-level margins from
  inferential behavioral analyses.
- Recorded run-level TAcc has only about 40 trials per run and therefore a
  substantial binomial noise floor.
- Dataset heterogeneity, including inferred Dataset-B metadata harmonisation,
  limits generalisation.
- Within-subject state variance is limited, reducing practical power for a
  subtle interaction.
- Eye/muscle artifacts and electrode quality can be entangled with the
  EEG-derived state despite regression, covariates, and sensitivity analyses.
- Motivation deltas are not interpreted because pre/post composites are not
  metadata-equivalent.
- The confirmatory endpoint is null. All analyses added after the frozen plan
  are explicitly exploratory or sensitivity analyses.
- Physiology figures prefer the matched ERD preprocess branch when present;
  they remain sanity checks, not a substitute for the confirmatory null.
- Cross-state tangent PCA is fit on all retained subject trials before the
  parity holdout (matching the frozen endpoint). A train-parity-only PCA is
  not the accepted confirmatory definition.
- Full confirmatory inference must be run serial (`--n-jobs 1`) on this stack;
  parallel Loky workers and forced BLAS thread limits broke frozen-endpoint
  numerical parity.
- `configs/primary.yaml` is still not loaded by every script; where defaults
  differ, run records and CLI arguments are authoritative.
- True clean-clone Git reproduction remains blocked while
  `Dreyer2023BCIData/` is an untracked tree beside unrelated parent-repo
  staged changes.
