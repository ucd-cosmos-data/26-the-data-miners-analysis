# Results summary — completion update

## Confirmatory scientific conclusion (unchanged)

The pre-declared confirmatory cross-state generalisation endpoint remains null:

- AUC gap: **0.0044**
- 95% CI (reduced-compute provenance): **[-0.0165, 0.0251]**
- Full 10,000-bootstrap CI (accepted): **[-0.0159, 0.0247]**

There is **no strong evidence for state-dependent motor-imagery coding** in the
confirmatory analysis. The conclusion has **not** been changed.

## Full confirmatory inference (accepted)

Accepted artifacts (`validity_label=full_predeclared_inference`, config hash
`e90eda294b47248d`):

- `cross_state_subject_effects_confirm.csv`
- `cross_state_confirm_full_permutation.csv`
- `cross_state_null_summary_confirm.json`
- `cross_state_bootstrap_confirm.json`
- `cross_state_gap_full_nulls.png`
- run record: `results/runs/cross_state_full_e90eda294b47248d.json`

Observed group mean gap **0.0044076093** matches the frozen
`sctm_summary_confirm.json` endpoint (difference ≈3e-11).
`conclusion_changed=false`.

Null tests (1,000 within-run replicates, subject-mean aggregation):

- state-permutation one-sided p = **0.275**
- random within-run variable one-sided p = **0.296**

Controls remain larger than the observed gap: trial-index control group mean
**0.0190** vs observed **0.0044**; H1b common-shift-removed mean **0.0020**.

Earlier non-equivalent / crashed parallel attempts remain under
`results/runs/INVALIDATED_*` and `BLOCKED_*` and must not be interpreted.

## CP1 and CP3 gates (unchanged)

- **CP1 failed.** Replayed trial-level margins remain excluded from inferential
  behavioural analyses. Recorded run-level TAcc is secondary only
  (state coefficient −0.073 log odds, p=0.321).
- **CP3 failed.** The conditional VAE remains `gated_off_due_to_CP3` and was
  not trained. Shape/import tests pass.

## Completed in this completion effort

### Ablations executed

| Ablation | Validity label | Mean differential partial R² | Mean cross-state gap |
|---|---|---:|---:|
| primary reject-excluded | sensitivity | 0.00586 | n/a (SCTM-only table) |
| include artifact-flagged | sensitivity | 0.00583 | n/a |
| open-loop R1/R2 | sensitivity | 0.01222 | n/a |
| circular sensorimotor-idle proxy | circularity, not primary | 0.00548 | n/a |
| **no-EOG-regression preprocessing** | preprocessing sensitivity | **0.01116** | **0.00818** |
| **MI-window state (circularity)** | circularity, not primary | 0.00589 | −0.01147 |
| latent d=1/2/3 components | fixed dimension series | 0.0046–0.0059 | −0.0022–0.0093 |
| template / time / nuisance grid | fixed ablation sensitivity | see summary JSON | n/a |

The MI-window circularity branch did **not** produce a clear inflation of the
differential or cross-state effects relative to the pre-cue primary analysis.
A larger effect there would not have been admissible as support for H1 anyway.

### Deep baselines executed

Pooled GroupKFold by subject, train-only normalisation, fixed seed:

- EEGNet: balanced accuracy 0.524 ± 0.036, AUC 0.548 ± 0.069 (87 subjects)
- ShallowConvNet: balanced accuracy 0.509 ± 0.019, AUC 0.526 ± 0.045

These are decoding baselines only (`decoding_baseline_not_state_inference`).

Classical run-ordered means remain: CSP+LDA 0.605; FBCSP 0.574; Riemannian MDM
0.565; tangent logreg 0.652; engineered 0.573.

### Physiology / topography suite

Generated from the matched ERD stream when available:

- `figure_erd_ers_confirm.png`, `figure_erd_ers_explore.png`
- `figure_sensor_space_topographies_confirm.png`
- `time_frequency_cluster_confirm.csv`
- `figure_manifest.csv`
- `cross_state_gap_full_nulls.png`

No cluster survived FWER 0.05 on the confirmatory contralateral−ipsilateral
difference map (smallest cluster p≈0.052). Maps are therefore physiological
sanity checks, not inferential claims from the grand average alone.

### Tests and reproducibility

- `python -m compileall -q src scripts` passed
- `python -m pytest -q tests`: **13 passed**
- Compact reproducibility: `passed_with_cached_preprocessing`
- Clean Git worktree reproduction: `blocked_by_untracked_project_tree`
  (project lives as `?? Dreyer2023BCIData/` beside unrelated staged parent-repo
  changes that were not touched)
- Isolated scratch setup:
  `scripts/setup_isolated_repro.py` → non-git scratch with junctions

## Deviations from the frozen plan

1. Full confirmatory inference must run serial (`--n-jobs 1`); Loky workers on
   this Windows stack returned non-equivalent observed gaps / native crashes.
   Restricting BLAS threads also broke frozen Riemannian/PCA parity.
2. Ablation SCTM permutations used 250 (not 1,000) for the sensitivity grid;
   they are labelled sensitivity/circularity, not confirmatory.
3. Motivation deltas remain uninterpreted because pre/post composites differ.
