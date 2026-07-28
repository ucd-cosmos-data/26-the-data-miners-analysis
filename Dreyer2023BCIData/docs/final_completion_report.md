# Final completion report

## Executive summary

This completion effort finished the frozen Dreyer2023 pipeline without changing
the scientific conclusion. The confirmatory cross-state endpoint remains null
(AUC gap 0.0044). Full 1,000-permutation / 10,000-bootstrap confirmatory
inference is **accepted** under `validity_label=full_predeclared_inference`
(config hash `e90eda294b47248d`), exactly matching the frozen observed mean.
CP1 and CP3 gates remain active. Ablations, deep baselines, matched-stream
physiology figures, tests, and compact reproducibility are complete.

## Final scientific conclusion

**Confound-controlled null for a practically meaningful, state-dependent
distortion of the MI code in this confirmatory analysis.** No endpoint switch,
subgroup selection, or favourable reinterpretation was performed.

Full-inference support for that null:

- observed mean gap = **0.0044076093** (frozen match; Δ≈3e-11)
- subject-bootstrap 95% CI = **[-0.0159, 0.0247]**
- state-permutation one-sided p = **0.275**
- random-variable one-sided p = **0.296**
- trial-index control mean = **0.0190** (≥ observed gap)
- `conclusion_changed=false`

## Completed work

- Repository audit and gap matrix (`docs/completion_audit.md`)
- Exact frozen-endpoint full-inference runner with equivalence gate + checkpoints
- Accepted full 1k/10k confirmatory inference artifacts
- Invalidation records for non-equivalent / crashed parallel attempts
- Genuine no-EOG-regression preprocessing branch + evaluation
- MI-window circularity branch, labelled invalid for primary inference
- Template / time / nuisance ablation grid
- Latent dimensions 1/2/3 fixed series with SCTM/cross-state summaries
- Consolidated ablation tables + forest plot
- EEGNet and ShallowConvNet pooled GroupKFold baselines
- Matched ERD stream physiology figures + topography suite + figure manifest
- Expanded invariant tests (**13** passing)
- Reproducibility orchestrator, compact validation, isolated scratch setup
- Methods/results/limitations/decisions/reproducibility docs

## Remaining gated or impossible work

| Item | Status | Evidence |
|---|---|---|
| Conditional VAE training | gated_off | CP3 failed |
| Trial-margin behavioural inference | not_applicable_due_to_CP1 | CP1 failed |
| Clean Git worktree of this project | blocked | untracked `Dreyer2023BCIData/` in parent repo with unrelated staged changes |

## Commands executed (selected)

```text
python scripts/run_cross_state_full_inference.py --permutations 1000 --bootstraps 10000 --n-jobs 1
python scripts/run_template_nuisance_ablations.py
python scripts/s2_preprocess.py --branch matched_erd ...
python scripts/generate_physiology_figures.py
python scripts/consolidate_ablation_tables.py
python scripts/setup_isolated_repro.py
python -m compileall -q src scripts
python -m pytest -q tests
python scripts/reproduce_final_results.py --mode compact
```

## Tests and results

- compileall: passed
- pytest: **13 passed**
- Compact reproducibility status: `passed_with_cached_preprocessing`
- Frozen analysis-plan hash: valid
- Frozen subject split: valid (30/57)

## Output inventory (accepted full inference)

- `results/tables/cross_state_subject_effects_confirm.csv`
- `results/tables/cross_state_confirm_full_permutation.csv`
- `results/tables/cross_state_null_summary_confirm.json`
- `results/tables/cross_state_bootstrap_confirm.json`
- `results/tables/cross_state_preliminary_vs_full.json`
- `results/figures/cross_state_gap_full_nulls.png`
- `results/runs/cross_state_full_e90eda294b47248d.json`

## Old versus new numerical results

| Quantity | Value | Role |
|---|---|---|
| Frozen confirmatory gap | 0.0044 [−0.0165, 0.0251] | **retained scientific endpoint** |
| Full-inference gap + CI | 0.0044 [−0.0159, 0.0247] | accepted; same conclusion |
| Full-inference state-perm p | 0.275 | null retained |
| Trial-index control mean | 0.0190 | ≥ observed; confound caution |
| No-EOG differential R² | 0.01116 | sensitivity only |
| MI-window circularity differential R² | 0.00589 | circularity only; no clear inflation |
| EEGNet balanced accuracy | 0.524 | decoding baseline |
| ShallowConvNet balanced accuracy | 0.509 | decoding baseline |

## Deviations from the frozen plan

1. Full inference required `--n-jobs 1` for frozen-endpoint equivalence on this
   Windows stack; parallel Loky workers were non-equivalent / unstable.
2. Ablation SCTM permutations used 250 rather than 1,000.
3. Clean-clone Git reproduction blocked by untracked project tree and unrelated
   parent-repo staged deletions.

## Reproducibility status

`passed_with_cached_preprocessing` for compact verification.
Not a full raw-data regeneration claim.
Clean worktree: `blocked_by_untracked_project_tree`.
Isolated scratch: available via `scripts/setup_isolated_repro.py`.

## Git commit/tag status

No project commit or tag was created. Parent-repo unrelated staged/deleted
disease-data files and `.gitignore` changes were left untouched.

## Known limitations

See `docs/limitations.md`. Motivation deltas remain uninterpreted.

## Genuine blockers remaining

1. Place `Dreyer2023BCIData/` under a dedicated clean repository/worktree if a
   true clean-clone reproduction is required.
2. CP1/CP3 continue to gate behavioural margins and cVAE training.
