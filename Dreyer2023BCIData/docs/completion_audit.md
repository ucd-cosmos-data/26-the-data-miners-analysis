# Completion audit

Audit timestamp: 2026-07-27 (updated after accepted full confirmatory inference).

Statuses:
`complete_and_validated`, `implemented_not_executed`, `partially_executed`,
`missing`, `gated_off`, `not_applicable_due_to_CP1`, `not_applicable_due_to_CP3`.

| Plan section | Required output | Implementation status | Execution status | Validation status | Existing artifact path | Missing work | Resolution |
|---|---|---|---|---|---|---|---|
| S0/S1 QC | Manifest, metadata, dictionary | complete_and_validated | complete_and_validated | complete_and_validated | `data/interim/`, `docs/data_dictionary.md` | none | retain |
| Split freeze | 30/57 split + hash | complete_and_validated | complete_and_validated | hash test passes | `subject_split.csv`, `docs/subject_split.json` | none | retain |
| Analysis plan | Frozen plan + sha256 | complete_and_validated | complete_and_validated | hash test passes | `docs/analysis_plan.md` | none | retain |
| S2 primary preprocess | Dual-stream epochs | complete_and_validated | complete_and_validated | 20,792 trials | `data/processed/epochs/` | none | retain |
| S3 replay / CP1 | Replay validation | complete_and_validated | complete_and_validated | CP1 failed as required | `decoder_replay_summary.json` | none | gate active |
| Behavioural inference | Trial margins | not_applicable_due_to_CP1 | not_applicable_due_to_CP1 | margins excluded | `behavioural_endpoint_note.md` | none | keep excluded |
| Run-level TAcc | Secondary GLM | complete_and_validated | complete_and_validated | secondary only | `run_tacc_binomial_glm.csv` | none | retain |
| S4/S5 state | Features + d=1 latent | complete_and_validated | complete_and_validated | CP2-style AC checks | `state_features.parquet`, `state_latent.parquet` | none | retain |
| SCTM / cross-state reduced | Preliminary confirm tables | complete_and_validated | complete_and_validated | null endpoint retained | `cross_state_confirm.csv`, `sctm_summary_confirm.json` | none | provenance |
| Full confirmatory inference | 1,000 perm + 10,000 bootstrap artifacts | complete_and_validated | complete_and_validated | frozen mean matched; null retained | `cross_state_*_confirm*`, `cross_state_gap_full_nulls.png`, `runs/cross_state_full_e90eda294b47248d.json` | none | accepted |
| No-EOG ablation | Separate preprocess + evaluation | complete_and_validated | complete_and_validated | 57 confirm subjects | `eog_preprocessing_ablation.csv` | none | executed |
| MI-window circularity | Isolated circularity branch | complete_and_validated | complete_and_validated | labelled invalid for primary | `mi_window_state_circularity_ablation.csv` | none | executed |
| Latent d=2/d=3 | Fixed dimension series | complete_and_validated | complete_and_validated | summary written | `latent_dimension_ablation*.csv` | none | executed |
| Template/nuisance grid | Dedicated sensitivity table | complete_and_validated | complete_and_validated | consolidated | `template_nuisance_ablation_*`, `ablation_summary_full.csv` | none | executed |
| Deep baselines | EEGNet / ShallowConvNet GroupKFold | complete_and_validated | complete_and_validated | no subject leakage by GroupKFold | `deep_baselines_*.csv`, training curves | none | executed |
| cVAE | Trainable only if CP3 passes | gated_off | gated_off | shape test only | `src/models/cvae.py` | none | remain gated |
| Physiology figures | Matched ERD/ERS + topographies | complete_and_validated | complete_and_validated | cluster table + manifest | `figure_erd_ers_*.png`, topographies, full-null figure | none | executed |
| Leakage/gate tests | Invariant suite | complete_and_validated | complete_and_validated | 13 tests pass | `tests/test_pipeline_invariants.py` | none | executed |
| Reproducibility | Orchestrator + validation report | complete_and_validated | complete_and_validated | compact passed; clean worktree blocked | `scripts/reproduce_final_results.py`, `results/reproducibility/` | dedicated clean clone if required | documented |
| Documentation | methods/results/limitations/final report | complete_and_validated | complete_and_validated | updated for accepted full inference | `docs/*` | none | this update |

## Hard blockers (evidence)

1. **Clean Git worktree reproduction.** Parent repo shows `?? Dreyer2023BCIData/`
   plus unrelated staged/deleted disease-data files. Those unrelated files were
   not discarded, restored, or committed.
2. **CP1 / CP3.** Behavioural trial-margin inference and cVAE training remain
   gated off by failed checkpoints.
