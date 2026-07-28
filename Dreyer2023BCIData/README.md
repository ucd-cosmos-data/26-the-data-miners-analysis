# State-dependent motor-imagery coding in Dreyer2023

This repository implements a confound-controlled analysis of whether pre-cue,
EEG-derived state modulates left/right motor-imagery representations.

## Reproducible execution order

```powershell
python scripts/s0_manifest.py --jobs 6
python scripts/s1_metadata.py
python scripts/freeze_split.py
python scripts/s2_preprocess.py --shard-index 0 --n-shards 4
# run the other three shard indices on the other machines
python scripts/s4_features.py --shard-index 0 --n-shards 4
# run remaining feature shard indices, then combine their compact artifacts
python scripts/s4_features.py --combine --n-shards 4
python scripts/s5_state_latent.py
python scripts/s3_replay.py --jobs 4
python scripts/eda.py
python scripts/s6_sctm.py --split explore
python scripts/s6_sctm.py --split confirm
python scripts/s7_controls.py --repeats 50  # retained reduced-compute provenance
python scripts/run_cross_state_full_inference.py --permutations 1000 --bootstraps 10000 --n-jobs 4
python scripts/s8_behaviour.py
python scripts/baselines.py
python scripts/interpretability.py
python scripts/s2_preprocess.py --branch no_eog_regression
python scripts/extract_ablation_features.py --branch no_eog_regression
python scripts/s2_preprocess.py --branch mi_window_circularity
python scripts/extract_ablation_features.py --branch mi_window_circularity
python scripts/run_latent_dimension_ablation.py
python scripts/run_deep_baselines.py
python scripts/generate_physiology_figures.py
python scripts/reproduce_final_results.py --mode compact
```

## Scientific guardrails

- State is computed exclusively from the pre-cue `[-3.0, -0.5]` s window.
- The headline test is a `class × state` interaction, not a class-common
  EEG shift.
- Cross-state generalisation splits state **within each run**; time-on-task
  is therefore matched by construction.
- Artifact flags are retained as covariates and sensitivity analyses, never
  silently dropped.
- The subject-level confirmatory split is frozen in
  `data/processed/subject_split.csv`.

Read `docs/analysis_plan.md` and `docs/data_dictionary.md` before interpreting
any model output.
