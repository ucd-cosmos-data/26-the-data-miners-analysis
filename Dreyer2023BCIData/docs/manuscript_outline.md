# Manuscript package

## Working title

Pre-cue EEG state and state-dependent sensorimotor representations in a
large-scale motor-imagery BCI dataset

## Claim boundary

The study tests whether the **class-discriminative** MI representation changes
with pre-cue state. A class-common EEG shift is reported as non-stationarity
but is not called state-dependent coding.

## Figures

1. Dataset and trial-timing schematic; participant/run inclusion flow.
2. Pre-cue state-latent loadings and example within-subject trajectories.
3. SCTM differential versus class-common state effects with subject bootstrap
   confidence intervals and within-run permutation null.
4. Within-run cross-state generalisation gap versus all controls.
5. Topographies/feature associations and negative controls.
6. Classical decoder baseline comparison.

## Tables

1. Dataset, exclusions, questionnaire missingness, and signal QC.
2. Confirmatory primary and secondary endpoints: effect sizes, bootstrap CIs,
   and permutation p-values.
3. Sensitivity analyses and all pre-specified ablations.

## Required limitations

- State self-reports have only two time points, so they are external anchors,
  not per-run labels.
- Feedback runs are closed-loop; reverse causality is possible.
- Run-level TAcc has a finite-trial noise floor and is secondary.
- EOG, EMG, and electrode quality can mimic user state; all are explicit
  nuisance variables and sensitivity analyses.
- Dataset B pre-session labels were harmonised by documented inference.

## Reproducibility checklist

- Raw-file manifest with SHA-256 hashes.
- Exact package versions in `requirements.txt`.
- Frozen subject split + SHA-256.
- One command per pipeline stage in `README.md`.
- All figures generated from files in `results/tables`, never edited manually.
