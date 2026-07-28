# Frozen analysis plan

## Primary endpoint

The primary endpoint is the within-subject class-by-state interaction in the
State-Conditioned Tangent-Space Model (SCTM):

`tangent_covariance ~ class + state + class:state + run/trial time + blink + EMG + rejection`.

The `class:state` term is differential warping and is tested with a
within-run state-label permutation. The class-common state term is reported as
ordinary non-stationarity and is not interpreted as evidence for
state-dependent motor-imagery coding.

## State definition

State is estimated only from the pre-cue interval `[-3.0, -0.5]` seconds.
It is a subject-specific PCA observation model over posterior alpha,
frontal-midline theta, aperiodic spectral features, alpha peak frequency,
slow eye activity, blink proxy, and EMG, smoothed by an AR(1) Kalman model.
Sensorimotor idle mu/beta are stored separately and excluded from the primary
state latent.

## Confirmation split

The deterministic stratified 30 exploratory / 57 confirmatory split is
defined in `data/processed/subject_split.csv`; SHA-256 and seed are in
`docs/subject_split.json`. It was generated from subject ID and dataset only.

## Secondary endpoints

1. Within-run low/high state cross-generalisation gap (AUC).
2. Trial-level replayed online-decoder margin as a mixed-model outcome.
3. Run-level online accuracy only as an external, low-power secondary outcome.

## Exclusions and controls

- A59 R5/R6 do not exist.
- A40 R3 has 32 trials and is retained with its recorded trial count.
- Artifact trials are flagged rather than dropped; no-rejection and
  EOG-regressed sensitivity analyses are required.
- All state permutations are within run.
- Dataset B metadata label harmonisation is flagged as an inferred decision.
- Pre/post Mood and Mindfulness are external anchors; Motivation deltas are
  explicitly not interpreted because the pre/post composites differ.

## Decision rule

The claim is supported only if confirmation-set differential warping exceeds
the within-run permutation null with a bootstrap confidence interval excluding
zero. A common state effect alone is reported as non-stationarity. A null
result is reported as a confound-controlled null, not retrofitted into a
positive finding.
