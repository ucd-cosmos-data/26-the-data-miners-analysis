# Methods

The frozen primary analysis estimates a within-subject state only from the
pre-cue interval `[-3.0, -0.5]` s. Its input features are posterior alpha,
frontal-midline theta, spectral aperiodic features, alpha-peak frequency,
slow-eye/blink measurements, and EMG. A within-subject PCA observation model
is smoothed by a scalar AR(1) Kalman model. Sensorimotor idle power is stored
separately and excluded from this latent.

MI covariance matrices are Ledoit-Wolf regularized and represented in
Riemannian tangent space. The SCTM contains class, state, class-by-state,
within-run trial time, run time, blink, EMG, and artifact-flag terms. Only the
class-by-state term is differential warping. A class-common state term is
ordinary non-stationarity, not evidence of state-dependent coding.

The confirmation split consists of 57 subjects; the remaining 30 are
exploratory. The cross-state endpoint splits low/high state separately within
each run and uses the fixed trial-parity held-out rule. Subject is the unit for
bootstrap confidence intervals and group null tests. State permutations are
only within run.

CP1 failed: replayed online-decoder margins cannot enter behavioral inference.
The remaining behavioral outcome is recorded run-level TAcc, marked secondary.
CP3 failed: conditional-VAE training is gated off. Deep networks, when run,
are pooled GroupKFold decoding baselines only and do not replace the SCTM
endpoint.
