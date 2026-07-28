# Figure captions

1. **Dataset and quality control.** Participants, usable trials, recorded
   missing runs, and artifact flags; all exclusions are represented as flags.
2. **Pre-cue state latent.** Within-subject PCA/Kalman loadings and
   trajectories. Self-report is an external pre/post anchor only.
3. **Confirmatory SCTM effects.** Differential class-by-state and class-common
   state terms. Only the former is a state-dependent-coding test.
4. **Cross-state generalisation.** Subject-level bidirectional AUC gaps,
   within-run state-permutation and random-variable nulls, plus the
   within-run trial-index control. The full run is labelled
   `full_predeclared_inference`.
5. **Physiological sanity checks.** Contralateral/ipsilateral ERD/ERS with
   cluster-corrected maps and confirmatory sensor-space diagnostics. The
   differential/common topographies are labelled sensor-space proxies, not
   exact inverse reconstructions of PCA-compressed SCTM coefficients.
6. **Decoding baselines.** Classical and pooled GroupKFold deep-decoder
   performance. These are decoding benchmarks, not state-inference tests.
7. **Fixed sensitivity series.** EOG-regression, artifact, state-dimension,
   open-loop, and circularity ablations. MI-window state is prominently
   labelled invalid for primary inference.
