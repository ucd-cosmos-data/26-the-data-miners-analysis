# Analysis decisions and gates

| Decision | Rationale | Status |
|---|---|---|
| Freeze the 30/57 subject split before modelling | Prevent subject-level selection leakage | Immutable |
| Define primary state from pre-cue signal only | Avoid defining state from the tested MI representation | Immutable |
| Use within-run state permutations | Preserve run composition and time-on-task structure | Immutable |
| Treat subject as the group inference unit | Trials are repeated measurements, not independent participants | Immutable |
| Exclude replayed margins from inference | CP1 reconstruction validation failed | Gate active |
| Retain recorded TAcc only as secondary | It is externally recorded but noisy at 40 trials/run | Gate active |
| Gate cVAE off | CP3 failed | Gate active |
| Label MI-window state as circularity ablation | It leaks representation-period information | Never primary |
| Preserve reduced-compute tables | They are provenance, not full inference | Retained |

Invalidated full-inference attempts are kept under `results/runs/` with a
machine-readable reason. They are not scientific results.
