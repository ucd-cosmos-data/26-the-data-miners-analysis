# Results

The frozen confirmatory conclusion is a null for practically meaningful
state-dependent motor-imagery coding. The endpoint estimate is an average
bidirectional cross-state AUC gap of 0.0044. The reduced-compute subject-
bootstrap 95% CI was `[-0.0165, 0.0251]`. The accepted full 10,000-bootstrap
CI is `[-0.0159, 0.0247]`. Both intervals include zero.

Full 1,000-permutation confirmatory inference is accepted
(`validity_label=full_predeclared_inference`, config hash `e90eda294b47248d`).
The observed group mean exactly matches the frozen
`sctm_summary_confirm.json` value. State-permutation and random-variable
one-sided p-values are 0.275 and 0.296. `conclusion_changed=false`.

The within-run trial-index control (group mean 0.0190) is larger than the
observed state-split estimate, preventing a clean separation of state from
time-on-task. The class-by-state SCTM coefficient remains small and is not
considered sufficient evidence without the cross-state null result above.

CP1 failed: 109/336 replayed decoder runs matched online TAcc within 2.5
points. Trial-level replayed margins are excluded. Secondary recorded run-level
TAcc has a state coefficient of -0.073 log odds (p=0.321).
