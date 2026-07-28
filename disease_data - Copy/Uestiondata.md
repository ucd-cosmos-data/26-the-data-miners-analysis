RQ4 — Is the neural coding of motor imagery state-dependent, i.e., does moment-to-moment user state (alertness, mood, stimulants, fatigue) systematically warp the sensorimotor representation?
Scientific motivation. Neural codes are increasingly understood as state- and context-dependent, but human MI research typically treats the MI representation as a fixed per-subject template. The dataset's pre/post NeXT state measures (alertness, mood, mindfulness, motivation) and stimulant/sleep logs let us ask whether the same imagined movement is encoded differently under different internal states — a direct test of state-dependent coding, and a mechanistic explanation for the large within-subject run-to-run variability (up to 27.7%).

Hypothesis. A latent-variable model conditioning the MI representation on inferred/reported state will show that a low-dimensional "state" latent significantly modulates class-conditional neural distributions, and that within-subject TAcc fluctuation across runs tracks this state latent better than time-on-task alone. Null: state adds nothing beyond a fixed per-subject template +   linear time/fatigue trend.

ML objective. Conditional latent-variable / generative modeling (state-conditioned representation), i.e., disentangling a "content" (which hand) latent from a "state/nuisance" latent.

Inputs → targets. Inputs: single-trial EEG features + run index; conditioning variables: pre/post state, alertness, sleep hours, stimulant doses. Targets: reconstruction + class separability as a function of state; per-run TAcc as external validation.

Modeling approaches. Conditional VAE / disentangled representation (content vs state factors) with adversarial or mutual-information penalties to prevent state leaking into content; or hierarchical Bayesian state-space model with a per-run latent state. Test whether decoders trained in one state degrade out-of-state.

Evaluation. Within-subject, run-ordered; cross-state generalization gap as the key metric; permutation of state labels as null. Interpretability: does the "state" latent map onto known physiology (e.g., alertness ↔ posterior alpha, fatigue ↔ frontal-midline theta)? Confound guard: pre vs post are only two timepoints — interpolate cautiously and treat run index and state jointly to avoid attributing time-on-task to "state."

Expected contribution. Direct human evidence for state-dependent sensorimotor coding, a mechanistic account of BCI performance instability, and motivation for state-adaptive decoders that track internal context.