# Data dictionary and S1 harmonisation notes

## Provenance

Source: `data/raw/BCI Database/Perfomances.csv`, three stacked blocks (datasets A, B, C) with semicolon separators and comma decimals.

## Harmonisation decisions

1. **Dataset B pre-session labels renamed.** B labels three columns `PRE_Nervousness` -> `PRE_Mood`, `PRE_Awakening` -> `PRE_Mindfulness`, `PRE_Concentration` -> `PRE_Motivation`. They occupy identical column positions, derive from the same questionnaire document, and share value granularity (multiples of 4, 10 and 20, implying 5-, 2- and 1-item Likert composites). Rows carry `pre_labels_harmonised = True`. **This is an inference, not a documented fact.**

2. **Sleep hours.** Dataset A uses comma decimals; dataset B mixes decimal hours with hours-and-minutes. A value with exactly two decimals forming a valid minute count is read as minutes and flagged in `pre_sleep_last_night_ambiguous`.

3. **Motivation delta is not interpretable.** `pre_motivation` has 1-item granularity and `post_motivation` has 7-item granularity; they correlate at r = 0.10 against 0.49 and 0.50 for mood and mindfulness. The column is named `delta_motivation_uninterpretable` so it cannot be used by accident.

4. **Item-level responses are unavailable.** Only composites are distributed, so composites cannot be recomputed.

## Tables

- `data/processed/subjects.parquet`: 87 rows x 86 columns, one per participant.
- `data/processed/runs.parquet`: 522 rows, one per (subject, motor-imagery run).
- `data/interim/comment_flags.csv`: 155 rows expanded from the experimenter COMMENTS column.

## Key columns

- `tacc_run3..6`: online accuracy in percent, feedback runs only. Calibration runs R1/R2 have no online score.
- `pre_*` / `post_*`: 0-100 rescaled questionnaire composites.
- `delta_mood`, `delta_mindfulness`: comparable pre-post changes.
- `pf16_*`: 16PF5 personality factors. `LEARNING_STYLE`: index of learning style. `mental_rotation_*`: spatial ability.

## Missingness (non-text columns with any missing value)

- `post_expectations`: 15 of 87
- `pre_stimulant_12h`: 14 of 87
- `interrogation`: 14 of 87
- `neuro_knowledge`: 13 of 87
- `pre_last_meal`: 13 of 87
- `meditation_practice`: 8 of 87
- `experimenter_gender`: 2 of 87
- `pre_tobacco_normal`: 2 of 87
- `pf16_A`: 1 of 87
- `pf16_B`: 1 of 87
- `pf16_C_`: 1 of 87
- `pf16_E`: 1 of 87
- `pf16_F`: 1 of 87
- `pf16_G`: 1 of 87
- `pf16_H`: 1 of 87
- `pf16_I`: 1 of 87
- `pf16_L`: 1 of 87
- `pf16_M`: 1 of 87
- `pf16_N`: 1 of 87
- `pf16_O`: 1 of 87
- `pf16_Q1`: 1 of 87
- `pf16_Q2`: 1 of 87
- `pf16_Q3`: 1 of 87
- `pf16_Q4`: 1 of 87
- `pf16_IM`: 1 of 87
- `pf16_EX`: 1 of 87
- `pf16_AX`: 1 of 87
- `pf16_TM`: 1 of 87
- `pf16_IN`: 1 of 87
- `pf16_SC`: 1 of 87
- `active`: 1 of 87
- `reflexive`: 1 of 87
- `sensory`: 1 of 87
- `intuitive`: 1 of 87
- `visual`: 1 of 87
- `verbal`: 1 of 87
- `sequential`: 1 of 87
- `global`: 1 of 87
- `tacc_run5`: 1 of 87
- `tacc_run6`: 1 of 87

## Experimenter comment flags

- subjects with at least one flagged channel: 14
- total (subject, run, channel) flags: 139
- subjects with noted session events: 5

  - `drowsiness`: ['A31', 'A6']
  - `environment`: ['A5']
  - `interruption`: ['A60', 'B72']