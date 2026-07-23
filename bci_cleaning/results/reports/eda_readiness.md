# EDA Readiness

- Result: **PASS**
- Processed dataset: `/Users/samarth.gaggar/COSMOS/26-the-data-miners-analysis/bci_cleaning/data/processed`
- Checks passed: 10/10
- Blocking issues: 0
- Documented source exceptions: 6
- Open validation warnings: 4

## Readiness checks

| Check | Status | Observed |
|---|---|---|
| `archive_verification` | **PASS** | passed |
| `cleaning_manifest` | **PASS** | 1073/1073 verified |
| `post_clean_traceability` | **PASS** | 1073/1073 passed |
| `processed_membership` | **PASS** | 1068 files; extras=0; missing=0 |
| `blocking_validation_issues` | **PASS** | 0 |
| `participant_directories` | **PASS** | 87 unique IDs |
| `gdf_handoff` | **PASS** | 694 files; byte-identical=True |
| `canonical_workbook` | **PASS** | 87 unique participant rows across 1 sheet(s) |
| `processed_csv_interfaces` | **PASS** | 86 files; 43798 parsed records; failures=0 |
| `raw_processed_separation` | **PASS** | raw=/Users/samarth.gaggar/COSMOS/26-the-data-miners-analysis/BCI Database; processed=/Users/samarth.gaggar/COSMOS/26-the-data-miners-analysis/bci_cleaning/data/processed |

## EDA entry points

- `data/processed/Perfomances.xlsx`: canonical participant, performance, questionnaire, and profile table. Preserve its three-section layout when importing.
- `data/processed/Signals/DATA A|B|C/<participant>/*.gdf`: unpreprocessed EEG/EOG/EMG recordings. Read with MNE and keep participant/run identifiers.
- `data/processed/Signals/**/frequency-band-selected*.csv`: UTF-8/LF-normalized MDFB outputs with scientific values unchanged.
- `results/reports/cleaning_manifest.csv`: raw-to-processed lineage for every publisher file.

## Documented source exceptions

- `A1` — R1 and R2 were extracted from a concatenated recording and lack end-of-trial and end-of-run triggers.
- `A9` — Only R1 was used for frequency-band selection; both acquisition recordings remain available.
- `A11` — Only R1 was used for frequency-band selection; both acquisition recordings remain available.
- `A59` — R5, R6, associated EEG data, and filters are missing because the participant left early.
- `B_questionnaire_loss` — Thirteen Dataset B participants have documented losses for specified questionnaire items.
- `C83` — ILS and 16PF5 results are documented as lost. Missing configuration assets remain an unresolved source warning.

## Open validation warnings

- `A40/R3` — `cue_balance`: left=14; right=18. No value was changed.
- `A40/R3` — `trial_end_count`: 32. No value was changed.
- `A40/R3` — `trial_start_count`: 33. No value was changed.
- `C83` — `configuration_files`: none. No value was changed.

A `PASS` authorizes downstream EDA on `data/processed/`; it does not imply that noisy channels, missing questionnaire values, or documented recording anomalies should be removed. Any analytical exclusion must be explicit and separate from this cleaning pipeline.
