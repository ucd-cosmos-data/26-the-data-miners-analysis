# Dreyer EEG Dataset Cleaning and Validation

This notebook-based project inventories, validates, and conservatively cleans the Dreyer et al. EEG motor-imagery dataset. It does not perform exploratory analysis, signal preprocessing, feature engineering, visualization, or modeling.

## Project structure

The project follows the same repository convention as `group_survey/`:

```text
bci_cleaning/
├── data/
│   ├── data_cleaning.ipynb     # fail-closed cleaning phase
│   ├── documentation_rules.json
│   ├── raw -> ../../BCI Database
│   ├── interim/
│   └── processed/              # APFS copy-on-write cleaned dataset; git-ignored
├── models/
├── notebooks/
│   ├── 01_inventory.ipynb
│   ├── 02_profile.ipynb
│   ├── 03_validate.ipynb
│   └── 05_verify.ipynb
├── results/
│   ├── figures/
│   ├── logs/
│   └── reports/
├── support/bci_core.py
├── tests/
├── requirements.txt
└── README.md
```

The publisher-supplied `BCI Database/` is a separate sibling directory and remains untouched. `data/raw/` is only a relative symbolic link to it. All derived data is written to `data/processed/`; reports and audit logs are written under `results/`.

## Documentation authority

Variable meanings and structural expectations come from:

- Zenodo record: https://zenodo.org/records/8089820
- Scientific Data descriptor: https://www.nature.com/articles/s41597-023-02445-z
- The local English and French instructions, checklists, questionnaires, participant notes, OpenViBE scenarios, scripts, and channel list

The documentation defines 87 participants split across A1-A60, B61-B81, and C82-C87. A complete session contains two baselines, acquisition runs R1-R2, and online runs R3-R6. The recordings contain 32 physiological channels sampled at 512 Hz. These rules and documented exceptions are encoded in `data/documentation_rules.json`.

No meaning is inferred from a filename or column name alone. Unknown or undocumented cases are preserved unchanged and reported.

## Dependencies

Use Python 3.12. From `bci_cleaning/`:

```bash
source ../../.venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

All direct dependencies are exactly pinned. The standard library handles hashing, CSV parsing, XML, ZIP/OOXML integrity, range requests, reporting, and file operations. `openpyxl` reads the canonical workbook without rewriting it; MNE reads GDF headers and annotations with `preload=False`; `pypdf` checks PDF integrity; and the Jupyter packages execute and validate the notebooks. `pandas` is retained as the requested standard tabular dependency.

## Reproduction

Activate the environment above, remain in `bci_cleaning/`, and execute the notebooks in this order:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_inventory.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_profile.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_validate.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace data/data_cleaning.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_verify.ipynb
```

The inventory reads only the Zenodo ZIP central directory through HTTP byte ranges and compares every local publisher path, size, and CRC32 without downloading the 27.5 GB archive. SHA-256 progress is checkpointed in `results/logs/inventory_hash_cache.jsonl`.

The first three notebooks are independently runnable and read-only. `data/data_cleaning.ipynb` refuses to run when archive verification is absent or failed, a fatal/error validation issue remains, the raw SHA-256 snapshot changed, fewer than 10 GiB are free, `data/raw/` is not the expected symlink, or `data/processed/` is nonempty. Rebuilding processed data must never remove or modify anything in `BCI Database/`.

## Cleaning operations

Only these operations are authorized:

1. Frequency-band CSV files

   - Confirm the observed `CRCRLF` record-boundary artifact.
   - Remove only parser-generated completely empty rows.
   - Write UTF-8 with LF record separators.
   - Require the ordered nonempty cell matrix to remain identical.

2. `Perfomances.csv`

   - Decode losslessly as CP1252.
   - Re-serialize as UTF-8 with LF record separators.
   - Preserve the semicolon delimiter, decimal commas, blank separator rows, repeated section headers, quoted multiline fields, spelling, column order, and every cell value.
   - Require the complete parsed cell matrix to remain identical.

3. Administrative artifacts

   - Omit exact `.DS_Store` files and invalid `~$` Office owner-lock files from `data/processed/`.
   - Retain them unchanged in raw and record each exclusion in the cleaning manifest.

All GDF, XLSX, XML, configuration, PDF, DOCX, script, questionnaire, note, and miscellaneous research assets are preserved byte-for-byte. No whitespace trimming, column renaming, scientific-row deduplication, timestamp changes, interpolation, signal filtering, trial reordering, or missing-value fabrication is performed.

## Reports and logs

Generated reports are stored in `results/reports/`:

- `dataset_inventory.csv`: one row per publisher file with size, mode, readability, hashes, and duplicate flags.
- `dataset_structure.md`: type counts and the complete publisher-data tree.
- `archive_manifest.csv` and `archive_comparison.csv`: Zenodo traceability evidence.
- `documentation_summary.md`: documented structure, meanings, and exceptions.
- `tabular_profile.csv`: privacy-conscious table shape, type, missingness, formula, and duplicate counts.
- `validation_issues.csv`: stable issue register with severity, evidence, source, and resolution.
- `cleaning_manifest.csv`: one-to-one raw/processed lineage and action record.
- `cleaning_report.md`: human-readable cleaning summary.
- `post_clean_validation.csv`: independent raw and processed hash/equivalence results.

`results/logs/pipeline.jsonl` contains timestamped audit events without questionnaire free text or medication responses.

## Assumptions and unresolved issues

- `Perfomances.xlsx` is the documented canonical source for performance and questionnaire information. The similarly named CSV is never used to silently correct it.
- Publisher column names and the three-section table layout are preserved. Documentation-backed aliases are descriptive only.
- A1 and A59 exceptions are accepted only as documented. A9/A11 frequency-selection behavior, Dataset B questionnaire losses, and C83 questionnaire losses are reported without correction.
- C83's absent configuration files remain an unresolved warning.
- A40 R3 contains 33 trial-start, 14 left-cue, 18 right-cue, and 32 trial-end events rather than 40/20/20/40. Its GDF matches Zenodo and remains byte-identical; no event is added, removed, or relabeled.
- Noisy channels, noisy trials, experimenter comments, and unusual scientific values are intentionally retained.

## Current verified result

- Raw inventory: 1,073 publisher files, 34,164,095,627 bytes, with no empty or unreadable files.
- Zenodo archive: 1,071 members matched by path, size, and CRC32; two `.DS_Store` files were classified as administrative extras.
- Recording validation: all 694 GDF files opened at 512 Hz with 32 channels.
- Cleaning: 982 files preserved byte-for-byte, 85 frequency CSVs normalized, one performance CSV normalized, and five administrative artifacts omitted.
- Final verification: all 1,073 raw hashes unchanged, all 1,068 processed files traceable, all 694 GDF hashes identical, and all participant IDs preserved.
- Final status: `PASS`, with the A40 R3 and C83 warnings retained in `results/reports/validation_issues.csv`.

The processed dataset is complete only when `results/reports/post_clean_summary.md` reports `PASS`.
