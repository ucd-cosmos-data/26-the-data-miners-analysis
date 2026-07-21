# Cleaning Report

- Run ID: `ade01978-d864-432b-a446-fedbe6ae0dff`
- Raw source: `/Users/samarth.gaggar/COSMOS/26-the-data-miners-analysis/BCI Database`
- Cleaned tree: `/Users/samarth.gaggar/COSMOS/26-the-data-miners-analysis/bci_cleaning/data/processed`
- Raw files represented: 1,073
- Raw dataset modified: no
- GDF recordings modified: no
- Participant IDs, timestamps, runs, and trial order modified: no

## Actions

- `excluded_administrative_artifact`: 5 files
- `normalized_frequency_csv`: 85 files
- `normalized_performance_csv`: 1 files
- `preserved_byte_identical`: 982 files

## Safeguards

- Cleaning ran only after the Zenodo path/size/CRC32 comparison passed.
- The raw SHA-256 snapshot was rechecked immediately before cloning.
- CSV changes were written atomically and accepted only after parsed-cell equivalence checks.
- All unknown or scientifically meaningful file types were preserved byte-for-byte.
- Final independent hashing remains the responsibility of `05_verify.ipynb`.

## Unresolved issues

See `validation_issues.csv`. Warnings remain unchanged in the dataset and are not silently corrected.
