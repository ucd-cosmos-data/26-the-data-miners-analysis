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
