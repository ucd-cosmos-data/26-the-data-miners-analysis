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
│   ├── 05_verify.ipynb
│   └── 06_eda_readiness.ipynb
├── results/
│   ├── figures/
│   ├── logs/
│   └── reports/
├── support/
│   ├── bci_core.py
│   └── eda_readiness.py
├── tests/
├── requirements.txt
└── README.md
```

## EDA readiness

From `bci_cleaning/`, activate the project environment and register its kernel once:

```bash
source ../../.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name bci-cleaning --display-name "Python 3 (BCI cleaning)"
```

Run notebooks `01`, `02`, `03`, `data/data_cleaning.ipynb`, `05`, and `06` in order with `python -m nbconvert --to notebook --execute --inplace <notebook>`. Begin EDA only when both `results/reports/post_clean_summary.md` and `results/reports/eda_readiness.md` report `PASS`; use `data/processed/` and retain all documented warnings.
