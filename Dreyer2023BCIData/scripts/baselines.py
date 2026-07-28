"""Run classical within-subject, run-ordered decoder baselines.

EEGNet and ShallowConvNet architectures live in ``src.models.deep_models`` and
are deliberately not trained per subject: 240 trials/person makes such a
comparison invalid. They are reserved for a pooled GroupKFold extension.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extract import ledoit_wolf_covariances  # noqa: E402
from src.paths import DATA_PROCESSED, RESULTS  # noqa: E402


def _band_features(x: np.ndarray, fs: float) -> np.ndarray:
    pieces = []
    for lo, hi in [(4, 8), (8, 12), (12, 16), (16, 24), (24, 30)]:
        filt = sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), x, axis=-1)
        pieces.append(np.log(np.var(filt, axis=-1) + 1e-12))
    return np.concatenate(pieces, axis=1)


def main() -> None:
    rows = []
    for epoch_file in sorted((DATA_PROCESSED / "epochs").glob("*_epochs.npz")):
        subject = epoch_file.name.split("_")[0]
        arrays = np.load(epoch_file)
        table = pd.read_parquet(DATA_PROCESSED / "epochs" / f"{subject}_trials.parquet")
        x, y = arrays["mi"], (table["target"].to_numpy() == "right").astype(int)
        run = table["run_index"].to_numpy()
        # Strict time-order split: early calibration/R3/R4 -> late R5/R6.
        test = run >= 5
        if test.sum() < 20 or (~test).sum() < 40 or len(np.unique(y[~test])) < 2:
            continue
        cov = ledoit_wolf_covariances(x)
        models = {
            "CSP_LDA": (
                make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                x,
            ),
            "FBCSP_proxy_logreg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)), _band_features(x, float(arrays["sfreq"]))),
            "Riemannian_MDM": (MDM(metric="riemann"), cov),
            "Tangent_logreg": (make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)), cov),
            "engineered_sensorimotor_logreg": (
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)),
                _band_features(x[:, [6, 7, 9, 10, 22, 24], :], float(arrays["sfreq"])),
            ),
        }
        for name, (model, data) in models.items():
            try:
                model.fit(data[~test], y[~test])
                accuracy = model.score(data[test], y[test])
                rows.append({"subject": subject, "dataset": subject[0], "baseline": name, "n_train": int((~test).sum()), "n_test": int(test.sum()), "accuracy": accuracy})
            except Exception as exc:
                rows.append({"subject": subject, "dataset": subject[0], "baseline": name, "error": repr(exc)})
        print(subject, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "tables" / "classical_baselines.csv", index=False)
    summary = result.groupby("baseline")["accuracy"].agg(["count", "mean", "std"]).reset_index()
    summary.to_csv(RESULTS / "tables" / "classical_baselines_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
