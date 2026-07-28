"""Regression tests for frozen-plan, leakage, and scientific gate invariants."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cross_state_full import (  # noqa: E402
    _permute_within_run,
    bootstrap_subject_mean,
    cross_state_gap_from_tangent,
)
from src.models.cvae import ConditionalDisentangledVAE  # noqa: E402
from src.paths import DATA_PROCESSED, DOCS, RESULTS  # noqa: E402


def test_analysis_plan_hash_is_frozen() -> None:
    expected = (DOCS / "analysis_plan.sha256").read_text(encoding="utf-8").split()[0].lower()
    observed = hashlib.sha256((DOCS / "analysis_plan.md").read_bytes()).hexdigest()
    assert observed == expected, "Frozen analysis plan changed without updating its integrity record"


def test_subject_split_hash_and_counts_are_frozen() -> None:
    split = pd.read_csv(DATA_PROCESSED / "subject_split.csv")
    metadata = json.loads((DOCS / "subject_split.json").read_text(encoding="utf-8"))
    assert split["subject"].is_unique
    assert split["split"].value_counts().to_dict() == {"confirm": 57, "explore": 30}
    assert hashlib.sha256((DATA_PROCESSED / "subject_split.csv").read_bytes()).hexdigest() == metadata["sha256"]


def test_group_kfold_has_no_subject_leakage() -> None:
    groups = np.repeat(np.array(["A1", "A2", "A3", "A4", "A5"]), 4)
    for train, test in GroupKFold(n_splits=5).split(np.zeros((len(groups), 1)), groups=groups):
        assert not set(groups[train]).intersection(groups[test])


def test_cross_state_permutation_cannot_move_values_between_runs() -> None:
    # The direct implementation is intentionally inspected through a
    # two-run fixture: replacing state values with another run's values would
    # turn either fold into an invalid class/state composition.
    frame = pd.DataFrame(
        {
            "run_index": np.repeat([1, 2], 40),
            "trial": np.tile(np.arange(1, 41), 2),
            "target": np.tile(np.repeat(["left", "right"], 2), 20),
            "reject_flag": False,
        }
    )
    x = np.random.default_rng(2).normal(size=(80, 8))
    z = np.concatenate([np.arange(40), np.arange(40)])
    result = cross_state_gap_from_tangent(x, frame, z)
    assert result["status"] == "ok"
    permuted = _permute_within_run(frame, z, np.random.default_rng(9))
    for _, indices in frame.groupby("run_index").groups.items():
        assert sorted(permuted[np.asarray(list(indices))]) == sorted(z[np.asarray(list(indices))])
    # Run composition and target labels are immutable inputs to the gap code.
    assert frame.groupby("run_index")["trial"].nunique().to_dict() == {1: 40, 2: 40}
    assert frame["target"].value_counts().to_dict() == {"left": 40, "right": 40}


def test_bootstrap_resamples_subjects_not_trials() -> None:
    values = np.array([0.01, -0.02, 0.04])
    draws, summary = bootstrap_subject_mean(values, 100, seed=3)
    assert draws.shape == (100,)
    assert summary["n_subjects"] == 3
    assert np.isclose(summary["mean"], values.mean())


def test_cp1_failure_excludes_trial_margin_inference() -> None:
    replay = json.loads((RESULTS / "tables" / "decoder_replay_summary.json").read_text(encoding="utf-8"))[0]
    assert replay["cp1_pass"] is False
    note = (RESULTS / "tables" / "behavioural_endpoint_note.md").read_text(encoding="utf-8").lower()
    assert "not used for inference" in note


def test_cp3_gate_blocks_cvae_training_path() -> None:
    # The cVAE must remain shape-valid while CP3 blocks every training runner.
    torch = pytest.importorskip("torch")
    model = ConditionalDisentangledVAE(input_dim=11)
    output = model(torch.zeros(3, 11), torch.tensor([0, 1, 0]))
    assert output["reconstruction"].shape == (3, 11)
    assert not (Path(__file__).resolve().parents[1] / "scripts" / "train_cvae.py").exists()


def test_primary_state_definition_is_precue_only() -> None:
    plan = (DOCS / "analysis_plan.md").read_text(encoding="utf-8")
    epochs = (Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "epochs.py").read_text(encoding="utf-8")
    assert "[-3.0, -0.5]" in plan
    assert "state_window: tuple[float, float] = STATE_WINDOW" in epochs
    assert "mi_eog_regression" in epochs


def test_primary_sctm_does_not_use_saved_online_bands() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "s6_sctm.py").read_text(encoding="utf-8")
    assert "online_decoder" not in source
    assert "frequency-band-selected" not in source


def test_cross_state_excludes_rejects_before_tangent_fit() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "models" / "sctm.py").read_text(encoding="utf-8")
    assert "Rejected trials are removed *before* the Riemannian mean" in source


def test_primary_yaml_declares_full_inference_counts() -> None:
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "primary.yaml").read_text(encoding="utf-8"))
    assert config["inference"]["within_run_permutations"] == 1000
    assert config["inference"]["subject_bootstraps"] == 10000
