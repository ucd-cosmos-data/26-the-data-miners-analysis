"""Leakage-controlled pooled EEGNet and ShallowConvNet decoding baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.deep_models import EEGNet, ShallowConvNet  # noqa: E402
from src.paths import DATA_PROCESSED, FIGURES, RESULTS, all_subjects  # noqa: E402


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _subject_seed(subject: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{subject}".encode()).digest()[:4], "little")


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts, y_parts, group_parts, run_parts = [], [], [], []
    for subject in all_subjects():
        source = DATA_PROCESSED / "epochs" / f"{subject}_epochs.npz"
        table = DATA_PROCESSED / "epochs" / f"{subject}_trials.parquet"
        if not source.exists() or not table.exists():
            continue
        arrays = np.load(source)
        metadata = pd.read_parquet(table)
        # Downsampling from 128 to 64 Hz is deterministic and occurs after
        # primary preprocessing; spatial/temporal normalisation remains fold-fit.
        x_parts.append(arrays["mi"][:, :, ::2].astype(np.float32))
        y_parts.append((metadata["target"].to_numpy() == "right").astype(np.int64))
        group_parts.append(np.repeat(subject, len(metadata)))
        run_parts.append(metadata["run_index"].to_numpy())
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(group_parts), np.concatenate(run_parts)


def _choose_validation_subjects(train_groups: np.ndarray) -> np.ndarray:
    labels = sorted(np.unique(train_groups))
    n = max(1, round(len(labels) * 0.1))
    return np.asarray(labels[-n:])


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else np.nan,
        "left_precision": float(precision[0]),
        "left_recall": float(recall[0]),
        "left_f1": float(f1[0]),
        "right_precision": float(precision[1]),
        "right_recall": float(recall[1]),
        "right_f1": float(f1[1]),
    }


def _train_one(
    architecture: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    fold: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    val_subjects = _choose_validation_subjects(groups[train_index])
    val_mask = np.isin(groups[train_index], val_subjects)
    fit_index, val_index = train_index[~val_mask], train_index[val_mask]
    # Fit strictly on training subjects; no held-out-subject samples enter this transform.
    mean = x[fit_index].mean(axis=(0, 2), keepdims=True)
    sd = x[fit_index].std(axis=(0, 2), keepdims=True)
    sd[sd < 1e-6] = 1.0
    def transform(index: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((x[index] - mean) / sd)[:, None].astype(np.float32))

    _seed(args.seed + fold + (0 if architecture == "EEGNet" else 1000))
    model = EEGNet(samples=x.shape[-1]) if architecture == "EEGNet" else ShallowConvNet(samples=x.shape[-1])
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    fit_loader = DataLoader(TensorDataset(transform(fit_index), torch.from_numpy(y[fit_index])), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + fold))
    val_x, val_y = transform(val_index).to(device), torch.from_numpy(y[val_index]).to(device)
    curve, best_state, best_loss, wait = [], None, np.inf, 0
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).cpu())
            val_probability = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        record = {
            "architecture": architecture, "fold": fold, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "validation_loss": val_loss,
            "validation_balanced_accuracy": _metrics(y[val_index], val_probability)["balanced_accuracy"],
        }
        curve.append(record)
        if val_loss < best_loss - 1e-5:
            best_loss, wait = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    test_x = transform(test_index).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(test_x), dim=1)[:, 1].cpu().numpy()
    metrics = _metrics(y[test_index], probability)
    metric_rows = []
    for subject in np.unique(groups[test_index]):
        mask = groups[test_index] == subject
        metric_rows.append(
            {
                "architecture": architecture, "fold": fold, "subject": subject,
                "n_test": int(mask.sum()), "n_train": int(len(fit_index)),
                "n_validation": int(len(val_index)), "validation_subjects": ";".join(val_subjects),
                "test_subjects": ";".join(sorted(np.unique(groups[test_index]))),
                "runtime_seconds": time.perf_counter() - started,
                "device": str(device), "input_representation": "EOG-regressed 8-30Hz MI epoch, 64Hz downsampled",
                "normalization_fit_scope": "training_subjects_only", "seed": args.seed,
                "epochs_completed": epoch, **_metrics(y[test_index][mask], probability[mask]),
            }
        )
    metadata = {
        "architecture": architecture, "fold": fold, "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "epochs_completed": epoch, "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "runtime_seconds": time.perf_counter() - started, "test_metrics_pooled": metrics,
    }
    return curve, metric_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    _seed(args.seed)
    torch.set_num_threads(4)
    x, y, groups, _ = _load()
    splits = list(GroupKFold(n_splits=args.folds).split(x, y, groups))
    curves, metric_rows, runs = [], [], []
    for architecture in ("EEGNet", "ShallowConvNet"):
        for fold, (train, test) in enumerate(splits, start=1):
            curve, metrics, metadata = _train_one(architecture, x, y, groups, train, test, fold, args)
            curves.extend(curve)
            metric_rows.extend(metrics)
            runs.append(metadata)
            print({**metadata, "completed": True}, flush=True)
    metrics = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curves)
    metrics.to_csv(RESULTS / "tables" / "deep_baselines_fold_metrics.csv", index=False)
    summary = metrics.groupby("architecture", as_index=False).agg(
        n_subjects=("subject", "nunique"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        left_recall_mean=("left_recall", "mean"),
        right_recall_mean=("right_recall", "mean"),
        total_runtime_seconds=("runtime_seconds", "sum"),
    )
    summary["validity_label"] = "decoding_baseline_not_state_inference"
    summary.to_csv(RESULTS / "tables" / "deep_baselines_summary.csv", index=False)
    curves_df.to_csv(RESULTS / "tables" / "deep_baseline_training_curves.csv", index=False)
    (RESULTS / "runs" / "deep_baselines_run.json").write_text(json.dumps({"arguments": vars(args), "folds": runs}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for architecture, part in curves_df.groupby("architecture"):
        grouped = part.groupby("epoch")[["train_loss", "validation_loss"]].mean()
        axes[0].plot(grouped.index, grouped["train_loss"], label=architecture)
        axes[1].plot(grouped.index, grouped["validation_loss"], label=architecture)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_baseline_training_curves.png", dpi=220)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
