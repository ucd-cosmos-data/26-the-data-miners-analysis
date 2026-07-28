"""Canonical reproducibility entry point for compact final artifacts.

`full` is intentionally explicit about expensive analyses. `compact` validates
the frozen inputs, cached compact outputs, figures, and tests; it does not
falsely claim raw-data regeneration. This project directory is currently
untracked by its parent repository, so a Git clean-worktree reproduction is
reported as blocked rather than fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_PROCESSED, DATA_RAW, DOCS, PROJECT_ROOT, RESULTS  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], log: list[dict], dry_run: bool) -> None:
    start = time.perf_counter()
    item = {"command": command, "started_at_utc": datetime.now(UTC).isoformat()}
    if dry_run:
        item.update({"status": "dry_run", "runtime_seconds": 0.0})
        log.append(item)
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    item.update(
        {
            "returncode": completed.returncode,
            "status": "passed" if completed.returncode == 0 else "failed",
            "runtime_seconds": time.perf_counter() - start,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    )
    log.append(item)
    if completed.returncode:
        raise RuntimeError(f"failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "compact", "full"], default="compact")
    parser.add_argument("--reuse-cached-preprocessing", action="store_true")
    args = parser.parse_args()
    dry_run = args.mode == "dry-run"
    started = datetime.now(UTC).isoformat()
    logs: list[dict] = []
    expected_plan = (DOCS / "analysis_plan.sha256").read_text(encoding="utf-8").split()[0].lower()
    plan_ok = _hash(DOCS / "analysis_plan.md") == expected_plan
    split = DATA_PROCESSED / "subject_split.csv"
    split_ok = split.exists() and len(__import__("pandas").read_csv(split)) == 87
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    }
    reproducibility_dir = RESULTS / "reproducibility"
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    (reproducibility_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (reproducibility_dir / "pip_freeze.txt").write_text(environment["pip_freeze"], encoding="utf-8")
    if not plan_ok or not split_ok:
        raise RuntimeError("frozen plan hash or subject split validation failed")
    if args.mode == "full":
        if not DATA_RAW.exists():
            raise RuntimeError("raw data unavailable for full regeneration")
        if not args.reuse_cached_preprocessing:
            _run([sys.executable, "scripts/s2_preprocess.py"], logs, dry_run)
            _run([sys.executable, "scripts/s4_features.py"], logs, dry_run)
        _run([sys.executable, "scripts/run_cross_state_full_inference.py"], logs, dry_run)
        _run([sys.executable, "scripts/generate_physiology_figures.py"], logs, dry_run)
        _run([sys.executable, "scripts/run_deep_baselines.py"], logs, dry_run)
    elif args.mode == "compact":
        _run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], logs, dry_run)
        _run([sys.executable, "-m", "pytest", "-q", "tests"], logs, dry_run)
    outputs = [path for path in (RESULTS / "tables").glob("*") if path.is_file()] + [path for path in FIGURES.glob("*") if path.is_file()] if (FIGURES := RESULTS / "figures").exists() else []
    output_hashes = {str(path.relative_to(PROJECT_ROOT)): _hash(path) for path in outputs}
    try:
        parent_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT.parent, text=True)
        git_status = (
            "blocked_by_untracked_project_tree"
            if "?? Dreyer2023BCIData/" in parent_status
            else "clean_parent_tree"
            if not parent_status.strip()
            else "parent_tree_has_unrelated_changes"
        )
    except Exception as exc:
        parent_status = ""
        git_status = f"isolated_non_git_workspace:{type(exc).__name__}"
    if args.mode == "compact":
        status = "passed_with_cached_preprocessing"
    elif args.mode == "full" and git_status in {"clean_parent_tree", "isolated_non_git_workspace:CalledProcessError"}:
        status = "passed_with_cached_preprocessing" if args.reuse_cached_preprocessing else "passed_clean_reproduction"
    elif args.mode == "dry-run":
        status = "dry_run"
    else:
        status = "blocked_by_untracked_project_tree"
    result = {
        "status": status,
        "mode": args.mode,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "frozen_plan_hash_valid": plan_ok,
        "frozen_split_valid": split_ok,
        "raw_data_available": DATA_RAW.exists(),
        "git_clean_worktree_status": git_status,
        "commands": logs,
        "output_hashes": output_hashes,
    }
    (reproducibility_dir / "final_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "output_hashes"}, indent=2))


if __name__ == "__main__":
    main()
