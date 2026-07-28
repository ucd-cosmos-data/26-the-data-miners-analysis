"""Create an isolated reproducibility workspace outside the parent Git tree.

This never modifies unrelated staged/deleted parent-repo files. It copies only
the analysis code/docs/configs/tests, junctions raw data read-only, and points
processed/results either at the live cache or a fresh scratch tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT.parent
DEFAULT_SCRATCH = PARENT.parent / "dreyer2023_repro_scratch"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        ignored: set[str] = set()
        for name in names:
            if name in {".pytest_cache", "__pycache__", ".git"} or name.endswith(".pyc"):
                ignored.add(name)
            # Only ignore the project-level data/results trees, never src/data.
            if directory_path.resolve() == src.resolve() and name in {"data", "results"}:
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)


def _junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        if link.is_symlink() or link.is_junction():
            link.unlink()
        else:
            raise RuntimeError(f"refusing to replace non-junction path: {link}")
    # Windows directory junction; read-only use by convention.
    subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--reuse-cached-processed", action="store_true", default=True)
    parser.add_argument("--fresh-processed", action="store_true")
    parser.add_argument("--run-compact", action="store_true")
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / "Dreyer2023BCIData"
    _copy_tree(PROJECT, work)
    # Raw data junction
    raw_src = PROJECT / "data" / "raw"
    if not raw_src.exists():
        raise RuntimeError(f"raw data missing at {raw_src}")
    _junction(work / "data" / "raw", raw_src)
    if args.fresh_processed:
        (work / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (work / "data" / "interim").mkdir(parents=True, exist_ok=True)
        (work / "results").mkdir(parents=True, exist_ok=True)
    else:
        _junction(work / "data" / "processed", PROJECT / "data" / "processed")
        _junction(work / "data" / "interim", PROJECT / "data" / "interim")
        _junction(work / "results", PROJECT / "results")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scratch": str(scratch),
        "worktree": str(work),
        "raw_junction": str(raw_src),
        "reuse_cached_processed": not args.fresh_processed,
        "pythonhashseed": "0",
    }
    if args.run_compact:
        completed = subprocess.run(
            [sys.executable, "scripts/reproduce_final_results.py", "--mode", "compact"],
            cwd=work,
            env=env,
            text=True,
            capture_output=True,
        )
        record["compact_returncode"] = completed.returncode
        record["compact_stdout_tail"] = completed.stdout[-2000:]
        record["compact_stderr_tail"] = completed.stderr[-2000:]
        record["status"] = "passed_with_cached_preprocessing" if completed.returncode == 0 else "failed"
    else:
        record["status"] = "scratch_prepared"
    (scratch / "isolation_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
