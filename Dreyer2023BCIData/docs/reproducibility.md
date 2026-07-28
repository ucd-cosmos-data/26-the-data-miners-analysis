# Reproducibility

Run from `Dreyer2023BCIData/`:

```powershell
python scripts/reproduce_final_results.py --mode dry-run
python scripts/reproduce_final_results.py --mode compact
python scripts/reproduce_final_results.py --mode full --reuse-cached-preprocessing
```

The command verifies the frozen analysis-plan hash and split, captures Python,
platform, CPU and `pip freeze` metadata, records commands and hashes compact
outputs, and writes `results/reproducibility/final_validation.json`.

`compact` is a cached-artifact verification run and is accurately labelled
`passed_with_cached_preprocessing`; it is not a raw-data reproduction. `full`
requires raw data and executes the specified expensive stages.

Accepted full confirmatory inference must use:

```powershell
python scripts/run_cross_state_full_inference.py --permutations 1000 --bootstraps 10000 --n-jobs 1
```

Do not restrict `OMP_NUM_THREADS` / OpenBLAS / MKL for that run: forced
single-thread BLAS settings broke Riemannian/PCA parity with the frozen
endpoint on this machine. Subject checkpoints are written under
`results/interim/cross_state_full_checkpoint/<config_hash>/`.

For an isolated non-git scratch copy with junctions to large artifacts:

```powershell
python scripts/setup_isolated_repro.py
```

At audit time this project tree is untracked in a parent repository that also
contains unrelated staged/deleted files. A clean Git worktree cannot recreate
the untracked project contents, so isolated-Git reproduction is reported as
`blocked_by_untracked_project_tree`, not as a clean-clone success.
