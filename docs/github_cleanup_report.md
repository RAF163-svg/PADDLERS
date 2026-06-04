# GitHub Cleanup Report

## 1. Paths

- Original project path: `${HOME}/PaddleRS`
- Clean GitHub project path: `${HOME}/PaddleRS_github_clean`

## 2. Copied Content

Root files copied:

- `LICENSE`
- `MANIFEST.in`
- `README.md`
- `README_CN.md`
- `README_EN.md`
- `requirements.txt`
- `setup.py`
- `Dockerfile`
- `.pre-commit-config.yaml`
- `.style.yapf`
- `.gitignore`

Directories copied:

- `.github/`
- `deploy/`
- `docs/`
- `examples/`
- `paddlers/`
- `release/`
- `test_tipc/`
- `tests/`
- `tools/`
- `tutorials/`

## 3. Excluded Content

The clean export excluded:

- `.git/`
- nested Git metadata
- `output/`, `**/output/`, `**/output_stable/`
- `**/runs/`, `**/validation/`
- `**/checkpoints/`, `**/best_model/`, `**/latest_model/`
- `**/vdl_log/`, `**/distributed_logs/`
- `**/prediction_samples/`, `**/config_snapshot/`
- `**/__pycache__/`, `*.pyc`, `*.pyo`
- model and optimizer artifacts: `*.pdparams`, `*.pdopt`, `*.pdstates`, `*.pth`, `*.onnx`
- numpy caches: `*.npy`, `*.npz`
- logs and process files: `*.log`, `*.out`, `*.pid`

## 4. DORGM Handling

The full root `DORGM/` directory was not copied into the main GitHub source tree.

Only lightweight helper scripts were copied to:

```text
tools/dorgm/
```

Copied scripts:

- `build_gid15_dataset.py`
- `build_gid15_clustered_dataset.py`
- `build_gid15_split_dataset.py`
- `audit_gid15_split_dataset.py`

DORGM checkpoints, nested `.git`, model files, and outputs were not kept.

## 5. Old Experiment Handling

The following old or comparison experiment directories were moved inside the clean export to:

```text
archive/experiments/
```

Archived experiment directories:

- `deeplabv3_gid_nromal`
- `farseg_gid15_normal`
- `fastscnn_gid15_normal`
- `fastseg_gid15_nromal`
- `hrnet_gid15_normal`
- `deeplabv3_data_split`
- `farseg_data_split`
- `fastscnn_data_split`
- `hrnet_data_split`

The current main line remains in the official source location:

```text
paddlers/unet_data_split/
```

## 6. release/custom_code Handling

`release/custom_code/` was moved to:

```text
archive/release_custom_code_snapshot/custom_code/
```

The official GitHub version should use the main source tree as the maintained source of truth, especially:

```text
paddlers/unet_data_split/
```

The archived release snapshot is retained only for historical comparison.

## 7. Local Path Cleanup

The clean export was scanned for these local path patterns:

- `${HOME}`
- `<NAS_ROOT>`
- `<PUBLIC_USERS_ROOT>`

Result:

```text
0 matches
```

Main-line configs were converted to portable/example style:

- `paddlers/unet_data_split/train_config_singlehead_example.json`
- `paddlers/unet_data_split/train_config_multihead_example.json`
- `paddlers/unet_data_split/train_config_singlehead_baseline.json`
- `paddlers/unet_data_split/train_config_multihead.json`
- `release/configs/unet_singlehead_portable.json`
- `release/configs/unet_multihead_portable.json`

## 8. Generated GitHub Files

Generated or updated:

- `.gitignore`
- `.env.example`
- `README.md`
- `docs/github_cleanup_report.md`

`.gitignore` now covers:

- local tool files and `.env`
- Python caches
- IDE files
- training outputs
- model weights and exported models
- logs
- numpy caches
- large archives
- local dataset directories
- release output directories

`README.md` now includes:

```text
GID15 Routed UNet Experiment
```

## 9. Validation Results

Clean project size:

```text
28M ${HOME}/PaddleRS_github_clean
```

Git metadata check:

```bash
find ${HOME}/PaddleRS_github_clean -name ".git" -type d
```

Result: no `.git` directories found.

Model artifact check:

```bash
find ${HOME}/PaddleRS_github_clean -type f \( -name "*.pdparams" -o -name "*.pdopt" -o -name "*.pdstates" -o -name "*.pth" -o -name "*.onnx" \)
```

Result: no model artifacts found.

Log artifact check:

```bash
find ${HOME}/PaddleRS_github_clean -type f \( -name "*.log" -o -name "*.out" -o -name "*.pid" \)
```

Result: no log artifacts found.

Training output and cache directory check:

```bash
find ${HOME}/PaddleRS_github_clean -type d \( -name "__pycache__" -o -name "vdl_log" -o -name "distributed_logs" -o -name "output" -o -name "output_stable" -o -name "runs" -o -name "validation" -o -name "checkpoints" -o -name "best_model" -o -name "latest_model" -o -name "prediction_samples" -o -name "config_snapshot" \)
```

Result: no matching directories found.

Local absolute path check:

```bash
grep -RInE '${HOME}|<NAS_ROOT>|<PUBLIC_USERS_ROOT>' ${HOME}/PaddleRS_github_clean --exclude-dir=.git
```

Result: no local path matches found.

Large file check:

```bash
find ${HOME}/PaddleRS_github_clean -type f -size +50M
```

Result: no files larger than 50M found.

Python syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/paddlers_clean_compileall_cache python -m compileall paddlers release tools tests
```

Result: passed. Only SyntaxWarning messages were emitted from existing upstream-style code; no syntax errors were reported.

Release self-check:

```bash
bash release/scripts/check_release.sh
```

Result: failed because `GID15_DATASET_ROOT` is not set.

Failure output:

```text
GID15_DATASET_ROOT is not set
```

This is an environment/data precondition failure, not a syntax failure.

Smoke test:

Skipped because `GID15_DATASET_ROOT` is not set in the current shell.

## 10. GitHub Upload Recommendation

Before staging, inspect:

```bash
cd ${HOME}/PaddleRS_github_clean
git status
find . -type f -size +50M
git check-ignore -v <suspicious-file>
```

Then initialize and push manually:

```bash
cd ${HOME}/PaddleRS_github_clean
git init
git status
git add .
git commit -m "Initial clean PaddleRS GID15 routed UNet project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Do not upload datasets, checkpoints, model weights, optimizer states, logs, or training outputs.
