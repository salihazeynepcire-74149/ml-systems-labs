# Vistula University - Machine Learning Lab 6 (CII5SE039CI) | 74149

**Student:** Saliha Zeynep Cire | **Group:** 01 | **Lab date:** 10 November 2026.

This project implements the 8 tasks specified on **page 8 of the uploaded Lab 6 manual**. The CNN class is extracted directly from the supplied Lab 3 notebook. It achieved 90.03% test accuracy with 20,490 parameters (MLP 87.63%, attention 83.15%). No trained checkpoint was included; Lab 6 retrains this architecture. See `docs/lab3_evidence/`.

## Individual project proposal

**AI-Based Flower Recognition**. The new dataset requires instructor approval. The Lab 6 training experiments remain on Fashion-MNIST; the separate project is a proposal, not a completed application. See the updated PDF in `docs/`.

## Quick start (Mac, Windows, Linux; Python 3.10+)

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python run_all.py --only baseline
python run_all.py --only profile
python run_all.py --only workers
python run_all.py --only batches
python run_all.py --only optimizers
python run_all.py --only accumulation
python run_all.py --only resume
# Or run all seven coding components:
python run_all.py --only all
mlflow ui --backend-store-uri ./mlruns
```

Run from the project directory. Official Fashion-MNIST downloads automatically if your machine is online. Set `data.download: false` after the data is cached. The initial run will require network access to download Fashion-MNIST.

**Do NOT use `--smoke` as submission evidence.** It uses synthetic randomly generated tensors solely for code verification; no academic accuracy or throughput conclusions should be drawn from it.

For a first code check without network or MLflow: `python run_all.py --only all --smoke`. Synthetic outputs are labeled in `results/SMOKE_SYNTHETIC_NOT_FOR_SUBMISSION`.

## Deliverables and requirements

| Lab manual | Code | Evidence |
|---|---|---|
| [20] YAML, seed, checkpoints, MLflow | `config.yaml`, `src/core.py`, baseline | `01_baseline/`, `mlruns/` after real run |
| [15] torch.profiler one epoch | `src/experiments.py` `profile_epoch` | `02_profile/` CSV / PNG / top-ops |
| [15] workers 0/1/2/4/8 | `workers_study` | `03_workers/` CSV / PNG |
| [15] batches 32..512 5 epochs | `batch_study` | `04_batches/` CSV / 3 PNGs |
| [15] SGD+momentum / Adam x 3 LR schedules, 10 epochs | `optimizer_study` | `05_optimizers/` CSV / PNG |
| [10] effective 256 = 64 x 4 | `accumulation_study` | `06_accumulation/` CSV / PNG |
| [10] interrupt epoch 3, resume | `resume_study` | `07_resume/` CSV / PNG / checkpoint |
| [15] one-page proposal | PDF in `docs/` | teacher signature still needed |

## Important experimental details

- Split: original 60,000 Fashion-MNIST training data divided using seed 42 into 55,000 train and 5,000 validation; official separate 10,000 test set. Original Lab 3 normalization `(0.2860,), (0.3530,)`.
- CNN: Conv(1,16) / Pool / Conv(16,32) / Pool / Linear(1568,10) per Lab 3 manual. No dropout or batch norm; precise 256-vs-64x4 comparison is meaningful.
- Determinism: seeds Python, NumPy, PyTorch, DataLoader. Scheduler, optimizer and RNG states are part of checkpoints. GPU operations can still differ slightly across hardware versions.
- Accumulation uses cross-entropy summed across each **complete effective batch** and scales by exact sample count, including the last partial batch.
- Val metrics determine comparison. The official test set is evaluated only after each run's last training epoch; it is not used for early stopping.
- For GPU: peak CUDA allocated memory is measured. For CPU: **highest sampled process RSS**, not true peak per-operator memory. The labeling avoids overstating accuracy.
- DataLoader worker counts above 0 are slower or unstable in some notebook/Windows environments: run this as a script, not interactively, and use workers 0 if debugging. Do not remove worker settings from the graded table.
- `torch.profiler` captures full one epoch; GPU activities on CUDA and CPU activities otherwise. Profiling and standard run time are not directly comparable due to profiler overhead.
- Each experiment has independent initialization and identical seeds, but this is a controlled comparison rather than a formal multi-seed study.

## What is / is not completed

**Completed:** implementation of all training tasks, tests runnable with synthetic data, prewritten English one-page individual proposal and reporting template.

**NOT yet completed here:** actual Fashion-MNIST experiments, MLflow histories and measured plots, instructor approval, `git commit` into the student's repository. These steps require the user's machine/data/access. Do not claim the graded experimental evidence exists before the real runs.

## Suggested commands to commit after real runs

```bash
git init
git add config.yaml requirements.txt run_all.py src tests docs README.md .gitignore results/README.md
git commit -m "Lab 6 training pipeline, experiments and individual proposal"
```

The default `.gitignore` excludes large `data/`, `mlruns/` and results. If the instructor requires evidence in Git, explicitly add a curated `results` directory with `git add -f results/<file>`, or submit evidence separately; avoid committing large checkpoint files indiscriminately.
