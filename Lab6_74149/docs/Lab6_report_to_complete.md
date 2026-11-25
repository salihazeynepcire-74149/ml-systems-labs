# Lab 6 Experimental Report - Actual runs only

**Course:** CII5SE039CI, Group 01  
**Student:** Saliha Zeynep Cire (74149)  
**Dataset:** Fashion-MNIST 55k train / 5k validation / 10k official test  
**Model:** CNN extracted from the original Lab 3 notebook (90.03% historical test accuracy, 20,490 parameters); weights are retrained.

## 1. YAML pipeline and MLflow (20)

`config.yaml` is the single source of configuration. Save the output from `python run_all.py --only baseline` and capture the MLflow UI from `mlflow ui --backend-store-uri ./mlruns`.

## 2. Full-epoch PyTorch profile (15)

Insert `results/02_profile/profile_breakdown.csv`, the chart and selected rows of `torch_profiler_top_ops.txt` here. Explain what the largest share means. Stage shares are synchronized wall time, not exclusive profiler kernel time.

## 3. Worker study (15)

Insert `results/03_workers/workers_study.csv` and plot; report optimal workers measured on the actual machine, and discuss loader-only vs end-to-end throughput.

## 4. Batch study (15)

Insert `results/04_batches/batch_study.csv` and three plots. Memory is peak allocated CUDA memory on CUDA; CPU is **sampled process resident set size** (not exact peak memory).

## 5. Optimizer/scheduler study (15)

Insert `results/05_optimizers/optimizer_scheduler_study.csv` and curves. All six conditions are trained for equal 10-epoch budgets and start with the same seed; same configured learning rate is used unless intentionally changed.

## 6. Gradient accumulation (10)

Insert `results/06_accumulation/accumulation_comparison.csv` and curves. Explain small floating-point differences if measured.

## 7. Checkpoint/restart (10)

Insert `results/07_resume/resume_verification.csv` and curves. The checkpoint includes model weights, optimizer, scheduler, Python/NumPy/PyTorch RNG and DataLoader RNG.

## 8. Individual proposal (15)

See `docs/Individual_Project_Proposal_74149.pdf`. *Approval is a teacher action and cannot be pre-filled.*

## Integrity statement

All tables/screenshots above must be populated with actual local execution outputs. No measured performance results were invented for this package.
