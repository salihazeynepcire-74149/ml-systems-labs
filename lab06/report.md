# Lab 06: Training Pipelines and Project Proposal

**Student:** Saliha Zeynep Cire  
**Student ID:** 74149

## Aim
Build a reproducible YAML-configured training pipeline, measure data loading and training trade-offs, compare optimizers and schedulers, test gradient accumulation and checkpoint recovery, and propose an individual project.

## Model and data
The baseline is Tughanbulut's Lab 03 CNN: 16- and 32-channel convolutions on Fashion-MNIST, normalized with mean 0.2860 and standard deviation 0.3530, using the same seeded 55,000/5,000 train-validation split. For another student this is a shared reference model; replace it with that student's best Lab 03 model if different. Each student configuration has a distinct random seed. Test accuracy is evaluated only for the initial baseline.

## Reproduce
Run from the repository root. The existing root requirements include PyTorch, torchvision and MLflow. PyYAML is also required. Python 3.9–3.11 is recommended for the repository's pinned versions.

```bash
pip install -r requirements.txt pyyaml
python lab06/src/run_lab06.py
mlflow ui --backend-store-uri "$(pwd)/lab06/results/mlruns"
```

The first run downloads Fashion-MNIST. GPU is recommended; this is a substantial experiment (more than 70 training epochs plus profiling). Results are hardware-dependent. The script creates CSVs, plots, PyTorch checkpoints, a profiler trace, histories, and MLflow runs under `lab06/results/`. The trace, downloaded data, and MLflow records can be large; consider Git LFS or other storage when sharing them.

## Results
**Pending execution.** Run the script and replace this paragraph with the observed profiling shares, worker bottleneck, batch-size throughput/memory/accuracy trade-offs, optimizer/scheduler comparison, accumulation difference, and interrupted-vs-resumed comparison. No measurements have been invented.

The CPU memory column uses process high-water RSS and may include prior notebook/process allocations; GPU measurements use peak allocated tensor memory. Profiling synchronizes GPU operations and therefore affects absolute timing. The script retains its checkpoint and MLflow history, including a distinct resumed run.

## Individual project proposal: Detecting plant leaf diseases

**Student:** Saliha Zeynep Cire | **ID:** 74149

**Problem and motivation.** Develop a compact model for detecting plant leaf diseases that can be evaluated under realistic resource constraints. The initial objective is a reliable baseline, followed by an optimized version and a clear comparison of quality and deployment cost.

**Dataset.** PlantVillage leaf images. Verify licensing and availability before use; document cleaning, class balance, and exclusions. Separate training, validation and test samples, avoiding leakage across related subjects or recordings where applicable. Fit preprocessing only on the training split.

**Success metric.** Macro F1 on a held-out test split. Report a baseline and the optimized model on the same untouched test split, together with class-level errors and the accuracy change caused by optimization.

**Deployment requirements.** Target model size ≤ 5 MB and single-sample inference latency ≤ 50 ms on a specified CPU. Measure serialized model size in bytes and report median and 95th-percentile latency after warm-up, hardware, software versions, input preprocessing and sample count. These are targets, not verified results.

**Optimization and plan.** Quantize a compact CNN and evaluate on a mobile-class CPU. Train a baseline, compare the optimized model under the same evaluation protocol, profile inference, and record experiments and model artifacts. If either constraint fails, reduce model width or input size and retest, while documenting the quality trade-off. Obtain instructor approval before treating this proposal as approved.

