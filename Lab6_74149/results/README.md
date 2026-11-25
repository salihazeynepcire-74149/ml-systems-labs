# Experiment outputs

This folder is intentionally empty in the submitted starter package because *real Fashion-MNIST experiments could not run in the preparation environment*. The software requires the official dataset and the MLflow package.

Run `python run_all.py --only all` once the requirements are installed and the dataset is available. The following authentic measurements will then be generated:

- `01_baseline/`: epoch CSV, model checkpoint, test metrics.
- `02_profile/`: full-epoch profiling CSV, PyTorch top ops, breakdown PNG.
- `03_workers/`: 0/1/2/4/8 workers table and plot.
- `04_batches/`: 5-epoch batches 32/64/128/256/512 accuracy, throughput, memory.
- `05_optimizers/`: 10-epoch, 2x3 optimizer/scheduler comparison and plot.
- `06_accumulation/`: actual batch 256 vs microbatch 64 x 4 training curves and differences.
- `07_resume/`: interruption after epoch 3, checkpoint, resumed and uninterrupted curves.

Do not submit synthetic smoke-test CSVs as measured real-data results.
