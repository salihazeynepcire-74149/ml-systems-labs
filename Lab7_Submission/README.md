# Lab 7 - Compression: Pruning and Knowledge Distillation

**Course:** Machine Learning [CII5SE039CI] | **Language:** English

## Contents

- `lab7.py` - full, reproducible experimental implementation and report/chart generator
- `requirements.txt` - dependencies
- `README.md` - precise instructions and evidence map

After a real Fashion-MNIST run, `results/` contains:

- `report.md` - complete report, efficiency-pillars table and evidence-based conclusion
- `metrics.csv` - measured accuracy, parameter counts, measured sparsity, serialized size, CPU latency and MAC proxies for **18 models** (1 teacher + 12 magnitude + 3 structured + 2 student variants)
- `accuracy_vs_sparsity.png` - mandatory global/layer-wise sparsity curves
- `structured_pruning.png` - structured filter pruning vs accuracy/CPU latency
- `dense_vs_sparse_storage.png` - measured serialization sizes (dense vs COO)
- `student_comparison.png` - plain vs distilled student and teacher

## Run on your PC (Internet required on first run)

Python 3.10+ recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python lab7.py --self-test
python lab7.py --output results
```

Fashion-MNIST is downloaded by torchvision into `data/` automatically. The **default** processes the full Fashion-MNIST train/test data: 60,000/10,000 samples, trains the approximately 1M-parameter teacher, fine-tunes each pruning variant for 3 epochs, and trains the students. An NVIDIA GPU will automatically accelerate training, where available; all reported inference latencies are measured on the same **CPU** for comparable measurements.

> **Do not submit an unexecuted template as measured experimental results.** The report and plots are populated only after the official dataset has actually been loaded and experiments have run. The first download failed in the original creation environment because external DNS was disabled; therefore the ZIP does **not** claim any invented accuracy, latency, or energy measurements.

Optional **development-only** subset command:

```bash
python lab7.py --train-limit 4000 --test-limit 1000 --output results_quick
```

Such reports are prominently marked `SUBSET RUN - NOT FULL DATASET EVIDENCE`; use the default full run for graded evidence.

## Exact mapping to your one-page lab manual (page 9)

1. Baseline (15): teacher (32, 64, 128, 144), **997,434 parameters**; accuracy, serialized size, latency measured.
2. Pruning (35): absolute magnitude global and per-layer targets **0, 25, 50, 75, 90, 95%**; gradient-masked fixed zeros during **3 epochs** fine-tuning. Structured L2 filter removal **10, 25, 50%** with physically smaller dependent layers; accuracy curves.
3. Storage + hardware timing (15): actual CPU single-example median/p95 timing, dense `torch.save` vs sparse-COO `torch.save` file size. Dense kernels do not automatically skip zero weights.
4. Distillation (30): teacher original baseline, student (16, 32, 48, 34), **99,024 parameters**; student with/without teacher, matched initialization/order/epochs; **T=4, alpha=0.5**, **KL(student || teacher)** as specifically printed in the manual.
5. Table and conclusions (20): Ch. 9 pillars **data, model, compute, energy**. Energy is explicitly **not measured** without a power meter. Dense MAC and ideal nonzero MAC proxies are named honestly.

### Interpreting results

- Target sparsity is the share of **prunable weights** removed (conv/linear weights, excluding bias), not parameter storage savings.
- For structured pruning, `target` is the fraction of **filters per convolution removed**; the resulting effective parameter reduction is different and recorded separately.
- COO storage contains indices, so it can be **larger** than dense at low sparsity.
- Unstructured pruning leaves dense layer shapes and usually does **not** improve regular CPU kernel runtime; physical structured pruning can.
- Energy (J/inference) cannot be inferred accurately from measured latency alone.
- CPU timing is **hardware-dependent**. Run on the device on which you want to report the results.

## Submission

After running the default command successfully, submit `lab7.py`, `report.md`, `metrics.csv`, and the four `.png` plots in `results/` (and this README if your instructor permits). The original manual has **no explicit filename/format requirement**; this set covers all graded evidence it lists.
