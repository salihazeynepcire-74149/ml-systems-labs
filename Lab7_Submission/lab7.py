"""Lab 7 -- Fashion-MNIST CNN compression and knowledge distillation.

Runs all experiments specified in the supplied lab manual and writes a genuine,
measurement-based English report, tables, and plots. Requires the official
Fashion-MNIST dataset (downloaded by torchvision if network is available).

Usage:
    python lab7.py --output results
    python lab7.py --self-test
    python lab7.py --train-limit 4000 --test-limit 1000 --output results_quick

Important: quick/subset runs are explicitly labeled as such; they are NOT
represented as full 60,000/10,000 Fashion-MNIST experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import math
import os
import platform
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

TARGET_SPARSITIES = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95)
STRUCTURED_FRACTIONS = (0.10, 0.25, 0.50)
PRUNABLE = ("conv1.weight", "conv2.weight", "conv3.weight", "fc1.weight", "fc2.weight")


class CompressionCNN(nn.Module):
    """Same topology for teacher/student, with configurable channel widths.

    Input: 1x28x28 -> conv1 -> conv2 -> pool -> conv3 -> pool -> FC -> logits.
    Teacher (32,64,128,144) = 997,434 parameters.
    Student (16,32,48,34) = 99,024 parameters.
    """

    def __init__(self, c1: int, c2: int, c3: int, hidden: int):
        super().__init__()
        self.widths = (c1, c2, c3, hidden)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(c3 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.flatten(start_dim=1)
        return self.fc2(F.relu(self.fc1(x)))


def teacher_model() -> CompressionCNN:
    return CompressionCNN(32, 64, 128, 144)


def student_model() -> CompressionCNN:
    return CompressionCNN(16, 32, 48, 34)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(root: str, train_limit: int, test_limit: int, seed: int):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    try:
        tr = datasets.FashionMNIST(root=root, train=True,
                                  transform=transform, download=True)
        te = datasets.FashionMNIST(root=root, train=False,
                                  transform=transform, download=True)
    except Exception as exc:
        raise RuntimeError(
            "Fashion-MNIST could not be loaded. Connect to the Internet and retry; "
            "torchvision downloads the official dataset automatically. "
            "Do not replace it with MNIST or synthetic images for submission. "
            f"Original error: {exc}"
        ) from exc
    rng = np.random.default_rng(seed)
    if train_limit and train_limit < len(tr):
        tr = Subset(tr, sorted(rng.choice(len(tr), train_limit, replace=False).tolist()))
    if test_limit and test_limit < len(te):
        te = Subset(te, sorted(rng.choice(len(te), test_limit, replace=False).tolist()))
    return tr, te


def loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, pin_memory=False)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def total_prunable_weights(model: nn.Module) -> int:
    return sum(dict(model.named_parameters())[name].numel() for name in PRUNABLE)


def nonzero_prunable_weights(model: nn.Module) -> int:
    return sum(int(torch.count_nonzero(dict(model.named_parameters())[name]).item())
               for name in PRUNABLE)


def weight_sparsity(model: nn.Module) -> float:
    return 1.0 - nonzero_prunable_weights(model) / total_prunable_weights(model)


def masks_for_magnitude(model: nn.Module, sparsity: float, scope: str) -> dict[str, Tensor]:
    """Exactly prune the smallest-magnitude weights, globally or per tensor."""
    assert 0.0 <= sparsity < 1.0
    params = dict(model.named_parameters())
    if scope == "layerwise":
        result = {}
        for name in PRUNABLE:
            p = params[name]
            n_drop = int(round(p.numel() * sparsity))
            flat = torch.ones(p.numel(), device=p.device, dtype=p.dtype)
            if n_drop:
                smallest = torch.topk(p.detach().abs().reshape(-1), k=n_drop,
                                      largest=False).indices
                flat[smallest] = 0
            result[name] = flat.reshape_as(p)
        return result
    if scope != "global":
        raise ValueError(f"Unrecognized scope: {scope}")
    pieces = [params[name].detach().abs().reshape(-1) for name in PRUNABLE]
    absolute = torch.cat(pieces)
    n_drop = int(round(absolute.numel() * sparsity))
    flat = torch.ones_like(absolute)
    if n_drop:
        smallest = torch.topk(absolute, k=n_drop, largest=False).indices
        flat[smallest] = 0
    result = {}
    offset = 0
    for name in PRUNABLE:
        length = params[name].numel()
        result[name] = flat[offset: offset + length].reshape_as(params[name])
        offset += length
    return result


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, Tensor]) -> None:
    for name, p in model.named_parameters():
        if name in masks:
            p.mul_(masks[name])


def train_epochs(model: nn.Module, train_ds: Dataset, epochs: int,
                 device: torch.device, lr: float, batch_size: int,
                 seed: int, masks: dict[str, Tensor] | None = None,
                 teacher: nn.Module | None = None) -> list[float]:
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = loader(train_ds, batch_size, shuffle=True, seed=seed)
    if masks is not None:
        apply_masks(model, masks)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    losses: list[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        seen = 0
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            scores = model(images)
            ce = F.cross_entropy(scores, labels)
            if teacher is None:
                loss = ce
            else:
                with torch.no_grad():
                    teacher_scores = teacher(images)
                # EXACT lab-manual orientation: KL(student_soft || teacher_soft).
                # T=4 and alpha=.5, as specified, not the more common reverse KL.
                temperature = 4.0
                alpha = 0.5
                student_logp = F.log_softmax(scores / temperature, dim=1)
                teacher_logp = F.log_softmax(teacher_scores / temperature, dim=1)
                student_p = student_logp.exp()
                kl_student_teacher = (student_p * (student_logp - teacher_logp)).sum(1).mean()
                loss = alpha * ce + (1 - alpha) * temperature**2 * kl_student_teacher
            loss.backward()
            if masks is not None:
                # Gradient masking + reapplying weights keeps zeros fixed despite updates.
                for name, p in model.named_parameters():
                    if name in masks and p.grad is not None:
                        p.grad.mul_(masks[name])
            optim.step()
            if masks is not None:
                apply_masks(model, masks)
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(epoch_loss)
        print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_loss:.5f}", flush=True)
    return losses


@torch.inference_mode()
def accuracy(model: nn.Module, test_ds: Dataset, device: torch.device,
             batch_size: int) -> float:
    model.eval().to(device)
    correct, total = 0, 0
    for images, labels in loader(test_ds, batch_size, shuffle=False, seed=0):
        predictions = model(images.to(device)).argmax(dim=1)
        correct += int((predictions == labels.to(device)).sum().item())
        total += labels.shape[0]
    return correct / max(total, 1)


def structured_prune(base: CompressionCNN, proportion: float) -> CompressionCNN:
    """Physical filter/channel removal selected by original per-filter L2 norm.

    Unlike zeroing whole convolution kernels, this creates smaller dense layers,
    propagates channel choices to downstream layers, and copies matching weights.
    """
    assert 0.0 < proportion < 1.0
    device = next(base.parameters()).device
    convs = (base.conv1, base.conv2, base.conv3)
    keep: list[Tensor] = []
    for conv in convs:
        output_count = conv.out_channels
        remaining = max(1, output_count - int(round(output_count * proportion)))
        scores = torch.linalg.vector_norm(conv.weight.detach().reshape(output_count, -1),
                                          ord=2, dim=1)
        selection = torch.topk(scores, k=remaining).indices.sort().values
        keep.append(selection)
    k1, k2, k3 = keep
    new = CompressionCNN(len(k1), len(k2), len(k3), base.fc1.out_features).to(device)
    with torch.no_grad():
        new.conv1.weight.copy_(base.conv1.weight[k1])
        new.conv1.bias.copy_(base.conv1.bias[k1])
        new.conv2.weight.copy_(base.conv2.weight[k2][:, k1])
        new.conv2.bias.copy_(base.conv2.bias[k2])
        new.conv3.weight.copy_(base.conv3.weight[k3][:, k2])
        new.conv3.bias.copy_(base.conv3.bias[k3])
        # Each conv3 output channel feeds an adjacent block of 7*7 FC inputs.
        fc_columns = torch.cat([torch.arange(int(k.item()) * 49,
                                                 (int(k.item()) + 1) * 49,
                                                 device=device) for k in k3])
        new.fc1.weight.copy_(base.fc1.weight[:, fc_columns])
        new.fc1.bias.copy_(base.fc1.bias)
        new.fc2.weight.copy_(base.fc2.weight)
        new.fc2.bias.copy_(base.fc2.bias)
    return new


def dense_macs(model: CompressionCNN) -> int:
    # Multiply-accumulates for convolution / linear; excludes bias, pool and ReLU.
    return (28 * 28 * model.conv1.weight.numel()
            + 28 * 28 * model.conv2.weight.numel()
            + 14 * 14 * model.conv3.weight.numel()
            + model.fc1.weight.numel() + model.fc2.weight.numel())


def ideal_nonzero_macs(model: CompressionCNN) -> int:
    # Theoretical useful operations if every zero could be skipped at no overhead.
    # This is NOT an observed MAC reduction in normal dense PyTorch kernels.
    count = lambda m: int(torch.count_nonzero(m.weight).item())
    return (28 * 28 * count(model.conv1) + 28 * 28 * count(model.conv2)
            + 14 * 14 * count(model.conv3) + count(model.fc1) + count(model.fc2))


def storage_sizes(model: nn.Module) -> tuple[int, int]:
    """True torch.save on-disk bytes: dense state vs COO sparse weights.

    This compares equivalent tensors. Indices make COO expensive at low sparsity.
    Models with physically smaller channel dimensions have smaller dense files.
    """
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    sparse_state = {k: (v.to_sparse_coo().coalesce() if k in PRUNABLE else v)
                    for k, v in state.items()}
    with tempfile.TemporaryDirectory(prefix="lab7_sizes_") as td:
        dense_file, sparse_file = Path(td) / "dense.pt", Path(td) / "coo.pt"
        torch.save(state, dense_file)
        torch.save(sparse_state, sparse_file)
        return dense_file.stat().st_size, sparse_file.stat().st_size


def cpu_latency_ms(model: CompressionCNN, repeats: int, threads: int) -> tuple[float, float]:
    """Real batch=1 dense-PyTorch CPU wall-clock latency, median and p95."""
    torch.set_num_threads(threads)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(10):
            cpu_model(x)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            cpu_model(x)
            timings.append((time.perf_counter_ns() - start) / 1e6)
    return float(statistics.median(timings)), float(np.percentile(timings, 95))


@dataclass
class Result:
    name: str
    group: str
    target: float
    actual_weight_sparsity: float
    accuracy_pct: float
    total_parameters: int
    prunable_weights: int
    nonzero_weights: int
    dense_size_bytes: int
    coo_size_bytes: int
    cpu_median_ms: float
    cpu_p95_ms: float
    dense_macs: int
    ideal_nonzero_macs: int


def measure_model(name: str, group: str, target: float,
                  model: CompressionCNN, test_ds: Dataset,
                  device: torch.device, batch_size: int,
                  benchmark_repeats: int, cpu_threads: int) -> Result:
    acc = accuracy(model, test_ds, device, batch_size)
    dense, coo = storage_sizes(model)
    p50, p95 = cpu_latency_ms(model, benchmark_repeats, cpu_threads)
    result = Result(name=name, group=group, target=target,
                    actual_weight_sparsity=weight_sparsity(model),
                    accuracy_pct=acc * 100, total_parameters=num_params(model),
                    prunable_weights=total_prunable_weights(model),
                    nonzero_weights=nonzero_prunable_weights(model),
                    dense_size_bytes=dense, coo_size_bytes=coo,
                    cpu_median_ms=p50, cpu_p95_ms=p95,
                    dense_macs=dense_macs(model),
                    ideal_nonzero_macs=ideal_nonzero_macs(model))
    print(f"  {name}: {result.accuracy_pct:.2f}% accuracy; "
          f"{result.total_parameters:,} parameters; "
          f"{result.cpu_median_ms:.3f} ms CPU p50; "
          f"actual weight sparsity {result.actual_weight_sparsity:.3f}", flush=True)
    return result


def save_plots(frame: pd.DataFrame, output: Path) -> None:
    plt.style.use("default")
    base = frame.loc[frame.name == "Baseline teacher"].iloc[0]
    plt.figure(figsize=(8.4, 5.3))
    for grp, label, marker in [("global", "Global magnitude", "o"),
                               ("layerwise", "Layer-wise magnitude", "s")]:
        part = frame[frame.group == grp].sort_values("target")
        plt.plot(part.target.to_numpy() * 100, part.accuracy_pct,
                 marker=marker, linewidth=2, label=label)
    plt.axhline(float(base.accuracy_pct), linestyle="--", alpha=0.55,
                color="gray", label="Unpruned baseline")
    plt.xlabel("Target weight sparsity (%)")
    plt.ylabel("Fashion-MNIST test accuracy (%)")
    plt.title("Global vs. layer-wise magnitude pruning (3-epoch fine-tuning)")
    plt.xticks([0, 25, 50, 75, 90, 95])
    plt.grid(alpha=0.24)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "accuracy_vs_sparsity.png", dpi=170)
    plt.close()

    struct = frame[frame.group == "structured"].sort_values("target")
    fig, left = plt.subplots(figsize=(8.4, 5.2))
    left.plot(struct.target.to_numpy() * 100, struct.accuracy_pct,
              marker="o", linewidth=2, label="Accuracy")
    left.set_xlabel("Original convolution filters removed per layer (%)")
    left.set_ylabel("Fashion-MNIST test accuracy (%)")
    left.axhline(float(base.accuracy_pct), linestyle="--", alpha=0.5, color="gray")
    left.grid(alpha=0.2)
    right = left.twinx()
    right.plot(struct.target.to_numpy() * 100, struct.cpu_median_ms,
               linestyle="--", marker="s", label="Measured CPU latency")
    right.set_ylabel("CPU latency, batch=1, median (ms)")
    fig.suptitle("Physically structured filter pruning: quality vs. CPU latency")
    fig.tight_layout()
    fig.savefig(output / "structured_pruning.png", dpi=170)
    plt.close(fig)

    sub = frame[frame.group.isin(["teacher", "student_plain", "student_kd"])]
    labels = ["Teacher", "Plain student", "Distilled student"]
    ordered = [sub[sub.group == k].iloc[0] for k in
               ("teacher", "student_plain", "student_kd")]
    plt.figure(figsize=(8.4, 4.9))
    bars = plt.bar(labels, [item.accuracy_pct for item in ordered])
    for bar, row in zip(bars, ordered):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{row.accuracy_pct:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Fashion-MNIST test accuracy (%)")
    plt.title("Teacher vs. same-initialization students")
    plt.ylim(0, max(100, max(r.accuracy_pct for r in ordered) * 1.10))
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "student_comparison.png", dpi=170)
    plt.close()

    size_data = frame[frame.group == "global"].sort_values("target")
    plt.figure(figsize=(8.4, 5))
    plt.plot(size_data.target.to_numpy() * 100,
             size_data.dense_size_bytes.to_numpy() / 1024, marker="o",
             label="Dense state_dict")
    plt.plot(size_data.target.to_numpy() * 100,
             size_data.coo_size_bytes.to_numpy() / 1024, marker="s",
             label="Sparse COO weights + dense biases")
    plt.xlabel("Global weight sparsity (%)")
    plt.ylabel("Serialized state size (KiB)")
    plt.title("Actual dense vs. sparse COO on-disk sizes")
    plt.xticks([0, 25, 50, 75, 90, 95])
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "dense_vs_sparse_storage.png", dpi=170)
    plt.close()


def create_report(frame: pd.DataFrame, output: Path, args: argparse.Namespace,
                  train_count: int, test_count: int, device: torch.device) -> None:
    find = lambda grp: frame[frame.group == grp].iloc[0]
    b, plain, kd = find("teacher"), find("student_plain"), find("student_kd")
    mb = lambda value: value / 1024**2
    macm = lambda value: value / 1e6
    accuracy_delta = kd.accuracy_pct - plain.accuracy_pct
    speed = b.cpu_median_ms / kd.cpu_median_ms if kd.cpu_median_ms else float("nan")
    is_full = train_count == 60000 and test_count == 10000
    full_label = "FULL DATASET RUN" if is_full else "SUBSET RUN - NOT FULL DATASET EVIDENCE"
    lines = [
        "# Machine Learning - Lab 7: Compression, Pruning and Distillation",
        "",
        "**Experiment status:** " + full_label,
        "",
        "> All numerical results below were computed by this script; no placeholder results are used. "
        "If a training/test subset was requested, this report MUST NOT be described as a full-data result.",
        "",
        "## 1. Experiment setup and baseline [15 points]",
        "",
        f"- Dataset: torchvision Fashion-MNIST; **{train_count:,} training** and **{test_count:,} test** images; "
        "28x28 grayscale, 10 classes. Inputs converted to tensors and normalized to [-1, 1].",
        f"- Python/PyTorch: {platform.python_version()} / {torch.__version__}; OS: {platform.platform()}.",
        f"- Training device: {device}; **CPU** inference latency device for all comparisons; "
        f"CPU threads: {args.cpu_threads}; fixed batch=1; warm-up=10, measured repeats={args.benchmark_repeats}.",
        f"- Seed: {args.seed}; train batch size: {args.batch_size}; optimizer: AdamW "
        "(weight decay=1e-4). No test-set training or augmentation.",
        f"- Epochs: teacher={args.teacher_epochs}, fixed-mask/structured fine-tuning={args.prune_epochs}, "
        f"each student={args.student_epochs}. Baseline/student LR={args.lr}; pruning fine-tune LR={args.prune_lr}.",
        "- Teacher architecture: Conv(1->32, 3x3), Conv(32->64, 3x3), max-pool, "
        "Conv(64->128, 3x3), max-pool, Linear(6272->144), Linear(144->10), ReLU between layers.",
        "- Student architecture: Conv(1->16), Conv(16->32), max-pool, Conv(32->48), "
        "max-pool, Linear(2352->34), Linear(34->10); same 3x3 kernels.",
        f"- Teacher: **{int(b.total_parameters):,}** parameters, "
        f"accuracy **{b.accuracy_pct:.2f}%**, dense state **{mb(b.dense_size_bytes):.3f} MiB**, "
        f"CPU latency **{b.cpu_median_ms:.3f} ms median / {b.cpu_p95_ms:.3f} ms p95**.",
        "",
        "## 2. Magnitude and structured pruning [35 points]",
        "",
        "Absolute weights were ranked; the smallest were zeroed either (i) across all convolution "
        "and linear weight tensors together (global) or (ii) independently within each tensor "
        "(layer-wise). Biases were not pruned. Targets were 0%, 25%, 50%, 75%, 90%, 95%. "
        "Each pruned copy of the *same original teacher checkpoint* was fine-tuned for **3 epochs** "
        "with fixed masks: gradients at pruned indices were zeroed and masks were re-applied after "
        "every optimizer step.",
        "",
        "Structured pruning ranked **convolution output filters by L2 norm**, removed 10%, 25%, "
        "or 50% from each convolution, and **physically rebuilt narrower convolutions and the "
        "first classifier layer** with the surviving channel indices. These three derived models "
        "were also fine-tuned for 3 epochs.",
        "",
        "![Magnitude pruning accuracy curves](accuracy_vs_sparsity.png)",
        "",
        "![Physically structured pruning](structured_pruning.png)",
        "",
        "## 3. Real CPU latency vs. dense/sparse storage [15 points]",
        "",
        "Each reported latency is based on 10 warm-up forward passes and "
        f"{args.benchmark_repeats} independent wall-clock forward-pass measurements on "
        f"the CPU with {args.cpu_threads} threads; **median and p95** are reported. "
        "Measurements use dense PyTorch modules with 32-bit floats at batch size 1. "
        "Latency is hardware-/workload-specific: it is not a guaranteed device speedup. "
        "The code does not label the theoretical nonzero-MAC count as measured execution.",
        "",
        "Dense storage uses `torch.save(state_dict)`; sparse storage uses "
        "`torch.save(state_dict with weight tensors converted to coalesced COO)`. "
        "COO stores nonzero values **plus indices**: at modest sparsity it can occupy "
        "more disk space than a dense checkpoint. Full zero-filled dense tensors keep "
        "their original shapes and generally run through the **same dense CPU kernels**, "
        "so unstructured pruning rarely reduces CPU latency without a specialized sparse "
        "runtime or algorithm. Structured pruning physically changes tensor shapes, "
        "so it may reduce dense MACs and measured latency.",
        "",
        "![Actual dense vs COO storage size](dense_vs_sparse_storage.png)",
        "",
        "## 4. Knowledge distillation [30 points]",
        "",
        "The original oversized teacher was frozen. Plain and distilled students both use "
        "the **same architecture, initial weights, train order and number of epochs**. "
        "Plain student: cross-entropy only. Distilled student: "
        "`0.5 * CE(student, label) + 0.5 * 4^2 * KL(p_student(T=4) || p_teacher(T=4))`. "
        "The KL **direction exactly follows the provided lab manual**, though "
        "KL(teacher || student) is common elsewhere. Teacher logits are detached.",
        "",
        f"- Plain student: {plain.accuracy_pct:.2f}% accuracy, {int(plain.total_parameters):,} parameters, "
        f"{mb(plain.dense_size_bytes):.3f} MiB dense, {plain.cpu_median_ms:.3f} ms CPU median.",
        f"- Distilled student: {kd.accuracy_pct:.2f}% accuracy, {int(kd.total_parameters):,} parameters, "
        f"{mb(kd.dense_size_bytes):.3f} MiB dense, {kd.cpu_median_ms:.3f} ms CPU median.",
        f"- Distillation minus plain accuracy: **{accuracy_delta:+.2f} percentage points**. "
        "A positive change indicates an improvement under these training settings; "
        "a negative change must be reported rather than hidden.",
        "",
        "![Plain student vs distilled student](student_comparison.png)",
        "",
        "## 5. Chapter 9 efficiency pillars [20 points]",
        "",
        "**Energy measurements:** No wattmeter, power telemetry, or energy meter was used. "
        "Actual energy per inference (J) is therefore **not measured**, not fabricated. "
        "The table provides dense-MAC percentage relative to the teacher ONLY as a "
        "workload/energy *proxy*, not a power-consumption estimate. "
        "Latency and MAC counts are only imperfect compute/energy *proxies*; "
        "frequency, memory traffic, and device power also matter.",
        "",
        "| Model / experiment | Data pillar | Model pillar: test accuracy; parameters; dense/COO MiB | Compute pillar: CPU median/p95 ms; dense/ideal nonzero MMAC | Energy pillar |",
        "|:--|:--|:--|:--|:--|",
    ]
    for _, row in frame.iterrows():
        name = str(row["name"])
        label = f"{train_count:,} train / {test_count:,} test"
        model = (f"{row['accuracy_pct']:.2f}%; {int(row['total_parameters']):,}; "
                 f"{mb(row['dense_size_bytes']):.3f} / {mb(row['coo_size_bytes']):.3f}")
        compute = (f"{row['cpu_median_ms']:.3f} / {row['cpu_p95_ms']:.3f}; "
                   f"{macm(row['dense_macs']):.2f} / {macm(row['ideal_nonzero_macs']):.2f}")
        mac_relative = 100.0 * float(row["dense_macs"]) / float(b.dense_macs)
        lines.append(f"| {name} | {label} | {model} | {compute} | "
                     f"Not measured (J); dense-MAC proxy {mac_relative:.1f}% of teacher |")
    lines += [
        "",
        "`dense/ideal nonzero MMAC` describes dense arithmetic and an **optimistic zero-skipping "
        "lower bound**. It is not a claim about actual PyTorch CPU operation counts.",
        "",
        "## 6. Conclusion",
        "",
        f"The teacher reached **{b.accuracy_pct:.2f}%** test accuracy with "
        f"**{int(b.total_parameters):,}** parameters. The distilled student uses "
        f"**{100 * kd.total_parameters / b.total_parameters:.2f}%** of its parameters "
        f"and shows a CPU latency ratio (teacher/student) of **{speed:.2f}x** on the "
        "measured setup. "
        f"With exactly the same student architecture and training budget, distillation "
        f"changed accuracy by **{accuracy_delta:+.2f} percentage points** compared "
        "with training without a teacher. Structured filter removal can shrink "
        "actual dense layers, whereas global/layer-wise zero masks alone do not. "
        "COO may or may not reduce disk size depending on how many indices must be "
        "stored. No direct inference energy measurement is available; making a "
        "Joules/inference claim would require hardware power telemetry.",
        "",
        "### Measurement and interpretation cautions",
        "",
        "- Model size means serialized **weight state**, not the Python runtime, source code or deployment package.",
        "- Test-set accuracy is the fraction of correct Fashion-MNIST predictions; the test set is never used for optimization.",
        "- Repeat CPU benchmarks on the **target hardware** before drawing mobile/MCU latency conclusions.",
        "- `metrics.csv` contains every exact experiment row, including target and *measured* weight sparsity.",
        "",
        "## References",
        "",
        "1. Course manual, *Lab 7: Compression: pruning and distillation*, page 9.",
        "2. Vijay Janapa Reddi, *Introduction to Machine Learning Systems*, Ch. 9, Ch. 10 ss. 10.3-10.4 (assigned course reading).",
        "3. Xiao, Rasul and Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, 2017, arXiv:1708.07747.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    """Tests mathematical/mechanical correctness without pretending synthetic data is Fashion-MNIST."""
    seed_everything(10)
    base = teacher_model()
    student = student_model()
    assert num_params(base) == 997434
    assert num_params(student) == 99024
    assert base(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert student(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    for scope in ("global", "layerwise"):
        m = copy.deepcopy(base)
        masks = masks_for_magnitude(m, 0.75, scope)
        apply_masks(m, masks)
        assert abs(weight_sparsity(m) - 0.75) < 0.0001
        # Fixed masks must remain zero after a masked gradient update.
        ds = TensorDataset(torch.rand(8, 1, 28, 28), torch.randint(0, 10, (8,)))
        train_epochs(m, ds, 1, torch.device("cpu"), 1e-3, 8, 123, masks=masks)
        for name, p in m.named_parameters():
            if name in masks:
                assert torch.count_nonzero(p[masks[name] == 0]).item() == 0
    for p in STRUCTURED_FRACTIONS:
        pruned = structured_prune(base, p)
        assert num_params(pruned) < num_params(base)
        assert pruned(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
        assert pruned.conv2.in_channels == pruned.conv1.out_channels
        assert pruned.conv3.in_channels == pruned.conv2.out_channels
    d, s = storage_sizes(base)
    assert d > 0 and s > 0
    assert dense_macs(base) == ideal_nonzero_macs(base)
    log_s = F.log_softmax(torch.randn(2, 10), dim=1)
    assert torch.allclose((log_s.exp() * (log_s - log_s)).sum(1).mean(),
                          torch.tensor(0.0))
    print("SELF-TEST PASS: shapes, parameter counts, both exact sparsity modes, "
          "fixed masks, physical channel removal, MACs, storage, and KL identity.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", default="results", help="Directory for report/plots/metrics")
    p.add_argument("--data-dir", default="data", help="Torchvision Fashion-MNIST cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--teacher-epochs", type=int, default=5)
    p.add_argument("--prune-epochs", type=int, default=3)
    p.add_argument("--student-epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--prune-lr", type=float, default=0.0005)
    p.add_argument("--train-limit", type=int, default=0,
                   help="0=ALL 60000; nonzero marks report as subset")
    p.add_argument("--test-limit", type=int, default=0,
                   help="0=ALL 10000; nonzero marks report as subset")
    p.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--benchmark-repeats", type=int, default=100)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--self-test", action="store_true",
                   help="Validate code on synthetic tensors; DO NOT create submission results")
    args = p.parse_args()
    if args.prune_epochs != 3 and not args.self_test:
        print("WARNING: Lab manual requires EXACTLY 3 pruning fine-tuning epochs.")
    if min(args.batch_size, args.cpu_threads, args.benchmark_repeats) < 1:
        p.error("batch-size, cpu-threads and benchmark-repeats must be >= 1")
    if min(args.teacher_epochs, args.prune_epochs, args.student_epochs) < 1:
        p.error("all epoch counts must be >= 1")
    if args.train_limit < 0 or args.test_limit < 0:
        p.error("train-limit and test-limit must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and
                              torch.backends.mps.is_available() else "cpu")
    train_ds, test_ds = build_datasets(args.data_dir, args.train_limit,
                                       args.test_limit, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Fashion-MNIST samples: train={len(train_ds)}, test={len(test_ds)}; "
          f"training device={device}; CPU benchmark threads={args.cpu_threads}", flush=True)
    results: list[Result] = []
    print("\n[1/4] Train original oversized baseline", flush=True)
    base = teacher_model()
    train_epochs(base, train_ds, args.teacher_epochs, device, args.lr,
                 args.batch_size, args.seed)
    base.eval()
    results.append(measure_model("Baseline teacher", "teacher", 0, base, test_ds,
                                 device, args.batch_size, args.benchmark_repeats,
                                 args.cpu_threads))
    # Reuse ORIGINAL trained checkpoint, not a previously modified pruned model.
    original = copy.deepcopy(base)
    print("\n[2/4] Global and layer-wise magnitude pruning", flush=True)
    for scope in ("global", "layerwise"):
        for target in TARGET_SPARSITIES:
            print(f"{scope} target={target:.0%}", flush=True)
            m = copy.deepcopy(original)
            masks = masks_for_magnitude(m, target, scope)
            apply_masks(m, masks)
            train_epochs(m, train_ds, args.prune_epochs, device, args.prune_lr,
                         args.batch_size, args.seed + 1, masks=masks)
            results.append(measure_model(f"{scope} {target:.0%}", scope,
                                         target, m, test_ds, device, args.batch_size,
                                         args.benchmark_repeats, args.cpu_threads))
            del m, masks
    print("\n[3/4] Physical structured filter pruning", flush=True)
    for fraction in STRUCTURED_FRACTIONS:
        print(f"structured filters removed={fraction:.0%}", flush=True)
        m = structured_prune(original, fraction)
        train_epochs(m, train_ds, args.prune_epochs, device, args.prune_lr,
                     args.batch_size, args.seed + 1)
        results.append(measure_model(f"Structured filters {fraction:.0%}",
                                     "structured", fraction, m, test_ds, device,
                                     args.batch_size, args.benchmark_repeats,
                                     args.cpu_threads))
        del m
    print("\n[4/4] Train plain student and distillation student", flush=True)
    for grp, name, use_teacher in (("student_plain", "Plain student", False),
                                    ("student_kd", "Distilled student", True)):
        print(name, flush=True)
        seed_everything(args.seed + 100)
        m = student_model()
        train_epochs(m, train_ds, args.student_epochs, device, args.lr,
                     args.batch_size, args.seed + 101,
                     teacher=original if use_teacher else None)
        results.append(measure_model(name, grp, 0, m, test_ds, device,
                                     args.batch_size, args.benchmark_repeats,
                                     args.cpu_threads))
        del m
    frame = pd.DataFrame([asdict(r) for r in results])
    frame.to_csv(output / "metrics.csv", index=False, float_format="%.8f")
    save_plots(frame, output)
    create_report(frame, output, args, len(train_ds), len(test_ds), device)
    print(f"\nSUCCESS. Real measurements, 4 plots and English report in: {output.resolve()}")


if __name__ == "__main__":
    main()
