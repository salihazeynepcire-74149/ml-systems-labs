"""
PyTorch Execution Graph Profiling & Memory Contiguity Mechanics
Student: Saliha Zeynep Cire (Student ID: 74149)
Course: Machine Learning Systems (Lab 05)
"""

import time
import json
import os
import torch
import torch.nn as nn


class MathematicalMoonsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.affine2 = nn.Linear(8, 1)
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.output_sigmoid(self.affine2(self.activation(self.affine1(x))))


def profile_execution_and_strides():
    print("[SALIHA CIRE] Profiling execution modes and memory contiguity...")
    model = MathematicalMoonsNet().eval()
    sample = torch.randn(64, 2, dtype=torch.float32)

    traced = torch.jit.trace(model, sample)
    ir_str = str(traced.inlined_graph)

    # Warm-up phase
    for _ in range(150):
        _ = model(sample)
        _ = traced(sample)

    loops = 1000
    t0 = time.perf_counter()
    for _ in range(loops):
        _ = model(sample)
    t_eager = ((time.perf_counter() - t0) / loops) * 1e6

    t0 = time.perf_counter()
    for _ in range(loops):
        _ = traced(sample)
    t_traced = ((time.perf_counter() - t0) / loops) * 1e6

    print(f"  Eager Latency:  {t_eager:.2f} µs")
    print(f"  Traced Latency: {t_traced:.2f} µs (Speedup: {t_eager / t_traced:.2f}x)")

    # Memory Strides Evaluation
    N = 2048
    tensor_c = torch.randn(N, N, dtype=torch.float32)
    tensor_nc = tensor_c.t()

    c_props = {
        'shape': list(tensor_c.shape),
        'stride': list(tensor_c.stride()),
        'is_contiguous': tensor_c.is_contiguous()
    }
    nc_props = {
        'shape': list(tensor_nc.shape),
        'stride': list(tensor_nc.stride()),
        'is_contiguous': tensor_nc.is_contiguous()
    }

    reps = 200
    for _ in range(20):
        _ = torch.sum(tensor_c, dim=1)
        _ = torch.sum(tensor_nc, dim=1)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(tensor_c, dim=1)
    sum_c = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.sum(tensor_nc, dim=1)
    sum_nc = ((time.perf_counter() - t0) / reps) * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = tensor_nc.contiguous()
    clone_time = ((time.perf_counter() - t0) / reps) * 1000.0

    penalty = sum_nc / sum_c
    print(f"  Contiguous Sum:     {sum_c:.3f} ms")
    print(f"  Non-Contiguous Sum: {sum_nc:.3f} ms")
    print(f"  Penalty:            {penalty:.2f}x")
    print(f"  Memory Copy Cost:   {clone_time:.3f} ms")

    audit = {
        'student': 'Saliha Zeynep Cire (74149)',
        'eager_latency_us': t_eager,
        'traced_latency_us': t_traced,
        'speedup_traced': t_eager / t_traced,
        'contiguous_props': c_props,
        'non_contiguous_props': nc_props,
        'sum_contiguous_ms': sum_c,
        'sum_non_contiguous_ms': sum_nc,
        'stride_penalty_factor': penalty,
        'contiguous_allocation_ms': clone_time,
        'graph_ir_summary': ir_str[:600]
    }

    os.makedirs('results', exist_ok=True)
    with open('results/stride_benchmarks.json', 'w') as f:
        json.dump(audit, f, indent=2)

    return audit


if __name__ == '__main__':
    profile_execution_and_strides()
