# Lab 05: Frameworks and Automatic Differentiation

**Student Researcher:** Saliha Zeynep Cire  
**Student ID:** 74149  
**Institution:** Vistula University – Machine Learning Systems (2025/2026)  
**Curriculum Reference:** Vijay Janapa Reddi, *Machine Learning Systems*, Chapter 7 (AI Frameworks: §§7.3–7.8)  

---

## 1. Executive Summary

This report presents an architectural and empirical investigation into automatic differentiation mechanisms and execution engine performance:
1. **Functional Reverse-Mode Autograd Engine:** Implementation of `Variable`, encapsulating forward dynamic expression DAG construction, topological post-order traversal, and non-overwriting adjoint accumulation.
2. **Empirical Numerical Gradient Checking:** Verification against PyTorch autograd and numerical central finite differences on `((a * b + c) ** 2).relu().sum()`, establishing $< 10^{-6}$ mathematical accuracy.
3. **Moons Binary Classification Benchmark:** 5-trial empirical comparison of scalar automatic differentiation against vectorized PyTorch C++ execution on `make_moons`.
4. **Intermediate Representation & Graph Lowering:** TorchScript IR inspection (`torch.jit.trace`) and eager dispatch overhead characterization across 1,000 warm-up cycles.
5. **Tensor Strides & Cache Hierarchy:** Microarchitectural analysis of memory strides, quantifying the cache-miss penalty and SIMD throughput loss in non-contiguous views.

---

## 2. Reverse-Mode Automatic Differentiation Core (`Variable`)

The custom automatic differentiation core is detailed in [autograd_tape.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/src/autograd_tape.py).

### 2.1 State Representation
Each `Variable` stores:
- `x`: Scalar value computed during forward propagation.
- `grad`: Accumulated gradient adjoint $\partial \mathcal{L} / \partial x$, initialized to 0.0.
- `_deps`: Antecedent dependency set in the directed acyclic graph.
- `_vjp`: Vector-Jacobian Product closure implementing the local backward chain rule.
- `op_code`: Operation symbol.

### 2.2 Mathematical Primitive Operations
- **Addition ($z = x + y$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += \frac{\partial \mathcal{L}}{\partial z}$
- **Multiplication ($z = x \cdot y$):** $\frac{\partial \mathcal{L}}{\partial x} += y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += x \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Power ($z = x^p$):** $\frac{\partial \mathcal{L}}{\partial x} += (p \cdot x^{p-1}) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **ReLU ($z = \max(0, x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \mathbf{1}_{\{x > 0\}} \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):** $\frac{\partial \mathcal{L}}{\partial x} += \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Logarithm ($z = \ln(x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$

### 2.3 Topological Traversal & In-Place Adjoint Accumulation
Depth-first post-order topological sorting guarantees that before evaluating the VJP of any node, all downstream dependency paths have finished propagating. Utilizing in-place addition `+=` preserves the multivariate chain rule across nodes with fan-out $>1$.

---

## 3. Mathematical Verification Against PyTorch & Central Differences

Target evaluation expression:
$$\mathcal{L} = \sum_{i=1}^4 \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

- **Test Vectors:**
  - $a = [1.60, -1.90, 0.85, -1.25]$
  - $b = [-2.20, 1.80, 2.40, 1.30]$
  - $c = [1.40, -0.90, -1.80, 3.10]$
- **Forward Output:** Custom = `25.753425`, PyTorch = `25.753425` ($\Delta = 0.00$)
- **Max Absolute Error vs PyTorch:** **$0.00 \times 10^0$** (Strictly $< 10^{-6}$, PASS)
- **Max Absolute Error vs Central Differences:** $2.60 \times 10^{-8}$ (PASS)

---

## 4. `make_moons` Training Benchmark (5-Trial Telemetry)

- **Architecture:** $2 \to 8 \to 1$ (ReLU hidden, Sigmoid output, BCE loss).
- **Dataset:** `make_moons(n_samples=100, noise=0.1, random_state=42)`.
- **Optimization:** SGD ($\eta = 0.25$, 35 epochs).

| Trial | Custom Engine (ms) | PyTorch (ms) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| 1 | 2276.50 | 16.33 | 139.4x |
| 2 | 2607.73 | 13.66 | 190.9x |
| 3 | 2650.63 | 14.44 | 183.6x |
| 4 | 2685.81 | 13.00 | 206.6x |
| 5 | 2546.14 | 14.64 | 173.9x |
| **Median** | **2607.73 ms** | **14.44 ms** | **180.6x** |

### Systems Analysis
PyTorch's $180.6\times$ advantage is driven by:
1. Heap object proliferation in Python: Allocating individual `Variable` instances for every scalar arithmetic operation imposes large heap overhead.
2. Bytecode interpretation: Python evaluates opcodes individually, whereas PyTorch calls pre-compiled C++ kernels.
3. Memory locality and SIMD: PyTorch executes dense BLAS matrix multiplications across all 100 samples simultaneously.

---

## 5. TorchScript IR Inspection & Execution Modes

We analyzed the computational IR of `MathematicalMoonsNet` using `torch.jit.trace`:
- **Eager Dispatch Latency:** $45.83\ \mu\text{s}$
- **Traced Graph Latency:** $38.60\ \mu\text{s}$
- **Speedup Factor:** **$1.19\times$**

TorchScript tracing removes Python layer dispatch and attribute lookups, producing a static unrolled graph.

---

## 6. Tensor Memory Layout & Strides Penalty

Using a $2048 \times 2048$ single-precision tensor (16 MB buffer):
- **Contiguous Array:** `stride = [2048, 1]`, `is_contiguous() = True`.
- **Transposed Non-Contiguous Array:** `stride = [1, 2048]`, `is_contiguous() = False`.

### Reduction Benchmark (Row-Wise Sum)
- **Contiguous Row-Sum Latency:** $0.187\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $0.359\text{ ms}$
- **Stride Penalty:** **$1.92\times$ slowdown**
- **Buffer Copy Cost (`.contiguous()`):** $8.585\text{ ms}$

### Hardware Mechanics
- **L1/L2 Cache Locality:** Contiguous memory reads read sequential 64-byte cache lines. Non-contiguous strides skip across memory addresses, causing cache-line thrashing and misses.
- **SIMD Instructions:** Contiguous layouts enable full 256/512-bit register utilization (`vmovups`). Non-contiguous layouts force slow gather operations.
- **Amortization Trade-off:** Calling `.contiguous()` takes $8.585\text{ ms}$, which is $50\times$ more expensive than a single reduction's slowdown. Copying is only amortized when the tensor is reused across many subsequent operations.

---

## 7. Technical Note: When Graph Compilation Delivers Value

According to Reddi Chapter 7, graph compilers provide substantial benefits when:
1. **Fusing Pointwise Kernels:** Fusing `Linear + ReLU` keeps intermediate tensors in CPU registers and L1/L2 cache, eliminating redundant DRAM reads and writes.
2. **Eliminating Python Dispatch Overhead:** For low-latency inference or small batch sizes, dispatch overhead dominates silicon execution time. Compilers lower the computation graph to native C++/machine code.
3. **Static Memory Planning:** Compilers analyze variable lifetimes and allocate memory statically, avoiding dynamic `malloc`/`free` calls during the training loop.

---

## 8. Artifact Directory

- [autograd_tape.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/src/autograd_tape.py): Custom scalar autograd engine.
- [gradient_test.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/src/gradient_test.py): Mathematical gradient verification suite.
- [train_classifier.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/src/train_classifier.py): 5-trial benchmark on `make_moons`.
- [memory_stride_benchmarks.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/src/memory_stride_benchmarks.py): TorchScript tracing and memory striding analysis.
- `results/`: Contains [gradient_verification_log.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/results/gradient_verification_log.json), [moons_telemetry_5runs.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/results/moons_telemetry_5runs.json), and [stride_benchmarks.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_74149_saliha/lab05/results/stride_benchmarks.json).
