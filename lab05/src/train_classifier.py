"""
End-to-End Classifier Optimization on make_moons: Custom Variable vs. PyTorch
Student: Saliha Zeynep Cire (Student ID: 74149)
Network Topology: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE loss)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from autograd_tape import Variable


class LinearUnit:
    def __init__(self, in_features: int, is_active: bool = True):
        # Glorot normal initialization
        std = (2.0 / in_features) ** 0.5
        self.weights = [Variable(random.gauss(0.0, std)) for _ in range(in_features)]
        self.bias = Variable(0.0)
        self.is_active = is_active

    def __call__(self, x_in):
        acc = self.bias
        for w, x in zip(self.weights, x_in):
            acc = acc + (w * x)
        return acc.relu() if self.is_active else acc

    def get_variables(self):
        return self.weights + [self.bias]


class DenseProjection:
    def __init__(self, in_dim: int, out_dim: int, is_active: bool = True):
        self.units = [LinearUnit(in_dim, is_active=is_active) for _ in range(out_dim)]

    def __call__(self, inputs):
        outputs = [u(inputs) for u in self.units]
        return outputs[0] if len(outputs) == 1 else outputs

    def get_variables(self):
        v = []
        for u in self.units:
            v.extend(u.get_variables())
        return v


class MoonsClassifier:
    def __init__(self):
        self.hidden = DenseProjection(2, 8, is_active=True)
        self.output = DenseProjection(8, 1, is_active=False)

    def forward(self, x):
        h = self.hidden(x)
        raw = self.output(h)
        return raw.sigmoid()

    def get_all_variables(self):
        return self.hidden.get_variables() + self.output.get_variables()

    def zero_gradients(self):
        for v in self.get_all_variables():
            v.grad = 0.0


class PyTorchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)


def binary_cross_entropy(y_true, y_pred):
    n = len(y_true)
    losses = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            term = -yp.log()
        else:
            term = -(Variable(1.0) - yp).log()
        losses.append(term)
    
    total = losses[0]
    for li in losses[1:]:
        total = total + li
    return total * (1.0 / n)


def train_custom_model(X, y, epochs=35, lr=0.25):
    random.seed(74149)
    model = MoonsClassifier()
    X_vars = [[Variable(val) for val in row] for row in X]
    
    t0 = time.perf_counter()
    loss_curve = []
    
    for epoch in range(epochs):
        preds = [model.forward(x) for x in X_vars]
        loss = binary_cross_entropy(y, preds)
        
        model.zero_gradients()
        loss.backward()
        
        for v in model.get_all_variables():
            v.x -= lr * v.grad
            
        loss_curve.append(loss.x)
        
    duration = time.perf_counter() - t0
    acc = sum((1 if p.x >= 0.5 else 0) == yt for p, yt in zip(preds, y)) / len(y)
    return duration, loss_curve[-1], acc


def train_pytorch_model(X, y, epochs=35, lr=0.25):
    torch.manual_seed(74149)
    model = PyTorchClassifier()
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    loss_curve = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(tx)
        loss = criterion(out, ty)
        loss.backward()
        opt.step()
        loss_curve.append(loss.item())
        
    duration = time.perf_counter() - t0
    with torch.no_grad():
        preds = (model(tx) >= 0.5).float()
        acc = (preds == ty).float().mean().item()
    return duration, loss_curve[-1], acc


def run_benchmark_trials():
    print("[SALIHA CIRE] Executing 5-trial make_moons benchmark...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    custom_times = []
    pytorch_times = []
    custom_losses = []
    pytorch_losses = []
    
    for trial in range(5):
        tc, lc, _ = train_custom_model(X, y, epochs=35, lr=0.25)
        tp, lp, _ = train_pytorch_model(X, y, epochs=35, lr=0.25)
        
        custom_times.append(tc * 1000.0)
        pytorch_times.append(tp * 1000.0)
        custom_losses.append(lc)
        pytorch_losses.append(lp)
        print(f"  Trial {trial+1}/5 | Custom: {custom_times[-1]:.2f} ms | PyTorch: {pytorch_times[-1]:.2f} ms")
        
    med_custom = statistics.median(custom_times)
    med_pytorch = statistics.median(pytorch_times)
    ratio = med_custom / med_pytorch
    
    output = {
        'student': 'Saliha Zeynep Cire (74149)',
        'trials': 5,
        'custom_median_ms': med_custom,
        'pytorch_median_ms': med_pytorch,
        'slowdown_factor': ratio,
        'custom_losses': custom_losses,
        'pytorch_losses': pytorch_losses,
        'custom_times_ms': custom_times,
        'pytorch_times_ms': pytorch_times
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_telemetry_5runs.json', 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\n[SUMMARY] Custom Median: {med_custom:.2f} ms | PyTorch Median: {med_pytorch:.2f} ms | Ratio: {ratio:.1f}x")
    return output


if __name__ == '__main__':
    run_benchmark_trials()
