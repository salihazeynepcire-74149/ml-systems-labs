"""
Analytical & Numerical Gradient Test Suite: Saliha Zeynep Cire (74149)
Expression: ((a * b + c) ** 2).relu().sum()
Target: Validate against PyTorch Autograd and Central Finite Differences (< 1e-6)
"""

import json
import os
import torch
from autograd_tape import Variable


def evaluate_expression(a_vars, b_vars, c_vars):
    terms = []
    for a, b, c in zip(a_vars, b_vars, c_vars):
        v = ((a * b + c) ** 2).relu()
        terms.append(v)
    
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


def central_difference(a_data, b_data, c_data, param_name, idx, h=1e-7):
    def f(a, b, c):
        return sum(max(0.0, (ai * bi + ci) ** 2) for ai, bi, ci in zip(a, b, c))

    ap, am = list(a_data), list(a_data)
    bp, bm = list(b_data), list(b_data)
    cp, cm = list(c_data), list(c_data)

    if param_name == 'a':
        ap[idx] += h
        am[idx] -= h
    elif param_name == 'b':
        bp[idx] += h
        bm[idx] -= h
    else:
        cp[idx] += h
        cm[idx] -= h

    return (f(ap, bp, cp) - f(am, bm, cm)) / (2.0 * h)


def run_gradient_verification():
    print("[SALIHA CIRE] Executing automatic differentiation validation suite...")
    a_init = [1.60, -1.90, 0.85, -1.25]
    b_init = [-2.20, 1.80, 2.40, 1.30]
    c_init = [1.40, -0.90, -1.80, 3.10]

    # Custom Variable Pass
    a_nodes = [Variable(x, tag=f'a_{i}') for i, x in enumerate(a_init)]
    b_nodes = [Variable(x, tag=f'b_{i}') for i, x in enumerate(b_init)]
    c_nodes = [Variable(x, tag=f'c_{i}') for i, x in enumerate(c_init)]

    loss_custom = evaluate_expression(a_nodes, b_nodes, c_nodes)
    loss_custom.backward()

    # PyTorch Reference Pass
    ta = torch.tensor(a_init, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c_init, dtype=torch.float64, requires_grad=True)

    loss_torch = (((ta * tb + tc) ** 2).relu()).sum()
    loss_torch.backward()

    # Numerical Central Differences
    max_torch_err = 0.0
    max_cd_err = 0.0
    records = []

    for name, v_nodes, t_ref in [('a', a_nodes, ta), ('b', b_nodes, tb), ('c', c_nodes, tc)]:
        for i in range(len(a_init)):
            g_cust = v_nodes[i].grad
            g_torch = t_ref.grad[i].item()
            g_cd = central_difference(a_init, b_init, c_init, name, i)

            diff_torch = abs(g_cust - g_torch)
            diff_cd = abs(g_cust - g_cd)

            max_torch_err = max(max_torch_err, diff_torch)
            max_cd_err = max(max_cd_err, diff_cd)

            records.append({
                'identifier': f"{name}[{i}]",
                'custom_gradient': g_cust,
                'torch_gradient': g_torch,
                'central_diff_gradient': g_cd,
                'error_vs_torch': diff_torch,
                'error_vs_cd': diff_cd
            })

    passed = bool(max_torch_err < 1e-6)
    print(f"  Max Absolute Difference vs PyTorch: {max_torch_err:.2e}")
    print(f"  Max Absolute Difference vs Central Diff: {max_cd_err:.2e}")
    print(f"  Verification Status: {'PASSED (< 1e-6)' if passed else 'FAILED'}")

    summary = {
        'student': 'Saliha Zeynep Cire (74149)',
        'target_expression': '((a * b + c) ** 2).relu().sum()',
        'forward_loss_custom': loss_custom.x,
        'forward_loss_torch': loss_torch.item(),
        'max_torch_error': max_torch_err,
        'max_central_diff_error': max_cd_err,
        'threshold': 1e-6,
        'status': 'PASSED' if passed else 'FAILED',
        'table': records
    }

    os.makedirs('results', exist_ok=True)
    with open('results/gradient_verification_log.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    run_gradient_verification()
