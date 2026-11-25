"""Lab 6 graded experiments: profiler, workers, batch, optimizers, accumulation, resume."""
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from .core import (ExperimentLogger, dataloader, datasets_for, evaluate, fit, load_config,
                   make_loaders, make_optimizer, memory_peak, memory_reset,
                   rss_mb, seed_everything, selected_device, sync,
                   train_epoch, write_table)
from .models import build_model


def plot_lines(df, x, ys, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=140)
    for col in ys:
        ax.plot(df[x], df[col], marker='o', label=col)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.25)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cloned(cfg):
    return copy.deepcopy(cfg)


def profile_epoch(cfg, data, mlflow_enabled=True):
    """Profile one *complete* training epoch using torch.profiler + stage wall timing."""
    cfg = cloned(cfg)
    cfg['training']['scheduler'] = 'none'
    cfg['training']['grad_accum_steps'] = 1
    seed_everything(int(cfg['project']['seed']), cfg['training']['deterministic'])
    device = selected_device(cfg['training']['device'])
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    train, _, _, _ = make_loaders(cfg, data)
    out = Path(cfg['project']['output_dir']) / '02_profile'
    out.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    memory_reset(device)
    with ExperimentLogger(cfg, '02_profile_one_full_epoch', mlflow_enabled) as log:
        # Tensorboard trace can be extremely large for a 55k-example epoch.
        with profile(activities=activities, profile_memory=True,
                     record_shapes=False, with_stack=False) as prof:
            epoch = train_epoch(model, train, optimizer, device, 1, instrument=True)
        durations = epoch['stages_s']
        total_staged = sum(durations.values())
        rows = [dict(stage=stage, wall_seconds=seconds,
                     share_pct=100 * seconds / total_staged if total_staged else 0)
                for stage, seconds in durations.items()]
        csv_file = write_table(rows, out / 'profile_breakdown.csv')
        summary = dict(epoch_seconds=epoch['seconds'], samples=epoch['samples'],
                       throughput_samples_s=epoch['throughput'], device=str(device),
                       stage_seconds_total=total_staged,
                       timing_note='Stage times are synchronized wall time, not exclusive CPU kernel time.')
        (out / 'profile_metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        sort_key = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        (out / 'torch_profiler_top_ops.txt').write_text(
            prof.key_averages().table(sort_by=sort_key, row_limit=35), encoding='utf-8')
        if cfg['experiments'].get('profile_export_trace', False):
            prof.export_chrome_trace(str(out / 'profile_trace.json'))
        fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
        ax.barh([x['stage'] for x in rows], [x['share_pct'] for x in rows])
        ax.invert_yaxis()
        ax.set(xlabel='Share of instrumented stage time (%)',
               title='One full epoch: profile stage breakdown')
        fig.tight_layout()
        fig.savefig(out / 'profile_breakdown.png')
        plt.close(fig)
        log.metrics({f'profile_{r["stage"]}_pct': r['share_pct'] for r in rows}, 1)
        for p in [csv_file, out / 'profile_metadata.json',
                  out / 'torch_profiler_top_ops.txt', out / 'profile_breakdown.png']:
            log.artifact(p)
    return rows


def workers_study(cfg, data, mlflow_enabled=True):
    """Measure isolated DataLoader and end-to-end training throughput with fixed seed/model."""
    rows = []
    base = cloned(cfg)
    train_data, _, _ = data
    out = Path(cfg['project']['output_dir']) / '03_workers'
    out.mkdir(parents=True, exist_ok=True)
    for workers in cfg['experiments']['workers']:
        c = cloned(base)
        c['data']['num_workers'] = int(workers)
        seed_everything(int(c['project']['seed']), c['training']['deterministic'])
        device = selected_device(c['training']['device'])
        g = torch.Generator().manual_seed(int(c['project']['seed']) + 1000)
        only_loader = dataloader(train_data, c['training']['batch_size'], workers,
                                 shuffle=True, generator=g, pin_memory=c['data']['pin_memory'])
        start = time.perf_counter()
        counted = 0
        for x, _ in only_loader:
            counted += len(x)
        elapsed = time.perf_counter() - start
        model = build_model(c).to(device)
        opt = make_optimizer(model, c)
        g2 = torch.Generator().manual_seed(int(c['project']['seed']) + 1000)
        training_loader = dataloader(train_data, c['training']['batch_size'], workers,
                                     shuffle=True, generator=g2, pin_memory=c['data']['pin_memory'])
        with ExperimentLogger(c, f'03_workers_{workers}', mlflow_enabled) as log:
            metrics = train_epoch(model, training_loader, opt, device, 1)
            row = dict(num_workers=int(workers), loader_only_samples_s=counted / elapsed,
                       training_samples_s=metrics['throughput'],
                       training_epoch_seconds=metrics['seconds'],
                       data_loading_share_pct=100 * metrics['stages_s']['data_loading'] /
                       max(sum(metrics['stages_s'].values()), 1e-12))
            rows.append(row)
            log.metrics({k: v for k, v in row.items() if k != 'num_workers'}, 1)
            print('workers:', row, flush=True)
    csv = write_table(rows, out / 'workers_study.csv')
    df = pd.DataFrame(rows).sort_values('num_workers')
    plot_lines(df, 'num_workers', ['loader_only_samples_s', 'training_samples_s'],
               'DataLoader workers vs throughput', 'DataLoader workers', 'Samples/second',
               out / 'workers_throughput.png')
    best = max(rows, key=lambda r: r['training_samples_s'])
    bottleneck = ('data loading is a likely bottleneck (large loading-time share)'
                  if best['data_loading_share_pct'] > 30
                  else 'compute/optimizer appears more important than data loading at the best worker count')
    (out / 'interpretation.txt').write_text(
        f'Fastest end-to-end worker count: {best["num_workers"]}\n'
        f'Training throughput: {best["training_samples_s"]:.1f} samples/s\n'
        f'Data loading share: {best["data_loading_share_pct"]:.1f}%\n'
        f'Bottleneck diagnostic: {bottleneck}. Confirm on the target machine.\n', encoding='utf-8')
    return rows


def batch_study(cfg, data, mlflow_enabled=True):
    out = Path(cfg['project']['output_dir']) / '04_batches'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in cfg['experiments']['batches']:
        c = cloned(cfg)
        c['training'].update(batch_size=int(batch), scheduler='none', grad_accum_steps=1)
        model, hist, summary = fit(c, data, f'04_batches/batch_{batch}',
                                   epochs=int(cfg['experiments']['batch_epochs']),
                                   mlflow_enabled=mlflow_enabled)
        last = hist[-1]
        rows.append(dict(batch_size=int(batch), final_val_accuracy=last['val_accuracy'],
                         final_test_accuracy=summary['test_accuracy'],
                         final_train_accuracy=last['train_accuracy'],
                         avg_train_throughput_samples_s=float(np.mean([r['throughput_samples_s'] for r in hist])),
                         peak_memory_mb=max(r['peak_memory_mb'] for r in hist if r['peak_memory_mb'] is not None),
                         memory_measurement=last['memory_measurement'], epochs=len(hist)))
    csv = write_table(rows, out / 'batch_study.csv')
    df = pd.DataFrame(rows).sort_values('batch_size')
    plot_lines(df, 'batch_size', ['avg_train_throughput_samples_s'],
               'Batch size vs training throughput', 'Batch size', 'Samples/s', out / 'batch_throughput.png')
    plot_lines(df, 'batch_size', ['peak_memory_mb'],
               'Batch size vs measured memory', 'Batch size', 'MiB', out / 'batch_memory.png')
    plot_lines(df, 'batch_size', ['final_val_accuracy', 'final_test_accuracy'],
               'Five-epoch classification accuracy', 'Batch size', 'Accuracy', out / 'batch_accuracy.png')
    return rows


def optimizer_study(cfg, data, mlflow_enabled=True):
    out = Path(cfg['project']['output_dir']) / '05_optimizers'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for optimizer in cfg['experiments']['optimizers']:
        for schedule in cfg['experiments']['schedulers']:
            c = cloned(cfg)
            c['training'].update(optimizer=str(optimizer), scheduler=str(schedule), grad_accum_steps=1)
            name = f'{optimizer}_{schedule}'
            _, hist, summary = fit(c, data, f'05_optimizers/{name}',
                                   epochs=int(cfg['experiments']['optimizer_epochs']),
                                   mlflow_enabled=mlflow_enabled)
            rows.append(dict(optimizer=optimizer, scheduler=schedule, epochs=len(hist),
                             final_val_accuracy=hist[-1]['val_accuracy'],
                             final_val_loss=hist[-1]['val_loss'],
                             final_test_accuracy=summary['test_accuracy'],
                             final_learning_rate=hist[-1]['lr']))
    write_table(rows, out / 'optimizer_scheduler_study.csv')
    fig, ax = plt.subplots(figsize=(8.3, 4.8), dpi=140)
    for row in rows:
        tag = row['optimizer'] + '_' + row['scheduler']
        hist = pd.read_csv(Path(cfg['project']['output_dir']) / '05_optimizers' / tag / 'history.csv')
        ax.plot(hist['epoch'], hist['val_accuracy'], label=tag)
    ax.set(xlabel='Epoch (equal 10-epoch budget)', ylabel='Validation accuracy',
           title='SGD+momentum vs Adam; LR schedule comparison')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'optimizer_scheduler_curves.png')
    plt.close(fig)
    return rows


def accumulation_study(cfg, data, mlflow_enabled=True):
    out = Path(cfg['project']['output_dir']) / '06_accumulation'
    out.mkdir(parents=True, exist_ok=True)
    a = cloned(cfg)
    b = cloned(cfg)
    a['training'].update(batch_size=256, grad_accum_steps=1, scheduler='none')
    b['training'].update(batch_size=64, grad_accum_steps=4, scheduler='none')
    ma, ha, sa = fit(a, data, '06_accumulation/physical256',
                     epochs=int(cfg['experiments']['accumulation_epochs']), mlflow_enabled=mlflow_enabled)
    mb, hb, sb = fit(b, data, '06_accumulation/micro64_accum4',
                     epochs=int(cfg['experiments']['accumulation_epochs']), mlflow_enabled=mlflow_enabled)
    params_a, params_b = dict(ma.named_parameters()), dict(mb.named_parameters())
    max_param_abs_diff = max((params_a[name].detach().cpu() - params_b[name].detach().cpu()).abs().max().item()
                             for name in params_a)
    val_accuracy_delta = abs(ha[-1]['val_accuracy'] - hb[-1]['val_accuracy'])
    val_loss_delta = abs(ha[-1]['val_loss'] - hb[-1]['val_loss'])
    result = dict(physical_batch=256, micro_batch=64, accumulation_steps=4,
                  base_val_accuracy=ha[-1]['val_accuracy'], accum_val_accuracy=hb[-1]['val_accuracy'],
                  val_accuracy_abs_delta=val_accuracy_delta, val_loss_abs_delta=val_loss_delta,
                  max_parameter_abs_difference=max_param_abs_diff,
                  comparison_note='Same example ordering, sample-weighted loss; floating-point rounding may vary.')
    write_table([result], out / 'accumulation_comparison.csv')
    df = pd.DataFrame({'epoch': [r['epoch'] for r in ha],
                       'physical_256': [r['val_loss'] for r in ha],
                       '64_x_4': [r['val_loss'] for r in hb]})
    plot_lines(df, 'epoch', ['physical_256', '64_x_4'],
               'Gradient accumulation vs physical 256', 'Epoch', 'Validation loss',
               out / 'accumulation_curves.png')
    return result


def resume_study(cfg, data, mlflow_enabled=True):
    out = Path(cfg['project']['output_dir']) / '07_resume'
    out.mkdir(parents=True, exist_ok=True)
    c = cloned(cfg)
    c['training'].update(grad_accum_steps=1, scheduler='step', checkpoint_every=1)
    total_epochs = int(cfg['experiments']['resume_epochs'])
    interrupt = int(cfg['experiments']['resume_interrupt_epoch'])
    if not 0 < interrupt < total_epochs:
        raise ValueError('resume_interrupt_epoch must be between 1 and resume_epochs - 1')
    checkpoint = out / 'resume_checkpoint.pt'
    ref_model, ref_hist, ref_summary = fit(c, data, '07_resume/uninterrupted',
                                            epochs=total_epochs, mlflow_enabled=mlflow_enabled)
    _, halted_hist, _ = fit(c, data, '07_resume/interrupted', epochs=total_epochs,
                            checkpoint_path=checkpoint, stop_after=interrupt,
                            mlflow_enabled=mlflow_enabled)
    resumed_model, resumed_hist, resumed_summary = fit(c, data, '07_resume/resumed',
                            epochs=total_epochs, checkpoint_path=checkpoint,
                            resume=True, mlflow_enabled=mlflow_enabled)
    max_weight_diff = max((u.detach().cpu() - v.detach().cpu()).abs().max().item()
                          for u, v in zip(ref_model.parameters(), resumed_model.parameters()))
    max_curve_diff = max(abs(a['val_loss'] - b['val_loss']) for a, b in zip(ref_hist, resumed_hist))
    result = dict(interrupt_epoch=interrupt, resumed_at_epoch=interrupt + 1,
                  uninterrupted_final_val_acc=ref_hist[-1]['val_accuracy'],
                  resumed_final_val_acc=resumed_hist[-1]['val_accuracy'],
                  max_val_loss_curve_difference=max_curve_diff,
                  max_parameter_abs_difference=max_weight_diff,
                  exactly_matching_within_1e_minus_6=(max_curve_diff < 1e-6 and max_weight_diff < 1e-6))
    write_table([result], out / 'resume_verification.csv')
    df = pd.DataFrame(dict(epoch=[x['epoch'] for x in ref_hist],
                           uninterrupted=[x['val_loss'] for x in ref_hist],
                           resumed=[x['val_loss'] for x in resumed_hist]))
    plot_lines(df, 'epoch', ['uninterrupted', 'resumed'],
               'Checkpoint interruption at epoch 3', 'Epoch', 'Validation loss',
               out / 'resume_curves.png')
    return result


def run(cfg, modes, smoke=False):
    """Do not interpret smoke results as graded Fashion-MNIST evidence."""
    modes = ['baseline', 'profile', 'workers', 'batches', 'optimizers', 'accumulation', 'resume'] if 'all' in modes else modes
    if smoke:
        cfg = cloned(cfg)
        cfg['project']['output_dir'] = 'results/SMOKE_SYNTHETIC_NOT_FOR_SUBMISSION'
        cfg['experiments']['workers'] = [0]
        cfg['experiments']['batches'] = [32]
        cfg['experiments']['optimizers'] = ['adam']
        cfg['experiments']['schedulers'] = ['step']
        cfg['experiments']['batch_epochs'] = 1
        cfg['experiments']['optimizer_epochs'] = 1
        cfg['experiments']['accumulation_epochs'] = 1
        cfg['experiments']['resume_epochs'] = 4
        cfg['experiments']['resume_interrupt_epoch'] = 3
        cfg['training']['epochs'] = 1
        cfg['data']['num_workers'] = 0
    data = datasets_for(cfg, smoke=smoke)
    mlflow_enabled = not smoke
    if 'baseline' in modes:
        cp = Path(cfg['project']['output_dir']) / '01_baseline' / 'checkpoint.pt'
        fit(cfg, data, '01_baseline', checkpoint_path=cp,
            epochs=int(cfg['training']['epochs']), mlflow_enabled=mlflow_enabled)
    if 'profile' in modes:
        profile_epoch(cfg, data, mlflow_enabled)
    if 'workers' in modes:
        workers_study(cfg, data, mlflow_enabled)
    if 'batches' in modes:
        batch_study(cfg, data, mlflow_enabled)
    if 'optimizers' in modes:
        optimizer_study(cfg, data, mlflow_enabled)
    if 'accumulation' in modes:
        accumulation_study(cfg, data, mlflow_enabled)
    if 'resume' in modes:
        resume_study(cfg, data, mlflow_enabled)
