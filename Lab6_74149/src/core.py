"""Shared, reproducible training and evaluation primitives."""
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from .models import build_model, model_count


def load_config(path='config.yaml'):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or 'project' not in cfg or 'training' not in cfg:
        raise ValueError('Invalid YAML: expected project and training mappings.')
    return cfg


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def selected_device(request='auto'):
    if request != 'auto':
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def datasets_for(cfg, smoke=False):
    """Official Lab 3 split; synthetic tensors are used ONLY by explicitly requested smoke tests."""
    seed = int(cfg['project']['seed'])
    if smoke:
        g = torch.Generator().manual_seed(seed)
        trainx = torch.randn((96, 1, 28, 28), generator=g)
        trainy = torch.randint(0, 10, (96,), generator=g)
        valx = torch.randn((32, 1, 28, 28), generator=g)
        valy = torch.randint(0, 10, (32,), generator=g)
        return (TensorDataset(trainx, trainy), TensorDataset(valx, valy),
                TensorDataset(valx.clone(), valy.clone()))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    root = str(cfg['data']['root'])
    raw_train = datasets.FashionMNIST(root=root, train=True,
                                       download=bool(cfg['data']['download']), transform=transform)
    official_test = datasets.FashionMNIST(root=root, train=False,
                                         download=bool(cfg['data']['download']), transform=transform)
    train_size = int(cfg['data']['train_size'])
    val_size = int(cfg['data']['val_size'])
    if train_size + val_size != len(raw_train) or val_size <= 0:
        raise ValueError('Lab 3 requires a 55,000/5,000 split of the official 60,000 training examples.')
    # Dedicated generator for fixed split, not reused for shuffling.
    order = torch.randperm(len(raw_train), generator=torch.Generator().manual_seed(seed)).tolist()
    train = Subset(raw_train, order[:train_size])
    val = Subset(raw_train, order[train_size:])
    for key, obj in [('max_train_examples', train), ('max_val_examples', val),
                     ('max_test_examples', official_test)]:
        n = cfg['training'].get(key)
        if n is not None:
            n = int(n)
            if not 0 < n <= len(obj):
                raise ValueError(f'{key} must be between 1 and {len(obj)}')
            if key == 'max_train_examples': train = Subset(train, range(n))
            if key == 'max_val_examples': val = Subset(val, range(n))
            if key == 'max_test_examples': official_test = Subset(official_test, range(n))
    return train, val, official_test


def dataloader(data, batch, workers, shuffle=False, generator=None, pin_memory=False):
    opts = dict(dataset=data, batch_size=int(batch), num_workers=int(workers),
                shuffle=shuffle, pin_memory=bool(pin_memory), drop_last=False,
                generator=generator)
    if int(workers) > 0:
        opts['persistent_workers'] = False
    return DataLoader(**opts)


def make_loaders(cfg, datasets_, batch_size=None, workers=None, generator=None):
    batch = int(batch_size or cfg['training']['batch_size'])
    workers = cfg['data']['num_workers'] if workers is None else workers
    generator = generator or torch.Generator().manual_seed(int(cfg['project']['seed']) + 1000)
    pin = bool(cfg['data']['pin_memory'])
    train, val, test = datasets_
    return (dataloader(train, batch, workers, True, generator, pin),
            dataloader(val, batch, workers, False, pin_memory=pin),
            dataloader(test, batch, workers, False, pin_memory=pin), generator)


def make_optimizer(model, cfg):
    t = cfg['training']
    name = str(t['optimizer']).lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=float(t['lr']),
                                weight_decay=float(t.get('weight_decay', 0)))
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=float(t['lr']),
                               momentum=float(t.get('momentum', .9)),
                               weight_decay=float(t.get('weight_decay', 0)))
    raise ValueError(f'Unknown optimizer: {name}')


def make_scheduler(optimizer, cfg, epochs):
    t = cfg['training']
    kind = str(t.get('scheduler', 'none')).lower()
    if kind == 'none':
        return None
    if kind == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(t['step_size']), gamma=float(t['gamma']))
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    if kind == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=float(t.get('gamma', 0.5)),
            patience=int(t.get('plateau_patience', 2)))
    raise ValueError(f'Unknown scheduler: {kind}')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def memory_reset(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def memory_peak(device, fallback=None):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024), 'cuda_allocated_peak_mb'
    return fallback, 'sampled_cpu_rss_mb'


def train_epoch(model, loader, optimizer, device, accum_steps=1, instrument=False):
    """One epoch; loss uses sample-weighted groups for exact effective batch semantics."""
    model.train()
    n_seen, n_correct, total_loss = 0, 0, 0.0
    stage = {'data_loading': 0., 'forward': 0., 'backward': 0., 'optimizer': 0.}
    peak_rss = rss_mb()
    sync(device)
    started = time.perf_counter()
    iterator = iter(loader)
    finished = False
    while not finished:
        microbatches = []
        for _ in range(accum_steps):
            try:
                t0 = time.perf_counter()
                with torch.profiler.record_function('STAGE_data_loading') if instrument else nullcontext():
                    x, y = next(iterator)
                stage['data_loading'] += time.perf_counter() - t0
                microbatches.append((x, y))
            except StopIteration:
                finished = True
                break
        if not microbatches:
            break
        total_window_examples = sum(y.numel() for _, y in microbatches)
        optimizer.zero_grad(set_to_none=True)
        for x, y in microbatches:
            x, y = x.to(device), y.to(device)
            sync(device)
            t0 = time.perf_counter()
            with torch.profiler.record_function('STAGE_forward') if instrument else nullcontext():
                logits = model(x)
                loss_sum = F.cross_entropy(logits, y, reduction='sum')
            sync(device)
            stage['forward'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.profiler.record_function('STAGE_backward') if instrument else nullcontext():
                (loss_sum / total_window_examples).backward()
            sync(device)
            stage['backward'] += time.perf_counter() - t0
            with torch.no_grad():
                n_seen += y.numel()
                total_loss += float(loss_sum.detach().item())
                n_correct += int((logits.argmax(1) == y).sum().item())
        t0 = time.perf_counter()
        with torch.profiler.record_function('STAGE_optimizer') if instrument else nullcontext():
            optimizer.step()
        sync(device)
        stage['optimizer'] += time.perf_counter() - t0
        sample = rss_mb()
        if sample is not None:
            peak_rss = sample if peak_rss is None else max(peak_rss, sample)
    sync(device)
    elapsed = time.perf_counter() - started
    if n_seen == 0:
        raise ValueError('Empty train dataloader')
    return dict(train_loss=total_loss / n_seen, train_acc=n_correct / n_seen,
                samples=n_seen, seconds=elapsed, throughput=n_seen / elapsed,
                sampled_cpu_rss_mb=peak_rss, stages_s=stage)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n, n_correct, total_loss = 0, 0, 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction='sum').item())
        n_correct += int((logits.argmax(1) == y).sum().item())
        n += y.numel()
    return {'loss': total_loss / n, 'accuracy': n_correct / n}


def capture_rng(generator):
    states = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch=torch.get_rng_state(), dataloader=generator.get_state())
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states, generator):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    generator.set_state(states['dataloader'])
    if 'cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])


def save_checkpoint(path, model, opt, sched, epoch, history, generator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # contains Python/NumPy RNG states; load only your own trusted checkpoints.
    torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                    scheduler=sched.state_dict() if sched is not None else None,
                    history=history, rng=capture_rng(generator)), path)


def load_checkpoint(path, model, opt, sched, generator, device):
    # weights_only=False is required for saved RNG states: only load locally trusted files.
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    if sched is not None and state['scheduler'] is not None:
        sched.load_state_dict(state['scheduler'])
    restore_rng(state['rng'], generator)
    return int(state['epoch']), list(state['history'])


class ExperimentLogger:
    """Write MLflow params/metrics/artifacts; smoke mode may skip missing MLflow."""
    def __init__(self, cfg, run_name, enabled=True):
        self.enabled = enabled
        self.mlflow = None
        self.run = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError('MLflow is required for graded runs: pip install -r requirements.txt') from exc
        self.mlflow = mlflow
        mlflow.set_tracking_uri(cfg['project']['tracking_uri'])
        mlflow.set_experiment(cfg['project']['experiment_name'])
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(dict(seed=cfg['project']['seed'], architecture=cfg['model']['architecture'],
                               batch_size=cfg['training']['batch_size'],
                               optimizer=cfg['training']['optimizer'],
                               scheduler=cfg['training']['scheduler'],
                               learning_rate=cfg['training']['lr'],
                               num_workers=cfg['data']['num_workers'],
                               grad_accum_steps=cfg['training']['grad_accum_steps'],
                               device=cfg['training']['device']))

    def param(self, key, value):
        if self.mlflow is not None:
            self.mlflow.log_param(key, value)

    def metrics(self, metrics, step):
        if self.mlflow is not None:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                     if isinstance(v, (int, float)) and not isinstance(v, bool)}, step=step)

    def artifact(self, path):
        if self.mlflow is not None and Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def end(self):
        if self.mlflow is not None and self.run is not None:
            self.mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.mlflow is not None:
            self.mlflow.end_run(status='FAILED' if exc_type is not None else 'FINISHED')
        return False


def write_table(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def fit(cfg, data, tag, epochs=None, checkpoint_path=None, resume=False,
        stop_after=None, mlflow_enabled=True, instrument=False):
    """Training session. If resumed, continues from exact epoch boundary with RNG restored."""
    seed_everything(int(cfg['project']['seed']), bool(cfg['training']['deterministic']))
    device = selected_device(cfg['training']['device'])
    model = build_model(cfg).to(device)
    opt = make_optimizer(model, cfg)
    total_epochs = int(epochs or cfg['training']['epochs'])
    scheduler = make_scheduler(opt, cfg, total_epochs)
    train, val, test, generator = make_loaders(cfg, data)
    start_epoch = 0
    history = []
    if resume:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f'Missing resume checkpoint: {checkpoint_path}')
        start_epoch, history = load_checkpoint(checkpoint_path, model, opt, scheduler, generator, device)
    path = Path(cfg['project']['output_dir']) / tag
    path.mkdir(parents=True, exist_ok=True)
    with ExperimentLogger(cfg, tag + ('_resumed' if resume else ''), enabled=mlflow_enabled) as log:
        log.param('epochs', total_epochs)
        log.param('parameters', model_count(model))
        if resume:
            log.param('resumed_from_epoch', start_epoch)
        for ep in range(start_epoch, total_epochs):
            memory_reset(device)
            tr = train_epoch(model, train, opt, device,
                             int(cfg['training']['grad_accum_steps']), instrument=instrument)
            ev = evaluate(model, val, device)
            lr_used = float(opt.param_groups[0]['lr'])
            if scheduler is not None:
                if cfg['training']['scheduler'] == 'plateau':
                    scheduler.step(ev['loss'])
                else:
                    scheduler.step()
            mem, memory_type = memory_peak(device, tr['sampled_cpu_rss_mb'])
            row = dict(epoch=ep + 1, train_loss=tr['train_loss'], train_accuracy=tr['train_acc'],
                       val_loss=ev['loss'], val_accuracy=ev['accuracy'], lr=lr_used,
                       seconds=tr['seconds'], throughput_samples_s=tr['throughput'],
                       peak_memory_mb=mem, memory_measurement=memory_type)
            history.append(row)
            log.metrics({k: v for k, v in row.items() if isinstance(v, (int, float))}, ep + 1)
            # Capture checkpoint at END of epoch: RNG state corresponds to next epoch.
            if checkpoint_path and (ep + 1) % int(cfg['training']['checkpoint_every']) == 0:
                save_checkpoint(checkpoint_path, model, opt, scheduler, ep + 1, history, generator)
            if instrument:
                profile_row = dict(epoch=ep + 1, **tr['stages_s'], total_epoch_seconds=tr['seconds'])
                write_table([profile_row], path / f'profile_epoch_{ep+1}.csv')
            print(f'{tag}: epoch {ep+1}/{total_epochs} '
                  f'train={tr["train_acc"]:.4f} val={ev["accuracy"]:.4f} '
                  f'{tr["throughput"]:.0f} samples/s', flush=True)
            if stop_after is not None and ep + 1 == stop_after:
                break
        path_csv = write_table(history, path / 'history.csv')
        log.artifact(path_csv)
        if history and (stop_after is None or history[-1]['epoch'] == total_epochs):
            # Test set evaluated AFTER final training only, not used for hyperparameter selection.
            test_eval = evaluate(model, test, device)
            summary = dict(test_accuracy=test_eval['accuracy'], test_loss=test_eval['loss'],
                           parameters=model_count(model), completed_epochs=history[-1]['epoch'])
            (path / 'final_test.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            log.metrics({'test_accuracy': test_eval['accuracy'], 'test_loss': test_eval['loss']},
                        history[-1]['epoch'])
            log.artifact(path / 'final_test.json')
        else:
            summary = None
    return model, history, summary
