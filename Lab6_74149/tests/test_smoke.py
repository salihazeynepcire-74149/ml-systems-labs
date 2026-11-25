import copy
import torch
from src.core import (datasets_for, load_config, make_loaders, make_optimizer,
                      seed_everything, train_epoch, evaluate, save_checkpoint,
                      load_checkpoint, selected_device)
from src.models import build_model


def test_cnn_output_shape():
    cfg = load_config()
    model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == 20490
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_tiny_training_and_evaluation():
    cfg = load_config()
    cfg['data']['num_workers'] = 0
    data = datasets_for(cfg, smoke=True)
    train, val, _, _ = make_loaders(cfg, data)
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)
    result = train_epoch(model, train, opt, torch.device('cpu'))
    assert result['samples'] == 96
    accuracy = evaluate(model, val, torch.device('cpu'))['accuracy']
    assert 0 <= accuracy <= 1


def test_accumulation_matches_physical_batch_one_epoch():
    cfg = load_config()
    cfg['data']['num_workers'] = 0
    data = datasets_for(cfg, smoke=True)
    models = []
    for batch, accum in [(32, 1), (8, 4)]:
        c = copy.deepcopy(cfg)
        c['training']['batch_size'] = batch
        seed_everything(42)
        model = build_model(c)
        opt = make_optimizer(model, c)
        train, _, _, _ = make_loaders(c, data)
        train_epoch(model, train, opt, torch.device('cpu'), accum_steps=accum)
        models.append(model)
    worst = max((a - b).abs().max().item()
                for a, b in zip(models[0].parameters(), models[1].parameters()))
    assert worst < 1e-5
