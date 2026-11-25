"""CLI for the eight Lab 6 requirements (the proposal is in docs/)."""
import argparse
from pathlib import Path

from src.core import load_config
from src.experiments import run


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Machine Learning Lab 6 | Fashion-MNIST')
    ap.add_argument('--config', default='config.yaml', help='YAML file')
    ap.add_argument('--only', nargs='+', choices=['all', 'baseline', 'profile', 'workers',
                                                  'batches', 'optimizers', 'accumulation', 'resume'],
                    default=['all'])
    ap.add_argument('--smoke', action='store_true', help='synthetic data, NOT graded evidence')
    args = ap.parse_args()
    # Run relative to project root even when CLI called from another working directory.
    import os
    os.chdir(Path(__file__).resolve().parent)
    config = load_config(args.config)
    run(config, args.only, smoke=args.smoke)
