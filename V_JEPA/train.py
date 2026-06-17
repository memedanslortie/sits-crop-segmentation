"""
train.py — entry point called by torchrun.

torchrun spawns one process per GPU and sets:
  - LOCAL_RANK  → which GPU this process should use
  - RANK        → global process index
  - WORLD_SIZE  → total number of processes

Everything else (process-group init, DDP wrapping, DistributedSampler) is
handled inside train_head.train().
"""

import argparse

import yaml

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_head import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-JEPA segmentation head training")
    parser.add_argument("--config", default="config.yaml", help="Path to base YAML config")
    # parser.add_argument("--experiments", default="experiments.yaml", help="Experiments YAML (optional)")
    # parser.add_argument("--group", default=None, help="Run only experiments from this group (default: all groups)")
    # parser.add_argument("--single", action="store_true", help="Ignore experiments.yaml and run the base config once")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()