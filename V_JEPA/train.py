"""
train.py — entry point called by torchrun.
"""

import argparse

import yaml

import sys
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_head import train, setup_ddp, cleanup_ddp
from utils.experiments import build_experiments, Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-JEPA segmentation head training")
    parser.add_argument("--config", default="config.yaml", required=True, help="Path to base YAML config")
    parser.add_argument("--experiments", default=None, help="Experiments YAML. If omitted, runs base config once.")
    parser.add_argument("--group", default=None, help="Run only experiments from this group (requires --experiments).")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved experiments without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.experiments is None:
        experiments = [Experiment(name="base", group="base", config=base_cfg)]
    else:
        experiments = build_experiments(args.config, args.experiments)
        if args.group is not None:
            experiments = [e for e in experiments if e.group == args.group]
            if not experiments:
                raise ValueError(
                    f"No experiments found for group '{args.group}'. "
                    f"Check your experiments.yaml."
                )

    if args.dry_run:
        for i, exp in enumerate(experiments):
            print(f"[{i+1}/{len(experiments)}] {exp.group} / {exp.name}")
        return

    rank, world_size, device = setup_ddp()

    try:
        for i, exp in enumerate(experiments):
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Experiment [{i+1}/{len(experiments)}]: {exp.name}")
                print(f"{'='*60}")

            dist.barrier()

            train(exp.config, rank, world_size, device)

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()