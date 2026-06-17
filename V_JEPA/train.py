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
import importlib

import yaml

from train_head import train

# Map config backbone name -> function in backbones.py
_BACKBONE_FN = {
    "vjepa2_1_vit_base_384":    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384":   "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384":   "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384":"vjepa2_1_vit_gigantic_384",
    "vit_large":                "vjepa2_vit_large",
    "vit_huge":                 "vjepa2_vit_huge",
    "vit_giant":                "vjepa2_vit_giant",
    "vit_giant_384":            "vjepa2_vit_giant_384",
    "vit_ac_giant":             "vjepa2_ac_vit_giant",
}


def load_encoder_and_predictor(cfg: dict):
    """
    Build encoder + predictor from backbones.py using the backbone name in cfg.
    Models are loaded on CPU; train_head.train() moves them to the correct GPU.
    """
    backbone_name = cfg["model"]["backbone"]
    fn_name = _BACKBONE_FN.get(backbone_name)
    if fn_name is None:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Valid options: {list(_BACKBONE_FN)}"
        )

    backbones = importlib.import_module("backbones")  # adjust path if needed
    build_fn = getattr(backbones, fn_name)

    print(f"Loading backbone '{backbone_name}' via backbones.{fn_name}() ...")
    encoder, predictor = build_fn(pretrained=True)
    print("Backbone loaded.")
    return encoder, predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-JEPA segmentation head training")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    encoder, predictor = load_encoder_and_predictor(cfg)
    train(cfg, encoder, predictor)


if __name__ == "__main__":
    main()