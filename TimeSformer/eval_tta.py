"""Test-time augmentation (TTA) evaluation — NO retraining.

Loads an already-trained checkpoint and evaluates the test fold twice:
  1. plain inference
  2. TTA: average the softmax over {identity, h-flip, v-flip, 180° rotation}
TTA only changes inference, so it is free given your training budget. It typically
adds +1-3% mIoU and is standard, fully reportable practice.

Usage:
  python TimeSformer/eval_tta.py                       # uses base.yaml + default ckpt
  python TimeSformer/eval_tta.py --fold 1 --ckpt results/checkpoints/best_model_fold1.pth
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import yaml
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data import get_pastis_dataloaders
from utils.metrics import SITS_Evaluator
from timesformer_sits import SITS_TimeSformer
from run_baseline import apply_training_strategy


def tta_ops():
    """Spatial transforms acting on the last two dims (H, W) with their inverses.
    Identity is its own inverse; flips and 180° rotation are involutions too."""
    return [
        (lambda t: t,                              lambda t: t),
        (lambda t: torch.flip(t, dims=[-1]),       lambda t: torch.flip(t, dims=[-1])),   # h-flip
        (lambda t: torch.flip(t, dims=[-2]),       lambda t: torch.flip(t, dims=[-2])),   # v-flip
        (lambda t: torch.rot90(t, 2, dims=[-2, -1]), lambda t: torch.rot90(t, 2, dims=[-2, -1])),  # 180°
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="TimeSformer/configs/base.yaml")
    ap.add_argument("--ckpt", default=None, help="path to .pth (default: results/checkpoints/best_model_fold{fold}.pth)")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fold = args.fold if args.fold is not None else cfg['experiment']['run_fold']

    ckpt = args.ckpt or f"results/checkpoints/best_model_fold{fold}.pth"
    if not os.path.exists(ckpt):
        found = glob.glob(f"**/best_model_fold{fold}.pth", recursive=True)
        assert found, f"Checkpoint not found at {ckpt} and no fallback located. Pass --ckpt."
        ckpt = found[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_pastis_dataloaders(
        data_path=cfg['data']['dataset_path'],
        batch_size=cfg['training']['batch_size'],
        run_fold=fold, fraction=1.0, cloud_rate=0.0,
    )

    model = SITS_TimeSformer(
        in_channels=cfg['data']['in_channels'],
        num_classes=cfg['data']['num_classes'],
        high_res_decoder=cfg['model'].get('high_res_decoder', False),
    )
    model = apply_training_strategy(model, cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt}")

    ev_plain = SITS_Evaluator(cfg['data']['num_classes'], cfg['data']['ignore_classes'])
    ev_tta = SITS_Evaluator(cfg['data']['num_classes'], cfg['data']['ignore_classes'])
    ops = tta_ops()

    with torch.no_grad():
        for x, y, mask in test_loader:
            x, mask = x.to(device), mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                base = model(x, mask)
                ev_plain.add_batch(base.argmax(1).cpu(), y)

                probs = torch.zeros_like(base, dtype=torch.float32)
                for fwd, inv in ops:
                    out = model(fwd(x), mask)                       # transform input (H,W)
                    probs += inv(torch.softmax(out.float(), dim=1))  # inverse-transform output
            ev_tta.add_batch(probs.argmax(1).cpu(), y)

    miou_plain = ev_plain.compute_metrics()['mIoU']
    miou_tta = ev_tta.compute_metrics()['mIoU']
    print(f"Test mIoU (plain): {miou_plain:.4f}")
    print(f"Test mIoU (TTA  ): {miou_tta:.4f}   (delta {miou_tta - miou_plain:+.4f})")


if __name__ == "__main__":
    main()
