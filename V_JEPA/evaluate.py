"""
evaluate.py
Loads a trained V-JEPA encoder + segmentation head checkpoint 
and evaluates it on the PASTIS test fold.
"""

import time
from pathlib import Path

import torch
from torch.amp import autocast
import torch.distributed as dist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_head import (
    load_encoder, 
    TemporalMeanPool, 
    SegHead, 
    _encode_sits, 
    _apply_lora, 
    is_main
)
from utils.metrics import SITS_Evaluator
from utils.load_data import get_pastis_dataloaders

from utils.logging import get_logger
log = get_logger()


def evaluate(cfg: dict, rank: int, world_size: int, device: torch.device, ckpt_path: Path) -> dict:
    main = is_main(rank)
    
    # Free up memory from training
    torch.cuda.empty_cache()

    if main:
        log.info(f"\n{'='*60}\n"
                 f"Starting Evaluation on Test Set\n"
                 f"Loading checkpoint: {ckpt_path}\n"
                 f"{'='*60}")

    # 1. Load Data (Force full evaluation with fraction=1.0 and cloud_rate=0.0)
    _, _, test_loader = get_pastis_dataloaders(
        data_path=cfg["data"]["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        run_fold=cfg["run_fold"],
        fraction=1.0, 
        cloud_rate=0.0, 
        to_rgb=cfg["data"]["to_rgb"],
        rank=rank,
        world_size=world_size,
    )

    # 2. Setup Model Architecture
    encoder = load_encoder(cfg, rank, device)
    lora_cfg = cfg["training"].get("lora")
    lora_active = lora_cfg is not None
    img_size = cfg["training"].get("img_size", 384)

    if lora_active:
        encoder = _apply_lora(encoder, lora_cfg)

    # Infer patch grid dynamically
    with torch.no_grad():
        sample_x, _, _ = test_loader.dataset[0]
        max_seq_len, in_channels, _, _ = sample_x.shape
        dummy_input = torch.zeros(1, in_channels, max_seq_len, img_size, img_size, device=device)
        
        encoder.eval()
        dummy_out = encoder(dummy_input)
        
        embed_dim = dummy_out.shape[-1]
        num_tokens = dummy_out.shape[1]
        patch_size = getattr(encoder, "patch_size", 16)
        h_p = w_p = img_size // patch_size
        t_p = num_tokens // (h_p * w_p)
        patch_grid_3d = (t_p, h_p, w_p)

    num_classes = int(cfg.get("training", {}).get("num_classes", 20))
    head = SegHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["training"].get("hidden_dim", 512),
        patch_grid=(h_p, w_p),
    ).to(device)

    pool = TemporalMeanPool(patch_grid_3d)

    # 3. Load Checkpoint Weights
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    if lora_active:
        encoder.load_state_dict(ckpt["encoder"])

    head.eval()
    encoder.eval()

    # 4. Evaluation Loop
    noloss_classes = list(set(cfg["data"]["noloss_classes"]))
    evaluator = SITS_Evaluator(
        num_classes=num_classes,
        ignore_indices=noloss_classes,
        device=device,
    )
    evaluator.reset()

    t0 = time.perf_counter()
    
    with torch.no_grad(), autocast("cuda"):
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            tokens = _encode_sits(encoder, x, img_size)
            logits = head(pool(tokens, mask))
            preds = logits.argmax(1)
            evaluator.add_batch(preds, y)

    # 5. DDP Sync: Aggregate confusion matrices from all GPUs
    if dist.is_initialized():
        dist.all_reduce(evaluator.conf_matrix, op=dist.ReduceOp.SUM)

    metrics = None
    if main:
        results_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
        suffix = "lora" if lora_active else "frozen"
        modality = "rgb" if cfg["data"].get("to_rgb", False) else "multi"
        frac = cfg["ablation"].get("dataset_fraction", 1.0)
        cloud = cfg["ablation"].get("cloud_drop_rate", 0.0)
        fold = cfg['run_fold']
        
        unique_id = f"{suffix}_{modality}_frac{frac}_cloud{cloud}_fold{fold}"
        results_file = results_dir / f"test_metrics_{unique_id}.json"
        
        # Save JSON to disk and print to terminal
        metrics = evaluator.save_results(results_file)
        log.info(f"Evaluation elapsed time: {time.perf_counter() - t0:.1f}s")
            
    # Keep ranks in sync before exiting
    dist.barrier()
    return metrics