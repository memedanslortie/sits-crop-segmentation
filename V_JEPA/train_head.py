"""
Segmentation head training on a frozen (or LoRA-adapted) V-JEPA 2.1 encoder
for PASTIS crop segmentation using native 3D video encoding.
"""

import os
import time
from pathlib import Path
import importlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.metrics import SITS_Evaluator

def setup_ddp() -> tuple[int, int, torch.device]:
    """Initialise the process group and return (rank, world_size, device)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device

def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0

def load_encoder_and_predictor(cfg: dict, rank: int, device: torch.device):
    """
    Build encoder + predictor from backbones.py using the backbone name in cfg.
    Only rank 0 loads the model, then broadcasts to all ranks.
    """
    backbone_name = cfg["model"]["backbone"]
    if backbone_name is None:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
        )

    # Only rank 0 loads the model
    if rank == 0:
        backbones = importlib.import_module(".backbones", "vjepa2.src.hub")
        build_fn = getattr(backbones, backbone_name)
        print(f"Loading backbone '{backbone_name}' via backbones.{backbone_name}() ...")
        encoder, predictor = build_fn(pretrained=True)
        print("Backbone loaded.")
    else:
        encoder, predictor = None, None

    # Broadcast model state dicts to all ranks
    if rank == 0:
        encoder_state = encoder.state_dict()
        predictor_state = predictor.state_dict()
    else:
        encoder_state = None
        predictor_state = None

    # Synchronize
    dist.barrier()

    # All ranks create the model architecture (but only rank 0 has weights)
    backbones = importlib.import_module(".backbones", "vjepa2.src.hub")
    build_fn = getattr(backbones, backbone_name)
    encoder, predictor = build_fn(pretrained=False)

    # Only rank 0 has the weights; others will load from rank 0
    if rank != 0:
        encoder.load_state_dict(encoder_state)
        predictor.load_state_dict(predictor_state)

    return encoder.to(device), predictor.to(device)

class TemporalMeanPool(nn.Module):
    """Mean-pool 3D video patch tokens over valid (non-padded) temporal patches."""
    def __init__(self, patch_grid_3d: tuple[int, int, int]):
        super().__init__()
        self.t_p, self.h_p, self.w_p = patch_grid_3d

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape
        
        # Reshape flat sequence back to (B, T_p, H_p * W_p, D)
        tokens = tokens.view(B, self.t_p, self.h_p * self.w_p, D)
        
        # Map frame-level mask (B, T) to temporal-patch-level mask (B, T_p)
        B_m, T = mask.shape
        frames_per_patch = T // self.t_p
        T_processed = self.t_p * frames_per_patch
        truncated_mask = mask[:, :T_processed]
        
        # True = invalid. A temporal patch is invalid ONLY if ALL its frames are invalid.
        patch_mask = truncated_mask.view(B, self.t_p, frames_per_patch).all(dim=-1)
        
        valid = (~patch_mask).float()[:, :, None, None]                 # (B, T_p, 1, 1)
        
        # Mean pool over the temporal patch dimension
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1) # (B, H_p * W_p, D)


class SegHead(nn.Module):
    """MLP decoder: spatial patch tokens -> dense segmentation map."""
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        patch_grid: tuple[int, int] = (24, 24),
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(tokens)                                       # (B, N_spatial, C)
        B, N_spatial, C = logits.shape
        H_p, W_p = self.patch_grid
        logits = logits.permute(0, 2, 1).reshape(B, C, H_p, W_p)
        return F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)


def _compute_class_weights(
    train_dataset,
    num_classes: int,
    noloss_classes: set[int],
    cache_path: Path,
    device: torch.device,
    rank: int,
) -> torch.Tensor:
    """Compute / load class weights. Only rank-0 writes to disk; all ranks load."""
    # Rank-0 computes/writes, then we barrier so all ranks can read safely.
    if is_main(rank):
        if cache_path.exists():
            print(f"Loading cached class weights from {cache_path}")
            weights = torch.load(cache_path, map_location="cpu")
        else:
            print("Computing class weights (one-time scan of training labels)...")
            counts = train_dataset.label_counts(num_classes)
            for c in noloss_classes:
                counts[c] = 0.0
            total = counts.sum()
            weights = torch.where(
                counts > 0,
                total / (num_classes * counts),
                torch.zeros_like(counts),
            ).float()
            torch.save(weights, cache_path)

    dist.barrier()  # Non-rank-0 processes wait here

    if not is_main(rank):
        weights = torch.load(cache_path, map_location="cpu")

    return weights.to(device)


def _apply_lora(encoder: nn.Module, lora_cfg: dict) -> nn.Module:
    config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg.get("alpha", lora_cfg["rank"]),
        target_modules=["qkv", "proj"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(encoder, config)
    encoder.print_trainable_parameters()
    return encoder


def _encode_sits(encoder: nn.Module, x: torch.Tensor, img_size: int) -> torch.Tensor:
    """Encode the full time series natively as a 3D video tensor."""
    B, T, C, H, W = x.shape
    
    # Safely interpolate spatially first to avoid smearing clouds/padding over time
    if H != img_size or W != img_size:
        x_flat = x.flatten(0, 1)                                        # (B*T, C, H, W)
        x_flat = F.interpolate(x_flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = x_flat.unflatten(0, (B, T))                                 # (B, T, C, H_new, W_new)
        
    # Transpose to V-JEPA expected format: (Batch, Channels, Time, Height, Width)
    video_tensor = x.permute(0, 2, 1, 3, 4)
    return encoder(video_tensor)                                        # (B, L, D)

def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, stripping DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


def _one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    pool: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    img_size: int,
    is_train: bool,
    lora_active: bool,
    evaluator: SITS_Evaluator | None = None,
) -> dict[str, float]:
    head.train(is_train)
    _unwrap(encoder).train(is_train and lora_active)

    if evaluator is not None:
        evaluator.reset()

    total_loss = 0.0
    n_batches = 0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with autocast("cuda"):
            enc_ctx = torch.enable_grad() if (is_train and lora_active) else torch.no_grad()
            with enc_ctx:
                tokens = _encode_sits(encoder, x, img_size)

            logits = head(pool(tokens, mask))
            loss = criterion(logits, y).mean()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if evaluator is not None:
            preds = logits.argmax(1)
            evaluator.add_batch(preds, y)

    out = {"loss": total_loss / max(n_batches, 1)}
    if evaluator is not None:
        out.update(evaluator.compute_metrics())
    return out


def train(cfg: dict) -> nn.Module:
    from utils.load_data import get_pastis_dataloaders

    # DDP init
    rank, world_size, device = setup_ddp()
    main = is_main(rank)

    # Load encoder and predictor on the correct GPU
    encoder, _ = load_encoder_and_predictor(cfg, rank, device)

    if main:
        print(f"DDP active: {world_size} GPUs")

    # Reproducibility
    torch.manual_seed(cfg["seed"] + rank)  # Different seed per rank for data augmentation

    # Config
    num_classes: int = int(cfg.get("training", {}).get("num_classes", 20))
    noloss_classes: set[int] = set(cfg["data"]["noloss_classes"])
    img_size: int = cfg["training"].get("img_size", 384)
    lora_cfg: dict | None = cfg["training"].get("lora")
    lora_active = lora_cfg is not None

    evaluator = SITS_Evaluator(
        num_classes=num_classes,
        ignore_indices=list(noloss_classes),
        device=device,
    )

    # Data
    # Pass rank and world_size so DistributedSampler is configured correctly.
    train_loader, val_loader, _ = get_pastis_dataloaders(
        data_path=cfg["data"]["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        run_fold=cfg["run_fold"],
        fraction=cfg["ablation"]["dataset_fraction"],
        cloud_rate=cfg["ablation"]["cloud_drop_rate"],
        to_rgb=cfg["data"]["to_rgb"],
        rank=rank,
        world_size=world_size,
    )

    # Encoder
    encoder = encoder.to(device)
    if lora_active:
        encoder = _apply_lora(encoder, lora_cfg)
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    # Wrap encoder in DDP only when it has trainable params (LoRA).
    if lora_active:
        encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=True)

    # Infer 3-D patch grid (rank-0 only, then broadcast)
    with torch.no_grad():
        sample_x, _, _ = train_loader.dataset[0]
        max_seq_len, in_channels, _, _ = sample_x.shape

        dummy_input = torch.zeros(1, in_channels, max_seq_len, img_size, img_size, device=device)
        raw_encoder = _unwrap(encoder)
        raw_encoder.eval()
        dummy_out = raw_encoder(dummy_input)
        if lora_active:
            raw_encoder.train()

        embed_dim = dummy_out.shape[-1]
        num_tokens = dummy_out.shape[1]
        patch_size = getattr(raw_encoder, "patch_size", 16)
        h_p = w_p = img_size // patch_size
        t_p = num_tokens // (h_p * w_p)
        patch_grid_3d = (t_p, h_p, w_p)

    if main:
        print(f"Patch grid 3D: {patch_grid_3d} | embed_dim: {embed_dim}")

    # Head + Pool
    head = SegHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["training"].get("hidden_dim", 512),
        patch_grid=(h_p, w_p),
    ).to(device)
    head = DDP(head, device_ids=[rank])

    pool = TemporalMeanPool(patch_grid_3d)  # Stateless; no DDP needed

    # Loss
    class_weight: torch.Tensor | None = None
    if cfg["training"].get("class_weighting", False):
        ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cache = ckpt_dir / f"class_weights_fold{cfg['run_fold']}.pt"
        class_weight = _compute_class_weights(
            train_loader.dataset, num_classes, noloss_classes, cache, device, rank
        )
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-1)

    # Optimiser
    param_groups = [{"params": _unwrap(head).parameters(), "lr": cfg["training"]["lr"]}]
    if lora_active:
        lora_lr = lora_cfg.get("lr", cfg["training"]["lr"])
        param_groups.append({
            "params": [p for p in _unwrap(encoder).parameters() if p.requires_grad],
            "lr": lora_lr,
        })

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg["training"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["training"].get("lr_patience", 10), factor=0.5,
    )
    scaler = GradScaler()

    # Checkpointing
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lora" if lora_active else "frozen"
    best_ckpt = ckpt_dir / f"best_{suffix}_fold{cfg['run_fold']}.pt"
    best_val_loss = float("inf")

    # Training loop
    n_epochs: int = cfg["training"]["epochs"]
    for epoch in range(1, n_epochs + 1):
        # Shuffle is correctly seeded per epoch by DistributedSampler
        train_loader.sampler.set_epoch(epoch)

        t0 = time.perf_counter()

        train_m = _one_epoch(
            encoder, head, pool,
            train_loader,
            optimizer, scaler, criterion,
            device, img_size,
            is_train=True,
            lora_active=lora_active,
            evaluator=None,
        )

        val_m = _one_epoch(
            encoder, head, pool,
            val_loader,
            None, scaler, criterion,
            device, img_size,
            is_train=False,
            lora_active=lora_active,
            evaluator=evaluator,
        )

        scheduler.step(val_m["loss"])

        if main:
            print(
                f"[Fold {cfg['run_fold']} | {epoch:03d}/{n_epochs}]  "
                f"train loss={train_m['loss']:.4f} | "
                f"val loss={val_m['loss']:.4f} | "
                f"OA={val_m.get('Overall_Accuracy', 0):.4f} | "
                f"mIoU={val_m.get('mIoU', 0):.4f} | "
                f"F1={val_m.get('Macro_F1', 0):.4f}  "
                f"({time.perf_counter() - t0:.1f}s)"
            )

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                # Save raw (unwrapped) state dicts
                torch.save(
                    {
                        "head": _unwrap(head).state_dict(),
                        "encoder": _unwrap(encoder).state_dict(),
                    },
                    best_ckpt,
                )

        # All ranks must agree on whether to save; barrier keeps them in sync
        dist.barrier()

    # Restore best weights
    ckpt = torch.load(best_ckpt, map_location=device)
    _unwrap(head).load_state_dict(ckpt["head"])
    if lora_active:
        _unwrap(encoder).load_state_dict(ckpt["encoder"])
        if lora_cfg.get("merge_after_training", False) and main:
            encoder = _unwrap(encoder).merge_and_unload()
            print("LoRA weights merged into encoder.")

    if main:
        print(f"Best val loss {best_val_loss:.4f} -> {best_ckpt}")

    cleanup_ddp()
    return _unwrap(head)