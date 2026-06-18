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
from custom_build import build_vjepa2_1_encoder

from utils.logging import get_logger
log = get_logger()

def setup_ddp() -> tuple[int, int, torch.device]:
    """Initialise the process group and return (rank, world_size, device)."""
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size, torch.device(f"cuda:{rank}")

def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0

def load_encoder(cfg: dict, rank: int, device: torch.device) -> nn.Module:
    """
    Memory-safe encoder loading for large models under DDP.

    Strategy
    --------
    1. Rank 0 downloads the checkpoint to a known cache path on disk.
    2. All ranks barrier — now the file is guaranteed to be on disk.
    3. Every rank builds the model with pretrained=False (cheap, no download),
       then loads the cached state dict directly onto its own GPU
       with map_location=device.  Peak extra RAM = one state dict on CPU
       per rank (loaded and freed immediately), never two full GPU copies.
    4. The patch-embed is replaced for multi-spectral input inside
       build_vjepa2_1_encoder; we only need to load the base weights here.
    """
    from vjepa2.src.hub.backbones import ARCH_NAME_MAP, _clean_backbone_key

    backbone_name = cfg["model"]["backbone"]
    is_rgb = cfg["data"].get("to_rgb", False)

    VJEPA_BASE_URL = os.getenv("VJEPA_BASE_URL", "https://dl.fbaipublicfiles.com/vjepa2")
    model_file = ARCH_NAME_MAP[backbone_name][-1]
    url = VJEPA_BASE_URL + f"/{model_file}.pt"

    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
    encoder_cache = ckpt_dir / f"encoder_weights_{model_file}.pt"

    _CHECKPOINT_KEY_MAP = {
        "vjepa2_1_vit_base_384":     "ema_encoder",
        "vjepa2_1_vit_large_384":    "ema_encoder",
        "vjepa2_1_vit_giant_384":    "target_encoder",
        "vjepa2_1_vit_gigantic_384": "target_encoder",
    }
    checkpoint_key = _CHECKPOINT_KEY_MAP[backbone_name]

    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not encoder_cache.exists():
            log.info(f"[rank 0] Ensuring checkpoint is cached: {url}")
            raw = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            
            encoder_sd = _clean_backbone_key(raw[checkpoint_key])
            torch.save(encoder_sd, encoder_cache)
            log.info(f"[rank 0] Encoder state dict cached to {encoder_cache}")
            
            del raw
            del encoder_sd
            torch.cuda.empty_cache()
        else:
            log.info(f"[rank 0] Using existing encoder cache {encoder_cache}")

    dist.barrier()

    encoder = build_vjepa2_1_encoder(
        model_name=backbone_name,
        is_rgb=is_rgb,
        pretrained=False,
    )

    encoder_sd = torch.load(encoder_cache, map_location=device)
    
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if is_main(rank) and (missing or unexpected):
        log.info(f"[rank 0] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    
    del encoder_sd
    torch.cuda.empty_cache()

    encoder = encoder.to(device)

    if is_main(rank):
        log.info(f"Backbone '{backbone_name}' loaded on all ranks.")

    return encoder

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
            log.info(f"Loading cached class weights from {cache_path}")
            weights = torch.load(cache_path, map_location="cpu")
        else:
            log.info("Computing class weights (one-time scan of training labels)...")
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


def _encode_sits(encoder: nn.Module, x: torch.Tensor, img_size: int = 128) -> torch.Tensor:
    """Encode the full time series natively as a 3D video tensor."""
    B, T, C, H, W = x.shape

    assert H == img_size and W == img_size, f"Input must be {img_size}x{img_size}, got {H}x{W}"

    # Transpose to (B, C, T, H, W)
    video_tensor = x.permute(0, 2, 1, 3, 4)
    return encoder(video_tensor)  # (B, L, D)

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

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with ctx, autocast("cuda"):
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
        if dist.is_initialized():
            dist.all_reduce(evaluator.conf_matrix, op=dist.ReduceOp.SUM)
        
        out.update(evaluator.compute_metrics())
    return out


def train(cfg: dict, rank, world_size, device) -> nn.Module:
    from utils.load_data import get_pastis_dataloaders

    # DDP init
    # rank, world_size, device = setup_ddp()
    main = is_main(rank)

    # Load encoder and predictor on the correct GPU
    encoder = load_encoder(cfg, rank, device)

    if main:
        log.info(f"DDP active: {world_size} GPUs")

    # Reproducibility
    torch.manual_seed(cfg["seed"] + rank)  # Different seed per rank for data augmentation

    # Config
    num_classes: int = int(cfg.get("training", {}).get("num_classes", 20))
    noloss_classes: set[int] = set(cfg["data"]["noloss_classes"])
    img_size: int = cfg["training"].get("img_size", 128)
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
        log.info(f"Patch grid 3D: {patch_grid_3d} | embed_dim: {embed_dim}")

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
    _lora_mlt = lora_cfg.get("patience_mlt", 2) if lora_active else 1
    _lr_patience = cfg["training"].get("lr_patience", 10) * _lora_mlt
    _stp_patience = cfg["training"].get("stp_patience", 25) * _lora_mlt

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=_lr_patience, factor=0.5,
    )
    scaler = GradScaler()

    # Checkpointing
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lora" if lora_active else "frozen"
    best_ckpt = ckpt_dir / f"best_{suffix}_fold{cfg['run_fold']}.pt"
    best_val_miou = -1.0

    # Training loop
    n_epochs: int = cfg["training"]["epochs"]
    epochs_since_improve = 0
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

        val_miou = val_m["mIoU"]

        scheduler.step(val_miou)

        improved = val_miou > best_val_miou
        if improved:
            best_val_miou = val_miou

        if main:
            log.info(
                f"[Fold {cfg['run_fold']} | {epoch:03d}/{n_epochs}]  "
                f"train loss={train_m['loss']:.4f} | "
                f"val loss={val_m['loss']:.4f} | "
                f"OA={val_m.get('Overall_Accuracy', 0):.4f} | "
                f"mIoU={val_miou:.4f}{'*' if improved else ' '} | "
                f"F1={val_m.get('Macro_F1', 0):.4f}  "
                f"({time.perf_counter() - t0:.1f}s)"
            )

            if improved:
                torch.save(
                    {
                        "head": _unwrap(head).state_dict(),
                        "encoder": _unwrap(encoder).state_dict(),
                    },
                    best_ckpt,
                )

        # broadcast improved flag so early stopping is consistent
        improved_tensor = torch.tensor(int(improved), device=device)
        dist.broadcast(improved_tensor, src=0)
        epochs_since_improve = 0 if improved_tensor.item() else epochs_since_improve + 1

        dist.barrier()

        if epochs_since_improve >= _stp_patience:
            if main:
                log.info(f"Early stopping: no mIoU improvement for {_stp_patience} epochs.")
            break

    # Restore best weights
    ckpt = torch.load(best_ckpt, map_location=device)
    _unwrap(head).load_state_dict(ckpt["head"])
    if lora_active:
        _unwrap(encoder).load_state_dict(ckpt["encoder"])
        if lora_cfg.get("merge_after_training", False) and main:
            encoder = _unwrap(encoder).merge_and_unload()
            log.info("LoRA weights merged into encoder.")

    if main:
        log.info(f"Best val mIoU {best_val_miou:.4f} -> {best_ckpt}")

    # cleanup_ddp()
    return best_ckpt