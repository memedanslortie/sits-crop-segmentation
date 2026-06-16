"""
Segmentation head training on a frozen (or LoRA-adapted) V-JEPA 2.1 encoder
for PASTIS crop segmentation using native 3D video encoding.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.amp import GradScaler, autocast

from utils.metrics import SITS_Evaluator

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
        
        # Truncate mask in case the 3D conv dropped residual frames at the end
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
    train_dataset, num_classes: int, noloss_classes: set[int], cache_path: Path, device: torch.device
) -> torch.Tensor:
    if cache_path.exists():
        print(f"Loading cached class weights from {cache_path}")
        weights = torch.load(cache_path, map_location="cpu")
    else:
        print("Computing class weights (one-time scan of training labels)...")
        counts = train_dataset.label_counts(num_classes)
        for c in noloss_classes:
            counts[c] = 0.0
        total = counts.sum()
        weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))
        weights = weights.float()
        torch.save(weights, cache_path)
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
    encoder.train(is_train and lora_active)

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

            # tokens: (B, L, D) -> pool -> (B, H_p * W_p, D)
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


def train(cfg: dict, encoder: nn.Module, predictor: nn.Module) -> nn.Module:
    from utils.load_data import get_pastis_dataloaders

    num_classes: int = int(cfg.get("training", {}).get("num_classes", 20))
    noloss_classes: set[int] = set(cfg["data"]["noloss_classes"])

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = SITS_Evaluator(num_classes=num_classes, ignore_indices=list(noloss_classes), device=device)

    train_loader, val_loader, _ = get_pastis_dataloaders(
        data_path=cfg["data"]["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        run_fold=cfg["run_fold"],
        fraction=cfg["ablation"]["dataset_fraction"],
        cloud_rate=cfg["ablation"]["cloud_drop_rate"],
        to_rgb=cfg["data"]["to_rgb"],
    )

    # Encoder Setup
    encoder = encoder.to(device)
    lora_cfg: dict | None = cfg["training"].get("lora")
    lora_active = lora_cfg is not None

    if lora_active:
        encoder = _apply_lora(encoder, lora_cfg)
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    # Dynamic 3D Grid Calculation
    img_size: int = cfg["training"].get("img_size", 384)
    with torch.no_grad():
        # Dynamically read exact sequence length and channels from the dataloader
        sample_x, _, _ = train_loader.dataset[0]
        max_seq_len, in_channels, _, _ = sample_x.shape
        
        # Pass dummy 5D video to infer correct token grid
        dummy_input = torch.zeros(30, in_channels, max_seq_len, img_size, img_size, device=device)
        dummy_out = encoder(dummy_input)
        
        embed_dim = dummy_out.shape[-1]
        num_tokens = dummy_out.shape[1]
        
        patch_size = getattr(encoder, "patch_size", 16)
        h_p = w_p = img_size // patch_size
        t_p = num_tokens // (h_p * w_p)
        
        patch_grid_3d = (t_p, h_p, w_p)

    head = SegHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["training"].get("hidden_dim", 512),
        patch_grid=(h_p, w_p),  # Seg head only needs the 2D spatial dimensions
    ).to(device)
    
    pool = TemporalMeanPool(patch_grid_3d)

    # Loss Setup
    class_weight: torch.Tensor | None = None
    if cfg["training"].get("class_weighting", False):
        ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cache = ckpt_dir / f"class_weights_fold{cfg['run_fold']}.pt"
        class_weight = _compute_class_weights(
            train_loader.dataset, num_classes, noloss_classes, cache, device
        )
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-1)

    # Optimiser
    param_groups = [{"params": head.parameters(), "lr": cfg["training"]["lr"]}]
    if lora_active:
        lora_lr = lora_cfg.get("lr", cfg["training"]["lr"])
        param_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": lora_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["training"].get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["training"].get("lr_patience", 10), factor=0.5,
    )
    scaler = GradScaler()

    # Checkpointing
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix    = "lora" if lora_active else "frozen"
    best_ckpt = ckpt_dir / f"best_{suffix}_fold{cfg['run_fold']}.pt"
    best_val_loss = float("inf")

    # Loop
    n_epochs: int = cfg["training"]["epochs"]
    for epoch in range(1, n_epochs + 1):
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
            torch.save({"head": head.state_dict(), "encoder": encoder.state_dict()}, best_ckpt)

    # Restore best
    ckpt = torch.load(best_ckpt, map_location=device)
    head.load_state_dict(ckpt["head"])
    if lora_active:
        encoder.load_state_dict(ckpt["encoder"])
        if lora_cfg.get("merge_after_training", False):
            encoder = encoder.merge_and_unload()
            print("LoRA weights merged into encoder.")

    print(f"Best val loss {best_val_loss:.4f} -> {best_ckpt}")
    return head