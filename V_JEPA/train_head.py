"""
Segmentation head training on a frozen V-JEPA 2.1 encoder for PASTIS crop segmentation.

Architecture:
  - Encoder (frozen ViT-B): each frame (B, 3, H, W) → (B, N_patches, D)
  - Temporal aggregation : mean-pool over valid (non-padded) timesteps → (B, N_patches, D)
  - Seg head (trainable)  : multi-scale MLP decoder → (B, num_classes, 128, 128)

The predictor returned by backbones.vjepa2_1_vit_base_384() is only used during
V-JEPA pretraining (mask-token prediction). It is intentionally ignored here,
exactly like the reconstruction decoder in U-BARN / Dumeur et al. 2024.
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler


import sys

from utils.load_data import get_pastis_dataloaders


class TemporalMeanPool(nn.Module):
    """Mean-pool patch tokens over valid (non-padded) timesteps.

    Args:
        mask: bool tensor (B, T) — True where the timestep is *padded* (to skip).
    """
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x    : (B, T, N, D)   patch tokens for every timestep
        # mask : (B, T)          True = padded/invalid frame
        valid = (~mask).float()                    # (B, T)
        valid = valid[:, :, None, None]            # (B, T, 1, 1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return pooled                              # (B, N, D)


class SegHead(nn.Module):
    """Lightweight MLP decoder: patch tokens → dense segmentation map.

    Follows V-JEPA 2.1 §8.2 "linear probe on patch tokens + bilinear upsample",
    extended with one hidden layer for the crop-class temporal complexity.

    Spatial layout: ViT-B/16 on 128×128 input → 8×8 = 64 patches (patch_size=16).
    ViT-B/16 on 384×384 input (vjepa2_1_vit_base_384) → 24×24 = 576 patches.
    We bilinearly upsample the spatial logit map to (128, 128).

    Args:
        embed_dim  : encoder hidden dim (768 for ViT-B).
        num_classes: number of segmentation classes.
        hidden_dim : inner MLP dimension.
        patch_grid : (H_patches, W_patches) spatial grid from the encoder.
    """
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
        # tokens: (B, N, D)  where N = H_p * W_p
        logits = self.mlp(tokens)                              # (B, N, C)
        B, N, C = logits.shape
        H_p, W_p = self.patch_grid
        assert N == H_p * W_p, f"Token count mismatch: {N} vs {H_p}×{W_p}"
        logits = logits.permute(0, 2, 1).reshape(B, C, H_p, W_p)
        return F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)

def _compute_patch_grid(encoder: nn.Module, img_size: int = 384) -> tuple[int, int]:
    """Infer (H_patches, W_patches) from the encoder's patch_size attribute."""
    patch_size = getattr(encoder, "patch_size", 16)
    n = img_size // patch_size
    return n, n


@torch.no_grad()
def _encode_sits(
    encoder: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    """Run the frozen encoder frame-by-frame, return (B, T, N, D).

    Args:
        x    : (B, T, C, H, W)  — input SITS (H=W=128, resized internally).
        mask : (B, T)            — True where timestep is padded.
        img_size: spatial resolution expected by the encoder (e.g. 384).
    """
    B, T, C, H, W = x.shape
    # Resize each frame to the resolution the ViT was pretrained on.
    if H != img_size or W != img_size:
        flat = x.view(B * T, C, H, W)
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    else:
        flat = x.view(B * T, C, H, W)

    # Encode in one batched forward pass (memory permitting) or in T chunks.
    tokens = encoder(flat)          # (B*T, N, D)  — standard ViT output
    _, N, D = tokens.shape
    return tokens.view(B, T, N, D)


def _one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    temporal_pool: TemporalMeanPool,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    img_size: int,
    noloss_classes: set[int],
    is_train: bool,
) -> dict[str, float]:
    head.train(is_train)
    encoder.eval()  # always frozen

    total_loss = total_correct = total_valid = n_batches = 0

    for x, y, mask in loader:
        # x    : (B, T, C, 128, 128)
        # y    : (B, 128, 128)         long
        # mask : (B, T)                bool, True = padded frame
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with torch.autocast('cuda'):
            with torch.no_grad():
                tokens = _encode_sits(encoder, x, mask, img_size)   # (B, T, N, D)
            pooled = temporal_pool(tokens, mask)                      # (B, N, D)
            logits = head(pooled)                                     # (B, C, 128, 128)

            # Mask out noloss classes (e.g. background/void = 19 in PASTIS)
            valid_px = ~torch.isin(y, torch.tensor(list(noloss_classes), device=device))
            loss = criterion(logits, y)
            loss = (loss * valid_px).sum() / valid_px.sum().clamp(min=1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = logits.argmax(1)
        total_correct += ((preds == y) & valid_px).sum().item()
        total_valid   += valid_px.sum().item()
        total_loss    += loss.item()
        n_batches     += 1

    return {
        "loss": total_loss / n_batches,
        "oa":   total_correct / max(total_valid, 1),
    }


def train(cgf: dict, encoder: nn.Module, predictor: nn.Module) -> nn.Module:  # noqa: ARG001
    """Train a segmentation head on frozen V-JEPA 2.1 features for PASTIS.

    Args:
        cgf      : parsed config dict (see config.yaml).
        encoder  : frozen V-JEPA 2.1 ViT encoder.
        predictor: V-JEPA predictor — unused here (pretraining artifact).

    Returns:
        Trained SegHead module.
    """
    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(cgf["seed"])

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_pastis_dataloaders(
        data_path=cgf["data"]["dataset_path"],
        batch_size=cgf["training"]["batch_size"],
        run_fold=cgf["run_fold"],
        fraction=cgf["ablation"]["dataset_fraction"],
        cloud_rate=cgf["ablation"]["cloud_drop_rate"],
        to_rgb=cgf["data"]["to_rgb"],
    )

    # ── Encoder (frozen) ───────────────────────────────────────────────────
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Infer encoder output dim from a dummy forward pass
    img_size: int = cgf["training"].get("img_size", 384)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        dummy_out = encoder(dummy)               # (1, N, D)
        embed_dim = dummy_out.shape[-1]
    patch_grid = _compute_patch_grid(encoder, img_size)

    # ── Segmentation head ──────────────────────────────────────────────────
    num_classes: int = cgf["training"]["num_classes"]
    head = SegHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=cgf["training"].get("hidden_dim", 512),
        patch_grid=patch_grid,
    ).to(device)

    temporal_pool = TemporalMeanPool()

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=cgf["training"]["lr"],
        weight_decay=cgf["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cgf["training"].get("lr_patience", 10), factor=0.5
    )
    scaler = GradScaler()

    # Cross-entropy with per-pixel reduction so we can mask void classes ourselves.
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    noloss_classes = set(cgf["data"]["noloss_classes"])

    # ── Checkpointing ──────────────────────────────────────────────────────
    ckpt_dir = Path(cgf["training"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt = ckpt_dir / f"best_seg_head_fold{cgf['run_fold']}.pt"

    # ── Training loop ──────────────────────────────────────────────────────
    n_epochs: int = cgf["training"]["epochs"]
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        train_metrics = _one_epoch(
            encoder, head, temporal_pool, train_loader,
            optimizer, scaler, criterion, device,
            img_size, noloss_classes, is_train=True,
        )
        val_metrics = _one_epoch(
            encoder, head, temporal_pool, val_loader,
            None, scaler, criterion, device,
            img_size, noloss_classes, is_train=False,
        )
        scheduler.step(val_metrics["loss"])
        elapsed = time.perf_counter() - t0

        print(
            f"[Fold {cgf['run_fold']} | Epoch {epoch:03d}/{n_epochs}]  "
            f"train loss={train_metrics['loss']:.4f}  oa={train_metrics['oa']:.4f}  |  "
            f"val loss={val_metrics['loss']:.4f}  oa={val_metrics['oa']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(head.state_dict(), best_ckpt)

    # Load best weights before returning
    head.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Best val loss {best_val_loss:.4f} — checkpoint: {best_ckpt}")
    return head