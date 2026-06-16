"""
Segmentation head training on a frozen (or LoRA-adapted) V-JEPA 2.1 encoder
for PASTIS crop segmentation.

Architecture:
  - Encoder (frozen ViT-B, optionally + LoRA via peft): frame → (B, N_patches, D)
  - Temporal aggregation : masked mean-pool over valid timesteps → (B, N_patches, D)
  - Seg head (trainable)  : MLP decoder → (B, num_classes, 128, 128)

PASTIS label conventions (from dataset doc):
  - 0          : Background (non-agricultural land — valid class, included in loss)
  - 1-18       : Crop types
  - 19         : Void (out-of-scope parcels — excluded from loss AND metrics)

Class weighting (training.class_weighting: true):
  Inverse-frequency weights computed from pixel counts across the training set.
  Void (class 19) weight is forced to 0 regardless so it never affects the loss.
  Weights are cached to <checkpoint_dir>/class_weights_fold<N>.pt to avoid
  reloading every .npy file on each run.
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
    """Mean-pool patch tokens over valid (non-padded) timesteps.

    Args:
        x   : (B, T, N, D) — patch tokens for every timestep.
        mask: (B, T) bool  — True where the timestep is padded (skip it).
    """
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float()[:, :, None, None]               # (B, T, 1, 1)
        return (x * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B, N, D)


class SegHead(nn.Module):
    """MLP decoder: patch tokens → dense segmentation map.

    Follows V-JEPA 2.1 §8.2 (linear probe on patch tokens + bilinear upsample),
    extended with one hidden layer for the 18-class crop complexity.

    vjepa2_1_vit_base_384: patch_size=16, input=384 → 24×24=576 tokens, D=768.
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
        logits = self.mlp(tokens)                               # (B, N, C)
        B, N, C = logits.shape
        H_p, W_p = self.patch_grid
        logits = logits.permute(0, 2, 1).reshape(B, C, H_p, W_p)
        return F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)

def _compute_class_weights(
    train_dataset,
    num_classes: int,
    noloss_classes: set[int],
    cache_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """Inverse-frequency weights over training pixels; noloss classes → 0.

    Weights are cached to ``cache_path`` so the full npy scan only runs once.

    Returns:
        Float tensor of shape (num_classes,) on ``device``.
    """
    if cache_path.exists():
        print(f"Loading cached class weights from {cache_path}")
        weights = torch.load(cache_path, map_location="cpu")
    else:
        print("Computing class weights (one-time scan of training labels)…")
        counts = train_dataset.label_counts(num_classes)        # (C,) float64
        # Zero out noloss classes before computing inverse freq so they don't
        # steal weight from valid classes.
        for c in noloss_classes:
            counts[c] = 0.0
        total = counts.sum()
        # Inverse-frequency: rarer classes get higher weight.
        weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))
        weights = weights.float()
        torch.save(weights, cache_path)
        print(f"Class weights cached to {cache_path}")

    return weights.to(device)

def _apply_lora(encoder: nn.Module, lora_cfg: dict) -> nn.Module:
    """Wrap encoder with peft LoRA adapters targeting attention linears."""
    config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg.get("alpha", lora_cfg["rank"]),
        # qkv is a single fused Linear(768→2304); proj is the output projection.
        target_modules=["qkv", "proj"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(encoder, config)
    encoder.print_trainable_parameters()
    return encoder


def _patch_grid(encoder: nn.Module, img_size: int) -> tuple[int, int]:
    patch_size = getattr(encoder, "patch_size", 16)
    n = img_size // patch_size
    return n, n


def _encode_sits(encoder: nn.Module, x: torch.Tensor, img_size: int) -> torch.Tensor:
    """Encode all frames independently, return (B, T, N, D)."""
    B, T, C, H, W = x.shape
    flat = x.flatten(0, 1)                                      # (B*T, C, H, W)
    if H != img_size or W != img_size:
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return encoder(flat).unflatten(0, (B, T))                   # (B, T, N, D)


def _one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    pool: TemporalMeanPool,
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

def train(cfg: dict, encoder: nn.Module, predictor: nn.Module) -> nn.Module:  # noqa: ARG001
    """Train a segmentation head on V-JEPA 2.1 features for PASTIS.

    Args:
        cfg      : parsed config dict.
        encoder  : V-JEPA 2.1 ViT encoder.
        predictor: unused (V-JEPA pretraining artifact).

    Returns:
        Trained SegHead.
    """
    from utils.load_data import get_pastis_dataloaders

    num_classes: int = cfg.get("training", {}).get("num_classes", 20)
    noloss_classes: set[int] = set(cfg["data"]["noloss_classes"])

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = SITS_Evaluator(num_classes=num_classes, ignore_indices=list(noloss_classes))

    train_loader, val_loader, _ = get_pastis_dataloaders(
        data_path=cfg["data"]["dataset_path"],
        batch_size=cfg["training"]["batch_size"],
        run_fold=cfg["run_fold"],
        fraction=cfg["ablation"]["dataset_fraction"],
        cloud_rate=cfg["ablation"]["cloud_drop_rate"],
        to_rgb=cfg["data"]["to_rgb"],
    )

    # Encoder
    encoder = encoder.to(device)
    lora_cfg: dict | None = cfg["training"].get("lora")
    lora_active = lora_cfg is not None

    if lora_active:
        encoder = _apply_lora(encoder, lora_cfg)
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    # Head
    img_size: int = cfg["training"].get("img_size", 384)
    with torch.no_grad():
        embed_dim = encoder(torch.zeros(1, 3, img_size, img_size, device=device)).shape[-1]

    head = SegHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["training"].get("hidden_dim", 512),
        patch_grid=_patch_grid(encoder, img_size),
    ).to(device)
    pool = TemporalMeanPool()

    # Loss
    void_mask_t = torch.tensor(sorted(noloss_classes), device=device)

    class_weight: torch.Tensor | None = None
    if cfg["training"].get("class_weighting", False):
        ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "./checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cache = ckpt_dir / f"class_weights_fold{cfg['run_fold']}.pt"
        class_weight = _compute_class_weights(
            train_loader.dataset, num_classes, noloss_classes, cache, device
        )

    # weight=None → uniform; weight[19]=0 → void never penalised.
    # ignore_index=-1 is a safety net (no label should ever be -1 in PASTIS).
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
            device, img_size, void_mask_t,
            is_train=True,
            lora_active=lora_active,
            evaluator=None,
        )

        val_m = _one_epoch(
            encoder, head, pool,
            val_loader,
            None, scaler, criterion,
            device, img_size, void_mask_t,
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

    print(f"Best val loss {best_val_loss:.4f} → {best_ckpt}")
    return head