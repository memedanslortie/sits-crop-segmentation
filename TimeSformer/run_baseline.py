import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import yaml
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data import get_pastis_dataloaders
from utils.metrics import SITS_Evaluator
from timesformer_sits import SITS_TimeSformer

# ==========================================
# 1. SCIENTIFIC RIGOR: ABSOLUTE DETERMINISM
# ==========================================
def set_seed(seed=42):
    """Locks all random number generators to ensure 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="TimeSformer/configs/base.yaml"):
    """Loads YAML config and overrides run_fold if passed via CLI."""
    parser = argparse.ArgumentParser(description="Run SITS-TimeSformer Baseline")
    parser.add_argument("--fold", type=int, default=None, help="Force the execution of a specific fold")
    args, _ = parser.parse_known_args()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.fold is not None:
        config['experiment']['run_fold'] = args.fold
        print(f"CLI Override: Setting run_fold to {args.fold}")

    return config

def get_base_model(model):
    """Unwraps DataParallel if needed to access the raw model."""
    return model.module if isinstance(model, nn.DataParallel) else model

# ==========================================
# 2. CORE TRAINING STRATEGY
# ==========================================
def apply_training_strategy(model, config):
    """Applies Linear Probing, Full Fine-Tuning, or LoRA based on the config."""
    if config['model'].get('use_lora', False):
        print("Strategy: LoRA (Low-Rank Adaptation)")
        from peft import LoraConfig, get_peft_model

        for param in model.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.temporal_aggregator.parameters():
            param.requires_grad = True
        for param in model.feature_fusion.parameters():
            param.requires_grad = True

        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["attention.attention.qkv", "temporal_attention.attention.qkv"],
            lora_dropout=0.1, bias="none"
        )
        model.backbone = get_peft_model(model.backbone, lora_config)

        # CRITICAL: get_peft_model() freezes EVERY non-adapter weight, including the
        # input embeddings. Three input layers carry no usable pretrained signal for SITS
        # and must be allowed to adapt, otherwise the LoRA adapters learn on top of a
        # frozen, ill-initialised input — which is the true bottleneck of this transfer:
        #   - patch_embeddings.projection : our 10-band conv, inflated from 3 RGB (crude init)
        #   - position_embeddings         : randomly RE-INITIALISED (224px K400 -> 128px mismatch)
        #   - time_embeddings             : 8-frame K400 table interpolated to 60 SITS frames
        # We re-enable them AFTER get_peft_model so they train alongside the LoRA adapters.
        unfrozen = []
        for name, param in model.backbone.named_parameters():
            if any(k in name for k in ("patch_embeddings.projection",
                                       "position_embeddings",
                                       "time_embeddings")):
                param.requires_grad = True
                unfrozen.append(name)
        print(f"   ↳ Unfroze {len(unfrozen)} input-embedding tensors for domain adaptation.")

    elif config['model'].get('freeze_backbone', False):
        print("Strategy: Linear Probing (Frozen Backbone)")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.temporal_aggregator.parameters():
            param.requires_grad = True
        for param in model.feature_fusion.parameters():
            param.requires_grad = True

    else:
        print("Strategy: Full Fine-Tuning (All weights are learning)")
        for param in model.parameters():
            param.requires_grad = True

    return model

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================
def compute_class_weights(counts, ignore_indices, num_classes, device, max_weight=10.0):
    """Median-frequency balancing (Eigen & Fergus, 2015) over the SCORED classes.

    w_c = median(freq) / freq_c  → frequent classes (e.g. background) get w<1, rare crops
    get w>1. Capped at `max_weight` so ultra-rare classes (e.g. Soybeans) can't destabilise
    training. Ignored classes (void) receive weight 0. Returns a [num_classes] tensor.
    """
    counts = counts.double()
    scored = [c for c in range(num_classes) if c not in ignore_indices]
    scored_counts = counts[scored]
    freq = scored_counts / scored_counts.sum().clamp(min=1)
    nonzero = freq[freq > 0]
    med = torch.median(nonzero) if len(nonzero) > 0 else torch.tensor(1.0, dtype=torch.float64)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c, f in zip(scored, freq):
        if f > 0:
            weights[c] = float((med / f).clamp(max=max_weight))
    return weights.to(device)


class SITS_LossBuilder(nn.Module):
    def __init__(self, loss_type="ce", ignore_indices=None, class_weights=None):
        """
        Unified loss builder controlled by TimeSformer/configs/base.yaml.
        Supported types: 'ce' | 'focal' | 'dice_ce'
        `class_weights`: optional [num_classes] tensor for class-balanced CE.
        """
        super().__init__()
        if ignore_indices is None:
            ignore_indices = [19]
        self.loss_type = loss_type
        self.ignore_indices = ignore_indices
        self.class_weights = class_weights

    def forward(self, logits, targets):
        valid_mask = ~torch.isin(
            targets,
            torch.tensor(self.ignore_indices, device=targets.device)
        )

        if not valid_mask.any():
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')

        if self.loss_type == "ce":
            return ce_loss[valid_mask].mean()

        elif self.loss_type == "focal":
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** 2.0) * ce_loss
            return focal_loss[valid_mask].mean()

        elif self.loss_type == "dice_ce":
            ce_mean = ce_loss[valid_mask].mean()
            probs = torch.softmax(logits, dim=1)
            targets_onehot = (
                F.one_hot(targets.clamp(0), logits.shape[1])
                .permute(0, 3, 1, 2)
                .float()
            )
            vm = valid_mask.unsqueeze(1)
            intersection = (probs * targets_onehot * vm).sum(dim=(0, 2, 3))
            union = ((probs + targets_onehot) * vm).sum(dim=(0, 2, 3))
            dice_per_class = 2.0 * intersection / (union + 1e-6)
            # Average Dice ONLY over scored classes. Ignored channels (background/void)
            # never appear as valid targets, so their Dice is structurally ~0 and would
            # otherwise inject a constant penalty that dilutes the gradient signal.
            keep = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
            keep[torch.tensor(self.ignore_indices, device=logits.device)] = False
            dice_loss = 1.0 - dice_per_class[keep].mean()
            return 0.5 * ce_mean + 0.5 * dice_loss

        else:
            raise ValueError(f"Loss function '{self.loss_type}' not recognized. "
                             f"Choose from: 'ce', 'focal', 'dice_ce'.")

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    config = load_config()
    set_seed(config['experiment']['seed'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpus = torch.cuda.device_count() if use_cuda else 0
    print(f"Initializing training on: {device} ({n_gpus} GPU(s) detected)")

    # --- DataLoaders ---
    train_loader, val_loader, test_loader = get_pastis_dataloaders(
        data_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        run_fold=config['experiment']['run_fold'],
        fraction=config['ablation']['dataset_fraction'],
        cloud_rate=config['ablation']['cloud_drop_rate']
    )

    # --- Model Initialization ---
    model = SITS_TimeSformer(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        high_res_decoder=config['model'].get('high_res_decoder', False)
    )
    model = apply_training_strategy(model, config)

    # FIX: Enable DataParallel BEFORE moving to device
    if n_gpus > 1:
        print(f" {n_gpus} GPUs detected — enabling DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Optimizer & Scheduler ---
    # FIX: Use get_base_model() to access parameters through DataParallel wrapper
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, get_base_model(model).parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    class_weights = None
    if config['training'].get('class_weighting', False):
        print("Computing median-frequency class weights from the training set...")
        counts = train_loader.dataset.label_counts(config['data']['num_classes'])
        class_weights = compute_class_weights(
            counts, config['data']['ignore_classes'], config['data']['num_classes'], device,
            max_weight=config['training'].get('class_weight_cap', 3.0)
        )
        print(f"   ↳ Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    criterion = SITS_LossBuilder(
        loss_type=config['training']['loss_function'],
        ignore_indices=config['data']['ignore_classes'],
        class_weights=class_weights
    )
    evaluator = SITS_Evaluator(
        num_classes=config['data']['num_classes'],
        ignore_indices=config['data']['ignore_classes']
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    best_val_miou = 0.0
    os.makedirs("results/checkpoints", exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_f1": []}

    # ==========================================
    # EPOCH LOOP
    # ==========================================
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['num_epochs']} ---")

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training")

        for batch_x, batch_y, batch_mask in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                get_base_model(model).parameters(),
                config['training']['gradient_clipping']
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        evaluator.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, batch_mask in tqdm(val_loader, desc="Validation"):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)

                with torch.amp.autocast('cuda', enabled=use_cuda):
                    logits = model(batch_x, batch_mask)
                    loss = criterion(logits, batch_y)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                evaluator.add_batch(preds, batch_y)

        metrics = evaluator.compute_metrics()
        current_miou = metrics['mIoU']
        current_f1 = metrics['Macro_F1']
        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val mIoU:   {current_miou:.4f} | Val Macro F1: {current_f1:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_miou"].append(current_miou)
        history["val_f1"].append(current_f1)

        # FIX: Use get_base_model() to save weights without DataParallel wrapper
        if current_miou > best_val_miou:
            best_val_miou = current_miou
            torch.save(
                get_base_model(model).state_dict(),
                f"results/checkpoints/best_model_fold{config['experiment']['run_fold']}.pth"
            )
            print(f"New best model saved! (mIoU: {best_val_miou:.4f})")

    # ==========================================
    # FINAL TEST EVALUATION
    # ==========================================
    print("\n🚀 Launching Final Evaluation on Test Fold...")

    # Rebuild architecture first (critical for LoRA: layers must exist before loading weights)
    model = SITS_TimeSformer(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        high_res_decoder=config['model'].get('high_res_decoder', False)
    )
    model = apply_training_strategy(model, config)
    model.load_state_dict(torch.load(
        f"results/checkpoints/best_model_fold{config['experiment']['run_fold']}.pth",
        weights_only=True
    ))
    # No DataParallel for final inference — single GPU is sufficient
    model = model.to(device)
    model.eval()
    evaluator.reset()

    with torch.no_grad():
        for batch_x, batch_y, batch_mask in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(batch_x, batch_mask)

            evaluator.add_batch(torch.argmax(logits, dim=1), batch_y)

    # ==========================================
    # EXPORT RESULTS & METADATA
    # ==========================================
    exp_name = config['experiment']['name']
    fold = config['experiment']['run_fold']

    if config['model'].get('use_lora'):
        strategy = "lora"
    elif config['model'].get('freeze_backbone'):
        strategy = "linear_probing"
    else:
        strategy = "full_finetuning"

    final_results = evaluator.compute_metrics()
    final_results["metadata"] = {
        "fold": fold,
        "seed": config['experiment']['seed'],
        "strategy": strategy,
        "loss_function": config['training']['loss_function'],
        "dataset_fraction": config['ablation']['dataset_fraction'],
        "cloud_drop_rate": config['ablation']['cloud_drop_rate'],
        "best_val_miou": best_val_miou,
        "config": config
    }

    results_path = f"results/{exp_name}_fold{fold}_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results and metadata saved to {results_path}")

    history_path = f"results/{exp_name}_fold{fold}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training curves saved to {history_path}")


if __name__ == "__main__":
    main()