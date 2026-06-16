import os
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
from TimeSformer.timesformer_sits import SITS_TimeSformer

# DETERMINISM
def set_seed(seed=42):
    """Locks all random number generators to ensure 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="configs/base.yaml"):
    """Loads YAML config and overrides run_fold if passed via CLI."""
    parser = argparse.ArgumentParser(description="Run SITS-TimeSformer Baseline")
    parser.add_argument("--fold", type=int, default=None, help="Force the execution of a specific fold")
    
    # parse_known_args is safer in case other arguments are passed (e.g., by Kaggle/Jupyter)
    args, _ = parser.parse_known_args()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Override the fold from YAML if provided in CLI
    if args.fold is not None:
        config['experiment']['run_fold'] = args.fold
        print(f"CLI Override: Setting run_fold to {args.fold}")
        
    return config
def apply_training_strategy(model, config):
    """Applies Linear Probing, Full Fine-Tuning, or LoRA based on the config."""
    if config['model'].get('use_lora', False):
        print("Strategy: LoRA (Low-Rank Adaptation)")
        from peft import LoraConfig, get_peft_model
 
        # Freeze all weights first
        for param in model.parameters():
            param.requires_grad = False
 
        # Unfreeze the spatial decoder head AND our custom temporal layers
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.temporal_aggregator.parameters(): 
            param.requires_grad = True
        for param in model.feature_fusion.parameters(): 
            param.requires_grad = True
 
        # Inject trainable low-rank matrices into the Attention blocks
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value"],
            lora_dropout=0.1, bias="none"
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
 
    elif config['model'].get('freeze_backbone', False):
        print("Strategy: Linear Probing (Frozen Backbone)")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the spatial decoder head AND our custom temporal layers
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

#LOSS FUNCTIONS
class SITS_LossBuilder(nn.Module):
    def __init__(self, loss_type="ce", ignore_indices=None):
        """
        Unified loss builder controlled by configs/base.yaml.
        Supported types: 'ce' | 'focal' | 'dice_ce'
        """
        super().__init__()
        if ignore_indices is None:
            ignore_indices = [0, 19]
        self.loss_type = loss_type
        self.ignore_indices = ignore_indices

    def forward(self, logits, targets):
        # Build a boolean mask to strictly exclude Background (0) and Void (19)
        valid_mask = ~torch.isin(
            targets,
            torch.tensor(self.ignore_indices, device=targets.device)
        )

        # Edge case: if the entire batch is ignored (e.g., only background), return 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        # Compute the base unreduced cross-entropy for all pixels
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Route to the correct loss function based on the YAML config
        if self.loss_type == "ce":
            # Standard Cross-Entropy — the PASTIS state-of-the-art default (U-TAE, TSViT)
            return ce_loss[valid_mask].mean()

        elif self.loss_type == "focal":
            # Focal Loss (gamma=2.0) — down-weights easy examples, focuses on hard ones
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** 2.0) * ce_loss
            return focal_loss[valid_mask].mean()

        elif self.loss_type == "dice_ce":
            # Dice + CE hybrid — robust to spatial class imbalance
            ce_mean = ce_loss[valid_mask].mean()

            probs = torch.softmax(logits, dim=1)
            targets_onehot = (
                F.one_hot(targets.clamp(0), logits.shape[1])
                .permute(0, 3, 1, 2)
                .float()
            )

            # Mask intersection and union to exclude ignored classes
            intersection = (probs * targets_onehot * valid_mask.unsqueeze(1)).sum(dim=(0, 2, 3))
            union = ((probs + targets_onehot) * valid_mask.unsqueeze(1)).sum(dim=(0, 2, 3))
            dice_loss = 1.0 - (2.0 * intersection / (union + 1e-6)).mean()

            return 0.5 * ce_mean + 0.5 * dice_loss

        else:
            raise ValueError(f"Loss function '{self.loss_type}' not recognized. "
                             f"Choose from: 'ce', 'focal', 'dice_ce'.")


# MAIN TRAINING LOOP

def main():
    config = load_config()
    set_seed(config['experiment']['seed'])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"🚀 Initializing training on: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader = get_pastis_dataloaders(
        data_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        run_fold=config['experiment']['run_fold'],
        fraction=config['ablation']['dataset_fraction'],
        cloud_rate=config['ablation']['cloud_drop_rate']
    )

    # Model Initialization
    model = SITS_TimeSformer(
        in_channels=config['data']['in_channels'], 
        num_classes=config['data']['num_classes']
    )
    model = apply_training_strategy(model, config).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss & Evaluator
    criterion = SITS_LossBuilder(
        loss_type=config['training']['loss_function'],
        ignore_indices=config['data']['ignore_classes']
    )  
    evaluator = SITS_Evaluator(num_classes=config['data']['num_classes'], ignore_indices=config['data']['ignore_classes'])
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    
    best_val_miou = 0.0
    os.makedirs("results/checkpoints", exist_ok=True)

    # FIX: Training History Tracking
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_f1": []}

    # Epoch Loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['num_epochs']} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch_x, batch_y, batch_mask in train_pbar:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        evaluator.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in tqdm(val_loader, desc="Validation"):
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                
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
        print(f"Val mIoU: {current_miou:.4f} | Val Macro F1: {current_f1:.4f}")

        # Save History
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_miou"].append(current_miou)
        history["val_f1"].append(current_f1)

        # Checkpoint Saving
        if current_miou > best_val_miou:
            best_val_miou = current_miou
            torch.save(model.state_dict(), f"results/checkpoints/best_model_fold{config['experiment']['run_fold']}.pth")
            print(f"🌟 New best model saved! (mIoU: {best_val_miou:.4f})")

    # --- FINAL TESTING PHASE ---
    print("\n Launching Final Evaluation on Test Fold...")
    
    model = SITS_TimeSformer(in_channels=config['data']['in_channels'], num_classes=config['data']['num_classes'])
    model = apply_training_strategy(model, config)
    
    model.load_state_dict(torch.load(f"results/checkpoints/best_model_fold{config['experiment']['run_fold']}.pth", weights_only=True))
    if torch.cuda.device_count() > 1:
        print(f" {torch.cuda.device_count()} GPUs via DataParallel ")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    model.eval()
    evaluator.reset()
    
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(batch_x, batch_mask)
            evaluator.add_batch(torch.argmax(logits, dim=1), batch_y)
            
    exp_name = config['experiment']['name']
    fold = config['experiment']['run_fold']
    
    final_results = evaluator.compute_metrics()
    final_results["metadata"] = {
        "fold": fold,
        "seed": config['experiment']['seed'],
        "strategy": "lora" if config['model'].get('use_lora') else (
                    "linear_probing" if config['model'].get('freeze_backbone') else "full_finetuning"),
        "dataset_fraction": config['ablation']['dataset_fraction'],
        "cloud_drop_rate": config['ablation']['cloud_drop_rate'],
        "best_val_miou": best_val_miou,
        "config": config
    }
    
    with open(f"results/{exp_name}_fold{fold}_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results and Metadata successfully saved to results/{exp_name}_fold{fold}_results.json")
    
    # Export Training Curves History
    with open(f"results/{exp_name}_fold{fold}_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training curves saved to results/{exp_name}_fold{fold}_history.json")

if __name__ == "__main__":
    main()