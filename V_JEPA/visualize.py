"""
visualize.py
Parses evaluation JSON files, computes 5-fold statistics, and generates plots.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Adjust this to match your config.yaml
CHECKPOINT_DIR = "../data/checkpoints/vjepa"
PLOT_DIR = "../data/plots"

# PASTIS classes for the confusion matrix axis
CLASSES = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed", 
    "Spring barley", "Sunflower", "Grapevine", "Beet", "Winter triticale", 
    "Winter durum wheat", "Fruits, veg, flowers", "Potatoes", "Leguminous fodder", 
    "Soybeans", "Orchard", "Mixed cereal", "Sorghum"
] # Assuming Background (0) and Void (19) are ignored in your final matrix

def load_and_group_results(results_dir):
    """Scan directory for JSONs and group them by experiment config (ignoring fold)."""
    search_pattern = os.path.join(results_dir, "test_metrics_*.json")
    files = glob.glob(search_pattern)
    
    experiments = {}
    
    for f in files:
        filename = os.path.basename(f)
        name_parts = filename.replace("test_metrics_", "").replace(".json", "").split("_fold")
        
        if len(name_parts) != 2:
            continue
            
        exp_group = name_parts[0]
        fold_num = name_parts[1]
        
        with open(f, 'r') as file:
            data = json.load(file)
            
        if exp_group not in experiments:
            experiments[exp_group] = []
            
        experiments[exp_group].append(data)
        
    return experiments

def aggregate_metrics(experiments):
    """Compute Mean and Std Dev across folds for primary metrics."""
    summary = []
    
    for exp_name, folds_data in experiments.items():
        mious = [f["mIoU"] for f in folds_data]
        f1s = [f["Macro_F1"] for f in folds_data]
        oas = [f["Overall_Accuracy"] for f in folds_data]
        
        summary.append({
            "Experiment": exp_name,
            "Folds Found": len(folds_data),
            "mIoU (Mean)": np.mean(mious),
            "mIoU (Std)": np.std(mious),
            "F1 (Mean)": np.mean(f1s),
            "F1 (Std)": np.std(f1s),
            "OA (Mean)": np.mean(oas),
        })
        
    df = pd.DataFrame(summary)
    return df.sort_values(by="mIoU (Mean)", ascending=False)

def plot_miou_comparison(df, save_path):
    """Bar chart comparing mIoU across different experiments with error bars."""
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=df, 
        x="mIoU (Mean)", 
        y="Experiment", 
        hue="Experiment",
        palette="viridis", 
        legend=False
    )
    
    # Add error bars
    plt.errorbar(
        df["mIoU (Mean)"], 
        range(len(df)), 
        xerr=df["mIoU (Std)"], 
        fmt='none', 
        c='black', 
        capsize=5
    )
    
    plt.title("5-Fold Cross-Validation: Mean mIoU Comparison")
    plt.xlabel("Mean mIoU")
    plt.ylabel("Configuration")
    plt.tight_layout()
    plt.savefig(save_path / "miou_comparison.png", dpi=300)
    plt.close()

def plot_aggregated_confusion_matrix(folds_data, exp_name, save_path):
    """Sum confusion matrices across all available folds and plot."""
    # Initialize an empty matrix of the correct size
    first_cm = np.array(folds_data[0]["Confusion_Matrix_Raw"])
    agg_cm = np.zeros_like(first_cm, dtype=np.float64)
    
    for fold in folds_data:
        agg_cm += np.array(fold["Confusion_Matrix_Raw"])
        
    # Row-normalize to show percentages
    row_sums = agg_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(agg_cm, row_sums, out=np.zeros_like(agg_cm), where=row_sums != 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        norm_cm, 
        annot=False, 
        cmap="Blues", 
        xticklabels=CLASSES[:len(norm_cm)], 
        yticklabels=CLASSES[:len(norm_cm)]
    )
    
    plt.title(f"Aggregated Confusion Matrix\n({exp_name})")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    filename = f"conf_matrix_{exp_name}.png"
    plt.savefig(save_path / filename, dpi=300)
    plt.close()

def main():
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for JSON results in {CHECKPOINT_DIR}...")
    experiments = load_and_group_results(CHECKPOINT_DIR)
    
    if not experiments:
        print("No evaluation JSON files found. Run evaluations first.")
        return
        
    print(f"Found {len(experiments)} distinct experiment configurations.")
    
    summary_df = aggregate_metrics(experiments)
    print("\n--- 5-Fold Cross Validation Summary ---")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(Path(PLOT_DIR) / "metrics_summary.csv", index=False)
    
    plot_miou_comparison(summary_df, Path(PLOT_DIR))
    print(f"\nSaved mIoU comparison plot to {PLOT_DIR}/miou_comparison.png")
    
    for exp_name, folds_data in experiments.items():
        plot_aggregated_confusion_matrix(folds_data, exp_name, Path(PLOT_DIR))
        
    print("Saved all aggregated confusion matrices.")

if __name__ == "__main__":
    main()