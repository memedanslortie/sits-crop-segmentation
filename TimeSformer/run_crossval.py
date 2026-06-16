import os
import subprocess
import json
import numpy as np
import yaml

def main():
    print("STARTING 5-FOLD CROSS-VALIDATION ")
    
    config_path = "configs/base.yaml"
    
    # Read the config exactly ONCE to safely extract the experiment name
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    exp_name = config['experiment']['name']
    
    # Run the experiment across all 5 geographical folds
    for fold in range(1, 6):
        print("\n" + "="*50)
        print(f"LAUNCHING FOLD {fold}/5")
        print("="*50)
        
        # check=True ensures the script completely stops if one fold crashes.
        subprocess.run(
            ["python", "baselines/run_baseline.py", "--fold", str(fold)], 
            check=True
        )
        
    # Aggregate the final results
    print("\n" + "="*50)
    print("📊 FINAL STATISTICAL RESULTS (5-FOLD CROSS-VALIDATION)")
    print("="*50)
    
    mious = []
    f1s = []
    
    for fold in range(1, 6):
        result_file = f"results/{exp_name}_fold{fold}_results.json"
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
                mious.append(data["mIoU"])
                f1s.append(data["Macro_F1"])
        else:
            print(f"Warning: The file {result_file} could not be found.")
            
    if len(mious) == 5:
        mean_miou = float(np.mean(mious))
        std_miou = float(np.std(mious))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        
        print(f"Overall mIoU    : {mean_miou:.4f} ± {std_miou:.4f}")
        print(f"Overall Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
        # Export the final statistical summary for the jury
        summary = {
            "experiment_name": exp_name,
            "mIoU_per_fold": mious,
            "F1_per_fold": f1s,
            "mIoU_mean": round(mean_miou, 4),
            "mIoU_std": round(std_miou, 4),
            "F1_mean": round(mean_f1, 4),
            "F1_std": round(std_f1, 4)
        }
        
        summary_path = f"results/{exp_name}_crossval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Full Cross-Validation Summary successfully saved to: {summary_path}")
        
    else:
        print("Cannot compute final statistics; some folds are missing.")

if __name__ == "__main__":
    main()