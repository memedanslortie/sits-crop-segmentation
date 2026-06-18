import os
import subprocess
import json
import numpy as np
import yaml


def main():
    print("STARTING 5-FOLD CROSS-VALIDATION")

    config_path = "TimeSformer/configs/base.yaml"

    # Read the config exactly ONCE to safely extract the experiment name
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    exp_name = config['experiment']['name']

    # Run the experiment across all 5 geographical folds
    for fold in range(1, 6):
        print("\n" + "=" * 50)
        print(f"LAUNCHING FOLD {fold}/5")
        print("=" * 50)

        # FIX: Correct path — run_baseline.py lives in TimeSformer/, not baselines/
        subprocess.run(
            ["python", "TimeSformer/run_baseline.py", "--fold", str(fold)],
            check=True  # Stops immediately if one fold crashes
        )

    # ==========================================
    # AGGREGATE FINAL RESULTS
    # ==========================================
    print("\n" + "=" * 50)
    print("STATISTICAL RESULTS (5-FOLD CROSS-VALIDATION)")
    print("=" * 50)

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
            print(f"Warning: {result_file} not found — fold {fold} may have crashed.")

    if len(mious) == 5:
        mean_miou = float(np.mean(mious))
        std_miou  = float(np.std(mious))
        mean_f1   = float(np.mean(f1s))
        std_f1    = float(np.std(f1s))

        print(f"Overall mIoU    : {mean_miou:.4f} ± {std_miou:.4f}")
        print(f"Overall Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"U-TAE reference : 0.6310 ± —")

        # Export the final statistical summary for the jury
        summary = {
            "experiment_name": exp_name,
            "mIoU_per_fold":   mious,
            "F1_per_fold":     f1s,
            "mIoU_mean":       round(mean_miou, 4),
            "mIoU_std":        round(std_miou,  4),
            "F1_mean":         round(mean_f1,   4),
            "F1_std":          round(std_f1,    4),
            "utae_reference_miou": 0.631
        }

        os.makedirs("results", exist_ok=True)
        summary_path = f"results/{exp_name}_crossval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Cross-Validation Summary saved to: {summary_path}")

    else:
        print(f" Only {len(mious)}/5 folds completed. Cannot compute final statistics.")


if __name__ == "__main__":
    main()