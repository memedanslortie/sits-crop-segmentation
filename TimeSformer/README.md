# 🚀 SITS-TimeSformer: Training & Experiments

This repository contains the official training pipeline for adapting a pre-trained video Transformer (TimeSformer) to 10-channel Satellite Image Time Series (SITS) using the PASTIS dataset.

---

## 🛠️ Prerequisites

If you are running this locally or on a custom GPU server, ensure you have the required libraries installed.

> **Note:** If you are using Kaggle, most of these are pre-installed.

```bash
pip install torch torchvision
pip install transformers scikit-learn tqdm pyyaml
pip install peft  # Required for LoRA experiments
```

---

## ⚙️ Configuration (`configs/base.yaml`)

We use a strict **configuration-driven approach** to ensure 100% scientific reproducibility. Do not modify the Python scripts directly to change hyperparameters. Instead, edit `configs/base.yaml`.

Key parameters to understand:

| Parameter | Description |
|---|---|
| `experiment.run_fold` | Which of the 5 spatial folds to use for testing (1–5) |
| `ablation.dataset_fraction` | Set between `0.1` and `1.0` for Data Efficiency tests |
| `ablation.cloud_drop_rate` | Set between `0.0` and `1.0` to simulate cloud cover |
| `model.use_lora` | Set to `true` to inject Low-Rank Adapters, `false` otherwise |

---

## 🏃 Running an Experiment

Once your data is located at `data/PASTIS` (or your Kaggle input path), simply execute the baseline script. It will automatically read the YAML config, load the model, apply weight inflation, and start the training loop.

```bash
python baselines/run_baseline.py
```

---

## 🧪 The 3 Training Strategies (Experiment 1)

Our architecture supports three distinct training paradigms. Switch between them by adjusting `configs/base.yaml`.

### 1. Linear Probing (Baseline)

**Goal:** Test the raw quality of the pre-trained features. Freezes the entire 768-dim backbone; only trains the final spatial decoder.

Config setup:
```yaml
model.use_lora: false
# Set freeze_backbone: true in the training script initialization
```

### 2. Full Fine-Tuning (Performance Ceiling)

**Goal:** Maximize mIoU by updating all 100M+ parameters of the TimeSformer.

Config setup:
```yaml
model.use_lora: false
training.learning_rate: 0.0001  # Keep very low to avoid destroying pre-trained weights
```

### 3. LoRA — Low-Rank Adaptation (SOTA Efficiency)

**Goal:** Achieve fine-tuning performance while only training ~1% of the parameters, by injecting tiny matrices into the Attention blocks.

Config setup:
```yaml
model.use_lora: true
```

---

## 📊 Outputs & Evaluation

We strictly evaluate using **mIoU** and **Macro F1-Score**, formally excluding the `Background` (index 0) and `Void` (index 19) classes to prevent metric distortion.

Upon completion of a run, the evaluator automatically generates a detailed JSON report in the `results/` directory:

```
results/TimeSformer_Base_Fold1_results.json
```

The report includes:
- Raw confusion matrix
- Per-class IoU and F1 scores
- Aggregated mIoU and Macro F1 for the fold

---

## 🗂️ Repository Structure

```
.
├── baselines/
│   └── run_baseline.py         # Main training & evaluation entry point
├── configs/
│   └── base.yaml               # Single source of truth for all hyperparameters
├── data/
│   └── PASTIS/                 # Dataset root (local or Kaggle input path)
├── models/
│   └── timesformer_sits.py     # TimeSformer adapted for 10-channel SITS
├── utils/
│   └── load_data.py            # DataLoader with padding mask generation
├── results/                    # Auto-generated JSON reports per fold/run
├── PROTOCOL.md                 # Scientific protocol, hypotheses, and split definitions
└── README.md
```

---

## 📋 Next Steps

The next step is to write `baselines/run_baseline.py`, the script that ties together the DataLoader, the Model, the Metrics, and the YAML config. Once complete, push everything to GitHub and trigger your first Kaggle run.