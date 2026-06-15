# PASTIS DataLoader Utility

A unified, deterministic PyTorch DataLoader for the PASTIS dataset. It handles sequence zero-padding, strict spatial cross-validation, data efficiency subsampling, and cloud simulation to ensure 100% fair comparisons across different architectures.

## Quick Reference

Drop `load_data.py` into your `utils/` directory and import it into your training or evaluation scripts:

```python
from utils.load_data import get_pastis_dataloaders

train_loader, val_loader, test_loader = get_pastis_dataloaders(
    data_path="./data/PASTIS",
    batch_size=4,
    run_fold=1,       # Cross-validation target fold (1 to 5)
    fraction=1.0,     # Data efficiency: e.g., 0.10 for 10% of training data
    cloud_rate=0.0    # Cloud ablation: e.g., 0.30 to drop 30% of valid time steps
)

for x, y, mask in train_loader:
    # x: [Batch, 60, 10, 128, 128] -> Sentinel-2 SITS Tensors
    # y: [Batch, 128, 128]         -> Semantic Labels (0-19)
    # mask: [Batch, 60]            -> Padding & Cloud Attention Mask (True = Ignore)
    pass