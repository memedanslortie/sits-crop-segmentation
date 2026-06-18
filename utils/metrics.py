import os
import json
import torch
import numpy as np

from utils.logging import get_logger
log = get_logger()

# PASTIS official class names (0-19) based on official documentation.
PASTIS_CLASSES = {
    0: "Background",
    1: "Meadow", 
    2: "Soft winter wheat", 
    3: "Corn", 
    4: "Winter barley", 
    5: "Winter rapeseed", 
    6: "Spring barley", 
    7: "Sunflower", 
    8: "Grapevine", 
    9: "Beet", 
    10: "Winter triticale", 
    11: "Winter durum wheat", 
    12: "Fruits, vegetables, flowers", 
    13: "Potatoes", 
    14: "Leguminous fodder", 
    15: "Soybeans", 
    16: "Orchard", 
    17: "Mixed cereal", 
    18: "Sorghum", 
    19: "Void label"
}

class SITS_Evaluator:
    def __init__(self, num_classes=20, ignore_indices=None):
        if ignore_indices is None:
            ignore_indices = [19]  # U-TAE protocol: ignore ONLY void; background(0) is scored
        self.num_classes = num_classes
        self.ignore_indices = set(ignore_indices or [19])

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.reset()

    def reset(self):
        """GPU confusion matrix accumulator (flat bincount form)."""
        self.conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def add_batch(self, predictions, targets):
        """
        GPU-optimized accumulation using torch.bincount.
        No sklearn, no CPU roundtrip per batch.
        """

        preds = predictions.to(self.device).flatten().long()
        targs = targets.to(self.device).flatten().long()

        # Build valid mask (exclude ignore classes)
        if len(self.ignore_indices) > 0:
            mask = torch.ones_like(targs, dtype=torch.bool)
            for ign in self.ignore_indices:
                mask &= (targs != ign)

            preds = preds[mask]
            targs = targs[mask]

        if preds.numel() == 0:
            return

        # Flat index: (pred * C + target)
        idx = preds * self.num_classes + targs

        bincount = torch.bincount(
            idx,
            minlength=self.num_classes * self.num_classes,
        )

        cm = bincount.reshape(self.num_classes, self.num_classes)

        self.conf_matrix += cm

    def compute_metrics(self):
        """CPU metrics computation (unchanged logic, now faster pipeline)."""

        cm = self.conf_matrix.detach().cpu().numpy()

        valid_classes = [
            i for i in range(self.num_classes) if i not in self.ignore_indices
        ]

        cm = cm[np.ix_(valid_classes, valid_classes)]

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP

        overall_accuracy = np.sum(TP) / (np.sum(cm) + 1e-10)

        # Ground-truth support per class. Classes ABSENT from this fold are undefined and
        # must be excluded (set to NaN) rather than scored as 0 — counting them as 0 would
        # artificially deflate mIoU/F1 and break comparability with the U-TAE protocol,
        # which averages only over classes present in the split.
        support = TP + FN

        # Intersection over Union (IoU) per class & Mean IoU
        iou_per_class = np.where(support > 0, TP / np.maximum(TP + FP + FN, 1), np.nan)
        # Guard against an all-NaN slice (e.g. a single patch with no scored pixels)
        mIoU = float(np.nanmean(iou_per_class)) if np.any(support > 0) else 0.0

        # F1-Score per class & Macro F1
        precision = TP / np.maximum(TP + FP, 1)
        recall = TP / np.maximum(TP + FN, 1)
        f1_per_class = np.where(
            support > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-10),
            np.nan,
        )
        macro_f1 = float(np.nanmean(f1_per_class)) if np.any(support > 0) else 0.0

        results = {
            "Overall_Accuracy": round(float(overall_accuracy), 4),
            "mIoU": round(float(mIoU), 4),
            "Macro_F1": round(float(macro_f1), 4),

            "Per_Class_IoU": {
                PASTIS_CLASSES.get(c, f"Class_{c}"): (
                    None if np.isnan(iou_per_class[idx]) else round(float(iou_per_class[idx]), 4)
                )
                for idx, c in enumerate(valid_classes)
            },

            "Per_Class_F1": {
                PASTIS_CLASSES.get(c, f"Class_{c}"): (
                    None if np.isnan(f1_per_class[idx]) else round(float(f1_per_class[idx]), 4)
                )
                for idx, c in enumerate(valid_classes)
            },

            "Confusion_Matrix_Raw": cm.tolist(),
        }

        return results

    def save_results(self, filepath):
        results = self.compute_metrics()

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        log.info(f"✅ Evaluation complete → {filepath}")
        log.info(
            f"📊 mIoU: {results['mIoU']:.4f} | Macro F1: {results['Macro_F1']:.4f}"
        )

        return results