import os
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

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
        self.ignore_indices = ignore_indices
        self.reset()

    def reset(self):
        """Resets the confusion matrix for a new epoch/evaluation."""
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add_batch(self, predictions, targets):
        """
        Updates the confusion matrix with a new batch.
        """
        # Flatten tensors to 1D arrays
        preds = predictions.cpu().numpy().flatten()
        targs = targets.cpu().numpy().flatten()

        mask = ~np.isin(targs, self.ignore_indices)
        preds = preds[mask]
        targs = targs[mask]

        # Update confusion matrix securely
        if len(targs) > 0:
            self.conf_matrix += confusion_matrix(
                targs, preds, labels=np.arange(self.num_classes)
            )

    def compute_metrics(self):
        """Computes OA, mIoU, and Macro-F1 from the confusion matrix."""
        valid_classes = [i for i in range(self.num_classes) if i not in self.ignore_indices]
        
        # Isolate valid classes
        cm = self.conf_matrix[valid_classes][:, valid_classes]

        # Calculate True Positives, False Positives, False Negatives
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP

        # Pixel Accuracy (Standard but biased)
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

        # Package results into a clean dictionary
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
            "Confusion_Matrix_Raw": self.conf_matrix.tolist()
        }
        return results

    def save_results(self, filepath):
        """Computes metrics and exports them to a highly readable JSON file."""
        results = self.compute_metrics()
        
        # Ensure directory exists ONLY if a directory is specified in the path
        dirname = os.path.dirname(filepath)
        if dirname: 
            os.makedirs(dirname, exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"✅ Evaluation complete. Results saved perfectly to: {filepath}")
        print(f"📊 Quick Recap -> mIoU: {results['mIoU']:.4f} | Macro F1: {results['Macro_F1']:.4f}")
        
        return results