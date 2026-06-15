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
            ignore_indices = [0, 19]
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

        # Intersection over Union (IoU) per class & Mean IoU
        iou_per_class = TP / (TP + FP + FN + 1e-10)
        mIoU = np.nanmean(iou_per_class)

        # F1-Score per class & Macro F1
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)
        macro_f1 = np.nanmean(f1_per_class)

        # Package results into a clean dictionary
        results = {
            "Overall_Accuracy": round(float(overall_accuracy), 4),
            "mIoU": round(float(mIoU), 4),
            "Macro_F1": round(float(macro_f1), 4),
            "Per_Class_IoU": {
                PASTIS_CLASSES.get(c, f"Class_{c}"): round(float(iou_per_class[idx]), 4) 
                for idx, c in enumerate(valid_classes)
            },
            "Per_Class_F1": {
                PASTIS_CLASSES.get(c, f"Class_{c}"): round(float(f1_per_class[idx]), 4) 
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