import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PASTIS_Dataset(Dataset):
    def __init__(self, data_path, target_folds, max_seq_len=60, dataset_fraction=1.0, seed=42, cloud_drop_rate=0.0):
        """
        Initializes the PASTIS Dataset with Data Efficiency and Cloud Simulation features.
        
        Args:
            data_path (str): Path to PASTIS folder (DATA_S2, ANNOTATIONS, metadata.geojson).
            target_folds (list): List of folds to load (e.g., [1, 2, 3] for train).
            max_seq_len (int): Fixed temporal length for zero-padding.
            dataset_fraction (float): Percentage of data to keep (0.10, 0.25, 0.50, 1.0).
            seed (int): Fixed seed to ensure V-JEPA and TimeSformer use the EXACT same subsets.
            cloud_drop_rate (float): Percentage of valid time steps to artificially mask (0.0 to 1.0).
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.target_folds = target_folds
        self.cloud_drop_rate = cloud_drop_rate
        
        #  Parse metadata and filter by fold
        metadata_path = os.path.join(data_path, "metadata.geojson")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing file: {metadata_path}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        all_patch_ids = []
        for feature in metadata["features"]:
            fold = feature["properties"]["Fold"]
            if fold in self.target_folds:
                all_patch_ids.append(feature["properties"]["ID_PATCH"])
                
        #  DATA EFFICIENCY: Deterministic Subsampling
        if dataset_fraction < 1.0:
            # Sort first to ensure deterministic order before shuffling
            all_patch_ids.sort()
            random.seed(seed)
            random.shuffle(all_patch_ids)
            
            keep_count = int(len(all_patch_ids) * dataset_fraction)
            self.patch_ids = all_patch_ids[:keep_count]
            print(f"Data Efficiency [{dataset_fraction*100}%]: Kept {len(self.patch_ids)} out of {len(all_patch_ids)} patches (Seed: {seed}).")
        else:
            self.patch_ids = all_patch_ids
            print(f"Loaded Folds {self.target_folds}: {len(self.patch_ids)} patches found.")
                
    def __len__(self):
        return len(self.patch_ids)
        
    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        
        # Load Sentinel-2 SITS (Shape: [T, 10, 128, 128])
        s2_path = os.path.join(self.data_path, "DATA_S2", f"S2_{patch_id}.npy")
        x = np.load(s2_path).astype(np.float32)
        x = torch.from_numpy(x)
        
        # Load Target (Semantic class is at index 0)
        target_path = os.path.join(self.data_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy")
        y = np.load(target_path)[0] 
        y = torch.from_numpy(y).long()
        
        # CLOUD SIMULATION (Ablation Study)
        valid_seq_len = x.shape[0]
        cloud_mask = torch.zeros(valid_seq_len, dtype=torch.bool)
        
        if self.cloud_drop_rate > 0.0:
            # Calculate how many dates to drop
            num_clouds = int(valid_seq_len * self.cloud_drop_rate)
            if num_clouds > 0:
                # Randomly pick indices to obscure
                cloud_indices = torch.randperm(valid_seq_len)[:num_clouds]
                # Zero out the data to simulate thick cloud cover
                x[cloud_indices] = 0.0
                # Flag them so the attention mechanism can (optionally) ignore them entirely
                cloud_mask[cloud_indices] = True
        
        # TEMPORAL PADDING
        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool) 
        
        if valid_seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - valid_seq_len
            zeros_pad = torch.zeros((pad_size, x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)
            x = torch.cat([x, zeros_pad], dim=0)
            
            # Combine cloud mask and padding mask
            final_mask = torch.cat([cloud_mask, torch.ones(pad_size, dtype=torch.bool)], dim=0)
            padding_mask = final_mask
            
        elif valid_seq_len > self.max_seq_len:
            x = x[:self.max_seq_len]
            padding_mask = cloud_mask[:self.max_seq_len]
        else:
            padding_mask = cloud_mask
        
        # x: [60, 10, 128, 128]
        # y: [128, 128]
        # padding_mask: [60] (True where there is padding OR a simulated cloud)
        return x, y, padding_mask

def get_pastis_dataloaders(data_path, batch_size=4, num_workers=4, run_fold=1, fraction=1.0, cloud_rate=0.0):
    """
    Generates DataLoaders with optional data efficiency and cloud simulation parameters.
    """
    all_folds = [1, 2, 3, 4, 5]
    test_fold = run_fold
    val_fold = 5 if test_fold == 1 else test_fold - 1
    train_folds = [f for f in all_folds if f not in [val_fold, test_fold]]
    
    print(f"--- Init Dataloaders | Test Fold: {test_fold} | Fraction: {fraction} | Cloud Rate: {cloud_rate} ---")
    
    # Subsampling is ONLY applied to the training set. 
    # Val and Test must always evaluate on 100% of their respective folds.
    train_dataset = PASTIS_Dataset(data_path, target_folds=train_folds, dataset_fraction=fraction, cloud_drop_rate=cloud_rate)
    
    # Val and Test get fraction=1.0 and cloud_rate=0.0 (unless you specifically want to test cloud robustness on the Test set)
    val_dataset   = PASTIS_Dataset(data_path, target_folds=[val_fold], dataset_fraction=1.0, cloud_drop_rate=0.0)
    
    # If evaluating Exp 3 (Cloud Robustness), apply the cloud_rate to the test set!
    test_dataset  = PASTIS_Dataset(data_path, target_folds=[test_fold], dataset_fraction=1.0, cloud_drop_rate=cloud_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader