import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PASTISDataset(Dataset):
    """PASTIS-HD Sentinel-2 time series dataset with optional subsampling and cloud simulation."""

    def __init__(
        self,
        data_path: str | Path,
        target_folds: list[int],
        max_seq_len: int = 60,
        dataset_fraction: float = 1.0,
        seed: int = 42,
        cloud_drop_rate: float = 0.0,
        norm: bool = True,
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.cloud_drop_rate = cloud_drop_rate
        self.norm = norm

        metadata_path = self.data_path / "metadata.geojson"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.metadata = {}
        patch_ids = []
        for feature in metadata["features"]:
            props = feature["properties"]
            if props["Fold"] in target_folds:
                p_id = props["ID_PATCH"]
                self.metadata[p_id] = props
                patch_ids.append(p_id)

        if dataset_fraction < 1.0:
            patch_ids.sort()
            rng = random.Random(seed)
            rng.shuffle(patch_ids)
            keep_count = int(len(patch_ids) * dataset_fraction)
            self.patch_ids = patch_ids[:keep_count]
            print(
                f"Data Efficiency [{dataset_fraction:.0%}]: kept {len(self.patch_ids)}/{len(patch_ids)} "
                f"patches (seed={seed})."
            )
        else:
            self.patch_ids = patch_ids
            print(f"Loaded folds {target_folds}: {len(self.patch_ids)} patches.")

        if self.norm:
            with open(self.data_path / "NORM_S2_patch.json") as f:
                norm_data = json.load(f)
            stats = norm_data.get(f"Fold_{target_folds[0]}", next(iter(norm_data.values())))
            self.mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1)
            self.std = torch.tensor(stats["std"]).view(1, -1, 1, 1)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_id = self.patch_ids[idx]

        x = torch.from_numpy(np.load(self.data_path / "DATA_S2" / f"S2_{patch_id}.npy")).float()
        if self.norm:
            x = (x - self.mean) / self.std

        y = torch.from_numpy(np.load(self.data_path / "ANNOTATIONS" / f"TARGET_{patch_id}.npy")[0]).long()

        valid_seq_len = x.shape[0]
        cloud_mask = torch.zeros(valid_seq_len, dtype=torch.bool)

        if self.cloud_drop_rate > 0.0:
            num_clouds = int(valid_seq_len * self.cloud_drop_rate)
            if num_clouds > 0:
                cloud_indices = torch.randperm(valid_seq_len)[:num_clouds]
                x[cloud_indices] = 0.0
                cloud_mask[cloud_indices] = True

        # Pad or truncate to a fixed temporal length; mask is True for padding OR simulated clouds.
        if valid_seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - valid_seq_len
            x = torch.cat([x, torch.zeros((pad_size, *x.shape[1:]), dtype=x.dtype)], dim=0)
            mask = torch.cat([cloud_mask, torch.ones(pad_size, dtype=torch.bool)], dim=0)
        else:
            x = x[: self.max_seq_len]
            mask = cloud_mask[: self.max_seq_len]

        # x: [max_seq_len, 10, 128, 128], y: [128, 128], mask: [max_seq_len]
        return x, y, mask


def get_pastis_dataloaders(
    data_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    run_fold: int = 1,
    fraction: float = 1.0,
    cloud_rate: float = 0.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders following the PASTIS 5-fold cross-validation protocol.

    Subsampling (`fraction`) and cloud simulation only apply to train; val is always
    evaluated on 100% clean data, and test optionally inherits `cloud_rate` for
    cloud-robustness evaluation.
    """
    all_folds = [1, 2, 3, 4, 5]
    test_fold = run_fold
    val_fold = 5 if test_fold == 1 else test_fold - 1
    train_folds = [f for f in all_folds if f not in (val_fold, test_fold)]

    print(f"--- Init dataloaders | test fold: {test_fold} | fraction: {fraction} | cloud rate: {cloud_rate} ---")

    train_dataset = PASTISDataset(data_path, train_folds, dataset_fraction=fraction, cloud_drop_rate=cloud_rate)
    val_dataset = PASTISDataset(data_path, [val_fold], dataset_fraction=1.0, cloud_drop_rate=0.0)
    test_dataset = PASTISDataset(data_path, [test_fold], dataset_fraction=1.0, cloud_drop_rate=cloud_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader