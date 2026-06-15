import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import Dataset

REPO_ID = "IGNF/PASTIS-HD"


class PASTISDataset(Dataset):
    def __init__(self, folds: list[int] | None = None, norm: bool = True, cache_dir: str | None = None):
        folds = folds or [1]
        self.norm = norm

        norm_path = hf_hub_download(REPO_ID, "NORM_S2_patch.json", repo_type="dataset", cache_dir=cache_dir)
        with open(norm_path) as f:
            norm_data = json.load(f)
        stats = norm_data.get(f"Fold_{folds[0]}", next(iter(norm_data.values())))
        self.mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1)
        self.std = torch.tensor(stats["std"]).view(1, -1, 1, 1)

        geojson_path = hf_hub_download(REPO_ID, "metadata.geojson", repo_type="dataset", cache_dir=cache_dir)
        with open(geojson_path) as f:
            geojson_data = json.load(f)

        self.metadata = {}
        self.patch_ids = []
        for feature in geojson_data["features"]:
            props = feature["properties"]
            p_id, f_id = str(props["ID_PATCH"]), int(props["Fold"])
            if f_id in folds:
                self.metadata[p_id] = props
                self.patch_ids.append(p_id)

        allow_patterns = [f"DATA_S2/S2_{p_id}.npy" for p_id in self.patch_ids] + [
            f"ANNOTATIONS/ParcelIDs_{p_id}.npy" for p_id in self.patch_ids
        ]
        self.data_dir = snapshot_download(
            REPO_ID, repo_type="dataset", allow_patterns=allow_patterns, cache_dir=cache_dir
        )

        print(f"Loaded {len(self.patch_ids)} patches for folds {folds}")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_id = self.patch_ids[idx]

        x = torch.from_numpy(np.load(f"{self.data_dir}/DATA_S2/S2_{patch_id}.npy")).float()
        if self.norm:
            x = (x - self.mean) / self.std

        y = torch.from_numpy(np.load(f"{self.data_dir}/ANNOTATIONS/ParcelIDs_{patch_id}.npy")).long()
        if y.ndim == 3:
            y = y[0]

        dates = torch.tensor([float(v) for v in self.metadata[patch_id]["dates-S2"].values()])
        dates = (dates - dates.min()) / (dates.max() - dates.min() + 1e-5)

        return x, dates, y