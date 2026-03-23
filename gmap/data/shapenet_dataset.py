import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from gmap.data.transforms import PointCloudTransforms

class ShapeNetDataset(Dataset):
    def __init__(self, h5_path: str, n_points: int = 8192, augment: bool = False):
        self.h5_path = h5_path
        self.n_points = n_points
        self.transform = PointCloudTransforms(n_points, normalize=True, augment=augment)
        with h5py.File(h5_path, "r") as f:
            self.data = f["data"][:].astype(np.float32)
            self.labels = f["label"][:].astype(np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        pc = self.data[idx]
        label = self.labels[idx]
        pc = self.transform(pc)
        return torch.from_numpy(pc), torch.tensor(label, dtype=torch.long)
