import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from gmap.data.transforms import PointCloudTransforms

JOINT_TYPE_MAP = {"revolute": 0, "prismatic": 1}

class PartNetMobilityDataset(Dataset):
    def __init__(self, data_root: str, split_file: str, n_points: int = 8192, augment: bool = False):
        self.data_root = data_root
        self.n_points = n_points
        self.transform = PointCloudTransforms(n_points, normalize=True, augment=augment)
        with open(split_file, "r") as f:
            self.obj_ids = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.obj_ids)

    def __getitem__(self, idx: int) -> dict:
        obj_id = self.obj_ids[idx]
        obj_dir = os.path.join(self.data_root, obj_id)
        points = np.load(os.path.join(obj_dir, "point_cloud.npy")).astype(np.float32)
        seg_label = np.load(os.path.join(obj_dir, "seg_label.npy")).astype(np.int64)
        movable_label = np.load(os.path.join(obj_dir, "movable_label.npy")).astype(np.int64)

        n = points.shape[0]
        if n >= self.n_points:
            choice = np.random.choice(n, self.n_points, replace=False)
        else:
            choice = np.random.choice(n, self.n_points, replace=True)
        points = points[choice]
        seg_label = seg_label[choice]
        movable_label = movable_label[choice]

        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        with open(os.path.join(obj_dir, "joint_params.json"), "r") as f:
            joint = json.load(f)

        joint_type = JOINT_TYPE_MAP.get(joint["joint_type"], 0)
        axis_direction = np.array(joint["axis_direction"], dtype=np.float32)
        axis_position = np.array(joint.get("axis_position", [0, 0, 0]), dtype=np.float32)
        joint_state = float(joint.get("joint_state", 0.0))

        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "seg_label": torch.from_numpy(seg_label),
            "movable_label": torch.from_numpy(movable_label),
            "joint_type": joint_type,
            "axis_direction": torch.from_numpy(axis_direction),
            "axis_position": torch.from_numpy(axis_position),
            "joint_state": torch.tensor(joint_state, dtype=torch.float32),
        }
