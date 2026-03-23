import torch
import numpy as np
import json
import pytest
from gmap.data.partnet_dataset import PartNetMobilityDataset

def test_partnet_mock(tmp_path):
    obj_dir = tmp_path / "100710"
    obj_dir.mkdir()
    np.save(obj_dir / "point_cloud.npy", np.random.randn(8192, 3).astype(np.float32))
    np.save(obj_dir / "seg_label.npy", np.random.randint(0, 4, 8192).astype(np.int64))
    np.save(obj_dir / "movable_label.npy", np.random.randint(0, 2, 8192).astype(np.int64))
    joint_data = {
        "joint_type": "revolute",
        "axis_direction": [0.0, 1.0, 0.0],
        "axis_position": [0.1, 0.2, 0.3],
        "joint_state": 0.5,
    }
    with open(obj_dir / "joint_params.json", "w") as f:
        json.dump(joint_data, f)
    split_file = tmp_path / "train.txt"
    split_file.write_text("100710\n")
    ds = PartNetMobilityDataset(
        data_root=str(tmp_path), split_file=str(split_file), n_points=8192,
    )
    assert len(ds) == 1
    sample = ds[0]
    assert sample["points"].shape == (8192, 3)
    assert sample["seg_label"].shape == (8192,)
    assert sample["movable_label"].shape == (8192,)
    assert sample["joint_type"] in [0, 1]
    assert sample["axis_direction"].shape == (3,)
