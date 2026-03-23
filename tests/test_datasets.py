import torch
import numpy as np
import pytest
from gmap.data.transforms import PointCloudTransforms
from gmap.data.shapenet_dataset import ShapeNetDataset

def test_transforms_normalize():
    t = PointCloudTransforms(n_points=1024, normalize=True, augment=False)
    pc = np.random.randn(2048, 3).astype(np.float32) * 5 + 10
    result = t(pc)
    assert result.shape == (1024, 3)
    assert np.abs(result.mean(axis=0)).max() < 0.1

def test_transforms_augment():
    t = PointCloudTransforms(n_points=1024, normalize=True, augment=True)
    pc = np.random.randn(2048, 3).astype(np.float32)
    r1 = t(pc)
    r2 = t(pc)
    assert not np.allclose(r1, r2)

def test_shapenet_dataset_mock(tmp_path):
    import h5py
    h5_path = tmp_path / "shapenet_train.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=np.random.randn(10, 8192, 3).astype(np.float32))
        f.create_dataset("label", data=np.arange(10, dtype=np.int64))
    ds = ShapeNetDataset(str(h5_path), n_points=8192, augment=False)
    assert len(ds) == 10
    pc, label = ds[0]
    assert pc.shape == (8192, 3)
    assert isinstance(pc, torch.Tensor)
