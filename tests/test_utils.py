import numpy as np
import torch
import pytest
from gmap.utils.pc_utils import normalize_point_cloud, random_sample_points, fps_torch, knn_torch
from gmap.utils.checkpoint import save_checkpoint, load_checkpoint
from gmap.utils.logger import get_logger

def test_normalize_point_cloud():
    pc = np.random.randn(100, 3) * 5 + 10
    normed = normalize_point_cloud(pc)
    assert np.allclose(normed.mean(axis=0), 0, atol=1e-6)
    max_dist = np.max(np.sqrt(np.sum(normed ** 2, axis=1)))
    assert np.isclose(max_dist, 1.0, atol=1e-6)

def test_random_sample_points():
    pc = np.random.randn(200, 3)
    sampled = random_sample_points(pc, 100)
    assert sampled.shape == (100, 3)
    sampled_up = random_sample_points(pc, 300)
    assert sampled_up.shape == (300, 3)

def test_fps_torch():
    xyz = torch.randn(2, 64, 3)
    idx = fps_torch(xyz, 16)
    assert idx.shape == (2, 16)
    assert idx.max() < 64
    assert idx.min() >= 0

def test_knn_torch():
    xyz = torch.randn(2, 64, 3)
    centers = xyz[:, :8, :]
    idx = knn_torch(xyz, centers, k=4)
    assert idx.shape == (2, 8, 4)

def test_checkpoint(tmp_path):
    path = str(tmp_path / "test.pth")
    state = {"epoch": 10, "model": {"weight": torch.randn(3, 3)}}
    save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 10

def test_logger():
    logger = get_logger("test")
    assert logger is not None
    assert logger.name == "test"
