import torch
import pytest
from gmap.models.pointnet2_utils import (
    farthest_point_sample,
    knn_query,
    group_points,
    MultiScaleGrouping,
)

@pytest.fixture
def sample_pc():
    return torch.randn(2, 1024, 3)

def test_farthest_point_sample(sample_pc):
    idx = farthest_point_sample(sample_pc, 256)
    assert idx.shape == (2, 256)

def test_knn_query(sample_pc):
    centers = sample_pc[:, :64, :]
    idx = knn_query(sample_pc, centers, k=16)
    assert idx.shape == (2, 64, 16)

def test_group_points(sample_pc):
    idx = torch.randint(0, 1024, (2, 64, 16))
    grouped = group_points(sample_pc, idx)
    assert grouped.shape == (2, 64, 16, 3)

def test_multi_scale_grouping():
    pc = torch.randn(2, 8192, 3)
    msg = MultiScaleGrouping(
        n_points=8192,
        scales=[(512, 32), (256, 8), (64, 8)],
    )
    patches_list, centers_list = msg(pc)
    assert len(patches_list) == 3
    assert patches_list[0].shape == (2, 512, 32, 3)
    assert patches_list[1].shape == (2, 256, 8, 3)
    assert patches_list[2].shape == (2, 64, 8, 3)
    assert centers_list[0].shape == (2, 512, 3)
