import torch
import pytest
from gmap.models.paranet import ParaNet

def test_paranet_forward():
    model = ParaNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6)
    xyz = torch.randn(2, 8192, 3)
    seg_pred = torch.randint(0, 6, (2, 8192))
    out = model(xyz, seg_pred)
    assert out["joint_type_logits"].shape[0] == 2
    assert out["axis_direction"].shape[-1] == 3
    assert out["axis_position"].shape[-1] == 3
    assert out["joint_state"].shape[-1] == 1

def test_paranet_loss():
    model = ParaNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6)
    xyz = torch.randn(2, 8192, 3)
    seg_pred = torch.randint(0, 6, (2, 8192))
    targets = {"joint_type": torch.tensor([0, 1]), "axis_direction": torch.randn(2, 3), "axis_position": torch.randn(2, 3), "joint_state": torch.randn(2)}
    loss = model.compute_loss(xyz, seg_pred, targets)
    assert loss["loss"].requires_grad
