import torch
import pytest
from gmap.models.affordnet import AffordNet

def test_affordnet_forward():
    model = AffordNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, top_k=64)
    xyz = torch.randn(2, 8192, 3)
    out = model(xyz)
    assert out["affordance_scores"].shape == (2, 8192)
    assert out["best_point"].shape == (2, 3)
    assert out["best_direction"].shape == (2, 3)

def test_affordnet_loss():
    model = AffordNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, top_k=64)
    xyz = torch.randn(2, 8192, 3)
    target_scores = torch.rand(2, 8192)
    loss = model.compute_loss(xyz, target_scores)
    assert loss["loss"].requires_grad
