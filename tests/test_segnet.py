import torch
import pytest
from gmap.models.segnet import SegNet

def test_segnet_forward():
    model = SegNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6)
    xyz = torch.randn(2, 8192, 3)
    out = model(xyz)
    assert out["seg_logits"].shape == (2, 8192, 6)
    assert out["mov_logits"].shape == (2, 8192, 2)

def test_segnet_loss():
    model = SegNet(n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6)
    xyz = torch.randn(2, 8192, 3)
    seg_label = torch.randint(0, 6, (2, 8192))
    mov_label = torch.randint(0, 2, (2, 8192))
    loss = model.compute_loss(xyz, seg_label, mov_label)
    assert "loss" in loss
    assert loss["loss"].requires_grad
