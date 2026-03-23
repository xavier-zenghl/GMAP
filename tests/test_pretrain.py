import torch
import pytest
from gmap.models.pretrain import PretrainModel

def test_pretrain_forward():
    model = PretrainModel(
        n_points=8192,
        scales=[(512, 32), (256, 8), (64, 8)],
        embed_dim=384,
        depth=6,
        heads=6,
        codebook_size=8192,
        codebook_dim=256,
        mask_ratio=0.6,
    )
    xyz = torch.randn(2, 8192, 3)
    loss_dict = model(xyz)
    assert "loss" in loss_dict
    assert "loss_recon" in loss_dict
    assert "loss_token" in loss_dict
    assert loss_dict["loss"].requires_grad

def test_pretrain_extract_features():
    model = PretrainModel(
        n_points=8192,
        scales=[(512, 32), (256, 8), (64, 8)],
        embed_dim=384,
        depth=6,
        heads=6,
        codebook_size=8192,
        codebook_dim=256,
        mask_ratio=0.6,
    )
    xyz = torch.randn(2, 8192, 3)
    features, centers = model.extract_features(xyz)
    assert len(features) == 3
    assert features[0].shape == (2, 512, 384)
