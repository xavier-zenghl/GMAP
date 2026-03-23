import torch
import pytest
from gmap.models.pfe import PFE

def test_pfe_output_shape():
    pfe = PFE(
        embed_dim=384,
        n_points=8192,
        scale_centers=[512, 256, 64],
    )
    features = [
        torch.randn(2, 512, 384),
        torch.randn(2, 256, 384),
        torch.randn(2, 64, 384),
    ]
    centers = [
        torch.randn(2, 512, 3),
        torch.randn(2, 256, 3),
        torch.randn(2, 64, 3),
    ]
    xyz = torch.randn(2, 8192, 3)

    point_features = pfe(features, centers, xyz)
    assert point_features.shape == (2, 8192, 384)
