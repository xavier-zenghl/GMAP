import torch
import torch.nn as nn

def three_nn_interpolate(
    target_xyz: torch.Tensor,
    source_xyz: torch.Tensor,
    source_feat: torch.Tensor,
) -> torch.Tensor:
    """反距离加权插值 (类似 PointNet++ Feature Propagation)。

    Args:
        target_xyz: (B, N, 3) 目标点坐标
        source_xyz: (B, M, 3) 源点坐标 (M < N)
        source_feat: (B, M, C) 源点特征
    Returns:
        (B, N, C) 插值后的目标点特征
    """
    dist = torch.cdist(target_xyz, source_xyz)  # (B, N, M)
    dist_top3, idx_top3 = dist.topk(3, dim=-1, largest=False)  # (B, N, 3)

    dist_top3 = dist_top3.clamp(min=1e-8)
    weight = 1.0 / dist_top3  # (B, N, 3)
    weight = weight / weight.sum(dim=-1, keepdim=True)

    B, N, _ = target_xyz.shape
    C = source_feat.shape[-1]

    idx_expanded = idx_top3.unsqueeze(-1).expand(B, N, 3, C)
    source_expanded = source_feat.unsqueeze(1).expand(B, N, -1, C)
    gathered = torch.gather(source_expanded, 2, idx_expanded)  # (B, N, 3, C)

    interpolated = (weight.unsqueeze(-1) * gathered).sum(dim=2)  # (B, N, C)
    return interpolated


class PFE(nn.Module):
    """点级特征传播 (Point-level Feature Extraction/Propagation)。
    将 MSFE 三个尺度的特征逐层上采样回原始 N 个点。
    Scale3(64) -> Scale2(256) -> Scale1(512) -> Original(8192)
    """

    def __init__(self, embed_dim: int = 384, n_points: int = 8192, scale_centers: list[int] = [512, 256, 64]):
        super().__init__()
        self.mlp_3to2 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp_2to1 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp_1toN = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        centers: list[torch.Tensor],
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        feat1, feat2, feat3 = features
        ctr1, ctr2, ctr3 = centers

        # Scale3 -> Scale2
        up_3to2 = three_nn_interpolate(ctr2, ctr3, feat3)
        fused_2 = self.mlp_3to2(torch.cat([feat2, up_3to2], dim=-1))

        # Scale2 -> Scale1
        up_2to1 = three_nn_interpolate(ctr1, ctr2, fused_2)
        fused_1 = self.mlp_2to1(torch.cat([feat1, up_2to1], dim=-1))

        # Scale1 -> Original N points
        up_1toN = three_nn_interpolate(xyz, ctr1, fused_1)
        point_feat = self.mlp_1toN(up_1toN)

        return point_feat
