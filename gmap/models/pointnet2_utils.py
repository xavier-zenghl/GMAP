import torch
import torch.nn as nn

try:
    from pointnet2_ops.pointnet2_utils import (
        furthest_point_sample as _cuda_fps,
        ball_query as _cuda_ball_query,
    )
    HAS_CUDA_OPS = True
except ImportError:
    HAS_CUDA_OPS = False

from gmap.utils.pc_utils import fps_torch, knn_torch


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """FPS 采样。xyz: (B, N, 3) -> (B, n_points) indices"""
    if HAS_CUDA_OPS and xyz.is_cuda:
        return _cuda_fps(xyz.contiguous(), n_points).long()
    return fps_torch(xyz, n_points)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """根据索引取点。points: (B,N,C), idx: (B,M) or (B,M,K) -> (B,M,C) or (B,M,K,C)"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    expand_shape = list(idx.shape)
    batch_idx = torch.arange(B, device=points.device).view(view_shape).expand(expand_shape)
    return points[batch_idx, idx, :]


def knn_query(xyz: torch.Tensor, center_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """KNN 查询。xyz: (B,N,3), center_xyz: (B,M,3) -> (B,M,K)"""
    return knn_torch(xyz, center_xyz, k)


def group_points(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引分组。xyz: (B,N,C), idx: (B,M,K) -> (B,M,K,C)"""
    return index_points(xyz, idx)


class MultiScaleGrouping(nn.Module):
    """多尺度 FPS+KNN 分组，产生三个尺度的 patch。"""

    def __init__(self, n_points: int, scales: list[tuple[int, int]]):
        super().__init__()
        self.n_points = n_points
        self.scales = scales

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            patches_list: list of (B, M_i, K_i, 3) 局部坐标 patch
            centers_list: list of (B, M_i, 3) 中心点坐标
        """
        patches_list = []
        centers_list = []
        for n_centers, k in self.scales:
            fps_idx = farthest_point_sample(xyz, n_centers)
            centers = index_points(xyz, fps_idx)
            knn_idx = knn_query(xyz, centers, k)
            grouped = group_points(xyz, knn_idx)
            grouped = grouped - centers.unsqueeze(2)
            patches_list.append(grouped)
            centers_list.append(centers)
        return patches_list, centers_list
