import torch
import numpy as np

def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    """归一化点云到单位球。pc: (N, 3)"""
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / max_dist
    return pc

def random_sample_points(pc: np.ndarray, n_points: int) -> np.ndarray:
    """随机采样固定数量点。"""
    n = pc.shape[0]
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.random.choice(n, n_points, replace=True)
    return pc[idx]

def fps_torch(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """纯 PyTorch FPS 备用实现 (无需 CUDA 编译)。xyz: (B, N, 3) -> (B, n_points)"""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid_xyz = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    return centroids

def knn_torch(xyz: torch.Tensor, center_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """纯 PyTorch KNN。xyz: (B,N,3), center_xyz: (B,M,3) -> (B,M,K) indices"""
    dist = torch.cdist(center_xyz, xyz)  # (B, M, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, M, K)
    return idx
