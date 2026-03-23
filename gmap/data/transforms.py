import numpy as np
from gmap.utils.pc_utils import normalize_point_cloud, random_sample_points

class PointCloudTransforms:
    def __init__(self, n_points: int = 8192, normalize: bool = True, augment: bool = False):
        self.n_points = n_points
        self.normalize = normalize
        self.augment = augment

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        pc = random_sample_points(pc, self.n_points)
        if self.normalize:
            pc = normalize_point_cloud(pc)
        if self.augment:
            pc = self._augment(pc)
        return pc.astype(np.float32)

    def _augment(self, pc: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
        pc = pc @ R.T
        scale = np.random.uniform(0.8, 1.2)
        pc = pc * scale
        pc = pc + np.random.normal(0, 0.02, size=pc.shape)
        return pc
