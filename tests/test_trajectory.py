import numpy as np
import pytest
from gmap.planner.trajectory import compute_revolute_trajectory, compute_prismatic_trajectory

def test_revolute_trajectory():
    traj = compute_revolute_trajectory(
        contact_point=np.array([1.0, 0, 0]), axis_direction=np.array([0, 0, 1.0]),
        axis_position=np.array([0, 0, 0]), target_angle=np.pi / 2, n_steps=10,
    )
    assert traj.shape == (10, 3)
    assert np.allclose(traj[-1], [0, 1, 0], atol=0.2)

def test_prismatic_trajectory():
    traj = compute_prismatic_trajectory(
        contact_point=np.array([0, 0, 0]), axis_direction=np.array([1.0, 0, 0]),
        target_distance=0.5, n_steps=10,
    )
    assert traj.shape == (10, 3)
    assert np.allclose(traj[-1], [0.5, 0, 0], atol=0.1)
