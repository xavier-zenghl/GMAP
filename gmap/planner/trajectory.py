import numpy as np
from scipy.spatial.transform import Rotation

def compute_revolute_trajectory(contact_point, axis_direction, axis_position, target_angle, n_steps=20):
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    angles = np.linspace(0, target_angle, n_steps)
    trajectory = []
    relative = contact_point - axis_position
    for angle in angles:
        rot = Rotation.from_rotvec(axis_direction * angle)
        rotated = rot.apply(relative) + axis_position
        trajectory.append(rotated)
    return np.array(trajectory)

def compute_prismatic_trajectory(contact_point, axis_direction, target_distance, n_steps=20):
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    distances = np.linspace(0, target_distance, n_steps)
    trajectory = contact_point + np.outer(distances, axis_direction)
    return trajectory
