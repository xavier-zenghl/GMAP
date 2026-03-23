"""Panda robot controller."""
import numpy as np

class PandaController:
    def __init__(self, robot):
        self.robot = robot
        self.ee_link_name = "panda_hand"

    def move_to_pose(self, target_pos, target_quat, scene, n_steps=100):
        for step in range(n_steps):
            scene.step()

    def follow_trajectory(self, trajectory, scene, steps_per_waypoint=50):
        for waypoint in trajectory:
            self.move_to_pose(waypoint, np.array([1, 0, 0, 0]), scene, steps_per_waypoint)

    def close_gripper(self):
        qpos = self.robot.get_qpos()
        qpos[-2:] = 0.0
        self.robot.set_qpos(qpos)

    def open_gripper(self):
        qpos = self.robot.get_qpos()
        qpos[-2:] = 0.04
        self.robot.set_qpos(qpos)
