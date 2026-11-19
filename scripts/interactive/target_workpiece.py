"""Persistent workpiece with geometric pickup/release for both viewers.

While held, preserve its transform relative to the tool. On release it stays
at that pose. This is kinematic attachment; no contact forces or gravity.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


class TargetWorkpiece:
    START = np.array([.30, .02, .030, 0.])

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.mocap_id = int(model.body("connector_target").mocapid[0])
        self.status_id = model.geom("connector_status").id
        if self.mocap_id < 0:
            raise ValueError("The connector target must be a mocap body.")
        self.reset()

    def reset(self):
        self.pose = self.START.copy()
        self.held = False
        self.offset = np.zeros(3)
        self.yaw_offset = 0.
        self.finger_positions = np.zeros(2)
        self.update()

    @staticmethod
    def rotation(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def try_grasp(self, tool_pose):
        offset = self.rotation(tool_pose[3]).T @ (self.pose[:3] - tool_pose[:3])
        yaw = math.atan2(math.sin(self.pose[3]-tool_pose[3]), math.cos(self.pose[3]-tool_pose[3]))
        # Parallel jaws can approach the housing at either 0 or 180 degrees.
        if (np.linalg.norm(offset[:2]) > .002 or abs(offset[2]) > .004
                or abs(math.sin(yaw)) > math.sin(math.radians(5))):
            return False
        self.offset = offset
        self.yaw_offset = yaw
        self.held = True
        # Stop each finger at the housing instead of closing through it.
        points = []
        for centre, half in [([0,0,0], [.016,.022,.010]), ([.019,0,.004], [.003,.005,.003])]:
            points.extend(np.array(centre) + np.array([x,y,z])*half
                          for x in (-1,1) for y in (-1,1) for z in (-1,1))
        bounds = np.asarray(points) @ self.rotation(yaw).T + offset
        self.finger_positions = np.maximum(0, [bounds[:,1].max()-.022, -bounds[:,1].min()-.022])
        self.update()
        return True

    def release(self):
        self.held = False
        self.update()

    def follow(self, tool_pose):
        if self.held:
            self.pose[:3] = tool_pose[:3] + self.rotation(tool_pose[3]) @ self.offset
            angle = tool_pose[3] + self.yaw_offset
            self.pose[3] = math.atan2(math.sin(angle), math.cos(angle))
        self.update()

    def update(self):
        x, y, z, yaw = self.pose
        self.data.mocap_pos[self.mocap_id] = [x, y, z]
        self.data.mocap_quat[self.mocap_id] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.model.geom_rgba[self.status_id] = (
            [.20, .55, .90, 1] if self.held else [.15, .65, .30, 1])
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def yaw_arrow(target):
        """Lift the yaw cue above the housing so it cannot hide inside it."""
        start = np.asarray(target[:3], dtype=float) + [0, 0, .016]
        end = start + .050 * np.array([math.cos(target[3]), math.sin(target[3]), 0])
        return start, end
