"""Checks for the eye-in-hand IBVS: each piece against MuJoCo ground truth, then the task.

    python scripts/task/check_ibvs.py
"""
import unittest

import mujoco
import numpy as np

from ibvs import (CAMERA, HAND_EYE_ANGLE, HAND_EYE_OFFSET, LOOK_POSE, OPEN, WristCamera,
                  camera_to_joint_rates, intrinsics, interaction_matrix, load_goal, match_order,
                  observe, rot_z, run, shift_block, teach)
from pick_place import load_scene
from planner import HOME, continuous_ik, load_settings
from scara_kinematics import forward_kinematics, forward_transform

OPENCV = np.diag([1, -1, -1])   # MuJoCo camera axes -> lecture (OpenCV) camera axes


class IbvsChecks(unittest.TestCase):
    def setUp(self):
        self.model = load_scene()
        self.data = mujoco.MjData(self.model)
        self.camera = WristCamera(self.model)
        limits, _ = load_settings()
        self.look_q = continuous_ik([LOOK_POSE], HOME, limits)[0]
        self.pose(self.look_q)

    def tearDown(self):
        self.camera.close()

    def pose(self, q):
        self.data.qpos[:4], self.data.qpos[4:6] = q, OPEN
        mujoco.mj_forward(self.model, self.data)

    def test_hand_eye_transform_matches_the_model(self):
        """T4c used by the controller equals the camera pose in models/scara.xml."""
        cam = self.model.camera(CAMERA).id
        R04 = forward_transform(self.look_q)[:3, :3]
        tool = forward_kinematics(self.look_q)[:3]
        model_rotation = self.data.cam_xmat[cam].reshape(3, 3) @ OPENCV
        np.testing.assert_allclose(R04 @ rot_z(HAND_EYE_ANGLE), model_rotation, atol=1e-9)
        np.testing.assert_allclose(tool + R04 @ HAND_EYE_OFFSET, self.data.cam_xpos[cam], atol=1e-9)

    def test_corners_match_the_projected_block(self):
        """Extracted corners agree with the true top corners projected by u = lam x/z + cx."""
        lam, cx, cy = intrinsics(self.model)
        cam = self.model.camera(CAMERA).id
        block = self.model.body("payload").id
        half = self.model.geom("workpiece").size
        top = [self.data.xpos[block] + self.data.xmat[block].reshape(3, 3) @ [i*half[0], j*half[1], half[2]]
               for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        R = self.data.cam_xmat[cam].reshape(3, 3) @ OPENCV
        x, y, z = ((np.array(top) - self.data.cam_xpos[cam]) @ R).T
        truth = np.c_[lam*x/z + cx, lam*y/z + cy]
        s, depth, _ = observe(self.camera, self.data)
        error = max(np.min(np.linalg.norm(s - p, axis=1)) for p in truth)
        self.assertLess(error, 1.0)
        self.assertAlmostEqual(depth, z[0], places=4)

    def test_interaction_matrix_predicts_image_motion(self):
        """L xi_c matches the measured feature velocity for each camera motion."""
        lam, cx, cy = intrinsics(self.model)
        s0, z, _ = observe(self.camera, self.data)
        L = interaction_matrix(s0, z, lam, cx, cy)
        # Rendered Harris corners need a multi-pixel perturbation to avoid
        # amplification of sub-pixel localization noise by differentiation.
        h = .03
        for xi in np.diag([.1, .1, .1, 1.0]):
            qd = camera_to_joint_rates(xi, self.look_q)
            images = []
            for sign in (1, -1):
                self.pose(self.look_q + sign*h*qd)
                images.append(match_order(observe(self.camera, self.data)[0], s0))
            measured = (images[0] - images[1]).ravel() / (2*h)
            predicted = L @ xi
            self.assertLess(np.linalg.norm(measured - predicted), .03*np.linalg.norm(predicted), xi)
        self.pose(self.look_q)

    def test_goal_file_is_up_to_date(self):
        np.testing.assert_allclose(teach(self.model)["s_star_px"], load_goal()["s_star_px"], atol=.01)

    def test_servo_then_pick_and_place(self):
        report, log = run(self.model, load_goal())
        self.assertTrue(report["success"], report["failure"])
        self.assertLess(log["rms_px"][-1], 1.0)
        self.assertLess(report["servo"]["grasp_xy_error_mm"], 1.0)
        self.assertLess(report["placement_error_m"], .001)

    def test_shifted_block_is_still_picked(self):
        shift_block(self.model, .008, -.006, .15)
        report, _ = run(self.model, load_goal())
        self.assertTrue(report["success"], report["failure"])
        self.assertLess(report["servo"]["grasp_xy_error_mm"], 1.0)

    def test_camera_mounting_error_beyond_90_degrees_diverges(self):
        """Lec 37 slide 12: converges only if cos(alpha) > 0."""
        report, _ = run(self.model, load_goal(), hand_eye_angle=HAND_EYE_ANGLE + np.radians(100))
        self.assertFalse(report["servo"]["converged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
