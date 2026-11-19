"""Meaningful numerical, boundary, limit and physics regressions for part 6."""
from __future__ import annotations

import argparse
import unittest

import numpy as np

from planner import (HOME, WORKED_GOAL, Segment, Trajectory, continuous_ik,
                     example_trajectory, load_settings, plan_segment, plan_waypoints,
                     validate)
from scara_kinematics import forward_kinematics


class TrajectoryChecks(unittest.TestCase):
    def setUp(self):
        self.limits, self.settings = load_settings()

    def test_derivatives_from_independent_position_differences(self):
        for profile in ("cubic", "quintic", "lspb"):
            segment = Segment(HOME, WORKED_GOAL, 2, profile)
            for time in (0.21, 0.91, 1.71):
                eps = 2e-4
                q, v, a = segment.evaluate(time)
                plus = segment.evaluate(time+eps)[0]
                minus = segment.evaluate(time-eps)[0]
                np.testing.assert_allclose(v, (plus-minus)/(2*eps), atol=2e-8)
                np.testing.assert_allclose(a, (plus-2*q+minus)/eps**2, atol=2e-7)

    def test_endpoints_reverse_stationary_and_holds(self):
        for profile in ("cubic", "quintic", "lspb"):
            for start, goal in ((HOME, WORKED_GOAL), (WORKED_GOAL, HOME), (HOME, HOME)):
                segment = plan_segment(start, goal, self.limits, profile=profile)
                for time, expected in ((0, start), (segment.duration, goal)):
                    q, v, a = segment.evaluate(time)
                    np.testing.assert_allclose(q, expected, atol=1e-12)
                    np.testing.assert_allclose(v, 0, atol=1e-12)
                    if profile == "quintic":
                        np.testing.assert_allclose(a, 0, atol=1e-12)
                for time, expected in ((-1, start), (segment.duration+1, goal)):
                    q, v, a = segment.evaluate(time)
                    np.testing.assert_allclose(q, expected)
                    np.testing.assert_allclose(v, 0)
                    np.testing.assert_allclose(a, 0)

    def test_dense_samples_respect_limits_including_lspb_triangle(self):
        for profile, blend in (("cubic", 1/3), ("quintic", 1/3), ("lspb", 1/3), ("lspb", 0.5)):
            segment = plan_segment(HOME, WORKED_GOAL, self.limits, profile=profile,
                                   blend_fraction=blend, margin=1)
            sample = np.array([segment.evaluate(t) for t in np.linspace(0, segment.duration, 5001)])
            self.assertTrue(np.all(np.abs(sample[:, 1]) <= self.limits.velocity+1e-10))
            self.assertTrue(np.all(np.abs(sample[:, 2]) <= self.limits.acceleration+1e-10))
            peak_v, peak_a = segment.peak_rates()
            np.testing.assert_allclose(np.max(np.abs(sample[:, 1]), axis=0), peak_v, rtol=1e-6)
            np.testing.assert_allclose(np.max(np.abs(sample[:, 2]), axis=0), peak_a, rtol=1e-6)

    def test_input_rejections(self):
        for duration in (0, -1, np.nan, np.inf, .001):
            with self.assertRaises(ValueError):
                plan_segment(HOME, WORKED_GOAL, self.limits, duration=duration)
        for point in ([0, 0, .5, 0], [0, 0, np.nan, 0], [1, 2]):
            with self.assertRaises(ValueError):
                plan_segment(HOME, point, self.limits)
        for blend in (0, -.1, .51, np.nan):
            with self.assertRaises(ValueError):
                Segment(HOME, WORKED_GOAL, 1, "lspb", blend)
        with self.assertRaises(ValueError):
            plan_waypoints([HOME, WORKED_GOAL], self.limits, holds=[-1, 0])
        with self.assertRaises(ValueError):
            plan_segment(HOME, WORKED_GOAL, self.limits, margin=.5)
        with self.assertRaises(ValueError):
            plan_segment(HOME, WORKED_GOAL, self.limits, profile="unknown")
        with self.assertRaises(ValueError):
            Segment(HOME, HOME, 1).evaluate(np.nan)
        with self.assertRaises(ValueError):
            validate(Trajectory([Segment(HOME, WORKED_GOAL, .001)]), self.limits)

    def test_joint_angles_do_not_cross_limited_joint_wrap_boundary(self):
        start, goal = HOME.copy(), HOME.copy()
        start[3], goal[3] = np.deg2rad(170), np.deg2rad(-170)
        segment = plan_segment(start, goal, self.limits)
        # The valid interval route goes through zero; +20 degrees crosses the stop.
        self.assertAlmostEqual(segment.evaluate(segment.duration/2)[0][3], 0)

    def test_waypoint_joins_and_ik_branch(self):
        trajectory = example_trajectory(self.limits, self.settings)
        validate(trajectory, self.limits)
        for left, right in zip(trajectory.segments, trajectory.segments[1:]):
            for a, b in zip(left.evaluate(left.duration), right.evaluate(0)):
                np.testing.assert_allclose(a, b, atol=1e-10)
        targets = [[.28, -.1, .10, 0], [.28, -.1, .04, 0], [.26, .14, .10, .35]]
        joints = continuous_ik(targets, HOME, self.limits)
        for q, target in zip(joints, targets):
            np.testing.assert_allclose(forward_kinematics(q), target, atol=1e-10)
            self.assertLess(q[1], 0)
        with self.assertRaises(ValueError):
            continuous_ik([[.5, 0, .1, 0]], HOME, self.limits)

    def test_immutable_endpoints_and_disconnected_path_rejection(self):
        start = HOME.copy()
        segment = Segment(start, WORKED_GOAL, 1)
        start[:] = 0
        np.testing.assert_allclose(segment.q0, HOME)
        with self.assertRaises(ValueError):
            Trajectory([segment, Segment(HOME, HOME, 1)])


def physics_checks():
    from simulation import simulate
    limits, settings = load_settings()
    trajectory = example_trajectory(limits, settings)
    cases = [("computed_torque", None), ("pd_bias", None),
             ("computed_torque", [np.deg2rad(2), np.deg2rad(-2), .002, np.deg2rad(2)])]
    for controller, offset in cases:
        logs, metrics = simulate(trajectory, settings, controller=controller,
                                 initial_offset=offset, limits=limits, check_clearance=True)
        assert metrics["completed"], metrics
        assert metrics["saturated_samples"] == 0, metrics
        assert metrics["model_position_limits_respected"], metrics
        assert metrics["final_tool_position_error_m"] < 1e-5, metrics
        assert np.all(np.abs(metrics["final_joint_error_SI"]) < 1e-4), metrics
        assert metrics["maximum_active_contacts"] == 0, metrics
        assert metrics["minimum_clearance_m"] > 0, metrics
        assert all(np.isfinite(values).all() for values in logs.values()), metrics
        if offset is None:
            tolerance = .001 if controller == "computed_torque" else .01
            assert metrics["max_tool_position_error_m"] < tolerance, metrics
            assert metrics["actual_velocity_limits_respected"], metrics
            assert metrics["actual_acceleration_limits_respected"], metrics
        print(f"PASS: {controller} physics tracking ({'nominal' if offset is None else 'perturbed start'}); "
              f"max tool error={metrics['max_tool_position_error_m']:.6g} m; "
              f"final={metrics['final_tool_position_error_m']:.6g} m; no saturation.")

    # A closed viewer must interrupt execution without reporting a completed run.
    samples = []
    def stop_early(model, data, desired_q):
        samples.append(float(data.time))
        return len(samples) < 11
    logs, metrics = simulate(trajectory, settings, limits=limits, on_step=stop_early)
    assert not metrics["completed"] and len(logs["time"]) == 11, metrics
    np.testing.assert_allclose(samples, np.arange(11)*.001, atol=1e-12)
    print("PASS: viewer observation and early-stop completion reporting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrajectoryChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.physics:
        physics_checks()
