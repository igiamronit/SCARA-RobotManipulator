"""Meaningful path-planning regressions, including a complete friction grasp."""
import unittest

import numpy as np

from rrt import PlanningFailure, RRTConnect
from pick_place_rrt import CarryCollision, EDGE_RESOLUTION, make_rrt_plan, run
from pick_place import Scene, load_scene, make_plan, preflight
from planner import load_settings


class PathPlanningChecks(unittest.TestCase):
    def test_edge_checks_interior_not_only_endpoints(self):
        p = RRTConnect([0, 0], [1, 1], lambda q: abs(q[0]-.5) > .1, [.02, .02])
        self.assertTrue(p.state_valid([.1, .5]))
        self.assertTrue(p.state_valid([.9, .5]))
        self.assertFalse(p.edge_valid([.1, .5], [.9, .5]))
        with self.assertRaises(PlanningFailure):
            RRTConnect([0, 0], [1, 1], lambda q: abs(q[0]-.5) > .1,
                       [.02, .02], max_iterations=50).plan([.1, .5], [.9, .5])

    def test_free_direct_and_invalid_endpoints(self):
        p = RRTConnect([-1, -1], [1, 1], lambda q: True, [.02, .02])
        result = p.plan([-.8, .2], [.8, -.2])
        self.assertTrue(result.direct_path_valid)
        np.testing.assert_array_equal(result.waypoints, [[-.8, .2], [.8, -.2]])
        with self.assertRaises(ValueError):
            p.plan([2, 0], [0, 0])
        with self.assertRaises(ValueError):
            p.plan([0, 0], [np.nan, 0])

    def test_detour_and_reproducibility(self):
        def valid(q):
            return np.linalg.norm(q-[.5, .5]) > .2
        results = [RRTConnect([0, 0], [1, 1], valid, [.01, .01], seed=4).plan(
            [.1, .5], [.9, .5]) for _ in range(2)]
        self.assertFalse(results[0].direct_path_valid)
        np.testing.assert_array_equal(results[0].waypoints, results[1].waypoints)
        p = RRTConnect([0, 0], [1, 1], valid, [.005, .005])
        for a, b in zip(results[0].waypoints[:-1], results[0].waypoints[1:]):
            self.assertTrue(p.edge_valid(a, b))

    def test_loaded_scara_direct_fails_rrt_and_dense_recheck_pass(self):
        model = load_scene()
        baseline, phases = make_plan(model, .090)
        with self.assertRaisesRegex(ValueError, "workpiece / obstacle"):
            preflight(Scene(model), baseline, phases)
        trajectory, phases, report = make_rrt_plan(model, seed=0)
        self.assertFalse(report["direct_transfer_valid"])
        limits, _ = load_settings()
        phase = next(p for p in phases if p.carrying and p.name.startswith("rrt"))
        p = RRTConnect(limits.lower, limits.upper, CarryCollision(model, phase), EDGE_RESOLUTION/2)
        for a, b in zip(report["waypoints"][:-1], report["waypoints"][1:]):
            self.assertTrue(p.edge_valid(np.array(a), np.array(b)))
        for left, right in zip(trajectory.segments[:-1], trajectory.segments[1:]):
            for a, b in zip(left.evaluate(left.duration), right.evaluate(0)):
                np.testing.assert_allclose(a, b, atol=1e-10)

    def test_real_grasp_and_full_task(self):
        report = run(0)
        self.assertTrue(report["success"], report["failure"])
        self.assertLess(report["placement_error_m"], .001)
        self.assertEqual(report["saturated_samples"], 0)
        self.assertTrue(report["actual_speed_limits_respected"])
        self.assertGreater(report["measured_clearance"]["payload_obstacle_clearance_m"], .001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
