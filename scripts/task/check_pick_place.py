"""Regression checks for the nominal task and two distinct failure cases."""
import unittest

from pick_place import Scene, load_scene, make_plan, preflight, simulate


class PickPlaceChecks(unittest.TestCase):
    def setUp(self):
        self.model = load_scene()

    def test_friction_grasp_and_placement(self):
        trajectory, phases = make_plan(self.model)
        report = simulate(self.model, trajectory, phases)

        self.assertTrue(report["success"], report["failure"])
        self.assertEqual(list(report["events"]), ["grasp_s", "lift_s", "seat_s", "release_s"])
        self.assertLess(report["events"]["release_s"], report["simulated_duration_s"])
        self.assertLess(report["simulated_duration_s"], 3.0)
        self.assertLess(report["placement_error_m"], .001)
        self.assertLess(report["placement_yaw_error_rad"], .01)
        self.assertGreater(report["measured_clearance"]["payload_obstacle_clearance_m"], .005)
        self.assertEqual(report["saturated_samples"], 0)

    def test_low_transfer_is_rejected_between_waypoints(self):
        trajectory, phases = make_plan(self.model, transfer_height=.090)
        with self.assertRaisesRegex(ValueError, r"transfer at .*workpiece / obstacle"):
            preflight(Scene(self.model), trajectory, phases)

    def test_low_friction_cannot_lift_the_payload(self):
        trajectory, phases = make_plan(self.model)
        self.model.geom("workpiece").friction[0] = .01
        report = simulate(self.model, trajectory, phases)

        self.assertFalse(report["success"])
        self.assertIn("Lost the grasp during lift", report["failure"])
        self.assertIn("grasp_s", report["events"])
        self.assertNotIn("release_s", report["events"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
