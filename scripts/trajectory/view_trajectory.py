"""Watch Part 6 motor-driven tracking in the native MuJoCo viewer.

Run with mjpython on macOS, or python on Windows/Linux. The viewer observes
the same simulation used by check_trajectory.py; it never sets joint poses.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from planner import example_trajectory, load_settings, positive, validate
from scara_kinematics import forward_kinematics
from simulation import simulate


class LiveViewer:
    def __init__(self, trajectory, speed=1):
        self.speed = positive(speed, "playback speed")
        self.handle = None
        self.started = None
        self.next_frame = 0.0
        self.path = np.array([forward_kinematics(trajectory.evaluate(t)[0])[:3]
                              for t in np.linspace(0, trajectory.duration, 180)])

    def __call__(self, model, data, desired_q):
        if self.handle is None:
            self.handle = mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False)
            with self.handle.lock():
                camera = self.handle.cam
                camera.lookat[:] = [0.16, 0.0, 0.16]
                camera.distance = 0.85
                camera.azimuth = 135
                camera.elevation = -25
            self.started = time.perf_counter()
        if not self.handle.is_running():
            return False
        if data.time + 1e-10 < self.next_frame:
            return True
        delay = self.started + float(data.time)/self.speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        with self.handle.lock():
            scene = self.handle.user_scn
            scene.ngeom = 0
            for start, end in zip(self.path[:-1], self.path[1:]):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                   np.array([0.1, 0.75, 0.85, 0.6], dtype=np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     0.001, start, end)
                scene.ngeom += 1
            target = forward_kinematics(desired_q)[:3]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, 0.006), target, np.eye(3).ravel(),
                               np.array([1.0, 0.5, 0.1, 0.8], dtype=np.float32))
            scene.ngeom += 1
        self.handle.sync()
        self.next_frame = float(data.time) + self.speed/60
        return self.handle.is_running()

    def close(self):
        if self.handle is not None:
            self.handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("computed_torque", "pd_bias"),
                        default="computed_torque")
    parser.add_argument("--speed", type=float, default=1,
                        help="Playback speed; 0.5 shows the same physics at half speed")
    parser.add_argument("--benchmark-cycles", type=int,
                        help="Repeat the 0.5 s unloaded benchmark this many times")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not 0.05 <= args.speed <= 5:
        parser.error("--speed must be between 0.05 and 5")
    if args.benchmark_cycles is None:
        limits, settings = load_settings()
        trajectory = example_trajectory(limits, settings)
        validate(trajectory, limits)
        path = trajectory
        print(f"Part 6: {trajectory.duration:.3f} s route; {args.controller} motor control.")
    else:
        if args.benchmark_cycles < 1:
            parser.error("--benchmark-cycles must be positive")
        from benchmark_cycle import load_benchmark
        trajectory, limits, settings = load_benchmark(args.benchmark_cycles)
        path = trajectory.cycle
        print(f"Benchmark: {args.benchmark_cycles} round trips at {60/path.duration:.0f} cycles/min "
              f"in simulation; {args.controller} motor control.")
    print("Cyan: planned tool path. Orange: current target. Close the window to stop.")
    viewer = LiveViewer(path, args.speed)
    try:
        _, metrics = simulate(trajectory, settings, controller=args.controller,
                              limits=limits, on_step=viewer, check_clearance=True)
    finally:
        viewer.close()
    status = "Completed" if metrics["completed"] else "Stopped early"
    print(f"{status}: {metrics['simulated_duration_s']:.3f} s simulated; "
          f"peak tool error {1000*metrics['max_tool_position_error_m']:.4f} mm; "
          f"{metrics['saturated_samples']} saturated samples.")
    part_a, part_b = metrics["closest_clearance_pair"]
    print(f"Closest sampled clearance: {1000*metrics['minimum_clearance_m']:.2f} mm "
          f"between {part_a} and {part_b} at {metrics['closest_clearance_time_s']:.3f} s.")


if __name__ == "__main__":
    main()
