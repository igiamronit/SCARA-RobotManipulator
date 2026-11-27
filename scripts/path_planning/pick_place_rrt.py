"""Post-final-presentation extension: obstacle-aware RRT-Connect task routes.

Run: python scripts/path_planning/pick_place_rrt.py --view
The original pick_place.py and main-branch scene remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / name) for name in ("task", "trajectory")]
from pick_place import (CLOSED, GRASP_OFFSET, Scene,
                        TASK_ACCELERATION_LIMIT, load_scene, make_plan, preflight, simulate)
from planner import Limits, Trajectory, load_settings, plan_segment, validate
from simulation import check_motor_layout
from rrt import RRTConnect


EDGE_RESOLUTION = np.array([np.deg2rad(.5), np.deg2rad(.5), .001, np.deg2rad(.5)])
SEARCH_CLEARANCE = .006  # Reserve margin for physical tracking and payload slip.


class CarryCollision:
    """Evaluate robot, fixtures and payload for a free-motion task phase.

    Loaded phases predict an ideal rigid grasp; unloaded phases leave the
    block on its stand. The executed payload is a free body held by friction.
    This kinematic prediction never attaches it in the physics simulation.
    """
    def __init__(self, model, phase, clearance=SEARCH_CLEARANCE):
        self.model, self.phase = model, phase
        self.scene = Scene(model)
        self.data = mujoco.MjData(model)
        self.clearance = clearance
        self.last_problem = None
        finger = self.scene.finger_boxes[0]
        body = int(model.geom_bodyid[finger])
        touching = model.geom_size[self.scene.payload, 1] - (
            abs(model.body_pos[body, 1] + model.geom_pos[finger, 1]) - model.geom_size[finger, 1])
        self.data.qpos[4:6] = max(CLOSED, touching) if phase.carrying else phase.grip[0]
        if not phase.carrying and phase.support == "place_stand":
            place = model.site("place_target")
            self.data.qpos[self.scene.payload_qpos:self.scene.payload_qpos+7] = np.r_[place.pos, place.quat]

    def place(self, q):
        self.data.qpos[:4] = q
        mujoco.mj_kinematics(self.model, self.data)
        if self.phase.carrying:
            pose = self.data.qpos[self.scene.payload_qpos:self.scene.payload_qpos+7]
            pose[:3] = self.data.site_xpos[self.scene.tool] - [0, 0, GRASP_OFFSET]
            mujoco.mju_mat2Quat(pose[3:], self.data.site_xmat[self.scene.tool])
        mujoco.mj_kinematics(self.model, self.data)

    def __call__(self, q):
        self.place(q)
        self.last_problem = self.scene.check_clearance(self.data, self.phase, self.clearance, 0)
        return self.last_problem is None


def make_rrt_plan(model, *, seed=0, transfer_height=.090, max_iterations=3000,
                  shortcut_attempts=80):
    trajectory, phases = make_plan(model, transfer_height=transfer_height)
    base, settings = load_settings()
    lower, upper = base.lower.copy(), base.upper.copy()
    # Keep the demonstrated elbow-down branch away from the straight-arm singularity.
    upper[1] = -np.arcsin(.12)
    limits = Limits(lower, upper, base.velocity/settings["velocity_scale"], TASK_ACCELERATION_LIMIT)
    replacements, plans = [], []
    for phase_index, phase in enumerate(phases):
        if phase.name not in ("approach", "transfer", "return"):
            replacements.append(phase)
            continue
        oracle = CarryCollision(model, phase)
        planner = RRTConnect(lower, upper, oracle, EDGE_RESOLUTION, seed=seed+phase_index,
                             max_iterations=max_iterations, shortcut_attempts=shortcut_attempts)
        try:
            planned = planner.plan(phase.segment.q0, phase.segment.qf)
        except ValueError as exc:
            raise ValueError(f"{phase.name}: {exc} {oracle.last_problem or ''}") from exc
        plans.append((phase.name, planned))
        if planned.direct_path_valid:
            replacements.append(phase)
        else:
            for i, (a, b) in enumerate(zip(planned.waypoints[:-1], planned.waypoints[1:])):
                segment = plan_segment(a, b, limits, margin=1.2, minimum_duration=.52,
                                       quantum=.001, profile="quintic")
                replacements.append(replace(phase, name=f"rrt_{phase.name}_{i+1}", segment=segment))
    phases = tuple(replacements)
    result = next(result for name, result in plans if name == "transfer")
    trajectory = Trajectory(p.segment for p in phases)
    kinematic_checks = validate(trajectory, limits)
    # Independent time-domain sampling also checks all unchanged task phases.
    reference_clearance = preflight(Scene(model), trajectory, phases)
    details = {
        "addition": "Implemented after feedback from the final presentation",
        "algorithm": "bidirectional RRT-Connect with collision-checked shortcuts",
        "seed": seed, "requested_transfer_height_m": transfer_height,
        "direct_transfer_valid": result.direct_path_valid,
        "iterations": sum(r.iterations for _, r in plans), "nodes": sum(r.nodes for _, r in plans),
        "state_checks": sum(r.state_checks for _, r in plans),
        "planning_time_s": sum(r.planning_time_s for _, r in plans),
        "search_clearance_m": SEARCH_CLEARANCE,
        "free_motion_plans": [{"phase": name, "direct_valid": r.direct_path_valid,
                               "waypoints": r.waypoints.tolist(), "iterations": r.iterations}
                              for name, r in plans],
        "raw_waypoints": result.raw_waypoints.tolist(), "waypoints": result.waypoints.tolist(),
        "edge_resolution_SI": EDGE_RESOLUTION.tolist(),
        "kinematic_checks": kinematic_checks, "reference_clearance": reference_clearance,
        "collision_validation": "sampled joint edges, 5 ms reference, 1 ms execution; not continuous certification",
    }
    return trajectory, phases, details


def run(seed=0, *, view=False, speed=1, transfer_height=.090, obstacle_height=.074,
        obstacle_x=.296, on_step=None):
    if (not np.isfinite([transfer_height, obstacle_height, obstacle_x, speed]).all() or
            obstacle_height <= 0 or speed <= 0):
        raise ValueError("Finite scene dimensions and positive obstacle height/playback speed are required.")
    # Recompile changed static geometry so collision bounds match its dimensions
    # and location, rather than editing only the run-time geom arrays.
    spec = mujoco.MjSpec.from_file(str(ROOT/"models/scara_pick_place.xml"))
    obstacle = spec.geom("obstacle")
    obstacle.size[2] = obstacle_height/2
    obstacle.pos[:] = [obstacle_x, .002, obstacle_height/2]
    model = spec.compile()
    check_motor_layout(model)
    trajectory, phases, planning = make_rrt_plan(model, seed=seed, transfer_height=transfer_height)
    viewer = None
    peaks = np.zeros(6)

    def observe(m, d, q):
        nonlocal peaks
        peaks = np.maximum(peaks, np.abs(d.qvel[:6]))
        if on_step is not None and on_step(m, d, q) is False:
            return False
        return viewer(m, d, q) if viewer is not None else True

    try:
        if view:
            from view_trajectory import LiveViewer
            viewer = LiveViewer(trajectory, speed)
        report = simulate(model, trajectory, phases, on_step=observe)
    finally:
        if viewer is not None:
            viewer.close()
    report["planning"] = planning
    report["scene"] = {"obstacle_height_m": obstacle_height, "obstacle_x_m": obstacle_x}
    report["peak_joint_speed_SI"] = peaks.tolist()
    report["actual_speed_limits_respected"] = bool(np.all(peaks <= [6, 8, .8, 12, .25, .25]))
    if not report["actual_speed_limits_respected"]:
        report["success"], report["failure"] = False, "An executed joint exceeded its speed limit."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--transfer-height", type=float, default=.090)
    parser.add_argument("--obstacle-height", type=float, default=.074)
    parser.add_argument("--obstacle-x", type=float, default=.296)
    parser.add_argument("--output", type=Path, default=ROOT/"output/path_planning/rrt_result.json")
    args = parser.parse_args()
    try:
        report = run(args.seed, view=args.view, speed=args.speed,
                     transfer_height=args.transfer_height, obstacle_height=args.obstacle_height,
                     obstacle_x=args.obstacle_x)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Planning/execution failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
