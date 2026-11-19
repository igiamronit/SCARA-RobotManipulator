"""Simulate a friction-grasp pick and place with the SCARA in MuJoCo."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trajectory"))
from planner import HOME, ROOT, Limits, Segment, Trajectory, continuous_ik, load_settings, plan_segment
from planner import positive, validate
from scara_kinematics import forward_kinematics, wrap_to_pi
from simulation import MeshClearance, check_motor_layout, computed_torque


OPEN = .013                  # Outward slide displacement per finger, in m.
CLOSED = .001                # Commanded displacement under load.
GRASP_OFFSET = .006          # Tool height above the workpiece center, in m.
TRANSFER_HEIGHT = .111       # Tool height during horizontal travel, in m.
PLAN_CLEARANCE = .002        # Minimum sampled separation, in m.
RUN_CLEARANCE = .001         # Minimum measured separation during execution, in m.
PLAN_SAMPLE_PERIOD = .005    # Maximum time between reference samples, in s.
CONTACT_TOLERANCE = .002     # Allowed penetration at intended interfaces, in m.

TASK_ACCELERATION_LIMIT = (96, 128, 9.6, 192)  # rad/s², rad/s², m/s², rad/s².
GRIP_RAMP_S = .12
TRANSFER_MIN_S = .52        # Limit payload slip during the lateral transfer.
LOWER_MIN_S = .52           # Reduce the contact speed at the place stand.
SETTLE_S = .10
ARM_FREQUENCY_RAD_S = 50


@dataclass(frozen=True)
class Phase:
    name: str
    segment: Segment
    support: str | None = None
    fingers_may_touch: bool = False
    carrying: bool = False
    grip: tuple[float, float] = (OPEN, OPEN)

    def grip_reference(self, elapsed):
        u = np.clip(elapsed / self.segment.duration, 0, 1)
        start, end = self.grip
        delta = end - start
        return (start + delta*u*u*(3 - 2*u),
                delta*6*u*(1 - u) / self.segment.duration)


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara_pick_place.xml"))
    check_motor_layout(model)
    return model


def site_yaw(site):
    return 2*np.arctan2(site.quat[3], site.quat[0])


def make_plan(model, transfer_height=TRANSFER_HEIGHT, start_q=None, pick_pose=None):
    """Plan the task phases from home to the pick_target site.

    ibvs.py passes start_q (the servoed pose above the block) and pick_pose
    (the tool grasp pose), so the plan starts with the descent instead.
    """
    limits, settings = load_settings()
    robot_velocity = limits.velocity / float(settings["velocity_scale"])
    limits = Limits(limits.lower, limits.upper, robot_velocity, TASK_ACCELERATION_LIMIT)
    pick, place = (model.site(name) for name in ("pick_target", "place_target"))

    def pose(site, height):
        return [*site.pos[:2], height, site_yaw(site)]

    if pick_pose is None:
        pick_pose = pose(pick, pick.pos[2] + GRASP_OFFSET)
    above_pick_pose = [pick_pose[0], pick_pose[1], transfer_height, pick_pose[3]]
    home_pose = forward_kinematics(HOME)
    home_pose[2] = transfer_height
    home, above_pick, at_pick, above_place, at_place = continuous_ik([
        home_pose, above_pick_pose, pick_pose,
        pose(place, transfer_height), pose(place, place.pos[2] + GRASP_OFFSET),
    ], HOME if start_q is None else start_q, limits)

    def move(start, end, minimum_duration=0):
        return plan_segment(start, end, limits, margin=settings["timing_margin"],
                            minimum_duration=max(minimum_duration, settings["minimum_segment_s"]),
                            quantum=settings["time_quantum_s"])

    if start_q is None:
        reach = (Phase("approach", move(home, above_pick), "pick_stand"),
                 Phase("descend", move(above_pick, at_pick), "pick_stand"))
    else:
        reach = (Phase("descend", move(start_q, at_pick), "pick_stand"),)

    phases = (
        *reach,
        Phase("grasp", Segment(at_pick, at_pick, GRIP_RAMP_S), "pick_stand",
              fingers_may_touch=True, grip=(OPEN, CLOSED)),
        Phase("lift", move(at_pick, above_pick), "pick_stand",
              fingers_may_touch=True, carrying=True, grip=(CLOSED, CLOSED)),
        Phase("transfer", move(above_pick, above_place, TRANSFER_MIN_S),
              fingers_may_touch=True, carrying=True, grip=(CLOSED, CLOSED)),
        Phase("lower", move(above_place, at_place, LOWER_MIN_S), "place_stand",
              fingers_may_touch=True, carrying=True, grip=(CLOSED, CLOSED)),
        Phase("release", Segment(at_place, at_place, GRIP_RAMP_S), "place_stand",
              fingers_may_touch=True, grip=(CLOSED, OPEN)),
        Phase("retreat", move(at_place, above_place), "place_stand"),
        Phase("return", move(above_place, home), "place_stand"),
        Phase("settle", Segment(home, home, SETTLE_S), "place_stand"),
    )
    trajectory = Trajectory(phase.segment for phase in phases)
    validate(trajectory, limits)
    return trajectory, phases


class Scene:
    """Track contact policy and sampled separation for robot and workpiece."""

    def __init__(self, model):
        self.model = model
        self.tool = model.site("tool").id
        self.payload = model.geom("workpiece").id
        self.payload_body = model.geom_bodyid[self.payload]
        free_joint = model.joint("payload_free").id
        self.payload_qpos = int(model.jnt_qposadr[free_joint])
        self.payload_dof = int(model.jnt_dofadr[free_joint])
        self.obstacle = model.geom("obstacle").id
        self.fixtures = (model.geom("pick_stand").id, model.geom("place_stand").id, self.obstacle)
        self.floor = model.geom("floor").id
        fingers = {model.body(name).id for name in ("left_finger", "right_finger")}
        self.finger_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in fingers}
        self.finger_boxes = tuple(i for i in sorted(self.finger_geoms)
                                  if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        if len(self.finger_boxes) != 2 or {model.geom_bodyid[i] for i in self.finger_boxes} != fingers:
            raise ValueError("Task requires one collision box on each finger.")
        robot = [i for i in range(model.ngeom)
                 if model.geom_bodyid[i] not in (0, self.payload_body)]
        self.pairs = [(a, b) for a, b, _ in MeshClearance(model).pairs]
        self.pairs.append((model.geom("upper_arm_collision").id,
                           model.geom("forearm_collision").id))
        self.pairs.extend((a, b) for a in robot for b in (*self.fixtures, self.payload))
        self.pairs.extend((self.payload, b) for b in (*self.fixtures, self.floor))
        self.minimum = (None, None, None)
        self.payload_obstacle_minimum = None

    def geom_name(self, i):
        if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            return self.model.mesh(self.model.geom_dataid[i]).name
        return self.model.geom(i).name or f"{self.model.body(self.model.geom_bodyid[i]).name} proxy"

    def check_clearance(self, data, phase, required, time):
        support = self.model.geom(phase.support).id if phase.support else None
        for a, b in self.pairs:
            allowed = ((a == self.payload and b == support) or
                       (phase.fingers_may_touch and a in self.finger_geoms and b == self.payload))
            distance = mujoco.mj_geomDistance(self.model, data, a, b, 1.0, None)
            if a == self.payload and b == self.obstacle:
                best = self.payload_obstacle_minimum
                self.payload_obstacle_minimum = distance if best is None else min(best, distance)
            if not allowed and (self.minimum[0] is None or distance < self.minimum[0]):
                self.minimum = (distance, (self.geom_name(a), self.geom_name(b)), float(time))
            threshold = -CONTACT_TOLERANCE if allowed else required
            if distance < threshold:
                return (f"{phase.name} at {time:.3f} s: {self.geom_name(a)} / {self.geom_name(b)} "
                        f"separation {distance*1000:.2f} mm (minimum {threshold*1000:.2f} mm)")
        return None

    def payload_contacts(self, data):
        forces = {}
        wrench = np.empty(6)
        for index, contact in enumerate(data.contact):
            a, b = contact.geom
            if self.payload in (a, b):
                other = b if a == self.payload else a
                mujoco.mj_contactForce(self.model, data, index, wrench)
                forces[other] = forces.get(other, 0) + float(wrench[0])
        return forces

    def grip_is_loaded(self, contacts):
        return all(contacts.get(finger, 0) > .15 for finger in self.finger_boxes)

    def clearance_report(self):
        distance, pair, time = self.minimum
        return {"minimum_clearance_m": distance, "closest_pair": pair, "time_s": time,
                "payload_obstacle_clearance_m": self.payload_obstacle_minimum}


def preflight(scene, trajectory, phases):
    """Sample reference motion with the workpiece at its nominal phase pose."""
    model = scene.model
    data = mujoco.MjData(model)
    workpiece_pose = data.qpos[scene.payload_qpos:scene.payload_qpos + 7]
    pick_pose = workpiece_pose.copy()
    place = model.site("place_target")
    place_pose = np.r_[place.pos, place.quat]
    # Match the opposed jaw faces to the nominal block width in the geometry test.
    finger = scene.finger_boxes[0]
    jaw_body = int(model.geom_bodyid[finger])
    closed_gap = model.geom_size[scene.payload, 1] - (
        abs(model.body_pos[jaw_body, 1] + model.geom_pos[finger, 1]) - model.geom_size[finger, 1])
    for i, phase in enumerate(phases):
        samples = int(np.ceil(phase.segment.duration / PLAN_SAMPLE_PERIOD)) + 1
        for elapsed in np.linspace(0, phase.segment.duration, samples):
            data.qpos[:4] = phase.segment.evaluate(elapsed)[0]
            grip, _ = phase.grip_reference(elapsed)
            data.qpos[4:6] = max(grip, closed_gap)
            if phase.carrying:
                mujoco.mj_kinematics(model, data)
                workpiece_pose[:3] = data.site_xpos[scene.tool] - [0, 0, GRASP_OFFSET]
                mujoco.mju_mat2Quat(workpiece_pose[3:], data.site_xmat[scene.tool])
            else:
                workpiece_pose[:] = pick_pose if phase.support == "pick_stand" else place_pose
            mujoco.mj_kinematics(model, data)
            problem = scene.check_clearance(data, phase, PLAN_CLEARANCE, trajectory.times[i] + elapsed)
            if problem:
                raise ValueError(f"Reference collision check failed. {problem}")
    return scene.clearance_report()


def simulate(model, trajectory, phases, *, on_step=None, data=None):
    """Run the plan; pass data to continue an existing simulation."""
    reference_clearance = preflight(Scene(model), trajectory, phases)
    scene = Scene(model)
    if data is None:
        data = mujoco.MjData(model)
        data.qpos[:4], data.qpos[4:6] = trajectory.segments[0].q0, OPEN
    start_time = float(data.time)
    mass = np.empty((model.nv, model.nv))
    place_id = model.geom("place_stand").id
    pick_id = model.geom("pick_stand").id
    dt = float(model.opt.timestep)
    steps = int(np.ceil(trajectory.duration / dt - 1e-9))
    phase_index = 0
    events = {}
    effort = np.zeros(model.nu)
    peak_tool_error = 0.0
    saturated_samples = 0
    problem = None

    for step in range(steps + 1):
        phase = phases[phase_index]
        elapsed = step*dt - trajectory.times[phase_index]
        q, velocity, acceleration = phase.segment.evaluate(elapsed)
        grip, grip_velocity = phase.grip_reference(elapsed)
        mujoco.mj_step1(model, data)
        arm = computed_torque(model, data, mass, acceleration, q - data.qpos[:4],
                              velocity - data.qvel[:4], ARM_FREQUENCY_RAD_S, 1)
        fingers = 1000*(grip - data.qpos[4:6]) + 12*(grip_velocity - data.qvel[4:6])
        request = np.r_[arm, fingers]
        data.ctrl[:] = np.clip(request, *model.actuator_ctrlrange.T)
        saturated_samples += int(np.any(request != data.ctrl))
        effort = np.maximum(effort, np.abs(data.ctrl))
        mujoco.mj_forward(model, data)

        time = float(data.time)
        if (abs(time - start_time - step*dt) > 1e-6 or not np.isfinite(data.qpos).all() or
                not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all()):
            raise RuntimeError(f"Invalid physics state during {phase.name} at step {step}.")
        peak_tool_error = max(peak_tool_error, float(np.linalg.norm(
            forward_kinematics(q)[:3] - data.site_xpos[scene.tool])))
        problem = scene.check_clearance(data, phase, RUN_CLEARANCE, time)
        if problem:
            break
        contacts = scene.payload_contacts(data)
        center = data.xpos[scene.payload_body]
        if phase.carrying:
            relative_error = center - data.site_xpos[scene.tool] + [0, 0, GRASP_OFFSET]
            if not scene.grip_is_loaded(contacts) or np.linalg.norm(relative_error) > .005:
                problem = f"Lost the grasp during {phase.name} at {time:.3f} s."
                break

        if on_step is not None and on_step(model, data, q) is False:
            problem = "Viewer closed before the task finished."
            break

        if elapsed + 1e-9 >= phase.segment.duration:
            if phase.name == "grasp":
                if not scene.grip_is_loaded(contacts):
                    problem = "Grasp did not establish bilateral finger contact."
                    break
                events["grasp_s"] = time
            elif phase.name == "lift":
                if contacts.get(pick_id, 0) > 0 or center[2] < model.site("pick_target").pos[2] + .03:
                    problem = "Workpiece did not lift clear of the pick stand."
                    break
                events["lift_s"] = time
            elif phase.name == "lower":
                if contacts.get(place_id, 0) <= 0:
                    problem = "Workpiece did not seat on the place stand."
                    break
                events["seat_s"] = time
            elif phase.name == "release":
                if contacts.get(place_id, 0) <= 0 or any(contacts.get(f, 0) > 0 for f in scene.finger_boxes):
                    problem = "Workpiece did not separate from the fingers on the place stand."
                    break
                events["release_s"] = time
            phase_index += 1

        if step < steps:
            mujoco.mj_step2(model, data)

    # Success requires the workpiece to stay at the destination after retreat.
    center = data.xpos[scene.payload_body]
    rotation = data.xmat[scene.payload_body].reshape(3, 3)
    target = model.site("place_target")
    final_error = float(np.linalg.norm(center - target.pos))
    object_yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    yaw_error = abs(wrap_to_pi(object_yaw - site_yaw(target)))
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
    payload_velocity = data.qvel[scene.payload_dof:scene.payload_dof + 6]
    if problem is None and (contacts.get(place_id, 0) <= 0 or final_error > .005 or
                            yaw_error > .10 or tilt > .10 or
                            np.linalg.norm(payload_velocity[:3]) > .005 or
                            np.linalg.norm(payload_velocity[3:]) > .05):
        problem = "Workpiece did not finish stationary and within the placement tolerance."

    return {
        "success": problem is None,
        "failure": problem,
        "simulated_duration_s": float(data.time),
        "events": events,
        "placement_error_m": final_error,
        "placement_yaw_error_rad": float(yaw_error),
        "placement_tilt_rad": tilt,
        "final_payload_center_m": center.tolist(),
        "max_tool_error_m": peak_tool_error,
        "peak_motor_effort_SI": effort.tolist(),
        "saturated_samples": saturated_samples,
        "reference_clearance": reference_clearance,
        "measured_clearance": scene.clearance_report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Watch the task in the native MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1, help="Viewer playback speed")
    parser.add_argument("--transfer-height", type=float, default=TRANSFER_HEIGHT,
                        help="Tool transfer height in meters")
    args = parser.parse_args()
    viewer = None
    try:
        model = load_scene()
        trajectory, phases = make_plan(model, args.transfer_height)
        if args.view:
            from view_trajectory import LiveViewer
            viewer = LiveViewer(trajectory, positive(args.speed, "playback speed"))
        report = simulate(model, trajectory, phases, on_step=viewer)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
