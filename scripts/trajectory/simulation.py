"""Physics tracking of the existing unloaded SCARA; no scene/model edits.

Uses MuJoCo's own M and bias/passive forces, not an independently derived
part-5 dynamics model. Grippers hold a fixed opening. No object task is claimed.
API reference: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
Force convention: https://mujoco.readthedocs.io/en/stable/computation/index.html
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import mujoco

from planner import ROOT, positive, vector
from scara_kinematics import forward_kinematics, wrap_to_pi


# Physical interfaces that intentionally touch or interlock in this assembly.
ASSEMBLY_INTERFACES = {frozenset(pair) for pair in (
    ("floor", "base_plate"), ("pedestal", "shoulder_hub"),
    ("pedestal", "upper_arm"), ("upper_arm", "elbow_pin"),
    ("forearm", "slide_shaft"), ("slide_shaft", "wrist_hub"),
    ("palm", "finger_left"), ("palm", "finger_right"),
)}


class MeshClearance:
    """Sample signed distances independently of contact masks and parent filtering."""

    def __init__(self, model):
        self.model = model
        geoms = [(model.mesh(model.geom_dataid[i]).name, i) for i in range(model.ngeom)
                 if model.geom_type[i] == int(mujoco.mjtGeom.mjGEOM_MESH)]
        geoms.extend((name, model.geom(name).id) for name in ("floor", "elbow_pin"))
        self.pairs = [(a, b, (name_a, name_b))
                      for (name_a, a), (name_b, b) in combinations(geoms, 2)
                      if model.geom_bodyid[a] != model.geom_bodyid[b] and
                      frozenset((name_a, name_b)) not in ASSEMBLY_INTERFACES]
        self.minimum = float("inf")
        self.pair = None
        self.time = None

    def observe(self, data):
        for a, b, pair in self.pairs:
            distance = mujoco.mj_geomDistance(self.model, data, a, b, 1.0, None)
            if distance < self.minimum:
                self.minimum, self.pair, self.time = distance, pair, float(data.time)

    def metrics(self):
        return dict(minimum_clearance_m=self.minimum, closest_clearance_pair=self.pair,
                    closest_clearance_time_s=self.time)


def setup_model():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara.xml"))
    names = ["shoulder", "elbow", "vertical", "wrist_roll", "left_grip", "right_grip"]
    if model.nq != 6 or model.nv != 6 or model.nu != 6:
        raise ValueError("Tracking demo requires the original six-joint SCARA model.")
    if [model.joint(i).name for i in range(6)] != names:
        raise ValueError("Unexpected model joint order.")
    # This demo explicitly requires six unit-gear joint motors in joint order.
    if not np.array_equal(model.actuator_trnid[:, 0], np.arange(6)):
        raise ValueError("Actuator transmissions do not match the supported joint order.")
    if not np.allclose(model.actuator_gear[:, 0], 1):
        raise ValueError("Non-unit motor gearing requires an explicit force mapping.")
    return model


def contact_pairs(model, data):
    return sorted({f"{int(c.geom1)}:{int(c.geom2)}" for c in data.contact})


def simulate(trajectory, settings, *, controller="computed_torque", initial_offset=None,
             limits=None, on_step=None, check_clearance=False):
    """Track a reference with motor forces and return aligned logs and metrics.

    ``on_step(model, data, desired_q)`` optionally observes each logged state.
    Returning False stops the run; ``completed`` then records the interruption.
    Observers must not alter the model, state, or controls.
    ``check_clearance`` measures signed distances on the unloaded assembly.
    """
    if controller not in ("computed_torque", "pd_bias"):
        raise ValueError("controller must be computed_torque or pd_bias.")
    model = setup_model()
    data = mujoco.MjData(model)
    clearance = MeshClearance(model) if check_clearance else None
    offset = np.zeros(4) if initial_offset is None else vector(initial_offset, "initial_offset")
    data.qpos[:4] = trajectory.evaluate(0)[0] + offset
    data.qpos[4:] = 0.009  # Hold away from the gripper travel stops.
    if np.any(data.qpos < model.jnt_range[:, 0]) or np.any(data.qpos > model.jnt_range[:, 1]):
        raise ValueError("Initial state exceeds a model joint limit.")
    target_grip = data.qpos[4:].copy()
    wn = positive(settings["controller_natural_frequency_rad_s"], "natural frequency")
    damping = positive(settings["controller_damping_ratio"], "damping ratio")
    settle = float(settings["settle_time_s"])
    if not np.isfinite(settle) or settle < 0:
        raise ValueError("settle time must be finite and nonnegative.")
    dt = float(model.opt.timestep)
    steps = int(np.ceil((trajectory.duration + settle) / dt))
    # Fixed SI direct-force gains for the PD+bias comparison, with force units
    # distinct from computed-torque's acceleration feedback gains.
    kp_direct = np.array([25, 12, 120, 0.8, 20, 20])
    kd_direct = np.array([3, 1, 12, 0.08, 1, 1])
    mass = np.empty((model.nv, model.nv))
    logs = {name: [] for name in ("time", "desired_q", "desired_qdot", "desired_qddot",
                                "q", "qdot", "qddot", "requested_effort", "effort",
                                "tool_error", "contacts")}
    pairs = set()
    for step in range(steps + 1):
        # Refresh kinematic and velocity-dependent fields before feedback.
        mujoco.mj_step1(model, data)
        qd, vd, ad = trajectory.evaluate(float(data.time))
        desired = np.r_[qd, target_grip]
        desired_velocity = np.r_[vd, 0, 0]
        desired_acceleration = np.r_[ad, 0, 0]
        error, rate_error = desired - data.qpos, desired_velocity - data.qvel
        if controller == "computed_torque":
            # MuJoCo 3.17 API: mj_fullM(model, data, destination).
            mujoco.mj_fullM(model, data, mass)
            v = desired_acceleration + 2*damping*wn*rate_error + wn**2*error
            requested = mass @ v + data.qfrc_bias - data.qfrc_passive
        else:
            requested = kp_direct*error + kd_direct*rate_error + data.qfrc_bias - data.qfrc_passive
        effort = np.clip(requested, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = effort
        # Compute acceleration with the just-applied command for aligned logs.
        mujoco.mj_forward(model, data)
        if clearance is not None:
            clearance.observe(data)
        actual_tool = forward_kinematics(data.qpos[:4])
        tool_error = forward_kinematics(qd) - actual_tool
        tool_error[3] = wrap_to_pi(tool_error[3])
        pairs.update(contact_pairs(model, data))
        values = (float(data.time), qd, vd, ad, data.qpos[:4].copy(),
                  data.qvel[:4].copy(), data.qacc[:4].copy(), requested.copy(),
                  effort.copy(), tool_error, data.ncon)
        for key, value in zip(logs, values):
            logs[key].append(value)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite simulation state.")
        if on_step is not None and on_step(model, data, qd) is False:
            break
        if step < steps:
            mujoco.mj_step2(model, data)
    logs = {key: np.array(value) for key, value in logs.items()}
    error = logs["desired_q"] - logs["q"]
    peak_effort = np.max(np.abs(logs["effort"]), axis=0)
    metric = {
        "controller": controller, "mujoco_version": mujoco.__version__,
        "completed": len(logs["time"]) == steps + 1,
        "physics_timestep_s": dt, "trajectory_duration_s": trajectory.duration,
        "simulated_duration_s": float(logs["time"][-1]),
        "initial_joint_offset_SI": offset.tolist(),
        "feedback_gain_convention": "commanded acceleration" if controller == "computed_torque" else "direct generalized force",
        "kp": ([wn**2]*6 if controller == "computed_torque" else kp_direct.tolist()),
        "kd": ([2*damping*wn]*6 if controller == "computed_torque" else kd_direct.tolist()),
        "max_joint_error_SI": np.max(np.abs(error), axis=0).tolist(),
        "rms_joint_error_SI": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "final_joint_error_SI": error[-1].tolist(),
        "peak_effort_SI": peak_effort.tolist(),
        "effort_limits_SI": model.actuator_ctrlrange[:, 1].tolist(),
        "peak_actual_velocity_SI": np.max(np.abs(logs["qdot"]), axis=0).tolist(),
        "peak_actual_acceleration_SI": np.max(np.abs(logs["qddot"]), axis=0).tolist(),
        "saturated_samples": int(np.count_nonzero(np.any(np.abs(logs["requested_effort"]-logs["effort"]) > 1e-10, axis=1))),
        "max_tool_position_error_m": float(np.max(np.linalg.norm(logs["tool_error"][:, :3], axis=1))),
        "final_tool_position_error_m": float(np.linalg.norm(logs["tool_error"][-1, :3])),
        "max_tool_yaw_error_rad": float(np.max(np.abs(logs["tool_error"][:, 3]))),
        "maximum_active_contacts": int(np.max(logs["contacts"])),
        "observed_contact_geom_id_pairs": sorted(pairs),
        "model_position_limits_respected": bool(np.all(logs["q"] >= model.jnt_range[:4, 0]-1e-8)
                                                and np.all(logs["q"] <= model.jnt_range[:4, 1]+1e-8)),
        "scope": "Unloaded existing model. No objects, grasp/release, or collision-free certificate. Dynamics supplied by the same simulator as the plant."
    }
    if limits is not None:
        metric["actual_velocity_limits_respected"] = bool(np.all(np.abs(logs["qdot"]) <= limits.velocity + 1e-8))
        metric["actual_acceleration_limits_respected"] = bool(np.all(np.abs(logs["qddot"]) <= limits.acceleration + 1e-8))
    if clearance is not None:
        metric.update(clearance.metrics())
    return logs, metric
