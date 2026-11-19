"""SCARA simulator in MuJoCo's native viewer.

Everything on screen is the standard MuJoCo viewer: its own left/right panels,
its Control sliders as the inputs, and its own overlay text and figures.

Modes (switch with F6-F12, or start one with --mode):
  F6  Model                 look at the robot; joint axes and body frames shown
  F7  Forward kinematics    sliders = joint values      -> tool pose
  F8  Inverse kinematics    sliders = target x,y,z,yaw  -> joint values
  F9  Velocity kinematics   sliders = joint rates       -> tool velocity (J qdot)
  F10 Resolved rate         sliders = tool velocity     -> joint rates (DLS J^-1)
  F11 Dynamics              sliders = motor efforts     -> physics + our model
  F12 Trajectory            computed-torque tracking of the planned route

Space pauses the motion modes. Backspace (Reset) returns to the home pose.

    python scripts/app/scara_sim.py [--mode fk]
    python scripts/app/scara_sim.py --check      # headless self-test
"""
from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("kinematics", "dynamics", "trajectory"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

from scara_kinematics import (PARAMETERS, JointLimitError,  # noqa: E402
                              UnreachableTargetError, damped_least_squares,
                              forward_kinematics, inverse_kinematics, task_jacobian)
from scara_dynamics import forward_dynamics, load_dynamic_parameters  # noqa: E402

MODEL_PATH = ROOT / "models" / "scara.xml"
HOME = np.array([math.radians(25), math.radians(-65), 0.045, math.radians(20)])
ARM_JOINTS = ["shoulder", "elbow", "vertical", "wrist_roll"]
DEG = 180 / math.pi
TO_DISPLAY = np.array([DEG, DEG, 1000, DEG])          # rad, rad, m, rad -> deg, deg, mm, deg
UNITS = ["deg", "deg", "mm", "deg"]
F6, SPACE = 295, 32                                    # GLFW key codes


# ---------------------------------------------------------------------------
# Model variants: same robot, different native Control sliders per mode
# ---------------------------------------------------------------------------
def robot_tree(title: str) -> ET.Element:
    root = ET.parse(MODEL_PATH).getroot()
    root.set("model", f"SCARA - {title}")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((MODEL_PATH.parent / compiler.get("meshdir", ".")).resolve()))
    return root


def set_sliders(root: ET.Element, sliders) -> None:
    """Replace the actuators with named sliders (name, joint, low, high).

    In the kinematic modes nothing is ever stepped, so these 'actuators' are
    only input sliders shown in MuJoCo's own Control panel.
    """
    actuators = root.find("actuator")
    for child in list(actuators):
        actuators.remove(child)
    for name, joint, low, high in sliders:
        ET.SubElement(actuators, "general", name=name, joint=joint,
                      ctrllimited="true", ctrlrange=f"{low:g} {high:g}")


def add_home_key(root: ET.Element, qpos, ctrl) -> None:
    keyframe = ET.SubElement(root, "keyframe")
    attrs = {"name": "home", "qpos": " ".join(f"{v:.9g}" for v in qpos)}
    if len(ctrl):
        attrs["ctrl"] = " ".join(f"{v:.9g}" for v in ctrl)
    ET.SubElement(keyframe, "key", **attrs)


def compile_tree(root: ET.Element) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# Small drawing helpers for viewer.user_scn (native MuJoCo geoms)
# ---------------------------------------------------------------------------
def add_geom(scene, kind, size, pos, rgba, end=None, width=0.0):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    if end is not None:
        mujoco.mjv_connector(geom, kind, width, np.asarray(pos, float), np.asarray(end, float))
    scene.ngeom += 1


def add_arrow(scene, start, end, rgba, width=0.004):
    if np.linalg.norm(np.asarray(end) - np.asarray(start)) > 1e-6:
        add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, [width, width, 0], start, rgba, end, width)


def add_polyline(scene, points, rgba, width=0.0015):
    for a, b in zip(points[:-1], points[1:]):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, [width, 0, 0], a, rgba, b, width)


def columns(rows):
    """Turn [(label, value), ...] into the two text columns of a native overlay."""
    return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)


def singular_status(q2: float) -> str:
    s = abs(math.sin(q2))
    return f"{'SINGULAR' if s < 1e-6 else 'near singular' if s <= 0.05 else 'regular'} (|sin q2| = {s:.3f})"


def fmt_q(q) -> str:
    return "  ".join(f"{v * f:7.1f}" for v, f in zip(q, TO_DISPLAY))


def joint_rows(q, label="q", rate=False):
    """Two compact rows: q1, q2 [deg] and q3 [mm], q4 [deg] (or their rates)."""
    s = "/s" if rate else ""
    v = np.asarray(q) * TO_DISPLAY
    return [(f"{label}1, {label}2", f"{v[0]:.1f}, {v[1]:.1f} deg{s}"),
            (f"{label}3, {label}4", f"{v[2]:.1f} mm{s}, {v[3]:.1f} deg{s}")]


def short_singular(q2: float) -> str:
    s = abs(math.sin(q2))
    return f"{'SINGULAR' if s < 1e-6 else 'near singular' if s <= 0.05 else 'regular'}, |sin q2| {s:.2f}"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
class Mode:
    title = ""
    physics = False

    def __init__(self):
        self.paused = False
        self.model = self.build()
        self.data = mujoco.MjData(self.model)
        self.reset()

    # Overridden by each mode -------------------------------------------------
    def build(self) -> mujoco.MjModel: ...
    def update(self, dt: float) -> None: ...
    def draw(self, scene) -> None: ...
    def info(self) -> list: return []
    def figure(self): return None

    def setup_view(self, viewer) -> None:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE   # tool frame

    # Shared ------------------------------------------------------------------
    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.clock = 0.0

    def step(self, dt: float) -> None:
        # The viewer's Reset button zeroes data.time; treat that as "go home".
        if self.data.time < self.clock - 1e-12:
            self.reset()
        if not self.paused:
            self.update(dt)
        if not self.physics:
            self.data.time += dt
        self.clock = self.data.time

    def tool(self) -> np.ndarray:
        return self.data.site("tool").xpos.copy()

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[:4].copy()


class ModelMode(Mode):
    title = "Model"

    def build(self):
        root = robot_tree(self.title)
        set_sliders(root, [])
        add_home_key(root, [*HOME, 0.009, 0.009], [])
        return compile_tree(root)

    def setup_view(self, viewer):
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1

    def info(self):
        p = PARAMETERS
        return [
            ("MODEL", "R-R-P-R SCARA + gripper"),
            ("a1, a2", f"{p.a1*1000:.0f}, {p.a2*1000:.0f} mm"),
            ("h, d4", f"{p.base_height*1000:.0f}, {p.d4*1000:.0f} mm"),
            ("d3*", f"{p.d3_min*1000:.0f}-{p.d3_max*1000:.0f} mm"),
            ("q1, q2 range", "+-150, +-145 deg"),
            ("q4 range", "+-180 deg"),
            ("Shown", "body frames, joint axes"),
        ]


class ForwardKinematics(Mode):
    title = "Forward kinematics"

    def build(self):
        root = robot_tree(self.title)
        set_sliders(root, [("shoulder [deg]", "shoulder", -150, 150),
                           ("elbow [deg]", "elbow", -145, 145),
                           ("slide q3 [mm]", "vertical", 0, 110),
                           ("wrist [deg]", "wrist_roll", -180, 180),
                           ("gripper opening [mm]", "left_grip", 0, 18)])
        add_home_key(root, [*HOME, 0.009, 0.009], [*(HOME * TO_DISPLAY), 9])
        return compile_tree(root)

    def update(self, dt):
        c = self.data.ctrl
        self.data.qpos[:4] = c[:4] / TO_DISPLAY
        self.data.qpos[4:6] = c[4] / 1000
        mujoco.mj_forward(self.model, self.data)

    def info(self):
        pose = forward_kinematics(self.q)
        error = np.linalg.norm(pose[:3] - self.tool()) * 1e6
        return [("FORWARD KINEMATICS", "inputs: Control panel"),
                *joint_rows(self.q),
                ("tool x, y", f"{pose[0]*1000:.1f}, {pose[1]*1000:.1f} mm"),
                ("tool z", f"{pose[2]*1000:.1f} mm"),
                ("yaw q1+q2-q4", f"{pose[3]*DEG:.1f} deg"),
                ("FK vs MuJoCo", f"{error:.3f} um"),
                ("det J", f"{PARAMETERS.a1*PARAMETERS.a2*math.sin(self.q[1]):+.4f}"),
                ("Jacobian", short_singular(self.q[1]))]


class InverseKinematics(Mode):
    title = "Inverse kinematics"

    def build(self):
        root = robot_tree(self.title)
        reach = (PARAMETERS.a1 + PARAMETERS.a2) * 1000 + 30
        zmax = (PARAMETERS.base_height - PARAMETERS.d3_min - PARAMETERS.d4) * 1000
        zmin = (PARAMETERS.base_height - PARAMETERS.d3_max - PARAMETERS.d4) * 1000
        set_sliders(root, [("target x [mm]", "shoulder", -reach, reach),
                           ("target y [mm]", "elbow", -reach, reach),
                           ("target z [mm]", "vertical", zmin - 15, zmax + 15),
                           ("target yaw [deg]", "wrist_roll", -180, 180),
                           ("elbow branch (-/+)", "left_grip", -1, 1)])
        home = forward_kinematics(HOME)
        add_home_key(root, [*HOME, 0.009, 0.009],
                     [*(home[:3] * 1000), home[3] * DEG, -1])
        return compile_tree(root)

    def reset(self):
        super().reset()
        self.status, self.reached, self.solutions = "home", True, []

    def target(self):
        c = self.data.ctrl
        return np.array([c[0] / 1000, c[1] / 1000, c[2] / 1000, c[3] / DEG])

    def update(self, dt):
        target = self.target()
        branch = 1 if self.data.ctrl[4] >= 0 else -1
        try:
            self.solutions = inverse_kinematics(target)
            chosen = [q for q in self.solutions if q[1] * branch >= -1e-9]
            if chosen:
                self.data.qpos[:4] = chosen[0]
                self.status, self.reached = "reached", True
            else:
                self.status, self.reached = "this elbow branch violates a joint limit", False
        except UnreachableTargetError as error:
            self.solutions, self.status, self.reached = [], f"unreachable: {error}", False
        except JointLimitError:
            self.solutions, self.status, self.reached = [], "reachable geometrically, but outside joint limits", False
        mujoco.mj_forward(self.model, self.data)

    def draw(self, scene):
        target = self.target()
        color = [0.1, 0.8, 0.2, 0.9] if self.reached else [0.9, 0.1, 0.1, 0.9]
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], target[:3], color)
        end = target[:3] + 0.05 * np.array([math.cos(target[3]), math.sin(target[3]), 0])
        add_arrow(scene, target[:3], end, color, 0.003)

    def info(self):
        t = self.target()
        rows = [("INVERSE KINEMATICS", "inputs: Control panel"),
                ("target x, y", f"{t[0]*1000:.1f}, {t[1]*1000:.1f} mm"),
                ("target z, yaw", f"{t[2]*1000:.1f} mm, {t[3]*DEG:.1f} deg"),
                ("r", f"{math.hypot(*t[:2])*1000:.1f} mm (139-420 ok)"),
                ("status", self.status if len(self.status) < 30 else self.status[:28] + "..")]
        for q in self.solutions:
            v = q * TO_DISPLAY
            rows.append(("elbow +" if q[1] >= 0 else "elbow -",
                         f"{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}, {v[3]:.1f}"))
        return rows


class VelocityKinematics(Mode):
    title = "Velocity kinematics"
    RATES = np.array([60, 60, 60, 90])        # slider ranges: deg/s, deg/s, mm/s, deg/s

    def build(self):
        root = robot_tree(self.title)
        r = self.RATES
        set_sliders(root, [("shoulder rate [deg/s]", "shoulder", -r[0], r[0]),
                           ("elbow rate [deg/s]", "elbow", -r[1], r[1]),
                           ("slide rate [mm/s]", "vertical", -r[2], r[2]),
                           ("wrist rate [deg/s]", "wrist_roll", -r[3], r[3])])
        add_home_key(root, [*HOME, 0.009, 0.009], np.zeros(4))
        return compile_tree(root)

    def reset(self):
        super().reset()
        self.qdot = np.zeros(4)
        self.trail = deque(maxlen=300)
        self.limited = False

    def update(self, dt):
        wanted = self.data.ctrl[:4] / TO_DISPLAY
        low, high = self.model.jnt_range[:4, 0], self.model.jnt_range[:4, 1]
        new = np.clip(self.q + wanted * dt, low, high)
        self.limited = not np.allclose(new, self.q + wanted * dt)
        self.qdot = (new - self.q) / dt if dt > 0 else np.zeros(4)
        self.data.qpos[:4] = new
        mujoco.mj_forward(self.model, self.data)
        if not self.trail or np.linalg.norm(self.tool() - self.trail[-1]) > 0.001:
            self.trail.append(self.tool())

    def velocity(self):
        return task_jacobian(self.q) @ self.qdot

    def draw(self, scene):
        add_polyline(scene, list(self.trail), [0.2, 0.6, 0.9, 0.8])
        v = self.velocity()
        add_arrow(scene, self.tool(), self.tool() + 0.5 * v[:3], [1.0, 0.7, 0.1, 1])

    def info(self):
        v = self.velocity()
        return [("VELOCITY KINEMATICS", "inputs: Control panel"),
                *joint_rows(self.qdot, "qd", rate=True),
                ("v = J qd  x, y", f"{v[0]*1000:.1f}, {v[1]*1000:.1f} mm/s"),
                ("v z, yaw rate", f"{v[2]*1000:.1f} mm/s, {v[3]*DEG:.1f} deg/s"),
                ("tool speed", f"{np.linalg.norm(v[:3])*1000:.1f} mm/s (arrow)"),
                ("det J", f"{np.linalg.det(task_jacobian(self.q)):+.4f}"),
                ("Jacobian", short_singular(self.q[1])),
                ("limits", "CLIPPED" if self.limited else "free")]


class ResolvedRate(VelocityKinematics):
    title = "Resolved rate"
    MAX_RATES = np.array([math.radians(90), math.radians(90), 0.08, math.radians(120)])

    def build(self):
        root = robot_tree(self.title)
        set_sliders(root, [("tool vx [mm/s]", "shoulder", -60, 60),
                           ("tool vy [mm/s]", "elbow", -60, 60),
                           ("tool vz [mm/s]", "vertical", -40, 40),
                           ("tool yaw rate [deg/s]", "wrist_roll", -60, 60)])
        add_home_key(root, [*HOME, 0.009, 0.009], np.zeros(4))
        return compile_tree(root)

    def commanded(self):
        c = self.data.ctrl
        return np.array([c[0] / 1000, c[1] / 1000, c[2] / 1000, c[3] / DEG])

    def update(self, dt):
        rates = damped_least_squares(task_jacobian(self.q), self.commanded())
        wanted = np.clip(rates, -self.MAX_RATES, self.MAX_RATES)
        low, high = self.model.jnt_range[:4, 0], self.model.jnt_range[:4, 1]
        new = np.clip(self.q + wanted * dt, low, high)
        self.limited = not np.allclose(new, self.q + wanted * dt)
        self.qdot = (new - self.q) / dt if dt > 0 else np.zeros(4)
        self.data.qpos[:4] = new
        mujoco.mj_forward(self.model, self.data)
        if not self.trail or np.linalg.norm(self.tool() - self.trail[-1]) > 0.001:
            self.trail.append(self.tool())

    def draw(self, scene):
        add_polyline(scene, list(self.trail), [0.2, 0.6, 0.9, 0.8])
        add_arrow(scene, self.tool(), self.tool() + 0.5 * self.commanded()[:3], [1.0, 0.7, 0.1, 1])
        add_arrow(scene, self.tool(), self.tool() + 0.5 * self.velocity()[:3], [0.1, 0.8, 0.9, 1], 0.003)

    def info(self):
        cmd, got = self.commanded(), self.velocity()
        sigma = np.linalg.svd(task_jacobian(self.q), compute_uv=False)[-1]
        return [("RESOLVED RATE", "inputs: Control panel"),
                ("wanted v (orange)", ", ".join(f"{x*1000:.1f}" for x in cmd[:3]) + " mm/s"),
                ("got J qd (cyan)", ", ".join(f"{x*1000:.1f}" for x in got[:3]) + " mm/s"),
                *joint_rows(self.qdot, "qd", rate=True),
                ("sigma_min(J)", f"{sigma:.4f} {'DLS damped' if sigma < 0.01 else 'exact J^-1'}"),
                ("Jacobian", short_singular(self.q[1])),
                ("limits", "CLIPPED" if self.limited else "free")]


class Figure:
    """A native MuJoCo figure (mjvFigure) with a rolling time window."""

    def __init__(self, title, names, ylabel, window=6.0):
        self.fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig)
        self.fig.title = title
        self.fig.xlabel = "time [s]"
        self.fig.flg_legend = 1
        self.fig.figurergba[3] = 0.6
        for i, name in enumerate(names):
            self.fig.linename[i] = name
        self.names, self.window = names, window
        self.history = deque()

    def add(self, t, values):
        self.history.append((t, np.asarray(values, float)))
        while self.history and self.history[0][0] < t - self.window:
            self.history.popleft()
        n = min(len(self.history), self.fig.linedata.shape[1] // 2)
        samples = list(self.history)[-n:]
        times = np.array([s[0] for s in samples])
        values = np.array([s[1] for s in samples])
        for i in range(len(self.names)):
            self.fig.linepnt[i] = n
            self.fig.linedata[i, 0:2 * n:2] = times
            self.fig.linedata[i, 1:2 * n:2] = values[:, i]
        # Explicit axis ranges (the default figure range is a fixed 0..1).
        low, high = float(values.min()), float(values.max())
        pad = max(0.1 * (high - low), 1e-9 + 1e-3 * max(abs(low), abs(high)))
        self.fig.range[0] = [max(0.0, t - self.window), max(t, self.window)]
        self.fig.range[1] = [low - pad, high + pad]

    def clear(self):
        self.history.clear()
        self.fig.linepnt[:] = 0


class Dynamics(Mode):
    title = "Dynamics"
    physics = True

    def build(self):
        root = robot_tree(self.title)
        # Fingers welded to the wrist (rigid body 4), as in the derivation.
        for body in root.iter("body"):
            for joint in body.findall("joint"):
                if joint.get("name") in ("left_grip", "right_grip"):
                    body.remove(joint)
        params = load_dynamic_parameters(grip=0.0)
        actuators = root.find("actuator")
        for child in list(actuators):
            actuators.remove(child)
        for name, joint, limit in (("shoulder torque [Nm]", "shoulder", 12), ("elbow torque [Nm]", "elbow", 8),
                                   ("slide force [N]", "vertical", 80), ("wrist torque [Nm]", "wrist_roll", 2)):
            ET.SubElement(actuators, "motor", name=name, joint=joint, ctrllimited="true",
                          ctrlrange=f"{-limit} {limit}")
        # Gravity hold: 1 = add F3 = -mu g on the slide (upward push).
        ET.SubElement(actuators, "general", name="gravity hold (0/1)", joint="vertical",
                      ctrllimited="true", ctrlrange="0 1", gainprm=f"{-params.mu * params.g:.9g}")
        add_home_key(root, [-0.5, 0.5, 0.02, 0.0], [0, 0, 0, 0, 1])
        self.params = params
        return compile_tree(root)

    def reset(self):
        super().reset()
        self.qdd_ours = np.zeros(4)
        self.plot = Figure("shoulder & elbow accel: MuJoCo vs our D,C,G",
                           ["q1dd MuJoCo", "q1dd ours", "q2dd MuJoCo", "q2dd ours"], "rad/s^2")
        self.worst = 0.0
        self.compare()

    def compare(self):
        d = self.data
        tau = d.qfrc_actuator[:4] + d.qfrc_passive[:4] + d.qfrc_constraint[:4]
        self.qdd_ours = forward_dynamics(d.qpos[:4], d.qvel[:4], tau, self.params)
        self.error = float(np.max(np.abs(self.qdd_ours - d.qacc[:4])))

    def update(self, dt):
        steps = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.compare()
        self.worst = max(self.worst, self.error)
        a = self.data.qacc
        self.plot.add(self.data.time, [a[0], self.qdd_ours[0], a[1], self.qdd_ours[1]])

    def figure(self):
        return self.plot.fig

    def info(self):
        d = self.data
        return [("DYNAMICS", "inputs: Control panel"),
                *joint_rows(d.qpos[:4]),
                ("gravity hold F3", f"{-self.params.mu*self.params.g*d.ctrl[4]:.2f} N"),
                ("limit force", "ACTIVE" if d.nefc else "none"),
                ("our D qdd+C qd+G=tau", ""),
                ("|qdd ours-MuJoCo|", f"{self.error:.1e} (worst {self.worst:.1e})"),
                ("Space", "pause")]


class Trajectory(Mode):
    title = "Trajectory"
    physics = True

    def build(self):
        from planner import example_trajectory, load_settings
        self.limits, self.settings = load_settings()
        self.route = example_trajectory(self.limits, self.settings)
        root = robot_tree(self.title)
        names = ["shoulder torque [Nm]", "elbow torque [Nm]", "slide force [N]", "wrist torque [Nm]",
                 "left finger [N]", "right finger [N]"]
        for motor, name in zip(root.find("actuator"), names):
            motor.set("name", name)
        add_home_key(root, [*self.route.evaluate(0)[0], 0.009, 0.009], np.zeros(6))
        self.path = np.array([forward_kinematics(self.route.evaluate(t)[0])[:3]
                              for t in np.linspace(0, self.route.duration, 200)])
        return compile_tree(root)

    def reset(self):
        super().reset()
        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.plot = Figure("joint tracking error", ["shoulder [deg]", "elbow [deg]", "slide [mm]", "wrist [deg]"],
                           "error")
        self.saturated = 0
        self.peak = 0.0

    def control(self):
        d, wn, zeta = self.data, self.settings["controller_natural_frequency_rad_s"], self.settings["controller_damping_ratio"]
        qd, vd, ad = self.route.evaluate(min(d.time, self.route.duration))
        desired = np.r_[qd, 0.009, 0.009]
        error, rate_error = desired - d.qpos, np.r_[vd, 0, 0] - d.qvel
        mujoco.mj_fullM(self.model, d, self.mass)
        v = np.r_[ad, 0, 0] + 2 * zeta * wn * rate_error + wn**2 * error
        requested = self.mass @ v + d.qfrc_bias - d.qfrc_passive
        low, high = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        d.ctrl[:] = np.clip(requested, low, high)
        self.saturated += int(np.any(np.abs(d.ctrl - requested) > 1e-10))
        return qd

    def update(self, dt):
        steps = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(steps):
            if self.data.time > self.route.duration + 1.0:     # loop the route
                self.reset()
            mujoco.mj_step1(self.model, self.data)
            self.control()
            mujoco.mj_step2(self.model, self.data)
        # Compare against the target at the same instant as the measured pose.
        self.qd = self.route.evaluate(min(self.data.time, self.route.duration))[0]
        error = (self.qd - self.data.qpos[:4]) * TO_DISPLAY
        tool_error = np.linalg.norm(forward_kinematics(self.qd)[:3] - forward_kinematics(self.q)[:3])
        self.peak = max(self.peak, tool_error)
        self.tool_error = tool_error
        self.plot.add(self.data.time, error)

    def draw(self, scene):
        add_polyline(scene, self.path, [0.1, 0.75, 0.85, 0.7])
        qd = self.route.evaluate(min(self.data.time, self.route.duration))[0]
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0],
                 forward_kinematics(qd)[:3], [1.0, 0.5, 0.1, 0.9])

    def figure(self):
        return self.plot.fig

    def info(self):
        d = self.data
        return [("TRAJECTORY", "computed torque"),
                ("time", f"{min(d.time, self.route.duration):.2f} / {self.route.duration:.2f} s"),
                ("tool error", f"{getattr(self, 'tool_error', 0)*1000:.3f} mm"),
                ("peak error", f"{self.peak*1000:.3f} mm"),
                ("tau1, tau2", f"{d.ctrl[0]:.2f}, {d.ctrl[1]:.2f} Nm"),
                ("F3, tau4", f"{d.ctrl[2]:.2f} N, {d.ctrl[3]:.3f} Nm"),
                ("saturated steps", str(self.saturated)),
                ("cyan / orange", "path / target"),
                ("Space", "pause")]


MODES = [ModelMode, ForwardKinematics, InverseKinematics, VelocityKinematics,
         ResolvedRate, Dynamics, Trajectory]
NAMES = ["model", "fk", "ik", "velocity", "rate", "dynamics", "trajectory"]


# ---------------------------------------------------------------------------
# Native viewer loop
# ---------------------------------------------------------------------------
def mode_list(current: int) -> tuple[str, str]:
    keys = "\n".join(f"F{6 + i}" for i in range(len(MODES)))
    titles = "\n".join(m.title + ("  <" if i == current else "") for i, m in enumerate(MODES))
    return keys, titles


def run(start: int, duration: float = 0.0) -> None:
    import mujoco.viewer

    request = {"mode": start, "switch": None}
    camera = None
    started = time.perf_counter()
    while True:
        mode = MODES[request["mode"]]()

        def on_key(key, mode=mode):
            if F6 <= key < F6 + len(MODES):
                request["switch"] = key - F6
            elif key == SPACE and mode.physics:
                mode.paused = not mode.paused

        with mujoco.viewer.launch_passive(mode.model, mode.data, key_callback=on_key) as viewer:
            with viewer.lock():
                if camera is None:
                    viewer.cam.lookat[:] = [0.15, 0.0, 0.17]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, 135, -25
                else:
                    viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera
                mode.setup_view(viewer)
            last = time.perf_counter()
            while viewer.is_running() and request["switch"] is None:
                now = time.perf_counter()
                dt, last = min(now - last, 0.05), now
                with viewer.lock():
                    mode.step(dt)
                    viewer.user_scn.ngeom = 0
                    mode.draw(viewer.user_scn)
                    camera = (viewer.cam.lookat.copy(), viewer.cam.distance,
                              viewer.cam.azimuth, viewer.cam.elevation)
                label, value = columns(mode.info())
                if mode.physics and mode.paused:
                    label, value = "PAUSED (Space)\n" + label, "\n" + value
                keys, titles = mode_list(request["mode"])
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, value),
                                  (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, keys, titles)])
                figure = mode.figure()
                rect = viewer.viewport
                if figure is not None and rect is not None and rect.width > 0:
                    # Bottom right, beside the mode list; it grows with the window.
                    w = int(min(rect.width * 0.5, max(rect.width - 330, 220)))
                    h = int(min(rect.height * 0.4, 0.75 * w))
                    viewer.set_figures((mujoco.MjrRect(rect.left + rect.width - w, rect.bottom, w, h), figure))
                viewer.sync()
                if duration and now - started > duration:
                    return
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
        if request["switch"] is None:
            return                       # window closed
        request["mode"], request["switch"] = request["switch"], None


# ---------------------------------------------------------------------------
# Headless self-test of every mode (no window)
# ---------------------------------------------------------------------------
def check() -> None:
    m = ModelMode()
    assert m.model.nu == 0 and np.allclose(m.q, HOME)

    fk = ForwardKinematics()
    fk.data.ctrl[:4] = [30, -50, 40, 10]
    fk.step(0.02)
    assert np.allclose(fk.q, np.array([30, -50, 40, 10]) / TO_DISPLAY)
    assert np.linalg.norm(forward_kinematics(fk.q)[:3] - fk.tool()) < 1e-9

    ik = InverseKinematics()
    for branch in (-1, 1):
        ik.data.ctrl[:] = [300, 50, 80, 15, branch]
        ik.step(0.02)
        assert ik.reached and np.sign(ik.q[1]) == branch
        assert np.allclose(forward_kinematics(ik.q), ik.target(), atol=1e-9)
    ik.data.ctrl[:] = [500, 0, 80, 0, -1]
    ik.step(0.02)
    assert not ik.reached

    vel = VelocityKinematics()
    vel.data.ctrl[:] = [20, -10, 5, 30]
    before = forward_kinematics(vel.q)
    vel.step(1e-4)
    numeric = (forward_kinematics(vel.q) - before) / 1e-4
    assert np.allclose(numeric[:3], vel.velocity()[:3], atol=1e-5)

    rate = ResolvedRate()
    rate.data.ctrl[:] = [20, -10, 5, 10]
    rate.step(1e-6)
    assert np.allclose(rate.velocity(), rate.commanded(), atol=1e-7), (rate.velocity(), rate.commanded())

    dyn = Dynamics()
    dyn.data.ctrl[:] = [0.3, -0.1, 0, 0.01, 1]
    for _ in range(50):
        dyn.step(0.02)
    assert dyn.worst < 1e-8, dyn.worst
    assert abs(dyn.data.qvel[2]) < 0.05        # gravity hold keeps the slide nearly still

    traj = Trajectory()
    for _ in range(100):
        traj.step(0.02)
    assert traj.peak < 1e-3 and traj.saturated == 0, (traj.peak, traj.saturated)

    reset = ForwardKinematics()
    reset.step(0.1)
    mujoco.mj_resetData(reset.model, reset.data)    # what the viewer's Reset does
    reset.step(0.02)
    assert np.allclose(reset.q, HOME)
    print("PASS: all 7 modes build; FK/IK/velocity/resolved-rate math, dynamics vs MuJoCo "
          f"(worst {dyn.worst:.1e}), trajectory tracking (peak {traj.peak*1000:.3f} mm), reset to home.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=NAMES, default="model")
    parser.add_argument("--check", action="store_true", help="headless self-test, no window")
    parser.add_argument("--duration", type=float, default=0.0, help="auto-close after N seconds (testing)")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        run(NAMES.index(args.mode), args.duration)


if __name__ == "__main__":
    main()
