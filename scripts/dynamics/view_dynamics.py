"""Watch the dynamics in the MuJoCo viewer and compare against the hand derivation.

Two modes:

  forward  Apply a chosen torque profile to the MuJoCo robot.  At the same time,
           integrate our forward_dynamics() from the same start state with the
           same torques, and report how far the two motions drift apart.

  inverse  Choose a smooth joint trajectory, compute the torques with our
           inverse_dynamics(), and send them open loop (no feedback) to MuJoCo.
           If the derivation is right, the robot follows the planned path.

Usage (from the repository root):

    python scripts/dynamics/view_dynamics.py                  # forward mode, viewer
    python scripts/dynamics/view_dynamics.py --mode inverse
    python scripts/dynamics/view_dynamics.py --tau 0.8 0 0 0  # constant extra torques
    python scripts/dynamics/view_dynamics.py --damping        # keep joint damping
    python scripts/dynamics/view_dynamics.py --headless       # no window, print only

The script makes a few changes to the model so that it matches the
derivation exactly:
  * the finger joints are removed, so the fingers are welded to the wrist as
    the rigid body 4 at grip = 0,
  * joint damping is set to zero (unless you pass --damping),
  * MuJoCo uses its RK4 integrator, so integration error doesn't hide model error.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scara_dynamics import DYNAMICS, MODEL_PATH, inverse_dynamics, rk4_step  # noqa: E402

JOINTS = ["shoulder", "elbow", "vertical", "wrist_roll"]
UNITS = ["rad", "rad", "m", "rad"]


def load_arm_model(damping: bool) -> mujoco.MjModel:
    """Load scara.xml with the fingers rigidly attached to the wrist."""
    root = ET.parse(MODEL_PATH).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((MODEL_PATH.parent / compiler.get("meshdir", ".")).resolve()))
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("name") in ("left_grip", "right_grip"):
                body.remove(joint)
    actuators = root.find("actuator")
    for motor in list(actuators):
        if motor.get("joint") not in JOINTS:
            actuators.remove(motor)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nv == 4 and model.nu == 4
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    if not damping:
        model.dof_damping[:] = 0.0
    return model


# Gravity compensation on the slider: holding still needs F3 = -mu g.
GRAVITY_HOLD = np.array([0.0, 0.0, -DYNAMICS.mu * DYNAMICS.g, 0.0])

# Forward-mode default: tau = A cos(w t).  Starting from rest, each joint swings
# out and back, so the arm stays well inside its limits.
Q0_FORWARD = np.array([-0.5, 0.5, 0.02, 0.0])
AMPLITUDE = np.array([0.5, 0.05, 0.25, 0.004])   # Nm, Nm, N, Nm
OMEGA = np.array([1.5 * math.pi, 1.5 * math.pi, math.pi, 2 * math.pi])

# Inverse-mode trajectory: q_d = q0 + B (1 - cos(w t)), which starts at rest.
Q0_INVERSE = np.array([-0.8, 0.6, 0.01, 0.0])
B = np.array([0.7, -0.6, 0.045, 1.2])
W = np.array([1.2, 1.7, 2.0, 2.5])


def desired(t: float):
    q = Q0_INVERSE + B * (1 - np.cos(W * t))
    qd = B * W * np.sin(W * t)
    qdd = B * W**2 * np.cos(W * t)
    return q, qd, qdd


def run(args) -> None:
    model = load_arm_model(args.damping)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    if args.mode == "forward":
        q0 = Q0_FORWARD
        extra = np.array(args.tau) if args.tau else None

        def torque(t):
            if extra is not None:
                return GRAVITY_HOLD + extra
            return GRAVITY_HOLD + AMPLITUDE * np.cos(OMEGA * t)
    else:
        q0 = Q0_INVERSE

        def torque(t):
            # MuJoCo holds ctrl constant over a step; the mid-step torque is the
            # best constant approximation of the continuous profile.
            return inverse_dynamics(*desired(t + dt / 2))

    warned_limit = [False]

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        return q0.copy(), np.zeros(4), 0.0

    def report(t, reference, header=False):
        if header:
            label = "our model" if args.mode == "forward" else "planned"
            print(f"\n{'t [s]':>6} | {'MuJoCo q':^36} | {label + ' q':^36} | max |err|")
        print(f"{t:6.2f} | " + " ".join(f"{x:8.4f}" for x in data.qpos) + " | "
              + " ".join(f"{x:8.4f}" for x in reference)
              + f" | {np.max(np.abs(data.qpos - reference)):.2e}")

    def step(state):
        q_ref, qd_ref, t = state
        tau = torque(t)
        if np.any(tau < lo) or np.any(tau > hi):
            print(f"warning: torque {tau} exceeds actuator limits and will be clipped")
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if data.nefc and not warned_limit[0]:
            warned_limit[0] = True
            print(f"note: a joint limit is active at t={t + dt:.2f} s. MuJoCo now applies a limit "
                  "force that our 4-DOF model doesn't have, so the two motions will diverge.")
        if args.mode == "forward":
            q_ref, qd_ref = rk4_step(q_ref, qd_ref, tau, dt)
        else:
            q_ref, qd_ref, _ = desired(t + dt)
        return q_ref, qd_ref, t + dt

    def run_once(viewer=None):
        state = reset()
        warned_limit[0] = False
        worst = 0.0
        report(0.0, state[0], header=True)
        start = time.perf_counter()
        while state[2] < args.duration - 1e-9:
            if viewer is not None and not viewer.is_running():
                return False
            state = step(state)
            worst = max(worst, float(np.max(np.abs(data.qpos - state[0]))))
            if round(state[2] / dt) % round(0.25 / dt) == 0:
                report(state[2], state[0])
            if viewer is not None:
                viewer.sync()
                lag = state[2] / args.speed - (time.perf_counter() - start)
                if lag > 0:
                    time.sleep(lag)
        print(f"max |MuJoCo - reference| over {args.duration:.1f} s: {worst:.2e} "
              f"(joint order {', '.join(f'{j} [{u}]' for j, u in zip(JOINTS, UNITS))})")
        return True

    if args.headless:
        run_once()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.15, 0.0, 0.2]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.2, 135, -25
        while run_once(viewer):
            print("replaying (close the window to stop)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["forward", "inverse"], default="forward")
    parser.add_argument("--tau", type=float, nargs=4, metavar=("T1", "T2", "F3", "T4"),
                        help="forward mode: constant torques added to gravity compensation")
    parser.add_argument("--duration", type=float, default=4.0, help="seconds per run")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = slow motion)")
    parser.add_argument("--damping", action="store_true", help="keep MuJoCo joint damping (not in our model)")
    parser.add_argument("--headless", action="store_true", help="no viewer; print the comparison only")
    run(parser.parse_args())
