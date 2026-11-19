"""Eye-in-hand image-based visual servoing (IBVS) for the SCARA pick.

Follows AI4000 Lectures 34 and 36:

  camera model   u = lam*x/z + cx,  v = lam*y/z + cy   (x right, y down, z forward)
  features       s = (u1, v1, ..., u4, v4), the block's 4 top corners  (k = 8)
  camera motion  xi_c = (vx, vy, vz, wz)                                (m = 4)
  control law    e = s - s*,  xi_c = -gamma (L^T L)^-1 L^T e
  to the joints  xi_4 = [R  S(d)R; 0  R] xi_c,  xi_0 = diag(R04, R04) xi_4,
                 qdot = J(q)^-1 xi_0   (frame 4 is our tool frame, the lecture's frame 6)

After the image error is below EPSILON_PX the arm descends with the vertical
joint only and the pick-and-place plan in pick_place.py takes over.

    python scripts/task/ibvs.py --teach     # record s* (once)
    python scripts/task/ibvs.py             # look, servo, pick and place
    python scripts/task/ibvs.py --view
"""
from __future__ import annotations

import argparse
import json
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pick_place import (ARM_FREQUENCY_RAD_S, GRASP_OFFSET, OPEN, TASK_ACCELERATION_LIMIT,
                        TRANSFER_HEIGHT, load_scene, make_plan, simulate, site_yaw)
from planner import HOME, ROOT, Limits, continuous_ik, load_settings, plan_segment
from scara_kinematics import forward_kinematics, forward_transform
from inverse_velocity import inverse_velocity
from simulation import computed_torque

GOAL_FILE = ROOT / "config" / "vision" / "ibvs_goal.json"
CAMERA = "wrist_cam"
WIDTH, HEIGHT = 640, 480

# Camera pose in tool frame 4: on the tool axis, 17 mm above the tool point, turned -90 deg about z4.
HAND_EYE_ANGLE = -np.pi/2
HAND_EYE_OFFSET = np.array([0, 0, -.017])

LOOK_POSE = (.29, -.09, .12, .25)     # tool x, y, z, yaw; the block only has to be in view
GOAL_TOOL_HEIGHT = .105               # pre-grasp tool height where s* is taught

GAMMA = 2.0                           # 1/s
CAMERA_PERIOD_S = 1/30
MAX_SPEED = .10                       # m/s
MAX_YAW_RATE = 1.0                    # rad/s
EPSILON_PX = 1.0
MAX_SERVO_S = 8.0
MIN_BLOCK_PIXELS = 500


class WristCamera:
    """Colour and depth images of the wrist camera."""

    def __init__(self, model):
        self.renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0      # the red tool-site marker sits in front of the lens

    def capture(self, data):
        self.renderer.update_scene(data, camera=CAMERA, scene_option=self.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=CAMERA, scene_option=self.option)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def close(self):
        self.renderer.close()


def intrinsics(model):
    """Focal length lam in pixels and principal point (cx, cy)."""
    fovy = np.radians(model.cam_fovy[model.camera(CAMERA).id])
    return (HEIGHT/2) / np.tan(fovy/2), WIDTH/2, HEIGHT/2


def block_top_face(rgb, depth):
    """Pixels of the block's top face: yellow, and the closest yellow surface."""
    r, g, b = (rgb[..., i] / 255 for i in range(3))
    yellow = (r - b > .3) & (g - b > .2)
    if yellow.sum() < MIN_BLOCK_PIXELS:
        return None
    return yellow & (depth < depth[yellow].min() + .001)


def corners(face):
    """The 4 corners of the rectangular face, from its centroid and second moments.

    A uniform rectangle n pixels long has variance (n^2 - 1)/12 along that side,
    so its half-length is sqrt(3*variance + 1/4).
    """
    rows, cols = np.nonzero(face)
    points = np.c_[cols, rows] + .5
    centre = points.mean(axis=0)
    variance, axes = np.linalg.eigh(np.cov(points.T))
    long_axis = axes[:, 1]
    short_axis = np.array([-long_axis[1], long_axis[0]])
    a, b = np.sqrt(3*variance[::-1] + .25)
    return np.array([centre + i*a*long_axis + j*b*short_axis
                     for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def match_order(s, s_star):
    """A rectangle looks the same after a half turn: use the corner order closest to s*."""
    turned = np.roll(s, 2, axis=0)
    return s if np.linalg.norm(s - s_star) <= np.linalg.norm(turned - s_star) else turned


def observe(camera, data):
    """Return (corners s, depth z, colour image), or None if the block is not fully in view."""
    rgb, depth = camera.capture(data)
    face = block_top_face(rgb, depth)
    if face is None or face[0].any() or face[-1].any() or face[:, 0].any() or face[:, -1].any():
        return None
    # The top face is parallel to the image plane, so all 4 corners share one depth.
    return corners(face), float(np.median(depth[face])), rgb


def draw_features(rgb, s, s_star):
    """Camera image with the current corners s (red) and goal s* (green)."""
    picture = Image.fromarray(rgb).convert("RGB")
    pen = ImageDraw.Draw(picture)
    pen.polygon([tuple(p) for p in s_star], outline=(30, 160, 60), width=3)
    pen.polygon([tuple(p) for p in s], outline=(210, 40, 40), width=3)
    for (u, v), (us, vs) in zip(s, s_star):
        pen.line([(u, v), (us, vs)], fill=(20, 20, 20), width=2)
        pen.ellipse([us - 7, vs - 7, us + 7, vs + 7], outline=(30, 160, 60), width=3)
        pen.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(210, 40, 40))
    return picture


def interaction_matrix(s, z, lam, cx, cy):
    """Stack L_p for each corner, keeping the columns vx, vy, vz, wz."""
    rows = []
    for u, v in s:
        ub, vb = u - cx, v - cy
        rows.append([-lam/z, 0, ub/z, vb])
        rows.append([0, -lam/z, vb/z, -ub])
    return np.array(rows)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skew(d):
    return np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def camera_to_joint_rates(xi_c, q, hand_eye_angle=HAND_EYE_ANGLE):
    """Camera velocity -> tool frame -> world -> joint rates."""
    vx, vy, vz, wz = xi_c
    xi_cam = np.array([vx, vy, vz, 0, 0, wz])
    R, d = rot_z(hand_eye_angle), HAND_EYE_OFFSET
    xi_4 = np.block([[R, skew(d) @ R], [np.zeros((3, 3)), R]]) @ xi_cam
    R04 = forward_transform(q)[:3, :3]
    xi_0 = np.r_[R04 @ xi_4[:3], R04 @ xi_4[3:]]
    return inverse_velocity(np.r_[xi_0[:3], xi_0[5]], q)


def servo_command(s, s_star, z, q, lam, cx, cy, gamma=GAMMA, hand_eye_angle=HAND_EYE_ANGLE):
    """One IBVS step. Returns joint rates, image error and xi_c."""
    e = (s - s_star).ravel()
    L = interaction_matrix(s, z, lam, cx, cy)
    xi = -gamma * np.linalg.solve(L.T @ L, L.T @ e)
    speed = np.linalg.norm(xi[:3])
    if speed > MAX_SPEED:
        xi[:3] *= MAX_SPEED / speed
    xi[3] = np.clip(xi[3], -MAX_YAW_RATE, MAX_YAW_RATE)
    return camera_to_joint_rates(xi, q, hand_eye_angle), e, xi


def drive(model, data, mass, q_ref, qd_ref, qdd_ref):
    """Computed-torque arm control with the fingers open; call mj_step1 first."""
    arm = computed_torque(model, data, mass, qdd_ref, q_ref - data.qpos[:4],
                          qd_ref - data.qvel[:4], ARM_FREQUENCY_RAD_S, 1)
    fingers = 1000*(OPEN - data.qpos[4:6]) - 12*data.qvel[4:6]
    data.ctrl[:] = np.clip(np.r_[arm, fingers], *model.actuator_ctrlrange.T)
    mujoco.mj_step2(model, data)


def look_segment():
    """Timed move from home (at transfer height) to LOOK_POSE."""
    limits, settings = load_settings()
    limits = Limits(limits.lower, limits.upper, limits.velocity / settings["velocity_scale"],
                    TASK_ACCELERATION_LIMIT)
    home_pose = forward_kinematics(HOME)
    home_pose[2] = TRANSFER_HEIGHT
    home, look = continuous_ik([home_pose, LOOK_POSE], HOME, limits)
    return plan_segment(home, look, limits, margin=settings["timing_margin"],
                        minimum_duration=settings["minimum_segment_s"],
                        quantum=settings["time_quantum_s"])


def follow(model, data, segment, on_step=None):
    mass = np.empty((model.nv, model.nv))
    dt = model.opt.timestep
    for step in range(round(segment.duration / dt) + 1):
        q, qd, qdd = segment.evaluate(step*dt)
        mujoco.mj_step1(model, data)
        if on_step is not None and on_step(model, data, q) is False:
            raise RuntimeError("Viewer closed.")
        drive(model, data, mass, q, qd, qdd)


def servo(model, data, camera, s_star, *, gamma=GAMMA, depth_scale=1.0,
          hand_eye_angle=HAND_EYE_ANGLE, on_step=None, on_frame=None):
    """Run IBVS from the current pose and return a log of every camera frame.

    depth_scale and hand_eye_angle give the controller a wrong depth or camera
    angle for experiments. on_frame(rgb, s, s_star, rms) is called per frame.
    """
    lam, cx, cy = intrinsics(model)
    limits, _ = load_settings()
    dt = model.opt.timestep
    frame_steps = round(CAMERA_PERIOD_S / dt)
    mass = np.empty((model.nv, model.nv))
    q_ref, qd_ref = data.qpos[:4].copy(), np.zeros(4)
    start = float(data.time)
    log = dict(converged=False, failure=None, t=[], rms_px=[], s=[], z=[], xi=[], tool=[],
               first_image=None, last_image=None)

    for step in range(round(MAX_SERVO_S / dt)):
        mujoco.mj_step1(model, data)
        if step % frame_steps == 0:
            seen = observe(camera, data)
            if seen is None:
                log["failure"] = "The block left the camera view."
                break
            s, z, rgb = seen
            s = match_order(s, s_star)
            rms = float(np.sqrt(np.mean(np.sum((s - s_star)**2, axis=1))))
            qd_ref, e, xi = servo_command(s, s_star, depth_scale*z, data.qpos[:4].copy(),
                                          lam, cx, cy, gamma, hand_eye_angle)
            log["t"].append(float(data.time) - start)
            log["rms_px"].append(rms)
            log["s"].append(s)
            log["z"].append(z)
            log["xi"].append(xi)
            log["tool"].append(forward_kinematics(data.qpos[:4]))
            log["first_image"] = rgb if log["first_image"] is None else log["first_image"]
            log["last_image"] = rgb
            if on_frame is not None:
                on_frame(rgb, s, s_star, rms)
            if rms < EPSILON_PX:
                log["converged"] = True
                break
        q_ref = q_ref + qd_ref*dt
        if np.any(q_ref < limits.lower) or np.any(q_ref > limits.upper):
            log["failure"] = "Visual servoing reached a joint limit."
            break
        if on_step is not None and on_step(model, data, q_ref) is False:
            log["failure"] = "Viewer closed."
            break
        drive(model, data, mass, q_ref, qd_ref, np.zeros(4))
    else:
        log["failure"] = f"No convergence within {MAX_SERVO_S} s."
    return log


def teach(model):
    """Teach by showing: record s* with the camera at the pre-grasp pose."""
    pick = model.site("pick_target")
    limits, _ = load_settings()
    q = continuous_ik([[pick.pos[0], pick.pos[1], GOAL_TOOL_HEIGHT, site_yaw(pick)]], HOME, limits)[0]
    data = mujoco.MjData(model)
    data.qpos[:4], data.qpos[4:6] = q, OPEN
    mujoco.mj_forward(model, data)
    camera = WristCamera(model)
    seen = observe(camera, data)
    camera.close()
    if seen is None:
        raise ValueError("The block is not fully visible from the pre-grasp pose.")
    s_star, z_star, _ = seen
    return {
        "s_star_px": np.round(s_star, 3).tolist(),
        "z_star_m": round(z_star, 6),
        "tool_height_m": GOAL_TOOL_HEIGHT,
        "descent_m": round(GOAL_TOOL_HEIGHT - (float(pick.pos[2]) + GRASP_OFFSET), 6),
        "image_size_px": [WIDTH, HEIGHT],
    }


def load_goal():
    if not GOAL_FILE.exists():
        raise ValueError(f"{GOAL_FILE} is missing. Run: python scripts/task/ibvs.py --teach")
    return json.loads(GOAL_FILE.read_text())


def shift_block(model, dx=0., dy=0., dyaw=0.):
    """Move the block's starting pose without telling the robot."""
    adr = model.jnt_qposadr[model.joint("payload_free").id]
    model.qpos0[adr:adr + 2] += [dx, dy]
    turn = np.array([np.cos(dyaw/2), 0, 0, np.sin(dyaw/2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, turn, model.qpos0[adr + 3:adr + 7].copy())
    model.qpos0[adr + 3:adr + 7] = quat


def run(model, goal, *, on_step=None, **servo_options):
    """Look, servo, then descend and pick and place. Returns (report, servo log)."""
    data = mujoco.MjData(model)
    approach = look_segment()
    data.qpos[:4], data.qpos[4:6] = approach.q0, OPEN
    follow(model, data, approach, on_step)

    camera = WristCamera(model)
    try:
        log = servo(model, data, camera, np.array(goal["s_star_px"]), on_step=on_step,
                    **servo_options)
    finally:
        camera.close()
    servo_report = {
        "converged": log["converged"],
        "time_s": log["t"][-1] if log["t"] else None,
        "frames": len(log["t"]),
        "initial_rms_px": log["rms_px"][0] if log["rms_px"] else None,
        "final_rms_px": log["rms_px"][-1] if log["rms_px"] else None,
    }
    if not log["converged"]:
        return {"success": False, "failure": log["failure"], "servo": servo_report}, log

    q = data.qpos[:4].copy()
    tool = forward_kinematics(q)
    pick_pose = [tool[0], tool[1], tool[2] - goal["descent_m"], tool[3]]
    # Evaluation only: the block's true pose is not used for control.
    block = model.body("payload").id
    rotation = data.xmat[block].reshape(3, 3)
    yaw_error = (pick_pose[3] - np.arctan2(rotation[1, 0], rotation[0, 0]) + np.pi/2) % np.pi - np.pi/2
    servo_report["grasp_xy_error_mm"] = 1000*float(np.linalg.norm(
        np.array(pick_pose[:2]) - data.xpos[block][:2]))
    servo_report["grasp_yaw_error_deg"] = float(np.degrees(abs(yaw_error)))

    trajectory, phases = make_plan(model, start_q=q, pick_pose=pick_pose)
    report = simulate(model, trajectory, phases, on_step=on_step, data=data)
    report["servo"] = servo_report
    return report, log


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teach", action="store_true", help="Record the goal image s* and exit")
    parser.add_argument("--view", action="store_true", help="Watch in the MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1, help="Viewer playback speed")
    parser.add_argument("--block-shift", type=float, nargs=3, default=(0, 0, 0),
                        metavar=("DX", "DY", "DYAW"), help="Move the block (m, m, rad)")
    args = parser.parse_args()
    model = load_scene()
    if args.teach:
        GOAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOAL_FILE.write_text(json.dumps(teach(model), indent=2) + "\n")
        print(f"Saved the goal image features to {GOAL_FILE}")
        return 0

    shift_block(model, *args.block_shift)
    viewer = None
    on_frame = None
    try:
        if args.view:
            from view_trajectory import LiveViewer
            from planner import Trajectory
            viewer = LiveViewer(Trajectory([look_segment()]), args.speed)

            def show_camera(rgb, s, s_star, rms):
                if viewer.handle is None:
                    return
                w, h = WIDTH // 2, HEIGHT // 2
                picture = np.asarray(draw_features(rgb, s, s_star).resize((w, h)))
                viewer.handle.set_images((mujoco.MjrRect(0, 0, w, h), picture))
                viewer.handle.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                         "Wrist camera: red = s, green = s*",
                                         f"RMS error {rms:6.1f} px"))
            on_frame = show_camera
        report, _ = run(model, load_goal(), on_step=viewer, on_frame=on_frame)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
