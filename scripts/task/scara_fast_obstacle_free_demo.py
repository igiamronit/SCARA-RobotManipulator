"""Run the obstacle-free SCARA speed demo in MuJoCo."""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'models' / 'scara_pick_place.xml'
if not SCENE.is_file():
    raise SystemExit(f'Missing MuJoCo task scene: {SCENE}')
sys.path[:0] = [str(ROOT / 'scripts' / name) for name in ('task', 'trajectory')]

import pick_place as task
from planner import Limits, Segment, Trajectory, plan_segment
from scara_kinematics import PARAMETERS, forward_kinematics, inverse_kinematics, task_jacobian
from simulation import MeshClearance, check_motor_layout


class OpenScene(task.Scene):
    """Use the task's contact checks with just the two stands and one block."""

    def __init__(self, model):
        self.model = model
        self.tool = model.site('tool').id
        self.payload = model.geom('workpiece').id
        self.payload_body = model.geom_bodyid[self.payload]
        joint = model.joint('payload_free').id
        self.payload_qpos = int(model.jnt_qposadr[joint])
        self.payload_dof = int(model.jnt_dofadr[joint])
        stands = tuple(model.geom(name).id for name in ('pick_stand', 'place_stand'))
        floor = model.geom('floor').id
        fingers = {model.body(name).id for name in ('left_finger', 'right_finger')}
        self.finger_geoms = {i for i, body in enumerate(model.geom_bodyid) if body in fingers}
        self.finger_boxes = tuple(i for i in sorted(self.finger_geoms)
                                  if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        robot = [i for i, body in enumerate(model.geom_bodyid) if body not in (0, self.payload_body)]
        self.pairs = [(a, b) for a, b, _ in MeshClearance(model).pairs]
        self.pairs.append((model.geom('upper_arm_collision').id,
                           model.geom('forearm_collision').id))
        self.pairs.extend((a, b) for a in robot for b in (*stands, self.payload))
        self.pairs.extend((self.payload, b) for b in (*stands, floor))
        self.minimum = (None, None, None)
        self.obstacle = -1
        self.payload_obstacle_minimum = None


@dataclass
class Window:
    motion: object
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start

    @property
    def q0(self):
        return self.motion.evaluate(self.start)[0]

    @property
    def qf(self):
        return self.motion.evaluate(self.end)[0]

    def evaluate(self, t):
        return self.motion.evaluate(self.start + np.clip(t, 0, self.duration))


@dataclass
class Overlap:
    planar: Segment
    vertical: Segment
    planar_delay: float
    vertical_delay: float

    @property
    def duration(self):
        return max(self.planar_delay + self.planar.duration,
                   self.vertical_delay + self.vertical.duration)

    def evaluate(self, t):
        arm = self.planar.evaluate(t - self.planar_delay)
        vertical = self.vertical.evaluate(t - self.vertical_delay)
        for arm_value, vertical_value in zip(arm, vertical):
            arm_value[2] = vertical_value[2]
        return arm


class Carry:
    duration = .376

    def __init__(self, q0, qf, limits):
        self.q0, self.qf = q0, qf
        self.path = Segment(forward_kinematics(q0), forward_kinematics(qf), self.duration)
        self.bump = np.array([-.009, 0, .032, 0])
        sample_count = int(np.ceil(self.duration / .0005)) + 1
        samples = np.linspace(0, self.duration, sample_count)
        q, qd, qdd = np.array([self.evaluate(t) for t in samples]).transpose(1, 0, 2)
        for bound in (q.min(axis=0), q.max(axis=0)):
            limits.check_position(bound)
        if (np.any(np.max(np.abs(qd), axis=0) > limits.velocity) or
                np.any(np.max(np.abs(qdd), axis=0) > limits.acceleration)):
            raise ValueError('Carry exceeds the configured joint rate limits.')

    def evaluate(self, t):
        if t <= 0 or t >= self.duration:
            q = self.q0 if t <= 0 else self.qf
            return q.copy(), np.zeros(4), np.zeros(4)
        pose, velocity, acceleration = self.path.evaluate(t)
        u = t / self.duration
        r = u * (1 - u)
        pose += 64 * r**3 * self.bump
        velocity += 192 * r**2 * (1 - 2*u) / self.duration * self.bump
        acceleration += 384 * r * ((1 - 2*u)**2 - r) / self.duration**2 * self.bump
        q = next(candidate for candidate in inverse_kinematics(pose) if candidate[1] < 0)
        jacobian = task_jacobian(q)
        qd = np.linalg.solve(jacobian, velocity)
        # Invert x_ddot = J q_ddot + J_dot q_dot.
        shoulder, elbow = q[0], q[0] + q[1]
        acceleration[:2] += PARAMETERS.a1 * qd[0]**2 * np.array([np.cos(shoulder), np.sin(shoulder)])
        acceleration[:2] += PARAMETERS.a2 * (qd[0] + qd[1])**2 * np.array([np.cos(elbow), np.sin(elbow)])
        return q, qd, np.linalg.solve(jacobian, acceleration)


def make_plan(model):
    base, settings = task.load_settings()
    limits = Limits(base.lower, base.upper, base.velocity / float(settings['velocity_scale']),
                    task.TASK_ACCELERATION_LIMIT)
    grip_time = .077
    task.GRIP_RAMP_S, task.SETTLE_S, task.ARM_FREQUENCY_RAD_S = grip_time, .060, 70
    _, phases = task.make_plan(model, transfer_height=.080)
    approach, descend, grasp, lift, _, lower, release, retreat, back, settle = phases

    def retime(phase):
        return plan_segment(phase.segment.q0, phase.segment.qf, limits, margin=1.025)

    inbound = Overlap(retime(approach), retime(descend), 0, .133)
    outbound = Overlap(retime(back), retime(retreat), .041, 0)
    carry = Carry(lift.segment.q0, lower.segment.qf, limits)
    grip_start = round(inbound.duration - .050, 3)
    middle = carry.duration / 2
    open_overlap = .065
    phases = (
        replace(approach, segment=Window(inbound, 0, grip_start)),
        replace(grasp, segment=Window(inbound, grip_start, grip_start + grip_time)),
        replace(lift, segment=Window(carry, 0, middle)),
        replace(lower, segment=Window(carry, middle, carry.duration)),
        replace(release, segment=Window(outbound, open_overlap - grip_time, open_overlap)),
        replace(back, segment=Window(outbound, open_overlap, outbound.duration)),
        settle,
    )
    return Trajectory(phase.segment for phase in phases), phases


def main():
    parser = argparse.ArgumentParser(description='Run the live obstacle-free SCARA demo.')
    parser.add_argument('--headless', action='store_true', help='Run physics without a window')
    parser.add_argument('--cycles', type=int, default=1,
                        help='Number of runs, resetting the block to the pick stand each time')
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('--cycles must be at least 1')

    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.delete(spec.geom('obstacle'))  # Remove it from this in-memory model.
    model = spec.compile()
    check_motor_layout(model)
    task.Scene = OpenScene
    trajectory, phases = make_plan(model)
    peak_speed = np.zeros(6)
    viewer = None
    if not args.headless:
        from view_trajectory import LiveViewer
        viewer = LiveViewer(trajectory)
        view_data = mujoco.MjData(model)

    def observe(model, data, desired_q):
        np.maximum(peak_speed, np.abs(data.qvel[:6]), out=peak_speed)
        if viewer is None:
            return True
        opening = viewer.handle is None
        if opening:
            mujoco.mj_copyData(view_data, model, data)
        else:
            if not viewer.handle.is_running():
                return False
            # The window keeps its own data buffer across independent physics runs.
            with viewer.handle.lock():
                mujoco.mj_copyData(view_data, model, data)
        if data.time == 0:
            viewer.next_frame = 0.0
        running = viewer(model, view_data, desired_q)
        if data.time == 0 and running:
            if opening:
                time.sleep(1)  # Give the window time to appear before the first cycle.
            viewer.started = time.perf_counter()
        return running

    try:
        for cycle in range(1, args.cycles + 1):
            if viewer is not None and viewer.handle is not None and not viewer.handle.is_running():
                return
            peak_speed.fill(0)
            result = task.simulate(model, trajectory, phases, on_step=observe)
            if not result['success']:
                if viewer is not None and result['failure'] == 'Viewer closed before the task finished.':
                    return
                raise SystemExit(f"Cycle {cycle}/{args.cycles}: {result['failure']}")
            if (np.any(peak_speed > [6, 8, .8, 12, .25, .25]) or
                    result['saturated_samples'] or result['placement_error_m'] >= .001):
                raise SystemExit(f'Cycle {cycle}/{args.cycles} exceeded a speed, motor, or placement limit.')
            print(f"Cycle {cycle}/{args.cycles}: {result['simulated_duration_s']:.3f} s  |  "
                  f"Release: {result['events']['release_s']:.3f} s  |  "
                  f"Placement error: {1000*result['placement_error_m']:.3f} mm", flush=True)
            if viewer is not None and viewer.handle.is_running():
                viewer.handle.sync()
                if cycle < args.cycles:
                    time.sleep(.5)  # Hold the placed block briefly before the reset.
        if viewer is not None and viewer.handle.is_running():
            while viewer.handle.is_running():
                time.sleep(.05)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()
