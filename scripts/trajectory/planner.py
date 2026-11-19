"""Synchronized rest-to-rest SCARA trajectories, independent of MuJoCo.

Theory: lectures on trajectory planning. Project-specific timing bounds are
obtained by differentiating the normalized profiles. SI units throughout.
Finite revolute joint ranges are treated as intervals, never wraparound axes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "kinematics"))
from scara_kinematics import inverse_kinematics


def vector(value, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have four finite SI values.")
    return result


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def profile_factors(profile, blend_fraction=1 / 3):
    """Return exact max |ds/du| and max |d2s/du2| on [0,1]."""
    if profile == "cubic":
        return 1.5, 6.0
    if profile == "quintic":
        return 15 / 8, 10 / np.sqrt(3)
    if profile == "lspb":
        if not np.isfinite(blend_fraction) or not 0 < blend_fraction <= 0.5:
            raise ValueError("LSPB blend_fraction must be in (0, 0.5].")
        return 1 / (1 - blend_fraction), 1 / (blend_fraction * (1 - blend_fraction))
    raise ValueError("profile must be cubic, quintic, or lspb.")


@dataclass(frozen=True)
class Limits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        for field in ("lower", "upper", "velocity", "acceleration"):
            value = vector(getattr(self, field), field)
            value.setflags(write=False)
            object.__setattr__(self, field, value)
        if np.any(self.upper <= self.lower):
            raise ValueError("Every upper position limit must exceed its lower limit.")
        if np.any(self.velocity <= 0) or np.any(self.acceleration <= 0):
            raise ValueError("Velocity and acceleration limits must be positive.")

    def check_position(self, q):
        q = vector(q, "joint configuration")
        if np.any(q < self.lower - 1e-12) or np.any(q > self.upper + 1e-12):
            raise ValueError("Joint configuration exceeds position limits.")
        return q


def load_settings():
    robot = json.loads((ROOT / "config" / "robot.json").read_text())
    settings = json.loads((ROOT / "config" / "trajectory" / "defaults.json").read_text())
    v = robot["velocity_limit"]
    bounds = np.array([
        np.deg2rad(robot["shoulder_limit_deg"]),
        np.deg2rad(robot["elbow_limit_deg"]),
        [0, robot["d3_max_m"] - robot["d3_min_m"]],
        np.deg2rad(robot["wrist_limit_deg"]),
    ])
    speed = np.array([v["shoulder_rad_s"], v["elbow_rad_s"],
                      v["vertical_m_s"], v["wrist_rad_s"]])
    scale = positive(settings["velocity_scale"], "velocity_scale")
    if scale > 1:
        raise ValueError("velocity_scale must not exceed the configured robot limits.")
    return Limits(bounds[:, 0], bounds[:, 1], speed * scale,
                  settings["acceleration_limit"]), settings


@dataclass(frozen=True)
class Segment:
    q0: np.ndarray
    qf: np.ndarray
    duration: float
    profile: str = "quintic"
    blend_fraction: float = 1 / 3

    def __post_init__(self):
        for name in ("q0", "qf"):
            value = vector(getattr(self, name), name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "duration", positive(self.duration, "duration"))
        profile_factors(self.profile, self.blend_fraction)

    def evaluate(self, time):
        """Evaluate scalar time, holding position outside the segment.

        Acceleration at the exact cubic/LSPB endpoints is the one-sided
        segment value; outside the interval it is zero. These profiles have
        acceleration discontinuities. Quintic acceleration is continuous.
        """
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite.")
        if time < 0 or time > self.duration:
            return (self.q0.copy() if time < 0 else self.qf.copy(),
                    np.zeros(4), np.zeros(4))
        u = time / self.duration
        if self.profile == "cubic":
            s, sd, sdd = 3*u*u - 2*u**3, 6*u - 6*u*u, 6 - 12*u
        elif self.profile == "quintic":
            s = 10*u**3 - 15*u**4 + 6*u**5
            sd, sdd = 30*u*u*(1-u)**2, 60*u*(1-u)*(1-2*u)
        else:
            b = self.blend_fraction
            a, v = 1 / (b * (1-b)), 1 / (1-b)
            if u < b:
                s, sd, sdd = a*u*u/2, a*u, a
            elif u < 1-b:
                s, sd, sdd = v*(u-b/2), v, 0.0
            else:
                s, sd, sdd = 1-a*(1-u)**2/2, a*(1-u), -a
        delta = self.qf - self.q0
        return self.q0 + delta*s, delta*sd/self.duration, delta*sdd/self.duration**2

    def peak_rates(self):
        fv, fa = profile_factors(self.profile, self.blend_fraction)
        delta = np.abs(self.qf - self.q0)
        return fv*delta/self.duration, fa*delta/self.duration**2


def plan_segment(q0, qf, limits, *, profile="quintic", duration=None,
                 margin=1.2, minimum_duration=0.05, quantum=0.001,
                 blend_fraction=1/3):
    q0, qf = limits.check_position(q0), limits.check_position(qf)
    margin = positive(margin, "margin")
    if margin < 1:
        raise ValueError("Timing margin must be at least one.")
    minimum_duration = positive(minimum_duration, "minimum_duration")
    quantum = positive(quantum, "quantum")
    fv, fa = profile_factors(profile, blend_fraction)
    distance = np.abs(qf - q0)
    required = max(float(np.max(fv*distance/limits.velocity)),
                   float(np.max(np.sqrt(fa*distance/limits.acceleration))))
    if duration is None:
        duration = float(np.ceil(max(minimum_duration, margin*required)/quantum)*quantum)
    else:
        duration = positive(duration, "duration")
        if duration + 1e-12 < required:
            raise ValueError(f"Duration {duration:g}s is infeasible; need at least {required:.6g}s.")
    return Segment(q0, qf, duration, profile, blend_fraction)


class Trajectory:
    """Continuous-position rest-to-rest segments; quintic joins are C2."""
    def __init__(self, segments):
        self.segments = tuple(segments)
        if not self.segments:
            raise ValueError("A trajectory needs at least one segment.")
        for left, right in zip(self.segments, self.segments[1:]):
            if not np.allclose(left.qf, right.q0, atol=1e-12, rtol=0):
                raise ValueError("Adjacent segments must share their endpoint.")
        self.times = np.r_[0.0, np.cumsum([s.duration for s in self.segments])]
        self.times.setflags(write=False)
        self.duration = float(self.times[-1])

    def evaluate(self, time):
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite.")
        index = int(np.clip(np.searchsorted(self.times[1:], time, side="right"),
                            0, len(self.segments)-1))
        return self.segments[index].evaluate(time - self.times[index])


def plan_waypoints(waypoints, limits, *, holds=None, **options):
    points = [limits.check_position(q) for q in waypoints]
    if len(points) < 2:
        raise ValueError("At least two joint waypoints are required.")
    holds = np.zeros(len(points)) if holds is None else np.asarray(holds, dtype=float)
    if holds.shape != (len(points),) or not np.isfinite(holds).all() or np.any(holds < 0):
        raise ValueError("holds must contain one finite nonnegative duration per waypoint.")
    segments = []
    for index, q in enumerate(points):
        if index:
            segments.append(plan_segment(points[index-1], q, limits, **options))
        if holds[index] > 0:
            segments.append(Segment(q, q, holds[index]))
    return Trajectory(segments)


def continuous_ik(targets, seed, limits):
    """Choose nearest valid IK on one elbow branch, with normalized SI distance.

    This converts waypoints only. It does not certify Cartesian interpolation,
    obstacle clearance or continuous IK feasibility between those waypoints.
    """
    previous = limits.check_position(seed)
    if abs(np.sin(previous[1])) < 0.12:
        raise ValueError("Use an IK seed away from the straight-arm singularity.")
    branch = np.sign(previous[1])
    result = []
    for target in targets:
        candidates = [q for q in inverse_kinematics(target)
                      if np.sign(q[1]) == branch and abs(np.sin(q[1])) >= 0.12]
        if not candidates:
            raise ValueError("No nonsingular IK solution on the selected elbow branch.")
        previous = min(candidates, key=lambda q: np.linalg.norm((q-previous)/limits.velocity))
        previous = limits.check_position(previous)
        result.append(previous.copy())
    return result


def validate(trajectory, limits, samples_per_segment=101):
    """Exact rest-to-rest position/rate bounds plus sampled join/derivative data.

    All supported profiles are monotone in normalized progress, so checking
    endpoint positions proves interval position bounds. Collision freedom is
    deliberately not implied by this kinematic validation.
    """
    if not isinstance(samples_per_segment, int) or samples_per_segment < 2:
        raise ValueError("samples_per_segment must be an integer >= 2.")
    speeds, accelerations = [], []
    min_sine = 1.0
    for segment in trajectory.segments:
        limits.check_position(segment.q0)
        limits.check_position(segment.qf)
        speed, acceleration = segment.peak_rates()
        if np.any(speed > limits.velocity + 1e-10) or np.any(acceleration > limits.acceleration + 1e-10):
            raise ValueError("Trajectory exceeds velocity or acceleration limits.")
        speeds.append(speed)
        accelerations.append(acceleration)
        for time in np.linspace(0, segment.duration, samples_per_segment):
            q, _, _ = segment.evaluate(time)
            min_sine = min(min_sine, abs(float(np.sin(q[1]))))
    return {"duration_s": trajectory.duration,
            "segments": len(trajectory.segments),
            "peak_velocity": np.max(speeds, axis=0).tolist(),
            "peak_acceleration": np.max(accelerations, axis=0).tolist(),
            "sampled_min_abs_sin_elbow": min_sine,
            "position_and_rate_bounds": "passed (analytic for these rest-to-rest profiles)",
            "collision_free": "not certified by kinematic checks"}


HOME = np.array([np.deg2rad(25), np.deg2rad(-65), 0.045, np.deg2rad(20)])
WORKED_GOAL = np.array([np.deg2rad(50), np.deg2rad(-45), 0.065, np.deg2rad(35)])


def example_trajectory(limits=None, settings=None):
    if limits is None or settings is None:
        limits, settings = load_settings()
    targets = [[0.28, -0.10, 0.10, 0.0], [0.28, -0.10, 0.04, 0.0],
               [0.28, -0.10, 0.10, 0.0], [0.26, 0.14, 0.10, 0.35],
               [0.26, 0.14, 0.04, 0.35], [0.26, 0.14, 0.10, 0.35]]
    joints = continuous_ik(targets, HOME, limits)
    return plan_waypoints([HOME, *joints, HOME], limits,
                          holds=[0, 0, 0.25, 0, 0, 0.25, 0, 0],
                          profile=settings["profile"], margin=settings["timing_margin"],
                          minimum_duration=settings["minimum_segment_s"],
                          quantum=settings["time_quantum_s"],
                          blend_fraction=settings["blend_fraction"])
