"""Forward/inverse kinematics and Jacobians for the project's four arm joints.

Joint vector order is ``[shoulder, elbow, vertical, wrist]``.  The vertical
coordinate is MuJoCo's slider coordinate q3, not the DH distance d3*.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def find_project_root(source_file: Path) -> Path:
    """Find the repository containing the shared robot configuration."""
    for candidate in (source_file.parent, *source_file.parents):
        if (candidate / "config" / "robot.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find config/robot.json. Keep this module inside the SCARA repository "
        "or call load_parameters() with an explicit configuration path.")


ROOT = find_project_root(Path(__file__).resolve())


class UnreachableTargetError(ValueError):
    """Raised when a Cartesian target lies outside the geometric workspace."""


class JointLimitError(ValueError):
    """Raised when a geometrically reachable target violates a joint limit."""


@dataclass(frozen=True)
class ScaraParameters:
    """Dimensions and arm-joint limits shared with the MuJoCo model."""

    base_height: float
    a1: float
    a2: float
    d3_min: float
    d3_max: float
    d4: float
    shoulder_limits: tuple[float, float]
    elbow_limits: tuple[float, float]
    wrist_limits: tuple[float, float]


def load_parameters(path: Path = ROOT / "config" / "robot.json") -> ScaraParameters:
    """Load the one source of truth for dimensions and joint limits."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ScaraParameters(
        base_height=raw["base_height_m"], a1=raw["a1_m"], a2=raw["a2_m"],
        d3_min=raw["d3_min_m"], d3_max=raw["d3_max_m"], d4=raw["d4_m"],
        shoulder_limits=tuple(math.radians(x) for x in raw["shoulder_limit_deg"]),
        elbow_limits=tuple(math.radians(x) for x in raw["elbow_limit_deg"]),
        wrist_limits=tuple(math.radians(x) for x in raw["wrist_limit_deg"]),
    )


PARAMETERS = load_parameters()


def _vector(values, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values; got shape {vector.shape}.")
    return vector


def wrap_to_pi(angle: float) -> float:
    """Return an equivalent angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def forward_kinematics(q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return tool pose ``[x, y, z, yaw]`` in the project world frame.

    ``yaw`` is the tool's vertical-axis orientation: q1 + q2 - q4.  The
    minus q4 follows the model's wrist hinge axis, which points downward.
    """
    q1, q2, q3, q4 = _vector(q, 4, "q")
    q12 = q1 + q2
    d3_star = parameters.d3_min + q3
    return np.array([
        parameters.a1 * math.cos(q1) + parameters.a2 * math.cos(q12),
        parameters.a1 * math.sin(q1) + parameters.a2 * math.sin(q12),
        parameters.base_height - d3_star - parameters.d4,
        wrap_to_pi(q12 - q4),
    ])


def forward_transform(q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return the standard-DH tool transform in the project world frame."""
    x, y, z, yaw = forward_kinematics(q, parameters)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0, x], [s, -c, 0.0, y], [0.0, 0.0, -1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def task_jacobian(q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return d[x, y, z, yaw]/d[q1, q2, q3, q4]."""
    q1, q2, _, _ = _vector(q, 4, "q")
    q12 = q1 + q2
    s1, c1, s12, c12 = math.sin(q1), math.cos(q1), math.sin(q12), math.cos(q12)
    return np.array([
        [-parameters.a1 * s1 - parameters.a2 * s12, -parameters.a2 * s12, 0.0, 0.0],
        [ parameters.a1 * c1 + parameters.a2 * c12,  parameters.a2 * c12, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 1.0, 0.0, -1.0],
    ])


def spatial_jacobian(q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return the 6x4 geometric Jacobian [linear velocity; angular velocity]."""
    task = task_jacobian(q, parameters)
    angular = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, -1.0]])
    return np.vstack((task[:3], angular))


def is_singular(q, parameters: ScaraParameters = PARAMETERS, tolerance: float = 1e-9) -> bool:
    """Check the arm singularity sin(q2)=0 (fully folded or extended arm)."""
    q2 = _vector(q, 4, "q")[1]
    return abs(parameters.a1 * parameters.a2 * math.sin(q2)) <= tolerance


def _within(value: float, limits: tuple[float, float], tolerance: float = 1e-9) -> bool:
    return limits[0] - tolerance <= value <= limits[1] + tolerance


def inverse_kinematics(target, parameters: ScaraParameters = PARAMETERS,
                       enforce_joint_limits: bool = True) -> list[np.ndarray]:
    """Return elbow-up/down joint solutions for ``target=[x,y,z,yaw]``.

    Raises ``UnreachableTargetError`` for a point outside the annular or
    vertical workspace.  Raises ``JointLimitError`` when geometry permits the
    target but no solution satisfies this project's configured arm limits.
    """
    x, y, z, yaw = _vector(target, 4, "target")
    radius = math.hypot(x, y)
    minimum, maximum = abs(parameters.a1 - parameters.a2), parameters.a1 + parameters.a2
    if radius < minimum - 1e-9 or radius > maximum + 1e-9:
        raise UnreachableTargetError(
            f"Horizontal radius {radius:.6f} m is outside [{minimum:.6f}, {maximum:.6f}] m.")

    d3_star = parameters.base_height - z - parameters.d4
    q3 = d3_star - parameters.d3_min
    q3_limits = (0.0, parameters.d3_max - parameters.d3_min)
    if not _within(q3, q3_limits):
        raise UnreachableTargetError(
            f"Vertical slider q3={q3:.6f} m is outside [{q3_limits[0]:.6f}, {q3_limits[1]:.6f}] m.")

    cosine_q2 = (radius**2 - parameters.a1**2 - parameters.a2**2) / (2 * parameters.a1 * parameters.a2)
    cosine_q2 = float(np.clip(cosine_q2, -1.0, 1.0))  # protects a target on a numerical boundary
    magnitude = math.acos(cosine_q2)
    solutions = []
    for q2 in (magnitude, -magnitude):
        q1 = math.atan2(y, x) - math.atan2(parameters.a2 * math.sin(q2),
                                            parameters.a1 + parameters.a2 * math.cos(q2))
        q1, q2 = wrap_to_pi(q1), wrap_to_pi(q2)
        q4 = wrap_to_pi(q1 + q2 - yaw)
        candidate = np.array([q1, q2, q3, q4])
        limits_ok = (_within(q1, parameters.shoulder_limits) and
                     _within(q2, parameters.elbow_limits) and
                     _within(q4, parameters.wrist_limits))
        if limits_ok or not enforce_joint_limits:
            if not any(np.allclose(candidate, old, atol=1e-10) for old in solutions):
                solutions.append(candidate)

    if not solutions and enforce_joint_limits:
        raise JointLimitError("Target is geometrically reachable but violates the configured joint limits.")
    return solutions
