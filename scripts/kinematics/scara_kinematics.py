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


class ConvergenceError(ValueError):
    """Raised when the numerical IK solver does not reach the target."""


# |sin(q2)| below this counts as singular for velocity kinematics.  The smallest
# singular value of the task Jacobian is roughly 0.085*|sin(q2)| for this arm,
# so 0.05 (about 2.9 degrees) is where J^-1 starts to demand very large rates.
SINGULARITY_SIN_TOLERANCE = 0.05

# Damped least squares: damping switches on smoothly once the smallest singular
# value of J falls below DLS_THRESHOLD (q2 within roughly 7 degrees of 0 or pi).
DLS_THRESHOLD = 0.01
DLS_MAX_DAMPING = 0.01


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


def is_singular(q, parameters: ScaraParameters = PARAMETERS,
                tolerance: float = SINGULARITY_SIN_TOLERANCE) -> bool:
    """Check whether the arm is at or near sin(q2)=0 (fully extended or folded).

    det(J) = a1*a2*sin(q2), so the exact singularity is q2 = 0 or pi.  A pose
    with |sin(q2)| <= ``tolerance`` is treated as singular, because nearby the
    exact inverse J^-1 needs very large joint rates for small Cartesian rates.
    """
    q2 = _vector(q, 4, "q")[1]
    return abs(math.sin(q2)) <= tolerance


def damped_least_squares(jacobian, vector, threshold: float = DLS_THRESHOLD,
                         max_damping: float = DLS_MAX_DAMPING) -> np.ndarray:
    """Solve J x = vector with adaptive damped least squares.

    Returns J^T (J J^T + lambda^2 I)^-1 vector.  lambda is zero while the
    smallest singular value of J is at least ``threshold`` (so the result
    equals the exact J^-1 solution there) and rises smoothly to
    ``max_damping`` at the singularity:
    lambda^2 = max_damping^2 * (1 - (sigma_min / threshold)^2).
    Near a singularity this trades a small tracking error for bounded rates.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    vector = np.asarray(vector, dtype=float)
    sigma_min = np.linalg.svd(jacobian, compute_uv=False)[-1]
    damping_sq = 0.0
    if sigma_min < threshold:
        damping_sq = max_damping**2 * (1.0 - (sigma_min / threshold) ** 2)
    rows = jacobian.shape[0]
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_sq * np.eye(rows), vector)


def _within(value: float, limits: tuple[float, float], tolerance: float = 1e-9) -> bool:
    return limits[0] - tolerance <= value <= limits[1] + tolerance


def _check_reachable(x: float, y: float, z: float,
                     parameters: ScaraParameters) -> tuple[float, float]:
    """Return (radius, q3) or raise UnreachableTargetError."""
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
    return radius, q3


def _clip_to_limits(q: np.ndarray, parameters: ScaraParameters) -> np.ndarray:
    """Wrap revolute joints to [-pi, pi), then clip every joint to its limits.

    Wrapping first matters for the +/-180 degree wrist: its limits are a full
    circle, so a step past +pi must come back in at -pi, not stick at +pi.
    """
    q = q.copy()
    q[0] = np.clip(wrap_to_pi(q[0]), *parameters.shoulder_limits)
    q[1] = np.clip(wrap_to_pi(q[1]), *parameters.elbow_limits)
    q[2] = np.clip(q[2], 0.0, parameters.d3_max - parameters.d3_min)
    q[3] = np.clip(wrap_to_pi(q[3]), *parameters.wrist_limits)
    return q


def inverse_kinematics(target, parameters: ScaraParameters = PARAMETERS,
                       enforce_joint_limits: bool = True) -> list[np.ndarray]:
    """Return elbow-up/down joint solutions for ``target=[x,y,z,yaw]``.

    Raises ``UnreachableTargetError`` for a point outside the annular or
    vertical workspace.  Raises ``JointLimitError`` when geometry permits the
    target but no solution satisfies this project's configured arm limits.
    """
    x, y, z, yaw = _vector(target, 4, "target")
    radius, q3 = _check_reachable(x, y, z, parameters)

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


def _dls_solve(target: np.ndarray, q: np.ndarray, parameters: ScaraParameters,
               enforce_joint_limits: bool, position_tolerance: float, yaw_tolerance: float,
               max_iterations: int, max_step: float) -> tuple[np.ndarray | None, np.ndarray]:
    """Run the DLS iteration from ``q``; return (solution or None, last error)."""
    if enforce_joint_limits:
        q = _clip_to_limits(q, parameters)
    error = np.full(4, np.inf)
    for _ in range(max_iterations):
        error = target - forward_kinematics(q, parameters)
        error[3] = wrap_to_pi(error[3])
        if np.linalg.norm(error[:3]) <= position_tolerance and abs(error[3]) <= yaw_tolerance:
            return np.array([wrap_to_pi(q[0]), wrap_to_pi(q[1]), q[2], wrap_to_pi(q[3])]), error
        damping = min(DLS_MAX_DAMPING, float(np.linalg.norm(error)))
        step = damped_least_squares(task_jacobian(q, parameters), error, max_damping=damping)
        largest = np.max(np.abs(step))
        if largest > max_step:
            step *= max_step / largest
        q = q + step
        if enforce_joint_limits:
            q = _clip_to_limits(q, parameters)
    return None, error


def inverse_kinematics_numerical(target, q0=None, parameters: ScaraParameters = PARAMETERS,
                                 enforce_joint_limits: bool = True,
                                 position_tolerance: float = 1e-9,
                                 yaw_tolerance: float = 1e-9,
                                 max_iterations: int = 200,
                                 max_step: float = 0.5,
                                 singular_nudge: float = 0.3,
                                 restarts: bool = True) -> np.ndarray:
    """Iterative IK using damped least squares; an alternative to the closed form.

    Update rule: q_{k+1} = q_k + DLS(J(q_k), e_k), with e = x_d - f(q) and the
    yaw error wrapped to [-pi, pi).  The maximum damping is also capped at
    |e| (Levenberg-Marquardt style), so damping fades out as the solver
    closes in and the final steps converge like Newton's method even for
    targets right next to the singularity.

    Returns one solution: normally the branch nearer the starting guess
    ``q0`` (default: the zero/home pose).  Unlike the closed form, it does not
    list both branches.

    Notes
    - At q2 = 0 (the home pose) the arm cannot move radially to first order,
      so any target that needs a radius change would leave the solver stuck.
      If q0 is singular, q2 is nudged by ``singular_nudge`` rad first.
    - Steps are capped at ``max_step`` (rad or m) and, when
      ``enforce_joint_limits`` is True, projected back inside the limits.
    - Being a local method, it can get pinned against a joint limit (often
      the +/-150 degree shoulder) when the path from q0 crosses it.  With
      ``restarts`` it then retries from two guesses pointed at the target's
      bearing, one per elbow branch.
    - Raises UnreachableTargetError for targets outside the workspace and
      ConvergenceError if no attempt meets the tolerance.
    """
    target = _vector(target, 4, "target")
    _check_reachable(target[0], target[1], target[2], parameters)
    q = np.zeros(4) if q0 is None else _vector(q0, 4, "q0").copy()
    if is_singular(q, parameters):
        q[1] += math.copysign(singular_nudge, q[1] if q[1] != 0 else 1.0)

    seeds = [q]
    if restarts:
        bearing = math.atan2(target[1], target[0])
        for elbow in (0.8, -0.8):
            seeds.append(np.array([bearing - elbow / 2, elbow, q[2], q[3]]))

    error = np.full(4, np.inf)
    for seed in seeds:
        solution, error = _dls_solve(target, seed, parameters, enforce_joint_limits,
                                     position_tolerance, yaw_tolerance, max_iterations, max_step)
        if solution is not None:
            return solution

    raise ConvergenceError(
        f"Numerical IK did not converge in {max_iterations} iterations "
        f"(position error {np.linalg.norm(error[:3]):.3e} m, yaw error {abs(error[3]):.3e} rad). "
        "Try another q0, or the target may need a joint outside its limits.")
