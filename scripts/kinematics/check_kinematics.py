"""Run deterministic structural checks for steps 3 and 4."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inverse_velocity import inverse_velocity, inverse_velocity_dls
from scara_kinematics import (ConvergenceError, JointLimitError, UnreachableTargetError,
                              forward_kinematics, forward_transform, inverse_kinematics,
                              inverse_kinematics_numerical, is_singular,
                              spatial_jacobian, task_jacobian, wrap_to_pi, PARAMETERS)


def assert_raises(error, function, *args):
    try:
        function(*args)
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}.")


samples = [
    np.array([math.radians(25), math.radians(55), 0.03, math.radians(-45)]),
    np.array([math.radians(-70), math.radians(-60), 0.08, math.radians(95)]),
]
for q in samples:
    target = forward_kinematics(q)
    solutions = inverse_kinematics(target)
    assert len(solutions) == 2, "Both elbow-up and elbow-down solutions are expected."
    assert {math.copysign(1, solution[1]) for solution in solutions} == {-1.0, 1.0}
    for solution in solutions:
        recovered = forward_kinematics(solution)
        assert np.allclose(recovered[:3], target[:3], atol=1e-10)
        assert abs(wrap_to_pi(recovered[3] - target[3])) < 1e-10

    transform = forward_transform(q)
    assert np.allclose(transform[:3, 3], target[:3])
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)

    # Independent standard-DH product: Tz(h) A1 A2 A3 A4.
    def dh(theta, d, a, alpha):
        ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
        return np.array([[ct, -st * ca, st * sa, a * ct],
                         [st, ct * ca, -ct * sa, a * st],
                         [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])

    q1, q2, q3, q4 = q
    world_to_zero = np.eye(4); world_to_zero[2, 3] = PARAMETERS.base_height
    dh_product = (world_to_zero @ dh(q1, 0.0, PARAMETERS.a1, 0.0) @
                  dh(q2, 0.0, PARAMETERS.a2, math.pi) @
                  dh(0.0, PARAMETERS.d3_min + q3, 0.0, 0.0) @
                  dh(q4, PARAMETERS.d4, 0.0, 0.0))
    assert np.allclose(transform, dh_product, atol=1e-12)
    yaw = target[3]
    assert np.allclose(transform[:3, 0], [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-12)
    assert np.allclose(transform[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)

q = samples[0]
analytic = task_jacobian(q)
step = 1e-7
numeric = np.empty((4, 4))
for column in range(4):
    dq = np.zeros(4); dq[column] = step
    delta = forward_kinematics(q + dq) - forward_kinematics(q - dq)
    delta[3] = wrap_to_pi(delta[3])
    numeric[:, column] = delta / (2 * step)
assert np.allclose(analytic, numeric, atol=1e-7)

spatial = spatial_jacobian(q)
assert spatial.shape == (6, 4)
assert np.allclose(spatial[:3], analytic[:3])
assert np.allclose(spatial[3:], [[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [1.0, 1.0, 0.0, -1.0]])

# Compare the spatial Jacobian to a numerical linear and angular velocity.
qdot = np.array([0.7, -0.4, 0.12, 0.9])
center_transform = forward_transform(q)
plus_transform = forward_transform(q + step * qdot)
minus_transform = forward_transform(q - step * qdot)
numeric_linear = (plus_transform[:3, 3] - minus_transform[:3, 3]) / (2 * step)
rotation_rate = ((plus_transform[:3, :3] - minus_transform[:3, :3]) / (2 * step) @
                 center_transform[:3, :3].T)
numeric_angular = np.array([rotation_rate[2, 1], rotation_rate[0, 2], rotation_rate[1, 0]])
assert np.allclose(spatial @ qdot, np.r_[numeric_linear, numeric_angular], atol=1e-7)
assert np.allclose(inverse_velocity(analytic @ qdot, q), qdot, atol=1e-12)
assert_raises(ValueError, inverse_velocity, [0.1, 0.0, 0.0, 0.0], [0, 0, 0.04, 0])

assert is_singular([0, 0, 0.04, 0])
assert is_singular([0, math.pi, 0.04, 0])
assert not is_singular(samples[0])
assert is_singular([0, 0.01, 0.04, 0])   # near-singular counts too, not only exactly 0
assert not is_singular([0, 0.2, 0.04, 0])

# Damped least squares velocity: equals J^-1 away from singularities, stays bounded near them.
assert np.allclose(inverse_velocity_dls(analytic @ qdot, q), qdot, atol=1e-12)
for q2 in (0.0, 1e-3, 0.02):
    near = [0.3, q2, 0.05, 0.0]
    rates = inverse_velocity_dls([0.01, 0.0, 0.0, 0.0], near)
    assert np.all(np.isfinite(rates)) and np.max(np.abs(rates)) < 1.0, rates
assert np.linalg.matrix_rank(task_jacobian([0, 0, 0.04, 0])) < 4
assert np.linalg.matrix_rank(task_jacobian([0, math.pi, 0.04, 0])) < 4
assert_raises(UnreachableTargetError, inverse_kinematics, [0.50, 0.0, 0.15, 0.0])
assert_raises(UnreachableTargetError, inverse_kinematics, [0.05, 0.0, 0.07, 0.0])

# Exact annular and vertical workspace boundaries remain geometrically valid.
assert inverse_kinematics([PARAMETERS.a1 + PARAMETERS.a2, 0.0, 0.07, 0.0])
assert inverse_kinematics([abs(PARAMETERS.a1 - PARAMETERS.a2), 0.0, 0.07, 0.0],
                          enforce_joint_limits=False)
assert inverse_kinematics([0.30, 0.0, 0.125, 0.0])  # q3 = 0
assert inverse_kinematics([0.30, 0.0, 0.015, 0.0])  # q3 = 0.110 m
assert_raises(UnreachableTargetError, inverse_kinematics, [0.30, 0.0, 0.126, 0.0])
assert_raises(UnreachableTargetError, inverse_kinematics, [0.30, 0.0, 0.014, 0.0])

outside_shoulder_limit = forward_kinematics([math.radians(170), 0.0, 0.05, 0.0])
assert_raises(JointLimitError, inverse_kinematics, outside_shoulder_limit)

# Numerical (DLS) IK agrees with the closed form, including starts at the singular home pose.
rng = np.random.default_rng(7)
for _ in range(300):
    q_true = np.array([rng.uniform(*PARAMETERS.shoulder_limits), rng.uniform(*PARAMETERS.elbow_limits),
                       rng.uniform(0.0, PARAMETERS.d3_max - PARAMETERS.d3_min),
                       rng.uniform(-math.pi, math.pi)])
    target = forward_kinematics(q_true)
    for q0 in (None, q_true + rng.normal(0.0, 0.3, 4)):
        numerical = inverse_kinematics_numerical(target, q0)
        error = forward_kinematics(numerical) - target
        error[3] = wrap_to_pi(error[3])
        assert np.max(np.abs(error)) < 1e-8
        assert any(np.allclose(c[:3], numerical[:3], atol=1e-4) and
                   abs(wrap_to_pi(c[3] - numerical[3])) < 1e-4 for c in inverse_kinematics(target))
assert_raises(UnreachableTargetError, inverse_kinematics_numerical, [0.50, 0.0, 0.15, 0.0])
assert_raises(ConvergenceError, inverse_kinematics_numerical, outside_shoulder_limit)

print("PASS: FK/IK covers both elbow branches, workspace boundaries, unreachable targets, and limits.")
print("PASS: DH-product transform orientation, task Jacobian, and 6x4 spatial Jacobian agree numerically.")
print("PASS: inverse velocity recovers joint rates and rejects singular configurations.")
print("PASS: singular and near-singular configurations detected; DLS rates stay bounded.")
print("PASS: numerical DLS IK matches the closed form from random and singular starting poses.")
