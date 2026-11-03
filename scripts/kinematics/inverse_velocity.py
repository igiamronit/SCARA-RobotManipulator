"""Inverse differential kinematics for the four-arm-joint SCARA task."""

from __future__ import annotations

import numpy as np

from scara_kinematics import PARAMETERS, ScaraParameters, is_singular, task_jacobian


def inverse_velocity(task_velocity, q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return qdot for desired ``[xdot, ydot, zdot, yawdot]``.

    The task Jacobian is square for this R-R-P-R arm.  The direct inverse is
    undefined when the two planar links are collinear (q2 = 0 or pi), so this
    function raises an error at those singular configurations.
    """
    task_velocity = np.asarray(task_velocity, dtype=float)
    if task_velocity.shape != (4,):
        raise ValueError(
            "task_velocity must contain [xdot, ydot, zdot, yawdot].")
    if is_singular(q, parameters):
        raise ValueError("Inverse velocity is undefined at the q2 = 0 or pi singularity.")
    return np.linalg.solve(task_jacobian(q, parameters), task_velocity)
