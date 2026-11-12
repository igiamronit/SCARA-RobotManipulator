"""Inverse differential kinematics for the four-arm-joint SCARA task."""

from __future__ import annotations

import numpy as np

from scara_kinematics import (PARAMETERS, ScaraParameters, damped_least_squares,
                              is_singular, task_jacobian)


def inverse_velocity(task_velocity, q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return qdot for desired ``[xdot, ydot, zdot, yawdot]``.

    The task Jacobian is square for this R-R-P-R arm.  The direct inverse is
    undefined when the two planar links are collinear (q2 = 0 or pi), and
    impractically large close to it, so this function raises an error when
    ``is_singular`` reports the pose as singular.  Use ``inverse_velocity_dls``
    to keep going through or near a singularity.
    """
    task_velocity = np.asarray(task_velocity, dtype=float)
    if task_velocity.shape != (4,):
        raise ValueError(
            "task_velocity must contain [xdot, ydot, zdot, yawdot].")
    if is_singular(q, parameters):
        raise ValueError("Inverse velocity is undefined at or near the q2 = 0 or pi singularity; "
                         "use inverse_velocity_dls instead.")
    return np.linalg.solve(task_jacobian(q, parameters), task_velocity)


def inverse_velocity_dls(task_velocity, q, parameters: ScaraParameters = PARAMETERS) -> np.ndarray:
    """Return qdot for ``[xdot, ydot, zdot, yawdot]`` using damped least squares.

    Away from singularities this equals ``inverse_velocity``.  Near q2 = 0 or
    pi it keeps joint rates bounded at the cost of a small Cartesian velocity
    error, mainly in the radial direction, which the arm cannot move in there.
    """
    task_velocity = np.asarray(task_velocity, dtype=float)
    if task_velocity.shape != (4,):
        raise ValueError(
            "task_velocity must contain [xdot, ydot, zdot, yawdot].")
    return damped_least_squares(task_jacobian(q, parameters), task_velocity)
