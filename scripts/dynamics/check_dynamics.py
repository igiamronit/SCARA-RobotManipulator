"""Deterministic checks for step 5: the Euler-Lagrange model in scara_dynamics.py.

1. D(q) is symmetric positive definite.
2. C(q, qd) equals the Christoffel-symbol construction from numerical dD/dq,
   and D_dot - 2C is skew-symmetric.
3. G(q) equals the numerical gradient of P(q).
4. D, C qd + G and inverse dynamics agree with MuJoCo (qM, qfrc_bias,
   mj_inverse) with damping and constraints disabled.
5. forward_dynamics inverts inverse_dynamics.
6. Unforced RK4 simulation conserves total energy K + P.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scara_dynamics import (MODEL_PATH, coriolis_matrix, forward_dynamics, gravity_vector,  # noqa: E402
                            inverse_dynamics, kinetic_energy, load_dynamic_parameters,
                            mass_matrix, potential_energy, simulate)

rng = np.random.default_rng(5)
STEP = 1e-6


def random_state():
    q = np.array([rng.uniform(-2.5, 2.5), rng.uniform(-2.4, 2.4),
                  rng.uniform(0.0, 0.11), rng.uniform(-3.0, 3.0)])
    return q, rng.normal(size=4), rng.normal(size=4)


def dD_dq(q, p):
    """Numerical partial derivatives; result[k] = dD/dq_k."""
    out = np.empty((4, 4, 4))
    for k in range(4):
        dq = np.zeros(4); dq[k] = STEP
        out[k] = (mass_matrix(q + dq, p) - mass_matrix(q - dq, p)) / (2 * STEP)
    return out


def christoffel_C(q, qd, p):
    """C_kj = sum_i c_ijk qd_i, c_ijk = 1/2 (dd_kj/dq_i + dd_ki/dq_j - dd_ij/dq_k)."""
    dD = dD_dq(q, p)
    C = np.zeros((4, 4))
    for k in range(4):
        for j in range(4):
            for i in range(4):
                C[k, j] += 0.5 * (dD[i][k, j] + dD[j][k, i] - dD[k][i, j]) * qd[i]
    return C


# --- 1-3: structural properties of D, C, G -----------------------------------
p = load_dynamic_parameters()
for _ in range(20):
    q, qd, _ = random_state()
    D = mass_matrix(q, p)
    assert np.allclose(D, D.T, atol=1e-15)
    assert np.linalg.eigvalsh(D).min() > 0, "D(q) must be positive definite."

    C = coriolis_matrix(q, qd, p)
    assert np.allclose(C, christoffel_C(q, qd, p), atol=1e-8)
    D_dot = np.einsum("kij,k->ij", dD_dq(q, p), qd)
    N = D_dot - 2 * C
    assert np.allclose(N, -N.T, atol=1e-8), "D_dot - 2C must be skew-symmetric."

    grad_P = np.array([(potential_energy(q + STEP * e, p) - potential_energy(q - STEP * e, p))
                       / (2 * STEP) for e in np.eye(4)])
    assert np.allclose(gravity_vector(q, p), grad_P, atol=1e-6)

# --- 4: compare with MuJoCo's rigid-body algorithms ----------------------------
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
model.dof_damping[:] = 0.0
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
data = mujoco.MjData(model)
full_M = np.empty((model.nv, model.nv))
for grip in (0.0, 0.009, 0.018):
    p_grip = load_dynamic_parameters(grip=grip)
    for _ in range(10):
        q, qd, qdd = random_state()
        data.qpos[:] = np.r_[q, grip, grip]
        data.qvel[:] = np.r_[qd, 0.0, 0.0]
        data.qacc[:] = np.r_[qdd, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, full_M)
        assert np.allclose(mass_matrix(q, p_grip), full_M[:4, :4], atol=1e-10)
        bias = coriolis_matrix(q, qd, p_grip) @ qd + gravity_vector(q, p_grip)
        assert np.allclose(bias, data.qfrc_bias[:4], atol=1e-10)

        data.qacc[:] = np.r_[qdd, 0.0, 0.0]
        mujoco.mj_inverse(model, data)
        assert np.allclose(inverse_dynamics(q, qd, qdd, p_grip), data.qfrc_inverse[:4], atol=1e-10)

# --- 5: forward/inverse round trip ---------------------------------------------
for _ in range(20):
    q, qd, qdd = random_state()
    tau = inverse_dynamics(q, qd, qdd, p)
    assert np.allclose(forward_dynamics(q, qd, tau, p), qdd, atol=1e-10)

# Static holding effort: only the slider needs force, F3 = -mu g (push upward).
hold = inverse_dynamics([0.3, -0.7, 0.05, 1.0], np.zeros(4), np.zeros(4), p)
assert np.allclose(hold, [0.0, 0.0, -p.mu * p.g, 0.0])

# --- 6: energy conservation with tau = 0 ---------------------------------------
q0 = np.array([0.2, math.radians(60), 0.05, 0.0])
qd0 = np.array([1.5, -2.0, 0.3, 4.0])
t, qs, qds = simulate(q0, qd0, lambda *_: np.zeros(4), duration=1.0, dt=1e-3, p=p)
energy = np.array([kinetic_energy(a, b, p) + potential_energy(a, p) for a, b in zip(qs, qds)])
drift = np.max(np.abs(energy - energy[0]))
assert drift < 1e-8 * max(1.0, abs(energy[0])), drift

print(f"Parameters: r1={p.r1:.3f} m, r2={p.r2:.3f} m, mu=m3+m4={p.mu:.3f} kg, "
      f"I1..I4=({p.I1:.2e}, {p.I2:.2e}, {p.I3:.2e}, {p.I4:.2e}) kg m^2")
print("PASS: D(q) symmetric positive definite; C from Christoffel symbols; D_dot - 2C skew-symmetric.")
print("PASS: G(q) = dP/dq; static holding effort F3 = -mu g.")
print("PASS: D, C qd + G and inverse dynamics match MuJoCo qM / qfrc_bias / mj_inverse at 3 grip openings.")
print(f"PASS: forward dynamics inverts inverse dynamics; unforced energy drift {drift:.1e} J over 1 s.")
