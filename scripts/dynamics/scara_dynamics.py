"""Euler-Lagrange forward and inverse dynamics for the four arm joints.

Implements the derivation in ``Handwritten-stuff/Forward_Inverse_Dynamics.pdf``:

    D(q) qdd + C(q, qd) qd + G(q) = tau

Joint vector order is ``[shoulder, elbow, vertical, wrist]`` (same as the
kinematics module).  The vertical coordinate is MuJoCo's slider q3, so the DH
distance is ``d3 = d3_min + q3`` and ``d(d3)/dt = qd3``.  Bodies:

1. upper arm (COM at r1 along link 1),
2. forearm (COM at r2 along link 2),
3. vertical carriage (COM on the vertical axis, yaws with the forearm),
4. wrist + palm + both fingers (COM on the vertical axis, yaws by q1+q2-q4).

Mass properties are read from the MuJoCo model so the equations and the
simulation always describe the same robot.  Joint damping, joint-limit forces
and the gripper slides are outside this 4-DOF model.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kinematics"))
from scara_kinematics import PARAMETERS, ROOT, ScaraParameters, _vector  # noqa: E402

MODEL_PATH = ROOT / "models" / "scara.xml"


@dataclass(frozen=True)
class DynamicParameters:
    """Lumped parameters used by the hand derivation (SI units).

    ``I1..I4`` are the inertias about each body's own vertical COM axis; only
    the z components matter because every body rotates about vertical axes.
    """

    a1: float
    a2: float
    r1: float
    r2: float
    m1: float
    m2: float
    m3: float
    m4: float
    I1: float
    I2: float
    I3: float
    I4: float
    d3_min: float
    g: float

    @property
    def mu(self) -> float:
        """Mass carried by the vertical slider, mu = m3 + m4."""
        return self.m3 + self.m4


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(x) for x in (text or default).split()])


def _inertial(body: ET.Element) -> tuple[float, np.ndarray, float]:
    """Return (mass, COM position in body frame, Izz) of one MJCF body."""
    inertial = body.find("inertial")
    if inertial is None or "diaginertia" not in inertial.attrib or "quat" in inertial.attrib:
        raise ValueError(f"Body {body.get('name')!r} needs an axis-aligned diaginertia.")
    return (float(inertial.get("mass")), _floats(inertial.get("pos"), "0 0 0"),
            float(_floats(inertial.get("diaginertia"), "")[2]))


def load_dynamic_parameters(model_path: Path = MODEL_PATH, grip: float = 0.0,
                            kinematics: ScaraParameters = PARAMETERS) -> DynamicParameters:
    """Extract the derivation's lumped parameters from the MJCF model.

    ``grip`` is the (symmetric) finger opening in metres.  The fingers sit off
    the wrist axis, so they add ``m_f * y_f**2`` to I4 by the parallel-axis
    theorem; their combined COM stays on the axis because they move
    symmetrically.
    """
    root = ET.parse(model_path).getroot()
    bodies = {body.get("name"): body for body in root.iter("body")}

    m1, com1, izz1 = _inertial(bodies["upper_arm"])
    m2, com2, izz2 = _inertial(bodies["forearm"])
    m3, com3, izz3 = _inertial(bodies["vertical_carriage"])
    for name, com in (("upper_arm", com1), ("forearm", com2)):
        if abs(com[1]) > 1e-12:
            raise ValueError(f"The derivation assumes the {name} COM lies on its link x axis.")
    if np.linalg.norm(com3[:2]) > 1e-12:
        raise ValueError("The derivation assumes the carriage COM lies on the vertical axis.")

    # Body 4: wrist/palm plus the fingers rigidly attached at the given grip.
    m4, com4, izz4 = _inertial(bodies["wrist"])
    if np.linalg.norm(com4[:2]) > 1e-12:
        raise ValueError("The derivation assumes the wrist COM lies on the vertical axis.")
    first_moment = np.zeros(2)
    for finger in bodies["wrist"].findall("body"):
        mass, com, izz = _inertial(finger)
        offset = _floats(finger.get("pos"), "0 0 0")[:2] + com[:2]
        joint = finger.find("joint")
        if joint is not None and joint.get("type") == "slide":
            offset = offset + grip * _floats(joint.get("axis"), "")[:2]
        m4 += mass
        izz4 += izz + mass * float(offset @ offset)
        first_moment += mass * offset
    if np.linalg.norm(first_moment) > 1e-12:
        raise ValueError("Wrist + finger COM must lie on the vertical axis (use a symmetric grip).")

    gravity = _floats(root.find("option").get("gravity"), "0 0 -9.81")
    return DynamicParameters(
        a1=kinematics.a1, a2=kinematics.a2, r1=float(com1[0]), r2=float(com2[0]),
        m1=m1, m2=m2, m3=m3, m4=m4, I1=izz1, I2=izz2, I3=izz3, I4=izz4,
        d3_min=kinematics.d3_min, g=float(-gravity[2]))


DYNAMICS = load_dynamic_parameters()


def mass_matrix(q, p: DynamicParameters = DYNAMICS) -> np.ndarray:
    """Return the 4x4 inertia matrix D(q)."""
    c2 = math.cos(_vector(q, 4, "q")[1])
    mu = p.mu
    i234 = p.I2 + p.I3 + p.I4
    d11 = (p.m1 * p.r1**2 + p.I1 + p.m2 * (p.a1**2 + p.r2**2 + 2 * p.a1 * p.r2 * c2)
           + mu * (p.a1**2 + p.a2**2 + 2 * p.a1 * p.a2 * c2) + i234)
    d12 = p.m2 * (p.r2**2 + p.a1 * p.r2 * c2) + mu * (p.a2**2 + p.a1 * p.a2 * c2) + i234
    d22 = p.m2 * p.r2**2 + mu * p.a2**2 + i234
    return np.array([
        [d11, d12, 0.0, -p.I4],
        [d12, d22, 0.0, -p.I4],
        [0.0, 0.0, mu, 0.0],
        [-p.I4, -p.I4, 0.0, p.I4],
    ])


def coriolis_matrix(q, qd, p: DynamicParameters = DYNAMICS) -> np.ndarray:
    """Return C(q, qd) built from the Christoffel symbols c_ijk.

    With h = (m2 a1 r2 + mu a1 a2) sin(q2), only the arm block is non-zero.
    This choice of C makes D_dot - 2C skew-symmetric.
    """
    s2 = math.sin(_vector(q, 4, "q")[1])
    qd1, qd2, _, _ = _vector(qd, 4, "qd")
    h = (p.m2 * p.a1 * p.r2 + p.mu * p.a1 * p.a2) * s2
    return np.array([
        [-h * qd2, -h * (qd1 + qd2), 0.0, 0.0],
        [h * qd1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def gravity_vector(q, p: DynamicParameters = DYNAMICS) -> np.ndarray:
    """Return G(q) = dP/dq.  Only the downward slider feels gravity: -mu g."""
    _vector(q, 4, "q")
    return np.array([0.0, 0.0, -p.mu * p.g, 0.0])


def kinetic_energy(q, qd, p: DynamicParameters = DYNAMICS) -> float:
    """Return K = 1/2 qd^T D(q) qd."""
    qd = _vector(qd, 4, "qd")
    return 0.5 * float(qd @ mass_matrix(q, p) @ qd)


def potential_energy(q, p: DynamicParameters = DYNAMICS) -> float:
    """Return P(q) = -mu g d3, measured from DH frame 0 (constant terms dropped)."""
    q3 = _vector(q, 4, "q")[2]
    return -p.mu * p.g * (p.d3_min + q3)


def inverse_dynamics(q, qd, qdd, p: DynamicParameters = DYNAMICS) -> np.ndarray:
    """Actuator effort for a specified motion: tau = D qdd + C qd + G.

    Returns ``[tau1 (Nm), tau2 (Nm), F3 (N), tau4 (Nm)]`` in the joint senses
    of the model (F3 positive pushes the slider downward).
    """
    qd, qdd = _vector(qd, 4, "qd"), _vector(qdd, 4, "qdd")
    return mass_matrix(q, p) @ qdd + coriolis_matrix(q, qd, p) @ qd + gravity_vector(q, p)


def forward_dynamics(q, qd, tau, p: DynamicParameters = DYNAMICS) -> np.ndarray:
    """Joint accelerations for applied efforts: solve D(q) qdd = tau - C qd - G."""
    qd, tau = _vector(qd, 4, "qd"), _vector(tau, 4, "tau")
    b = tau - coriolis_matrix(q, qd, p) @ qd - gravity_vector(q, p)
    return np.linalg.solve(mass_matrix(q, p), b)


def rk4_step(q, qd, tau, dt: float, p: DynamicParameters = DYNAMICS) -> tuple[np.ndarray, np.ndarray]:
    """Advance one RK4 step holding ``tau`` constant (like a MuJoCo ctrl value)."""
    q, qd = _vector(q, 4, "q"), _vector(qd, 4, "qd")

    def f(x):
        return np.r_[x[4:], forward_dynamics(x[:4], x[4:], tau, p)]

    x = np.r_[q, qd]
    k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
    x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[:4], x[4:]


def simulate(q0, qd0, torque, duration: float, dt: float = 1e-3,
             p: DynamicParameters = DYNAMICS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the forward dynamics with fixed-step RK4.

    ``torque`` is a function ``(t, q, qd) -> tau``.  Returns ``(t, q, qd)``
    arrays with one row per step.  Joint limits are not enforced.
    """
    steps = int(round(duration / dt))
    t = np.arange(steps + 1) * dt
    q = np.empty((steps + 1, 4)); qd = np.empty((steps + 1, 4))
    q[0], qd[0] = _vector(q0, 4, "q0"), _vector(qd0, 4, "qd0")

    def f(time, x):
        position, velocity = x[:4], x[4:]
        return np.r_[velocity, forward_dynamics(position, velocity, torque(time, position, velocity), p)]

    for k in range(steps):
        x = np.r_[q[k], qd[k]]
        k1 = f(t[k], x)
        k2 = f(t[k] + dt / 2, x + dt / 2 * k1)
        k3 = f(t[k] + dt / 2, x + dt / 2 * k2)
        k4 = f(t[k] + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q[k + 1], qd[k + 1] = x[:4], x[4:]
    return t, q, qd
