# Dynamics: implementation notes

The code for step 5 is in `scripts/dynamics/scara_dynamics.py`. It implements the
Euler–Lagrange derivation in `Handwritten-stuff/Forward_Inverse_Dynamics.pdf`
with the same joint order as the kinematics:
$q=[q_1,q_2,q_3,q_4]$ = [shoulder, elbow, vertical, wrist], $d_3=0.080+q_3$.

$$D(q)\ddot q + C(q,\dot q)\dot q + G(q) = \tau,\qquad \tau=[\tau_1,\tau_2,F_3,\tau_4]^T$$

## Bodies and parameters

The parameters are read from `models/scara.xml`, so the equations always match
the simulation.

| Body | Mass | COM | $I_{zz}$ about COM |
| --- | --- | --- | --- |
| 1 upper arm | $m_1=1.10$ kg | $r_1=0.120$ m along link 1 | $I_1=6.0\times10^{-3}$ |
| 2 forearm | $m_2=0.75$ kg | $r_2=0.090$ m along link 2 | $I_2=2.5\times10^{-3}$ |
| 3 vertical carriage | $m_3=0.30$ kg | on the vertical axis | $I_3=1.0\times10^{-4}$ |
| 4 wrist + palm + fingers | $m_4=0.30$ kg | on the vertical axis | $I_4=5.23\times10^{-4}$ (closed grip) |

Body 4 includes both fingers, which are held fixed at the chosen grip. They sit
$0.029+g$ m off the axis, so `load_dynamic_parameters(grip=g)` adds
$m_f(0.029+g)^2$ to $I_4$ using the parallel-axis theorem. Their combined COM
stays on the axis. $\mu=m_3+m_4$ is the mass carried by the slider.

## Equations

$$
D(q)=\begin{bmatrix}d_{11}&d_{12}&0&-I_4\\d_{12}&d_{22}&0&-I_4\\0&0&\mu&0\\-I_4&-I_4&0&I_4\end{bmatrix}
$$

$$
\begin{aligned}
d_{11}&=m_1r_1^2+I_1+m_2(a_1^2+r_2^2+2a_1r_2c_2)+\mu(a_1^2+a_2^2+2a_1a_2c_2)+I_2+I_3+I_4\\
d_{12}&=m_2(r_2^2+a_1r_2c_2)+\mu(a_2^2+a_1a_2c_2)+I_2+I_3+I_4\\
d_{22}&=m_2r_2^2+\mu a_2^2+I_2+I_3+I_4
\end{aligned}
$$

Let $h=(m_2a_1r_2+\mu a_1a_2)\sin q_2$. Then

$$
C(q,\dot q)=\begin{bmatrix}-h\dot q_2&-h(\dot q_1+\dot q_2)&0&0\\h\dot q_1&0&0&0\\0&0&0&0\\0&0&0&0\end{bmatrix},
\qquad G(q)=\begin{bmatrix}0\\0\\-\mu g\\0\end{bmatrix}.
$$

$G_3$ is negative because $d_3$ points downward: holding the carriage still needs
$F_3=-\mu g\approx-5.9$ N, an upward push.

- **Inverse dynamics:** `inverse_dynamics(q, qd, qdd)` returns $\tau=D\ddot q+C\dot q+G$.
- **Forward dynamics:** `forward_dynamics(q, qd, tau)` solves $D\ddot q=\tau-C\dot q-G$.
- `simulate(...)` integrates the forward dynamics with fixed-step RK4.

The model leaves out joint damping, joint-limit forces and the gripper slide
dynamics. MuJoCo adds the first two as passive and constraint forces.

## Corrections to the handwritten notes

The final $D$, $C$ and $G$ in the notes are correct. A few intermediate lines
have slips, which the code does not repeat:

- $J_{v_3}$ and $J_{v_4}$ need a third row of $[0,0,-1,0]$ (the $\dot d_3$ term). Also
  $J_{v_4}=J_{v_3}$, because both COMs lie on the vertical axis. The notes then
  use $v_{c_3}^2=v_{c_4}^2$ including $\dot d_3^2$, which is consistent with this.
- $J_{\omega_4}$ has a third row of $[1,1,0,-1]$, matching $\omega_4=\dot\theta_1+\dot\theta_2-\dot\theta_4$.
- In the first line for $K_\omega$, the factor $(I_1+I_2)$ should be $(I_2+I_3)$, as written in $K_{total}$.

## Notebook and viewers

[notebooks/SCARA_Dynamics_MuJoCo.ipynb](../notebooks/SCARA_Dynamics_MuJoCo.ipynb)
walks through the derivation step by step and checks each result against MuJoCo.
Its last cell opens an interactive viewer. In it you set joint efforts with the
keyboard. MuJoCo simulates the robot while a green ghost shows the motion our
equations predict. `scripts/dynamics/view_dynamics.py` provides the same comparison as a
command-line script.

## Run the checks

```powershell
python scripts/dynamics/check_dynamics.py
```

The script checks that $D$ is symmetric positive definite. It checks that $C$
equals the Christoffel-symbol construction, that $\dot D-2C$ is skew-symmetric,
and that $G=\partial P/\partial q$. It compares $D$, $C\dot q+G$ and $\tau$ with
MuJoCo's `qM`, `qfrc_bias` and `mj_inverse` at three grip openings, and they
agree to $10^{-10}$. It also confirms that forward dynamics inverts inverse
dynamics and that an unforced simulation conserves energy.
