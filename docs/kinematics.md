# Kinematics: implementation notes

The reusable code for steps 3 and 4 is in `scripts/kinematics/scara_kinematics.py`.
It uses the shared values in `config/robot.json` and the arm-joint order
`q = [q1, q2, q3, q4] = [shoulder, elbow, vertical, wrist]`.

## Forward kinematics

Define $q_{12}=q_1+q_2$ and $d_3^*=0.080+q_3$.  The world-frame tool pose is

$$
\begin{bmatrix}x\\y\\z\\\phi\end{bmatrix} =
\begin{bmatrix}
a_1\cos q_1+a_2\cos q_{12}\\
a_1\sin q_1+a_2\sin q_{12}\\
h-d_3^*-d_4\\
q_1+q_2-q_4
\end{bmatrix}.
$$

Here $h=0.280$ m is the world-to-shoulder height.  The negative wrist term
comes from the model's wrist axis pointing downward.  The homogeneous tool
transform also contains the DH frame's flipped $z$ axis:

$$
{}^WT_4 =
\begin{bmatrix}
\cos\phi & \sin\phi & 0 & x\\
\sin\phi &-\cos\phi & 0 & y\\
0&0&-1&z\\0&0&0&1
\end{bmatrix}.
$$

## Inverse kinematics and reachability

For desired pose $(x_d,y_d,z_d,\phi_d)$, let $r=\sqrt{x_d^2+y_d^2}$.  A
geometric solution exists only when

$$|a_1-a_2|\le r\le a_1+a_2,$$

and $d_3^*=h-z_d-d_4$ is within the configured vertical range.  The code
calculates both elbow branches:

$$
q_2=\pm\cos^{-1}\left(\frac{r^2-a_1^2-a_2^2}{2a_1a_2}\right),\quad
q_1=\operatorname{atan2}(y_d,x_d)-\operatorname{atan2}(a_2\sin q_2,a_1+a_2\cos q_2),
$$

$$q_3=d_3^*-0.080,\qquad q_4=q_1+q_2-\phi_d.$$

It raises a clear error for unreachable positions or for solutions excluded by
the model's joint limits.

## Velocity kinematics and singularities

The task Jacobian maps $\dot q$ to $[\dot x,\dot y,\dot z,\dot\phi]^T$:

$$
J=\begin{bmatrix}
-a_1s_1-a_2s_{12}&-a_2s_{12}&0&0\\
a_1c_1+a_2c_{12}& a_2c_{12}&0&0\\
0&0&-1&0\\1&1&0&-1
\end{bmatrix}.
$$

The arm is singular when $a_1a_2\sin q_2=0$, i.e. at $q_2=0$ or $q_2=\pi$
(subject to joint limits).  `spatial_jacobian` provides the matching 6x4
geometric Jacobian when linear and angular velocities are needed separately.

## Run the checks

```powershell
python scripts/kinematics/check_kinematics.py
python scripts/model/check_model.py
```
