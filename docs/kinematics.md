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

`inverse_velocity.py` supplies the inverse differential relation for a
non-singular pose: $\dot q=J(q)^{-1}[\dot x,\dot y,\dot z,\dot\phi]^T$.  It
raises an error at or near singular configurations rather than returning
unreliable joint velocities.  "Near" means $|\sin q_2|\le 0.05$ (about
2.9°): the smallest singular value of $J$ is roughly $0.085|\sin q_2|$ here,
so closer than that, $J^{-1}$ needs very large joint rates for small Cartesian
rates.  The physically reachable singularity is $q_2=0$ (arm fully extended,
including the home pose); $q_2=\pi$ lies outside the ±145° elbow limit.

Note that the joint limits shrink the reachable annulus: with $|q_2|\le145°$
the inner radius is about 0.139 m, not $|a_1-a_2|=0.06$ m, and the ±150°
shoulder limit leaves a 60° wedge behind the base unreachable.  The IK raises
`JointLimitError` for such targets.

## Damped least squares (optional alternative)

`inverse_velocity_dls` and `inverse_kinematics_numerical` use adaptive damped
least squares:

$$
\dot q = J^T\left(JJ^T+\lambda^2 I\right)^{-1}\dot x,\qquad
\lambda^2=\begin{cases}0,&\sigma_{\min}\ge\varepsilon\\
\lambda_{\max}^2\left(1-(\sigma_{\min}/\varepsilon)^2\right),&\sigma_{\min}<\varepsilon\end{cases}
$$

with $\varepsilon=0.01$ and $\lambda_{\max}=0.01$.  Away from singularities
this is exactly $J^{-1}$; near them it keeps joint rates bounded and drops the
radial velocity component the arm cannot produce.

`inverse_kinematics_numerical` iterates $q_{k+1}=q_k+\Delta q_k$ with the DLS
step on the pose error $e=x_d-f(q_k)$ (yaw error wrapped to $[-\pi,\pi)$).
The damping is also capped at $\lVert e\rVert$, so the final steps behave like
Newton's method.  Practical points:

- At the home pose ($q_2=0$) the arm cannot move radially to first order, so a
  singular starting guess is nudged by 0.3 rad in $q_2$ first.
- Steps are limited to 0.5 rad (or m) and projected back inside the joint limits.
- It returns one branch only.  If the path from the guess runs into a joint
  limit, it retries from guesses pointed at the target, one per elbow branch.

The closed form remains the default: it is exact, lists both branches, and
reports unreachable targets directly.  The numerical solver is kept for
comparison or for extensions without a closed-form solution.

## Run the checks

```powershell
python scripts/kinematics/check_kinematics.py
python scripts/model/check_model.py
```
