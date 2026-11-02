# Parameters and limits for the team

All numerical values below are **provisional educational design values** selected for a presentable simulation. The class images define the joint arrangement and the DH twist, but no lengths, physical parameters, or speed-test conditions. The machine-readable source is `config/robot.json`; edit that file and regenerate the outputs if the team changes values.

| Parameter | Value | Meaning |
| --- | ---: | --- |
| Shoulder plane height | 0.280 m | World-to-DH-frame-0 offset |
| Upper arm length $a_1$ | 0.240 m | Shoulder to elbow |
| Forearm length $a_2$ | 0.180 m | Elbow to vertical axis |
| $d_3$ | 0.080–0.190 m | Downward wrist offset |
| $d_4$ | 0.075 m | Wrist to tool site, downward |
| Finger opening increment | 0–0.018 m per finger | Both fingers move outward symmetrically |

The maximum horizontal reach from the shoulder axis is 0.420 m before joint limits and collision constraints. At the minimum $d_3$, the tool site is 0.125 m above the floor; at the maximum it is 0.015 m above the floor. At the most extended vertical pose, the finger tips have about 0.010 m nominal floor clearance in the zero arm orientation. Actual object dimensions and grasp clearances must be checked by the teammate doing the task scene.

| Joint | Range | Chosen speed metadata | Chosen effort limit |
| --- | --- | ---: | ---: |
| Shoulder | −150° to +150° | 6 rad/s | ±12 Nm |
| Elbow | −145° to +145° | 8 rad/s | ±8 Nm |
| Vertical | $q_3=0$ to 0.110 m, downward | 0.8 m/s | ±80 N |
| Wrist | −180° to +180° | 12 rad/s | ±2 Nm |
| Each finger | 0 to 0.018 m, outward | 0.25 m/s | ±8 N |

Joint ranges are encoded in MJCF and URDF. MuJoCo's joint limits are soft constraints: the 100 ms full-effort finger check reaches approximately 18.09 mm against the declared 18 mm limit. Effort limits are encoded as MJCF motor control ranges and URDF effort metadata. The URDF velocity fields are **metadata**, and MJCF does not enforce these speed values automatically; the teammate planning trajectories must enforce speed and acceleration limits in the controller. There is no validated one-pick-and-place-per-second claim. A full cycle definition, object weight, travel distance, actuator tracking, grasp success, and repeat trials are still needed to assess it.

The fixed base is visual/collision geometry. The moving body masses are 1.10 kg upper arm, 0.75 kg forearm, 0.30 kg vertical carriage, 0.22 kg wrist/palm, and 0.04 kg per finger. The declared diagonal COM inertias are simple conceptual approximations in `scripts/model/build.py`, not measured hardware data. The visual STL meshes do not set the mass properties. Before dynamics work, the team should agree on and document equivalent solids or measured inertia values, then update the model and derivation together.

The URDF and MJCF share joint origins, axes, visual meshes, and major collision shapes. URDF does not contain the MuJoCo floor, camera, or motor definitions. The FreeCAD FCStd/STEP zero-pose assembly contains the geometric parts but no kinematic constraints. STL files are individual mesh assets, not a complete robot description.

