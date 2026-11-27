# SCARA Robot Manipulator

A simulation project for the AI4000 Robotics course at IIT Hyderabad. The robot has a four-DOF R-R-P-R arm and a two-finger parallel gripper. MuJoCo provides the robot simulation, contact forces and wrist-camera images; the project implements the kinematics, dynamics, motion generation and control in Python.

## Branches

| Branch | Contents |
| --- | --- |
| [`main`](https://github.com/igiamronit/SCARA-RobotManipulator/tree/main) | Robot models, kinematics, dynamics, timed waypoint trajectories, physical pick-and-place and eye-in-hand visual servoing. |
| [`feature/rrt-path-planning`](https://github.com/igiamronit/SCARA-RobotManipulator/tree/feature/rrt-path-planning) | The core project plus RRT-Connect obstacle-avoidance path planning and the project report. |

**This is the core-project branch.** It includes timed trajectory generation; obstacle-avoidance route search is available on the RRT branch.

Trajectory generation assigns timing to a specified route. The RRT extension searches for an obstacle-avoiding route before assigning that timing.

## Setup

The project has been verified on Windows with Python 3.12. FreeCAD is optional and is needed only to inspect or regenerate the solid CAD model. The Python requirements include the MuJoCo runtime.

Run these commands in PowerShell:

```powershell
git clone --branch main https://github.com/igiamronit/SCARA-RobotManipulator.git
cd SCARA-RobotManipulator
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the repository is already cloned, run the environment setup from its root directory. All commands below use the virtual environment directly, so activation is optional. If an existing environment has no `pip`, install it with `.\.venv\Scripts\python.exe -m ensurepip --upgrade`.

## Run the interactive simulator

```powershell
.\simulator.cmd
```

Double-clicking `simulator.cmd` opens the same application. It starts in forward-kinematics mode and uses MuJoCo's native viewer.

| Key | Mode | What you control or observe |
| --- | --- | --- |
| F6 | Model | Inspect geometry, joint axes and body frames. |
| F7 | Forward kinematics | Set joint positions and observe the tool pose. |
| F8 | Inverse kinematics | Set a target position and yaw; inspect the joint solution. |
| F9 | Velocity kinematics | Set joint rates and observe the resulting tool velocity. |
| F10 | Resolved rate | Set tool velocity and inspect damped inverse-velocity control. |
| F11 | Dynamics | Apply motor efforts and compare analytical and MuJoCo accelerations. |
| F12 | Trajectory tracking | Watch an unloaded, timed route tracked with computed-torque control. |

Expand **Control** in the right panel to edit the inputs. **Up/Down** selects an input and **Left/Right** changes it. Use **Space** to pause motion modes. All editable mode inputs are in **Control**.

Start a particular mode directly:

```powershell
.\simulator.cmd --mode model
.\simulator.cmd --mode ik
.\simulator.cmd --mode dynamics
```

The same application can be launched with `.\.venv\Scripts\python.exe scripts/app/scara_sim.py`. The `.cmd` launchers are Windows-specific. For native-viewer scripts on macOS, use MuJoCo's `mjpython` launcher; see the [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html#interactive-viewer).

## Run the manipulation tasks

### Pick-and-place with an obstacle

```powershell
.\.venv\Scripts\python.exe scripts/task/pick_place.py --view
```

The gripper picks up a free 80 g block through frictional contact, carries it over a fixed obstacle and releases it onto a second stand. The simulation checks bilateral finger contact, clearance, grasp retention and final placement. This task follows prescribed waypoints.

### Fast pick-and-place without the obstacle

```powershell
.\.venv\Scripts\python.exe scripts/task/scara_fast_obstacle_free_demo.py --cycles 3
```

Runs three trials, resetting the scene between trials. Approach and descent overlap, and the carry uses an arched Cartesian trajectory. Use `--headless` to run without a viewer.

### Eye-in-hand visual servoing

```powershell
.\.venv\Scripts\python.exe scripts/task/ibvs.py --view
```

A wrist-mounted camera detects the block's four top corners. Image-based visual servoing aligns the tool with a taught pre-grasp view, then the robot descends and executes the physical pick-and-place task. The taught goal is included in [config/vision/ibvs_goal.json](config/vision/ibvs_goal.json).

To test an initially displaced object:

```powershell
.\.venv\Scripts\python.exe scripts/task/ibvs.py --view --block-shift 0.008 -0.006 0.15
```

The displacement is `(x, y, yaw)` in metres and radians. The controller observes the camera image rather than receiving this displacement. Removing `--view` from the obstacle task or IBVS command runs the simulation without a window and prints its result as JSON.

### Lightweight kinematics viewer

```powershell
.\kinematics.cmd
```

This separate viewer offers **Joints**, **Target** and **Velocity** modes. In **Joints**, choose **Go to object**, then **Close / pick up**, and move the joint sliders. **Open / release** releases the connector. F1/F2/F3 switch modes. Pickup in this viewer uses kinematic attachment; the block tasks above simulate a free body held by contact forces.

## Inspect the robot models

For an interactive view of the robot and its assigned axes, run `.\simulator.cmd --mode model`. To open the raw MJCF model in MuJoCo's standalone viewer:

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=models/scara.xml
```

| Asset | Purpose | How to inspect it |
| --- | --- | --- |
| [models/scara.xml](models/scara.xml) | Robot MJCF: joints, inertias, motors and collision geometry. | MuJoCo viewer or simulator Model mode. |
| [models/scara_pick_place.xml](models/scara_pick_place.xml) | Task scene with stands, an obstacle and a free payload. | Run the physical pick-and-place task. |
| [models/scara.urdf](models/scara.urdf) | Robot description for exchange with URDF-compatible tools. | Load it together with the mesh assets. |
| [models/meshes/](models/meshes/) | Individual link meshes in STL format. | Open in a mesh viewer or CAD application. |
| [cad/scara.FCStd](cad/scara.FCStd) | Native solid CAD assembly. | Open in FreeCAD. |
| [cad/scara_zero_pose.step](cad/scara_zero_pose.step) | Solid geometry in STEP format. | Open in FreeCAD or another STEP-compatible CAD application. |

Keep the mesh directory with the robot descriptions so their relative asset paths resolve. The MJCF is the reference for the current simulation geometry; the CAD exports predate the 52 mm forearm-offset correction.

Simulation and STL coordinates use metres; the FreeCAD and STEP assemblies use millimetres. STL does not encode units, so choose metres or apply a factor of 1000 when importing a mesh into a CAD document that uses millimetres. The [frame assignment](figures/frame_assignment.png) and [DH table](figures/dh_table.png) are available as separate diagrams.

Main dimensions and limits are defined in [config/robot.json](config/robot.json). The arm has 240 mm and 180 mm horizontal links, 110 mm of slide travel and a 75 mm wrist-to-tool offset. Masses, inertias and motor limits are educational simulation assumptions.

To regenerate the STL, MJCF and URDF assets from the shared parameters:

```powershell
.\.venv\Scripts\python.exe scripts/model/build.py
```

The [FreeCAD macro](cad/create_scara.FCMacro) separately generates the solid CAD assembly and STEP export.

## Validation

The following checks run without opening an interactive viewer:

```powershell
.\.venv\Scripts\python.exe scripts/model/check_model.py
.\.venv\Scripts\python.exe scripts/kinematics/check_kinematics.py
.\.venv\Scripts\python.exe scripts/dynamics/check_dynamics.py
.\.venv\Scripts\python.exe scripts/trajectory/check_trajectory.py --physics
.\.venv\Scripts\python.exe scripts/task/check_pick_place.py
.\.venv\Scripts\python.exe scripts/task/check_ibvs.py
.\.venv\Scripts\python.exe scripts/app/scara_sim.py --check

```

They cover the DH/FK agreement, inverse kinematics, Jacobians, analytical dynamics, trajectory bounds, motor-driven tracking, grasp failure cases and camera feedback. Camera checks still require an available rendering context.

On `main`, the original Harris feature-velocity regression can exceed its 3% tolerance when small rendered corner displacements are differentiated. The RRT branch uses a longer finite-difference interval for that check; the nominal and shifted-object manipulation trials are checked separately.

## Project scope

- [x] 1. Define the robot geometry, joint limits, masses, frames, and sign conventions.
- [x] 2. Build the model and assign frames/DH parameters.
- [x] 3. Derive forward and inverse kinematics, including unreachable targets.
- [x] 4. Derive velocity kinematics (the Jacobian), and identify singularities.
- [x] 5. Derive forward and inverse dynamics from the robot's energies.
- [x] 6. Plan a timed path, then control the actuators to track it.
- [x] 7. Add a simulated task: objects, grasp/release, and collision constraints.
- [x] 8. Validate the equations and measure task results: tracking error, placement accuracy, effort, failures, and cycle time.

## Mathematical background and results

| Topic | Documentation |
| --- | --- |
| Frames and DH convention | [Frame assignment and DH table](docs/frames_and_dh.md) |
| Dimensions and design references | [Parameters and limits](docs/parameters_and_limits.md), [parameter sources](docs/parameter_sources.md) |
| Position and velocity kinematics | [Kinematics](docs/kinematics.md) |
| Analytical forward and inverse dynamics | [Dynamics](docs/dynamics.md) |
| Timed trajectories and tracking | [Trajectory planning and control](docs/trajectory/README.md) |
| Physical task | [Pick-and-place](docs/task/README.md) |
| Camera feedback | [Eye-in-hand visual servoing](docs/task/ibvs.md) |
| Handwritten derivations | [Derivation PDFs](Handwritten-stuff/) |
