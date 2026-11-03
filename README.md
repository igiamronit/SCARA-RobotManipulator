# SCARA Robot Manipulator

This is a computer-only SCARA robot project built for MuJoCo. The robot has two rotary arm joints, a vertical sliding joint, a rotary wrist, and a moving two-finger gripper.

Double-click [kinematics.cmd](kinematics.cmd) to open the interactive viewer. See the [controls](docs/interactive_viewer.md).

## SCARA Robot

![SCARA Robot](figures/model_preview.png)

## Project checklist

- [x] 1. Define the robot geometry, joint limits, masses, frames, and sign conventions.
- [x] 2. Build the model and assign frames/DH parameters.
- [x] 3. Derive forward and inverse kinematics, including unreachable targets.
- [x] 4. Derive velocity kinematics (the Jacobian), and identify singularities.
- [ ] 5. Derive forward and inverse dynamics from the robot’s energies.
- [ ] 6. Plan a timed path, then control the actuators to track it.
- [ ] 7. Add a simulated task: objects, grasp/release, and collision constraints.
- [ ] 8. Validate the equations and measure task results: tracking error, placement accuracy, effort, failures, and cycle time.
