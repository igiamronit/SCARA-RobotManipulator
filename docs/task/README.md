# SCARA pick and place

Part 7 adds a pick stand, a place stand, a fixed obstacle, and a free 80 g
workpiece to the MuJoCo robot. The block measures 40 × 54 × 36 mm. The goal is
240.8 mm from the pick position and rotated by 0.35 rad (20.1°). The two stands
are 22 mm high. The obstacle between them is 74 mm high. [MuJoCo task scene](../../models/scara_pick_place.xml)

## Run the task

Use the Python environment from [Part 6](../trajectory/README.md#run-the-demos).
From the repository root, run the task headlessly to get a JSON report.

```sh
python scripts/task/pick_place.py
python scripts/task/check_pick_place.py
```

Watch the same physics run in the MuJoCo viewer on macOS. Use `python` in place
of `mjpython` on Linux and Windows. The cyan trace shows the tool reference.

```sh
mjpython scripts/task/pick_place.py --view
```

The 111 mm transfer is planned above the obstacle. A 90 mm transfer is
kinematically reachable and fails the workpiece clearance check between the
stations. The command reports the violating pair and exits with status 1.

```sh
python scripts/task/pick_place.py --transfer-height 0.090
```

## Grasp, release, and collision checks

The robot approaches with open fingers, descends vertically, closes the
gripper, lifts, carries the workpiece over the obstacle, lowers to the second
stand, opens the fingers, and returns home. Both fingers apply limited motor
forces. MuJoCo friction holds the free body during the transfer. The arm uses
the Part 6 quintic planner and computed-torque controller. The payload affects
the arm through contact forces. [Task runner](../../scripts/task/pick_place.py)

The simulation checks for force on both fingers before lifting and throughout
the carry. It checks for support contact before opening, separation from the
fingers after opening, and a stationary, upright placement after retreat.
The final success criteria use 5 mm position and 0.10 rad orientation
tolerances. The regression check with reduced friction confirms that the task
reports a lost grasp during lift. [Task checks](../../scripts/task/check_pick_place.py)

Before simulation, the runner samples the timed route at intervals of at most
5 ms with the workpiece at its expected pose for each phase. It checks robot
link meshes and collision proxies against the fixtures and workpiece, and
checks the workpiece against the fixtures and floor. It requires 2 mm of
reference clearance between forbidden pairs and checks for at least 1 mm of
measured clearance every 1 ms of physics. The policy permits finger–workpiece contact
during grasp, carry, and release, and workpiece–stand contact at the relevant
station. Reported clearances are sampled distances for this scene.

## Nominal run

The task uses the configured arm speed limits with a 20% timing margin and
task-specific acceleration limits. Closing and opening each take 120 ms.
The loaded transfer has extra time to limit slip, and the final descent
slows before the workpiece meets the stand. The regression test requires
the nominal cycle to finish in under 3 s with less than 1 mm placement error.

| Result | Original timing | Retimed |
| --- | ---: | ---: |
| Simulated duration, including return and settling | 8.421 s | 2.668 s |
| Time to complete release | 6.144 s | 2.062 s |
| Placement error | 0.291 mm | 0.779 mm |
| Minimum measured separation between forbidden pairs | 6.00 mm | 6.00 mm |
| Minimum measured workpiece–obstacle separation | 10.06 mm | 11.86 mm |
| Samples with an actuator at its command limit | 0 | 0 |

## Eye-in-hand variant

The same task can start with eye-in-hand visual servoing, so the robot finds the
block with a wrist camera instead of reading its pose from the scene.
[Eye-in-hand visual servoing](ibvs.md)
