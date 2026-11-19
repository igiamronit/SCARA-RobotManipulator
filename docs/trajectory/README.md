# SCARA trajectory planning and tracking

Part 6 plans synchronized joint motion and tracks it with motor commands in
MuJoCo. Both demos run on the unloaded robot.

## Run the demos

From the repository root on macOS or Linux, set up the Python environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows Command Prompt, use `python -m venv .venv` and
`.venv\Scripts\activate` before installing the requirements.

On macOS, run the default route or the fast benchmark in the viewer.

```sh
mjpython scripts/trajectory/view_trajectory.py
mjpython scripts/trajectory/view_trajectory.py --benchmark-cycles 12
```

Use `python` in place of `mjpython` on Linux and Windows.

## Reproduce the results

```sh
python scripts/trajectory/check_trajectory.py --physics
python scripts/trajectory/demo_trajectory.py --physics
python scripts/trajectory/benchmark_cycle.py > output/trajectory/benchmark.json
```

The checks cover profile endpoints, derivatives, joint limits, and motor-driven
tracking. The demo creates `output/trajectory/` and writes the reference and
tracking CSVs and `validation.json`. The benchmark runs 120 round trips and
reports tracking error, actuator effort, and minimum sampled clearance.

| Unloaded run | Planned duration | Peak tool position error | Saturated samples |
| --- | ---: | ---: | ---: |
| Default quintic route | 7.705 s | 0.076 mm | 0 |
| Fast round trip | 0.500 s | 0.208 mm | 0 |

The fast benchmark measured **6.32 mm** minimum clearance with the forearm
lowered by 52 mm. Its 0.500 s round trip includes two 10 ms station holds.

## Implementation

[The planner](../../scripts/trajectory/planner.py) generates cubic, quintic,
and linear segments with parabolic blends (LSPB). The two presets are in
[`config/trajectory/`](../../config/trajectory). [The tracker](../../scripts/trajectory/simulation.py)
uses MuJoCo's mass matrix, bias, and passive forces for computed-torque control
and includes a PD + bias comparison.

The handwritten derivation uses the DH distance `d3`. The simulator uses the
slider displacement `q3`, with `d3 = 0.080 m + q3`. Both use a downward-positive
slide. The offset cancels from displacements and time derivatives.
