# Eye-in-hand visual servoing

Before the pick, the robot finds the block with a camera on its wrist. It uses image-based visual servoing (IBVS) from AI4000 Lectures 34 and 36. The camera image steers the arm until it matches a goal image taken once at the pre-grasp pose. The arm then descends with the vertical joint only, and the [pick-and-place plan](README.md) takes over. [Servo code](../../scripts/task/ibvs.py)

## Run it

```sh
python scripts/task/ibvs.py --teach             # record the goal image s* (once)
python scripts/task/ibvs.py                     # look, servo, pick and place
python scripts/task/ibvs.py --block-shift 0.01 0.005 -0.2   # move the block; the robot is not told
python scripts/task/ibvs.py --view
python scripts/task/check_ibvs.py               # checks against MuJoCo ground truth
python scripts/task/ibvs_experiments.py         # CSVs and images in output/ibvs/
```

## Design

| Item | Choice |
| --- | --- |
| Camera | `wrist_cam` on the tool axis, 17 mm above the tool point, looking down, turned −90° about z4. Field of view 60°, 640 × 480 px, so λ = 415.7 px. |
| Features | The block's 4 top corners, s ∈ ℝ⁸ (k = 8). They come from the centroid and second moments of the yellow top-face pixels. |
| Camera motion | ξc = (vx, vy, vz, ωz), m = 4. A SCARA camera cannot tilt. |
| Control law | ξc = −γ L⁺ e with the left pseudoinverse L⁺ = (LᵀL)⁻¹Lᵀ and γ = 2 s⁻¹. Camera frames arrive at 30 Hz. Speed is limited to 0.1 m/s and 1 rad/s. |
| Depth ẑ | The median of the depth image over the top face. The face is parallel to the image plane, so all 4 corners share it. |
| Hand–eye | ξ4 = [R S(d)R; 0 R] ξc (frame 4 is our DH tool frame, the lecture's frame 6) with R = Rz(−90°) and d = (0, 0, −17) mm. S(d)Rω = 0 because d is parallel to ω. |
| Joints | q̇ = J⁻¹(q) ξ0 from the Part 4 task Jacobian. The Part 6 computed-torque controller tracks it. |
| Stop | The RMS corner error is below 1 px, about 0.15 mm at the goal depth. |

## Results

Each run starts at the look pose, offset 10 mm in x and y, 15 mm in height and 14° in yaw. The `ibvs_experiments.py` script measured:

| Run | Servo time | Fitted decay rate | Grasp error | Task |
| --- | ---: | ---: | ---: | --- |
| Nominal (γ = 2) | 2.14 s | 2.03 s⁻¹ | 0.12 mm | success |
| Depth estimate × 0.5 | 4.72 s | 1.01 s⁻¹ | 0.10 mm | success |
| Depth estimate × 2 | 2.08 s | 2.44 s⁻¹ | 0.02 mm | success |
| Camera angle error 30° | 2.44 s | 1.78 s⁻¹ (γ cos 30° = 1.73) | 0.16 mm | success |
| Camera angle error 100° | diverged: block left the view at 0.96 s | | | fails, as predicted for cos α < 0 |
| Block moved 8 mm, −6 mm, 8.6° | 2.28 s | 2.06 s⁻¹ | 0.11 mm | success |

Doubling the depth estimate only speeds up the translation. The ωz column of L does not depend on depth, so the rotation part still decays at γ.
