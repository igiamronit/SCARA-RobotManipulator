# SCARA dimensions and limits sources

The project uses ABB axis speeds for the fast unloaded benchmark and uses
published industrial pick-cycle dimensions as a benchmark reference. The
robot geometry and most simulation limits are defined in the project files.

## Axis speeds and travel

The fast preset takes its velocity caps from the **ABB IRB 920T-6/0.45**
with the **OmniCore C30/90XT** controller. The shoulder and elbow stops in the
original robot config also match ABB's published working ranges. See
§1.8.1, p. 51 for travel and §1.8.3, p. 54 for velocity. [ABB product specification, 3HAC075723-001 Rev. P][abb-spec]

| Project axis | ABB maximum speed | Fast preset (SI) | Project travel | ABB travel |
| --- | ---: | ---: | ---: | ---: |
| Shoulder (1) | 396°/s | 6.9115 rad/s | ±150° | ±150° |
| Elbow (2) | 780°/s | 13.6136 rad/s | ±145° | ±145° |
| Vertical slide (3) | 1,200 mm/s | 1.2000 m/s | 110 mm stroke | 180 or 300 mm stroke on the standard IP30 version |
| Wrist (4) | 2,500°/s | 43.6332 rad/s | ±180° | ±400° default |

The speed caps above apply to the fast preset. The default trajectory uses
25% of the project speed settings, giving `1.5 rad/s`, `2.0 rad/s`,
`0.2 m/s`, and `3.0 rad/s` for axes 1 through 4. [Fast preset](../config/trajectory/benchmark.json) · [Default preset](../config/trajectory/defaults.json) · [Robot config](../config/robot.json)

## Published pick-cycle references

The fast simulation uses **300 mm station spacing** and **25 mm of tool
lift** to match the scale of the published motions below. Its **0.500 s**
round trip is a project target for unloaded motion, with two 10 ms station
holds. The manufacturer times apply to their published payloads and paths.
[Benchmark config](../config/trajectory/benchmark.json) · [Simulation results](trajectory/README.md#reproduce-the-results)

| Manufacturer and model | Reach | Published cycle | Test conditions | Source |
| --- | ---: | ---: | --- | --- |
| ABB IRB 920T-6/0.45 | 450 mm | 0.346 s | 1 kg, optimized 25 × 300 × 25 mm picking motion | [ABB datasheet 9AKK107992A6817][abb-datasheet] |
| Yamaha YK400XE-4 | 400 mm | 0.41 s | 2 kg, reciprocating coarse-positioning arch, 300 mm horizontal and 25 mm vertical | [Yamaha specification, note 2][yamaha-spec] |
| Epson GX4-A351x | 350 mm | 0.35 s | 2 kg, optimized round-trip arch, 300 mm horizontal and 25 mm vertical, tabletop boost mode | [Epson GX4 specification sheet CPD-62177, p. 2, note 1][epson-spec] |

## Project design values

The class material supplies the RRPR joint arrangement and DH convention.
The **240 mm upper arm**, **180 mm forearm**, **280 mm shoulder height**,
**80–190 mm DH slide distance**, and **75 mm wrist-to-tool offset** come
from the project configuration. The **420 mm** maximum horizontal reach is
the sum of the two link lengths. The **52 mm** forearm drop is a project
clearance change in the model builder. [DH notes](frames_and_dh.md) · [Model parameters](parameters_and_limits.md) · [Model builder](../scripts/model/build.py)

The project also sets the body masses, inertias, gripper dimensions, motor
effort caps, and joint acceleration bounds. The fast acceleration limits are
`75 rad/s²`, `185 rad/s²`, `12 m/s²`, and `230 rad/s²` in axis order.
The controller gains and cycle timing are simulation settings.
[Robot config](../config/robot.json) · [Fast preset](../config/trajectory/benchmark.json) · [Default preset](../config/trajectory/defaults.json)

[abb-spec]: https://library.e.abb.com/public/4cbd884534ed4e5aa98bafb938d9f447/3HAC075723-001_en_P_Product%20specification%20IRB%20920.pdf?x-sign=%2BB9Zpq2Fu%2BY0ceNGxVKCIoBWk65Lynh570z%2BnWmgRj%2BfdCZydYk20tmVqu2lxeii
[abb-datasheet]: https://library.e.abb.com/public/e52a49c7ae7c4f009734dc7a859de739/IRB%20920T%20datasheet_20220105.pdf?x-sign=vaihQ0Ty5jWEF7%2FdLgicDAdy5mh6SSBAqZWIKtyM0x8gwciQY7KWmmDhhVdQswkE
[yamaha-spec]: https://global.yamaha-motor.com/business/robot/lineup/ykxg/yk-xe/spec/
[epson-spec]: https://files.support.epson.com/far/docs/scara_gx4_robot_specification_sheet_cpd-62177_final.pdf
