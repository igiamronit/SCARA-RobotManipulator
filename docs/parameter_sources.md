# SCARA dimensions and limits sources

Every simulation uses one set of robot limits, defined in
[`config/robot.json`](../config/robot.json). Published industrial SCARAs are
used only as a reference for scale; their speeds are not copied into the model.

## Axis speeds and travel

All presets, including the fast benchmark, use the rated axis speeds in
`config/robot.json`. For comparison, the **ABB IRB 920T-6/0.45** with the
**OmniCore C30/90XT** controller is listed below. Our shoulder and elbow
travel match ABB's published working ranges. See §1.8.1 for travel and §1.8.3
for velocity. [ABB product specification, 3HAC075723-001 Rev. P][abb-spec]

| Project axis | Project speed | Project travel | ABB maximum speed | ABB travel |
| --- | ---: | ---: | ---: | ---: |
| Shoulder (1) | 6 rad/s | ±150° | 396°/s (6.91 rad/s) | ±150° |
| Elbow (2) | 8 rad/s | ±145° | 780°/s (13.61 rad/s) | ±145° |
| Vertical slide (3) | 0.8 m/s | 110 mm stroke | 1,200 mm/s | 180 or 300 mm stroke on the standard IP30 version |
| Wrist (4) | 12 rad/s | ±180° | 2,500°/s (43.63 rad/s) | ±400° default |

The fast benchmark uses the full project speeds. The default trajectory uses
25% of them, giving `1.5 rad/s`, `2.0 rad/s`, `0.2 m/s`, and `3.0 rad/s` for
axes 1 through 4. [Fast preset](../config/trajectory/benchmark.json) · [Default preset](../config/trajectory/defaults.json) · [Robot config](../config/robot.json)

## Published pick-cycle references

The fast simulation uses **300 mm station spacing** and **25 mm of tool
lift** to match the scale of the published motions below. At the project's
own speed limits its fastest feasible round trip is **0.800 s** for unloaded
motion, with two 10 ms station holds; the elbow speed sets this limit. The
manufacturer times apply to their own robots, payloads and paths.
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
