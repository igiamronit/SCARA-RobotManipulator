"""Draw a lecture-style DH sketch from the MuJoCo model's frame positions.

The joints are posed directly for this structural diagram. This is not a
controller or a dynamic simulation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PARAM = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scara.xml"))
DATA = mujoco.MjData(MODEL)

# A bent pose shows both horizontal links, like the lecture sketch.
DATA.qpos[:4] = (0, math.radians(45), .055, 0)
mujoco.mj_forward(MODEL, DATA)


def matrix(name: str) -> np.ndarray:
    return DATA.body(name).xmat.reshape(3, 3).copy()


p0 = DATA.body("upper_arm").xpos.copy()
p1 = DATA.body("forearm").xpos.copy()
p2 = p1 + matrix("forearm") @ np.array([PARAM["a2_m"], 0, 0])
p3 = DATA.body("wrist").xpos.copy()
p4 = DATA.site("tool").xpos.copy()
points = (p0, p1, p2, p3, p4)

# The physical construction frames do not carry the DH alpha_2 flip. Apply
# Rx(pi) for DH frames 2-4; frame 1 follows the upper-arm rotation.
flip = np.diag([1, -1, -1])
frames = (np.eye(3), matrix("upper_arm"), matrix("forearm") @ flip,
          matrix("forearm") @ flip, matrix("wrist") @ flip)


def screen(v: np.ndarray) -> tuple[float, float]:
    x, y, z = v
    return 210 + 800 * (x + .35 * y), 430 - 800 * (z + .45 * y)


def fmt(v: float) -> str:
    return f"{v:.1f}"


svg: list[str] = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480">',
    '<defs><marker id="blueArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#3566a3" stroke-width="1.2"/></marker></defs>',
    '<rect width="900" height="480" fill="white"/>',
]
scale = 2
image = Image.new("RGB", (900 * scale, 480 * scale), "white")
pen = ImageDraw.Draw(image)
font_path = Path("C:/Windows/Fonts/times.ttf")
fonts = {}


def pixel(point):
    return tuple(round(scale * value) for value in point)


def line(a, b, color="#242424", width=1.7, arrow=False, dash=None):
    attrs = f' stroke="{color}" stroke-width="{width}" fill="none"'
    if arrow:
        attrs += ' marker-end="url(#blueArrow)"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    svg.append(f'<line x1="{fmt(a[0])}" y1="{fmt(a[1])}" x2="{fmt(b[0])}" y2="{fmt(b[1])}"{attrs}/>')
    pen.line((pixel(a), pixel(b)), fill=color, width=max(1, round(scale * width)))
    if arrow:
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        if norm:
            ux, uy = dx / norm, dy / norm
            left = (b[0] - 8 * ux - 3.5 * uy, b[1] - 8 * uy + 3.5 * ux)
            right = (b[0] - 8 * ux + 3.5 * uy, b[1] - 8 * uy - 3.5 * ux)
            pen.line((pixel(left), pixel(b), pixel(right)), fill=color,
                     width=max(1, round(scale * width)))


def text(x, y, label, size=16, color="#285c98", italic=False):
    style = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="Times New Roman, serif" font-size="{size}" fill="{color}"{style}>{label}</text>')
    if size not in fonts:
        fonts[size] = ImageFont.truetype(str(font_path), size * scale)
    pen.text(pixel((x, y)), label, font=fonts[size], fill=color, anchor="ls")


def cylinder(x, y, radius=14, height=24):
    top = y - height / 2
    bottom = y + height / 2
    svg.append(f'<path d="M {fmt(x-radius)} {fmt(top)} A {radius} 5 0 0 1 {fmt(x+radius)} {fmt(top)} L {fmt(x+radius)} {fmt(bottom)} A {radius} 5 0 0 1 {fmt(x-radius)} {fmt(bottom)} Z" fill="white" stroke="#242424" stroke-width="1.6"/>')
    svg.append(f'<ellipse cx="{fmt(x)}" cy="{fmt(top)}" rx="{radius}" ry="5" fill="white" stroke="#242424" stroke-width="1.6"/>')
    pen.line((pixel((x-radius,top)),pixel((x-radius,bottom))),fill="#242424",width=3)
    pen.line((pixel((x+radius,top)),pixel((x+radius,bottom))),fill="#242424",width=3)
    pen.arc((pixel((x-radius,bottom-5)),pixel((x+radius,bottom+5))),0,180,fill="#242424",width=3)
    pen.ellipse((pixel((x-radius,top-5)),pixel((x+radius,top+5))),fill="white",outline="#242424",width=3)


xy = [screen(p) for p in points]
x0, y0 = xy[0]
x1, y1 = xy[1]
x2, y2 = xy[2]
x3, y3 = xy[3]
x4, y4 = xy[4]

# Thin black geometry, drawn behind the coordinate frames.
cylinder(x0, 414, 16, 26)
line((x0, 397), (x0, y0))
line(xy[0], xy[1], width=2)
line(xy[1], xy[2], width=2)
cylinder(x1, y1, 14, 22)
svg.append(f'<path d="M {fmt(x2-17)} {fmt(y2-5)} L {fmt(x2+10)} {fmt(y2-12)} L {fmt(x2+22)} {fmt(y2-3)} L {fmt(x2-5)} {fmt(y2+4)} Z M {fmt(x2-17)} {fmt(y2-5)} L {fmt(x2-17)} {fmt(y2+17)} L {fmt(x2-5)} {fmt(y2+25)} L {fmt(x2-5)} {fmt(y2+4)} M {fmt(x2-5)} {fmt(y2+25)} L {fmt(x2+22)} {fmt(y2+17)} L {fmt(x2+22)} {fmt(y2-3)}" fill="white" stroke="#242424" stroke-width="1.5"/>')
top_shape = [(x2-17,y2-5),(x2+10,y2-12),(x2+22,y2-3),(x2-5,y2+4)]
pen.polygon([pixel(p) for p in top_shape],fill="white",outline="#242424",width=3)
pen.line([pixel(p) for p in [(x2-17,y2-5),(x2-17,y2+17),(x2-5,y2+25),(x2-5,y2+4)]],fill="#242424",width=3)
pen.line([pixel(p) for p in [(x2-5,y2+25),(x2+22,y2+17),(x2+22,y2-3)]],fill="#242424",width=3)
line((x2, y2 + 24), (x3, y3 - 13))
cylinder(x3, y3, 13, 22)
line((x3, y3 + 12), xy[4])

# Physical slide extent and wrist-to-tool offset are dimension lines.
for xa, ya, xb, yb, label in [(x2 + 60, y2 + 12, x3 + 60, y3, "d₃"),
                              (x3 + 83, y3, x4 + 83, y4, "d₄")]:
    line((xa, ya), (xb, yb), color="#3566a3", width=1.2)
    line((xa-5, ya), (xa+5, ya), color="#3566a3", width=1.2)
    line((xb-5, yb), (xb+5, yb), color="#3566a3", width=1.2)
    text(xa + 9, (ya + yb) / 2 + 5, label, 17)

# Each axis below is calculated from the DH frame orientation. The arrows
# point in the true projected positive directions of x_i, y_i, and z_i.
labels = (("x₀", "y₀", "z₀"), ("x₁", "y₁", "z₁"),
          ("x₂", "y₂", "z₂"), ("x₃", "y₃", "z₃"),
          ("x₄", "y₄", "z₄"))
for i, (p, rotation, names) in enumerate(zip(points, frames, labels)):
    start = screen(p)
    for axis in range(3):
        end = screen(p + .042 * rotation[:, axis])
        line(start, end, color="#3566a3", width=1.4, arrow=True)
        dx, dy = end[0] - start[0], end[1] - start[1]
        # Small hand-placed offsets keep labels off the arrowheads.
        nudge_x = 5 if dx >= 0 else -20
        nudge_y = -4 if dy <= 0 else 17
        if i == 2 and axis == 2:
            nudge_x = -19
        if i == 4 and axis == 2:
            nudge_x = -18
        text(end[0] + nudge_x, end[1] + nudge_y, names[axis], 15)

# Joint variables follow the lecture's notation, without extra decoration.
text(x0 + 12, y0 + 76, "θ₁", 17)
text(x1 + 7, y1 + 67, "θ₂", 17)
text(x3 + 22, y3 + 10, "θ₄", 17)

svg.append('</svg>')
(ROOT / "figures" / "frame_assignment.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
image.save(ROOT / "figures" / "frame_assignment.png")
print("Wrote figures/frame_assignment.svg and .png from MuJoCo frame positions and DH axes.")
