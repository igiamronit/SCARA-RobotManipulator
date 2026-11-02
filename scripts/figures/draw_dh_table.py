"""Draw the lecture-style standard-DH table as SVG and PNG."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PARAM = json.loads((ROOT / "config" / "robot.json").read_text(encoding="utf-8"))
assert PARAM["a1_m"] > 0 and PARAM["a2_m"] > 0
assert PARAM["d3_min_m"] < PARAM["d3_max_m"]

WIDTH, HEIGHT, SCALE = 900, 340, 2
BLUE = "#1f3653"
BLACK = "#222222"
FONT = Path("C:/Windows/Fonts/times.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")
image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
pen = ImageDraw.Draw(image)
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
    f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
]


def label(x, y, value, size=23, bold=False, color=BLACK, centered=False):
    face = FONT_BOLD if bold else FONT
    font = ImageFont.truetype(str(face), size * SCALE)
    anchor = "ms" if centered else "ls"
    pen.text((round(x * SCALE), round(y * SCALE)), value, font=font,
             fill=color, anchor=anchor)
    text_anchor = ' text-anchor="middle"' if centered else ""
    weight = ' font-weight="bold"' if bold else ""
    svg.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, serif" font-size="{size}"{text_anchor}{weight} fill="{color}">{value}</text>')


def rule(x1, y1, x2, y2):
    pen.line(((round(x1 * SCALE), round(y1 * SCALE)),
              (round(x2 * SCALE), round(y2 * SCALE))), fill=BLACK, width=2)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BLACK}" stroke-width="1"/>')


label(52, 57, "DH Parameters", size=25, bold=True, color=BLUE)
columns = [260, 365, 470, 575, 680]
header = ["Link", "aᵢ", "αᵢ", "dᵢ", "θᵢ"]
for x, value in zip(columns, header):
    label(x, 113, value, centered=True)
rule(215, 124, 725, 124)

# Keep the slide's symbolic variables. Their numerical project choices and
# MuJoCo's q3 offset are recorded in docs/frames_and_dh.md.
rows = [
    ("1", "a₁", "0", "0", "θ₁*"),
    ("2", "a₂", "180°", "0", "θ₂*"),
    ("3", "0", "0", "d₃*", "0"),
    ("4", "0", "0", "d₄", "θ₄*"),
]
for y, row in zip((162, 207, 252, 297), rows):
    for x, value in zip(columns, row):
        label(x, y, value, centered=True)

svg.append("</svg>")
(ROOT / "figures" / "dh_table.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
image.save(ROOT / "figures" / "dh_table.png")
print("Wrote figures/dh_table.svg and .png in the lecture's symbolic notation.")
