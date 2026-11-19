"""IBVS experiments: nominal, wrong depth, wrong camera angle and a moved block.

    python scripts/task/ibvs_experiments.py

Writes one CSV per run, the start and goal camera images and summary.json to output/ibvs/.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from ibvs import GAMMA, HAND_EYE_ANGLE, draw_features, load_goal, run, shift_block
from pick_place import load_scene
from planner import ROOT

OUT = ROOT / "output" / "ibvs"

EXPERIMENTS = [
    ("nominal", {}, None),
    ("depth_x0.5", {"depth_scale": .5}, None),
    ("depth_x2", {"depth_scale": 2.}, None),
    ("mount_error_30deg", {"hand_eye_angle": HAND_EYE_ANGLE + np.radians(30)}, None),
    ("mount_error_100deg", {"hand_eye_angle": HAND_EYE_ANGLE + np.radians(100)}, None),
    ("shifted_block", {}, (.008, -.006, .15)),
]


def decay_rate(t, rms):
    """Fit rms = A exp(-rate t) between 80% of the initial error and 2 px."""
    t, rms = np.asarray(t), np.asarray(rms)
    use = (rms < .8*rms[0]) & (rms > 2)
    if use.sum() < 3:
        return None
    return float(-np.polyfit(t[use], np.log(rms[use]), 1)[0])


def save_csv(name, log):
    with open(OUT / f"{name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "rms_px", *[f"{a}{i}" for i in range(1, 5) for a in "uv"],
                         "tool_x_mm", "tool_y_mm", "tool_z_mm", "tool_yaw_deg"])
        for t, rms, s, tool in zip(log["t"], log["rms_px"], log["s"], log["tool"]):
            writer.writerow([f"{t:.4f}", f"{rms:.4f}", *np.round(s.ravel(), 2),
                             *np.round(1000*tool[:3], 3), round(float(np.degrees(tool[3])), 3)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    goal = load_goal()
    s_star = np.array(goal["s_star_px"])
    summary = {}
    print(f"{'experiment':<20} {'servo':>9} {'time s':>7} {'rate 1/s':>9} {'grasp mm':>9} {'task':>6}")
    for name, options, shift in EXPERIMENTS:
        model = load_scene()
        if shift:
            shift_block(model, *shift)
        report, log = run(model, goal, **options)
        servo = report["servo"]
        rate = decay_rate(log["t"], log["rms_px"])
        summary[name] = {"success": report["success"], "failure": report["failure"],
                         "decay_rate_per_s": rate, "gamma": GAMMA, **servo}
        save_csv(name, log)
        print(f"{name:<20} {'converged' if servo['converged'] else 'FAILED':>9} "
              f"{servo['time_s'] or float('nan'):7.2f} {rate or float('nan'):9.2f} "
              f"{servo.get('grasp_xy_error_mm', float('nan')):9.2f} "
              f"{'ok' if report['success'] else 'fail':>6}")
        if name == "nominal":
            draw_features(log["first_image"], log["s"][0], s_star).save(OUT / "start_view.png")
            draw_features(log["last_image"], log["s"][-1], s_star).save(OUT / "goal_view.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote CSVs, images and summary.json to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
