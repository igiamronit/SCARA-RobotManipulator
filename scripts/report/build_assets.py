"""Build measured results and print-quality figures for report/main.tex.

Run from the repository: python scripts/report/build_assets.py --refresh
--refresh reruns the checks, vision trials and RRT trials. Without it the saved
RRT/vision trials are reused; baseline physics and plots are always refreshed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts"/name) for name in ("path_planning", "task", "trajectory", "dynamics")]
import pick_place as task
from pick_place_rrt import make_rrt_plan
from planner import example_trajectory, load_settings, HOME
from simulation import simulate as track
from scara_kinematics import forward_kinematics
from scara_dynamics import mass_matrix, load_dynamic_parameters

REPORT = ROOT/"report"
FIG = REPORT/"figures"
RESULT = REPORT/"results"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                     "figure.constrained_layout.use": True, "savefig.dpi": 220,
                     "pdf.fonttype": 42})


def save_plot(fig, name):
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sources():
    for name in ("frame_assignment.png", "kinematics_viewer.png", "forearm_offset_52mm_vertical.png"):
        source = ROOT/"figures"/name
        if source.exists():
            shutil.copy2(source, FIG/name)
    for name in ("lecture_scara_frames.png",):
        shutil.copy2(ROOT/"docs"/name, FIG/name)
    # Existing project presentation assets, never external stock illustrations.
    archive = ROOT/"presentation/scara-presentation-overleaf.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            wanted = {"basemodel.png", "dynamics_mode.png", "elbow_before.png",
                      "pick_place_sequence_free.png", "route_tool_path.png", "simulator.png",
                      "task_scene_obstacle.png"}
            for name in z.namelist():
                if name.startswith("figures/") and Path(name).name in wanted:
                    target = FIG/Path(name).name
                    if not target.exists():
                        target.write_bytes(z.read(name))


def run_checks():
    checks = [("model", ["scripts/model/check_model.py"]),
              ("kinematics", ["scripts/kinematics/check_kinematics.py"]),
              ("dynamics", ["scripts/dynamics/check_dynamics.py"]),
              ("trajectory", ["scripts/trajectory/check_trajectory.py", "--physics"]),
              ("pick_place", ["scripts/task/check_pick_place.py"]),
              ("ibvs", ["scripts/task/check_ibvs.py"]),
              ("path_planning", ["scripts/path_planning/check_path_planning.py"])]
    results = {}
    for name, args in checks:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=240)
        (RESULT/f"check_{name}.txt").write_text(result.stdout+result.stderr, encoding="utf-8")
        results[name] = {"command": "python "+" ".join(args), "exit_code": result.returncode}
        print(f"Check {name}: {result.returncode}", flush=True)
    write_json(RESULT/"checks.json", results)
    if any(r["exit_code"] for r in results.values()):
        raise RuntimeError("A project check failed; inspect report/results/check_*.txt")


class Snapshots:
    def __init__(self, targets):
        self.targets = list(targets)
        self.frames = []
        self.renderer = None
        self.names = []

    def __call__(self, model, data, desired):
        if not self.targets or data.time + 1e-8 < self.targets[0][0]:
            return True
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, 720, 960)
        _, caption = self.targets.pop(0)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [.16, .015, .145]
        camera.distance, camera.azimuth, camera.elevation = .78, 135, -28
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        self.renderer.update_scene(data, camera=camera, scene_option=option)
        self.frames.append(Image.fromarray(self.renderer.render().copy()))
        self.names.append(caption)
        return True

    def save(self, name):
        if self.renderer:
            self.renderer.close()
        # Images remain simulation renders. Only captions and white gutters are added.
        width, height, caption = 480, 360, 34
        grid = Image.new("RGB", (3*width, 2*(height+caption)), "white")
        draw = ImageDraw.Draw(grid)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 20)
        except OSError:
            font = None
        for i, (frame, text) in enumerate(zip(self.frames, self.names)):
            x, y = i%3*width, i//3*(height+caption)
            grid.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (x, y))
            draw.text((x+10, y+height+5), f"({chr(97+i)}) {text}", fill="black", font=font)
        grid.save(FIG/f"{name}.png")


def baseline_results():
    limits, settings = load_settings()
    trajectory = example_trajectory(limits, settings)
    logs, computed = track(trajectory, settings, limits=limits, check_clearance=True)
    other, pd = track(trajectory, settings, controller="pd_bias", limits=limits)
    labels = ("Shoulder (deg)", "Elbow (deg)", "Slide (mm)", "Wrist (deg)")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4))
    for i, ax in enumerate(axes.flat):
        scale = 1000 if i == 2 else 180/np.pi
        ax.plot(logs["time"], (logs["desired_q"][:, i]-logs["q"][:, i])*scale,
                color="black", label="Computed torque")
        ax.plot(other["time"], (other["desired_q"][:, i]-other["q"][:, i])*scale,
                color=".5", ls="--", label="PD + bias")
        ax.set(title=labels[i], xlabel="Time (s)", ylabel="Tracking error")
    axes[0, 0].legend(fontsize=8)
    save_plot(fig, "tracking_comparison")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4))
    for i, ax in enumerate(axes.flat):
        ax.plot(logs["time"], logs["effort"][:, i], color="black")
        ax.set(title=labels[i].split(" (")[0], xlabel="Time (s)",
               ylabel="Force (N)" if i == 2 else "Torque (N m)")
    save_plot(fig, "motor_efforts")
    model = task.load_scene()
    trajectory, phases = task.make_plan(model)
    times = trajectory.times
    shots = Snapshots([(0, "Approach"), (times[3], "Grasp"), (times[4], "Lift"),
                       ((times[4]+times[5])/2, "Transfer"), (times[6], "Seat"),
                       (times[8], "Release and retreat")])
    physical = task.simulate(model, trajectory, phases, on_step=shots)
    shots.save("pick_place_sequence")
    # This module exposes a modified contact policy for its in-memory obstacle-free scene.
    import scara_fast_obstacle_free_demo as fast
    spec = mujoco.MjSpec.from_file(str(fast.SCENE))
    spec.delete(spec.geom("obstacle"))
    open_model = spec.compile()
    original_scene = task.Scene
    try:
        task.Scene = fast.OpenScene
        motion, phases = fast.make_plan(open_model)
        fast_reports = [task.simulate(open_model, motion, phases) for _ in range(3)]
    finally:
        task.Scene = original_scene
    # Local unconstrained linearization of qdd = M^-1 tau + drift.
    M = mass_matrix(HOME, load_dynamic_parameters())
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [np.zeros((4, 8))]])
    B = np.vstack([np.zeros((4, 4)), np.linalg.inv(M)])
    controllability = int(np.linalg.matrix_rank(np.hstack([B, A@B])))
    result = dict(unloaded_computed_torque=computed, unloaded_pd_bias=pd,
                  physical_pick_place=physical, fast_obstacle_free=fast_reports,
                  local_controllability_rank=controllability)
    write_json(RESULT/"baseline.json", result)
    return result


def rrt_figures():
    trial = json.loads((RESULT/"path_planning/nominal_0.json").read_text())
    model = task.load_scene()
    motion, phases, planning = make_rrt_plan(model, seed=0)
    carrying = [i for i, p in enumerate(phases) if p.name.startswith("rrt_transfer")]
    begin, end = motion.times[carrying[0]], motion.times[carrying[-1]+1]
    shots = Snapshots([(0, "Approach"), (begin, "Payload lifted"),
                       (begin+(end-begin)/3, "RRT detour"),
                       (begin+2*(end-begin)/3, "RRT detour"),
                       (end, "Above destination"), (motion.duration-.1, "Placed")])
    executed = task.simulate(model, motion, phases, on_step=shots)
    shots.save("rrt_sequence")
    if not executed["success"]:
        raise RuntimeError(executed["failure"])
    # Show the payload centre for the planned transfer and the rejected direct edge.
    waypoints = np.array(planning["waypoints"])
    paths = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        paths.extend(forward_kinematics(a+(b-a)*u)[:3]-[0, 0, task.GRASP_OFFSET]
                     for u in np.linspace(0, 1, 100))
    paths = np.asarray(paths)*1000
    direct = np.array([forward_kinematics(waypoints[0]+(waypoints[-1]-waypoints[0])*u)[:3]
                       -[0, 0, task.GRASP_OFFSET] for u in np.linspace(0, 1, 200)])*1000
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    for ax, dims, names in zip(axes, [(0, 1), (1, 2)], [("x (mm)", "y (mm)"), ("y (mm)", "z (mm)")]):
        a, b = dims
        center = np.array([296, 2, 37]); half = np.array([27, 26, 37])
        ax.add_patch(Rectangle((center[a]-half[a], center[b]-half[b]), 2*half[a], 2*half[b],
                               facecolor=".8", edgecolor=".3", label="Obstacle"))
        ax.plot(direct[:, a], direct[:, b], ls="--", color=".5", label="Rejected direct transfer")
        ax.plot(paths[:, a], paths[:, b], color="black", label="RRT payload-centre path")
        ax.scatter(paths[[0, -1], a], paths[[0, -1], b], color="black", s=20)
        ax.set(xlabel=names[0], ylabel=names[1], title="Top view" if a == 0 else "Side view")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.12),
                   ncol=3, frameon=False)
    save_plot(fig, "rrt_path")
    experiments = json.loads((RESULT/"path_planning/experiments.json").read_text())
    rows = experiments["trials"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    for ax, key, label in zip(axes, ["planning_s", "placement_mm", "clearance_mm"],
                               ["Planning time (s)", "Placement error (mm)", "Payload-obstacle clearance (mm)"]):
        for name, marker, gray in [("nominal", "o", "black"), ("shifted_obstacle", "s", ".5")]:
            selected = [r for r in rows if r["case"] == name and r["success"]]
            ax.scatter([r["seed"] for r in selected], [r[key] for r in selected],
                       marker=marker, c=gray, s=25, label=name.replace("_", " "))
        ax.set(xlabel="Random seed", ylabel=label)
    axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)
    save_plot(fig, "rrt_experiments")
    return experiments


def vision_figures():
    source = ROOT/"output/ibvs"
    target = RESULT/"ibvs"
    target.mkdir(exist_ok=True)
    for file in source.iterdir():
        if file.suffix in (".csv", ".json"):
            shutil.copy2(file, target/file.name)
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    styles = ["-", "--", "-.", ":", "--", "-"]
    names = ["nominal", "depth_x0.5", "depth_x2", "mount_error_30deg", "mount_error_100deg", "shifted_block"]
    for i, (name, style) in enumerate(zip(names, styles)):
        data = np.genfromtxt(target/f"{name}.csv", delimiter=",", names=True)
        ax.semilogy(data["t"], data["rms_px"], color=str(.08+i*.13), ls=style,
                    label=name.replace("_", " "), lw=1.5)
    ax.axhline(1, color=".4", lw=.8)
    ax.set(xlabel="Servo time (s)", ylabel="RMS corner error (pixels)")
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, "ibvs_convergence_current")
    images = [Image.open(source/f"{name}_view.png").convert("RGB") for name in ("start", "goal")]
    canvas = Image.new("RGB", (1280, 480), "white")
    for i, img in enumerate(images):
        canvas.paste(img, (640*i, 0))
    canvas.save(FIG/"ibvs_views_current.png")
    return json.loads((target/"summary.json").read_text())


def supplementary_plots():
    from planner import Segment
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for name, style, gray in [("cubic", "--", ".5"), ("quintic", "-", "black"), ("lspb", ":", ".3")]:
        segment = Segment(np.zeros(4), np.ones(4), 1, profile=name)
        samples = np.array([segment.evaluate(u) for u in np.linspace(0, 1, 501)])
        for i, ax in enumerate(axes):
            ax.plot(np.linspace(0, 1, 501), samples[:, i, 0], ls=style, color=gray, label=name)
    for ax, label in zip(axes, ["Progress s", "Normalized speed s'", "Normalized acceleration s''"]):
        ax.set(xlabel="Normalized time u", ylabel=label)
    axes[0].legend(fontsize=7)
    save_plot(fig, "time_profiles")
    q1, q2 = np.meshgrid(np.deg2rad(np.linspace(-150, 150, 201)),
                          np.deg2rad(np.linspace(-145, 145, 201)))
    x, y = .24*np.cos(q1)+.18*np.cos(q1+q2), .24*np.sin(q1)+.18*np.sin(q1+q2)
    fig, ax = plt.subplots(figsize=(4.7, 3.8))
    ax.scatter(1000*x, 1000*y, s=.2, color=".4", rasterized=True)
    angle = np.linspace(0, 2*np.pi, 400)
    ax.plot(60*np.cos(angle), 60*np.sin(angle), "--", color="black", lw=.8, label="Geometric inner radius")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax.legend(fontsize=8)
    save_plot(fig, "workspace_current")


def latex_tables(baseline, planning, vision):
    nominal = [r for r in planning["trials"] if r["case"] == "nominal"]
    shifted = [r for r in planning["trials"] if r["case"] == "shifted_obstacle"]
    lines = []
    for name, rows in [("Nominal obstacle", nominal), ("Obstacle shifted +16 mm in x", shifted)]:
        ok = [r for r in rows if r["success"]]
        if not ok:
            lines.append(f"{name} & 0/{len(rows)} & -- & -- & -- & -- \\")
            continue
        values = []
        for key in ("planning_s", "cycle_s", "placement_mm", "clearance_mm"):
            v = [r[key] for r in ok]
            values.append(f"{min(v):.2f}--{max(v):.2f}")
        lines.append(f"{name} & {len(ok)}/{len(rows)} & "+" & ".join(values)+r" \\")
    (RESULT/"rrt_table.tex").write_text("\n".join(lines)+"\n")
    lines = []
    for name, key in [("Nominal", "nominal"), ("Depth estimate $\\times 0.5$", "depth_x0.5"),
                      ("Depth estimate $\\times 2$", "depth_x2"),
                      ("Mounting error $30^\\circ$", "mount_error_30deg"),
                      ("Mounting error $100^\\circ$", "mount_error_100deg"),
                      ("Shifted block", "shifted_block")]:
        row = vision[key]
        rate, error = row.get("decay_rate_per_s"), row.get("grasp_xy_error_mm")
        lines.append(f"{name} & {row['time_s']:.2f} & "+(f"{rate:.2f}" if rate is not None else "--")+
                     " & "+(f"{error:.2f}" if error is not None else "--")+
                     " & "+("Success" if row["success"] else "View lost")+r" \\")
    (RESULT/"ibvs_table.tex").write_text("\n".join(lines)+"\n")
    obstacle = baseline["physical_pick_place"]
    free = baseline["fast_obstacle_free"][0]
    lines = []
    for label, key, scale, fmt in [("Full cycle (s)", "simulated_duration_s", 1, ".3f"),
                                   ("Placement error (mm)", "placement_error_m", 1000, ".3f"),
                                   ("Peak tool error (mm)", "max_tool_error_m", 1000, ".3f"),
                                   ("Saturated samples", "saturated_samples", 1, ".0f")]:
        lines.append(label+" & "+format(obstacle[key]*scale, fmt)+" & "+format(free[key]*scale, fmt)+r" \\")
    lines.append("Release complete (s) & "+f"{obstacle['events']['release_s']:.3f} & {free['events']['release_s']:.3f}"+r" \\")
    (RESULT/"task_table.tex").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    sources()
    if args.refresh:
        run_checks()
        subprocess.run([sys.executable, "scripts/task/ibvs_experiments.py"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/path_planning/experiments.py", "--output",
                        str(RESULT/"path_planning")], cwd=ROOT, check=True)
    baseline = baseline_results()
    print("Baseline results and renders complete", flush=True)
    planning = rrt_figures()
    vision = vision_figures()
    supplementary_plots()
    latex_tables(baseline, planning, vision)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                **{p: importlib.metadata.version(p) for p in
                   ("mujoco", "numpy", "opencv-python-headless", "matplotlib", "pillow")}}
    write_json(RESULT/"environment.json", versions)
    print("Report assets complete:", REPORT, flush=True)


if __name__ == "__main__":
    main()
