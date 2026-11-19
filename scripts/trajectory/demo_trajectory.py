"""Generate trajectory CSV/plots and optional real physics tracking results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from planner import (ROOT, HOME, WORKED_GOAL, Segment, example_trajectory,
                     load_settings, plan_segment, validate)


def write_csv(path, arrays):
    columns, values = [], []
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.ndim == 1:
            columns.append(name)
            values.append(array[:, None])
        else:
            columns.extend(f"{name}_{index+1}" for index in range(array.shape[1]))
            values.append(array)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(np.column_stack(values))


def plots(output, reference, tracking=None, comparison=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scara_kinematics import forward_kinematics
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    factors = np.array([180/np.pi, 180/np.pi, 1000, 180/np.pi])
    names = ["Shoulder", "Elbow", "Vertical", "Wrist"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True, layout="constrained")
    for row, key in enumerate(("q", "qdot", "qddot")):
        for i in range(4):
            axes[row, i].plot(reference["time"], reference[key][:, i]*factors[i], color="#147d92", label="Desired")
            if tracking is not None:
                axes[row, i].plot(tracking["time"], tracking[key][:, i]*factors[i], color="#cf672d", lw=.8, alpha=.8, label="Actual")
            unit = "mm" if i == 2 else "deg"
            unit += ("", "/s", "/s^2")[row]
            axes[row, i].set_ylabel(unit)
            axes[row, i].grid(alpha=.2)
            if row == 0:
                axes[row, i].set_title(names[i])
            if row == 2:
                axes[row, i].set_xlabel("Time (s)")
    axes[0, 0].legend()
    fig.suptitle("SCARA quintic reference and unloaded physics tracking")
    fig.savefig(output / "joint_profiles.png")
    plt.close(fig)
    poses = np.array([forward_kinematics(q) for q in reference["q"]])
    fig = plt.figure(figsize=(7, 5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(*poses[:, :3].T, color="#147d92", label="Desired tool path")
    if tracking is not None:
        actual = np.array([forward_kinematics(q) for q in tracking["q"]])
        ax.plot(*actual[:, :3].T, color="#cf672d", ls="--", lw=.7, label="Actual")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Task-style waypoints; no object/grasp simulation")
    ax.legend()
    fig.savefig(output / "tool_path.png")
    plt.close(fig)
    if tracking is not None:
        fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, layout="constrained")
        for i in range(4):
            for log, label, color in ((tracking, "Computed torque", "#147d92"), (comparison, "PD + bias", "#cf672d")):
                if log is not None:
                    axes[0, i].plot(log["time"], (log["desired_q"][:, i]-log["q"][:, i])*factors[i], label=label, color=color)
                    axes[1, i].plot(log["time"], log["effort"][:, i], color=color)
            axes[0, i].set_title(names[i])
            axes[0, i].set_ylabel("Error (mm)" if i == 2 else "Error (deg)")
            axes[1, i].set_ylabel("Force (N)" if i == 2 else "Torque (Nm)")
            axes[1, i].set_xlabel("Time (s)")
            for row in axes:
                row[i].grid(alpha=.2)
        axes[0, 0].legend()
        fig.savefig(output / "tracking_errors_and_effort.png")
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, layout="constrained")
    for profile in ("cubic", "quintic", "lspb"):
        segment = Segment(np.zeros(4), np.ones(4), 1, profile)
        times = np.linspace(0, 1, 1001)
        samples = np.array([segment.evaluate(t) for t in times])[:, :, 0]
        for index, ax in enumerate(axes):
            ax.plot(times, samples[:, index], label=profile)
            ax.grid(alpha=.2)
    for ax, label in zip(axes, ("Progress s", "ds/du", "d2s/du2")):
        ax.set_ylabel(label)
    axes[0].legend()
    axes[2].set_xlabel("Normalized time u")
    fig.savefig(output / "profile_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", action="store_true", help="Run computed torque and PD+bias with real dynamics stepping")
    parser.add_argument("--plots", action="store_true", help="Requires matplotlib")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "trajectory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    limits, settings = load_settings()
    trajectory = example_trajectory(limits, settings)
    report = {"reference": validate(trajectory, limits), "settings": settings,
              "velocity_limits_SI": limits.velocity.tolist(),
              "acceleration_limits_SI": limits.acceleration.tolist()}
    times = np.unique(np.r_[np.arange(0, trajectory.duration, 0.005), trajectory.times])
    samples = np.array([trajectory.evaluate(t) for t in times])
    reference = dict(time=times, q=samples[:, 0], qdot=samples[:, 1], qddot=samples[:, 2])
    write_csv(args.output / "reference.csv", reference)
    worked = plan_segment(HOME, WORKED_GOAL, limits, margin=settings["timing_margin"])
    report["worked_example"] = {"start_SI": HOME.tolist(), "goal_SI": WORKED_GOAL.tolist(),
                                "duration_s": worked.duration,
                                "peak_velocity_SI": worked.peak_rates()[0].tolist(),
                                "peak_acceleration_SI": worked.peak_rates()[1].tolist()}
    tracking = comparison = None
    if args.physics:
        from simulation import simulate
        tracking, report["computed_torque"] = simulate(trajectory, settings, limits=limits)
        comparison, report["pd_bias"] = simulate(trajectory, settings, controller="pd_bias", limits=limits)
        _, report["perturbed_initial_state"] = simulate(
            trajectory, settings, initial_offset=[np.deg2rad(2), np.deg2rad(-2), 0.002, np.deg2rad(2)], limits=limits)
        write_csv(args.output / "tracking.csv", tracking)
        write_csv(args.output / "pd_bias.csv", comparison)
    if args.plots:
        plots(args.output, reference, tracking, comparison)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
