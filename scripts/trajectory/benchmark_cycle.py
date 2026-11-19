"""Run a repeated, unloaded 300 mm by 25 mm SCARA cycle in MuJoCo."""
from __future__ import annotations

import argparse
import json

import numpy as np
from numpy.polynomial import Polynomial, polynomial

from planner import HOME, ROOT, Limits, Segment, Trajectory, continuous_ik, load_settings
from simulation import simulate


PROGRESS = np.array([0, 0, 0, 10, -15, 6, 0])
ARCH = np.array([0, 0, 0, 64, -192, 192, -64])
START_OFFSET = [np.deg2rad(0.25), np.deg2rad(-0.25), 0.00025, np.deg2rad(0.25)]
PATH_TOLERANCE_M = 0.0005
ARRIVAL_TOLERANCE_M = 0.0001
MIN_CLEARANCE_M = 0.005


def extrema(poly):
    roots = poly.deriv().roots()
    times = [0, 1, *(r.real for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1)]
    values = poly(np.array(times))
    return float(values.min()), float(values.max())


class ArchTransfer:
    """Quintic joint transfer with simultaneous lift and shoulder shaping."""

    def __init__(self, start, goal, duration, lift, shoulder_blend, limits):
        self.q0, self.qf = start, goal
        self.duration = duration
        # Equal shoulder and wrist offsets preserve tool yaw. The slide axis
        # points down, so a negative offset raises the tool above the stations.
        offsets = np.array([shoulder_blend, 0, -lift, shoulder_blend])
        position = PROGRESS[:, None] * (goal - start) + ARCH[:, None] * offsets
        position[0] = start
        speed = polynomial.polyder(position) / duration
        acceleration = polynomial.polyder(speed) / duration
        lower, upper = np.array([extrema(Polynomial(p)) for p in position.T]).T
        limits.check_position(lower)
        limits.check_position(upper)
        self.peak_velocity = np.max(np.abs([extrema(Polynomial(v)) for v in speed.T]), axis=1)
        self.peak_acceleration = np.max(np.abs([extrema(Polynomial(a)) for a in acceleration.T]), axis=1)
        if (np.any(self.peak_velocity > limits.velocity) or
                np.any(self.peak_acceleration > limits.acceleration)):
            raise ValueError("Benchmark transfer exceeds its axis speed or acceleration limits.")
        self.coefficients = position, speed, acceleration

    def evaluate(self, time):
        if time <= 0:
            return self.q0.copy(), np.zeros(4), np.zeros(4)
        if time >= self.duration:
            return self.qf.copy(), np.zeros(4), np.zeros(4)
        return tuple(polynomial.polyval(time / self.duration, coeff)
                     for coeff in self.coefficients)


class RepeatedCycle:
    def __init__(self, cycle, count):
        self.cycle = cycle
        self.count = count
        self.duration = count * cycle.duration

    def evaluate(self, time):
        local_time = time % self.cycle.duration if 0 < time < self.duration else 0
        return self.cycle.evaluate(local_time)


def load_benchmark(count):
    if count < 1:
        raise ValueError("Benchmark needs at least one cycle.")
    base_limits, defaults = load_settings()
    settings = json.loads((ROOT / "config/trajectory/benchmark.json").read_text())
    # Rated axis speeds from config/robot.json (load_settings scales them down).
    rated_speed = base_limits.velocity / float(defaults["velocity_scale"])
    limits = Limits(base_limits.lower, base_limits.upper, rated_speed,
                    settings["acceleration_limit_SI"])
    settings["settle_time_s"] = 0.0

    r_pick, r_place = settings["station_radii_m"]
    spacing = settings["station_spacing_m"]
    # Place the endpoints at the specified radii and chord distance.
    angle = np.arccos((r_pick**2 + r_place**2 - spacing**2) / (2 * r_pick * r_place)) / 2
    height = settings["station_height_m"]
    pick_pose = [r_pick * np.cos(angle), -r_pick * np.sin(angle), height, 0]
    place_pose = [r_place * np.cos(angle), r_place * np.sin(angle), height, 0]
    pick, place = continuous_ik([pick_pose, place_pose], HOME, limits)
    dwell = settings["dwell_s"]
    move = settings["cycle_s"] / 2 - dwell
    if not 0 < dwell < settings["cycle_s"] / 2:
        raise ValueError("Benchmark dwell must fit within each half-cycle.")
    lift, blend = settings["lift_m"], settings["shoulder_blend_rad"]
    outbound = ArchTransfer(pick, place, move, lift, blend, limits)
    inbound = ArchTransfer(place, pick, move, lift, blend, limits)
    cycle = Trajectory([outbound, Segment(place, place, dwell), inbound, Segment(pick, pick, dwell)])
    return RepeatedCycle(cycle, count), limits, settings


def run_case(trajectory, limits, settings, offset=None):
    logs, metrics = simulate(trajectory, settings, limits=limits, initial_offset=offset,
                             check_clearance=True)
    errors = np.linalg.norm(logs["tool_error"][:, :3], axis=1)
    dt = metrics["physics_timestep_s"]
    starts = np.arange(trajectory.count) * trajectory.cycle.duration
    arrivals = starts[:, None] + trajectory.cycle.times[[1, 3]]
    arrival_indices = np.rint(arrivals / dt).astype(int)
    metrics["max_arrival_error_m"] = float(np.max(errors[arrival_indices]))
    metrics["max_dwell_end_error_m"] = float(np.max(errors[np.rint(
        (starts[:, None] + trajectory.cycle.times[[2, 4]]) / dt).astype(int)]))
    metrics["passed"] = bool(
        metrics["completed"] and metrics["saturated_samples"] == 0 and
        metrics["maximum_active_contacts"] == 0 and metrics["model_position_limits_respected"] and
        metrics["minimum_clearance_m"] >= MIN_CLEARANCE_M and
        metrics["actual_velocity_limits_respected"] and metrics["actual_acceleration_limits_respected"] and
        metrics["max_arrival_error_m"] <= ARRIVAL_TOLERANCE_M and
        metrics["max_dwell_end_error_m"] <= ARRIVAL_TOLERANCE_M and
        (offset is not None or metrics["max_tool_position_error_m"] <= PATH_TOLERANCE_M))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=120)
    parser.add_argument("--sweep", action="store_true", help="Compare feedback gains on three cycles each")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    trajectory, limits, settings = load_benchmark(args.cycles)
    short = RepeatedCycle(trajectory.cycle, 3)
    if args.sweep:
        report = []
        for frequency, damping in ((10, 1), (80, .7), (100, .7), (120, .7), (140, .7)):
            trial = dict(settings, controller_natural_frequency_rad_s=frequency,
                         controller_damping_ratio=damping)
            report.append(dict(frequency_rad_s=frequency, damping_ratio=damping,
                               nominal=run_case(short, limits, trial),
                               offset_start=run_case(short, limits, trial, START_OFFSET)))
        print(json.dumps(report, indent=2))
        return

    nominal = run_case(trajectory, limits, settings)
    perturbed = run_case(short, limits, settings, START_OFFSET)
    print(json.dumps(dict(cycles=args.cycles, cycles_per_minute=60 / trajectory.cycle.duration,
                          benchmark=settings,
                          nominal_path_tolerance_m=PATH_TOLERANCE_M,
                          station_tolerance_m=ARRIVAL_TOLERANCE_M,
                          minimum_required_clearance_m=MIN_CLEARANCE_M,
                          nominal=nominal, offset_start=perturbed), indent=2))
    if not (nominal["passed"] and perturbed["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
