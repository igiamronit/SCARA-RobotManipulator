"""Repeat seeded RRT physics trials and save machine-readable results."""
import argparse
import csv
import json
from pathlib import Path

from pick_place_rrt import ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/path_planning")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("nominal", i, .074, .296) for i in range(10)]
    cases += [("shifted_obstacle", i, .074, .312) for i in range(3)]
    rows = []
    for case, seed, height, x in cases:
        try:
            report = run(seed, obstacle_height=height, obstacle_x=x)
            (args.output/f"{case}_{seed}.json").write_text(json.dumps(report, indent=2)+"\n")
            p = report["planning"]
            row = dict(case=case, seed=seed, success=report["success"], failure=report["failure"],
                       planning_s=p["planning_time_s"], iterations=p["iterations"], nodes=p["nodes"],
                       state_checks=p["state_checks"], waypoints=len(p["waypoints"]),
                       direct_valid=p["direct_transfer_valid"], cycle_s=report["simulated_duration_s"],
                       placement_mm=1000*report["placement_error_m"],
                       clearance_mm=1000*report["measured_clearance"]["payload_obstacle_clearance_m"],
                       saturated_samples=report["saturated_samples"],
                       speed_pass=report["actual_speed_limits_respected"])
        except (ValueError, RuntimeError) as exc:
            row = dict(case=case, seed=seed, success=False, failure=str(exc))
        rows.append(row)
        print(json.dumps(row), flush=True)
    # A high obstacle makes the fixed home pose violate the 6 mm search margin.
    # Such a query must fail clearly rather than returning a route to invalid endpoints.
    try:
        run(0, obstacle_height=.090)
        negative = {"passed": False, "reason": "Invalid endpoint was unexpectedly accepted"}
    except ValueError as exc:
        negative = {"passed": "configuration is invalid" in str(exc), "reason": str(exc)}
    (args.output/"experiments.json").write_text(json.dumps({"trials": rows, "invalid_endpoint": negative}, indent=2)+"\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/"experiments.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["success"] for row in rows) and negative["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
