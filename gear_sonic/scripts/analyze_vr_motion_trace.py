"""Summarize live VR motion and audit the complete publisher output sequence.

Usage: python -m gear_sonic.scripts.analyze_vr_motion_trace path/to/trace.jsonl
Derivatives use logged command times. Results describe Cartesian commands;
robot feedback remains in the trace for separate joint-tracking analysis.
The live summary excludes returns/faults; output_sequence includes them and gaps.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.teleop.vr_motion_conditioner import rotation_error


def derivative_summary(groups, source):
    metrics = {name: [] for name in ("speed_m_s", "acceleration_m_s2", "jerk_m_s3",
                                   "speed_deg_s", "acceleration_deg_s2", "jerk_deg_s3")}
    for group in groups:
        pose = np.array([r[source] for r in group])
        dt = np.diff([r["time"] for r in group])
        linear = np.diff(pose[:, :, :3], axis=0)
        angular = rotation_error(pose[1:, :, 3:].reshape(-1, 4),
                                 pose[:-1, :, 3:].reshape(-1, 4)).reshape(-1, 3, 3)
        for delta, names, scale in (
            (linear, ("speed_m_s", "acceleration_m_s2", "jerk_m_s3"), 1),
            (angular, ("speed_deg_s", "acceleration_deg_s2", "jerk_deg_s3"), 180 / np.pi),
        ):
            derivative = delta
            for order, name in enumerate(names):
                derivative = derivative / dt[order:, None, None]
                metrics[name].extend((np.linalg.norm(derivative, axis=-1) * scale).flatten())
                derivative = np.diff(derivative, axis=0)
    return {name: {"peak": float(np.max(values)), "p95": float(np.percentile(values, 95))}
            for name, values in metrics.items() if len(values)}


def summarize(path):
    rows = []
    incomplete_lines = 0
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete_lines += 1  # May be inspecting a file still being written.
    samples = [r for r in rows if r.get("type") == "sample"]
    live = [r for r in samples if r["stream_mode"] == 5 and not r["generated"]]
    result = {"path": str(path), "settings": next((r.get("limits") for r in rows if r.get("type") == "metadata"), None),
              "samples": len(samples), "live_samples": len(live),
              "feedback_samples": sum(r.get("type") == "feedback" for r in rows),
              "dropped_samples": max((r.get("dropped", 0) for r in rows), default=0),
              "incomplete_lines": incomplete_lines,
              "fault_samples": sum(bool(r.get("fault")) for r in live)}
    # Audit the complete publisher output sequence separately. A normal-motion
    # summary must not hide the fault, return, and gap intervals that matter
    # most when checking continuity. Missing log rows / reseeds remain explicit
    # discontinuities; there is no way to reconstruct commands that were lost.
    output_groups = []
    discontinuities = 0
    gaps = 0
    for row in samples:
        if not np.isfinite(row["output"]).all():
            discontinuities += 1
            output_groups.append([])
            continue
        if not output_groups or not output_groups[-1]:
            output_groups.append([row])
            continue
        previous = output_groups[-1][-1]
        dt = row["time"] - previous["time"]
        if (row.get("seed_generation") != previous.get("seed_generation") or
                not np.isfinite(dt) or dt <= 0 or
                row.get("dropped", 0) != previous.get("dropped", 0)):
            discontinuities += 1
            output_groups.append([row])
        else:
            gaps += dt > 0.1
            output_groups[-1].append(row)
    result["output_sequence"] = derivative_summary([g for g in output_groups if len(g) >= 4], "output")
    result["output_sequence_scope"] = "Publisher timestamps including faults, returns, and timing gaps; not receiver or motor motion."
    result["output_sequence_discontinuities"] = discontinuities
    result["output_sequence_gaps_over_100ms"] = gaps
    if not live:
        result["status"] = "No live VR teleop samples; cannot assess this trial."
        return result
    groups = []
    for row in live:
        valid = not row.get("fault") and np.isfinite(row["input"]).all() and np.isfinite(row["output"]).all()
        if not valid:
            groups.append([])
            continue
        if not groups or not groups[-1]:
            groups.append([row])
        else:
            previous = groups[-1][-1]
            if (row.get("seed_generation") != previous.get("seed_generation") or
                    not 0 < row["time"] - previous["time"] <= 0.1 or
                    row.get("dropped", 0) != previous.get("dropped", 0)):
                groups.append([row])
            else:
                groups[-1].append(row)
    groups = [group for group in groups if len(group) >= 4]
    result["analyzed_seconds"] = sum(g[-1]["time"] - g[0]["time"] for g in groups)
    for source in ("input", "output"):
        result[source] = derivative_summary(groups, source)
    valid = [r for group in groups for r in group]
    if valid:
        error = np.array([np.asarray(r["input"])[:, :3] - np.asarray(r["output"])[:, :3] for r in valid])
        result["position_error_mm_p95"] = float(np.percentile(np.linalg.norm(error, axis=-1) * 1000, 95))
        limited = [r["limited"] for r in valid if "limited" in r]
        if limited:
            result["limited_fraction_per_point"] = np.mean(limited, axis=0).tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2))
