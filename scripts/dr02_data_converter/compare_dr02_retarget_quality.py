#!/usr/bin/env python3
"""Compare two DR02 retarget CSV files against one generated target-keypoint PKL.

The report is deliberately lightweight for the 25-motion QA pass: it checks
frame alignment/finite values, the full fine-task errors, hand/foot end-point
errors, joint ranges, and the configured inter-frame velocity budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from robot_retarget import RobotRetarget  # noqa: E402


def _read_qpos_csv(path: Path, nq: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] != nq:
        raise ValueError(f"{path} has {values.shape[1]} columns; expected MuJoCo qpos nq={nq}")
    # Exported files store the floating quaternion as xyzw for common tooling;
    # RobotRetarget/MuJoCo consumes wxyz internally.
    qpos = values.copy()
    qpos[:, 3:7] = values[:, [6, 3, 4, 5]]
    return qpos


def _joint_limit_violations(model: mujoco.MjModel, qpos: np.ndarray) -> int:
    violations = 0
    for joint_id in range(model.njnt):
        if model.jnt_limited[joint_id] and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            address = model.jnt_qposadr[joint_id]
            low, high = model.jnt_range[joint_id]
            violations += int(np.count_nonzero((qpos[:, address] < low - 1e-8) | (qpos[:, address] > high + 1e-8)))
    return violations


def _motion_metrics(retargeter: RobotRetarget, qpos: np.ndarray) -> dict:
    available_end_effectors = [
        name for name in ("left_fore_arm", "right_fore_arm", "left_calf", "right_calf")
        if name in retargeter.human_body_to_task
    ]
    all_errors, end_position, end_orientation = [], [], []
    for frame_idx, q in enumerate(qpos):
        retargeter.configuration.update(q)
        retargeter.update_targets(frame_idx)
        all_errors.append(retargeter.error())
        for name in available_end_effectors:
            error = retargeter.human_body_to_task[name].compute_error(retargeter.configuration)
            end_position.append(float(np.linalg.norm(error[:3])))
            end_orientation.append(float(np.linalg.norm(error[3:])))

    velocity = {"max_excess": 0.0, "violating_frames": 0}
    limit = retargeter.frame_velocity_limit
    if limit is not None and len(qpos) > 1:
        excesses = []
        for prior, current in zip(qpos[:-1], qpos[1:]):
            displacement = np.empty(retargeter.model.nv)
            mujoco.mj_differentiatePos(retargeter.model, displacement, 1.0, prior, current)
            excesses.append(float(np.max(np.abs(displacement[limit.indices]) - limit.frame_budget)))
        velocity = {
            "max_excess": max(0.0, max(excesses)),
            "violating_frames": int(sum(excess > 1e-8 for excess in excesses)),
        }
    return {
        "frames": int(len(qpos)),
        "finite": bool(np.all(np.isfinite(qpos))),
        "fine_task_error_mean": float(np.mean(all_errors)),
        "fine_task_error_first_100_mean": float(np.mean(all_errors[:100])),
        "four_end_effector_position_error_mean": float(np.mean(end_position)),
        "four_end_effector_position_error_first_100_mean": float(np.mean(end_position[: 4 * 100])),
        "four_end_effector_orientation_error_mean": float(np.mean(end_orientation)),
        "four_end_effector_orientation_error_first_100_mean": float(np.mean(end_orientation[: 4 * 100])),
        "joint_limit_violations": _joint_limit_violations(retargeter.model, qpos),
        "velocity": velocity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints-pkl", required=True, type=Path, help="Generated DR02 target-keypoint PKL")
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/robot/DR02_temporal_constraints_test.yaml")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, help="Optional one-row-per-method CSV summary")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    retargeter = RobotRetarget(
        model_path=str(PROJECT_ROOT / config["robot_xml_path"]), keypoint_path=str(args.keypoints_pkl),
        ik_match_table=config["ik_match_table"], verbose=False, render_debug=False,
        joints_limit_offset_degrees=config.get("joints_limit_offset_degrees"),
        contact_body_names=config.get("contact_links"), contact_position_cost=config.get("contact_pos_fixed_factor", 10.0),
        temporal_continuity_regularization=config.get("temporal_continuity_regularization"),
        inter_frame_velocity_limit=config.get("inter_frame_velocity_limit"), two_stage_optimization=None,
    )
    report = {}
    for label, path in (("baseline", args.baseline_csv), ("candidate", args.candidate_csv)):
        qpos = _read_qpos_csv(path, retargeter.model.nq)
        if len(qpos) != retargeter.num_frames:
            raise ValueError(f"{path} has {len(qpos)} frames; target PKL has {retargeter.num_frames}")
        report[label] = _motion_metrics(retargeter, qpos)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "frames", "finite", "fine_task_error_mean", "four_end_effector_position_error_mean", "joint_limit_violations", "velocity_max_excess", "velocity_violating_frames"])
            writer.writeheader()
            for method, metrics in report.items():
                writer.writerow({"method": method, **{key: metrics[key] for key in ("frames", "finite", "fine_task_error_mean", "four_end_effector_position_error_mean", "joint_limit_violations")}, "velocity_max_excess": metrics["velocity"]["max_excess"], "velocity_violating_frames": metrics["velocity"]["violating_frames"]})


if __name__ == "__main__":
    main()
