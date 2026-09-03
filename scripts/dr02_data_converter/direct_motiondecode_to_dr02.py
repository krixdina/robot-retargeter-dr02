#!/usr/bin/env python3
"""Direct, disk-free MotionDecode G1 to DR02 NPZ conversion.

Only the final NPZ is published.  The legacy CSV/PKL stages remain available
for replay and debugging, but batch jobs use this process-isolated entrypoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from motiondecode_adapter import load_motiondecode_qpos
from dr02_npz_exporter import export_qpos_array, parse_fps
from robot_replay import (
    append_target_contact_keypoints, build_robot_replay_buffers,
    build_robot_retarget_keypoints, compute_leg_displacement_scale,
    compute_robot_contact_sequence, compute_robot_link_lengths,
    load_effective_knee_angle_offset_degrees, load_path_config,
    load_robot_links_config, load_scalar_float_config, load_scalar_int_config,
    load_yaml_body_list_config, canonicalize_contact_name,
    offset_keypoints_by_contact_height, scale_keypoint_frame_displacements,
)
from robot_retarget import RobotRetarget


def build_keypoints_payload(qpos_frames, fps: int) -> dict:
    """Build the legacy retarget payload entirely in memory."""
    source_config = PROJECT_ROOT / "config/robot/g1.yaml"
    target_config = PROJECT_ROOT / "config/robot/DR02.yaml"
    source_mjcf = load_path_config(source_config, "robot_xml_path").expanduser().resolve()
    target_mjcf = load_path_config(target_config, "robot_xml_path").expanduser().resolve()
    source_links = load_robot_links_config(source_config)
    target_links = load_robot_links_config(target_config)
    source_contact_links = load_yaml_body_list_config(source_config, "contact_links")
    contact_names = tuple(canonicalize_contact_name(name) for name in source_contact_links)
    positions, quaternions, contact_positions = build_robot_replay_buffers(
        qpos_frames=qpos_frames, source_robot_mjcf_path=source_mjcf,
        source_robot_links=source_links, contact_body_names=source_contact_links,
    )
    vel_window = load_scalar_int_config(source_config, "contact_vel_calculate_window", default=6)
    vel_threshold = load_scalar_float_config(source_config, "contact_vel_threshold", default=0.5)
    height_threshold = load_scalar_float_config(source_config, "contact_height_threshold", default=0.025)
    height_lpf_alpha = load_scalar_float_config(source_config, "contact_height_lpf_alpha", default=0.2)
    contact_speeds, contact_states = compute_robot_contact_sequence(
        contact_positions=contact_positions, fps=fps, vel_window=vel_window,
        vel_threshold=vel_threshold, height_threshold=height_threshold,
    )
    target_lengths = compute_robot_link_lengths(target_config, target_mjcf)
    keypoints, keypoint_quats, raw_quats, source_vectors, _, _ = build_robot_retarget_keypoints(
        source_positions=positions, source_quaternions=quaternions,
        target_robot_link_lengths=target_lengths, source_robot_config_path=source_config,
        target_robot_config_path=target_config,
    )
    keypoints, keypoint_quats, extras = append_target_contact_keypoints(
        keypoints=keypoints, keypoint_quaternions=keypoint_quats,
        anchor_raw_quaternions=raw_quats, target_robot_links=target_links,
        target_robot_config_path=target_config, target_robot_mjcf_path=target_mjcf,
    )
    scale, _, _ = compute_leg_displacement_scale(
        robot_link_lengths=target_lengths, skeleton_link_vectors=source_vectors,
        knee_angle_offset_degrees=load_effective_knee_angle_offset_degrees(target_config),
    )
    keypoints = scale_keypoint_frame_displacements(keypoints, scale, root_keypoint_idx=0)
    keypoint_names = ["hips_mean", *target_links.keys(), *extras]
    keypoints, _ = offset_keypoints_by_contact_height(
        keypoints=keypoints, keypoint_names=keypoint_names, contact_names=contact_names,
        contact_positions=contact_positions, contact_states=contact_states,
        height_lpf_alpha=height_lpf_alpha,
    )
    return {"keypoint_names": keypoint_names, "positions": keypoints,
            "quaternions": keypoint_quats, "fps": fps, "contact_names": contact_names,
            "contact_states": contact_states, "contact_positions": contact_positions,
            "contact_speeds": contact_speeds}


def convert(input_path: Path, output_path: Path, fps: int, overwrite: bool) -> None:
    qpos = load_motiondecode_qpos(input_path)
    # ``load_motiondecode_qpos`` deliberately returns the legacy CSV boundary
    # (XYZW). MuJoCo FK consumes its native free-joint WXYZ ordering.
    qpos[:, 3:7] = qpos[:, [6, 3, 4, 5]]
    payload = build_keypoints_payload(qpos, fps)
    config_path = PROJECT_ROOT / "config/robot/DR02.yaml"
    config = yaml.safe_load(config_path.read_text())
    retargeter = RobotRetarget(
        model_path=str((PROJECT_ROOT / config["robot_xml_path"]).resolve()), keypoint_path=None,
        keypoints_data=payload, ik_match_table=config["ik_match_table"], solver="daqp",
        verbose=bool(config.get("verbose", False)), render_debug=False,
        joints_limit_offset_degrees=config.get("joints_limit_offset_degrees", {}),
        contact_body_names=config.get("contact_links"),
        contact_position_cost=config.get("contact_pos_fixed_factor", 10.0),
        temporal_continuity_regularization=config.get("temporal_continuity_regularization"),
        inter_frame_velocity_limit=config.get("inter_frame_velocity_limit"),
        two_stage_optimization=config.get("two_stage_optimization"),
    )
    retargeter.retarget()
    # MuJoCo stores the free-joint quaternion as WXYZ. Exporter accepts the
    # legacy XYZW boundary representation, so convert only this in-memory view.
    result = np.asarray(retargeter.result_pos, dtype=float)
    result[:, 3:7] = result[:, [4, 5, 6, 3]]
    export_qpos_array(result, output_path, fps, overwrite=overwrite)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Direct MotionDecode to DR02 NPZ converter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=parse_fps, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        convert(args.input.resolve(), args.output.resolve(), args.fps, args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
