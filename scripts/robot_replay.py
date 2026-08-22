#!/usr/bin/env python3
"""Replay robot motion CSV and export retargeted keypoints.

This script loads a source-robot motion CSV, replays it through MuJoCo,
retargets body keypoints to a target robot, and saves a keypoint payload.
You can run it with or without the viewer.

Usage:
	# Basic run (with viewer)
	python scripts/robot_replay.py

	# Run headless and only export keypoints
	python scripts/robot_replay.py --no-viewer

	# Specify source/target configs and motion file
	python scripts/robot_replay.py \
		--source-robot-config config/robot/g1.yaml \
		--target-robot-config config/robot/DR02.yaml \
		--motion-file dataset/lafan1_g1/dance1_subject1.csv \
		--fps 30
"""

from __future__ import annotations

import argparse
import pickle
import sys
import threading
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from smpl_replay import (
	REPLAY_BODY_NAMES,
	advance_frame_cursor,
	apply_axis_map_and_local_euler_offset_wxyz,
	apply_link_scales_to_positions,
	average_quaternions_wxyz,
	compute_contact_height_offsets,
	compute_leg_displacement_scale,
	compute_link_geometry_from_positions,
	compute_link_scale_factors,
	compute_robot_body_local_offset,
	compute_robot_link_lengths,
	compute_windowed_point_speeds,
	derive_leg_body_chain_from_links,
	format_progress_line,
	iter_progress,
	load_key_frame_config,
	load_path_config,
	load_robot_links_config,
	load_scalar_float_config,
	load_scalar_int_config,
	quat_rotate_vectors_wxyz,
	save_keypoints_pkl,
	scale_keypoint_frame_displacements,
	select_frame_slice,
	update_viewer_keypoints,
)


SOURCE_SEMANTIC_LINKS = {
	"left_hip": ("hips_mean", "left_up_leg"),
	"left_thigh": ("left_up_leg", "left_leg"),
	"left_calf": ("left_leg", "left_foot"),
	"right_hip": ("hips_mean", "right_up_leg"),
	"right_thigh": ("right_up_leg", "right_leg"),
	"right_calf": ("right_leg", "right_foot"),
	"neck": ("hips_mean", "shoulder_mean"),
	"head": ("shoulder_mean", "head"),
	"left_shoulder": ("shoulder_mean", "left_arm"),
	"left_arm": ("left_arm", "left_fore_arm"),
	"left_fore_arm": ("left_fore_arm", "left_hand"),
	"right_shoulder": ("shoulder_mean", "right_arm"),
	"right_arm": ("right_arm", "right_fore_arm"),
	"right_fore_arm": ("right_fore_arm", "right_hand"),
}

DEFAULT_SOURCE_ROBOT_CONFIG_PATH = Path("config/robot/g1.yaml")
DEFAULT_TARGET_ROBOT_CONFIG_PATH = Path("config/robot/h2.yaml")
DEFAULT_MOTION_FILE = Path("output_data/robot_motion/bones_g1/body_check_001__A548_M.csv")
DEFAULT_OUTPUT_DIR = Path("output_data/keypoints")

IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Replay a robot-motion CSV and generate target-robot keypoints."
	)
	parser.add_argument(
		"--motion-file",
		type=Path,
		default=DEFAULT_MOTION_FILE,
		help=(
			"Path to a robot motion CSV under output_data/robot_motion. "
			"If omitted, derive it from the source robot config's keypoints_path."
		),
	)
	parser.add_argument(
		"--source-robot-config",
		type=Path,
		default=DEFAULT_SOURCE_ROBOT_CONFIG_PATH,
		help="Path to the source robot YAML used to interpret the input motion.",
	)
	parser.add_argument(
		"--target-robot-config",
		type=Path,
		default=DEFAULT_TARGET_ROBOT_CONFIG_PATH,
		help="Path to the target robot YAML used to generate output keypoints.",
	)
	parser.add_argument(
		"--start-frame",
		type=int,
		default=0,
		help="First frame to replay.",
	)
	parser.add_argument(
		"--end-frame",
		type=int,
		default=-1,
		help="Last frame to replay, inclusive. Default replays until the end.",
	)
	parser.add_argument(
		"--stride",
		type=int,
		default=1,
		help="Replay every Nth frame.",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=120.0,
		help="Override playback fps. Default uses the source keypoints payload fps or 30.",
	)
	parser.add_argument(
		"--loop",
		type=bool,
		default=True,
		help="Loop the clip until the viewer window closes.",
	)
	parser.add_argument(
		"--print-summary",
		action="store_true",
		help="Print clip metadata before launching the viewer.",
	)
	parser.add_argument(
		"--no-viewer",
		action="store_true",
		help="Generate keypoints without launching the MuJoCo viewer.",
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=None,
		help="Optional explicit output .pkl path. Default uses output_data/keypoints/<target>/<motion>_from_<source>_keypoints.pkl.",
	)
	return parser.parse_args()


# 用途：确定机器人回放阶段应读取的运动 CSV 文件，并确保该文件已经存在。
# 输入：motion_file 表示调用者可选的显式运动文件路径；传入空值时改为按项目约定自动推导路径。
# 输入：source_robot_config_path 表示源机器人配置文件路径，自动推导时从中读取关键点文件信息和机器人名称。
# 输出：返回经过用户目录展开和绝对路径解析、且已确认存在的机器人运动 CSV 文件路径。
def resolve_motion_file(motion_file: Path | None, source_robot_config_path: Path) -> Path:
	# 优先采用 motion_file（调用者显式指定的机器人运动 CSV 路径），并在返回前验证文件是否存在。
	if motion_file is not None:
		resolved = motion_file.expanduser().resolve()
		if not resolved.exists():
			raise FileNotFoundError(f"Robot motion CSV not found: {resolved}")
		return resolved

	# 未显式指定运动文件时，从源机器人参数 yaml 中读取关键点文件路径，并提取不含约定后缀的动作名称。
	keypoints_path = load_path_config(source_robot_config_path, "keypoints_path").expanduser().resolve()
	keypoint_stem = keypoints_path.stem
	if keypoint_stem.endswith("_keypoints"):
		keypoint_stem = keypoint_stem[: -len("_keypoints")]
	# 按动作名称与源机器人配置名称组合出生成脚本所采用的 CSV 路径，并确认生成结果可用。
	derived = Path("output_data/robot_motion") / f"{keypoint_stem}_{source_robot_config_path.stem}.csv"
	resolved = derived.expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(
			"Robot motion CSV not found. Provide --motion-file or generate it first with robot_retarget.py: "
			f"{resolved}"
		)
	return resolved


def load_robot_motion_csv(csv_path: Path) -> np.ndarray:
	data = np.loadtxt(csv_path, delimiter=",")
	if data.ndim == 1:
		data = data[None, :]
	if data.shape[1] < 7:
		raise ValueError(f"Motion CSV must contain at least 7 columns, got {data.shape}")

	qpos = data.astype(np.float32, copy=True)
	quat_xyzw = qpos[:, 3:7].copy()
	qpos[:, 3] = quat_xyzw[:, 3]
	qpos[:, 4] = quat_xyzw[:, 0]
	qpos[:, 5] = quat_xyzw[:, 1]
	qpos[:, 6] = quat_xyzw[:, 2]
	return qpos


def load_robot_motion_fps(source_robot_config_path: Path, fps_override: float) -> float:
	if fps_override > 0.0:
		return float(fps_override)

	keypoints_path = load_path_config(source_robot_config_path, "keypoints_path").expanduser().resolve()
	if keypoints_path.exists():
		with keypoints_path.open("rb") as f:
			payload = pickle.load(f)
		fps = float(payload.get("fps", 0.0))
		if fps > 0.0:
			return fps
	return 30.0


def load_yaml_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	value = config.get(field_name)
	if not isinstance(value, list) or not value:
		raise ValueError(f"Missing or invalid list field '{field_name}' in: {config_path}")
	items = tuple(str(item).strip() for item in value if str(item).strip())
	if not items:
		raise ValueError(f"'{field_name}' must contain at least one body name in: {config_path}")
	return items


def load_scalar_bool_config(config_path: Path, field_name: str, default: bool = False) -> bool:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	value = config.get(field_name, default)
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		parsed = value.strip().lower()
		if parsed in {"true", "1", "yes", "on"}:
			return True
		if parsed in {"false", "0", "no", "off"}:
			return False
	raise ValueError(f"Missing or invalid bool field '{field_name}' in: {config_path}")


def load_effective_knee_angle_offset_degrees(config_path: Path) -> float:
	enabled = load_scalar_bool_config(
		config_path,
		"enable_knee_angle_offset_degrees",
		default=False,
	)
	if not enabled:
		return 0.0
	return load_scalar_float_config(config_path, "knee_angle_offset_degrees", default=0.0)


def validate_motion_shape(model: mujoco.MjModel, qpos_frames: np.ndarray, motion_file: Path) -> None:
	if qpos_frames.ndim != 2:
		raise ValueError(f"Motion CSV must be a 2D array, got {qpos_frames.shape}")
	if qpos_frames.shape[1] != model.nq:
		raise ValueError(
			f"Motion CSV qpos dim mismatch for {motion_file}: expected {model.nq}, got {qpos_frames.shape[1]}"
		)


def canonicalize_contact_name(body_name: str) -> str:
	lower = body_name.lower()
	if "left" in lower and "toe" in lower:
		return "left_toe"
	if "right" in lower and "toe" in lower:
		return "right_toe"
	if "left" in lower and "foot" in lower and "end" in lower:
		return "left_foot_end"
	if "right" in lower and "foot" in lower and "end" in lower:
		return "right_foot_end"
	if "left" in lower and ("hand" in lower or "wrist" in lower):
		return "left_hand"
	if "right" in lower and ("hand" in lower or "wrist" in lower):
		return "right_hand"
	return body_name


# 用途：将源机器人配置中的具体连杆端点转换为项目统一的语义身体节点，使不同机器人可复用同一套回放和重定向流程。
# 输入：source_robot_links 表示“语义连杆名 → 起点与终点 body 名称”的源机器人配置映射。
# 输出：返回统一的语义身体节点到具体 body 名称的映射；单个 body 表示普通节点，一对 body 表示由左右节点共同确定的中心节点。
def resolve_source_semantic_body_mapping(
	source_robot_links: dict[str, tuple[str, str]],
) -> dict[str, str | tuple[str, str]]:
	required_links = (
		"left_hip",
		"left_thigh",
		"left_calf",
		"right_hip",
		"right_thigh",
		"right_calf",
		"neck",
		"head",
		"left_shoulder",
		"left_arm",
		"left_fore_arm",
		"right_shoulder",
		"right_arm",
		"right_fore_arm",
	)
	# 先确认统一骨架所需的源连杆齐全，避免后续构造映射时因配置缺项产生不完整的关键点序列。
	for link_name in required_links:
		if link_name not in source_robot_links:
			raise ValueError(f"Missing source robot link required for replay: {link_name}")

	# 按统一骨架节点定义选择每条配置连杆的起点或终点；骨盆中心和肩部中心保留左右 body 对，供后续计算中心位置与平均朝向。
	return {
		"hips_mean": (source_robot_links["left_hip"][0], source_robot_links["right_hip"][0]),
		"hips": source_robot_links["left_hip"][0],
		"left_up_leg": source_robot_links["left_hip"][1],
		"left_leg": source_robot_links["left_thigh"][1],
		"left_foot": source_robot_links["left_calf"][1],
		"right_up_leg": source_robot_links["right_hip"][1],
		"right_leg": source_robot_links["right_thigh"][1],
		"right_foot": source_robot_links["right_calf"][1],
		"shoulder_mean": (
			source_robot_links["left_shoulder"][0],
			source_robot_links["right_shoulder"][0],
		),
		"spine1": source_robot_links["neck"][0],
		"spine2": source_robot_links["neck"][0],
		"chest": source_robot_links["neck"][1],
		"neck": source_robot_links["neck"][1],
		"head": source_robot_links["head"][1],
		"left_shoulder": source_robot_links["left_shoulder"][1],
		"left_arm": source_robot_links["left_shoulder"][1],
		"left_fore_arm": source_robot_links["left_arm"][1],
		"left_hand": source_robot_links["left_fore_arm"][1],
		"right_shoulder": source_robot_links["right_shoulder"][1],
		"right_arm": source_robot_links["right_shoulder"][1],
		"right_fore_arm": source_robot_links["right_arm"][1],
		"right_hand": source_robot_links["right_fore_arm"][1],
	}


# 用途：将 MJCF 中的 body 名称解析为 MuJoCo 模型内部索引，并在配置名称不匹配模型时尽早给出明确错误。
# 输入：model 表示已加载的 MuJoCo 机器人模型；body_name 表示配置文件中的目标刚体名称。
# 输出：返回 body_id，即该刚体用于访问位置、朝向等 MuJoCo 结果数组的非负整数索引；找不到时抛出异常而不返回索引。
def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
	# 限定查询对象为 body，避免与同名的关节、几何体或 site 混淆。
	body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
	if body_id < 0:
		# 让配置与 MJCF 模型名称不一致的问题在读取阶段暴露，避免后续以无效索引访问运动学结果。
		raise ValueError(f"Missing body in MJCF: {body_name}")
	return int(body_id)


# 用途：将源机器人关节状态的逐帧序列送入 MuJoCo 前向运动学，提取后续动作重定向与接触检测所需的身体节点位置和朝向。
# 输入：qpos_frames 表示每帧完整的 MuJoCo 广义坐标；source_robot_mjcf_path 表示源机器人的 MJCF 模型文件；source_robot_links 表示语义连杆与模型 body 名称的对应关系；contact_body_names 表示需要记录世界坐标的接触 body 名称。
# 输出：返回 positions（按统一语义节点顺序存放的逐帧世界坐标）、quaternions（这些节点对应的 wxyz 世界朝向），以及 contact_positions（用于后续速度和高度接触判定的逐帧接触 body 世界坐标）。
def build_robot_replay_buffers(
	qpos_frames: np.ndarray,
	source_robot_mjcf_path: Path,
	source_robot_links: dict[str, tuple[str, str]],
	contact_body_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	# 加载源机器人模型并校验每帧关节状态的维度，确保可安全写入 MuJoCo 的 qpos 状态向量。
	model = mujoco.MjModel.from_xml_path(str(source_robot_mjcf_path))
	data = mujoco.MjData(model)
	validate_motion_shape(model, qpos_frames, source_robot_mjcf_path)

	# 建立统一语义关键身体节点到数组序列号的索引，并读取目标输出所需的连杆顺序及双方关键帧坐标系修正参数。
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	# 建立统一语义关键身体节点到机器人具体 body 名称的映射。
	semantic_mapping = resolve_source_semantic_body_mapping(source_robot_links)
	# 预先把配置中的 body 名称转换为模型内部索引；部分语义节点由左右两个 body 的中心表示，因此保留一对索引。
	# 建立统一语义关键身体节点到具体 body 索引的映射，供后续循环中直接访问 MuJoCo 运动学结果。
	resolved_body_ids: dict[str, int | tuple[int, int]] = {}
	for semantic_name, body_spec in semantic_mapping.items():
		if isinstance(body_spec, tuple):
			resolved_body_ids[semantic_name] = tuple(get_body_id(model, name) for name in body_spec)
		else:
			resolved_body_ids[semantic_name] = get_body_id(model, body_spec)

	contact_body_ids = [get_body_id(model, body_name) for body_name in contact_body_names]
	# positions 形状为 [帧数, 语义节点数, 3]，末维依次存放世界坐标 x、y、z。
	# quaternions 形状为 [帧数, 语义节点数, 4]，末维依次存放 wxyz 朝向四元数分量。
	positions = np.zeros((qpos_frames.shape[0], len(REPLAY_BODY_NAMES), 3), dtype=np.float32)
	quaternions = np.broadcast_to(IDENTITY_QUAT_WXYZ, (qpos_frames.shape[0], len(REPLAY_BODY_NAMES), 4)).copy()
	contact_positions = np.zeros((qpos_frames.shape[0], len(contact_body_ids), 3), dtype=np.float32)

	# 对每帧关节状态执行前向运动学，使 data.xpos 与 data.xquat 更新为该姿态下所有 body 的世界位置和世界朝向。
	# 同时为逐帧回放创建终端进度条；它仍产出“帧编号、该帧关节状态”，只额外显示处理进度、总帧数和速度。
	# total 传入动作总帧数，desc 设置进度条说明文字，unit 指定计数单位为 frame。
	frame_iter = iter_progress(
		enumerate(qpos_frames),
		total=qpos_frames.shape[0],
		desc="Robot body replay",
		unit="frame",
	)
	for frame_idx, qpos in frame_iter:
		data.qpos[:] = qpos
		mujoco.mj_forward(model, data)

		# 将模型 body 的运动学结果写入统一骨架槽位；由两个 body 共同定义的中心节点同时取位置均值和平均朝向。
		# 为每一帧中的每个语义关键身体节点计算世界坐标和朝向，并存入 positions 与 quaternions。
		for semantic_name, slot in body_slots.items():
			if semantic_name not in resolved_body_ids:
				continue
			body_spec = resolved_body_ids[semantic_name]
			if isinstance(body_spec, tuple):
				left_id, right_id = body_spec
				positions[frame_idx, slot, :] = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
				quaternions[frame_idx, slot, :] = average_quaternions_wxyz(
					data.xquat[left_id][None, :].astype(np.float32),
					data.xquat[right_id][None, :].astype(np.float32),
				)[0]
			else:
				positions[frame_idx, slot, :] = data.xpos[body_spec]
				quaternions[frame_idx, slot, :] = data.xquat[body_spec]

		# 额外保留配置指定接触 body 的世界位置，供函数外部根据速度和离地高度识别支撑阶段。
		for contact_idx, body_id in enumerate(contact_body_ids):
			contact_positions[frame_idx, contact_idx, :] = data.xpos[body_id]

	return positions.astype(np.float32), quaternions.astype(np.float32), contact_positions


# 用途：
# 根据机器人各接触 body 在连续帧中的世界坐标，结合运动速度和离地高度，逐帧判定每个接触点是否处于支撑接触状态。
# 输入：
# - contact_positions：形状为 [帧数, 接触点数, 3] 的世界坐标；每个接触点由源机器人配置的 contact_links 指定，最后一维的第三个分量是高度。
# - fps：动作数据的采样帧率，用于将跨帧位移换算为每秒速度。
# - vel_window：计算速度时使用的时间窗口长度，用于在当前帧前后取样以减小逐帧位置噪声的影响。
# - vel_threshold：速度判定阈值；接触点的估计速度不高于该值才可视为静止。
# - height_threshold：高度判定阈值；接触点的世界坐标高度不高于该值才可视为接近地面。
# 输出：
# - 返回 contact_speeds：形状为 [帧数, 接触点数] 的浮点速度数组，以及 contact_states：相同形状的布尔接触掩码；只有同时低速且低于高度阈值的位置为 True。
def compute_robot_contact_sequence(
	contact_positions: np.ndarray,
	fps: float,
	vel_window: int,
	vel_threshold: float,
	height_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
	# 在每个接触点的相邻时间窗口内计算位移速度，得到用于抑制瞬时坐标抖动的逐帧运动量。
	contact_speeds = compute_windowed_point_speeds(
		contact_positions,
		fps=fps,
		window=vel_window,
		progress_desc="Contact speed",
	)
	# 仅当估计运动速度足够低，且世界坐标的竖直分量足够接近地面时，才将该帧标记为接触。
	contact_states = np.logical_and(
		contact_speeds <= float(vel_threshold),
		contact_positions[:, :, 2] <= float(height_threshold),
	)
	return contact_speeds.astype(np.float32), contact_states.astype(np.bool_)


def append_target_contact_keypoints(
	keypoints: np.ndarray,
	keypoint_quaternions: np.ndarray,
	anchor_raw_quaternions: np.ndarray,
	target_robot_links: dict[str, tuple[str, str]],
	target_robot_config_path: Path,
	target_robot_mjcf_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
	target_contact_links = load_yaml_body_list_config(target_robot_config_path, "contact_links")
	contact_name_to_body = {
		canonicalize_contact_name(body_name): body_name for body_name in target_contact_links
	}
	anchor_specs = (
		("left_foot_end", "left_calf"),
		("right_foot_end", "right_calf"),
		("left_toe", "left_calf"),
		("right_toe", "right_calf"),
	)
	keypoint_index = {name: idx for idx, name in enumerate(["hips_mean", *target_robot_links.keys()])}
	extra_positions: list[np.ndarray] = []
	extra_quaternions: list[np.ndarray] = []
	extra_names: list[str] = []

	for contact_name, anchor_link_name in anchor_specs:
		target_body_name = contact_name_to_body.get(contact_name)
		if target_body_name is None:
			continue
		if anchor_link_name not in target_robot_links:
			raise ValueError(f"Missing target robot link required for extra contact keypoints: {anchor_link_name}")
		anchor_body_name = target_robot_links[anchor_link_name][1]
		anchor_idx = keypoint_index[anchor_link_name]
		anchor_positions = keypoints[:, anchor_idx, :]
		anchor_quaternions = keypoint_quaternions[:, anchor_idx, :]
		anchor_raw_quat = anchor_raw_quaternions[:, anchor_idx, :]
		local_offset = compute_robot_body_local_offset(
			target_robot_mjcf_path,
			anchor_body=anchor_body_name,
			target_body=target_body_name,
		)
		world_offset = quat_rotate_vectors_wxyz(anchor_raw_quat, local_offset)
		extra_positions.append(anchor_positions + world_offset)
		extra_quaternions.append(anchor_quaternions)
		extra_names.append(target_body_name)

	if not extra_positions:
		return keypoints, keypoint_quaternions, extra_names

	extra_positions_arr = np.stack(extra_positions, axis=1)
	extra_quaternions_arr = np.stack(extra_quaternions, axis=1)
	return (
		np.concatenate([keypoints, extra_positions_arr], axis=1),
		np.concatenate([keypoint_quaternions, extra_quaternions_arr], axis=1),
		extra_names,
	)


def build_key_frame_adjustment_matrix(
	axis_map: np.ndarray,
	euler_xyz_degrees: np.ndarray,
) -> np.ndarray:
	if axis_map.shape != (3, 3):
		raise ValueError(f"axis_map shape must be (3, 3), got {axis_map.shape}")
	axis_map64 = axis_map.astype(np.float64)
	euler_rad = np.radians(euler_xyz_degrees.astype(np.float64))
	rx, ry, rz = float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2])
	rot_x = np.array(
		[[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]],
		dtype=np.float64,
	)
	rot_y = np.array(
		[[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]],
		dtype=np.float64,
	)
	rot_z = np.array(
		[[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
		dtype=np.float64,
	)
	return axis_map64 @ rot_x @ rot_y @ rot_z


def transform_key_frame_quaternions_wxyz(
	quaternions_wxyz: np.ndarray,
	source_axis_map: np.ndarray,
	source_euler_xyz_degrees: np.ndarray,
	target_axis_map: np.ndarray,
	target_euler_xyz_degrees: np.ndarray,
) -> np.ndarray:
	base_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
	base_mats = Rotation.from_quat(base_xyzw).as_matrix()
	source_adjust = build_key_frame_adjustment_matrix(source_axis_map, source_euler_xyz_degrees)
	target_adjust = build_key_frame_adjustment_matrix(target_axis_map, target_euler_xyz_degrees)
	adjusted_mats = np.einsum(
		"nij,jk,kl->nil",
		base_mats,
		np.linalg.inv(source_adjust),
		target_adjust,
	)
	adjusted_xyzw = Rotation.from_matrix(adjusted_mats).as_quat()
	return adjusted_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


# 用途：将源机器人的逐帧身体关键点与朝向，按目标机器人的连杆尺寸和坐标系约定完成重定向，并整理为可导出的固定顺序数据。
# 输入：source_positions 表示源机器人每一帧各语义身体节点的世界坐标；source_quaternions 表示对应节点的 wxyz 朝向四元数。
# 输入：target_robot_link_lengths 表示目标机器人各语义连杆在默认姿态下的长度；source_robot_config_path 与 target_robot_config_path 分别提供源、目标机器人的关键帧坐标系及连杆配置。
# 输出：依次返回目标顺序的关键点坐标、经过坐标系变换的关键点四元数、同一批用于后续接触点锚定的原始四元数、源连杆向量、各连杆缩放比例，以及缩放比例是否为静态值的标记。
def build_robot_retarget_keypoints(
	source_positions: np.ndarray,
	source_quaternions: np.ndarray,
	target_robot_link_lengths: dict[str, float],
	source_robot_config_path: Path,
	target_robot_config_path: Path,
) -> tuple[
		np.ndarray,
		np.ndarray,
		np.ndarray,
		dict[str, np.ndarray],
		dict[str, float | np.ndarray],
		dict[str, bool],
	]:
	# 建立统一语义关键身体节点到数组列号的索引，并读取目标输出所需的连杆顺序及双方关键帧坐标系修正参数。
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	target_robot_links = load_robot_links_config(target_robot_config_path)
	source_key_frame_offset_degrees, source_key_frame_axis_map = load_key_frame_config(
		source_robot_config_path,
		section_name="key_frame_config",
	)
	knee_angle_offset_degrees = load_effective_knee_angle_offset_degrees(target_robot_config_path)
	key_frame_offset_degrees, key_frame_axis_map = load_key_frame_config(
		target_robot_config_path,
		section_name="key_frame_config",
	)
	# 从源语义连杆表提取左右腿“髋—膝—脚”的有序节点链，供后续缩放后计算膝盖预弯时定位三处关节。
	# 膝盖预弯是指：在动作重定向时，给原本完全伸直或接近伸直的腿，预先加入一个很小的屈膝角度。
	left_leg_links = derive_leg_body_chain_from_links(SOURCE_SEMANTIC_LINKS, "left")
	right_leg_links = derive_leg_body_chain_from_links(SOURCE_SEMANTIC_LINKS, "right")
	# 从源动作逐帧提取语义连杆的长度、方向和端点关系，再以目标机器人尺寸计算缩放比例。
	# 返回值类型：设：
	# - 动作帧数为 F = source_positions.shape[0]；
	# - 语义连杆数量为 L = len(SOURCE_SEMANTIC_LINKS)
	# source_link_lengths 为 dict[str, np.ndarray]，每个连杆名称对应一个形状为：(F,) 的一维数组，表示该连杆在每一帧的长度。
	# source_link_vectors 为 dict[str, np.ndarray]，每个连杆名称对应一个形状为：(F,3) 的数组，表示每一帧从父节点指向子节点的三维向量。
	# resolved_source_links 为 dict[str, tuple[str, str]]。每个连杆名称对应一个元组，表示该连杆的父节点和子节点。
	source_link_lengths, source_link_vectors, resolved_source_links = compute_link_geometry_from_positions(
		SOURCE_SEMANTIC_LINKS,
		body_slots,
		source_positions,
	)
	# link_scales 为 dict[str, float | np.ndarray]，每个连杆名称对应目标连杆长度与源连杆长度的比值；当前实现中该值为形状 (F,) 的 float32 数组，表示每一帧的缩放比例。
	# link_scale_is_static 为 dict[str, bool]，每个连杆名称对应一个布尔标记，用于表示该连杆的缩放比例是否在所有帧中保持不变；当前实现均为 False。
	link_scales, link_scale_is_static = compute_link_scale_factors(
		target_robot_link_lengths,
		source_link_lengths,
	)

	# 按目标机器人与源机器人的比例完成重定向
	# retarget_keypoints 为 np.ndarray，形状为 (F, B, 3)，表示各源语义节点按目标连杆尺寸缩放并施加膝盖预弯后的逐帧世界坐标。
	# retarget_quaternions 为 np.ndarray，形状为 (F, B, 4)，表示与缩放后节点位置对应的逐帧 wxyz 朝向四元数。
	retarget_keypoints, retarget_quaternions = apply_link_scales_to_positions(
		source_positions,
		resolved_source_links,
		body_slots,
		link_scales,
		source_link_vectors,
		knee_angle_offset_degrees,
		left_leg_links,
		right_leg_links,
		quaternions=source_quaternions,
	)

	# 输出数组的第 0 项固定为骨盆中心；其余项按目标机器人配置的连杆顺序排列，保证下游保存和回放时的索引一致。
	# 三个数组形状均为 [帧数, 目标连杆数加 1, 分量数]；位置末维为 3，两个四元数数组末维为 4。
	ordered_keypoints = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 3), dtype=np.float32)
	ordered_quaternions = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 4), dtype=np.float32)
	ordered_raw_quaternions = np.zeros((source_positions.shape[0], len(target_robot_links) + 1, 4), dtype=np.float32)
	ordered_keypoints[:, 0, :] = retarget_keypoints[:, body_slots["hips_mean"], :]
	# 读取源、目标配置对骨盆中心定义的坐标轴映射与欧拉角偏置；未配置时分别采用单位映射和零偏置。
	source_hips_mean_offset = source_key_frame_offset_degrees.get("hips_mean", np.zeros(3, dtype=np.float32))
	source_hips_mean_axis_map = source_key_frame_axis_map.get("hips_mean", np.eye(3, dtype=np.float32))
	hips_mean_offset = key_frame_offset_degrees.get("hips_mean", np.zeros(3, dtype=np.float32))
	hips_mean_axis_map = key_frame_axis_map.get("hips_mean", np.eye(3, dtype=np.float32))

	# 将骨盆中心的朝向由源关键帧坐标系转换到目标关键帧坐标系；该版本同时保留给后续接触点锚定使用。
	ordered_raw_quaternions[:, 0, :] = transform_key_frame_quaternions_wxyz(
		retarget_quaternions[:, body_slots["hips_mean"], :],
		source_hips_mean_axis_map,
		source_hips_mean_offset,
		hips_mean_axis_map,
		hips_mean_offset,
	)
	ordered_quaternions[:, 0, :] = ordered_raw_quaternions[:, 0, :]

	# 输出约定为“骨盆中心 + 每条连杆的子端点”：hips_mean 作为根节点已在循环外固定写入第 0 项；循环只写入子端点，因为连杆的父端点要么是根节点，要么已作为另一条连杆的子端点写入，避免重复保存共享节点。
	# 同时将每个子端点的朝向由源关键帧坐标系转换到目标关键帧坐标系。
	for keypoint_idx, link_name in enumerate(target_robot_links, start=1):
		if link_name not in resolved_source_links:
			raise ValueError(f"Missing source semantic link for target robot link: {link_name}")
		_parent_body, child_body = resolved_source_links[link_name]
		child_idx = body_slots[child_body]
		ordered_keypoints[:, keypoint_idx, :] = retarget_keypoints[:, child_idx, :]
		source_euler_offset = source_key_frame_offset_degrees.get(child_body, np.zeros(3, dtype=np.float32))
		source_axis_map = source_key_frame_axis_map.get(child_body, np.eye(3, dtype=np.float32))
		euler_offset = key_frame_offset_degrees.get(child_body, np.zeros(3, dtype=np.float32))
		axis_map = key_frame_axis_map.get(child_body, np.eye(3, dtype=np.float32))
		ordered_raw_quaternions[:, keypoint_idx, :] = transform_key_frame_quaternions_wxyz(
			retarget_quaternions[:, child_idx, :],
			source_axis_map,
			source_euler_offset,
			axis_map,
			euler_offset,
		)
		ordered_quaternions[:, keypoint_idx, :] = ordered_raw_quaternions[:, keypoint_idx, :]

	return (
		ordered_keypoints,
		ordered_quaternions,
		ordered_raw_quaternions,
		source_link_vectors,
		link_scales,
		link_scale_is_static,
	)


def offset_keypoints_by_contact_height(
	keypoints: np.ndarray,
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
	contact_positions: np.ndarray,
	contact_states: np.ndarray,
	height_lpf_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
	height_offsets = compute_contact_height_offsets(
		keypoints=keypoints,
		keypoint_names=keypoint_names,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_states=contact_states,
		height_lpf_alpha=height_lpf_alpha,
	)
	adjusted_keypoints = keypoints.copy()
	adjusted_keypoints[:, :, 2] -= height_offsets[:, None]
	return adjusted_keypoints.astype(np.float32), height_offsets.astype(np.float32)


def build_output_path(
	output_path: Path | None,
	motion_file: Path,
	source_robot_config_path: Path,
	target_robot_config_path: Path,
) -> Path:
	if output_path is not None:
		return output_path.expanduser().resolve()
	return (
		(DEFAULT_OUTPUT_DIR / target_robot_config_path.stem / f"{motion_file.stem}_from_{source_robot_config_path.stem}_keypoints.pkl")
		.expanduser()
		.resolve()
	)


def load_model_with_ground(xml_file: Path) -> mujoco.MjModel:
	"""Load a robot MJCF and add a visual ground plane plus skybox."""
	spec = mujoco.MjSpec.from_file(str(xml_file))
	existing_tex = {texture.name for texture in spec.textures}
	existing_mat = {material.name for material in spec.materials}

	if "skybox" not in existing_tex:
		spec.add_texture(
			name="skybox",
			type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
			builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
			rgb1=[0.3, 0.5, 0.7],
			rgb2=[0.0, 0.0, 0.0],
			width=512,
			height=512,
		)
	if "groundplane" not in existing_tex:
		spec.add_texture(
			name="groundplane",
			type=mujoco.mjtTexture.mjTEXTURE_2D,
			builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
			rgb1=[0.2, 0.3, 0.4],
			rgb2=[0.1, 0.2, 0.3],
			width=300,
			height=300,
		)
	if "groundplane" not in existing_mat:
		spec.add_material(
			name="groundplane",
			textures=["", "groundplane"],
			texrepeat=[5, 5],
			texuniform=True,
			reflectance=0.2,
		)

	spec.worldbody.add_light(
		pos=[0, 0, 20.0],
		dir=[0, 0, -1],
		type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
		diffuse=[0.7, 0.7, 0.7],
		specular=[0.3, 0.3, 0.3],
	)
	spec.worldbody.add_light(
		pos=[4, 4, 6.0],
		dir=[-0.5, -0.5, -1],
		type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
		diffuse=[0.4, 0.4, 0.4],
		specular=[0.1, 0.1, 0.1],
	)

	ground = spec.worldbody.add_geom(
		name="ground",
		type=mujoco.mjtGeom.mjGEOM_PLANE,
		size=[0, 0, 0.05],
		material="groundplane",
		pos=[0, 0, 0],
	)
	ground.contype = 0
	ground.conaffinity = 0
	return spec.compile()


def print_robot_summary(
	motion_file: Path,
	source_robot_config_path: Path,
	target_robot_config_path: Path,
	positions: np.ndarray,
	fps: float,
	selected_frames: np.ndarray,
	output_path: Path,
	contact_names: tuple[str, ...],
) -> None:
	duration = positions.shape[0] / fps if fps > 0 else 0.0
	print(f"motion_file: {motion_file}")
	print(f"source_robot_config: {source_robot_config_path}")
	print(f"target_robot_config: {target_robot_config_path}")
	print(f"frames_total: {positions.shape[0]}")
	print(f"frames_selected: {selected_frames.shape[0]}")
	print(f"fps: {fps:.3f}")
	print(f"duration_sec: {duration:.3f}")
	print(f"output_keypoints: {output_path}")
	print(f"contact_names: {list(contact_names)}")


def play_robot_motion(
	model: mujoco.MjModel,
	data: mujoco.MjData,
	qpos_frames: np.ndarray,
	retarget_keypoints: np.ndarray | None,
	retarget_quaternions: np.ndarray | None,
	contact_positions: np.ndarray | None,
	contact_states: np.ndarray | None,
	fps: float,
	frame_ids: np.ndarray,
	loop: bool,
) -> None:
	frame_time = 1.0 / fps
	num_frames = int(frame_ids.shape[0])
	state = {
		"paused": False,
		"cursor": 0,
		"step_delta": 0,
		"resync_clock": False,
	}
	state_lock = threading.Lock()

	def key_callback(keycode: int) -> None:
		with state_lock:
			if keycode == glfw.KEY_SPACE:
				state["paused"] = not state["paused"]
				state["resync_clock"] = True
			elif keycode == glfw.KEY_COMMA:
				state["paused"] = True
				state["step_delta"] -= 1
				state["resync_clock"] = True
			elif keycode == glfw.KEY_PERIOD:
				state["paused"] = True
				state["step_delta"] += 1
				state["resync_clock"] = True
			elif keycode == glfw.KEY_LEFT_BRACKET:
				state["paused"] = True
				state["step_delta"] -= 10
				state["resync_clock"] = True
			elif keycode == glfw.KEY_RIGHT_BRACKET:
				state["paused"] = True
				state["step_delta"] += 10
				state["resync_clock"] = True
			elif keycode == glfw.KEY_R:
				state["cursor"] = 0
				state["step_delta"] = 0
				state["resync_clock"] = True

	def render_cursor(cursor: int, viewer=None) -> None:
		frame_idx = int(frame_ids[cursor])
		data.qpos[:] = qpos_frames[frame_idx]
		mujoco.mj_forward(model, data)
		if viewer is not None and retarget_keypoints is not None:
			update_viewer_keypoints(
				viewer,
				retarget_keypoints[frame_idx],
				None if retarget_quaternions is None else retarget_quaternions[frame_idx],
				contact_keypoints=None if contact_positions is None else contact_positions[frame_idx],
				contact_states=None if contact_states is None else contact_states[frame_idx],
			)

	print("Controls: Space play/pause, ',' back 1, '.' forward 1, '[' back 10, ']' forward 10, 'R' reset")
	with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
		last_advance_time = time.perf_counter()
		last_progress_line = ""
		while viewer.is_running():
			frame_start = time.perf_counter()
			with state_lock:
				paused = bool(state["paused"])
				cursor = int(state["cursor"])
				step_delta = int(state["step_delta"])
				resync_clock = bool(state["resync_clock"])
				state["step_delta"] = 0
				state["resync_clock"] = False

			if resync_clock:
				last_advance_time = time.perf_counter()

			if step_delta != 0:
				cursor = advance_frame_cursor(cursor, step_delta, num_frames, loop)
				paused = True
				with state_lock:
					state["paused"] = True
					state["cursor"] = cursor
				last_advance_time = time.perf_counter()
			elif not paused:
				now = time.perf_counter()
				if now - last_advance_time >= frame_time:
					if cursor == num_frames - 1 and not loop:
						render_cursor(cursor, viewer)
						viewer.sync()
						return
					cursor = advance_frame_cursor(cursor, 1, num_frames, loop)
					with state_lock:
						state["cursor"] = cursor
					last_advance_time = now

			render_cursor(cursor, viewer)
			progress_line = format_progress_line(cursor, num_frames, frame_ids, paused)
			if progress_line != last_progress_line:
				sys.stdout.write(progress_line)
				sys.stdout.flush()
				last_progress_line = progress_line
			viewer.sync()

			elapsed = time.perf_counter() - frame_start
			if paused or step_delta != 0:
				time.sleep(0.01)
			else:
				remaining = frame_time - elapsed
				if remaining > 0.0:
					time.sleep(remaining)
		sys.stdout.write("\n")
		sys.stdout.flush()


def main() -> None:
	args = parse_args()
	# 将命令行提供的源、目标机器人 YAML 路径统一为可定位的绝对路径，供后续加载配置、推导输出文件名及打印摘要共用。
	source_robot_config_path = args.source_robot_config.expanduser().resolve()
	target_robot_config_path = args.target_robot_config.expanduser().resolve()
	motion_file = resolve_motion_file(args.motion_file, source_robot_config_path)

	source_robot_mjcf_path = load_path_config(source_robot_config_path, "robot_xml_path").expanduser().resolve()
	target_robot_mjcf_path = load_path_config(target_robot_config_path, "robot_xml_path").expanduser().resolve()
	# 建立“统一的语义连杆名 → 起点与终点 body 名称”的对应映射关系
	source_robot_links = load_robot_links_config(source_robot_config_path)
	target_robot_links = load_robot_links_config(target_robot_config_path)
	source_contact_links = load_yaml_body_list_config(source_robot_config_path, "contact_links")
	contact_names = tuple(canonicalize_contact_name(name) for name in source_contact_links)

	# 加载形状为 [帧数, nq] 的动作数组；第 0 维为 CSV 中的动作帧数，第 1 维为每帧完整的 MuJoCo 广义坐标。
	qpos_frames = load_robot_motion_csv(motion_file)
	# 对源机器人执行 FK 计算
	positions, quaternions, contact_positions = build_robot_replay_buffers(
		qpos_frames=qpos_frames,
		source_robot_mjcf_path=source_robot_mjcf_path,
		source_robot_links=source_robot_links,
		contact_body_names=source_contact_links,
	)
	fps = load_robot_motion_fps(source_robot_config_path, args.fps)
	contact_vel_window = load_scalar_int_config(
		source_robot_config_path,
		"contact_vel_calculate_window",
		default=6,
	)
	contact_vel_threshold = load_scalar_float_config(
		source_robot_config_path,
		"contact_vel_threshold",
		default=0.5,
	)
	contact_height_threshold = load_scalar_float_config(
		source_robot_config_path,
		"contact_height_threshold",
		default=0.025,
	)
	contact_height_lpf_alpha = load_scalar_float_config(
		source_robot_config_path,
		"contact_height_lpf_alpha",
		default=0.2,
	)
	contact_speeds, contact_states = compute_robot_contact_sequence(
		contact_positions=contact_positions,
		fps=fps,
		vel_window=contact_vel_window,
		vel_threshold=contact_vel_threshold,
		height_threshold=contact_height_threshold,
	)

	target_robot_link_lengths = compute_robot_link_lengths(
		config_path=target_robot_config_path,
		robot_mjcf_path=target_robot_mjcf_path,
	)
	# 将源机器人的逐帧身体关键点与朝向，按目标机器人的连杆尺寸和坐标系约定完成重定向，并整理为可导出的固定顺序数据。
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		retarget_raw_quaternions,
		source_link_vectors,
		link_scales,
		link_scale_is_static,
	) = build_robot_retarget_keypoints(
		source_positions=positions,
		source_quaternions=quaternions,
		target_robot_link_lengths=target_robot_link_lengths,
		source_robot_config_path=source_robot_config_path,
		target_robot_config_path=target_robot_config_path,
	)
	# 此处不再修改已重定向关键点；重新取同一预弯参数，仅用于后续按预弯后的有效腿长缩放根节点位移。
	knee_angle_offset_degrees = load_effective_knee_angle_offset_degrees(target_robot_config_path)
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		extra_keypoint_names,
	) = append_target_contact_keypoints(
		keypoints=retarget_keypoints,
		keypoint_quaternions=retarget_keypoint_quaternions,
		anchor_raw_quaternions=retarget_raw_quaternions,
		target_robot_links=target_robot_links,
		target_robot_config_path=target_robot_config_path,
		target_robot_mjcf_path=target_robot_mjcf_path,
	)
	# 预弯会缩短髋到脚的有效伸展距离，因此用该角度计算步幅缩放，保证全局移动尺度与当前腿形一致。
	leg_displacement_scale, robot_leg_length, source_leg_length = compute_leg_displacement_scale(
		robot_link_lengths=target_robot_link_lengths,
		skeleton_link_vectors=source_link_vectors,
		knee_angle_offset_degrees=knee_angle_offset_degrees,
	)
	retarget_keypoints = scale_keypoint_frame_displacements(
		keypoints=retarget_keypoints,
		displacement_scale=leg_displacement_scale,
		root_keypoint_idx=0,
	)
	keypoint_names = ["hips_mean", *list(target_robot_links.keys()), *extra_keypoint_names]
	retarget_keypoints, contact_height_offsets = offset_keypoints_by_contact_height(
		keypoints=retarget_keypoints,
		keypoint_names=keypoint_names,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_states=contact_states,
		height_lpf_alpha=contact_height_lpf_alpha,
	)
	keypoint_output_path = build_output_path(
		output_path=args.output_path,
		motion_file=motion_file,
		source_robot_config_path=source_robot_config_path,
		target_robot_config_path=target_robot_config_path,
	)
	save_keypoints_pkl(
		output_path=keypoint_output_path,
		keypoint_names=keypoint_names,
		positions=retarget_keypoints,
		quaternions=retarget_keypoint_quaternions,
		fps=fps,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_speeds=contact_speeds,
		contact_states=contact_states,
		contact_vel_window=contact_vel_window,
		contact_vel_threshold=contact_vel_threshold,
		contact_height_threshold=contact_height_threshold,
	)

	frame_ids = select_frame_slice(positions.shape[0], args.start_frame, args.end_frame, args.stride)
	if args.print_summary:
		print_robot_summary(
			motion_file=motion_file,
			source_robot_config_path=source_robot_config_path,
			target_robot_config_path=target_robot_config_path,
			positions=positions,
			fps=fps,
			selected_frames=frame_ids,
			output_path=keypoint_output_path,
			contact_names=contact_names,
		)
		print(f"contact_states_shape: {contact_states.shape}")
		print(f"contact_height_lpf_alpha: {contact_height_lpf_alpha}")
		print(f"knee_angle_offset_degrees: {knee_angle_offset_degrees}")
		print(f"robot_leg_length: {robot_leg_length}")
		print(f"source_leg_length: {source_leg_length}")
		print(f"leg_displacement_scale: {leg_displacement_scale}")
		print(f"source_robot_mjcf_path: {source_robot_mjcf_path}")
		print(f"target_robot_mjcf_path: {target_robot_mjcf_path}")
		print(f"target_robot_link_lengths: {target_robot_link_lengths}")
		print(f"retarget_keypoints_shape: {retarget_keypoints.shape}")
		print(f"retarget_keypoint_quaternions_shape: {retarget_keypoint_quaternions.shape}")
		print(f"contact_height_offsets_shape: {contact_height_offsets.shape}")
		print(f"source_link_vectors: {source_link_vectors}")
		print(f"link_scales: {link_scales}")
		print(f"link_scale_is_static: {link_scale_is_static}")

	if args.no_viewer:
		return

	# 为源机器人创建带地面的 MuJoCo 模型及其运行时状态，作为 CSV qpos 帧的回放载体。
	source_model = load_model_with_ground(source_robot_mjcf_path)
	source_data = mujoco.MjData(source_model)
	# 在查看器中按 fps 写入每帧关节状态；同时叠加前面生成的目标机器人关键点和接触信息。
	play_robot_motion(
		model=source_model,
		data=source_data,
		qpos_frames=qpos_frames,
		retarget_keypoints=retarget_keypoints,
		retarget_quaternions=retarget_keypoint_quaternions,
		contact_positions=contact_positions,
		contact_states=contact_states,
		# frame_ids 控制回放的起止范围和抽帧；loop 决定到末帧后是否从头继续播放。
		fps=fps,
		frame_ids=frame_ids,
		loop=args.loop,
	)


if __name__ == "__main__":
	main()
