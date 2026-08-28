"""Retarget keypoint motions to a robot and export MuJoCo qpos CSV.

This script loads robot and keypoint settings from a robot YAML config,
runs IK-based retargeting frame by frame, and saves the resulting motion
as a CSV under output_data/robot_motion.

Usage:
    # Run with a specific robot config from terminal
    python scripts/robot_retarget.py --config config/robot/h2.yaml

    # Override render_debug from terminal (highest priority)
    python scripts/robot_retarget.py --config config/robot/agibot_x2.yaml --render-debug

    # Override keypoints_path from terminal by motion stem only
    python scripts/robot_retarget.py --config config/robot/h2.yaml --keypoints-name body_check_001__A548_M_from_g1
"""

import argparse
import mink
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import pickle
import time
import csv
import copy
import os
import glfw
from collections.abc import Mapping, Sequence
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global pause state and keyboard callback
_PAUSED = False

DEFAULT_TEMPORAL_JOINT_SUBSTRINGS = ("shoulder", "elbow")


def _joint_dof_width(joint_type):
    """Return the MuJoCo tangent-space width of one joint."""
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def _finite_number(value, setting_name, strictly_positive=False):
    """Validate a scalar numeric YAML value and return it as a float."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{setting_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{setting_name} must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"{setting_name} must be a finite number")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{setting_name} must be > 0")
    return number


def _enabled_flag(config, setting_name):
    """Read an optional YAML switch without accepting truthy non-boolean values."""
    enabled = config.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError(f"{setting_name}.enabled must be a bool")
    return bool(enabled)


def _joint_dof_indices_for_substrings(model, substrings, setting_name):
    """Return non-free DoF indices selected by configured joint-name substrings."""
    if not isinstance(substrings, Sequence) or isinstance(substrings, (str, bytes)):
        raise ValueError(f"{setting_name}.joint_name_substrings must be a list of strings")
    if not substrings:
        raise ValueError(f"{setting_name}.joint_name_substrings must not be empty")

    indices = []
    for substring in substrings:
        if not isinstance(substring, str) or not substring:
            raise ValueError(f"{setting_name}.joint_name_substrings must contain non-empty strings")
        matched = False
        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None or substring not in joint_name:
                continue
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_start = model.jnt_dofadr[joint_id]
            dof_end = dof_start + _joint_dof_width(model.jnt_type[joint_id])
            indices.extend(range(dof_start, dof_end))
            matched = True
        if not matched:
            raise ValueError(
                f"{setting_name}.joint_name_substrings contains unknown joint substring: {substring!r}"
            )
    return np.asarray(sorted(set(indices)), dtype=int)


class FrameVelocityLimit(mink.Limit):
    """Per-data-frame displacement budget for configured non-free joint DoFs.

    ``begin_frame`` fixes the qpos anchor.  Each QP then receives only the
    remaining part of ``max_velocity / fps``; unlike ``mink.VelocityLimit``,
    repeated IK iterations cannot exceed the frame-wide budget.

    Free joints (including a floating base) are deliberately not selected by
    this feature, matching Mink's ``VelocityLimit`` joint-velocity semantics.
    """

    def __init__(self, model, indices, max_velocities, fps):
        fps = _finite_number(fps, "inter_frame_velocity_limit.fps", strictly_positive=True)
        self.model = model
        self.indices = np.asarray(indices, dtype=int)
        self.max_velocities = np.asarray(max_velocities, dtype=float)
        if self.indices.ndim != 1 or self.max_velocities.shape != self.indices.shape:
            raise ValueError("FrameVelocityLimit indices and max_velocities must have matching shapes")
        if not np.all(np.isfinite(self.max_velocities)) or np.any(self.max_velocities <= 0.0):
            raise ValueError("inter_frame_velocity_limit velocities must be finite and > 0")
        self.projection_matrix = np.eye(model.nv)[self.indices]
        self.frame_budget = self.max_velocities / fps
        self.previous_q = None

    def begin_frame(self, previous_q):
        previous_q = np.asarray(previous_q, dtype=float)
        if previous_q.shape != (self.model.nq,):
            raise ValueError(
                f"FrameVelocityLimit previous_q must have shape ({self.model.nq},), got {previous_q.shape}"
            )
        self.previous_q = previous_q.copy()

    def compute_qp_inequalities(self, configuration, dt):
        del dt  # The budget is defined by source-data fps, not IK integration dt.
        if self.previous_q is None:
            raise RuntimeError("FrameVelocityLimit.begin_frame must be called before solve_ik")
        used_displacement = np.empty(self.model.nv)
        mujoco.mj_differentiatePos(
            self.model,
            used_displacement,
            1.0,
            self.previous_q,
            configuration.q,
        )
        used = used_displacement[self.indices]
        # G * delta_q <= h.  Both signs make the remaining frame budget explicit.
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([self.frame_budget - used, self.frame_budget + used])
        return mink.Constraint(G=G, h=h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget keypoint motions to a robot from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("config", "robot", "h2.yaml"),
        help="Robot YAML config path (default: config/robot/h2.yaml)",
    )
    parser.add_argument(
        "--keypoints-name",
        type=str,
        default=None,
        help=(
            "Override keypoints_path by motion stem only. "
            "Example: --keypoints-name body_check_001__A548_M_from_g1 -> "
            "output_data/keypoints/<config_stem>/<name>_keypoints.pkl"
        ),
    )
    render_debug_group = parser.add_mutually_exclusive_group()
    render_debug_group.add_argument(
        "--render-debug",
        dest="render_debug",
        action="store_true",
        help="Force enable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    render_debug_group.add_argument(
        "--no-render-debug",
        dest="render_debug",
        action="store_false",
        help="Force disable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    parser.set_defaults(render_debug=None)
    return parser.parse_args()


def key_callback(keycode: int) -> None:
    """Toggle pause/play with the space key."""
    global _PAUSED
    if keycode == glfw.KEY_SPACE:
        _PAUSED = not _PAUSED


class RobotRetarget:

    LEGACY_CONTACT_CONFIG_TO_NAMES = {
        "foot_end_link": ("left_foot_end", "right_foot_end"),
        "toe_link": ("left_toe", "right_toe"),
        "hand_link": ("left_hand", "right_hand"),
    }

    def __init__(
        self,
        model_path: str,
        keypoint_path: str,
        ik_match_table: dict,
        solver: str = "daqp",
        verbose: bool = False,
        damping: float = 1.0,
        render_debug: bool = False,
        joints_limit_offset_degrees: dict | None = None,
        contact_body_names: list | tuple | dict | None = None,
        contact_position_cost: float = 10.0,
        temporal_continuity_regularization: dict | None = None,
        inter_frame_velocity_limit: dict | None = None,
    ):
        self.xml_file = model_path
        self.keypoint_path = keypoint_path
        self.ik_match_table = ik_match_table
        self.model = self._load_model_with_ground(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.max_iter = 50
        self.damping = damping
        self.render_debug = render_debug
        self.joints_limit_offset_degrees = joints_limit_offset_degrees or {}
        self.contact_body_names = self._normalize_contact_body_names(contact_body_names)
        self.contact_position_cost = float(contact_position_cost)
        self.temporal_continuity_regularization = temporal_continuity_regularization
        self.inter_frame_velocity_limit_config = inter_frame_velocity_limit
        self.verbose = verbose
        self.solver = solver
        self.human_body_to_task = {}
        self.task_errors = {}
        self.result_pos = []
        self.frame_ik_iteration_counts = []
        self.contact_state_name_to_idx = {}
        self.keypoint_name_to_idx = {}
        self.body_name_to_source_keypoint = {}
        self.body_name_to_contact_task = {}
        self.contact_targets = []
        self.current_contact_points = []
        self.temporal_posture_task = None
        self.frame_velocity_limit = None
        self.convergence_tasks = []

        self.robot_motor_names = {}

        # 设置 IK 求解器的参数
        self.setup_retarget_configuration()
        # 读取上游流程写出的关键点 pkl 文件，并准备后续逐帧 IK 所需的位姿、接触和时间信息。
        self.load_keypoints()
        self.setup_contact_targets()
        # Posture regularization should guide the QP but not change the legacy
        # task-error stopping criterion.
        self.convergence_tasks = list(self.tasks)
        # 建立可选的时间连续性任务与帧间硬速度约束；缺省配置保持旧行为。
        self.setup_temporal_constraints()

    def _normalize_contact_body_names(self, contact_body_names):
        if contact_body_names is None:
            return []
        if isinstance(contact_body_names, dict):
            flattened_body_names = []
            for field_name, contact_names in self.LEGACY_CONTACT_CONFIG_TO_NAMES.items():
                body_names = contact_body_names.get(field_name, [])
                if len(body_names) != len(contact_names):
                    continue
                flattened_body_names.extend(body_names)
            return flattened_body_names
        if isinstance(contact_body_names, (list, tuple)):
            return list(contact_body_names)
        raise TypeError(
            "contact_body_names must be a list/tuple of body names or a legacy contact-body mapping"
        )

    def _load_model_with_ground(self, xml_file: str) -> mujoco.MjModel:
        """Load the robot XML and add a MuJoCo default-style ground (checker plane + skybox).

        The ground geom has collision disabled (contype/conaffinity=0) and serves only as a
        visual reference; it does not affect mink IK's pure kinematic solving.
        """
        spec = mujoco.MjSpec.from_file(xml_file)

        # Skip if ground/skybox assets already exist to avoid name conflicts
        existing_tex = {t.name for t in spec.textures}
        existing_mat = {m.name for m in spec.materials}

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

        # Lighting: main directional light + angled fill light to avoid an overly dark scene
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

        # Add the (infinite) ground plane, collision disabled, for visualization only
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

    # 用途：读取上游流程导出的关键点 pkl 文件，并将其中的运动、接触和帧率数据整理为重定向器的逐帧输入。
    # 输入：self 表示已保存的 keypoint_path（待加载关键点 pkl 文件路径）
    # 输出：不直接返回数值；会更新 self.keypoints_pos（逐帧关键点位置）、self.keypoints_quat（逐帧关键点朝向）、接触状态、名称索引及帧数和时间间隔等对象状态。
    def load_keypoints(self):
        # 以二进制反序列化方式恢复 pkl 中保存的字典，其中包含位置、朝向、关键点名称和接触信息。
        with open(self.keypoint_path, "rb") as f:
                keypoints_data = pickle.load(f)
        # 建立“关键点名称到数组列号”的索引，使后续能按配置名称快速提取某一关键点的逐帧位姿。
        self.keypoint_names = keypoints_data["keypoint_names"]
        self.keypoint_name_to_idx = {
            keypoint_name: idx for idx, keypoint_name in enumerate(self.keypoint_names)
        }
        # 读取每帧的三维位置与 wxyz 四元数朝向；两者的前两维分别对应相同的动作帧和关键点。
        self.keypoints_pos = keypoints_data["positions"] 
        self.keypoints_quat = keypoints_data["quaternions"]  
        # 建立接触点名称索引并读取每帧布尔接触状态，供支撑期末端位置锁定约束使用。
        self.contact_names = keypoints_data.get("contact_names", [])
        self.contact_state_name_to_idx = {
            contact_name: idx for idx, contact_name in enumerate(self.contact_names)
        }
        self.contact_seq = keypoints_data["contact_states"]
        # 使用文件记录的帧率；旧数据未记录时按 30 FPS 解释，并由位置数组的形状确定总帧数和关键点数。
        self.fps = _finite_number(keypoints_data.get("fps", 30), "keypoints fps", strictly_positive=True)
        self.time_step = 1.0 / self.fps
        self.num_frames = self.keypoints_pos.shape[0]
        self.num_keypoints = self.keypoints_pos.shape[1]

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.robot_frame_names = []  # Robot body name corresponding to each task

        # Apply offsets to the configured joint limits: raise the lower bound
        # 在进行 IK 求解之前收紧关节的可行范围，在 DR02 中，我们将收紧膝关节与肘关节的下限，避免 IK 求解中接近可能会导致腿/臂反向伸展的极端姿态。
        self._apply_joints_limit_offset()
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        # self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        for keypoint_name, entry in self.ik_match_table.items():
            robot_frame, pos_weight, rot_weight = entry
            if pos_weight != 0 or rot_weight != 0:
                # 为目标机器人中 robot_frame 所表示的 body 创建位姿跟踪任务；
                # pos_weight 和 rot_weight 分别决定该 body 跟随源关键点位置与朝向的相对优先级。
                task = mink.FrameTask(
                    frame_name=robot_frame,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    # 使用 Levenberg-Marquardt 阻尼抑制难以到达或任务冲突时的过大求解步长。
                    lm_damping=1,
                )
                # Use the keypoint name as the key for indexing keypoint data in update_targets
                self.human_body_to_task[keypoint_name] = task
                self.body_name_to_source_keypoint[robot_frame] = keypoint_name

                self.tasks.append(task)
                self.robot_frame_names.append(robot_frame)
                self.task_errors[task] = []
        pass

    def setup_temporal_constraints(self):
        """Create optional temporal IK terms after source fps is available."""
        temporal_config = self.temporal_continuity_regularization
        if temporal_config is not None:
            if not isinstance(temporal_config, Mapping):
                raise ValueError("temporal_continuity_regularization must be a mapping")
            if _enabled_flag(temporal_config, "temporal_continuity_regularization"):
                if "cost" not in temporal_config:
                    raise ValueError("temporal_continuity_regularization.cost is required when enabled")
                cost = _finite_number(
                    temporal_config["cost"],
                    "temporal_continuity_regularization.cost",
                    strictly_positive=True,
                )
                indices = _joint_dof_indices_for_substrings(
                    self.model,
                    temporal_config.get("joint_name_substrings", DEFAULT_TEMPORAL_JOINT_SUBSTRINGS),
                    "temporal_continuity_regularization",
                )
                posture_cost = np.zeros(self.model.nv)
                posture_cost[indices] = cost
                self.temporal_posture_task = mink.PostureTask(self.model, cost=posture_cost)
                self.tasks.append(self.temporal_posture_task)

        velocity_config = self.inter_frame_velocity_limit_config
        if velocity_config is not None:
            if not isinstance(velocity_config, Mapping):
                raise ValueError("inter_frame_velocity_limit must be a mapping")
            if _enabled_flag(velocity_config, "inter_frame_velocity_limit"):
                velocity_by_substring = velocity_config.get("joint_velocity_limits")
                if not isinstance(velocity_by_substring, Mapping) or not velocity_by_substring:
                    raise ValueError(
                        "inter_frame_velocity_limit.joint_velocity_limits must be a non-empty mapping"
                    )
                indices = []
                max_velocities = []
                for substring, max_velocity in velocity_by_substring.items():
                    max_velocity = _finite_number(
                        max_velocity,
                        "inter_frame_velocity_limit.joint_velocity_limits values",
                        strictly_positive=True,
                    )
                    matched_indices = _joint_dof_indices_for_substrings(
                        self.model,
                        [substring],
                        "inter_frame_velocity_limit",
                    )
                    indices.extend(matched_indices.tolist())
                    max_velocities.extend([max_velocity] * len(matched_indices))
                # Repeated substring matches are ambiguous, so reject them instead of silently picking one.
                if len(set(indices)) != len(indices):
                    raise ValueError("inter_frame_velocity_limit joint substrings must not overlap")
                self.frame_velocity_limit = FrameVelocityLimit(
                    self.model, indices, max_velocities, self.fps
                )
                self.ik_limits.append(self.frame_velocity_limit)

    def begin_temporal_frame(self):
        """Anchor all temporal terms to the final qpos of the preceding data frame."""
        previous_q = self.configuration.q.copy()
        if self.temporal_posture_task is not None:
            # Mink cost c is squared in its least-squares objective; target is fixed per frame.
            self.temporal_posture_task.set_target(previous_q)
        if self.frame_velocity_limit is not None:
            self.frame_velocity_limit.begin_frame(previous_q)

    # 用途：为可识别的接触 body 预先创建位置锁定任务，并保存每段连续接触所需的固定目标位置。
    # 不直接返回数值；会向 self.contact_targets（接触约束描述的列表）加入可用接触项，并将对应的位置任务加入总 IK 任务集合。
    def setup_contact_targets(self):
        # 未配置机器人接触 body，或关键点文件不含接触状态名称时，无法建立一一对应的锁定任务。
        if not self.contact_body_names or not self.contact_state_name_to_idx:
            return

        # 两个列表按相同下标配对；数量不同会造成接触状态和机器人末端错配，因此整体跳过接触约束。
        if len(self.contact_body_names) != len(self.contact_names):
            if self.verbose:
                print(
                    "[contact target] skip: robot contact_links count "
                    f"{len(self.contact_body_names)} != keypoint contact_names count {len(self.contact_names)}"
                )
            return

        # 为每对“机器人接触 body 与输入接触状态名称”建立描述，供每帧根据实际接触状态启用或解除位置锁定。
        for body_name, contact_name in zip(self.contact_body_names, self.contact_names):
            contact_state_idx = self.contact_state_name_to_idx.get(contact_name)
            if contact_state_idx is None:
                continue

            # 找到能代表该机器人 body 的关键点；没有对应关键点时，无法计算该接触末端应锁定的位置。
            source_keypoint_name = self._resolve_body_source_keypoint(body_name)
            if source_keypoint_name is None:
                if self.verbose:
                    print(f"[contact target] could not find a source keypoint for {body_name}")
                continue

            # 调用 _ensure_contact_task（为指定机器人接触 body 获取唯一位置锁定任务的辅助函数）：
            # 若此前已为该 body 创建位置锁定任务则直接复用；否则创建只约束位置、不约束朝向的 FrameTask，
            # 随后计算每段连续接触期间的平均位置，将这个平均位置作为该接触任务的目标。
            task = self._ensure_contact_task(body_name)
            self.contact_targets.append(
                {
                    "body_name": body_name,
                    "contact_name": contact_name,
                    "contact_state_idx": contact_state_idx,
                    "source_keypoint_name": source_keypoint_name,
                    "mean_positions": self._build_contact_mean_positions(
                        contact_state_idx=contact_state_idx,
                        source_keypoint_name=source_keypoint_name,
                    ),
                    "task": task,
                }
            )

        # 仅在详细模式下打印最终成功建立的接触状态、机器人 body 与源关键点对应关系。
        if self.verbose and self.contact_targets:
            summary = [
                f"{item['contact_name']}->{item['body_name']} (src={item['source_keypoint_name']})"
                for item in self.contact_targets
            ]
            print(f"[contact target] enabled: {summary}")

    def _resolve_body_source_keypoint(self, body_name):
        if body_name in self.keypoint_name_to_idx:
            return body_name
        return self.body_name_to_source_keypoint.get(body_name)

    def _ensure_contact_task(self, body_name):
        existing_task = self.body_name_to_contact_task.get(body_name)
        if existing_task is not None:
            return existing_task

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=self.contact_position_cost,
            orientation_cost=0.0,
            lm_damping=1,
        )
        self.body_name_to_contact_task[body_name] = task
        self.tasks.append(task)
        self.robot_frame_names.append(body_name)
        self.task_errors[task] = []
        return task

    def _get_target_pose(self, frame_idx, keypoint_name):
        keypoint_idx = self.keypoint_name_to_idx[keypoint_name]
        target_pos = self.keypoints_pos[frame_idx, keypoint_idx]
        target_quat = self.keypoints_quat[frame_idx, keypoint_idx]
        return target_pos, target_quat

    def _get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found: {body_name}")
        return self.configuration.data.xpos[body_id], self.configuration.data.xquat[body_id]

    def _build_contact_mean_positions(self, contact_state_idx, source_keypoint_name):
        mean_positions = np.zeros((self.num_frames, 3), dtype=np.float64)
        contact_states = self.contact_seq[:, contact_state_idx].astype(bool)
        if not np.any(contact_states):
            return mean_positions

        source_keypoint_idx = self.keypoint_name_to_idx[source_keypoint_name]
        source_positions = np.asarray(
            self.keypoints_pos[:, source_keypoint_idx, :], dtype=np.float64
        )

        frame_idx = 0
        while frame_idx < self.num_frames:
            if not contact_states[frame_idx]:
                frame_idx += 1
                continue

            interval_end = frame_idx + 1
            while interval_end < self.num_frames and contact_states[interval_end]:
                interval_end += 1

            interval_mean = np.mean(source_positions[frame_idx:interval_end], axis=0)
            mean_positions[frame_idx:interval_end] = interval_mean
            frame_idx = interval_end

        return mean_positions

    def _get_contact_locked_position(self, frame_idx, contact_target):
        contact_state_idx = contact_target["contact_state_idx"]
        source_keypoint_name = contact_target["source_keypoint_name"]

        target_pos, _ = self._get_target_pose(frame_idx, source_keypoint_name)
        in_contact = bool(self.contact_seq[frame_idx, contact_state_idx])
        if in_contact:
            return contact_target["mean_positions"][frame_idx], True
        return target_pos, False
    
    def _apply_joints_limit_offset(self):
        """Add offsets to the lower/upper limits of matching joints per joints_limit_offset_degrees.

        The key is a joint-name substring (e.g. knee_joint / elbow_joint) and matches all joints
        in the model whose name contains that substring (e.g. left_knee_joint, right_knee_joint).
        The value is [lower_offset_deg, upper_offset_deg]: the lower offset is added to the lower
        bound and the upper offset to the upper bound (positive raises, negative lowers). A single
        scalar value is also supported (applied to the lower bound only).
        """
        if not self.joints_limit_offset_degrees:
            return

        # Collect all joint names in the model
        all_joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                all_joint_names.append((joint_id, name))

        for key, offset_deg in self.joints_limit_offset_degrees.items():
            # Support [lower_offset, upper_offset] or a single scalar (lower bound only)
            if isinstance(offset_deg, (list, tuple)):
                if len(offset_deg) != 2:
                    raise ValueError(
                        f"joints_limit_offset_degrees['{key}'] must be two values [lower, upper], "
                        f"got {offset_deg}"
                    )
                lower_offset_rad = np.radians(float(offset_deg[0]))
                upper_offset_rad = np.radians(float(offset_deg[1]))
            else:
                lower_offset_rad = np.radians(float(offset_deg))
                upper_offset_rad = 0.0

            if lower_offset_rad == 0.0 and upper_offset_rad == 0.0:
                continue
            matched = [(jid, name) for jid, name in all_joint_names if key in name]
            if not matched:
                if self.verbose:
                    print(f"[joint limit] no matching joint found: {key}")
                continue
            for joint_id, joint_name in matched:
                lower, upper = self.model.jnt_range[joint_id]
                new_lower = lower + lower_offset_rad
                new_upper = upper + upper_offset_rad
                # Ensure the lower bound does not exceed the upper bound
                if new_lower > new_upper:
                    new_lower, new_upper = new_upper, new_lower
                self.model.jnt_range[joint_id, 0] = new_lower
                self.model.jnt_range[joint_id, 1] = new_upper
                if self.verbose:
                    print(
                        f"[joint limit] {joint_name} range: "
                        f"[{lower:.4f}, {upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] rad"
                    )
 
    # 用途：使用当前动作帧普通身体关键点和接触锁定约束来更新 IK 求解的目标位姿，使随后的逆运动学求解对应这一帧输入运动。
    # 输入：self 表示已加载关键点序列并建立 IK 任务的重定向器实例；frame_idx 表示要读取的关键点序列帧编号。
    # 输出：不直接返回数值；会改写各 IK 任务的目标位姿，并刷新 self.current_targets（用于调试绘制的普通目标记录）和 self.current_contact_points（当前激活接触位置的记录）。
    def update_targets(self, frame_idx):
        # Record the keypoint targets (pos, quat_wxyz) mapped to tasks for the current frame, for visualization
        # 清空上一帧的绘制缓存，保证调试窗口只显示当前帧的普通目标和激活接触点。
        self.current_targets = []
        self.current_contact_points = []
        # 将每个输入的关键点转为位置加朝向的刚体位姿，并写入对应机器人 body 的普通 IK 任务。
        for keypoint_name, task in self.human_body_to_task.items():
            target_pos, target_quat = self._get_target_pose(frame_idx, keypoint_name)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos))
            robot_frame, pos_weight, rot_weight = self.ik_match_table[keypoint_name]
            self.current_targets.append(
                {
                    "pos": np.asarray(target_pos, dtype=np.float64),
                    "quat": np.asarray(target_quat, dtype=np.float64),
                    "robot_frame": robot_frame,
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                }
            )

        # 接触约束仅在对应末端处于支撑阶段时锁定其位置；未接触时将目标设为当前 body 位姿，从而不额外牵制普通 IK 解。
        for contact_target in self.contact_targets:
            source_keypoint_name = contact_target["source_keypoint_name"]
            # 根据当前帧是否接触，取得接触任务的位置候选和激活标记：
            # 接触时，target_pos 是连续接触段的平均位置，is_active 为 True，后续将锁定该末端；
            # 未接触时，target_pos 是当前帧源关键点的原始位置，is_active 为 False，
            # 后续会忽略该候选位置并将任务目标改为该 body 的当前位姿。
            target_pos, is_active = self._get_contact_locked_position(frame_idx, contact_target)
            if is_active:
                _, target_quat = self._get_target_pose(frame_idx, source_keypoint_name)
            else:
                target_pos, target_quat = self._get_body_pose(contact_target["body_name"])
            contact_target["task"].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
            )
            if is_active:
                # 记录实际生效的锁定位置，供调试窗口以接触点标记绘制。
                self.current_contact_points.append(np.asarray(target_pos, dtype=np.float64))
    
    def error(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.convergence_tasks]
            )
        )
    
    def _draw_pose(self, scene, pos, quat_wxyz, point_rgba, axis_alpha=1.0,
                   point_radius=0.035, axis_radius=0.005, axis_length=0.08):
        """Draw a pose in the scene: one small sphere + red/green/blue (XYZ) axes. Returns whether it succeeded."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 4 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, axis_alpha], dtype=np.float32),  # X red
            np.array([0.0, 1.0, 0.0, axis_alpha], dtype=np.float32),  # Y green
            np.array([0.0, 0.0, 1.0, axis_alpha], dtype=np.float32),  # Z blue
        )

        # Position sphere
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1

        # Derive the rotation matrix from the quaternion (wxyz); its columns are the axis directions
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, np.asarray(quat_wxyz, dtype=np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        # A cylinder defaults along its local z axis, so rotate each axis to its target direction
        base_to_axis = (
            Rot.from_euler("y", 90, degrees=True).as_matrix(),   # z->x
            Rot.from_euler("x", -90, degrees=True).as_matrix(),  # z->y
            np.eye(3),                                           # z->z
        )
        for axis_idx in range(3):
            axis_world_rot = rot_mat @ base_to_axis[axis_idx]
            axis_dir = rot_mat[:, axis_idx]
            center = pos + axis_dir * (0.5 * axis_length)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([axis_radius, 0.5 * axis_length, 0.0], dtype=np.float64),
                center,
                axis_world_rot.reshape(-1),
                axis_colors[axis_idx],
            )
            scene.ngeom += 1
        return True

    def _draw_point(self, scene, pos, point_rgba, point_radius=0.04):
        """Draw only a single spherical point in the scene."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 1 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return True

    def draw_target_keypoints(self, viewer, point_radius=0.035, axis_radius=0.005, axis_length=0.08, targets=None):
        """Draw the positions and axes of the target keypoints and the corresponding robot bodies.

        - Target keypoint: yellow sphere + full-brightness axes.
        - Robot body: cyan sphere + semi-transparent axes.
        """
        if targets is None:
            targets = getattr(self, "current_targets", None)
        scene = viewer.user_scn
        scene.ngeom = 0

        target_rgba = np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32)   # yellow
        robot_rgba = np.array([0.0, 1.0, 1.0, 0.9], dtype=np.float32)    # cyan
        contact_rgba = np.array([1.0, 0.0, 0.0, 0.95], dtype=np.float32) # red

        # 1) Target keypoints
        if targets:
            for target in targets:
                if isinstance(target, dict):
                    target_pos = target["pos"]
                    target_quat_wxyz = target["quat"]
                    robot_frame = target.get("robot_frame")
                    pos_weight = target.get("pos_weight", 1.0)
                    rot_weight = target.get("rot_weight", 1.0)
                else:
                    # Backward compatibility with the old (pos, quat) tuple format
                    target_pos, target_quat_wxyz = target
                    robot_frame = None
                    pos_weight = 1.0
                    rot_weight = 1.0

                # Zero position weight: draw the sphere at the corresponding body's position (overlapping)
                if pos_weight == 0 and robot_frame is not None:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
                    if body_id >= 0:
                        target_pos = self.configuration.data.xpos[body_id]

                if rot_weight == 0:
                    # Zero orientation weight: show only the sphere, no axes
                    if not self._draw_point(
                        scene, target_pos, target_rgba,
                        point_radius=point_radius,
                    ):
                        break
                else:
                    if not self._draw_pose(
                        scene, target_pos, target_quat_wxyz, target_rgba,
                        axis_alpha=1.0,
                        point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
                    ):
                        break

        # 2) Current position and orientation of the corresponding robot bodies
        data = self.configuration.data
        for robot_frame in self.robot_frame_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
            if body_id < 0:
                continue
            body_pos = data.xpos[body_id]
            body_quat = data.xquat[body_id]  # wxyz
            if not self._draw_pose(
                scene, body_pos, body_quat, robot_rgba,
                axis_alpha=0.5,
                point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
            ):
                break

        # 3) Currently active contact points, drawn as red spheres
        for contact_pos in self.current_contact_points:
            if not self._draw_point(
                scene,
                contact_pos,
                contact_rgba,
                point_radius=max(point_radius * 1.5, 0.02),
            ):
                break
    
    # 用途：按关键点序列的时间顺序求解目标机器人的逆运动学姿态，并可选地实时显示求解过程。
    def retarget(self):
        global _PAUSED
        viewer = None
        _PAUSED = False

        # 仅在调试模式下创建被动查看器；空格回调控制全局暂停状态，不参与 IK 求解。
        if self.render_debug:
            viewer = mujoco.viewer.launch_passive(
                self.model, self.configuration.data, key_callback=key_callback
            )
            print("Controls: Space play/pause")

        try:
            # 逐帧将输入关键点及接触锁定位置设为当前 IK 目标；同一配置对象会保留上一帧解，作为下一帧的初始姿态。
            for frame_idx in tqdm(range(self.num_frames), desc="Retargeting", unit="frame"):
                start_time = time.time()
                self.update_targets(frame_idx)
                self.begin_temporal_frame()

                # 记录更新前所有普通关键点与接触任务的未加权总误差，作为本轮迭代的比较基准。
                curr_error = self.error()
                # 读取 MuJoCo 模型的内部时间步长，作为本次差分 IK 速度积分的持续时间。
                dt = self.configuration.model.opt.timestep

                # 根据当前机器人姿态、全部任务与关节限位，求解使任务误差尽量下降的广义速度 vel。
                vel = mink.solve_ik(
                    self.configuration, self.tasks, dt, self.solver, self.damping, limits=self.ik_limits
                )
                # 将求得的速度 vel 在时间步长 dt 内积分，原地更新当前机器人的 qpos 姿态。
                self.configuration.integrate_inplace(vel, dt)
                # 重新计算更新后的总任务误差，以衡量这一次 IK 更新实际带来的改善。
                next_error = self.error()
                num_iter = 0
                # 只要本次迭代仍能使总误差显著下降，便在单帧内继续细化；最大次数限制避免难解姿态无限迭代。
                while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                    curr_error = next_error
                    dt = self.configuration.model.opt.timestep
                    vel = mink.solve_ik(
                        self.configuration, self.tasks, dt, self.solver, self.damping, limits=self.ik_limits
                    )
                    self.configuration.integrate_inplace(vel, dt)
                    next_error = self.error()
                    num_iter += 1
                self.frame_ik_iteration_counts.append(num_iter + 1)
                # 复制当前帧的完整广义位置，避免后续帧的原地积分改写已经保存的历史结果。
                curr_pos = self.configuration.data.qpos.copy()
                self.result_pos.append(curr_pos)

                # 调试查看器存在时，前向更新显示状态、绘制目标关键点，并按输入序列帧率控制播放速度。
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    mujoco.mj_forward(self.model, self.configuration.data)
                    self.draw_target_keypoints(viewer)
                    viewer.sync()
                    # Stay on the current frame while paused, until unpaused or the window is closed
                    while _PAUSED and viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    if not viewer.is_running():
                        break
                    end_time = time.time()
                    elapsed = end_time - start_time
                    time.sleep(max(0, self.time_step - elapsed))
                    # time.sleep(0.02)  # reduce CPU usage
                    
        finally:
            if viewer is not None:
                viewer.close()
    
    def save_results_as_csv(self, output_path):
        if len(self.result_pos) == 0:
            raise ValueError("No retarget results to save. Run retarget() first.")

        # [num_frame, num_joint], the first 7 dims are posXYZ + quat(wxyz)
        result = np.asarray(self.result_pos, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7:
            raise ValueError(
                f"result_pos shape must be [num_frame, num_joint>=7], got {result.shape}"
            )

        # Reorder the first-7-column quaternion from wxyz (cols 3..6) to xyzw
        result_xyzw = result.copy()
        result_xyzw[:, 3:7] = result[:, [4, 5, 6, 3]]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result_xyzw.tolist())

        if self.verbose:
            print(f"Saved retarget results to: {output_path} (shape={result_xyzw.shape})")



if  __name__ == "__main__":
    # 用途：作为脚本入口，将一段已生成的关键点运动逐帧求解为目标机器人关节姿态，并导出可回放的 CSV 轨迹。
    # 输入：命令行参数 args 用于指定机器人配置、可选的关键点文件名覆盖和调试渲染开关；YAML 配置提供模型、关键点、IK 映射及接触约束信息。
    # 输出：会在 output_data/robot_motion 中写入按帧保存的 MuJoCo qpos 轨迹 CSV。
    args = parse_args()
    # 统一以项目根目录解析相对路径，使脚本可从任意当前工作目录启动。
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR,"..", "robot", ".."))
    config_path = os.path.expanduser(args.config)
    if not os.path.isabs(config_path):
        config_path = os.path.join(workspace_root, config_path)

    # 读取机器人配置；其中的映射表定义源关键点与目标机器人连杆之间的 IK 约束及其权重。
    print("mujoco version: ", mujoco.__version__)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ik_match_table = config.get("ik_match_table", {})
    robot_xml_path = config.get("robot_xml_path", "")
    keypoints_path = config.get("keypoints_path", "")
    keypoints_idx = config.get("keypoints_idx","")

    # 命令行指定运动类型名时，按“机器人配置名/运动类型名_keypoints.pkl”的约定覆盖配置中的关键点文件。
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if args.keypoints_name:
        keypoints_path = os.path.join(
            "output_data",
            "keypoints",
            config_name,
            f"{args.keypoints_name}_keypoints.pkl",
        )

    # 将配置内的相对路径转换为绝对路径，供模型和关键点加载阶段直接使用。
    if robot_xml_path and not os.path.isabs(robot_xml_path):
        robot_xml_path = os.path.join(workspace_root, robot_xml_path)
    if keypoints_path and not os.path.isabs(keypoints_path):
        keypoints_path = os.path.join(workspace_root, keypoints_path)

    # 合并 YAML 与命令行的运行选项；命令行显式给出的调试开关优先级更高。
    verbose_mode = config.get("verbose", False)
    config_render_debug = bool(config.get("render_debug", False))
    render_debug = config_render_debug if args.render_debug is None else bool(args.render_debug)
    joints_limit_offset_degrees = config.get("joints_limit_offset_degrees", {})
    contact_body_names = config.get("contact_links")
    # 若未提供统一的接触连杆列表，则尝试从三个分组字段收集接触连杆，供旧格式或自定义配置使用。
    # 但当前所有配置都使用了 contact_links 字段，所以 contact_body_names 不会为 None, 因此下面的代码不会执行。
    if contact_body_names is None:
        contact_body_names = {
            field_name: config.get(field_name, [])
            for field_name in RobotRetarget.LEGACY_CONTACT_CONFIG_TO_NAMES
        }
    contact_position_cost = config.get("contact_pos_fixed_factor", 10.0)
    temporal_continuity_regularization = config.get("temporal_continuity_regularization")
    inter_frame_velocity_limit = config.get("inter_frame_velocity_limit")
    
    # 初始化重定向器：加载目标机器人、关键点序列，并建立逐帧 IK 与接触锁定约束。
    robot_retarget = RobotRetarget(
        model_path=robot_xml_path,
        keypoint_path=keypoints_path,
        ik_match_table=ik_match_table,
        solver="daqp",
        verbose=verbose_mode,
        render_debug=render_debug,
        joints_limit_offset_degrees=joints_limit_offset_degrees,
        contact_body_names=contact_body_names,
        contact_position_cost=contact_position_cost,
        temporal_continuity_regularization=temporal_continuity_regularization,
        inter_frame_velocity_limit=inter_frame_velocity_limit,
    )

    # 对每一帧关键点求解目标机器人姿态，求解结果保存在重定向器的内存结果序列中。
    robot_retarget.retarget()

    # 以输入关键点运动名和机器人配置名命名输出文件，避免不同目标机器人结果互相覆盖。
    keypoint_stem = os.path.splitext(os.path.basename(keypoints_path))[0]
    if keypoint_stem.endswith("_keypoints"):
        keypoint_stem = keypoint_stem[: -len("_keypoints")]
    output_csv = os.path.join(
        workspace_root,
        "output_data/robot_motion",
        f"{keypoint_stem}_{config_name}.csv",
    )
    # 导出用于后续 robot_replay.py 回放的 qpos CSV，并在导出时完成四元数列顺序转换。
    robot_retarget.save_results_as_csv(output_csv)
