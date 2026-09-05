"""Small, generated DR02 target-keypoint fixtures for retargeting tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np


def write_minimal_dr02_keypoints(config: dict, destination: Path, frames: int = 3) -> Path:
    """Write a static FK-consistent target PKL without depending on output_data."""
    project_root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(str(project_root / config["robot_xml_path"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = list(config["ik_match_table"])
    positions = np.empty((frames, len(names), 3), dtype=np.float64)
    quaternions = np.empty((frames, len(names), 4), dtype=np.float64)
    for index, name in enumerate(names):
        body_name = config["ik_match_table"][name][0]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        positions[:, index, :] = data.xpos[body_id]
        quaternions[:, index, :] = data.xquat[body_id]
    payload = {
        "keypoint_names": names,
        "positions": positions,
        "quaternions": quaternions,
        "contact_names": ["left_foot_end", "left_toe", "right_foot_end", "right_toe", "left_hand", "right_hand"],
        "contact_states": np.zeros((frames, 6), dtype=bool),
        "fps": 30,
    }
    destination.write_bytes(pickle.dumps(payload))
    return destination
