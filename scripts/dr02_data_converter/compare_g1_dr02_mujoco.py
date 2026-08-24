#!/usr/bin/env python3
"""Compare a source Unitree G1 motion and a retargeted DR02 motion in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_robot_visualize import (  # noqa: E402
    build_combined_spec,
    get_qpos_start,
    get_robot_xml,
    load_motion,
)
from dr02_data_converter.motiondecode_adapter import EXPECTED_HEADER  # noqa: E402


SOURCE_ROBOT = "g1"
TARGET_ROBOT = "DR02"
QUATERNION_NORM_TOLERANCE = 1e-3


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def _validate_qpos(qpos: np.ndarray, expected_nq: int, label: str) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"{label} motion must be a non-empty 2D array, got {qpos.shape}")
    if qpos.shape[1] != expected_nq:
        raise ValueError(
            f"{label} qpos width mismatch: expected {expected_nq}, got {qpos.shape[1]}"
        )
    if not np.isfinite(qpos).all():
        bad = np.argwhere(~np.isfinite(qpos))[0]
        raise ValueError(
            f"{label} contains a non-finite value at row {bad[0] + 1}, column {bad[1] + 1}"
        )
    quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    invalid = np.abs(quaternion_norms - 1.0) > QUATERNION_NORM_TOLERANCE
    if np.any(invalid):
        frame = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"{label} root quaternion at frame {frame} has norm "
            f"{quaternion_norms[frame]:.8g}; expected 1 within "
            f"{QUATERNION_NORM_TOLERANCE:g}"
        )
    return qpos.astype(np.float64, copy=False)


def detect_source_format(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_row = next(csv.reader(handle), None)
    if first_row is None:
        raise ValueError(f"source CSV is empty: {csv_path}")
    if tuple(first_row) == EXPECTED_HEADER:
        return "motiondecode"
    try:
        for value in first_row:
            float(value)
    except ValueError as exc:
        raise ValueError(
            "source CSV header is neither the expected MotionDecode G1 header "
            "nor a headerless numeric project CSV"
        ) from exc
    return "project"


def load_motiondecode_source(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if tuple(header or ()) != EXPECTED_HEADER:
        raise ValueError(f"unexpected MotionDecode G1 header: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    # MotionDecode already stores the floating-base quaternion as MuJoCo WXYZ.
    return data


def load_source_csv(csv_path: Path, source_format: str, expected_nq: int) -> tuple[np.ndarray, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"source CSV not found: {csv_path}")
    resolved_format = detect_source_format(csv_path) if source_format == "auto" else source_format
    if resolved_format == "motiondecode":
        qpos = load_motiondecode_source(csv_path)
    else:
        qpos = load_motion(str(csv_path))
    return _validate_qpos(qpos, expected_nq, "source G1"), resolved_format


def load_target_csv(csv_path: Path, expected_nq: int) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"target CSV not found: {csv_path}")
    qpos = load_motion(str(csv_path))
    return _validate_qpos(qpos, expected_nq, "target DR02")


def recenter_for_comparison(
    qpos: np.ndarray,
    y_offset: float,
    preserve_world_xy: bool,
) -> np.ndarray:
    display_qpos = qpos.copy()
    if not preserve_world_xy:
        display_qpos[:, 0:2] -= display_qpos[0:1, 0:2]
    display_qpos[:, 1] += y_offset
    return display_qpos


class ComparisonPlayback:
    def __init__(
        self,
        source_qpos: np.ndarray,
        target_qpos: np.ndarray,
        source_fps: float,
        target_fps: float,
        render_fps: float,
        spacing: float,
        preserve_world_xy: bool,
        max_duration: float,
    ) -> None:
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.render_fps = render_fps
        self.source_qpos = recenter_for_comparison(
            source_qpos, spacing / 2.0, preserve_world_xy
        )
        self.target_qpos = recenter_for_comparison(
            target_qpos, -spacing / 2.0, preserve_world_xy
        )

        source_duration = (len(source_qpos) - 1) / source_fps
        target_duration = (len(target_qpos) - 1) / target_fps
        self.duration = min(source_duration, target_duration)
        if max_duration > 0.0:
            self.duration = min(self.duration, max_duration)
        self.num_render_frames = max(1, int(math.floor(self.duration * render_fps)) + 1)

        self.spec = build_combined_spec([SOURCE_ROBOT, TARGET_ROBOT])
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.source_start = get_qpos_start(
            self.model, f"{SOURCE_ROBOT}_floating_base_joint"
        )
        self.target_start = get_qpos_start(
            self.model, f"{TARGET_ROBOT}_floating_base_joint"
        )
        if self.source_start + source_qpos.shape[1] > self.model.nq:
            raise ValueError("source G1 qpos does not fit in the combined MuJoCo model")
        if self.target_start + target_qpos.shape[1] > self.model.nq:
            raise ValueError("target DR02 qpos does not fit in the combined MuJoCo model")

    def frame_indices(self, render_frame: int) -> tuple[int, int, float]:
        clamped = min(max(render_frame, 0), self.num_render_frames - 1)
        playback_time = min(clamped / self.render_fps, self.duration)
        source_frame = min(
            int(round(playback_time * self.source_fps)), len(self.source_qpos) - 1
        )
        target_frame = min(
            int(round(playback_time * self.target_fps)), len(self.target_qpos) - 1
        )
        return source_frame, target_frame, playback_time

    def apply(self, render_frame: int) -> tuple[int, int, float]:
        source_frame, target_frame, playback_time = self.frame_indices(render_frame)
        self.data.qpos[
            self.source_start : self.source_start + self.source_qpos.shape[1]
        ] = self.source_qpos[source_frame]
        self.data.qpos[
            self.target_start : self.target_start + self.target_qpos.shape[1]
        ] = self.target_qpos[target_frame]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return source_frame, target_frame, playback_time


def validate_playback(playback: ComparisonPlayback) -> None:
    for render_frame in tqdm(
        range(playback.num_render_frames),
        desc="Validating comparison",
        unit="frame",
        dynamic_ncols=True,
    ):
        playback.apply(render_frame)
        if not np.isfinite(playback.data.xpos).all():
            raise ValueError(f"non-finite MuJoCo body position at render frame {render_frame}")


def launch_viewer(
    playback: ComparisonPlayback,
    loop: bool,
    exit_on_finish: bool,
) -> None:
    state = {"frame": 0, "paused": False, "replay": False, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode in (ord("r"), ord("R")):
            state["replay"] = True
        elif keycode == ord(","):
            state["paused"] = True
            state["step"] = -1
        elif keycode == ord("."):
            state["paused"] = True
            state["step"] = 1

    print("Layout: source G1 at +Y, target DR02 at -Y")
    print("Controls: [Space] Pause/Resume  [,/.] Step  [R] Replay")
    frame_interval = 1.0 / playback.render_fps
    pbar = tqdm(
        total=playback.num_render_frames,
        desc="Comparison playback",
        unit="frame",
        dynamic_ncols=True,
    )

    with mujoco.viewer.launch_passive(
        playback.model, playback.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            frame_started = time.perf_counter()
            if state["replay"]:
                state.update(frame=0, paused=False, replay=False, step=0)
                pbar.reset(total=playback.num_render_frames)
            if state["step"]:
                state["frame"] = min(
                    max(state["frame"] + state["step"], 0),
                    playback.num_render_frames - 1,
                )
                state["step"] = 0

            source_frame, target_frame, playback_time = playback.apply(state["frame"])
            viewer.sync()
            pbar.n = min(state["frame"] + 1, playback.num_render_frames)
            pbar.set_postfix_str(
                f"t={playback_time:.2f}s g1={source_frame} dr02={target_frame}"
            )
            pbar.refresh()

            if not state["paused"]:
                state["frame"] += 1
                if state["frame"] >= playback.num_render_frames:
                    if loop:
                        state["frame"] = 0
                        pbar.reset(total=playback.num_render_frames)
                    elif exit_on_finish:
                        break
                    else:
                        state["frame"] = playback.num_render_frames - 1
                        state["paused"] = True

            elapsed = time.perf_counter() - frame_started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    pbar.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display source G1 and retargeted DR02 CSV motions side by side in MuJoCo."
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument(
        "--source-format",
        choices=("auto", "motiondecode", "project"),
        default="auto",
        help="Source CSV convention; auto detects MotionDecode header or project headerless CSV.",
    )
    parser.add_argument("--source-fps", type=positive_float, default=120.0)
    parser.add_argument("--target-fps", type=positive_float, default=120.0)
    parser.add_argument("--render-fps", type=positive_float, default=60.0)
    parser.add_argument("--spacing", type=positive_float, default=2.5)
    parser.add_argument(
        "--max-duration",
        type=nonnegative_float,
        default=0.0,
        help="Compare at most this many seconds; 0 uses the full shared duration.",
    )
    parser.add_argument(
        "--preserve-world-xy",
        action="store_true",
        help="Keep original global XY trajectories instead of recentering both at their first frame.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--exit-on-finish",
        action="store_true",
        help="Close the script after one playback instead of waiting at the last frame.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and forward-kinematics-check the comparison without opening a viewer.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_model = mujoco.MjModel.from_xml_path(get_robot_xml(SOURCE_ROBOT))
    target_model = mujoco.MjModel.from_xml_path(get_robot_xml(TARGET_ROBOT))
    source_qpos, resolved_source_format = load_source_csv(
        args.source_csv.expanduser().resolve(), args.source_format, source_model.nq
    )
    target_qpos = load_target_csv(args.target_csv.expanduser().resolve(), target_model.nq)

    playback = ComparisonPlayback(
        source_qpos=source_qpos,
        target_qpos=target_qpos,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        render_fps=args.render_fps,
        spacing=args.spacing,
        preserve_world_xy=args.preserve_world_xy,
        max_duration=args.max_duration,
    )
    print(
        f"source: format={resolved_source_format} frames={len(source_qpos)} "
        f"fps={args.source_fps:g} duration={(len(source_qpos) - 1) / args.source_fps:.3f}s"
    )
    print(
        f"target: frames={len(target_qpos)} fps={args.target_fps:g} "
        f"duration={(len(target_qpos) - 1) / args.target_fps:.3f}s"
    )
    print(
        f"comparison: duration={playback.duration:.3f}s "
        f"render_fps={args.render_fps:g} render_frames={playback.num_render_frames}"
    )

    if args.validate_only:
        validate_playback(playback)
        print("Validation successful: all synchronized MuJoCo frames are finite.")
    else:
        launch_viewer(playback, loop=args.loop, exit_on_finish=args.exit_on_finish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
