#!/usr/bin/env python3
"""Export headerless DR02 qpos CSV data to the RL motion NPZ format.

The CSV must contain exactly 38 numeric columns in every row, in MJCF qpos
order: root position (xyz), root quaternion (xyzw), then the 31 DR02 joints.
No resampling, quaternion conversion, or joint reordering is performed.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

import numpy as np


NUM_COLUMNS = 38
ROOT_POS_COLUMNS = slice(0, 3)
ROOT_ROT_COLUMNS = slice(3, 7)
DOF_POS_COLUMNS = slice(7, 38)
QUATERNION_NORM_TOLERANCE = 1e-3
INT64_MAX = np.iinfo(np.int64).max


def parse_fps(value: str) -> int:
    """Parse a positive, integer-valued FPS suitable for the int64 NPZ field."""
    try:
        fps = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(
            "fps must be a finite positive integer-valued number"
        ) from exc

    if not fps.is_finite() or fps <= 0 or fps != fps.to_integral_value():
        raise argparse.ArgumentTypeError(
            "fps must be a finite positive integer-valued number"
        )

    fps_int = int(fps)
    if fps_int > INT64_MAX:
        raise argparse.ArgumentTypeError(f"fps must not exceed {INT64_MAX}")
    return fps_int


def load_qpos_csv(input_path: Path) -> np.ndarray:
    """Load and validate a headerless, comma-delimited DR02 qpos CSV."""
    try:
        data = np.loadtxt(input_path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read headerless numeric CSV {input_path}: {exc}") from exc

    if data.ndim != 2:
        raise ValueError(f"CSV must be two-dimensional; got shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("CSV must contain at least one frame")
    if data.shape[1] != NUM_COLUMNS:
        raise ValueError(
            f"CSV must contain exactly {NUM_COLUMNS} columns "
            f"(xyz + xyzw + 31 joints); got {data.shape[1]}"
        )
    if not np.isfinite(data).all():
        bad_rows, bad_columns = np.where(~np.isfinite(data))
        raise ValueError(
            "CSV contains non-finite values; first occurrence at "
            f"row {bad_rows[0] + 1}, column {bad_columns[0] + 1}"
        )

    quaternion_norms = np.linalg.norm(data[:, ROOT_ROT_COLUMNS], axis=1)
    invalid_quaternions = np.abs(quaternion_norms - 1.0) > QUATERNION_NORM_TOLERANCE
    if np.any(invalid_quaternions):
        frame = int(np.flatnonzero(invalid_quaternions)[0])
        raise ValueError(
            f"root quaternion at row {frame + 1} has norm {quaternion_norms[frame]:.8g}; "
            f"expected 1 within {QUATERNION_NORM_TOLERANCE:g}"
        )
    return data


def export_npz(input_path: Path, output_path: Path, fps: int, overwrite: bool) -> None:
    """Validate one CSV and save it with the expected RL NPZ schema."""
    if not input_path.is_file():
        raise ValueError(f"input CSV does not exist or is not a file: {input_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"output already exists: {output_path} (pass --overwrite to replace it)")

    data = load_qpos_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        root_pos=data[:, ROOT_POS_COLUMNS].astype(np.float32),
        root_rot=data[:, ROOT_ROT_COLUMNS].astype(np.float32),
        dof_pos=data[:, DOF_POS_COLUMNS].astype(np.float32),
        fps=np.array(fps, dtype=np.int64),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert headerless 38-column DR02 qpos CSV to RL motion NPZ."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input headerless qpos CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output NPZ path")
    parser.add_argument(
        "--fps",
        required=True,
        type=parse_fps,
        help="Positive integer-valued frame rate (stored as int64)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing an existing output file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        export_npz(args.input, args.output, args.fps, args.overwrite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
