#!/usr/bin/env python3
"""Convert a MotionDecode G1 CSV into ``robot_replay.py`` motion CSV format.

MotionDecode stores the floating-base quaternion as WXYZ and includes a header.
``robot_replay.py`` reads a headerless matrix whose quaternion is XYZW.  This
adapter only changes that four-value order; it neither changes samples nor
performs interpolation/resampling.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


EXPECTED_HEADER = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
    "dof_left_hip_pitch_joint(rad)",
    "dof_left_hip_roll_joint(rad)",
    "dof_left_hip_yaw_joint(rad)",
    "dof_left_knee_joint(rad)",
    "dof_left_ankle_pitch_joint(rad)",
    "dof_left_ankle_roll_joint(rad)",
    "dof_right_hip_pitch_joint(rad)",
    "dof_right_hip_roll_joint(rad)",
    "dof_right_hip_yaw_joint(rad)",
    "dof_right_knee_joint(rad)",
    "dof_right_ankle_pitch_joint(rad)",
    "dof_right_ankle_roll_joint(rad)",
    "dof_waist_yaw_joint(rad)",
    "dof_waist_roll_joint(rad)",
    "dof_waist_pitch_joint(rad)",
    "dof_left_shoulder_pitch_joint(rad)",
    "dof_left_shoulder_roll_joint(rad)",
    "dof_left_shoulder_yaw_joint(rad)",
    "dof_left_elbow_joint(rad)",
    "dof_left_wrist_roll_joint(rad)",
    "dof_left_wrist_pitch_joint(rad)",
    "dof_left_wrist_yaw_joint(rad)",
    "dof_right_shoulder_pitch_joint(rad)",
    "dof_right_shoulder_roll_joint(rad)",
    "dof_right_shoulder_yaw_joint(rad)",
    "dof_right_elbow_joint(rad)",
    "dof_right_wrist_roll_joint(rad)",
    "dof_right_wrist_pitch_joint(rad)",
    "dof_right_wrist_yaw_joint(rad)",
)
COLUMN_COUNT = len(EXPECTED_HEADER)
QUATERNION_NORM_TOLERANCE = 1e-3


class MotionDecodeValidationError(ValueError):
    """Raised when a CSV is not a valid MotionDecode G1 trajectory."""


def _validate_header(header: Sequence[str] | None) -> None:
    if header is None:
        raise MotionDecodeValidationError("input CSV is empty; expected a MotionDecode header")
    if len(header) != COLUMN_COUNT:
        raise MotionDecodeValidationError(
            f"header has {len(header)} columns; expected {COLUMN_COUNT}"
        )
    if tuple(header) != EXPECTED_HEADER:
        mismatches = [
            f"column {index + 1}: expected {expected!r}, got {actual!r}"
            for index, (expected, actual) in enumerate(zip(EXPECTED_HEADER, header))
            if expected != actual
        ]
        detail = "; ".join(mismatches[:3])
        if len(mismatches) > 3:
            detail += f"; and {len(mismatches) - 3} more"
        raise MotionDecodeValidationError(f"unexpected MotionDecode G1 header ({detail})")


def _validate_row(row: Sequence[str], csv_line: int) -> None:
    if len(row) != COLUMN_COUNT:
        raise MotionDecodeValidationError(
            f"line {csv_line} has {len(row)} columns; expected {COLUMN_COUNT}"
        )

    values: list[float] = []
    for column, text in enumerate(row):
        try:
            value = float(text)
        except ValueError as exc:
            raise MotionDecodeValidationError(
                f"line {csv_line}, column {column + 1} ({EXPECTED_HEADER[column]}): "
                f"expected a numeric value, got {text!r}"
            ) from exc
        if not math.isfinite(value):
            raise MotionDecodeValidationError(
                f"line {csv_line}, column {column + 1} ({EXPECTED_HEADER[column]}): "
                "value must be finite"
            )
        values.append(value)

    quaternion_norm = math.sqrt(sum(component * component for component in values[3:7]))
    if abs(quaternion_norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise MotionDecodeValidationError(
            f"line {csv_line}: root quaternion norm is {quaternion_norm:.9g}; "
            f"expected 1 within {QUATERNION_NORM_TOLERANCE:g}"
        )


def convert_motiondecode_csv(input_path: Path, output_path: Path, *, overwrite: bool = False) -> int:
    """Validate and convert one MotionDecode G1 CSV, returning its frame count."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist or is not a file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path} (pass --overwrite to replace it)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    frame_count = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            _validate_header(next(reader, None))
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.writer(temporary, lineterminator="\n")
                for csv_line, row in enumerate(reader, start=2):
                    _validate_row(row, csv_line)
                    # Preserve every numeric text value and all 29 joint columns.
                    writer.writerow([*row[:3], row[4], row[5], row[6], row[3], *row[7:]])
                    frame_count += 1

        if frame_count == 0:
            raise MotionDecodeValidationError("input CSV contains no motion frames")
        os.replace(temporary_path, output_path)
        temporary_path = None
        return frame_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MotionDecode G1 CSV (header, quaternion WXYZ) to the "
            "headerless XYZW CSV read by scripts/robot_replay.py."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="MotionDecode G1 CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Headerless robot_replay CSV output")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frame_count = convert_motiondecode_csv(args.input, args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    print(f"Converted {frame_count} frames: {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
