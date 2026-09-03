"""Lightweight contract tests for the disk-free DR02 final-export boundary."""
from __future__ import annotations

import sys
import os
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "dr02_data_converter"))

from dr02_npz_exporter import export_qpos_array
from motiondecode_adapter import EXPECTED_HEADER, load_motiondecode_qpos


def test_export_qpos_array_writes_reference_schema_and_wxyz(tmp_path: Path) -> None:
    qpos = np.zeros((2, 38), dtype=np.float64)
    qpos[:, 3:7] = [[0.1, 0.2, 0.3, np.sqrt(0.86)], [0.5, 0.5, 0.5, 0.5]]  # legacy xyzw
    output = tmp_path / "motion.npz"

    export_qpos_array(qpos, output, 120, overwrite=False)

    with np.load(output) as data:
        assert set(data.files) == {"root_pos", "root_rot", "dof_pos", "fps"}
        np.testing.assert_allclose(
            data["root_rot"], [[np.sqrt(0.86), 0.1, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5]]
        )
        assert data["root_pos"].dtype == np.float32
        assert data["dof_pos"].shape == (2, 31)
        assert data["fps"].dtype == np.int64 and int(data["fps"]) == 120


def test_motiondecode_loader_keeps_xyzw_only_at_legacy_boundary(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    row = ["0"] * len(EXPECTED_HEADER)
    row[3] = "1"  # source WXYZ identity
    source.write_text(",".join(EXPECTED_HEADER) + "\n" + ",".join(row) + "\n", encoding="utf-8")

    qpos = load_motiondecode_qpos(source)

    np.testing.assert_allclose(qpos[0, 3:7], [0.0, 0.0, 0.0, 1.0])


def test_batch_runner_parallel_publish_has_no_legacy_intermediates(tmp_path: Path) -> None:
    """Exercise shell scheduling, quoting, manifests, and final-only publication."""
    dataset_root = tmp_path / "Motion Decode samples"
    input_dir = dataset_root / "set with spaces"
    input_dir.mkdir(parents=True)
    for name in ("one file.csv", "two file.csv"):
        (input_dir / name).write_text("not read by the isolated runner stub", encoding="utf-8")

    converter = tmp_path / "direct stub.py"
    converter.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import time
            from pathlib import Path
            import numpy as np
            parser = argparse.ArgumentParser()
            parser.add_argument('--input')
            parser.add_argument('--output')
            parser.add_argument('--fps')
            parser.add_argument('--overwrite', action='store_true')
            args = parser.parse_args()
            time.sleep(0.35)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, root_pos=np.zeros((1, 3), np.float32),
                root_rot=np.array([[1, 0, 0, 0]], np.float32),
                dof_pos=np.zeros((1, 31), np.float32), fps=np.array(120, np.int64))
            """
        ),
        encoding="utf-8",
    )
    runner = PROJECT_ROOT / "scripts/dr02_data_converter/run_dr02_batch.sh"
    env = os.environ | {"DIRECT_CONVERTER": str(converter), "PYTHON_BIN": sys.executable}
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", str(runner), "--input-dir", str(input_dir), "--dataset-root", str(dataset_root), "--jobs", "2"],
        cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False,
    )
    elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert elapsed < 0.65, f"jobs did not run concurrently ({elapsed:.3f}s)"

    output_root = PROJECT_ROOT / "output_data/dr02_data_converter"
    expected = {
        output_root / "npz/set with spaces/one file_from_g1_DR02.npz",
        output_root / "npz/set with spaces/two file_from_g1_DR02.npz",
    }
    assert all(path.is_file() for path in expected)
    assert not (output_root / "normalized").exists()
    assert not (PROJECT_ROOT / "output_data/keypoints/DR02/set with spaces").exists()
    assert not (PROJECT_ROOT / "output_data/robot_motion/set with spaces").exists()
    for output in expected:
        with np.load(output) as data:
            assert set(data.files) == {"root_pos", "root_rot", "dof_pos", "fps"}
        output.unlink()
