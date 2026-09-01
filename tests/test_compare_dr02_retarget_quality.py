from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/dr02_data_converter/compare_dr02_retarget_quality.py"
SPEC = importlib.util.spec_from_file_location("compare_dr02_retarget_quality", MODULE_PATH)
quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quality)


class CompareDr02RetargetQualityTest(unittest.TestCase):
    def test_read_qpos_csv_restores_exported_xyzw_to_mujoco_wxyz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            # qpos export stores xyz + xyzw; all remaining entries are hinges.
            np.savetxt(path, [[1, 2, 3, 0.1, 0.2, 0.3, 0.4, 9]], delimiter=",")
            qpos = quality._read_qpos_csv(path, nq=8)
        np.testing.assert_allclose(qpos[0], [1, 2, 3, 0.4, 0.1, 0.2, 0.3, 9])

    def test_read_qpos_csv_rejects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            np.savetxt(path, [[1, 2]], delimiter=",")
            with self.assertRaisesRegex(ValueError, "expected MuJoCo qpos"):
                quality._read_qpos_csv(path, nq=8)


if __name__ == "__main__":
    unittest.main()
