from __future__ import annotations

import importlib.util
import json
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

    def test_read_qpos_csv_rejects_nan_and_inf_before_mujoco(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in (np.nan, np.inf):
                path = Path(directory) / f"motion_{value}.csv"
                np.savetxt(path, [[1, 2, 3, 0.1, 0.2, 0.3, 0.4, value]], delimiter=",")
                with self.assertRaisesRegex(ValueError, "non-finite qpos"):
                    quality._read_qpos_csv(path, nq=8)

    def test_report_json_is_strictly_json_decodable(self):
        payload = {"finite": True, "metric": 0.25, "velocity": {"max_excess": 0.0}}
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded), payload)
        with self.assertRaises(ValueError):
            json.dumps({"invalid": np.nan}, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
