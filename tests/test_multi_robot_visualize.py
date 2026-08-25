from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_robot_visualize import build_combined_spec, get_robot_xml  # noqa: E402


class BuildCombinedSpecTest(unittest.TestCase):
    def test_attach_keeps_parent_integrator_without_conflict_warning(self) -> None:
        robot_names = ["g1", "DR02"]
        expected_nq = sum(
            mujoco.MjModel.from_xml_path(get_robot_xml(robot)).nq
            for robot in robot_names
        )
        parent_integrator = int(mujoco.MjSpec().option.integrator)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build_combined_spec(robot_names).compile()

        attach_conflicts = [
            str(item.message) for item in caught if "Attach conflict" in str(item.message)
        ]
        self.assertEqual(attach_conflicts, [])
        self.assertEqual(int(model.opt.integrator), parent_integrator)
        self.assertEqual(model.nq, expected_nq)
        for robot in robot_names:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot}_floating_base_joint",
            )
            self.assertGreaterEqual(joint_id, 0)


if __name__ == "__main__":
    unittest.main()
