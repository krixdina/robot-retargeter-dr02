from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import mink
import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from robot_retarget import RobotRetarget  # noqa: E402


HINGE_XML = """
<mujoco model="one_hinge"><worldbody><body name="link">
  <joint name="elbow_joint" type="hinge" axis="0 0 1" range="-10 10"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
</body></worldbody></mujoco>
"""


def make_owner(config):
    owner = RobotRetarget.__new__(RobotRetarget)
    owner.model = mujoco.MjModel.from_xml_string(HINGE_XML)
    owner.keypoint_name_to_idx = {"elbow": 0}
    owner.two_stage_optimization_config = config
    owner.coarse_tasks = []
    owner.coarse_human_body_to_task = {}
    owner.coarse_ik_limits = []
    owner.coarse_max_iter = 0
    owner.coarse_convergence_threshold = 0.001
    owner.two_stage_enabled = False
    owner.setup_two_stage_optimization()
    return owner


class TwoStageOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / "config/robot/DR02_two_stage_optimization_test.yaml").open() as stream:
            cls.config = yaml.safe_load(stream)

    def make_retargeter(self):
        return self.make_retargeter_with_two_stage(self.config["two_stage_optimization"])

    def make_retargeter_with_two_stage(self, two_stage_optimization):
        return RobotRetarget(
            model_path=str(PROJECT_ROOT / self.config["robot_xml_path"]),
            keypoint_path=str(PROJECT_ROOT / self.config["keypoints_path"]),
            ik_match_table=self.config["ik_match_table"],
            verbose=False,
            render_debug=False,
            joints_limit_offset_degrees=self.config["joints_limit_offset_degrees"],
            contact_body_names=self.config["contact_links"],
            contact_position_cost=self.config["contact_pos_fixed_factor"],
            temporal_continuity_regularization=self.config["temporal_continuity_regularization"],
            inter_frame_velocity_limit=self.config["inter_frame_velocity_limit"],
            two_stage_optimization=two_stage_optimization,
        )

    def test_omitted_and_disabled_keep_no_coarse_objects(self):
        self.assertFalse(make_owner(None).two_stage_enabled)
        disabled = make_owner({"enabled": False})
        self.assertFalse(disabled.two_stage_enabled)
        self.assertEqual(disabled.coarse_tasks, [])
        self.assertEqual(disabled.coarse_ik_limits, [])

    def test_omitted_and_disabled_preserve_legacy_short_retarget_result(self):
        omitted = self.make_retargeter_with_two_stage(None)
        disabled = self.make_retargeter_with_two_stage({"enabled": False})
        for retargeter in (omitted, disabled):
            retargeter.num_frames = 2
            retargeter.keypoints_pos = retargeter.keypoints_pos[:2]
            retargeter.keypoints_quat = retargeter.keypoints_quat[:2]
            retargeter.contact_seq = retargeter.contact_seq[:2]
            retargeter.retarget()
        np.testing.assert_allclose(omitted.result_pos, disabled.result_pos)
        self.assertEqual(omitted.frame_ik_iteration_counts, disabled.frame_ik_iteration_counts)

    def test_coarse_config_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "coarse must be a mapping"):
            make_owner({"enabled": True})
        with self.assertRaisesRegex(ValueError, "unknown keypoint"):
            make_owner(
                {"enabled": True, "coarse": {"max_iterations": 10, "convergence_threshold": 0.001,
                    "ik_match_table": {"missing": ["link", 0, 1]}}}
            )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            make_owner(
                {"enabled": True, "coarse": {"max_iterations": 1.5, "convergence_threshold": 0.001,
                    "ik_match_table": {"elbow": ["link", 0, 1]}}}
            )

    def test_gmr_style_coarse_tasks_are_isolated_and_have_explicit_weights(self):
        retargeter = self.make_retargeter()
        self.assertTrue(retargeter.two_stage_enabled)
        self.assertEqual(len(retargeter.coarse_tasks), 14)
        self.assertEqual(len(retargeter.coarse_ik_limits), 1)
        self.assertIsInstance(retargeter.coarse_ik_limits[0], mink.ConfigurationLimit)
        self.assertFalse(any(isinstance(task, mink.PostureTask) for task in retargeter.coarse_tasks))
        self.assertFalse(any(limit is retargeter.frame_velocity_limit for limit in retargeter.coarse_ik_limits))
        self.assertFalse(set(retargeter.coarse_tasks) & {item["task"] for item in retargeter.contact_targets})
        costs = {name: task.cost for name, task in retargeter.coarse_human_body_to_task.items()}
        np.testing.assert_allclose(costs["left_fore_arm"][:3], [10, 10, 10])
        np.testing.assert_allclose(costs["left_fore_arm"][3:], [10, 10, 10])
        np.testing.assert_allclose(costs["left_calf"][:3], [50, 50, 50])

    def test_short_two_stage_smoke_warm_starts_fine_and_enforces_fine_budget(self):
        retargeter = self.make_retargeter()
        retargeter.num_frames = 2
        retargeter.keypoints_pos = retargeter.keypoints_pos[:2]
        retargeter.keypoints_quat = retargeter.keypoints_quat[:2]
        retargeter.contact_seq = retargeter.contact_seq[:2]
        calls = []
        original = RobotRetarget.solve_ik_stage

        def wrapped(instance, tasks, *args, **kwargs):
            start = instance.configuration.q.copy()
            result = original(instance, tasks, *args, **kwargs)
            calls.append((tasks, start, instance.configuration.q.copy()))
            return result

        with mock.patch.object(RobotRetarget, "solve_ik_stage", new=wrapped):
            retargeter.retarget()

        self.assertEqual(len(retargeter.result_pos), 2)
        self.assertEqual(len(calls), 4)
        for coarse, fine in zip(calls[::2], calls[1::2]):
            self.assertIs(coarse[0], retargeter.coarse_tasks)
            self.assertIs(fine[0], retargeter.tasks)
            np.testing.assert_allclose(fine[1], coarse[2])
            displacement = np.empty(retargeter.model.nv)
            mujoco.mj_differentiatePos(retargeter.model, displacement, 1.0, fine[1], fine[2])
            np.testing.assert_array_less(
                np.abs(displacement[retargeter.frame_velocity_limit.indices]),
                retargeter.frame_velocity_limit.frame_budget + 1e-8,
            )


if __name__ == "__main__":
    unittest.main()
