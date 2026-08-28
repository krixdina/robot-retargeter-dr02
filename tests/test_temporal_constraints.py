from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mink
import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from robot_retarget import FrameVelocityLimit, RobotRetarget  # noqa: E402


HINGE_XML = """
<mujoco model="one_hinge">
  <worldbody>
    <body name="link">
      <joint name="elbow_joint" type="hinge" axis="0 0 1" range="-10 10"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_temporal_owner(temporal_config=None, velocity_config=None, fps=10):
    """Build only the state needed by RobotRetarget.setup_temporal_constraints."""
    owner = RobotRetarget.__new__(RobotRetarget)
    owner.model = mujoco.MjModel.from_xml_string(HINGE_XML)
    owner.fps = fps
    owner.tasks = []
    owner.ik_limits = [mink.ConfigurationLimit(owner.model)]
    owner.temporal_continuity_regularization = temporal_config
    owner.inter_frame_velocity_limit_config = velocity_config
    owner.temporal_posture_task = None
    owner.frame_velocity_limit = None
    owner.setup_temporal_constraints()
    return owner


class TemporalConstraintTest(unittest.TestCase):
    def test_omitted_and_explicitly_disabled_configs_are_identical(self):
        omitted = make_temporal_owner()
        disabled = make_temporal_owner({"enabled": False}, {"enabled": False})
        self.assertEqual(len(omitted.tasks), len(disabled.tasks))
        self.assertEqual(len(omitted.ik_limits), len(disabled.ik_limits))
        self.assertIsNone(omitted.temporal_posture_task)
        self.assertIsNone(disabled.frame_velocity_limit)
        task = mink.PostureTask(omitted.model, cost=1.0)
        task.set_target(np.array([1.0]))
        omitted_configuration = mink.Configuration(omitted.model)
        disabled_configuration = mink.Configuration(disabled.model)
        omitted_velocity = mink.solve_ik(
            omitted_configuration, [task], 0.01, "daqp", limits=omitted.ik_limits
        )
        disabled_velocity = mink.solve_ik(
            disabled_configuration, [task], 0.01, "daqp", limits=disabled.ik_limits
        )
        np.testing.assert_allclose(omitted_velocity, disabled_velocity)

    def test_all_feature_switch_combinations(self):
        temporal = {"enabled": True, "cost": 0.2, "joint_name_substrings": ["elbow"]}
        velocity = {"enabled": True, "joint_velocity_limits": {"elbow": 1.0}}
        for temporal_enabled in (False, True):
            for velocity_enabled in (False, True):
                owner = make_temporal_owner(
                    temporal if temporal_enabled else {"enabled": False},
                    velocity if velocity_enabled else {"enabled": False},
                )
                self.assertEqual(len(owner.tasks), int(temporal_enabled))
                self.assertEqual(len(owner.ik_limits), 1 + int(velocity_enabled))
                self.assertEqual(owner.temporal_posture_task is not None, temporal_enabled)
                self.assertEqual(owner.frame_velocity_limit is not None, velocity_enabled)

    def test_posture_target_is_fixed_at_frame_start(self):
        owner = make_temporal_owner(
            {"enabled": True, "cost": 0.2, "joint_name_substrings": ["elbow"]}
        )
        owner.configuration = mink.Configuration(owner.model)
        owner.begin_temporal_frame()
        anchor = owner.configuration.q.copy()
        np.testing.assert_allclose(owner.temporal_posture_task.target_q, anchor)
        owner.configuration.integrate_inplace(np.array([0.4]), 1.0)
        # Calling the QP task repeatedly during a frame must not refresh its target.
        owner.temporal_posture_task.compute_error(owner.configuration)
        owner.temporal_posture_task.compute_error(owner.configuration)
        np.testing.assert_allclose(owner.temporal_posture_task.target_q, anchor)
        owner.begin_temporal_frame()
        np.testing.assert_allclose(owner.temporal_posture_task.target_q, owner.configuration.q)

    def test_frame_velocity_budget_covers_multiple_ik_iterations(self):
        model = mujoco.MjModel.from_xml_string(HINGE_XML)
        configuration = mink.Configuration(model)
        task = mink.PostureTask(model, cost=1.0)
        task.set_target(np.array([2.0]))
        limit = FrameVelocityLimit(model, [0], [1.0], fps=10.0)
        limit.begin_frame(configuration.q)

        # The unbounded first IK update is intentionally much larger than 0.1 rad.
        unbounded_velocity = mink.solve_ik(configuration, [task], 0.01, "daqp")
        self.assertGreater(abs(unbounded_velocity[0] * 0.01), 0.1)

        for _ in range(3):  # More than one internal iteration in the same data frame.
            velocity = mink.solve_ik(configuration, [task], 0.01, "daqp", limits=[limit])
            configuration.integrate_inplace(velocity, 0.01)

        displacement = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, displacement, 1.0, limit.previous_q, configuration.q)
        self.assertGreater(displacement[0], 0.0)
        self.assertLessEqual(abs(displacement[0]), 0.1 + 1e-8)

    def test_invalid_temporal_configuration_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "cost must be > 0"):
            make_temporal_owner({"enabled": True, "cost": -0.1, "joint_name_substrings": ["elbow"]})
        with self.assertRaisesRegex(ValueError, "cost is required"):
            make_temporal_owner({"enabled": True, "joint_name_substrings": ["elbow"]})
        with self.assertRaisesRegex(ValueError, "unknown joint substring"):
            make_temporal_owner({"enabled": True, "cost": 0.1, "joint_name_substrings": ["missing"]})
        with self.assertRaisesRegex(ValueError, "values must be > 0"):
            make_temporal_owner(
                None,
                {"enabled": True, "joint_velocity_limits": {"elbow": -1.0}},
            )

    def test_strict_switch_and_numeric_validation(self):
        with self.assertRaisesRegex(ValueError, "enabled must be a bool"):
            make_temporal_owner({"enabled": 1})
        with self.assertRaisesRegex(ValueError, "enabled must be a bool"):
            make_temporal_owner(None, {"enabled": "false"})
        with self.assertRaisesRegex(ValueError, "finite number"):
            make_temporal_owner(
                {"enabled": True, "cost": float("nan"), "joint_name_substrings": ["elbow"]}
            )
        with self.assertRaisesRegex(ValueError, "finite number"):
            make_temporal_owner(
                None,
                {"enabled": True, "joint_velocity_limits": {"elbow": float("inf")}},
            )
        with self.assertRaisesRegex(ValueError, "fps"):
            make_temporal_owner(
                None,
                {"enabled": True, "joint_velocity_limits": {"elbow": 1.0}},
                fps=float("nan"),
            )

    def test_dr02_test_yaml_selects_eight_non_free_arm_dofs(self):
        config_path = PROJECT_ROOT / "config/robot/DR02_temporal_constraints_test.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / config["robot_xml_path"]))
        owner = RobotRetarget.__new__(RobotRetarget)
        owner.model = model
        owner.fps = 30
        owner.tasks = []
        owner.ik_limits = [mink.ConfigurationLimit(model)]
        owner.temporal_continuity_regularization = config["temporal_continuity_regularization"]
        owner.inter_frame_velocity_limit_config = config["inter_frame_velocity_limit"]
        owner.temporal_posture_task = None
        owner.frame_velocity_limit = None
        owner.setup_temporal_constraints()
        self.assertEqual(np.count_nonzero(owner.temporal_posture_task.cost), 8)
        self.assertEqual(len(owner.frame_velocity_limit.indices), 8)
        self.assertTrue(np.all(owner.frame_velocity_limit.indices >= 6))

    def test_short_dr02_retarget_keeps_frame_budget_across_inner_iterations(self):
        config_path = PROJECT_ROOT / "config/robot/DR02_temporal_constraints_test.yaml"
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        retargeter = RobotRetarget(
            model_path=str(PROJECT_ROOT / config["robot_xml_path"]),
            keypoint_path=str(PROJECT_ROOT / config["keypoints_path"]),
            ik_match_table=config["ik_match_table"],
            verbose=False,
            render_debug=False,
            joints_limit_offset_degrees=config["joints_limit_offset_degrees"],
            contact_body_names=config["contact_links"],
            contact_position_cost=config["contact_pos_fixed_factor"],
            temporal_continuity_regularization=config["temporal_continuity_regularization"],
            inter_frame_velocity_limit=config["inter_frame_velocity_limit"],
        )
        initial_q = retargeter.configuration.q.copy()
        retargeter.num_frames = 2
        retargeter.keypoints_pos = retargeter.keypoints_pos[:2]
        retargeter.keypoints_quat = retargeter.keypoints_quat[:2]
        retargeter.contact_seq = retargeter.contact_seq[:2]
        retargeter.retarget()  # In-memory only: no CSV is written by this test.
        self.assertEqual(len(retargeter.result_pos), 2)
        self.assertGreater(max(retargeter.frame_ik_iteration_counts), 1)
        for previous_q, current_q in zip([initial_q] + retargeter.result_pos[:-1], retargeter.result_pos):
            displacement = np.empty(retargeter.model.nv)
            mujoco.mj_differentiatePos(retargeter.model, displacement, 1.0, previous_q, current_q)
            selected = displacement[retargeter.frame_velocity_limit.indices]
            np.testing.assert_array_less(
                np.abs(selected), retargeter.frame_velocity_limit.frame_budget + 1e-8
            )


if __name__ == "__main__":
    unittest.main()
