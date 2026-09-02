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
    owner.coarse_max_additional_iterations = 0
    owner.coarse_convergence_threshold = 0.001
    owner.two_stage_optimization_mode = "first_frame_only"
    owner.initialize_root_from_target = False
    owner.root_target_keypoint = "hips_mean"
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
                {"enabled": True, "coarse": {"max_additional_iterations": 10, "convergence_threshold": 0.001,
                    "ik_match_table": {"missing": ["link", 0, 1]}}}
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            make_owner(
                {"enabled": True, "coarse": {"max_additional_iterations": 1.5, "convergence_threshold": 0.001,
                    "ik_match_table": {"elbow": ["link", 0, 1]}}}
            )
        owner = make_owner(
            {"enabled": True, "optimization_mode": "first_frame_only", "coarse": {
                "max_additional_iterations": 0, "convergence_threshold": 0.001,
                "ik_match_table": {"elbow": ["link", 0, 1]},
            }}
        )
        self.assertEqual(owner.coarse_max_additional_iterations, 0)
        with self.assertRaisesRegex(ValueError, "optimization_mode"):
            make_owner({"enabled": True, "optimization_mode": "every_frame", "coarse": {
                "max_additional_iterations": 0, "convergence_threshold": 0.001,
                "ik_match_table": {"elbow": ["link", 0, 1]},
            }})
        with self.assertRaisesRegex(ValueError, "renamed to optimization_mode"):
            make_owner({"enabled": True, "application": "first_frame_only", "coarse": {
                "max_additional_iterations": 0, "convergence_threshold": 0.001,
                "ik_match_table": {"elbow": ["link", 0, 1]},
            }})
        with self.assertRaisesRegex(ValueError, "initialize_root_from_target"):
            make_owner({"enabled": True, "initialize_root_from_target": "yes", "coarse": {
                "max_additional_iterations": 0, "convergence_threshold": 0.001,
                "ik_match_table": {"elbow": ["link", 0, 1]},
            }})
        with self.assertRaisesRegex(ValueError, "root_target_keypoint"):
            make_owner({"enabled": True, "initialize_root_from_target": True, "root_target_keypoint": "missing", "coarse": {
                "max_additional_iterations": 0, "convergence_threshold": 0.001,
                "ik_match_table": {"elbow": ["link", 0, 1]},
            }})

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

    def test_root_initialization_uses_target_translation_and_yaw_only(self):
        retargeter = self.make_retargeter()
        target_pos, target_quat = retargeter._get_target_pose(0, "hips_mean")
        retargeter._initialize_root_from_target(0)
        np.testing.assert_allclose(retargeter.configuration.q[:3], target_pos)
        expected_yaw = np.arctan2(
            2.0 * (target_quat[0] * target_quat[3] + target_quat[1] * target_quat[2]),
            1.0 - 2.0 * (target_quat[2] ** 2 + target_quat[3] ** 2),
        )
        np.testing.assert_allclose(
            retargeter.configuration.q[3:7],
            [np.cos(expected_yaw / 2.0), 0.0, 0.0, np.sin(expected_yaw / 2.0)],
        )

    def test_first_frame_only_warm_starts_fine_and_frame_one_is_legacy(self):
        retargeter = self.make_retargeter()
        retargeter.num_frames = 2
        retargeter.keypoints_pos = retargeter.keypoints_pos[:2]
        retargeter.keypoints_quat = retargeter.keypoints_quat[:2]
        retargeter.contact_seq = retargeter.contact_seq[:2]
        initial_q = retargeter.configuration.q.copy()
        calls = []
        original = RobotRetarget.solve_ik_stage

        def wrapped(instance, tasks, *args, **kwargs):
            start = instance.configuration.q.copy()
            temporal_target = (
                instance.temporal_posture_task.target_q.copy()
                if tasks is instance.tasks and instance.temporal_posture_task is not None
                else None
            )
            result = original(instance, tasks, *args, **kwargs)
            calls.append((tasks, start, instance.configuration.q.copy(), temporal_target))
            return result

        with mock.patch.object(RobotRetarget, "solve_ik_stage", new=wrapped):
            retargeter.retarget()

        self.assertEqual(len(retargeter.result_pos), 2)
        self.assertEqual(len(calls), 3)
        coarse, fine, legacy = calls
        self.assertIs(coarse[0], retargeter.coarse_tasks)
        self.assertIsNot(fine[0], retargeter.tasks)
        self.assertFalse(any(isinstance(task, mink.PostureTask) for task in fine[0]))
        self.assertTrue(all(target["task"] in fine[0] for target in retargeter.contact_targets))
        self.assertFalse(
            any(limit is retargeter.frame_velocity_limit
                for limit in retargeter._fine_limits_for_initialization_frame())
        )
        # Fine receives the exact in-place coarse endpoint; there is no partial
        # velocity-box projection on the initialization frame.
        np.testing.assert_allclose(fine[1], coarse[2])
        self.assertIsNone(fine[3])
        # Frame 1 follows the legacy path and anchors the temporal/FVL terms to
        # the finalized frame 0 result before using the shared stage solver.
        self.assertIs(legacy[0], retargeter.tasks)
        np.testing.assert_allclose(legacy[1], retargeter.result_pos[0])
        np.testing.assert_allclose(legacy[3], retargeter.result_pos[0])
        np.testing.assert_allclose(
            retargeter.temporal_posture_task.target_q, retargeter.result_pos[0]
        )
        np.testing.assert_allclose(
            retargeter.frame_velocity_limit.previous_q, retargeter.result_pos[0]
        )
        displacement = np.empty(retargeter.model.nv)
        mujoco.mj_differentiatePos(
            retargeter.model, displacement, 1.0, retargeter.result_pos[0], retargeter.result_pos[1]
        )
        np.testing.assert_array_less(
            np.abs(displacement[retargeter.frame_velocity_limit.indices]),
            retargeter.frame_velocity_limit.frame_budget + 1e-8,
        )
        self.assertFalse(np.allclose(initial_q, retargeter.result_pos[0]))

    def test_error_uses_default_or_explicit_task_collection(self):
        retargeter = self.make_retargeter()
        retargeter.update_targets(0)
        retargeter.update_coarse_targets(0)

        self.assertEqual(
            retargeter.error(),
            retargeter.error(retargeter.convergence_tasks),
        )
        expected_coarse_error = np.linalg.norm(
            np.concatenate([
                task.compute_error(retargeter.configuration)
                for task in retargeter.coarse_tasks
            ])
        )
        self.assertEqual(
            retargeter.error(retargeter.coarse_tasks),
            expected_coarse_error,
        )

    def test_shared_stage_solver_matches_the_legacy_explicit_loop(self):
        shared = self.make_retargeter_with_two_stage(None)
        reference = self.make_retargeter_with_two_stage(None)
        for retargeter in (shared, reference):
            retargeter.update_targets(0)
            retargeter.begin_temporal_frame()

        shared_solves = shared.solve_ik_stage(
            shared.tasks,
            shared.convergence_tasks,
            shared.ik_limits,
            shared.max_iter,
            0.001,
        )

        curr_error = reference.error()
        dt = reference.configuration.model.opt.timestep
        velocity = mink.solve_ik(
            reference.configuration,
            reference.tasks,
            dt,
            reference.solver,
            reference.damping,
            limits=reference.ik_limits,
        )
        reference.configuration.integrate_inplace(velocity, dt)
        next_error = reference.error()
        additional_iterations = 0
        while curr_error - next_error > 0.001 and additional_iterations < reference.max_iter:
            curr_error = next_error
            velocity = mink.solve_ik(
                reference.configuration,
                reference.tasks,
                dt,
                reference.solver,
                reference.damping,
                limits=reference.ik_limits,
            )
            reference.configuration.integrate_inplace(velocity, dt)
            next_error = reference.error()
            additional_iterations += 1

        self.assertEqual(shared_solves, additional_iterations + 1)
        np.testing.assert_allclose(shared.configuration.q, reference.configuration.q)
        self.assertEqual(shared.error(), reference.error())

    def test_frame_zero_refreshes_fine_targets_after_root_initialization(self):
        retargeter = self.make_retargeter()
        retargeter.num_frames = 1
        retargeter.keypoints_pos = retargeter.keypoints_pos[:1]
        retargeter.keypoints_quat = retargeter.keypoints_quat[:1]
        # Force every contact task down the non-active branch, where its target
        # is derived from the current robot FK rather than source keypoints.
        retargeter.contact_seq = np.zeros_like(retargeter.contact_seq[:1], dtype=bool)
        events = []
        target_snapshots = []
        original_update_targets = retargeter.update_targets
        original_root_init = retargeter._initialize_root_from_target
        original_update_coarse_targets = retargeter.update_coarse_targets
        original_solve = retargeter.solve_ik_stage

        def record_targets(frame_idx):
            original_update_targets(frame_idx)
            events.append("targets")
            target_snapshots.append((
                retargeter.contact_targets[0]["task"].transform_target_to_world.translation().copy(),
                retargeter._get_body_pose(retargeter.contact_targets[0]["body_name"])[0].copy(),
            ))

        def record_root_init(frame_idx):
            original_root_init(frame_idx)
            events.append("root_init")

        def record_coarse_targets(frame_idx):
            original_update_coarse_targets(frame_idx)
            events.append("coarse_targets")

        def record_solve(tasks, *args, **kwargs):
            events.append("coarse_solve" if tasks is retargeter.coarse_tasks else "fine_solve")
            return original_solve(tasks, *args, **kwargs)

        with mock.patch.object(retargeter, "update_targets", side_effect=record_targets), \
             mock.patch.object(retargeter, "_initialize_root_from_target", side_effect=record_root_init), \
             mock.patch.object(retargeter, "update_coarse_targets", side_effect=record_coarse_targets), \
             mock.patch.object(retargeter, "solve_ik_stage", side_effect=record_solve):
            retargeter.retarget()

        self.assertEqual(
            events,
            ["targets", "root_init", "targets", "coarse_targets", "coarse_solve", "fine_solve"],
        )
        self.assertEqual(len(target_snapshots), 2)
        self.assertFalse(np.allclose(target_snapshots[0][0], target_snapshots[1][0]))
        np.testing.assert_allclose(target_snapshots[1][0], target_snapshots[1][1])


if __name__ == "__main__":
    unittest.main()
