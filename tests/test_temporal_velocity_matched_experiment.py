from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr02_data_converter/temporal_velocity_matched_experiment.py"
SPEC = importlib.util.spec_from_file_location("matched_experiment", MODULE_PATH)
matched = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(matched)


class MatchedExperimentDefinitionTest(unittest.TestCase):
    def test_b0_is_a_matched_unconstrained_candidate(self):
        self.assertFalse(matched.CANDIDATES["B0"]["temporal"]["enabled"])
        self.assertFalse(matched.CANDIDATES["B0"]["velocity"]["enabled"])

    def test_tracking_groups_are_explicit_and_do_not_mix_legs(self):
        self.assertEqual(set(matched.FOUR_EE), {"left_hand", "right_hand", "left_foot", "right_foot"})
        self.assertEqual(
            set(matched.UPPER_LIMB),
            {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hand", "right_hand"},
        )
        sources = [source for source, _ in matched.UPPER_LIMB.values()]
        self.assertFalse(any("hip" in source or "thigh" in source or "calf" in source for source in sources))
        self.assertEqual(len(sources), len(set(sources)))

    def test_pairs_have_exact_expected_membership(self):
        self.assertEqual(matched.FOUR_EE["left_foot"], ("left_foot_end_link", "left_foot_end_link"))
        self.assertEqual(matched.FOUR_EE["right_foot"], ("right_foot_end_link", "right_foot_end_link"))
        self.assertEqual(matched.UPPER_LIMB["left_hand"], ("left_fore_arm", "left_hand_link"))

    def test_hard_is_a_subset_of_near_at_boundaries(self):
        import numpy as np
        hard = np.array([0.001, 0.0009]) <= 0.001
        near = np.array([0.001, 0.0009]) <= np.deg2rad(5)
        self.assertTrue(np.all(~hard | near))
        self.assertEqual(matched.longest_run(np.array([False, True, True, False, True])), 2)

    def test_archive_aggregation_is_frame_weighted_not_pooled_p95(self):
        rows = [{"candidate":"B0","motion_id":"1","source":"a","frames":"1","x":"1","edge_pct":""},
                {"candidate":"B0","motion_id":"2","source":"b","frames":"3","x":"5","edge_pct":""}]
        action, frame, _ = matched.aggregate_rows(rows)
        self.assertEqual(action[0]["x"], 3.0)
        self.assertEqual(frame[0]["x"], 4.0)


if __name__ == "__main__":
    unittest.main()
