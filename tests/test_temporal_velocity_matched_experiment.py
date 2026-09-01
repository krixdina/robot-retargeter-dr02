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


if __name__ == "__main__":
    unittest.main()
