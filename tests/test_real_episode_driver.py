from __future__ import annotations

import unittest

from rq1.hermes.episode_driver import parse_model_action
from rq1.recovery.controlled_failure import ControlledFailureError, select_reversible_navigation_action


class RealEpisodeDriverContractTests(unittest.TestCase):
    def test_accepts_only_one_exact_action_line(self) -> None:
        actions = ("go to desk 1", "look")
        self.assertEqual("go to desk 1", parse_model_action("ACTION: go to desk 1", actions))
        self.assertIsNone(parse_model_action("go to desk 1", actions))
        self.assertIsNone(parse_model_action("ACTION: Go to desk 1", actions))
        self.assertIsNone(parse_model_action("ACTION: go to desk 1\nextra", actions))
        self.assertIsNone(parse_model_action("ACTION: invent action", actions))

    def test_navigation_detour_is_sorted_and_never_the_expected_action(self) -> None:
        action = select_reversible_navigation_action(
            ("go to pantry 1", "go to desk 1", "look", "go to bed 1"), "go to desk 1"
        )
        self.assertEqual("go to bed 1", action)

    def test_non_navigation_checkpoint_fails_closed(self) -> None:
        with self.assertRaises(ControlledFailureError) as caught:
            select_reversible_navigation_action(("go to desk 1",), "take mug 1")
        self.assertEqual("checkpoint_not_navigation_eligible", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
