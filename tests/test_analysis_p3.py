"""Tests for P@3, Retrieval Noise, human labelling, and Cohen's kappa."""
from __future__ import annotations

import unittest

from rq1.analysis.kappa import cohens_kappa, cohens_kappa_details
from rq1.analysis.labelling import (
    IRRELEVANT,
    RELEVANT,
    DisagreementError,
    RaterRating,
    RetrievalLabelSet,
    adjudicate,
    build_label_set,
)
from rq1.analysis.metrics import precision_at_k, retrieval_noise
from rq1.analysis.p3 import compute_retrieval_quality


class PrecisionAtKTests(unittest.TestCase):
    def test_precision_at_3(self) -> None:
        self.assertAlmostEqual(precision_at_k([True, False, True], 3), 2 / 3)

    def test_empty_is_none_not_zero(self) -> None:
        self.assertIsNone(precision_at_k([], 3))

    def test_fewer_than_k_uses_actual_denominator(self) -> None:
        self.assertAlmostEqual(precision_at_k([True, False], 3), 0.5)

    def test_k_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            precision_at_k([True], 0)


class RetrievalNoiseTests(unittest.TestCase):
    def test_noise_is_one_minus_precision(self) -> None:
        self.assertAlmostEqual(retrieval_noise(2 / 3), 1 / 3)

    def test_noise_of_none_is_none(self) -> None:
        self.assertIsNone(retrieval_noise(None))


class CohenKappaTests(unittest.TestCase):
    def test_known_value(self) -> None:
        a = ["RELEVANT", "RELEVANT", "IRRELEVANT", "IRRELEVANT", "RELEVANT"]
        b = ["RELEVANT", "IRRELEVANT", "RELEVANT", "IRRELEVANT", "RELEVANT"]
        self.assertAlmostEqual(cohens_kappa(a, b), 0.16666666666666666)

    def test_perfect_agreement_is_one(self) -> None:
        a = ["RELEVANT", "IRRELEVANT", "RELEVANT"]
        self.assertAlmostEqual(cohens_kappa(a, a), 1.0)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(cohens_kappa([], []))

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            cohens_kappa(["RELEVANT"], [])

    def test_details_reports_sample_size(self) -> None:
        a = ["RELEVANT", "IRRELEVANT"]
        details = cohens_kappa_details(a, a)
        self.assertEqual(details["sample_size"], 2)
        self.assertEqual(details["cohens_kappa"], 1.0)


class LabellingTests(unittest.TestCase):
    def test_invalid_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RaterRating("e1", "s1", "r1", "MAYBE")

    def test_adjudicate_agreement(self) -> None:
        result = adjudicate(
            [
                RaterRating("e1", "s1", "r1", RELEVANT),
                RaterRating("e1", "s1", "r2", RELEVANT),
            ]
        )
        self.assertEqual(result[("e1", "s1")], RELEVANT)

    def test_adjudicate_disagreement_raises(self) -> None:
        with self.assertRaises(DisagreementError):
            adjudicate(
                [
                    RaterRating("e1", "s1", "r1", RELEVANT),
                    RaterRating("e1", "s1", "r2", IRRELEVANT),
                ]
            )

    def test_build_label_set_preserves_rank_order(self) -> None:
        adjudicated = {("e1", "s1"): RELEVANT, ("e1", "s2"): IRRELEVANT, ("e1", "s3"): RELEVANT}
        label_set = build_label_set("e1", ["s1", "s2", "s3"], adjudicated)
        self.assertEqual([item[0] for item in label_set.labels], ["s1", "s2", "s3"])
        self.assertAlmostEqual(label_set.precision_at_k(3), 2 / 3)

    def test_build_label_set_requires_all_labels(self) -> None:
        adjudicated = {("e1", "s1"): RELEVANT}
        with self.assertRaises(DisagreementError):
            build_label_set("e1", ["s1", "s2"], adjudicated)


class RetrievalQualityTests(unittest.TestCase):
    def _labels(self) -> list[RetrievalLabelSet]:
        return [
            RetrievalLabelSet("e1", (("s1", RELEVANT), ("s2", IRRELEVANT), ("s3", RELEVANT))),
            RetrievalLabelSet("e2", (("s1", RELEVANT), ("s2", RELEVANT), ("s3", IRRELEVANT))),
        ]

    def test_mean_precision_noise_and_kappa(self) -> None:
        rater_a = ["RELEVANT", "RELEVANT", "IRRELEVANT", "IRRELEVANT", "RELEVANT"]
        rater_b = ["RELEVANT", "IRRELEVANT", "RELEVANT", "IRRELEVANT", "RELEVANT"]
        quality = compute_retrieval_quality(self._labels(), rater_a, rater_b)
        self.assertAlmostEqual(quality["precision_at_k_mean"], 2 / 3)
        self.assertAlmostEqual(quality["retrieval_noise_mean"], 1 / 3)
        self.assertAlmostEqual(quality["cohens_kappa"], 0.16666666666666666)
        self.assertEqual(quality["kappa_sample_size"], 5)
        self.assertEqual(quality["no_retrieval_count"], 0)
        self.assertIsNotNone(quality["noise_bootstrap"]["lower"])
        self.assertIsNotNone(quality["noise_bootstrap"]["upper"])

    def test_no_retrieval_reported_separately(self) -> None:
        labels = self._labels() + [RetrievalLabelSet("e3", ())]
        quality = compute_retrieval_quality(labels, ["RELEVANT"], ["RELEVANT"])
        self.assertEqual(quality["no_retrieval_count"], 1)
        # Empty retrieval does not change the mean precision of retrieved events.
        self.assertAlmostEqual(quality["precision_at_k_mean"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
