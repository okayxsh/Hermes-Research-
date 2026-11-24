"""Human relevance-rating workflow -- SYNTHETIC submissions exercise the tooling; they are not human labels."""
from __future__ import annotations

import csv
import io
import unittest

from five_arm_fixtures import ORACLE_CONTINUATION, SKILL_TEXTS
from rq1.five_arm import rating

SALT = "fixture-salt-0123456789"
TASKS = [{"task_id": f"valid_seen:task-{i}/trial_T1", "task_goal": f"put a clean mug in coffeemachine {i}.",
          "recovery_context": "You arrive at cabinet 1. The cabinet 1 is closed."} for i in range(3)]
def DECL(rater, **override):
    """A declaration-SHAPED record for tooling tests (not a real declaration; no real rater exists here)."""
    base = {"declaration_version": rating.DECLARATION_VERSION, "label_source": "human", "rater_id": rater, "role": "rater",
            "declared_at": "2026-11-24T10:00:00Z", **rating.DECLARATION_STATEMENTS,
            "coordinator_receipt": {"received_by_code": "COORD", "received_at": "2026-11-24T11:00:00Z", "channel": "separate upload",
                                    "export_sha256": "a" * 64}}
    base.update(override)
    return base


def fill(csv_text: str, labeler) -> bytes:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=rating.EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        row["rating"] = labeler(row["item_id"])
        writer.writerow(row)
    return out.getvalue().encode()


class ItemExportTests(unittest.TestCase):
    def setUp(self):
        self.items, self.key = rating.build_items(TASKS, SKILL_TEXTS, design="full_library", salt=SALT)

    def test_stable_anonymous_ids_and_hidden_rank(self):
        again, _ = rating.build_items(TASKS, SKILL_TEXTS, design="full_library", salt=SALT)
        self.assertEqual([i["item_id"] for i in again], [i["item_id"] for i in self.items])
        other, _ = rating.build_items(TASKS, SKILL_TEXTS, design="full_library", salt=SALT + "x")
        self.assertNotEqual(other[0]["item_id"], self.items[0]["item_id"])
        self.assertEqual(len(self.items), 3 * 18)
        export_a, export_b = rating.export_for_rater(self.items, "A"), rating.export_for_rater(self.items, "B")
        header = export_a.splitlines()[0].split(",")
        self.assertEqual(tuple(header), rating.EXPORT_COLUMNS)
        for word in ("rank", "score", "arm", "condition", "success", "skill_id", "task_id"):
            self.assertNotIn(word, header)
        self.assertNotRegex(export_a, r"skill_\d")                       # no skill identifiers
        self.assertNotEqual(export_a.splitlines()[1:], export_b.splitlines()[1:])   # per-rater order
        with self.assertRaises(rating.RatingError):
            rating.item_id("t", "s", "short")

    def test_forbidden_content_and_candidate_sets(self):
        with self.assertRaises(rating.RatingError):
            rating.build_items([{**TASKS[0], "recovery_context": "next: " + ORACLE_CONTINUATION[1]}], SKILL_TEXTS, design="full_library",
                               salt=SALT, forbidden_strings={TASKS[0]["task_id"]: ORACLE_CONTINUATION})
        with self.assertRaises(rating.RatingError):
            rating.build_items(TASKS, SKILL_TEXTS, design="candidate_set", salt=SALT)
        with self.assertRaises(rating.RatingError):
            rating.build_items(TASKS[:1], SKILL_TEXTS, design="candidate_set", salt=SALT, candidate_sets={TASKS[0]["task_id"]: ["nope"]})
        items, _ = rating.build_items(TASKS, SKILL_TEXTS, design="candidate_set", salt=SALT,
                                      candidate_sets={t["task_id"]: ["skill_00", "skill_01", "skill_02"] for t in TASKS})
        self.assertEqual(len(items), 9)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.items, self.key = rating.build_items(TASKS, SKILL_TEXTS, design="full_library", salt=SALT)
        self.ids = set(self.key["items"])
        self.export_a = rating.export_for_rater(self.items, "A")
        self.export_b = rating.export_for_rater(self.items, "B")
        self.lab_a = lambda iid: "RELEVANT" if int(iid[-1], 16) % 3 == 0 else "IRRELEVANT"
        self.lab_b = lambda iid: "UNCERTAIN" if iid.endswith("0") else self.lab_a(iid) if int(iid[-2], 16) % 4 else "PARTIAL"

    def parse(self, data, rater="A", decl=None):
        return rating.parse_submission(data, expected_item_ids=self.ids, rater_id=rater, declaration=decl or DECL(rater))

    def test_independent_submission_validation(self):
        sub = self.parse(fill(self.export_a, self.lab_a))
        self.assertEqual(len(sub), len(self.ids))
        bad_cases = [
            (fill(self.export_a, self.lab_a), "A", DECL("A", label_source="ai_model")),
            (fill(self.export_a, self.lab_a), "A", DECL("A", saw_other_ratings=True)),
            (fill(self.export_a, self.lab_a), "A", DECL("A", no_ai_or_automated_assistance=False)),
            (fill(self.export_a, self.lab_a), "A", DECL("A", saw_retrieval_ranks_or_scores=True)),
            (fill(self.export_a, self.lab_a), "A", DECL("A", email="someone@example.org")),
            (fill(self.export_a, self.lab_a), "A", DECL("A", coordinator_receipt=None)),
            (fill(self.export_a, self.lab_a), "A", DECL("A", coordinator_receipt={**DECL("A")["coordinator_receipt"], "received_by_code": "A"})),
            (fill(self.export_a, lambda i: "MAYBE"), "A", None),
            (fill(self.export_a, lambda i: ""), "A", None),
            (b"\n".join(fill(self.export_a, self.lab_a).splitlines()[:-1]) + b"\n", "A", None),   # missing item
            (fill(self.export_a, self.lab_a), "B", DECL("B")),                                    # rater id mismatch in rows
        ]
        text = fill(self.export_a, self.lab_a).decode()
        lines = text.splitlines()
        bad_cases.append(("\n".join(lines + [lines[1]]).encode(), "A", None))                    # duplicate item
        bad_cases.append((text.replace("rationale", "rationale,extra", 1).encode(), "A", None))   # unexpected column
        for data, rater, decl in bad_cases:
            with self.assertRaises(rating.RatingError):
                self.parse(data, rater, decl)

    def test_agreement_adjudication_and_freeze(self):
        a, b = self.parse(fill(self.export_a, self.lab_a)), self.parse(fill(self.export_b, self.lab_b), "B")
        report = rating.agreement(a, b)
        conflicts = set(report["requires_adjudication"])
        self.assertTrue(conflicts)
        self.assertTrue(all(a[i]["label"] != b[i]["label"] or a[i]["label"] == "UNCERTAIN" for i in conflicts))
        self.assertIsNotNone(report["kappa_four_category"])
        with self.assertRaises(rating.RatingError):
            rating.adjudicate(a, b, {}, rater_ids=("A", "B"))
        with self.assertRaises(rating.RatingError):
            rating.adjudicate(a, b, {i: {"label": "RELEVANT", "adjudicator_id": "A"} for i in conflicts}, rater_ids=("A", "B"))
        with self.assertRaises(rating.RatingError):
            rating.adjudicate(a, b, {i: {"label": "UNCERTAIN", "adjudicator_id": "C"} for i in conflicts}, rater_ids=("A", "B"))
        final = rating.adjudicate(a, b, {i: {"label": "PARTIAL", "adjudicator_id": "C"} for i in conflicts}, rater_ids=("A", "B"))
        self.assertTrue(all(final[i] == "PARTIAL" for i in conflicts))
        prov = {k: "a" * 64 for k in ("export_sha256", "rater_a_submission_sha256", "rater_b_submission_sha256", "adjudication_sha256",
                                      "rater_a_declaration_sha256", "rater_b_declaration_sha256", "adjudicator_declaration_sha256")}
        self.assertIn("independence_screen", report)
        frozen = rating.freeze_labels(final, self.key, provenance=prov)
        self.assertEqual(frozen["status"], "frozen_pending_human_approval")
        self.assertEqual(len(rating.label_map(frozen)), len(self.ids))
        self.assertEqual(frozen["labels_sha256"], rating.freeze_labels(final, self.key, provenance=prov)["labels_sha256"])
        with self.assertRaises(rating.RatingError):
            rating.freeze_labels(dict(list(final.items())[1:]), self.key, provenance=prov)       # missing label
        with self.assertRaises(rating.RatingError):
            rating.freeze_labels(final, self.key, provenance={**prov, "adjudication_sha256": "x"})


class ProvenanceTests(unittest.TestCase):
    def test_independence_screen_and_export_leak_scan(self):
        same = {f"i{k}": {"label": "RELEVANT", "rationale": "same words"} for k in range(25)}
        screen = rating.independence_screen(same, same)
        self.assertEqual(len(screen["flags"]), 2)
        for leak in ("Retrieval rank 1 was shown.", "similarity 0.8123", "the D18 condition"):
            with self.assertRaises(rating.RatingError):
                rating.build_items([{**TASKS[0], "recovery_context": leak}], SKILL_TEXTS, design="full_library", salt=SALT)


class WorkloadTests(unittest.TestCase):
    def test_from_frozen_rankings(self):
        rankings = {f"t{i}": [f"s{(i + k) % 18:02d}" for k in range(18)] for i in range(79)}
        w = rating.workload_from_rankings(rankings, rel_scope="natural_top3", irr_scope="natural_top3",
                                          pilot_tasks=[f"t{i}" for i in range(12)])
        self.assertEqual((w["pairs"], w["judgments_before_adjudication"], w["pilot_pairs"], w["pilot_judgments_before_adjudication"]),
                         (237, 474, 36, 72))
        wide = rating.workload_from_rankings(rankings, rel_scope="natural_top3", irr_scope="full_ranking")
        self.assertEqual(wide["pairs"], 79 * 18)

    def test_exact_counts(self):
        w = rating.workload(tasks=79, library_skills=18, candidate_per_task=3, pilot_tasks=12)
        self.assertEqual((w["full_library"]["pairs"], w["full_library"]["judgments"]), (1422, 2844))
        self.assertEqual((w["candidate_set"]["pairs"], w["candidate_set"]["judgments"]), (237, 474))
        self.assertEqual((w["full_library"]["pilot_judgments"], w["candidate_set"]["pilot_judgments"]), (432, 72))


if __name__ == "__main__":
    unittest.main()
