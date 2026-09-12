"""Tests for authoritative named skill-library construction."""
from __future__ import annotations

import unittest
from pathlib import Path

from rq1.skills.library import (
    ACCUM_EXTRAS_PER_FAMILY,
    CONDITIONS,
    CORE_PER_FAMILY,
    LIBRARY_SIZES,
    TASK_FAMILIES,
    AcquiredSkill,
    LibraryConstructionError,
    build_libraries,
)
from rq1.utils.config import load_json_yaml


def _skill(skill_id: str, family: str, index: int, validated: bool = True) -> AcquiredSkill:
    return AcquiredSkill(
        skill_id=skill_id,
        title=f"Skill {skill_id}",
        body=f"Body of {skill_id}",
        task_family=family,
        acquisition_index=index,
        validated=validated,
    )


def _saturated_skills() -> list[AcquiredSkill]:
    """Enough validated + extra skills per family to fill Accum-96.

    Each family gets 4 validated (core) plus 12 extra successful skills.
    """
    skills: list[AcquiredSkill] = []
    index = 0
    for family in TASK_FAMILIES:
        for position in range(CORE_PER_FAMILY + ACCUM_EXTRAS_PER_FAMILY["Accum-96"]):
            index += 1
            skills.append(_skill(f"{family}-{position:02d}", family, index, validated=True))
    return skills


class LibrarySizeTests(unittest.TestCase):
    def test_condition_sizes_are_frozen(self) -> None:
        self.assertEqual(LIBRARY_SIZES, {"NoLib": 0, "Clean-24": 24, "Accum-60": 60, "Accum-96": 96})
        self.assertEqual(CONDITIONS, ("NoLib", "Clean-24", "Accum-60", "Accum-96"))
        self.assertEqual(len(TASK_FAMILIES), 6)
        self.assertEqual(CORE_PER_FAMILY, 4)

    def test_config_matches_code_constants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_json_yaml(root / "configs" / "libraries.yaml")
        self.assertEqual(tuple(config["conditions"]), CONDITIONS)
        self.assertEqual(config["library_sizes"], LIBRARY_SIZES)
        self.assertEqual(tuple(config["task_families"]), TASK_FAMILIES)
        self.assertEqual(config["core_per_family"], CORE_PER_FAMILY)
        self.assertEqual(config["accum_extras_per_family"], ACCUM_EXTRAS_PER_FAMILY)
        self.assertTrue(config["core_identical_across_conditions"])
        self.assertTrue(config["do_not_deduplicate"])


class BuildLibrariesTests(unittest.TestCase):
    def test_builds_all_four_conditions_with_correct_sizes(self) -> None:
        libraries = build_libraries(_saturated_skills())
        self.assertEqual(set(libraries), set(CONDITIONS))
        for condition in CONDITIONS:
            self.assertEqual(libraries[condition].library_size, LIBRARY_SIZES[condition])

    def test_no_lib_is_empty(self) -> None:
        libraries = build_libraries(_saturated_skills())
        self.assertEqual(libraries["NoLib"].skills, ())
        self.assertEqual(libraries["NoLib"].library_size, 0)

    def test_core_is_identical_across_conditions(self) -> None:
        libraries = build_libraries(_saturated_skills())
        self.assertEqual(
            libraries["Clean-24"].core_sha256(),
            libraries["Accum-60"].core_sha256(),
        )
        self.assertEqual(
            libraries["Clean-24"].core_sha256(),
            libraries["Accum-96"].core_sha256(),
        )

    def test_core_takes_earliest_validated_per_family(self) -> None:
        skills = []
        index = 0
        for family in TASK_FAMILIES:
            # first 4 validated, then 12 more successful
            for position in range(4 + 12):
                index += 1
                skills.append(_skill(f"{family}-{position}", family, index, validated=True))
        libraries = build_libraries(skills)
        core = libraries["Clean-24"]
        for family in TASK_FAMILIES:
            family_core = [s for s in core.skills if s.task_family == family]
            self.assertEqual(len(family_core), CORE_PER_FAMILY)
            self.assertEqual([s.skill_id for s in family_core], [f"{family}-{position}" for position in range(4)])

    def test_accum_60_and_96_are_nested(self) -> None:
        libraries = build_libraries(_saturated_skills())
        c60_ids = [s.skill_id for s in libraries["Accum-60"].skills]
        c96_ids = [s.skill_id for s in libraries["Accum-96"].skills]
        self.assertTrue(set(c60_ids).issubset(set(c96_ids)))
        self.assertEqual(len(c60_ids), 60)
        self.assertEqual(len(c96_ids), 96)

    def test_unvalidated_skills_can_be_extras_but_not_core(self) -> None:
        skills = []
        index = 0
        for family in TASK_FAMILIES:
            for position in range(4):
                index += 1
                skills.append(_skill(f"{family}-core{position}", family, index, validated=True))
            for position in range(12):
                index += 1
                skills.append(_skill(f"{family}-extra{position}", family, index, validated=False))
        libraries = build_libraries(skills)
        core_ids = {s.skill_id for s in libraries["Clean-24"].skills}
        for skill in skills:
            if skill.validated:
                self.assertIn(skill.skill_id, core_ids)
            else:
                self.assertNotIn(skill.skill_id, core_ids)

    def test_preserves_near_duplicate_text(self) -> None:
        # Two extras in pick_and_place share identical title/body; both must be
        # retained (never deduplicated) within the accumulated quotas.
        skills = []
        index = 0
        for family in TASK_FAMILIES:
            for position in range(4):
                index += 1
                skills.append(_skill(f"{family}-core{position}", family, index, validated=True))
            for position in range(12):
                index += 1
                skills.append(_skill(f"{family}-extra{position}", family, index, validated=True))
        # Overwrite the two earliest pick_and_place extras with identical text.
        duplicate_a = AcquiredSkill("dup-a", "Same title", "Same body", "pick_and_place", 5, True)
        duplicate_b = AcquiredSkill("dup-b", "Same title", "Same body", "pick_and_place", 6, True)
        # Remove the originals at those indices and insert the duplicates.
        keep = [s for s in skills if s.acquisition_index not in {5, 6}]
        keep.extend([duplicate_a, duplicate_b])
        libraries = build_libraries(keep)
        accum_96_texts = [s.text for s in libraries["Accum-96"].skills]
        self.assertIn(duplicate_a.retrieval_text(), accum_96_texts)
        self.assertIn(duplicate_b.retrieval_text(), accum_96_texts)
        self.assertEqual(accum_96_texts.count(duplicate_a.retrieval_text()), 2)


class QuotaFailureTests(unittest.TestCase):
    def test_fails_closed_when_core_quota_unmet(self) -> None:
        # Only 3 validated skills in pick_and_place -> cannot build core.
        skills = [_skill(f"pp-{i}", "pick_and_place", i, validated=True) for i in range(3)]
        with self.assertRaises(LibraryConstructionError):
            build_libraries(skills)

    def test_fails_closed_when_accum_quota_unmet(self) -> None:
        skills = []
        index = 0
        for family in TASK_FAMILIES:
            # 4 validated core but only 2 extras -> Accum-60 needs 6.
            for position in range(4):
                index += 1
                skills.append(_skill(f"{family}-core{position}", family, index, validated=True))
            for position in range(2):
                index += 1
                skills.append(_skill(f"{family}-extra{position}", family, index, validated=True))
        with self.assertRaises(LibraryConstructionError):
            build_libraries(skills)

    def test_rejects_unknown_family(self) -> None:
        with self.assertRaises(LibraryConstructionError):
            build_libraries([_skill("x", "not_a_family", 1, validated=True)])

    def test_rejects_duplicate_skill_id(self) -> None:
        with self.assertRaises(LibraryConstructionError):
            build_libraries(
                [_skill("dup", "pick_and_place", 1, True), _skill("dup", "pick_and_place", 2, True)]
            )


class ManifestTests(unittest.TestCase):
    def test_content_hash_is_stable_and_distinct(self) -> None:
        libraries = build_libraries(_saturated_skills())
        self.assertEqual(libraries["Clean-24"].content_sha256(), libraries["Clean-24"].content_sha256())
        self.assertNotEqual(libraries["Clean-24"].content_sha256(), libraries["Accum-60"].content_sha256())

    def test_to_dict_includes_hashes(self) -> None:
        libraries = build_libraries(_saturated_skills())
        payload = libraries["Clean-24"].to_dict()
        self.assertEqual(payload["condition"], "Clean-24")
        self.assertEqual(payload["library_size"], 24)
        self.assertIn("content_sha256", payload)
        self.assertIn("core_sha256", payload)


if __name__ == "__main__":
    unittest.main()
