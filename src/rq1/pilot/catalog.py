"""Immutable catalog for the ordered Phase 6 pilot sequence."""
from __future__ import annotations

from rq1.pilot.models import (
    BlockingClass,
    CapabilityRequirement,
    CleanupPolicy,
    EvidenceLevel,
    PilotMode,
    PilotTestSpec,
    RetryPolicy,
)


_NAMES = (
    "Repository self-test",
    "Machine doctor",
    "Installation verification",
    "Raw model response",
    "Raw model tool-call compatibility",
    "Basic Hermes profile test",
    "Hermes plugin discovery",
    "Hermes tool dispatch",
    "Profile isolation",
    "Native skill retrieval",
    "Skill persistence",
    "Evaluation write protection",
    "Standalone ALFWorld bridge",
    "Hermes-to-ALFWorld one-step test",
    "Multi-step Hermes-ALFWorld loop",
    "Complete episode test",
    "Checkpoint creation",
    "Checkpoint replay equality",
    "Controlled perturbation application",
    "Perturbation solvability",
    "Recovery-start context",
    "Controlled recovery without learned skills",
    "Controlled recovery with one relevant skill",
    "Controlled recovery with distractors",
    "Recovery log reconciliation",
    "Session contamination",
    "Crash and failure handling",
    "Mini acquisition",
    "Mini chronological snapshots",
    "Mini paired recovery evaluation",
    "Parallel worker test",
    "Resume test",
    "Runtime and capacity benchmark",
    "Recovery relevance-labelling test",
    "Candidate model decision",
    "Recovery protocol decision",
    "Final pilot report",
)


def _group(index: int) -> str:
    if index <= 2: return "foundation"
    if index <= 11: return "model-hermes"
    if index <= 15: return "environment"
    if index <= 24: return "recovery"
    if index <= 26: return "resilience"
    if index <= 33: return "mini-workflow"
    return "decisions"


def _timeout(index: int) -> int:
    if index in {0, 36}: return 180
    if index <= 2: return 60
    if index <= 11: return 300
    if index <= 26: return 900
    return 3600 if index <= 33 else 180


def _real_level(index: int) -> EvidenceLevel:
    if index == 0: return EvidenceLevel.STATIC
    if index <= 2: return EvidenceLevel.INSTALLED
    if index <= 12: return EvidenceLevel.REAL_COMPONENT
    if index <= 35: return EvidenceLevel.REAL_INTEGRATED
    return EvidenceLevel.STATIC


def _capabilities(index: int) -> tuple[CapabilityRequirement, ...]:
    if index <= 2 or index == 36: return ()
    if index <= 4: return (CapabilityRequirement("ollama_model", EvidenceLevel.REAL_COMPONENT),)
    if index <= 11: return (CapabilityRequirement("hermes", EvidenceLevel.REAL_COMPONENT),)
    if index == 12: return (CapabilityRequirement("alfworld", EvidenceLevel.REAL_COMPONENT),)
    return (
        CapabilityRequirement("hermes", EvidenceLevel.REAL_COMPONENT),
        CapabilityRequirement("alfworld", EvidenceLevel.REAL_COMPONENT),
    )


def _cleanup(index: int) -> CleanupPolicy:
    return CleanupPolicy.EXPLICIT_DESTRUCTIVE if index in {8, 9, 10, 11, 21, 22, 23, 27, 28, 29} else CleanupPolicy.NONE


PILOT_TESTS = tuple(
    PilotTestSpec(
        test_id=f"pilot_{index:02d}",
        name=name,
        purpose=name,
        group=_group(index),
        prerequisites=(() if index in {0, 36} else (f"pilot_{index - 1:02d}",)),
        supported_modes=(PilotMode.FAKE, PilotMode.REAL),
        required_capabilities=_capabilities(index),
        timeout_seconds=_timeout(index),
        retry_policy=RetryPolicy.READ_ONLY_ONCE if index in {0, 1, 2} else RetryPolicy.NEW_ATTEMPT_ONLY,
        cleanup_policy=_cleanup(index),
        blocking=BlockingClass.BLOCKING,
        fake_evidence=EvidenceLevel.STATIC if index == 0 else EvidenceLevel.MOCK,
        real_evidence=_real_level(index),
    )
    for index, name in enumerate(_NAMES)
)
PILOT_TEST_MAP = {test.test_id: test for test in PILOT_TESTS}
PILOT_GROUPS = tuple(dict.fromkeys(test.group for test in PILOT_TESTS))


def validate_catalog() -> list[str]:
    errors: list[str] = []
    if len(PILOT_TESTS) != 37 or len(PILOT_TEST_MAP) != 37:
        errors.append("Pilot catalog must contain unique pilot_00 through pilot_36 entries")
    expected = [f"pilot_{index:02d}" for index in range(37)]
    if [test.test_id for test in PILOT_TESTS] != expected:
        errors.append("Pilot IDs are not contiguous")
    for test in PILOT_TESTS:
        for prerequisite in test.prerequisites:
            if prerequisite not in PILOT_TEST_MAP:
                errors.append(f"{test.test_id} has unknown prerequisite {prerequisite}")
            elif expected.index(prerequisite) >= expected.index(test.test_id):
                errors.append(f"{test.test_id} has a non-acyclic prerequisite")
    return errors


def select_tests(
    *, test_id: str | None = None, group: str | None = None,
    start: str | None = None, end: str | None = None, include_prerequisites: bool = False,
) -> tuple[PilotTestSpec, ...]:
    selectors = sum(value is not None for value in (test_id, group, start, end))
    if test_id and selectors != 1:
        raise ValueError("--test cannot be combined with group or range selection")
    if group and selectors != 1:
        raise ValueError("--group cannot be combined with test or range selection")
    if bool(start) != bool(end):
        raise ValueError("--from and --to must be supplied together")
    if test_id:
        if test_id not in PILOT_TEST_MAP: raise ValueError(f"Unknown pilot test: {test_id}")
        selected = [PILOT_TEST_MAP[test_id]]
    elif group:
        selected = [item for item in PILOT_TESTS if item.group == group]
        if not selected: raise ValueError(f"Unknown pilot group: {group}")
    elif start and end:
        ids = [item.test_id for item in PILOT_TESTS]
        if start not in ids or end not in ids: raise ValueError("Unknown pilot range endpoint")
        left, right = ids.index(start), ids.index(end)
        if left > right: raise ValueError("Pilot range start must not follow its end")
        selected = list(PILOT_TESTS[left:right + 1])
    else:
        selected = list(PILOT_TESTS)
    if include_prerequisites:
        required = {item.test_id for item in selected}
        pending = list(required)
        while pending:
            current = PILOT_TEST_MAP[pending.pop()]
            for prerequisite in current.prerequisites:
                if prerequisite not in required:
                    required.add(prerequisite); pending.append(prerequisite)
        selected = [item for item in PILOT_TESTS if item.test_id in required]
    return tuple(selected)
