"""SYNTHETIC fixtures for the five-arm tests.  Nothing here is scientific evidence, a real skill, a real label, a real
tokenizer or an approved filler corpus (the 'approval' below is a fixture used only to exercise code paths)."""
from __future__ import annotations

import re

from rq1.five_arm.memory import FillerCorpus, Skill
from rq1.five_arm.prompting import CheckpointContext


class SyntheticTokenizer:
    """Word/punctuation tokenizer -- SYNTHETIC; not the Gemma tokenizer."""
    _pattern = re.compile(r"\w+|[^\w\s]")

    def encode(self, text: str):
        return [hash(tok) & 0xFFFF for tok in self._pattern.findall(text)]


SKILL_TEXTS = {f"skill_{i:02d}": text for i, text in enumerate([
    "To heat an item, carry it to the microwave, heat it, then place it in the target receptacle.",
    "To cool an item, carry it to the fridge, cool it, then put it in the target receptacle.",
    "To clean an item, take it to the sink basin, clean it, then place it where the goal requires.",
    "To examine an object under light, pick up the object and switch on a nearby desk lamp.",
    "To place an object, find it on a countertop or shelf, take it, and put it in the receptacle.",
    "When two objects are needed, collect the first, place it, then return for the second.",
    "Check drawers and cabinets systematically when an object is not visible on surfaces.",
    "If a receptacle is closed, open it before trying to put an object inside.",
    "After a detour, look around and return to the last location that held the target object.",
    "Slice food with a knife before heating or cooling when the goal mentions a sliced item.",
    "Use the inventory action to confirm what you are holding before placing it.",
    "Go to the garbage can when the goal asks to dispose of an item there.",
    "Visit each countertop in turn until the requested utensil is found.",
    "Put cleaned dishes back on the dining table if the goal names that table.",
    "A lamp must be turned on before an object can be examined in its light.",
    "Navigate to the sofa or armchair to find remote controls and pillows.",
    "Bathroom items such as soap bars are usually near the toilet or the bathtub.",
    "Kitchen tools like spatulas and ladles are often stored in drawers near the stove.",
])}
LIBRARY_SCORES = {skill_id: round(0.9 - index * 0.03, 6) for index, skill_id in enumerate(sorted(SKILL_TEXTS))}

NEUTRAL_SENTENCES = (
    "Autumn leaves drift slowly across the quiet meadow.", "The old lighthouse stands tall beside the harbour.",
    "Gentle rain softens the colours of the distant hills.", "A small orchestra rehearsed a cheerful melody yesterday.",
    "Clouds gathered above the valley before sunset.", "The library catalogue lists many historical maps.",
    "Migrating birds rested briefly near the river delta.", "An artist painted a calm seascape in pale blue tones.",
    "Snow covered the mountain pass during the winter months.", "A narrow footpath winds through the pine forest.",
    "The museum displays fossils from an ancient coastline.", "Bright lanterns decorated the festival streets.",
    "Quiet evenings in the countryside feel unhurried.", "The telescope revealed faint rings around a distant planet.",
    "Wildflowers bloom along the edges of the old railway.", "Sailors once relied on the stars for orientation.",
    "The poem describes a journey across a frozen lake.", "Morning fog lifted slowly from the harbour.",
    "A gentle breeze carried the scent of salt and seaweed.", "Rain fell.", "Birds sang softly.", "Waves rolled.",
    "Stars shone.", "Leaves rustled.", "Bells rang.", "Snow melted.", "Owls called.", "Tides turned.",
)


def corpus(approved: bool = True) -> FillerCorpus:
    base = FillerCorpus("synthetic-neutral-v0", NEUTRAL_SENTENCES)
    review = {"approved": True, "approved_by": "FIXTURE-NOT-A-PERSON", "approval_reference": "unit-test",
              "corpus_sha256": base.sha256} if approved else {}
    return FillerCorpus(base.corpus_id, base.sentences, review)


def natural_top3() -> list[Skill]:
    ranked = sorted(LIBRARY_SCORES.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return [Skill(skill_id, SKILL_TEXTS[skill_id], score, rank) for rank, (skill_id, score) in enumerate(ranked, 1)]


TOLERANCE = {"absolute_tokens": 8, "relative": 0.02}

CONTEXT = CheckpointContext(
    task_goal="put a clean mug in coffeemachine.",
    initial_observation="You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a coffeemachine 1, "
                        "a countertop 1, and a sinkbasin 1.\nYour task is to: put a clean mug in coffeemachine.",
    observation="You arrive at cabinet 1. The cabinet 1 is closed.",
    admissible_actions=("go to coffeemachine 1", "go to countertop 1", "go to sinkbasin 1", "look", "open cabinet 1"),
    history=(("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a mug 1."),
             ("take mug 1 from countertop 1", "You pick up the mug 1 from the countertop 1."),
             ("go to cabinet 1", "You arrive at cabinet 1. The cabinet 1 is closed.")),
)
ORACLE_CONTINUATION = ("go to sinkbasin 1", "clean mug 1 with sinkbasin 1", "go to coffeemachine 1", "put mug 1 in/on coffeemachine 1")
FUTURE_OBSERVATION = "You clean the mug 1 using the sinkbasin 1."


def approval(sha: str = "a" * 64, **extra) -> dict:
    return {"approved": True, "approved_by": "FIXTURE-NOT-A-PERSON", "approval_reference": "unit-test", "artifact_sha256": sha, **extra}
