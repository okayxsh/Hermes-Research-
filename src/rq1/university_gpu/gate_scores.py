"""Build valid-seen Gate calibration scores using the frozen retrieval query."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.retrieval.models import RetrievalQuery
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE
from rq1.retrieval.retriever import cosine_similarity


def compute_scores(contexts: Sequence[Mapping[str, Any]], skills: Mapping[str, str], embedder: Any) -> list[dict[str, Any]]:
    if len(contexts) != 6 or len({str(row.get("task_id")) for row in contexts}) != 6:
        raise ValueError("Gate score generation requires six distinct valid-seen contexts")
    if len(skills) != 18 or any(not key or not value for key, value in skills.items()):
        raise ValueError("Gate score generation requires 18 distinct nonempty skill documents")
    queries = []
    for row in contexts:
        task_id = str(row.get("task_id", ""))
        instruction, observation = str(row.get("task_instruction", "")), str(row.get("observation", ""))
        inventory = row.get("inventory")
        if (not task_id.startswith("valid_seen:") or not instruction.strip() or not observation.strip()
                or not isinstance(inventory, list) or not all(isinstance(value, str) for value in inventory)
                or row.get("failure_message") != CANONICAL_FAILURE_MESSAGE):
            raise ValueError(f"invalid frozen valid-seen failure context: {task_id}")
        queries.append(RetrievalQuery(instruction, observation, tuple(inventory), CANONICAL_FAILURE_MESSAGE).text())
    ordered = sorted(skills.items())
    vectors = embedder.encode(queries + [text for _, text in ordered])
    if len(vectors) != len(contexts) + len(ordered) or not vectors or any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("embedding count or dimension mismatch")
    if (not vectors[0] or any(not math.isfinite(float(value)) for vector in vectors for value in vector)
            or any(not any(float(value) != 0.0 for value in vector) for vector in vectors)):
        raise ValueError("invalid embedding values")
    rows = []
    for index, context in enumerate(contexts):
        for offset, (skill_id, _text) in enumerate(ordered):
            score = cosine_similarity(vectors[index], vectors[len(contexts) + offset])
            if not math.isfinite(score) or not -1.000000001 <= score <= 1.000000001:
                raise ValueError("invalid cosine score")
            rows.append({"task_id": context["task_id"], "skill_id": skill_id,
                         "cosine_similarity": min(1.0, max(-1.0, score))})
    return rows


def prepare_scores(*, tasks_path: Path, failures_path: Path, contexts_path: Path, library_path: Path,
                   relevance_path: Path, cache: Path, output: Path) -> dict[str, Any]:
    from rq1.acquisition.environment import sbert_snapshot_sha256
    from rq1.evaluation.amended_protocol import SBERT_SNAPSHOT_SHA256
    from rq1.university_gpu.manifest import SBERT_MODEL, SBERT_REVISION
    from rq1.university_gpu.models import atomic_write_json, sha256_file
    from rq1.university_gpu.ops import _library_payload, _tasks_and_failures
    from rq1.university_gpu.relevance import parse_relevance_csv
    from rq1.skills.library import TASK_FAMILIES

    if output.exists():
        raise ValueError("score output directory already exists")
    source_paths = {"contexts": contexts_path, "tasks": tasks_path, "failures": failures_path,
                    "library": library_path, "relevance": relevance_path}
    source_hashes = {key: sha256_file(path) for key, path in source_paths.items()}
    tasks = _tasks_and_failures(tasks_path, failures_path, stage="pilot")
    if len(tasks) != 6 or {task["task_family"] for task in tasks} != set(TASK_FAMILIES):
        raise ValueError("pilot requires six oracle-validated tasks, one per family")
    skills, texts, _ = _library_payload(library_path)
    parse_relevance_csv(relevance_path.read_bytes(), allowed_tasks={str(t["task_id"]) for t in tasks}, library_skill_ids=set(skills))
    if sbert_snapshot_sha256(cache) != SBERT_SNAPSHOT_SHA256:
        raise ValueError("local Sentence-BERT snapshot is missing or differs from the frozen hash")
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if (not isinstance(contexts, list) or any(not isinstance(row, dict) for row in contexts)
            or {row.get("task_id") for row in contexts} != {task["task_id"] for task in tasks}):
        raise ValueError("failure contexts must cover exactly the frozen pilot tasks")
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    observed = {row["task_id"]: row["post_detour_observation"] for row in failures["tasks"]}
    if any(row.get("observation") != observed[row["task_id"]] for row in contexts):
        raise ValueError("failure context observation differs from the frozen post-detour observation")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError("local sentence-transformers is unavailable; no download is attempted") from exc

    class CPUEmbedder:
        def __init__(self):
            self.model = SentenceTransformer(SBERT_MODEL, revision=SBERT_REVISION, cache_folder=str(cache),
                                             local_files_only=True, device="cpu")

        def encode(self, values):
            vectors = self.model.encode(values, convert_to_numpy=True, normalize_embeddings=True).tolist()
            if any(len(vector) != 768 for vector in vectors):
                raise ValueError("frozen Sentence-BERT embedding dimension must be 768")
            return vectors

    rows = compute_scores(contexts, texts, CPUEmbedder())
    if any(sha256_file(path) != source_hashes[key] for key, path in source_paths.items()):
        raise ValueError("a Gate scoring input changed during computation")
    if sbert_snapshot_sha256(cache) != SBERT_SNAPSHOT_SHA256:
        raise ValueError("Sentence-BERT snapshot changed during computation")
    output.mkdir(parents=True)
    with (output / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "skill_id", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(rows)
    with relevance_path.open(encoding="utf-8-sig", newline="") as src, (output / "labels.csv").open("x", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["task_id", "skill_id", "adjudicated"])
        writer.writeheader()
        for row in csv.DictReader(src):
            writer.writerow({key: row[key] for key in writer.fieldnames})
    evidence = {"schema": "rq1-gate-score-evidence-v1", "device": "cpu", "model": SBERT_MODEL,
                "revision": SBERT_REVISION, "snapshot_sha256": SBERT_SNAPSHOT_SHA256,
                "contexts_sha256": source_hashes["contexts"], "tasks_sha256": source_hashes["tasks"],
                "failures_sha256": source_hashes["failures"], "library_sha256": source_hashes["library"],
                "relevance_sha256": source_hashes["relevance"], "scores_sha256": sha256_file(output / "scores.csv"),
                "labels_sha256": sha256_file(output / "labels.csv"), "pairs": len(rows),
                "query_hashes": {row["task_id"]: RetrievalQuery(row["task_instruction"], row["observation"],
                                                                  tuple(row["inventory"]), row["failure_message"]).text_hash()
                                 for row in contexts},
                "warning": "Context task instructions and inventory require independent human review and comparison with observed pilot contexts."}
    atomic_write_json(output / "SCORE-EVIDENCE.json", evidence, overwrite=False)
    return evidence
