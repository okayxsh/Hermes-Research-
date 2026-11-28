"""University entry point for five-arm and pilot-diagnostic execution (gate first, then the real components).

Order inside ``build_university_queue`` (nothing real is imported or constructed before step 3):

1. artifact hashes are computed from the files named in the run configuration (CPU-only, no model);
2. readiness for the manifest's stage (``d0_failure_manipulation_pilot`` / ``five_arm_scientific_pilot`` /
   ``confirmatory``), observed-vs-approved runtime identity and the frozen, approved manifest are verified;
3. only then: provider metadata preflight, tokenizer + verified chat-template counter, the real episode driver, the
   Sentence-BERT embedder (arm mode only), the approved filler corpus (arm mode only), the memory policy, the executor
   and the resumable queue.

The old six-arm worker (``rq1.university_gpu``) cannot run this protocol: its manifests/conditions differ and this
module refuses anything that is not a five-arm or pilot-diagnostic manifest.  Real-provider behaviour is verified only
on the workstation; locally the same executor and queue run against a scripted fake driver (synthetic records only).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rq1.five_arm import readiness, runtime
from rq1.five_arm.executor import InterventionIntegrityError, ProtocolViolation  # noqa: F401  (re-exported)
from rq1.five_arm.manifest import DIAGNOSTIC_SCHEMA, SCHEMA as FIVE_ARM_SCHEMA, STAGE_LEVEL, validate_manifest
from rq1.five_arm.memory import FillerCorpus, Skill, Tokenizer, construct_arm_memory
from rq1.five_arm.protocol import BASE_CONDITION, CONDITIONS, UNDISTURBED_CONDITIONS


class ChatTemplateCounter:
    """Local count of the complete provider prompt: ``prefix + prompt + suffix`` (+ BOS).  Usable for real execution
    only when its spec is bound to reviewed provider-probe results (the probe establishes exact equality)."""

    def __init__(self, tokenizer: Tokenizer, spec: Mapping[str, Any]) -> None:
        for key in ("prefix", "suffix", "add_bos"):
            if key not in spec:
                raise ValueError(f"chat-template spec lacks {key}")
        self.tokenizer, self.spec = tokenizer, dict(spec)

    @property
    def sha256(self) -> str:
        body = {k: self.spec[k] for k in ("prefix", "suffix", "add_bos")}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def __call__(self, prompt: str) -> int:
        return len(self.tokenizer.encode(f"{self.spec['prefix']}{prompt}{self.spec['suffix']}")) + (1 if self.spec["add_bos"] else 0)


class FiveArmMemoryPolicy:
    """Memory transform for the executor: (unit, task, retrieval_outcome, failure_context) -> (memory, audit).
    Detoured conditions: verifies the live natural top-3 against the frozen ranking, the filler approval/tolerance and the
    frozen memory hash.  Undisturbed conditions (D0U, D18U): no retrieval may happen; the memory is the frozen block of the
    base condition (D18U shows exactly the D18 block retrieved at the post-detour state -- not an independent retrieval)."""

    def __init__(self, *, manifest: Mapping[str, Any], skill_texts: Mapping[str, str], corpus: FillerCorpus | None,
                 tokenizer: Tokenizer, tolerance: Mapping[str, Any], d0_representation: str) -> None:
        self.manifest, self.skill_texts, self.corpus = manifest, dict(skill_texts), corpus
        self.tokenizer, self.tolerance, self.d0_representation = tokenizer, dict(tolerance), d0_representation
        self.memory_hashes = {(u["task_id"], u["condition"]): u["memory_block_sha256"] for u in manifest["units"]}

    def __call__(self, unit: Any, task: Any, retrieval_outcome: Any, _failure_context: Mapping[str, Any]):
        arm = str(unit["condition"] if isinstance(unit, Mapping) else unit.condition)
        task_id = str(task.task_id)
        if arm not in CONDITIONS:
            raise InterventionIntegrityError(f"unsupported arm {arm!r}")
        selection = self.manifest["selections"][task_id]
        frozen_top3 = [item["skill_id"] for item in selection["natural_top3"]]
        if arm in UNDISTURBED_CONDITIONS:
            if retrieval_outcome.result.retrieved:
                raise InterventionIntegrityError(f"{task_id}/{arm}: an undisturbed condition performed a retrieval")
            live = [(item["skill_id"], item.get("score")) for item in selection["natural_top3"]]
        else:
            live = [(c.skill_id, float(c.score)) for c in retrieval_outcome.result.top]
            if [skill_id for skill_id, _ in live] != frozen_top3:
                raise InterventionIntegrityError(f"{task_id}: live natural top-3 {live} differs from the frozen ranking {frozen_top3}")
        natural = [Skill(skill_id, self.skill_texts[skill_id], score, rank) for rank, (skill_id, score) in enumerate(live, 1)]
        base = BASE_CONDITION.get(arm, arm)
        active = None
        if arm in ("Rel", "Irr"):
            chosen = selection[arm.lower()]
            if chosen is None:
                raise InterventionIntegrityError(f"{task_id}: {arm} has no frozen selection; the unit must not exist")
            active = Skill(chosen["skill_id"], self.skill_texts[chosen["skill_id"]], chosen.get("score"), chosen.get("rank"))
        if arm in ("Rel", "Irr", "Fill") and self.corpus is None:
            raise InterventionIntegrityError(f"{task_id}/{arm}: no filler corpus loaded")
        memory, audit = construct_arm_memory(base, natural_top3=natural, active_skill=active,
                                             corpus=self.corpus if arm in ("Rel", "Irr", "Fill") else None,
                                             tokenizer=self.tokenizer, tolerance=self.tolerance,
                                             d0_representation=self.d0_representation, salt=task_id)
        if arm in ("Rel", "Irr", "Fill") and (audit["within_token_tolerance"] is not True or audit["filler_corpus_approved"] is not True):
            raise InterventionIntegrityError(f"{task_id}/{arm}: unapproved filler or token mismatch; intervention refused")
        if audit["memory_block_sha256"] != self.memory_hashes.get((task_id, arm)):
            raise InterventionIntegrityError(f"{task_id}/{arm}: memory block differs from the frozen manifest")
        audit["natural_top3_skill_ids"] = frozen_top3
        audit["natural_top3_scores"] = [score for _, score in live]
        if arm in UNDISTURBED_CONDITIONS:
            audit["arm"] = arm
            audit["not_applicable"] = list(audit.get("not_applicable", [])) + ["retrieval_at_undisturbed_state"]
        return memory, audit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(config: Mapping[str, Any], approved_runtime: Mapping[str, Any]) -> dict[str, str]:
    """Hashes of the artifacts actually used: named files (bytes), the runtime configuration (canonical) and, when
    present, the filler corpus (canonical corpus hash)."""
    hashes = {key: file_sha256(Path(str(path))) for key, path in dict(config.get("artifact_files") or {}).items()}
    hashes["runtime_config"] = runtime.config_sha256(approved_runtime)
    if config.get("filler_corpus_path"):
        from rq1.five_arm.memory import load_filler_corpus
        hashes["filler_corpus"] = load_filler_corpus(Path(str(config["filler_corpus_path"]))).sha256
    return hashes


def load_exposure_eligibility(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Exposure eligibility evidence named in the run configuration (``exposure_eligibility``), read as exact bytes.

    developmental: ``{"kind": "developmental", <each of exposure_acceptance.DEVELOPMENTAL_EVIDENCE_FILES>: path,
    "approvals_record": path}`` -- the approvals record is the author record carrying
    ``developmental_pilot_exposure_acceptance`` and ``sequential_cpu_amendment``.
    independent: ``{"kind": "independent", "statement": path, "acceptance": path, "protocol_sha256": hex,
    "known_exposed_statements": [path, ...]}``."""
    from rq1.five_arm.exposure_acceptance import DEVELOPMENTAL_EVIDENCE_FILES
    spec = config.get("exposure_eligibility")
    if not spec:
        return None
    try:
        if spec["kind"] == "developmental":
            out: dict[str, Any] = {"kind": "developmental"}
            out.update({name: Path(str(spec[name])).read_bytes() for name in DEVELOPMENTAL_EVIDENCE_FILES})
            approvals = json.loads(Path(str(spec["approvals_record"])).read_text(encoding="utf-8"))
            out["exposure_acceptance"] = approvals.get("developmental_pilot_exposure_acceptance")
            out["cpu_amendment_approval"] = approvals.get("sequential_cpu_amendment")
            return out
        return {"kind": str(spec["kind"]), "statement": json.loads(Path(str(spec["statement"])).read_text(encoding="utf-8")),
                "acceptance": json.loads(Path(str(spec["acceptance"])).read_text(encoding="utf-8")),
                "protocol_sha256": str(spec["protocol_sha256"]),
                "known_exposed_statements": [Path(str(p)).read_bytes() for p in spec.get("known_exposed_statements") or []]}
    except (KeyError, OSError, ValueError) as exc:
        raise readiness.ExecutionBlocked(f"exposure eligibility evidence could not be loaded: {type(exc).__name__}: {exc}") from exc


def assert_execution_permitted(*, manifest: Mapping[str, Any], readiness_state: Mapping[str, Any], artifacts: Mapping[str, str],
                               observed_runtime: Mapping[str, Any] | None, approved_runtime: Mapping[str, Any],
                               exposure_eligibility: Mapping[str, Any] | None = None) -> str:
    """Raise unless REAL execution of this manifest is authorised.  Returns the readiness level that was checked.
    Seven-condition manifests additionally need stage-appropriate exposure eligibility (checked last, still before any
    provider contact)."""
    if manifest.get("schema") not in (FIVE_ARM_SCHEMA, DIAGNOSTIC_SCHEMA):
        raise readiness.ExecutionBlocked("not a five-arm or pilot-diagnostic manifest (six-arm manifests cannot run here)")
    if manifest.get("synthetic"):
        raise readiness.ExecutionBlocked("synthetic manifests can never be executed for real")
    level = STAGE_LEVEL.get(str(manifest.get("stage")))
    if level is None:
        raise readiness.ExecutionBlocked(f"unknown manifest stage {manifest.get('stage')!r}")
    readiness.require(level, readiness_state, artifacts, real=True)
    runtime.require_runtime(observed_runtime, approved_runtime)
    problems = validate_manifest(manifest)
    if problems or manifest.get("status") != "frozen" or not readiness.valid_approval(manifest.get("approval"),
                                                                                         manifest.get("manifest_sha256")):
        raise readiness.ExecutionBlocked("the manifest is not frozen with a human approval bound to its hash: " + "; ".join(problems))
    if manifest["input_hashes"].get("runtime_config_sha256") != runtime.config_sha256(approved_runtime):
        raise readiness.ExecutionBlocked("the manifest is bound to a different runtime configuration")
    if manifest["source_commit"] != readiness_state.get("code_commit"):
        raise readiness.ExecutionBlocked("the manifest was built from a different code commit than the reviewed one")
    from rq1.five_arm.exposure_acceptance import exposure_eligibility_problems
    exposure = exposure_eligibility_problems(manifest, exposure_eligibility, readiness_state=readiness_state,
                                             artifacts=artifacts, real=True)
    if exposure:
        raise readiness.ExecutionBlocked("exposure eligibility refused: " + "; ".join(exposure))
    return level


def build_university_queue(config: Mapping[str, Any], *, readiness_state: Mapping[str, Any],
                           observed_runtime: Mapping[str, Any] | None, approved_runtime: Mapping[str, Any]):
    """Sequential path: gate, then the real executor and the resumable queue.  Returns (queue, driver)."""
    from rq1.five_arm.queue import RunQueue
    built = build_university_executor(config, readiness_state=readiness_state, observed_runtime=observed_runtime,
                                      approved_runtime=approved_runtime)
    try:
        queue = RunQueue(output_root=Path(str(config["output_root"])), manifest=config["manifest"], executor=built["dispatch"],
                         kind=built["kind"], max_infrastructure_retries=int(config["max_infrastructure_retries"]),
                         synthetic=False, interruption_record=built["interruption_record"])
    except BaseException:
        built["driver"].__exit__(None, None, None)
        raise
    return queue, built["driver"]


def build_university_executor(config: Mapping[str, Any], *, readiness_state: Mapping[str, Any],
                              observed_runtime: Mapping[str, Any] | None, approved_runtime: Mapping[str, Any],
                              endpoint: str | None = None, execution_provenance: Mapping[str, Any] | None = None,
                              bridge_log_root: Path | None = None) -> dict[str, Any]:
    """Gate, then construct the real executor(s) for ONE provider endpoint (a parallel lane passes its own endpoint).
    Returns {dispatch, interruption_record, kind, driver}; call ``driver.__exit__`` when done."""
    if endpoint is not None:
        config = {**config, "ollama_endpoint": endpoint}
    manifest = config["manifest"]
    artifacts = artifact_hashes(config, approved_runtime)
    assert_execution_permitted(manifest=manifest, readiness_state=readiness_state, artifacts=artifacts,
                               observed_runtime=observed_runtime, approved_runtime=approved_runtime,
                               exposure_eligibility=load_exposure_eligibility(config))
    # --- only reachable after every scientific and runtime gate has passed -----------------------------------------
    from rq1.evaluation.amended_libraries import AmendedLibrary
    from rq1.evaluation.amended_matrix import EvaluationTask
    from rq1.five_arm.executor import FiveArmEpisodeExecutor
    from rq1.five_arm.memory import load_filler_corpus
    from rq1.five_arm.prompting import LeakageProbe
    from rq1.hermes.episode_driver import RealEpisodeDriver
    from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity
    from rq1.university_gpu.ollama import probe_endpoint

    probe = probe_endpoint(str(config["ollama_endpoint"]), expected_model=approved_runtime["model_tag"],
                           expected_digest=approved_runtime["model_manifest_digest"])
    if probe.get("status") != "PASS":
        raise runtime.RuntimeIdentityError(f"provider preflight failed: {probe}")
    tokenizer = LocalGemmaTokenizer(TokenizerIdentity(Path(str(config["tokenizer_path"])), approved_runtime["tokenizer_sha256"],
                                                      approved_runtime["tokenizer_revision"]))
    counter = ChatTemplateCounter(tokenizer, config["chat_template_spec"])
    if counter.sha256 != approved_runtime.get("chat_template_counter_sha256"):
        raise runtime.RuntimeIdentityError("chat-template counter spec differs from the approved chat_template_counter_sha256")
    diagnostic = manifest["schema"] == DIAGNOSTIC_SCHEMA
    tasks = {}
    for t in manifest["tasks"]:
        spec = dict(config["evaluation_tasks"][str(t["task_id"])])
        spec["prefix_actions"], spec["reference_actions"] = tuple(spec["prefix_actions"]), tuple(spec["reference_actions"])
        tasks[str(t["task_id"])] = EvaluationTask(**spec)
    task_meta = {str(t["task_id"]): t for t in manifest["tasks"]}
    arm_parts: dict[str, Any] = {}
    if not diagnostic:                      # everything that can fail on inputs is checked BEFORE the bridge starts
        from rq1.retrieval.embedder import SentenceBERTEmbedder
        corpus = load_filler_corpus(Path(str(config["filler_corpus_path"])))
        if corpus.sha256 != manifest["input_hashes"]["filler_corpus_sha256"] or not corpus.approved:
            raise InterventionIntegrityError("filler corpus differs from the manifest or is not approved")
        arm_parts["library"] = AmendedLibrary("Accum-18", tuple(dict(s) for s in config["accum18_library"]["skills"]))
        arm_parts["embedder"] = SentenceBERTEmbedder(approved_runtime["sbert_model"], cache_folder=Path(str(config["sbert_cache"])),
                                                     revision=approved_runtime["sbert_revision"], local_files_only=True)
        arm_parts["memory_policy"] = FiveArmMemoryPolicy(manifest=manifest, skill_texts=config["skill_texts"], corpus=corpus,
                                                         tokenizer=tokenizer, tolerance=config["token_tolerance"],
                                                         d0_representation=str(config["d0_representation"]))
        arm_parts["leakage_probes"] = {
            task_id: LeakageProbe(oracle_continuation=tuple(task.reference_actions[len(task.prefix_actions):]),
                                  expected_next_action=task.expected_next_action, skill_ids=tuple(config["skill_texts"]),
                                  library_texts=tuple(config["skill_texts"].values()))
            for task_id, task in tasks.items()}
        arm_parts["analysis_labels"] = dict(config.get("analysis_labels") or {})
    driver = RealEpisodeDriver(Path(str(config["repository_root"])).resolve(), data_dir=Path(str(config["alfworld_data_dir"])),
                               model_name=approved_runtime["model_tag"], ollama_url=str(config["ollama_endpoint"]),
                               bridge_log_root=bridge_log_root or Path(str(config["output_root"])) / "bridge")
    driver.__enter__()
    try:
        common = dict(tasks=tasks, task_meta=task_meta, split=manifest["split"], synthetic=False, tokenizer=tokenizer,
                      runtime_identity_sha256=runtime.config_sha256(approved_runtime), exposure_policy=str(config["exposure_policy"]),
                      d0_representation=str(config["d0_representation"]), num_ctx=int(approved_runtime["num_ctx"]),
                      max_output_tokens=int(approved_runtime["num_predict"]), chat_template_tokens=counter,
                      provider_token_tolerance=int(config["provider_token_tolerance"]),
                      execution_provenance=execution_provenance)
        if diagnostic:
            modes = {"UNDISTURBED": "diagnostic_undisturbed", "DETOURED": "diagnostic_detoured"}
            executors = {c: FiveArmEpisodeExecutor(driver, mode=m, **common) for c, m in modes.items()}
            dispatch = lambda unit, **kw: executors[unit["condition"]](unit, **kw)
            interrupted = lambda unit, attempt, detail: executors[unit["condition"]].interruption_record(unit, attempt, detail)
            kind = "diagnostic"
        else:
            executor = FiveArmEpisodeExecutor(driver, mode="arm", **arm_parts, **common)
            dispatch, interrupted, kind = executor, executor.interruption_record, "five_arm"
    except BaseException:
        driver.__exit__(None, None, None)
        raise
    return {"dispatch": dispatch, "interruption_record": interrupted, "kind": kind, "driver": driver}
