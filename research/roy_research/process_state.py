from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, Tuple


PROCESS_STATE_SCHEMA_VERSION = 1
SEMANTIC_LABELS = ("entail", "contradict", "unknown")

EXTRACTOR_SYSTEM_PROMPT = """You are a frozen epistemic event extractor. Return one JSON object.
It must contain exactly these six array fields: requirements, claims, assumptions, evidence,
external_observations, and blind_spots. Requirements need id and description; claims and
assumptions need id and statement; evidence needs id and content; external observations need id
and observation; blind spots are strings. Represent only entities explicit in the supplied event.
Give every entity a deterministic stable ID derived from its source event ID and local index. Do
not infer semantic relations and do not use benchmark answer keys or keyword fields."""

VERIFIER_SYSTEM_PROMPT = """You are a frozen semantic relation verifier. Return one JSON object
with exactly this schema: {"label":"unknown","probabilities":{"entail":0.0,
"contradict":0.0,"unknown":1.0}}. Label must be entail, contradict, or unknown. Probabilities
must be numeric, use those three named object keys, and sum to 1. Judge meaning in context;
surface word overlap is not evidence of entailment. Do not inspect benchmark answer keys,
keyword fields, embedding similarity, or reward."""


def canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GlobalEpistemicState:
    trajectory_id: str
    sequence: int
    task_id: str
    requirements: Tuple[Mapping[str, Any], ...] = ()
    claims: Tuple[Mapping[str, Any], ...] = ()
    assumptions: Tuple[Mapping[str, Any], ...] = ()
    evidence: Tuple[Mapping[str, Any], ...] = ()
    external_observations: Tuple[Mapping[str, Any], ...] = ()
    semantic_relations: Tuple[Mapping[str, Any], ...] = ()
    blind_spots: Tuple[Mapping[str, Any], ...] = ()
    dependencies: Tuple[Mapping[str, Any], ...] = ()
    nodes: Tuple[Mapping[str, Any], ...] = ()
    dag_edges: Tuple[Mapping[str, Any], ...] = ()
    active_subtree: Tuple[str, ...] = ()
    runtime_events: Tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    environment_revision: str = ""
    previous_fingerprint: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": PROCESS_STATE_SCHEMA_VERSION,
            "trajectory_id": self.trajectory_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
            "requirements": list(self.requirements),
            "claims": list(self.claims),
            "assumptions": list(self.assumptions),
            "evidence": list(self.evidence),
            "external_observations": list(self.external_observations),
            "semantic_relations": list(self.semantic_relations),
            "blind_spots": list(self.blind_spots),
            "dependencies": list(self.dependencies),
            "nodes": list(self.nodes),
            "dag_edges": list(self.dag_edges),
            "active_subtree": list(self.active_subtree),
            "runtime_events": list(self.runtime_events),
            "usage": dict(self.usage),
            "environment_revision": self.environment_revision,
            "previous_fingerprint": self.previous_fingerprint,
        }
        value["fingerprint"] = canonical_fingerprint(value)
        return value


class SemanticJSONClient(Protocol):
    def complete_json(self, prompt_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class FrozenDeepSeekSemanticClient:
    """Two prompt channels over one frozen DeepSeek endpoint with full cache provenance."""

    def __init__(self, client: Any, cache_path: Path, model_revision: str,
                 max_tokens: int = 4096, max_attempts: int = 3) -> None:
        self.client = client
        self.cache_path = cache_path
        self.model_revision = model_revision
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.cache: Dict[str, Mapping[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line:
                    value = json.loads(line)
                    self.cache[str(value["cache_key"])] = value["response"]

    def complete_json(self, prompt_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if prompt_name not in ("epistemic_extractor_v1", "semantic_verifier_v1"):
            raise ValueError(f"unknown semantic prompt {prompt_name}")
        system = EXTRACTOR_SYSTEM_PROMPT if prompt_name == "epistemic_extractor_v1" \
            else VERIFIER_SYSTEM_PROMPT
        request = {"prompt": prompt_name, "model": self.client.model,
                   "revision": self.model_revision, "payload": payload}
        cache_key = hashlib.sha256(json.dumps(
            request, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                     sort_keys=True)},
        ]
        response: Mapping[str, Any] | None = None
        last_error: Exception | None = None
        prior_content: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            completion = None
            try:
                attempt_messages = list(messages)
                if prior_content is not None:
                    attempt_messages.extend([
                        {"role": "assistant", "content": prior_content},
                        {"role": "user", "content":
                         "The preceding response failed the required JSON schema. Return only "
                         "a corrected object in the exact schema from the system message."},
                    ])
                completion = self.client.complete(
                    attempt_messages, max_tokens=self.max_tokens, temperature=0.0,
                    metadata={"semantic_operation": prompt_name, "cache_key": cache_key,
                              "model_revision": self.model_revision,
                              "semantic_attempt": attempt},
                    json_mode=True, thinking="disabled",
                )
                parsed = _parse_json_object(completion.content)
                _validate_semantic_response(prompt_name, parsed)
                response = parsed
                break
            except Exception as error:
                last_error = error
                prior_content = getattr(completion, "content", None)
                self._record_failure(cache_key, request, prompt_name, attempt,
                                     prior_content, error)
        if response is None:
            raise ValueError(
                f"semantic DeepSeek response failed schema validation after "
                f"{self.max_attempts} attempts: {last_error}"
            ) from last_error
        enriched = dict(response)
        enriched["_provenance"] = {
            "provider": "deepseek", "model": self.client.model,
            "model_revision": self.model_revision, "cache_key": cache_key,
            "temperature": 0.0, "prompt_name": prompt_name,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cache_key": cache_key, "request": request,
                                     "response": enriched}, sort_keys=True,
                                    ensure_ascii=False) + "\n")
        self.cache[cache_key] = enriched
        return enriched

    def _record_failure(self, cache_key: str, request: Mapping[str, Any], prompt_name: str,
                        attempt: int, content: str | None, error: Exception) -> None:
        path = self.cache_path.parent / "semantic-failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "cache_key": cache_key, "request": request, "prompt_name": prompt_name,
                "attempt": attempt, "response_content": content,
                "error_type": type(error).__name__, "error": str(error),
            }, sort_keys=True, ensure_ascii=False) + "\n")


def _parse_json_object(content: str) -> Mapping[str, Any]:
    text = content.strip()
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        body = text[3:-3].strip()
        if body.lower().startswith("json"):
            body = body[4:].strip()
        candidates.append(body)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("semantic DeepSeek response must contain one JSON object")


def _validate_semantic_response(prompt_name: str, response: Mapping[str, Any]) -> None:
    if prompt_name == "semantic_verifier_v1":
        probabilities = response.get("probabilities")
        if response.get("label") not in SEMANTIC_LABELS or not isinstance(probabilities, Mapping) \
                or set(probabilities) != set(SEMANTIC_LABELS):
            raise ValueError("semantic verifier response does not match its schema")
        if any(not isinstance(probabilities[label], (int, float))
               for label in SEMANTIC_LABELS):
            raise ValueError("semantic verifier probabilities must be numeric")
        values = [float(probabilities[label]) for label in SEMANTIC_LABELS]
        if any(value < 0.0 or value > 1.0 for value in values) \
                or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("semantic verifier probabilities must be in [0,1] and sum to 1")
        return
    fields = {
        "requirements": ("id", "description"),
        "claims": ("id", "statement"),
        "assumptions": ("id", "statement"),
        "evidence": ("id", "content"),
        "external_observations": ("id", "observation"),
    }
    if set(fields).union({"blind_spots"}) - set(response):
        raise ValueError("semantic extractor response is missing required arrays")
    for field_name, required in fields.items():
        values = response.get(field_name)
        if not isinstance(values, list) or any(
            not isinstance(value, Mapping) or any(value.get(key) is None for key in required)
            for value in values
        ):
            raise ValueError(f"semantic extractor field {field_name} does not match its schema")
    if not isinstance(response.get("blind_spots"), list) or any(
        not isinstance(value, str) for value in response["blind_spots"]
    ):
        raise ValueError("semantic extractor blind_spots does not match its schema")


class SemanticStateBuilder:
    """LLM-labelled semantics with embedding used only for candidate recall."""

    def __init__(self, client: SemanticJSONClient, encoder: Any, audit_path: Path, top_k: int = 8):
        self.client = client
        self.encoder = encoder
        self.audit_path = audit_path
        self.top_k = top_k

    def extract(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self.client.complete_json("epistemic_extractor_v1", {"event": dict(event)})
        self._audit("extractor", event, response, [])
        return response

    def verify_candidates(
        self, left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        candidates = embedding_candidate_pairs(left, right, self.encoder, self.top_k)
        relations: List[Mapping[str, Any]] = []
        for left_index, right_index, score in candidates:
            request = {"left": left[left_index], "right": right[right_index]}
            response = self.client.complete_json("semantic_verifier_v1", request)
            probabilities = dict(response.get("probabilities", {}))
            if set(probabilities) != set(SEMANTIC_LABELS):
                raise ValueError("semantic verifier must return all three probabilities")
            label = str(response.get("label"))
            if label not in SEMANTIC_LABELS:
                raise ValueError("semantic verifier returned an invalid label")
            relation = {
                "left_id": left[left_index]["id"], "right_id": right[right_index]["id"],
                "label": label, "probabilities": probabilities,
                "candidate_source": "minilm_top_k", "candidate_similarity": score,
                "provenance": response.get("_provenance", {}),
            }
            relations.append(relation)
            self._audit("verifier", request, response, [relation])
        return relations

    def _audit(self, operation: str, request: Mapping[str, Any], response: Mapping[str, Any],
               candidates: Sequence[Mapping[str, Any]]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"operation": operation, "request": request, "response": response,
                  "candidates": list(candidates)}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def embedding_candidate_pairs(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], encoder: Any, top_k: int = 8
) -> List[Tuple[int, int, float]]:
    """Return at most top_k recall pairs for one entity type, never semantic labels."""
    if not left or not right:
        return []
    import torch

    left_vectors = encoder.encode([str(value["statement"]) for value in left])
    right_vectors = encoder.encode([str(value["statement"]) for value in right])
    similarities = torch.as_tensor(left_vectors) @ torch.as_tensor(right_vectors).T
    count = min(top_k, similarities.numel())
    values, indices = torch.topk(similarities.reshape(-1), count)
    right_count = len(right)
    return [(int(index) // right_count, int(index) % right_count, float(value))
            for value, index in zip(values, indices)]


def append_state(path: Path, state: GlobalEpistemicState) -> str:
    value = state.to_dict()
    if path.exists():
        last = next(reversed([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]))
        if state.sequence != int(last["sequence"]) + 1:
            raise ValueError("process state sequence must be append-only")
        if state.previous_fingerprint != last["fingerprint"]:
            raise ValueError("process state fingerprint chain is broken")
    elif state.sequence != 0 or state.previous_fingerprint is not None:
        raise ValueError("the first process state must be M_0")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    return str(value["fingerprint"])
