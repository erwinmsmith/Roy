from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, Tuple


PROCESS_STATE_SCHEMA_VERSION = 1
SEMANTIC_LABELS = ("entail", "contradict", "unknown")

EXTRACTOR_SYSTEM_PROMPT = """You are a frozen epistemic event extractor. Return JSON only.
Represent only entities explicit in the supplied event: requirements, claims, assumptions,
evidence, external_observations, and blind_spots. Give every entity a deterministic stable ID
derived from its source event ID and local index. Do not infer semantic relations and do not use
benchmark answer keys or keyword fields."""

VERIFIER_SYSTEM_PROMPT = """You are a frozen semantic relation verifier. Return JSON only with
label (entail, contradict, or unknown) and probabilities for all three labels. Judge meaning in
context; surface word overlap is not evidence of entailment. Do not inspect benchmark answer
keys, keyword fields, embedding similarity, or reward."""


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
                 max_tokens: int = 4096) -> None:
        self.client = client
        self.cache_path = cache_path
        self.model_revision = model_revision
        self.max_tokens = max_tokens
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
        completion = self.client.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
            max_tokens=self.max_tokens, temperature=0.0,
            metadata={"semantic_operation": prompt_name, "cache_key": cache_key,
                      "model_revision": self.model_revision},
        )
        response = json.loads(completion.content)
        if not isinstance(response, Mapping):
            raise ValueError("semantic DeepSeek response must be a JSON object")
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
    """Return recall candidates only; similarity never becomes a semantic label."""
    if not left or not right:
        return []
    import torch

    left_vectors = encoder.encode([str(value["statement"]) for value in left])
    right_vectors = encoder.encode([str(value["statement"]) for value in right])
    similarities = torch.as_tensor(left_vectors) @ torch.as_tensor(right_vectors).T
    result: List[Tuple[int, int, float]] = []
    for left_index in range(len(left)):
        count = min(top_k, len(right))
        values, indices = torch.topk(similarities[left_index], count)
        result.extend((left_index, int(index), float(value)) for value, index in zip(values, indices))
    return result


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
