from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .model import FrozenTextEncoder
from .process_state import FrozenDeepSeekSemanticClient, SemanticStateBuilder
from .providers import DeepSeekClient


class AccountingLedger:
    """Usage accounting without a formal-run token ceiling."""

    def __init__(self) -> None:
        self.used = 0
        self.reserved = 0

    def reserve(self, amount: int) -> None:
        self.reserved += amount

    def settle(self, reserved: int, actual: int) -> None:
        self.reserved -= reserved
        self.used += actual

    def release(self, reserved: int) -> None:
        self.reserved -= reserved

    def snapshot(self) -> Dict[str, Any]:
        return {"used": self.used, "reserved": self.reserved, "limit": None,
                "remaining": None, "exhausted": False}


class SemanticServer:
    def __init__(self) -> None:
        root = Path(os.environ.get(
            "ROY_LHTB_SEMANTIC_ROOT", "research/output/lhtb/semantic"
        )) / f"process-{os.getpid()}"
        ledger = AccountingLedger()
        deepseek = DeepSeekClient(
            ledger, model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            timeout=float(os.environ.get("ROY_LHTB_SEMANTIC_TIMEOUT", "600")),
            event_log=root / "llm-events.jsonl",
        )
        client = FrozenDeepSeekSemanticClient(
            deepseek, root / "semantic-cache.jsonl",
            os.environ.get("DEEPSEEK_MODEL_REVISION", "deepseek-v4-flash-frozen"),
        )
        self.builder = SemanticStateBuilder(
            client, FrozenTextEncoder(device="cpu", local_only=True),
            root / "semantic-audit.jsonl", top_k=8,
        )

    def process(self, event: Mapping[str, Any], existing_state: Mapping[str, Any]) -> Dict[str, Any]:
        extracted = dict(self.builder.extract(event))
        claims = [_claim(value, str(event.get("nodeId") or "root"))
                  for value in _objects(extracted.get("claims"))]
        extracted["claims"] = claims
        relations = []
        for entity_type, text_field in (
            ("requirements", "description"), ("claims", "statement"),
            ("assumptions", "statement"), ("evidence", "content"),
            ("external_observations", "observation"),
        ):
            left = _semantic_entities(_objects(extracted.get(entity_type)), text_field)
            right = _semantic_entities(_objects(existing_state.get(entity_type)), text_field)
            if left and right:
                relations.extend({**relation, "entity_type": entity_type}
                                 for relation in self.builder.verify_candidates(left, right))
        return {
            "event_id": event.get("id"),
            "requirements": _objects(extracted.get("requirements")), "claims": claims,
            "assumptions": _objects(extracted.get("assumptions")),
            "evidence": _objects(extracted.get("evidence")),
            "external_observations": _objects(extracted.get("external_observations")),
            "blind_spots": [str(value) for value in extracted.get("blind_spots", [])],
            "relations": relations, "provenance": extracted.get("_provenance"),
        }


def _objects(value: Any) -> List[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _claim(value: Mapping[str, Any], origin: str) -> Dict[str, Any]:
    return {"id": str(value["id"]), "statement": str(value["statement"]),
            "status": str(value.get("status", "tentative")),
            "originNodeId": str(value.get("originNodeId", origin))}


def _semantic_entities(values: List[Mapping[str, Any]], text_field: str) -> List[Mapping[str, Any]]:
    return [{"id": str(value["id"]), "statement": str(value[text_field])}
            for value in values
            if value.get("id") is not None and value.get(text_field) is not None]


def main() -> None:
    server = SemanticServer()
    for line in sys.stdin:
        request: Dict[str, Any] = {}
        try:
            request = json.loads(line)
            result = server.process(request["event"], dict(request.get("existing_state", {})))
            response = {"id": request["id"], "result": result}
        except Exception as error:
            response = {"id": request.get("id", "unknown"), "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
