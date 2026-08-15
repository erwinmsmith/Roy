from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from .model import FrozenTextEncoder, StructuralPolicyNetwork, graph_tensors
from .training import ACTION_INDEX, resource_tensor


class PolicyServer:
    def __init__(self) -> None:
        checkpoint = os.environ.get("ROY_STRUCTURAL_MODEL")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.encoder = None
        if checkpoint:
            self.model, _ = StructuralPolicyNetwork.load_checkpoint(Path(checkpoint), map_location=str(self.device))
            self.model.to(self.device).eval()
            self.encoder = FrozenTextEncoder(device=str(self.device), local_only=True)

    def decide(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        legal = list(checkpoint.get("legalActions", []))
        if not legal:
            raise ValueError("checkpoint contains no legal actions")
        if self.model is None or self.encoder is None:
            nodes = checkpoint.get("eventGraph", {}).get("nodes", [])
            has_gap = any(
                node.get("status") == "failed" or node.get("kind") == "dependency"
                for node in nodes
            )
            action = "BRANCH" if has_gap and "BRANCH" in legal else "RETURN" if "RETURN" in legal else "CONTINUE"
            return {"action": action, "rationale": "Deterministic sidecar fallback policy.", "policyVersion": "fallback-v1"}
        tensors = tuple(
            tensor.to(self.device) for tensor in graph_tensors(checkpoint["eventGraph"], self.encoder)
        )
        mask = torch.tensor([action in legal for action in ACTION_INDEX], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(
                *tensors,
                resource_tensor(checkpoint.get("resources", {}), self.device),
                mask,
            )
            probabilities = torch.softmax(logits, dim=-1)
            index = int(torch.argmax(probabilities).item())
        action = list(ACTION_INDEX)[index]
        return {
            "action": action,
            "rationale": "Learned relational event-graph policy.",
            "policyVersion": "cs-grpo-v1",
            "confidence": float(probabilities[index].cpu()),
        }


def main() -> None:
    server = PolicyServer()
    for line in sys.stdin:
        request: Dict[str, Any] = {}
        try:
            request = json.loads(line)
            response = {"id": request["id"], "decision": server.decide(request["checkpoint"])}
        except Exception as error:
            response = {"id": request.get("id", "unknown") if isinstance(request, dict) else "unknown", "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
