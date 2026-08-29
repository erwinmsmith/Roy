from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .torch_runtime import configure_torch_runtime

configure_torch_runtime()

import torch

from .model import FrozenTextEncoder
from .organization_model import InformationRealizationPolicy, LHTB_ACTOR_MODEL_REVISION
from .organization_replay import sample_organization_decision


class LHTBPolicyServer:
    def __init__(self) -> None:
        checkpoint = os.environ.get("ROY_LHTB_MODEL")
        if not checkpoint:
            raise RuntimeError("ROY_LHTB_MODEL is required; learned mode has no heuristic fallback")
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        actor_revision = payload.get("metadata", {}).get("actor_model_revision")
        if actor_revision != LHTB_ACTOR_MODEL_REVISION:
            raise ValueError(
                "LHTB actor checkpoint is incompatible: expected "
                f"{LHTB_ACTOR_MODEL_REVISION}, found "
                f"{actor_revision or 'legacy-active-node-routing'}; initialize a fresh checkpoint"
            )
        self.model = InformationRealizationPolicy()
        state = payload.get("actor_state_dict", payload.get("state_dict"))
        if not isinstance(state, dict):
            raise ValueError("checkpoint contains no LHTB actor state")
        self.model.load_state_dict(state)
        self.model.eval()
        self.encoder = FrozenTextEncoder(device="cpu", local_only=True)
        self.generator = torch.Generator(device="cpu")

    def decide(self, policy_state: Dict[str, Any], seed: int) -> Dict[str, Any]:
        self.generator.manual_seed(seed)
        with torch.no_grad():
            candidate, record = sample_organization_decision(
                self.model, self.encoder, policy_state, self.generator, torch.device("cpu")
            )
        return {"candidate_id": candidate["id"], "policy_record": record}

def main() -> None:
    server = LHTBPolicyServer()
    for line in sys.stdin:
        request: Dict[str, Any] = {}
        try:
            request = json.loads(line)
            operation = str(request.get("operation", "select"))
            if operation == "select":
                result = server.decide(request["policy_state"], int(request["seed"]))
            else:
                raise ValueError(
                    f"unsupported policy operation without search/value rollout: {operation}"
                )
            response = {"id": request["id"], **result}
        except Exception as error:
            response = {"id": request.get("id", "unknown"), "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
