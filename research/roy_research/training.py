from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor

from .grpo import clipped_policy_loss, select_action_log_probs, standardized_advantages
from .io import read_jsonl
from .model import FrozenTextEncoder, StructuralPolicyNetwork, graph_tensors

ACTION_INDEX = {"CONTINUE": 0, "BRANCH": 1, "RETURN": 2}
TRAINING_VARIANTS = ("full_trajectory", "no_event_graph", "node_only", "full")


def resource_tensor(resources: Dict[str, Any], device: torch.device) -> Tensor:
    keys = ("computeTokens", "wallClockMs", "parallelSlots", "communicationEdges", "toolCalls")
    return torch.tensor([
        math.log1p(max(0.0, float(resources.get(key) or 0))) for key in keys
    ], dtype=torch.float32, device=device)


def train_groups(
    groups_path: Path,
    output_path: Path,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    device_name: str | None = None,
    local_encoder_only: bool = True,
    resume: bool = False,
    variant: str = "full",
    seed: int = 20260815,
) -> Dict[str, Any]:
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"unknown training variant: {variant}")
    torch.manual_seed(seed)
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    groups = [group for group in read_jsonl(groups_path) if group.get("task", {}).get("split") == "train"]
    if not groups:
        raise ValueError("no training groups found")
    encoder = FrozenTextEncoder(device=str(device), local_only=local_encoder_only)
    model = StructuralPolicyNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0
    if resume and output_path.exists():
        payload = torch.load(output_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("metadata", {}).get("completed_epochs", 0))
    old_model = StructuralPolicyNetwork().to(device)
    old_model.load_state_dict(model.state_dict())
    old_model.eval()
    losses: List[float] = []

    for epoch in range(start_epoch, epochs):
        for group in groups:
            checkpoint = group["checkpoint"]
            graph = checkpoint["event_graph"] if variant != "no_event_graph" else {
                "parentId": checkpoint["parent_id"], "nodes": [], "edges": [], "observedAt": 0,
            }
            tensors = tuple(tensor.to(device) for tensor in graph_tensors(graph, encoder))
            resources = resource_tensor(checkpoint["resources"], device)
            legal = checkpoint["legal_actions"]
            legal_mask = torch.tensor([action in legal for action in ACTION_INDEX], dtype=torch.bool, device=device)
            new_logits = model(*tensors, resources, legal_mask)
            with torch.no_grad():
                old_logits = old_model(*tensors, resources, legal_mask)
            if variant == "full_trajectory":
                action_names = [result["action"] for result in group["results"]]
                advantage_values = standardized_advantages([
                    float(result["utility"]) for result in group["results"]
                ]).tolist()
            else:
                action_names = list(group["outer_advantages"])
                advantage_values = [
                    float(group["outer_advantages"][name]) for name in action_names
                ]
            actions = torch.tensor([ACTION_INDEX[name] for name in action_names], device=device)
            advantages = torch.tensor(advantage_values, dtype=torch.float32, device=device)
            expanded_new = new_logits.unsqueeze(0).expand(len(action_names), -1)
            expanded_old = old_logits.unsqueeze(0).expand(len(action_names), -1)
            expanded_mask = legal_mask.unsqueeze(0).expand(len(action_names), -1)
            new_log_prob = select_action_log_probs(expanded_new, expanded_mask, actions)
            old_log_prob = select_action_log_probs(expanded_old, expanded_mask, actions)
            outer_loss = clipped_policy_loss(new_log_prob, old_log_prob, advantages)

            _, graph_state = model.encode_graph(*tensors)
            candidates = group.get("branch_specifications", [])
            derive_loss = torch.tensor(0.0, device=device)
            if candidates and variant == "full":
                candidate_embeddings = encoder.encode([candidate["task"] for candidate in candidates]).to(device)
                scores = model.score_derivations(graph_state, candidate_embeddings)
                targets = torch.tensor([
                    float(group["branch_advantages"].get(candidate["id"], 0.0)) for candidate in candidates
                ], dtype=torch.float32, device=device)
                derive_loss = torch.nn.functional.mse_loss(scores, targets)
            loss = outer_loss + 0.2 * derive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        old_model.load_state_dict(model.state_dict())

    metadata = {
        "schema_version": 1,
        "epochs": epochs,
        "completed_epochs": epochs,
        "resumed_from_epoch": start_epoch,
        "groups": len(groups),
        "mean_loss": sum(losses) / max(1, len(losses)),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "encoder_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "variant": variant,
        "seed": seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }, output_path)
    return metadata


def evaluate_groups(
    groups_path: Path,
    model_path: Path,
    split: str = "test",
    device_name: str | None = None,
) -> Dict[str, Any]:
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, metadata = StructuralPolicyNetwork.load_checkpoint(model_path, map_location=str(device))
    model.to(device).eval()
    encoder = FrozenTextEncoder(device=str(device), local_only=True)
    variant = str(metadata.get("variant", "full"))
    count = 0
    correct = 0
    regrets: List[float] = []
    utilities: List[float] = []
    confusion = {actual: {predicted: 0 for predicted in ACTION_INDEX} for actual in ACTION_INDEX}
    task_results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for group in read_jsonl(groups_path):
            task = group.get("task", {})
            if split == "ood":
                if not task.get("ood"):
                    continue
            elif task.get("split") != split:
                continue
            checkpoint = group["checkpoint"]
            graph = checkpoint["event_graph"] if variant != "no_event_graph" else {
                "parentId": checkpoint["parent_id"], "nodes": [], "edges": [], "observedAt": 0,
            }
            tensors = tuple(tensor.to(device) for tensor in graph_tensors(graph, encoder))
            legal = checkpoint["legal_actions"]
            mask = torch.tensor([action in legal for action in ACTION_INDEX], dtype=torch.bool, device=device)
            logits = model(*tensors, resource_tensor(checkpoint["resources"], device), mask)
            selected = list(ACTION_INDEX)[int(torch.argmax(logits).item())]
            values = {key: float(value) for key, value in group["action_values"].items()}
            branch_values = {key: float(value) for key, value in group.get("branch_values", {}).items()}
            oracle_values = dict(values)
            if branch_values:
                oracle_values["BRANCH"] = max(branch_values.values())
            oracle = max(oracle_values, key=oracle_values.get)
            selected_child = None
            selected_utility = values[selected]
            if selected == "BRANCH" and branch_values and variant == "full":
                _, graph_state = model.encode_graph(*tensors)
                candidates = group.get("branch_specifications", [])
                candidate_embeddings = encoder.encode([candidate["task"] for candidate in candidates]).to(device)
                scores = model.score_derivations(graph_state, candidate_embeddings)
                selected_child = candidates[int(torch.argmax(scores).item())]["id"]
                selected_utility = branch_values[selected_child]
            count += 1
            correct += int(selected == oracle)
            confusion[oracle][selected] += 1
            utilities.append(selected_utility)
            regrets.append(oracle_values[oracle] - selected_utility)
            task_results.append({
                "task_id": group["task"]["id"],
                "selected_action": selected,
                "selected_child_specification": selected_child,
                "oracle_action": oracle,
                "utility": selected_utility,
                "oracle_utility": oracle_values[oracle],
                "regret": oracle_values[oracle] - selected_utility,
            })
    return {
        "split": split,
        "count": count,
        "accuracy": correct / max(1, count),
        "mean_utility": sum(utilities) / max(1, count),
        "mean_regret": sum(regrets) / max(1, count),
        "action_confusion_matrix": confusion,
        "task_results": task_results,
        "model_metadata": metadata,
    }
