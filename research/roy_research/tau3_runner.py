from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .analysis import paired_bootstrap_interval
from .io import read_jsonl, write_jsonl
from .organization import (
    ORGANIZATION_GROUP_SIZE,
    ExplorationEnvelope,
    RuntimeBudget,
    training_envelope,
)
from .organization_replay import OrganizationGRPOTrainer
from .tau3 import TAU3_COMMIT
from .tau3_agent import (
    Tau3OrganizationAgentConfig,
    organization_trajectory_from_state,
    register_tau3_organization_agent,
)


def train_tau3_on_policy(
    manifest_path: Path,
    trajectories_path: Path,
    model_path: Path,
    agent_llm: str,
    user_llm: str,
    limit: int | None = None,
    max_steps: int = 100,
    max_tokens: int = 50000,
    temperature: float = 0.0,
    epochs: int = 4,
    runtime_budget: RuntimeBudget = RuntimeBudget(),
    seed: int = 20260818,
    resume: bool = False,
) -> Dict[str, Any]:
    """Collect fresh masked-policy groups and update immediately, with no teacher."""

    if resume and trajectories_path.exists() != model_path.exists():
        raise ValueError("resume requires both the trajectory log and model checkpoint")
    if trajectories_path.exists() and not resume:
        raise ValueError("tau3 trajectory output already exists; use --resume or a new path")
    if model_path.exists() and not resume:
        raise ValueError("tau3 model output already exists; use --resume or a new path")
    tasks = [value for value in read_jsonl(manifest_path) if value.get("split") == "train"]
    if limit is not None:
        tasks = tasks[:limit]
    trainer = OrganizationGRPOTrainer(model_path, seed=seed, resume=resume)
    if not model_path.exists():
        trainer.save()
    existing = list(read_jsonl(trajectories_path)) if resume and trajectories_path.exists() else []
    by_group: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for value in existing:
        by_group[str(value["group_id"])].append(value)

    collected = 0
    for epoch in range(epochs):
        epoch_tasks = list(tasks)
        random.Random(seed + epoch).shuffle(epoch_tasks)
        envelope = training_envelope(epoch, epochs)
        for task_index, task in enumerate(epoch_tasks):
            _validate_manifest_record(task, "train")
            group_id = f"{task['domain']}:{task['task_id']}:{epoch}"
            records = by_group[group_id]
            present = {int(value["rollout_index"]) for value in records}
            environment_seed = _stable_group_seed(seed, epoch, task)
            for rollout_index in range(ORGANIZATION_GROUP_SIZE):
                if rollout_index in present:
                    continue
                name = f"roy_train_{epoch}_{task_index}_{rollout_index}_{uuid.uuid4().hex[:8]}"
                organization_seed = environment_seed + rollout_index + 1
                agent_config = Tau3OrganizationAgentConfig(
                    seed=organization_seed,
                    policy=trainer.model,
                    encoder=trainer.encoder,
                    exploration_envelope=envelope,
                    runtime_budget=runtime_budget,
                )
                register_tau3_organization_agent(agent_config, name=name)
                simulation, state = _run_episode(
                    task,
                    name,
                    agent_llm,
                    user_llm,
                    max_steps,
                    max_tokens,
                    temperature,
                    environment_seed,
                )
                trajectory = organization_trajectory_from_state(
                    state,
                    _reward(simulation),
                    str(task["domain"]),
                    "train",
                    envelope,
                    group_id,
                    epoch,
                    rollout_index,
                    environment_seed,
                    organization_seed,
                    runtime_budget,
                    _resource_summary(simulation, state),
                    _simulation_record(simulation),
                )
                write_jsonl(
                    trajectories_path,
                    [trajectory],
                    append=trajectories_path.exists(),
                )
                records.append(trajectory)
                collected += 1
                print(json.dumps({
                    "event": "trajectory_saved",
                    "group_id": group_id,
                    "rollout_index": rollout_index,
                    "trajectory_id": trajectory["id"],
                    "terminal_utility": trajectory["terminal_utility"],
                    "termination_type": trajectory["termination_type"],
                    "nodes": trajectory["realized_resources"]["nodes"],
                    "maximum_depth": trajectory["realized_resources"]["maximum_depth"],
                    "path": str(trajectories_path),
                }, sort_keys=True), flush=True)
            if len(records) != ORGANIZATION_GROUP_SIZE:
                raise ValueError(f"incomplete organization group after collection: {group_id}")
            if group_id not in trainer.updated_group_ids:
                trainer.update_group(records)
                print(json.dumps({
                    "event": "group_updated",
                    "group_id": group_id,
                    "optimizer_steps": trainer.optimizer_steps,
                    "model": str(model_path),
                }, sort_keys=True), flush=True)
    all_records = [value for values in by_group.values() for value in values]
    return {
        **trainer.metadata(),
        "epochs": epochs,
        "new_trajectories": collected,
        "truncation_rate": sum(bool(value.get("truncated")) for value in all_records)
        / max(1, len(all_records)),
        "model": str(model_path),
    }


def evaluate_tau3_against_direct(
    manifest_path: Path,
    model_path: Path,
    output_path: Path,
    agent_llm: str,
    user_llm: str,
    split: str = "test",
    limit: int | None = None,
    max_steps: int = 100,
    max_tokens: int = 50000,
    temperature: float = 0.0,
    runtime_budget: RuntimeBudget = RuntimeBudget(),
    seed: int = 20260818,
) -> Dict[str, Any]:
    """Paired tau3 evaluation: single-agent direct versus one learned organization."""

    if split not in {"validation", "test", "heldout"}:
        raise ValueError("tau3 evaluation split must be validation, test, or heldout")
    if output_path.exists():
        raise ValueError("tau3 evaluation output already exists; choose a new path")
    tasks = [value for value in read_jsonl(manifest_path) if value.get("split") == split]
    if limit is not None:
        tasks = tasks[:limit]
    trainer = OrganizationGRPOTrainer(model_path, seed=seed, resume=True)
    evaluation_envelope = ExplorationEnvelope("evaluation", 0, 12, 0, 5, "expansive")
    rows: list[Dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        _validate_manifest_record(task, split)
        direct, _state = _run_episode(
            task,
            "llm_agent",
            agent_llm,
            user_llm,
            max_steps,
            max_tokens,
            temperature,
            seed + task_index,
        )
        name = f"roy_eval_{task_index}_{uuid.uuid4().hex[:8]}"
        config = Tau3OrganizationAgentConfig(
            seed=seed + task_index,
            policy=trainer.model,
            encoder=trainer.encoder,
            exploration_envelope=evaluation_envelope,
            runtime_budget=runtime_budget,
        )
        register_tau3_organization_agent(config, name=name)
        learned, state = _run_episode(
            task,
            name,
            agent_llm,
            user_llm,
            max_steps,
            max_tokens,
            temperature,
            seed + task_index,
        )
        rows.extend([
            _evaluation_row(task, "single_agent_direct", direct, None),
            _evaluation_row(task, "learned_information_realization", learned, state),
        ])
        write_jsonl(output_path, rows[-2:], append=output_path.exists() or len(rows) > 2)
    return _paired_summary(rows, split)


def _run_episode(
    task_record: Mapping[str, Any],
    agent_name: str,
    agent_llm: str,
    user_llm: str,
    max_steps: int,
    max_tokens: int,
    temperature: float,
    seed: int,
):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import build_text_orchestrator, get_tasks, run_simulation

    domain = str(task_record["domain"])
    retrieval_config = None
    if domain == "banking_knowledge":
        retrieval_config = _register_local_banking_all_tools()
    official_split = str(task_record["official_split"])
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=official_split,
        task_ids=[str(task_record["task_id"])],
    )
    llm_arguments: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    api_base = os.environ.get("DEEPSEEK_BASE_URL")
    if api_base:
        llm_arguments["api_base"] = api_base.rstrip("/")
    config_values: Dict[str, Any] = dict(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=agent_llm,
        llm_args_agent=dict(llm_arguments),
        llm_user=user_llm,
        llm_args_user=dict(llm_arguments),
        max_steps=max_steps,
        max_errors=10,
        seed=seed,
        max_concurrency=1,
    )
    if retrieval_config is not None:
        config_values["retrieval_config"] = retrieval_config
    config = TextRunConfig(**config_values)
    orchestrator = build_text_orchestrator(config, tasks[0], seed=seed)
    simulation = run_simulation(orchestrator)
    return simulation, getattr(orchestrator, "agent_state", None)


def _register_local_banking_all_tools() -> str:
    """Expose banking AllTools without an external embedding credential."""

    from tau2.domains.banking_knowledge.retrieval import (
        RETRIEVAL_VARIANTS,
        all_tools_variant,
    )
    from tau2.knowledge.document_preprocessors.embedding_indexer import (
        EMBEDDER_REGISTRY as DOCUMENT_EMBEDDERS,
    )
    from tau2.knowledge.input_preprocessors.embedding_encoder import (
        EMBEDDER_REGISTRY as QUERY_EMBEDDERS,
    )
    from .model import FrozenTextEncoder, MINILM_MODEL_ID

    class RoyLocalEmbedder:
        def __init__(self, model: str | None = None, **_kwargs: Any) -> None:
            self.model_name = model or MINILM_MODEL_ID
            self.encoder = FrozenTextEncoder(device="cpu", local_only=True)

        def embed(self, texts: list[str]):
            return self.encoder.encode(texts).cpu().numpy()

        def get_name(self) -> str:
            return f"roy-local:{self.model_name}"

    embedder_name = "roy_local_minilm"
    variant_name = "roy_alltools_local"
    DOCUMENT_EMBEDDERS[embedder_name] = RoyLocalEmbedder
    QUERY_EMBEDDERS[embedder_name] = RoyLocalEmbedder
    if variant_name not in RETRIEVAL_VARIANTS:
        RETRIEVAL_VARIANTS[variant_name] = all_tools_variant(
            variant_name,
            embedder_type=embedder_name,
            embedder_model=MINILM_MODEL_ID,
        )
    return variant_name


def _validate_manifest_record(value: Mapping[str, Any], split: str) -> None:
    if value.get("benchmark") != "tau3" or value.get("revision") != TAU3_COMMIT:
        raise ValueError("tau3 runner requires the pinned manifest revision")
    if value.get("split") != split:
        raise ValueError("tau3 manifest split mismatch")


def _stable_group_seed(seed: int, epoch: int, task: Mapping[str, Any]) -> int:
    material = f"{seed}:{epoch}:{task['domain']}:{task['task_id']}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _reward(simulation: Any) -> float:
    reward_info = getattr(simulation, "reward_info", None)
    if reward_info is None or getattr(reward_info, "reward", None) is None:
        raise ValueError("tau3 simulation did not produce terminal reward")
    return float(reward_info.reward)


def _resource_summary(simulation: Any, state: Any) -> Dict[str, Any]:
    payload = simulation.model_dump(mode="json") if hasattr(simulation, "model_dump") else {}
    prompt_tokens, completion_tokens = _token_counts(payload)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "organization_decisions": int(getattr(state, "decision_count", 0)),
        "llm_calls": int(getattr(state, "llm_call_count", 0)),
        "tool_calls": int(getattr(state, "tool_call_count", 0)),
        "nodes": len(getattr(state, "nodes", [])),
        "maximum_depth": max(
            (int(value["depth"]) for value in getattr(state, "nodes", [])), default=0
        ),
    }


def _simulation_record(simulation: Any) -> Dict[str, Any]:
    if not hasattr(simulation, "model_dump"):
        return {"representation": str(simulation)}
    value = simulation.model_dump(mode="json")
    return dict(value) if isinstance(value, Mapping) else {"value": value}


def _token_counts(value: Any) -> tuple[int, int]:
    prompt = 0
    completion = 0
    if isinstance(value, Mapping):
        prompt += int(value.get("prompt_tokens", 0) or 0)
        completion += int(value.get("completion_tokens", 0) or 0)
        for child in value.values():
            child_prompt, child_completion = _token_counts(child)
            prompt += child_prompt
            completion += child_completion
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            child_prompt, child_completion = _token_counts(child)
            prompt += child_prompt
            completion += child_completion
    return prompt, completion


def _evaluation_row(task: Mapping[str, Any], arm: str, simulation: Any, state: Any) -> Dict[str, Any]:
    utility = _reward(simulation)
    row = {
        "benchmark": "tau3",
        "revision": TAU3_COMMIT,
        "domain": str(task["domain"]),
        "task_id": str(task["task_id"]),
        "split": str(task["split"]),
        "arm": arm,
        "utility": utility,
        "success": utility >= 1.0 - 1e-9,
    }
    if state is not None:
        row["organization"] = {
            "nodes": len(state.nodes),
            "maximum_depth": max(int(value["depth"]) for value in state.nodes),
            "dependency_edges": len(state.dependency_edges),
            "communication_edges": len(state.communication_edges),
            "decisions": state.decision_count,
            "terminated": state.termination_type == "policy_stop",
            "truncated": state.termination_type != "policy_stop",
            "termination_type": state.termination_type,
        }
    return row


def _paired_summary(rows: Iterable[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for value in rows:
        key = f"{value['domain']}:{value['task_id']}"
        by_arm[str(value["arm"])][key] = value
    direct = by_arm["single_agent_direct"]
    learned = by_arm["learned_information_realization"]
    keys = sorted(set(direct) & set(learned))
    direct_utility = [float(direct[key]["utility"]) for key in keys]
    learned_utility = [float(learned[key]["utility"]) for key in keys]
    direct_success = [float(direct[key]["success"]) for key in keys]
    learned_success = [float(learned[key]["success"]) for key in keys]
    return {
        "benchmark": "tau3",
        "revision": TAU3_COMMIT,
        "split": split,
        "tasks": len(keys),
        "single_agent_direct": {
            "mean_utility": sum(direct_utility) / max(1, len(keys)),
            "success_rate": sum(direct_success) / max(1, len(keys)),
        },
        "learned_information_realization": {
            "mean_utility": sum(learned_utility) / max(1, len(keys)),
            "success_rate": sum(learned_success) / max(1, len(keys)),
            "truncation_rate": sum(
                bool(learned[key].get("organization", {}).get("truncated")) for key in keys
            ) / max(1, len(keys)),
            "paired_utility_vs_direct": paired_bootstrap_interval(
                learned_utility, direct_utility
            ),
            "paired_success_vs_direct": paired_bootstrap_interval(
                learned_success, direct_success
            ),
        },
    }
