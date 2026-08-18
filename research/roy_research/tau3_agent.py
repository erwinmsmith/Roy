from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .model import FrozenTextEncoder
from .organization import (
    ExplorationEnvelope,
    RuntimeBudget,
    envelope_legal_actions,
)
from .organization_model import InformationRealizationPolicy
from .organization_replay import sample_organization_decision


NODE_PROMPT = """
You are an autonomously derived reasoning node in Roy's recursive information-realization system.
You are not a predefined role. Work only on the local objective caused by a residual requirement.
Distinguish external observations from inference. Do not reveal hidden chain-of-thought.
Return one JSON object with: conclusion, reasoning_summary, claims, evidence,
external_observations, assumptions, uncertainty, conflicts, coverage, blind_spots,
residual_requirements, proposed_children, resolved_parent_gap, information_to_propagate.
Every proposed child must be a strict refinement tied to one residual requirement and have a
verifiable termination condition. Do not create a child merely to increase node count.
Inspect the available tool schemas and domain policy before deciding that information must be
requested from the user. When an available tool can resolve missing task information, emit a
residual requirement whose possible_external_access names that tool. Do not claim that a lookup
is unavailable when its tool is listed. Use proposed_children only for genuinely separable
reasoning, verification, evidence, or planning gaps; tool access itself belongs in ACQUIRE.
""".strip()


@dataclass(frozen=True)
class Tau3OrganizationAgentConfig:
    checkpoint: Path | None = None
    maximum_internal_decisions_per_turn: int = 64
    seed: int = 20260818
    local_encoder_only: bool = True
    policy: Any | None = None
    encoder: Any | None = None
    exploration_envelope: ExplorationEnvelope | None = None
    runtime_budget: RuntimeBudget = RuntimeBudget()
    organization_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.organization_temperature <= 0:
            raise ValueError("organization temperature must be positive")

    def envelope(self) -> ExplorationEnvelope:
        if self.exploration_envelope is not None:
            return self.exploration_envelope
        return ExplorationEnvelope("default", 0, 12, 0, 5, "expansive")


def register_tau3_organization_agent(
    config: Tau3OrganizationAgentConfig,
    name: str = "roy_information_realization",
) -> None:
    """Register Roy in an installed tau3 checkout without patching tau3 itself."""

    from pydantic import BaseModel, ConfigDict, Field
    from tau2.agent.base.llm_config import LLMConfigMixin
    from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
    from tau2.data_model.message import (
        APICompatibleMessage,
        AssistantMessage,
        Message,
        MultiToolMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate

    if registry.get_agent_factory(name) is not None:
        return

    class OrganizationState(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        system_messages: list[Any]
        messages: list[Any]
        task_id: str
        trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        nodes: list[Dict[str, Any]]
        derivation_edges: list[Dict[str, Any]] = Field(default_factory=list)
        dependency_edges: list[Dict[str, Any]] = Field(default_factory=list)
        communication_edges: list[Dict[str, Any]] = Field(default_factory=list)
        observations: list[Dict[str, Any]] = Field(default_factory=list)
        policy_records: list[Dict[str, Any]] = Field(default_factory=list)
        llm_events: list[Dict[str, Any]] = Field(default_factory=list)
        available_tools: list[Dict[str, Any]] = Field(default_factory=list)
        actions: list[Dict[str, Any]] = Field(default_factory=list)
        used_candidates: list[str] = Field(default_factory=list)
        pending_acquisition_node: str | None = None
        final_output: str | None = None
        stopped: bool = False
        decision_count: int = 0
        llm_call_count: int = 0
        tool_call_count: int = 0
        termination_type: str | None = None
        termination_reason: str | None = None
        initial_snapshot_fingerprint: str = ""

    class Tau3OrganizationAgent(LLMConfigMixin, HalfDuplexAgent[OrganizationState]):
        def __init__(self, tools, domain_policy, llm, llm_args=None, task=None):
            super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
            self.task = task
            torch.manual_seed(config.seed)
            self.device = torch.device("cpu")
            self.encoder = config.encoder or FrozenTextEncoder(
                device=str(self.device), local_only=config.local_encoder_only
            )
            if config.policy is not None:
                self.policy = config.policy
            elif config.checkpoint:
                self.policy, _metadata = InformationRealizationPolicy.load_checkpoint(
                    config.checkpoint, map_location=str(self.device)
                )
            else:
                self.policy = InformationRealizationPolicy()
            self.policy.to(self.device).eval()
            self.generator = torch.Generator().manual_seed(config.seed)

        @property
        def system_prompt(self) -> str:
            return (
                "You are the root of Roy's autonomous recursive information-realization system. "
                "Follow the domain policy exactly. Use tools only through valid tool calls.\n"
                f"<policy>\n{self.domain_policy}\n</policy>"
            )

        def get_init_state(self, message_history: list[Message] | None = None) -> OrganizationState:
            history = message_history or []
            if not all(is_valid_agent_history_message(value) for value in history):
                raise ValueError("invalid initial tau3 agent history")
            task_id = str(getattr(self.task, "id", "unknown"))
            state = OrganizationState(
                system_messages=[SystemMessage(role="system", content=self.system_prompt)],
                messages=list(history),
                task_id=task_id,
                available_tools=self._tool_context(),
                nodes=[{
                    "id": "root",
                    "parent_id": None,
                    "depth": 0,
                    "objective": "Resolve the user's tau3 task while following the domain policy.",
                    "triggering_gap_id": None,
                    "status": "ready",
                    "report": None,
                }],
            )
            state.initial_snapshot_fingerprint = _stable_json_hash({
                "task_id": task_id,
                "history": [str(getattr(value, "content", "")) for value in history],
            })
            return state

        def stop(self, message=None, state: OrganizationState | None = None) -> None:
            if state is None:
                return
            state.stopped = True
            if state.termination_type is None:
                state.termination_type = "truncated_environment"
                state.termination_reason = "tau3 orchestrator ended the episode"
            if state.final_output is None:
                for value in reversed(state.messages):
                    if isinstance(value, AssistantMessage) and getattr(value, "content", None):
                        state.final_output = str(value.content)
                        break

        def generate_next_message(self, message, state: OrganizationState):
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
                for value in message.tool_messages:
                    self._ingest_tool_observation(value, state)
            elif message is not None:
                state.messages.append(message)
                if isinstance(message, ToolMessage):
                    self._ingest_tool_observation(message, state)
                elif isinstance(message, UserMessage):
                    root = state.nodes[0]
                    root["objective"] = str(message.content)
                    root["status"] = "ready"

            for _ in range(config.maximum_internal_decisions_per_turn):
                if state.decision_count >= config.runtime_budget.maximum_decisions:
                    return self._truncate_and_finalize(
                        state, generate, AssistantMessage, SystemMessage,
                        "decision_budget_exhausted"
                    )
                if state.llm_call_count >= config.runtime_budget.maximum_llm_calls:
                    return self._truncate_and_finalize(
                        state, generate, AssistantMessage, SystemMessage,
                        "llm_call_budget_exhausted"
                    )
                candidates = self._candidates(state)
                if not any(bool(value.get("legal", True)) for value in candidates):
                    return self._truncate_and_finalize(
                        state, generate, AssistantMessage, SystemMessage,
                        "runtime_action_budget_exhausted"
                    )
                policy_state = self._policy_state(state, candidates)
                candidate, record = sample_organization_decision(
                    self.policy,
                    self.encoder,
                    policy_state,
                    generator=self.generator,
                    device=self.device,
                )
                state.policy_records.append(record)
                state.actions.append({
                    "kind": candidate["kind"],
                    "actor_node_id": record["active_node_id"],
                    "candidate_id": candidate["id"],
                })
                print(json.dumps({
                    "event": "organization_action",
                    "task_id": state.task_id,
                    "decision": state.decision_count + 1,
                    "kind": candidate["kind"],
                    "actor_node_id": record["active_node_id"],
                    "candidate_id": candidate["id"],
                    "tool_name": candidate.get("tool_name"),
                    "nodes": len(state.nodes),
                }, sort_keys=True), flush=True)
                state.used_candidates.append(str(candidate["id"]))
                state.decision_count += 1
                outward = self._apply(candidate, state, generate, AssistantMessage, SystemMessage)
                if outward is not None:
                    state.messages.append(outward)
                    return outward, state

            return self._truncate_and_finalize(
                state, generate, AssistantMessage, SystemMessage,
                "internal_turn_decision_limit"
            )

        def _ingest_tool_observation(self, message, state: OrganizationState) -> None:
            state.tool_call_count += 1
            actor_id = state.pending_acquisition_node or "root"
            state.observations.append({
                "id": f"observation-{len(state.observations)}",
                "node_id": actor_id,
                "source_type": "tool",
                "observation": str(getattr(message, "content", message)),
                "provenance": str(getattr(message, "tool_call_id", "tau3-environment")),
            })
            node = self._node(state, actor_id)
            node["status"] = "ready"
            state.pending_acquisition_node = None

        def _apply(self, candidate, state, generate_fn, assistant_type, system_type):
            kind = str(candidate["kind"])
            actor = self._node(state, str(candidate["actor_node_id"]))
            if kind == "EXECUTE":
                state.llm_call_count += 1
                actor["report"] = self._generate_report(actor, state, generate_fn, system_type)
                actor["status"] = "reported"
                return None
            if kind == "DERIVE":
                proposal = dict(candidate["proposal"])
                child_id = f"node-{len(state.nodes)}"
                state.nodes.append({
                    "id": child_id,
                    "parent_id": actor["id"],
                    "depth": int(actor["depth"]) + 1,
                    "objective": str(proposal["objective"]),
                    "triggering_gap_id": str(proposal["triggering_gap_id"]),
                    "status": "ready",
                    "report": None,
                })
                state.derivation_edges.append({"from": actor["id"], "to": child_id})
                known_nodes = {value["id"] for value in state.nodes}
                for producer_id in proposal.get("depends_on_node_ids", []):
                    if producer_id in known_nodes and producer_id != child_id:
                        producer = self._node(state, str(producer_id))
                        state.dependency_edges.append({
                            "from": str(producer_id),
                            "to": child_id,
                            "artifact_id": f"report:{producer_id}",
                            "resolved": producer.get("status") == "returned",
                        })
                return None
            if kind == "CONNECT":
                state.communication_edges.append({
                    "from": candidate["from"], "to": candidate["to"], "required": False
                })
                return None
            if kind == "RETURN":
                actor["status"] = "returned"
                if actor.get("parent_id"):
                    self._node(state, actor["parent_id"])["status"] = "ready"
                return None
            if kind == "PRUNE":
                self._node(state, str(candidate["target_node_id"]))["status"] = "pruned"
                return None
            if kind == "ACQUIRE":
                state.pending_acquisition_node = actor["id"]
                actor["status"] = "waiting"
                prompt = system_type(
                    role="system",
                    content=(
                        "The organization selected ACQUIRE for this residual requirement: "
                        f"{candidate['requirement']}. Call exactly the selected tau3 tool "
                        f"{candidate['tool_name']} once with policy-compliant arguments."
                    ),
                )
                selected_tools = [
                    tool for tool in self.tools
                    if self._tool_name(tool) == str(candidate["tool_name"])
                ]
                if len(selected_tools) != 1:
                    raise ValueError(
                        f"selected tau3 tool is unavailable or ambiguous: {candidate['tool_name']}"
                    )
                state.llm_call_count += 1
                outward = generate_fn(
                    model=self.llm,
                    tools=selected_tools,
                    messages=state.system_messages + state.messages + [prompt],
                    call_name="roy_acquire",
                    **self.llm_args,
                )
                tool_calls = list(getattr(outward, "tool_calls", None) or [])
                if len(tool_calls) > 1:
                    outward = outward.model_copy(update={"tool_calls": tool_calls[:1]})
                self._record_llm_event(state, "roy_acquire", actor["id"], outward)
                return outward
            if kind == "STOP":
                outward = self._final_message(state, generate_fn, system_type)
                state.final_output = str(outward.content or "")
                state.stopped = True
                state.termination_type = "policy_stop"
                state.termination_reason = "organization policy selected STOP"
                return outward
            raise ValueError(f"unsupported organization action: {kind}")

        def _generate_report(self, node, state, generate_fn, system_type) -> Dict[str, Any]:
            returned = [
                value for value in state.nodes
                if value.get("parent_id") == node["id"] and value.get("status") == "returned"
            ]
            context = {
                "local_objective": node["objective"],
                "triggering_gap_id": node.get("triggering_gap_id"),
                "external_observations": [
                    value for value in state.observations if value.get("node_id") == node["id"]
                ],
                "returned_child_reports": [value.get("report") for value in returned],
                "conversation": [str(getattr(value, "content", "")) for value in state.messages[-8:]],
                "domain_policy": self.domain_policy,
                "available_tools": state.available_tools,
            }
            response = generate_fn(
                model=self.llm,
                tools=[],
                messages=[
                    system_type(role="system", content=NODE_PROMPT),
                    system_type(role="system", content=json.dumps(context, ensure_ascii=False)),
                ],
                call_name="roy_epistemic_report",
                **self.llm_args,
            )
            self._record_llm_event(state, "roy_epistemic_report", node["id"], response)
            return _normalize_report(_parse_json_object(str(response.content or "{}")), node)

        def _tool_context(self) -> list[Dict[str, Any]]:
            result: list[Dict[str, Any]] = []
            for tool in self.tools:
                if hasattr(tool, "model_dump"):
                    value = tool.model_dump(mode="json", exclude_none=True)
                elif isinstance(tool, Mapping):
                    value = dict(tool)
                else:
                    value = {"representation": str(tool)}
                record = dict(value) if isinstance(value, Mapping) else {"value": value}
                record["name"] = self._tool_name(tool)
                result.append(record)
            return result

        @staticmethod
        def _tool_name(tool) -> str:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if name:
                return str(name)
            if isinstance(tool, Mapping) and tool.get("name"):
                return str(tool["name"])
            return type(tool).__name__

        def _final_message(self, state, generate_fn, system_type):
            reports = [value.get("report") for value in state.nodes if value.get("report")]
            synthesis = system_type(
                role="system",
                content=(
                    "Produce the final user-facing answer now. Use the epistemic reports below, "
                    "follow the domain policy, do not mention internal agents, and do not call a tool.\n"
                    + json.dumps(reports, ensure_ascii=False)
                ),
            )
            state.llm_call_count += 1
            response = generate_fn(
                model=self.llm,
                tools=[],
                messages=state.system_messages + state.messages + [synthesis],
                call_name="roy_final_answer",
                **self.llm_args,
            )
            self._record_llm_event(state, "roy_final_answer", "root", response)
            return response

        @staticmethod
        def _record_llm_event(state, call_name: str, node_id: str, response) -> None:
            if hasattr(response, "model_dump"):
                message = response.model_dump(mode="json", exclude_none=True)
            else:
                message = {"representation": str(response)}
            state.llm_events.append({
                "call_name": call_name,
                "node_id": node_id,
                "message": message,
            })

        def _truncate_and_finalize(
            self, state, generate_fn, assistant_type, system_type, reason: str
        ):
            if state.llm_call_count < config.runtime_budget.maximum_llm_calls:
                outward = self._final_message(state, generate_fn, system_type)
            else:
                reports = [
                    str(value.get("report", {}).get("conclusion", ""))
                    for value in state.nodes if value.get("report")
                ]
                outward = assistant_type(
                    role="assistant",
                    content="\n".join(value for value in reports if value),
                )
            state.final_output = str(outward.content or "")
            state.stopped = True
            state.termination_type = "truncated_resource"
            state.termination_reason = reason
            state.messages.append(outward)
            return outward, state

        def _candidates(self, state: OrganizationState) -> list[Dict[str, Any]]:
            result: list[Dict[str, Any]] = []
            used = set(state.used_candidates)
            for node in state.nodes:
                if node["status"] in {"pruned", "returned", "waiting"}:
                    continue
                actor = str(node["id"])
                if node["status"] == "ready":
                    result.append(_candidate(f"execute:{actor}:{state.decision_count}", "EXECUTE", actor, "Update the node epistemic report", 1.0))
                    continue
                report = node.get("report") or {}
                for index, proposal in enumerate(report.get("proposed_children", [])):
                    identifier = f"derive:{actor}:{proposal.get('triggering_gap_id')}:{index}"
                    if identifier not in used:
                        result.append({
                            **_candidate(identifier, "DERIVE", actor, str(proposal.get("objective")), 2.0),
                            "proposal": proposal,
                            "resulting_depth": int(node["depth"]) + 1,
                            "resolves_gap": True,
                            "depth_delta": 1,
                        })
                residuals = list(report.get("residual_requirements", []))
                external_residuals = [
                    value for value in residuals
                    if isinstance(value, Mapping) and value.get("possible_external_access")
                ]
                for tool in state.available_tools:
                    tool_name = str(tool["name"])
                    identifier = (
                        f"acquire-tool:{actor}:{tool_name}:{state.decision_count}"
                    )
                    result.append({
                        **_candidate(
                            identifier,
                            "ACQUIRE",
                            actor,
                            f"Use tau3 tool {tool_name}: {json.dumps(tool, ensure_ascii=False)}",
                            3.0,
                        ),
                        "requirement": {
                            "residual_requirements": external_residuals,
                            "selected_tool": tool_name,
                        },
                        "tool_name": tool_name,
                        "external_access": True,
                        "resolves_gap": bool(external_residuals),
                    })
                if actor != "root":
                    result.append(_candidate(f"return:{actor}", "RETURN", actor, "Return the epistemic report to the parent", 0.2))
                elif not any(value["status"] in {"ready", "reported", "waiting"} for value in state.nodes[1:]):
                    result.append(_candidate(f"stop:{state.decision_count}", "STOP", actor, "Synthesize the terminal answer", 1.0))

            returned = [value for value in state.nodes if value["status"] == "returned"]
            for node in returned:
                identifier = f"prune:{node['id']}"
                if identifier not in used:
                    result.append({
                        **_candidate(identifier, "PRUNE", "root", "Prune a returned redundant branch", 0.1),
                        "target_node_id": node["id"],
                    })
            active = [value for value in state.nodes if value["status"] not in {"pruned"}]
            existing = {(value["from"], value["to"]) for value in state.communication_edges}
            for source in active:
                for target in active:
                    pair = (source["id"], target["id"])
                    identifier = f"connect:{pair[0]}:{pair[1]}"
                    if pair[0] != pair[1] and pair not in existing and identifier not in used:
                        result.append({
                            **_candidate(identifier, "CONNECT", str(source["id"]), "Share relevant epistemic state", 0.2),
                            "from": pair[0], "to": pair[1],
                        })
            if not result:
                result.append(_candidate("stop:fallback", "STOP", "root", "Synthesize the terminal answer", 1.0))
            for candidate in result:
                candidate["legal"] = self._runtime_action_legal(candidate, state)
            return result

        def _runtime_action_legal(self, candidate, state) -> bool:
            budget = config.runtime_budget
            kind = str(candidate["kind"])
            llm_remaining = budget.maximum_llm_calls - state.llm_call_count
            if kind in {"EXECUTE", "ACQUIRE"} and llm_remaining <= 1:
                return False
            if kind == "STOP" and llm_remaining < 1:
                return False
            if kind == "ACQUIRE" and state.tool_call_count >= budget.maximum_tool_calls:
                return False
            if kind == "DERIVE":
                return (
                    len(state.nodes) < min(
                        budget.maximum_nodes, config.envelope().maximum_nodes
                    )
                    and int(candidate.get("resulting_depth", 0)) <= min(
                        budget.maximum_depth, config.envelope().maximum_depth
                    )
                )
            return state.decision_count < budget.maximum_decisions

        @staticmethod
        def _has_actionable_residual_gap(candidates) -> bool:
            return any(
                bool(value.get("legal", True))
                and value.get("kind") in {"DERIVE", "ACQUIRE"}
                and bool(value.get("resolves_gap", False))
                for value in candidates
            )

        def _policy_state(self, state, candidates):
            nodes = [{
                "id": value["id"],
                "kind": "agent",
                "text": f"{value['objective']} {json.dumps(value.get('report'), ensure_ascii=False)}",
                "status": value["status"],
                "attributes": {"signal": 1.0 if value.get("report") else 0.0},
            } for value in state.nodes]
            edges = [
                {"from": value["from"], "to": value["to"], "kind": "derivation"}
                for value in state.derivation_edges
            ] + [
                {"from": value["from"], "to": value["to"], "kind": "dependency"}
                for value in state.dependency_edges
            ] + [
                {"from": value["from"], "to": value["to"], "kind": "communication"}
                for value in state.communication_edges
            ]
            active_ids = sorted({str(value["actor_node_id"]) for value in candidates})
            unresolved_gap_exists = self._has_actionable_residual_gap(candidates)
            envelope_legal = envelope_legal_actions(
                candidates,
                config.envelope(),
                len(state.nodes),
                max(int(value["depth"]) for value in state.nodes),
                unresolved_gap_exists,
            )
            active_legal = [
                any(
                    legal and str(candidate["actor_node_id"]) == actor_id
                    for candidate, legal in zip(candidates, envelope_legal)
                )
                for actor_id in active_ids
            ]
            resources = self._resource_fractions(state)
            fingerprint_payload = {
                "nodes": nodes,
                "edges": edges,
                "candidate_ids": [str(value["id"]) for value in candidates],
                "candidate_legal": envelope_legal,
                "resources": resources,
                "envelope": config.envelope().to_dict(),
            }
            return {
                "state_fingerprint": _stable_json_hash(fingerprint_payload),
                "event_graph": {"nodes": nodes, "edges": edges},
                "active_node_ids": active_ids,
                "active_node_legal": active_legal,
                "candidates": candidates,
                "resources": resources,
                "node_count": len(state.nodes),
                "maximum_depth_reached": max(int(value["depth"]) for value in state.nodes),
                "envelope": config.envelope().to_dict(),
                "unresolved_gap_exists": unresolved_gap_exists,
                "organization_temperature": config.organization_temperature,
            }

        @staticmethod
        def _resource_fractions(state):
            budget = config.runtime_budget
            maximum_depth = max(int(value["depth"]) for value in state.nodes)
            return {
                "llm_calls_remaining_fraction": max(
                    0.0, (budget.maximum_llm_calls - state.llm_call_count)
                    / budget.maximum_llm_calls
                ),
                "tool_calls_remaining_fraction": max(
                    0.0, (budget.maximum_tool_calls - state.tool_call_count)
                    / max(1, budget.maximum_tool_calls)
                ),
                "nodes_remaining_fraction": max(
                    0.0, (budget.maximum_nodes - len(state.nodes)) / budget.maximum_nodes
                ),
                "depth_remaining_fraction": max(
                    0.0, (budget.maximum_depth - maximum_depth)
                    / max(1, budget.maximum_depth)
                ),
                "decisions_remaining_fraction": max(
                    0.0, (budget.maximum_decisions - state.decision_count)
                    / budget.maximum_decisions
                ),
            }

        @staticmethod
        def _node(state, node_id):
            for value in state.nodes:
                if value["id"] == node_id:
                    return value
            raise ValueError(f"unknown organization node: {node_id}")

    def factory(tools, domain_policy, **kwargs):
        return Tau3OrganizationAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
            task=kwargs.get("task"),
        )

    registry.register_agent_factory(factory, name)


def organization_trajectory_from_state(
    state: Any,
    terminal_utility: float,
    domain: str,
    split: str,
    envelope: ExplorationEnvelope,
    group_id: str,
    epoch: int,
    rollout_index: int,
    environment_seed: int,
    organization_seed: int,
    organization_temperature: float,
    runtime_budget: RuntimeBudget,
    realized_resources: Mapping[str, Any] | None = None,
    benchmark_episode: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    termination_type = str(state.termination_type or "truncated_environment")
    return {
        "schema_version": 2,
        "id": str(state.trajectory_id),
        "group_id": group_id,
        "benchmark": "tau3",
        "domain": domain,
        "task_id": str(state.task_id),
        "split": split,
        "epoch": epoch,
        "rollout_index": rollout_index,
        "environment_seed": environment_seed,
        "organization_seed": organization_seed,
        "organization_temperature": organization_temperature,
        "initial_snapshot_fingerprint": str(state.initial_snapshot_fingerprint),
        "envelope": envelope.to_dict(),
        "runtime_budget": runtime_budget.to_dict(),
        "actions": list(state.actions),
        "policy_records": list(state.policy_records),
        "terminal_utility": float(terminal_utility),
        "realized_resources": dict(realized_resources or {}),
        "benchmark_episode": dict(benchmark_episode or {}),
        "terminal": bool(state.stopped),
        "terminated": termination_type == "policy_stop",
        "truncated": termination_type != "policy_stop",
        "termination_type": termination_type,
        "termination_reason": state.termination_reason,
        "final_output": state.final_output,
        "organization": {
            "nodes": list(state.nodes),
            "derivation_edges": list(state.derivation_edges),
            "dependency_edges": list(state.dependency_edges),
            "communication_edges": list(state.communication_edges),
            "observations": list(state.observations),
            "llm_events": list(state.llm_events),
            "available_tools": list(state.available_tools),
        },
    }


def _candidate(
    identifier: str, kind: str, actor: str, description: str, complexity: float
) -> Dict[str, Any]:
    return {
        "id": identifier,
        "kind": kind,
        "actor_node_id": actor,
        "description": description,
        "scheduler_complexity": complexity,
        "legal": True,
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_report(value: Mapping[str, Any], node: Mapping[str, Any]) -> Dict[str, Any]:
    gaps = _as_list(value.get("residual_requirements"))
    proposals = _as_list(value.get("proposed_children"))
    normalized_gaps = []
    for index, gap in enumerate(gaps):
        if not isinstance(gap, Mapping):
            continue
        normalized_gaps.append({
            **dict(gap),
            "id": str(gap.get("id") or f"gap-{node['id']}-{index}"),
            "description": str(gap.get("description") or gap.get("required_information") or "unresolved requirement"),
            "possible_external_access": _as_list(gap.get("possible_external_access")),
        })
    normalized_proposals = []
    gap_ids = {value["id"] for value in normalized_gaps}
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            continue
        gap_id = str(proposal.get("triggering_gap_id") or proposal.get("triggering_gap") or "")
        if gap_id not in gap_ids:
            continue
        objective = str(proposal.get("objective") or "").strip()
        termination = str(proposal.get("termination_condition") or "").strip()
        if objective and termination:
            normalized_proposals.append({
                **dict(proposal),
                "id": str(proposal.get("id") or f"proposal-{node['id']}-{index}"),
                "triggering_gap_id": gap_id,
                "objective": objective,
                "termination_condition": termination,
                "depends_on_node_ids": [
                    str(identifier) for identifier in _as_list(
                        proposal.get("depends_on_node_ids")
                    )
                ],
            })
    return {
        "conclusion": str(value.get("conclusion") or ""),
        "reasoning_summary": str(value.get("reasoning_summary") or ""),
        "claims": _as_list(value.get("claims")),
        "evidence": _as_list(value.get("evidence")),
        "external_observations": _as_list(value.get("external_observations")),
        "assumptions": _as_list(value.get("assumptions")),
        "uncertainty": _as_mapping(value.get("uncertainty")),
        "conflicts": _as_list(value.get("conflicts")),
        "coverage": _as_mapping(value.get("coverage")),
        "blind_spots": _as_list(value.get("blind_spots")),
        "residual_requirements": normalized_gaps,
        "proposed_children": normalized_proposals,
        "resolved_parent_gap": bool(value.get("resolved_parent_gap", False)),
        "information_to_propagate": _as_list(value.get("information_to_propagate")),
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"summary": value}


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    import hashlib

    material = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
