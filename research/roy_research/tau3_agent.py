from __future__ import annotations

import hashlib
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
Do not propose a child whose only job is to ask the user for information, wait for the next
user message, or repeat an ancestor's interaction gap. Report that as a residual requirement;
Roy handles user interaction as ACQUIRE and resumes this node after the reply. When the task
contains several separable downstream investigations, expose those substantive gaps even when
they share an unresolved prerequisite.
When the local graph exposes another node whose report is genuinely required by a proposed
child, include that exact node id in depends_on_node_ids. Do not name the current node or one
of its ancestors as a dependency. Roy may derive a child directly from a residual requirement
when proposed_children is empty, so residual descriptions and termination conditions must be
specific and verifiable.
""".strip()


@dataclass(frozen=True)
class Tau3OrganizationAgentConfig:
    checkpoint: Path | None = None
    maximum_internal_decisions_per_turn: int = 4096
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
        return ExplorationEnvelope("default", 0, 24, 0, 8, "expansive")


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
        ToolCall,
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
        derived_gap_keys: list[str] = Field(default_factory=list)
        derived_objective_fingerprints: list[str] = Field(default_factory=list)
        child_spec_node_ids: Dict[str, str] = Field(default_factory=dict)
        failed_tool_candidate_ids: list[str] = Field(default_factory=list)
        pending_acquisition_node: str | None = None
        pending_user_acquisition_node: str | None = None
        final_output: str | None = None
        stopped: bool = False
        decision_count: int = 0
        llm_call_count: int = 0
        tool_call_count: int = 0
        termination_type: str | None = None
        termination_reason: str | None = None
        initial_snapshot_fingerprint: str = ""

    class Tau3OrganizationAgent(LLMConfigMixin, HalfDuplexAgent[OrganizationState]):
        STOP_TOKEN = "###ROY_ORGANIZATION_STOP###"

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

        @classmethod
        def is_stop(cls, message) -> bool:
            return _is_stop_message(message, cls.STOP_TOKEN)

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
                    if state.pending_user_acquisition_node:
                        self._ingest_user_observation(message, state)
                    else:
                        root = state.nodes[0]
                        root["objective"] = str(message.content)
                        root["status"] = "ready"

            for _ in range(config.maximum_internal_decisions_per_turn):
                if _limit_reached(
                    state.decision_count, config.runtime_budget.maximum_decisions
                ):
                    return self._truncate_and_finalize(
                        state, generate, AssistantMessage, SystemMessage,
                        "decision_budget_exhausted"
                    )
                if _limit_reached(
                    state.llm_call_count, config.runtime_budget.maximum_llm_calls
                ):
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
                action_record = {
                    "kind": candidate["kind"],
                    "actor_node_id": record["active_node_id"],
                    "candidate_id": candidate["id"],
                    "description": candidate.get("description"),
                }
                for key in (
                    "tool_name", "from", "to", "resulting_depth", "user_interaction"
                ):
                    if candidate.get(key) is not None:
                        action_record[key] = candidate[key]
                if candidate.get("proposal") is not None:
                    action_record["child_specification"] = dict(candidate["proposal"])
                state.actions.append(action_record)
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
            observation = str(getattr(message, "content", message))
            state.observations.append({
                "id": f"observation-{len(state.observations)}",
                "node_id": actor_id,
                "source_type": "tool",
                "observation": observation,
                "provenance": str(getattr(message, "tool_call_id", "tau3-environment")),
            })
            node = self._node(state, actor_id)
            node["status"] = "ready"
            if _is_failed_tool_observation(observation):
                for action in reversed(state.actions):
                    if (
                        action.get("kind") == "ACQUIRE"
                        and action.get("tool_name")
                        and action.get("candidate_id")
                    ):
                        candidate_id = str(action["candidate_id"])
                        if candidate_id not in state.failed_tool_candidate_ids:
                            state.failed_tool_candidate_ids.append(candidate_id)
                        break
            state.pending_acquisition_node = None

        def _ingest_user_observation(self, message, state: OrganizationState) -> None:
            actor_id = state.pending_user_acquisition_node or "root"
            state.observations.append({
                "id": f"observation-{len(state.observations)}",
                "node_id": actor_id,
                "source_type": "user",
                "observation": str(getattr(message, "content", message)),
                "provenance": "tau3-user-conversation",
            })
            self._node(state, actor_id)["status"] = "ready"
            state.pending_user_acquisition_node = None
            if state.failed_tool_candidate_ids:
                failed = set(state.failed_tool_candidate_ids)
                state.used_candidates = [
                    value for value in state.used_candidates if value not in failed
                ]
                state.failed_tool_candidate_ids = []

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
                dependency_ids = self._valid_dependency_ids(
                    state, actor["id"], proposal.get("depends_on_node_ids", [])
                )
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
                proposal_id = str(proposal.get("id") or "")
                if proposal_id:
                    state.child_spec_node_ids[
                        _child_spec_alias_key(actor["id"], proposal_id)
                    ] = child_id
                state.derived_gap_keys.append(
                    _derived_gap_key(actor["id"], proposal["triggering_gap_id"])
                )
                state.derived_objective_fingerprints.append(
                    _objective_fingerprint(proposal["objective"])
                )
                for producer_id in dependency_ids:
                    producer = self._node(state, producer_id)
                    state.dependency_edges.append({
                        "from": producer_id,
                        "to": child_id,
                        "artifact_id": f"report:{producer_id}",
                        "resolved": producer.get("status") == "returned",
                    })
                if dependency_ids and not all(
                    edge["resolved"] for edge in state.dependency_edges
                    if edge["to"] == child_id
                ):
                    self._node(state, child_id)["status"] = "waiting_dependency"
                return None
            if kind == "CONNECT":
                state.communication_edges.append({
                    "from": candidate["from"], "to": candidate["to"], "required": False
                })
                target = self._node(state, str(candidate["to"]))
                if target["status"] == "reported":
                    target["status"] = "ready"
                return None
            if kind == "RETURN":
                actor["status"] = "returned"
                for edge in state.dependency_edges:
                    if edge["from"] == actor["id"]:
                        edge["resolved"] = True
                self._wake_dependency_nodes(state)
                if actor.get("parent_id"):
                    self._node(state, actor["parent_id"])["status"] = "ready"
                return None
            if kind == "PRUNE":
                self._node(state, str(candidate["target_node_id"]))["status"] = "pruned"
                return None
            if kind == "ACQUIRE":
                if candidate.get("user_interaction"):
                    state.pending_user_acquisition_node = actor["id"]
                    actor["status"] = "waiting"
                    return assistant_type(
                        role="assistant",
                        content=_user_acquisition_question(candidate["requirement"]),
                    )
                state.pending_acquisition_node = actor["id"]
                actor["status"] = "waiting"
                selected_tools = [
                    tool for tool in self.tools
                    if self._tool_name(tool) == str(candidate["tool_name"])
                ]
                if len(selected_tools) != 1:
                    raise ValueError(
                        f"selected tau3 tool is unavailable or ambiguous: {candidate['tool_name']}"
                    )
                selected_tool = selected_tools[0]
                selected_name = str(candidate["tool_name"])
                prompt = system_type(
                    role="system",
                    content=_tool_argument_prompt(
                        selected_name,
                        dict(selected_tool.openai_schema),
                        dict(candidate["requirement"]),
                    ),
                )
                acquire_arguments = _non_thinking_arguments(self.llm_args)
                state.llm_call_count += 1
                argument_response = generate_fn(
                    model=self.llm,
                    tools=[],
                    messages=state.system_messages + state.messages + [prompt],
                    call_name="roy_acquire_arguments",
                    **acquire_arguments,
                )
                arguments = _parse_json_object(str(argument_response.content or "{}"))
                bound_call = _bound_tool_call_payload(
                    selected_name,
                    arguments,
                    f"call_roy_{uuid.uuid4().hex}",
                )
                outward = assistant_type(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(**bound_call)],
                    cost=getattr(argument_response, "cost", None),
                    usage=getattr(argument_response, "usage", None),
                    raw_data={
                        "roy_bound_tool": selected_name,
                        "argument_generation": getattr(argument_response, "raw_data", None),
                    },
                    generation_time_seconds=getattr(
                        argument_response, "generation_time_seconds", None
                    ),
                )
                self._record_llm_event(
                    state, "roy_acquire_arguments", actor["id"], argument_response
                )
                self._record_llm_event(state, "roy_acquire_bound", actor["id"], outward)
                return outward
            if kind == "STOP":
                outward = self._final_message(state, generate_fn, system_type)
                state.final_output = str(outward.content or "")
                state.stopped = True
                state.termination_type = "policy_stop"
                state.termination_reason = "organization policy selected STOP"
                return self._mark_stop(outward)
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
                "local_graph": _local_graph_context(state, node["id"]),
                "incoming_communications": [
                    {
                        "from": edge["from"],
                        "report": self._node(state, str(edge["from"])).get("report"),
                    }
                    for edge in state.communication_edges if edge["to"] == node["id"]
                ],
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
                **_non_thinking_arguments(self.llm_args),
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
                **_non_thinking_arguments(self.llm_args),
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
            if not _limit_reached(
                state.llm_call_count, config.runtime_budget.maximum_llm_calls
            ):
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
            outward = self._mark_stop(outward)
            state.messages.append(outward)
            return outward, state

        def _mark_stop(self, message):
            content = _stop_content(getattr(message, "content", None), self.STOP_TOKEN)
            return message.model_copy(update={"content": content})

        def _candidates(self, state: OrganizationState) -> list[Dict[str, Any]]:
            result: list[Dict[str, Any]] = []
            used = set(state.used_candidates)
            for node in state.nodes:
                if node["status"] in {
                    "pruned", "returned", "waiting", "waiting_dependency"
                }:
                    continue
                actor = str(node["id"])
                if node["status"] == "ready":
                    result.append(_candidate(f"execute:{actor}:{state.decision_count}", "EXECUTE", actor, "Update the node epistemic report", 1.0))
                    continue
                report = node.get("report") or {}
                proposals = _residual_child_specifications(
                    report, node, [str(value["id"]) for value in state.nodes]
                )
                derived_gap_keys = set(state.derived_gap_keys)
                derived_objectives = set(state.derived_objective_fingerprints)
                for index, proposal in enumerate(proposals):
                    gap_key = _derived_gap_key(actor, proposal.get("triggering_gap_id"))
                    objective_fingerprint = _objective_fingerprint(
                        proposal.get("objective")
                    )
                    if (
                        gap_key in derived_gap_keys
                        or objective_fingerprint in derived_objectives
                    ):
                        continue
                    identifier = f"derive:{actor}:{proposal.get('triggering_gap_id')}:{index}"
                    if identifier not in used:
                        result.append({
                            **_candidate(identifier, "DERIVE", actor, str(proposal.get("objective")), 2.0),
                            "proposal": proposal,
                            "dependencies_available": self._dependencies_available(
                                state, actor, proposal.get("depends_on_node_ids", [])
                            ),
                            "resulting_depth": int(node["depth"]) + 1,
                            "resolves_gap": True,
                            "depth_delta": 1,
                        })
                residuals = list(report.get("residual_requirements", []))
                for residual in residuals:
                    if not isinstance(residual, Mapping) or not _is_user_interaction_requirement(
                        residual
                    ):
                        continue
                    gap_id = str(residual.get("id") or "interaction")
                    identifier = f"acquire-user:{actor}:{gap_id}"
                    if identifier not in used:
                        result.append({
                            **_candidate(
                                identifier,
                                "ACQUIRE",
                                actor,
                                f"Acquire required information from the user: {residual.get('description', '')}",
                                0.5,
                            ),
                            "requirement": dict(residual),
                            "user_interaction": True,
                            "external_access": True,
                            "resolves_gap": True,
                        })
                result.extend(_tool_acquisition_candidates(
                    actor, residuals, state.available_tools, used
                ))
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
            active = [
                value for value in state.nodes
                if value["status"] not in {"pruned", "waiting", "waiting_dependency"}
            ]
            existing = {(value["from"], value["to"]) for value in state.communication_edges}
            incoming_targets = {value["to"] for value in state.communication_edges}
            for source in active:
                for target in active:
                    pair = (source["id"], target["id"])
                    identifier = f"connect:{pair[0]}:{pair[1]}"
                    if _communication_candidate_allowed(
                        source, target, pair, existing, incoming_targets, used, identifier
                    ):
                        result.append({
                            **_candidate(
                                identifier,
                                "CONNECT",
                                str(source["id"]),
                                (
                                    f"Share report from {source['id']} with {target['id']}: "
                                    f"{source.get('objective', '')} -> {target.get('objective', '')}"
                                ),
                                0.2,
                            ),
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
            if kind == "EXECUTE" and _remaining_at_most(
                state.llm_call_count, budget.maximum_llm_calls, 1
            ):
                return False
            if (
                kind == "ACQUIRE"
                and not candidate.get("user_interaction")
                and _remaining_at_most(
                    state.llm_call_count, budget.maximum_llm_calls, 1
                )
            ):
                return False
            if kind == "STOP" and _remaining_at_most(
                state.llm_call_count, budget.maximum_llm_calls, 0
            ):
                return False
            if (
                kind == "ACQUIRE"
                and not candidate.get("user_interaction")
                and _limit_reached(
                    state.tool_call_count, budget.maximum_tool_calls
                )
            ):
                return False
            if kind == "DERIVE":
                node_ceiling = _effective_ceiling(
                    budget.maximum_nodes, config.envelope().maximum_nodes
                )
                depth_ceiling = _effective_ceiling(
                    budget.maximum_depth, config.envelope().maximum_depth
                )
                return (
                    bool(candidate.get("dependencies_available", True))
                    and len(state.nodes) < node_ceiling
                    and int(candidate.get("resulting_depth", 0)) <= depth_ceiling
                )
            if kind == "RETURN":
                actor = self._node(state, str(candidate["actor_node_id"]))
                unresolved_dependencies = any(
                    edge["to"] == actor["id"] and not bool(edge.get("resolved"))
                    for edge in state.dependency_edges
                )
                active_children = any(
                    value.get("parent_id") == actor["id"]
                    and value.get("status") not in {"returned", "pruned"}
                    for value in state.nodes
                )
                return not (unresolved_dependencies or active_children)
            return not _limit_reached(
                state.decision_count, budget.maximum_decisions
            )

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
                "llm_calls_remaining_fraction": _remaining_fraction(
                    state.llm_call_count, budget.maximum_llm_calls
                ),
                "tool_calls_remaining_fraction": _remaining_fraction(
                    state.tool_call_count, budget.maximum_tool_calls
                ),
                "nodes_remaining_fraction": _remaining_fraction(
                    len(state.nodes), budget.maximum_nodes
                ),
                "depth_remaining_fraction": _remaining_fraction(
                    maximum_depth, budget.maximum_depth
                ),
                "decisions_remaining_fraction": _remaining_fraction(
                    state.decision_count, budget.maximum_decisions
                ),
            }

        @staticmethod
        def _node(state, node_id):
            for value in state.nodes:
                if value["id"] == node_id:
                    return value
            raise ValueError(f"unknown organization node: {node_id}")

        @staticmethod
        def _valid_dependency_ids(state, actor_id, identifiers):
            known = {str(value["id"]) for value in state.nodes}
            ancestors = {str(actor_id)}
            cursor = Tau3OrganizationAgent._node(state, str(actor_id))
            while cursor.get("parent_id"):
                ancestors.add(str(cursor["parent_id"]))
                cursor = Tau3OrganizationAgent._node(state, str(cursor["parent_id"]))
            result: list[str] = []
            for identifier in identifiers:
                value = str(identifier)
                resolved = (
                    value if value in known
                    else state.child_spec_node_ids.get(
                        _child_spec_alias_key(actor_id, value)
                    )
                )
                if resolved and resolved not in ancestors and resolved not in result:
                    result.append(str(resolved))
            return result

        @staticmethod
        def _dependencies_available(state, actor_id, identifiers):
            known = {str(value["id"]) for value in state.nodes}
            return all(
                str(identifier) in known
                or _child_spec_alias_key(actor_id, identifier) in state.child_spec_node_ids
                for identifier in identifiers
            )

        @staticmethod
        def _wake_dependency_nodes(state):
            for node in state.nodes:
                if node.get("status") != "waiting_dependency":
                    continue
                incoming = [
                    edge for edge in state.dependency_edges if edge["to"] == node["id"]
                ]
                if incoming and all(bool(edge.get("resolved")) for edge in incoming):
                    node["status"] = "ready"

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
    topology = _topology_summary(
        state.nodes,
        state.derivation_edges,
        state.dependency_edges,
        state.communication_edges,
    )
    return {
        "schema_version": 3,
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
        "topology": topology,
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


def _tool_acquisition_candidates(
    actor: str,
    residuals: Sequence[Any],
    available_tools: Sequence[Mapping[str, Any]],
    used_candidate_ids: set[str],
) -> list[Dict[str, Any]]:
    """Expose each relevant tool once for the current semantic requirement.

    The old decision-index identifier made the same tool and unchanged gap look
    new forever.  That let an untrained policy loop until a resource ceiling.
    Stable requirement-derived identifiers preserve open tool choice while
    removing those no-op repetitions from the on-policy action space.
    """

    external_residuals = _tool_access_residuals(residuals)
    if not external_residuals:
        return []
    available_names = [str(value["name"]) for value in available_tools]
    hinted_names = _matching_tool_names(external_residuals, available_names)
    selected_names = hinted_names or set(available_names)
    result: list[Dict[str, Any]] = []
    for tool in available_tools:
        tool_name = str(tool["name"])
        if tool_name not in selected_names:
            continue
        matched = [
            value for value in external_residuals
            if tool_name in _matching_tool_names([value], available_names)
        ]
        requirements = matched if hinted_names else external_residuals
        requirement = {
            "residual_requirements": requirements,
            "selected_tool": tool_name,
        }
        identifier = f"acquire-tool:{actor}:{tool_name}"
        if identifier in used_candidate_ids:
            continue
        short_description = str(
            tool.get("short_desc") or tool.get("description") or ""
        ).strip()
        description = f"Use tau3 tool {tool_name}"
        if short_description:
            description += f" to resolve the reported gap: {short_description}"
        result.append({
            **_candidate(identifier, "ACQUIRE", actor, description, 3.0),
            "requirement": requirement,
            "tool_name": tool_name,
            "external_access": True,
            "resolves_gap": True,
        })
    return result


def _limit_reached(used: int, limit: int | None) -> bool:
    return limit is not None and used >= limit


def _is_failed_tool_observation(observation: Any) -> bool:
    return bool(re.match(r"\s*(?:error\b|\{?\s*\"?error\"?\s*:)", str(observation), re.I))


def _remaining_at_most(used: int, limit: int | None, threshold: int) -> bool:
    return limit is not None and limit - used <= threshold


def _remaining_fraction(used: int, limit: int | None) -> float:
    if limit is None:
        return 1.0
    return max(0.0, (limit - used) / max(1, limit))


def _effective_ceiling(optional_limit: int | None, envelope_limit: int) -> int:
    return envelope_limit if optional_limit is None else min(optional_limit, envelope_limit)


def _topology_summary(
    nodes: Sequence[Mapping[str, Any]],
    derivation_edges: Sequence[Mapping[str, Any]],
    dependency_edges: Sequence[Mapping[str, Any]],
    communication_edges: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out_degree: Dict[str, int] = {}
    for edge in derivation_edges:
        source = str(edge["from"])
        out_degree[source] = out_degree.get(source, 0) + 1
    branched = any(value > 1 for value in out_degree.values())
    if dependency_edges and communication_edges:
        topology_class = "hybrid_dag"
    elif dependency_edges:
        topology_class = "dependency_dag"
    elif communication_edges:
        topology_class = "communication_dag"
    elif branched:
        topology_class = "fan_out_tree"
    elif len(nodes) > 1:
        topology_class = "chain"
    else:
        topology_class = "single"
    return {
        "class": topology_class,
        "node_count": len(nodes),
        "derived_agent_count": max(0, len(nodes) - 1),
        "maximum_depth": max((int(value.get("depth", 0)) for value in nodes), default=0),
        "derivation_edge_count": len(derivation_edges),
        "dependency_edge_count": len(dependency_edges),
        "communication_edge_count": len(communication_edges),
        "maximum_derivation_fanout": max(out_degree.values(), default=0),
    }


def _non_thinking_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Disable thinking because tau3 does not preserve DeepSeek reasoning metadata."""

    result = dict(arguments)
    extra_body = dict(result.get("extra_body") or {})
    extra_body["thinking"] = {"type": "disabled"}
    result["extra_body"] = extra_body
    return result


def _tool_argument_prompt(
    selected_tool: str,
    openai_schema: Mapping[str, Any],
    requirement: Mapping[str, Any],
) -> str:
    """Ask only for arguments; Roy, rather than the provider, binds the tool name."""

    return (
        "Roy has already selected exactly one tau3 tool. Return only one JSON object "
        "containing that tool's arguments. Do not return a tool name, wrapper, prose, "
        "Markdown, or a tool call. Use only information available in the conversation and "
        "domain policy. The official tau3 environment will validate and execute the object.\n"
        f"Selected tool: {selected_tool}\n"
        f"Residual requirement: {json.dumps(requirement, ensure_ascii=False)}\n"
        f"Official schema: {json.dumps(dict(openai_schema), ensure_ascii=False)}"
    )


def _bound_tool_call_payload(
    selected_tool: str,
    arguments: Mapping[str, Any],
    identifier: str,
) -> Dict[str, Any]:
    """Bind model-generated arguments to the policy-selected tool identity."""

    return {
        "id": identifier,
        "name": selected_tool,
        "arguments": dict(arguments),
        "requestor": "assistant",
    }


def _stop_content(content: Any, stop_token: str) -> str:
    value = str(content or "")
    return value if stop_token in value else f"{value}\n{stop_token}".strip()


def _is_stop_message(message: Any, stop_token: str) -> bool:
    return stop_token in str(getattr(message, "content", "") or "")


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


def _proposal_gap_id(
    proposal: Mapping[str, Any],
    index: int,
    normalized_gaps: Sequence[Mapping[str, Any]],
) -> str:
    explicit = str(
        proposal.get("triggering_gap_id") or proposal.get("triggering_gap") or ""
    )
    gap_ids = {str(value.get("id")) for value in normalized_gaps}
    if explicit in gap_ids:
        return explicit
    if index < len(normalized_gaps):
        return str(normalized_gaps[index]["id"])
    proposal_tokens = set(re.findall(
        r"[a-z0-9_]+",
        " ".join(
            str(proposal.get(key) or "").lower()
            for key in ("objective", "termination_condition", "possible_external_access")
        ),
    ))
    ranked = sorted(
        (
            len(proposal_tokens & set(re.findall(
                r"[a-z0-9_]+", str(gap.get("description") or "").lower()
            ))),
            str(gap["id"]),
        )
        for gap in normalized_gaps
    )
    return ranked[-1][1] if ranked and ranked[-1][0] > 0 else ""


def _normalize_report(value: Mapping[str, Any], node: Mapping[str, Any]) -> Dict[str, Any]:
    gaps = _as_list(value.get("residual_requirements"))
    proposals = _as_list(value.get("proposed_children"))
    normalized_gaps = []
    for gap in gaps:
        if isinstance(gap, Mapping):
            gap_value = dict(gap)
        elif str(gap).strip():
            gap_value = {"description": str(gap).strip()}
        else:
            continue
        description = str(
            gap_value.get("description")
            or gap_value.get("required_information")
            or "unresolved requirement"
        )
        fallback_material = f"{node['id']}:{description}".encode("utf-8")
        fallback_id = hashlib.sha256(fallback_material).hexdigest()[:12]
        normalized_gaps.append({
            **gap_value,
            "id": str(gap_value.get("id") or f"gap-{node['id']}-{fallback_id}"),
            "description": description,
            "possible_external_access": _as_list(
                gap_value.get("possible_external_access")
            ),
        })
    normalized_proposals = []
    gap_ids = {value["id"] for value in normalized_gaps}
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            continue
        gap_id = _proposal_gap_id(proposal, index, normalized_gaps)
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


def _derived_gap_key(actor_id: Any, gap_id: Any) -> str:
    return f"{actor_id}:{gap_id}"


def _child_spec_alias_key(actor_id: Any, proposal_id: Any) -> str:
    return f"{actor_id}:{proposal_id}"


def _objective_fingerprint(objective: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(objective).lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _residual_child_specifications(
    report: Mapping[str, Any],
    node: Mapping[str, Any],
    known_node_ids: Sequence[str],
) -> list[Dict[str, Any]]:
    """Build open child specifications from the node's own residual gaps.

    Explicit model proposals remain authoritative. A residual without an explicit
    proposal receives one conservative specification so the structural policy can
    actually choose DERIVE. This is neither a role catalog nor teacher annotation:
    its objective, stopping condition, access hints, and dependencies all come from
    the current node report.
    """

    residuals = [
        dict(value) for value in _as_list(report.get("residual_requirements"))
        if isinstance(value, Mapping)
    ]
    gaps = {str(value.get("id")): value for value in residuals if value.get("id")}
    known = set(str(value) for value in known_node_ids)
    actor_id = str(node.get("id"))
    result: list[Dict[str, Any]] = []
    explicit_gap_ids: set[str] = set()
    proposal_aliases = {
        str(value.get("id")) for value in _as_list(report.get("proposed_children"))
        if isinstance(value, Mapping) and value.get("id")
    }
    for proposal in _as_list(report.get("proposed_children")):
        if not isinstance(proposal, Mapping):
            continue
        gap_id = str(proposal.get("triggering_gap_id") or "")
        if gap_id not in gaps:
            continue
        if _is_user_interaction_requirement(gaps[gap_id]):
            continue
        child = dict(proposal)
        child["depends_on_node_ids"] = _safe_dependency_ids(
            child.get("depends_on_node_ids"), known, actor_id, proposal_aliases
        )
        result.append(child)
        explicit_gap_ids.add(gap_id)
    for index, gap in enumerate(residuals):
        gap_id = str(gap.get("id") or f"gap-{actor_id}-{index}")
        if gap_id in explicit_gap_ids:
            continue
        if _is_user_interaction_requirement(gap):
            continue
        if _as_list(gap.get("possible_external_access")):
            continue
        description = str(
            gap.get("description")
            or gap.get("required_information")
            or "Resolve the remaining information requirement"
        ).strip()
        termination = str(
            gap.get("termination_condition")
            or gap.get("verification_condition")
            or f"Return an evidence-backed resolution of: {description}"
        ).strip()
        result.append({
            "id": f"residual-spec-{actor_id}-{index}",
            "triggering_gap_id": gap_id,
            "objective": description,
            "termination_condition": termination,
            "possible_external_access": _as_list(gap.get("possible_external_access")),
            "depends_on_node_ids": _safe_dependency_ids(
                gap.get("depends_on_node_ids"), known, actor_id
            ),
            "origin": "model_reported_residual",
        })
    return result


def _is_user_interaction_requirement(requirement: Mapping[str, Any]) -> bool:
    access = {
        re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()
        for value in _as_list(requirement.get("possible_external_access"))
    }
    text = " ".join(
        str(requirement.get(key) or "").lower()
        for key in ("description", "required_information", "termination_condition")
    )
    interaction_phrases = (
        "ask the user",
        "ask user",
        "prompt the user",
        "user message",
        "user's user_id",
        "user's user id",
        "user id from the user",
        "obtain the user's user id",
        "provided by the user",
        "confirm with the user",
    )
    user_access = bool(access & {
        "user",
        "customer",
        "human",
        "user simulator",
        "user interaction",
    })
    return user_access or any(phrase in text for phrase in interaction_phrases)


def _user_acquisition_question(requirement: Mapping[str, Any]) -> str:
    description = str(
        requirement.get("description")
        or requirement.get("required_information")
        or "the missing information needed to continue"
    ).strip()
    return f"To continue, could you please provide this information: {description}?"


def _matching_tool_names(
    residuals: Sequence[Mapping[str, Any]], available_tool_names: Sequence[str]
) -> set[str]:
    names = {str(value) for value in available_tool_names}
    result: set[str] = set()
    for residual in residuals:
        for hint in _as_list(residual.get("possible_external_access")):
            normalized = re.sub(r"[^a-z0-9_]+", "_", str(hint).lower()).strip("_")
            for name in names:
                lowered = name.lower()
                if normalized == lowered or lowered in normalized.split("_"):
                    result.add(name)
                elif lowered in normalized and len(lowered) >= 6:
                    result.add(name)
    return result


def _tool_access_residuals(residuals: Sequence[Any]) -> list[Dict[str, Any]]:
    return [
        dict(value) for value in residuals
        if isinstance(value, Mapping)
        and value.get("possible_external_access")
        and not _is_user_interaction_requirement(value)
    ]


def _safe_dependency_ids(
    value: Any,
    known: set[str],
    actor_id: str,
    symbolic: set[str] | None = None,
) -> list[str]:
    allowed = known | set(symbolic or set())
    return list(dict.fromkeys(
        str(identifier) for identifier in _as_list(value)
        if str(identifier) in allowed and str(identifier) != actor_id
    ))


def _communication_candidate_allowed(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    pair: tuple[Any, Any],
    existing: set[tuple[Any, Any]],
    incoming_targets: set[Any],
    used: set[str],
    identifier: str,
) -> bool:
    """Offer sparse forward communication choices so the realized graph stays a DAG."""

    if pair[0] == pair[1] or pair in existing or identifier in used:
        return False
    if (
        pair[1] in incoming_targets
        or not source.get("report")
        or source.get("status") not in {"reported", "returned"}
    ):
        return False
    source_index = _node_sequence(source.get("id"))
    target_index = _node_sequence(target.get("id"))
    return source_index < target_index and target.get("status") in {"ready", "reported"}


def _node_sequence(identifier: Any) -> int:
    value = str(identifier)
    if value == "root":
        return 0
    match = re.fullmatch(r"node-(\d+)", value)
    return int(match.group(1)) + 1 if match else 2**31 - 1


def _local_graph_context(state: Any, actor_id: str) -> Dict[str, Any]:
    """Expose IDs the current node may use for autonomous dependency proposals."""

    return {
        "current_node_id": actor_id,
        "nodes": [
            {
                "id": str(node["id"]),
                "parent_id": node.get("parent_id"),
                "objective": node.get("objective"),
                "status": node.get("status"),
                "report_available": bool(node.get("report")),
            }
            for node in state.nodes
        ],
        "derivation_edges": list(state.derivation_edges),
        "dependency_edges": list(state.dependency_edges),
        "communication_edges": list(state.communication_edges),
    }


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"summary": value}


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    material = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
