from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Protocol

from .harness import AgentHarness, AgentHarnessConfig
from .mia import SemanticInformationLandscape
from .tools import TaskToolRegistry, ToolRequest
from .types import (
    AgentState,
    AgentStatus,
    CandidateGraph,
    RealizedSubgraph,
    ResultState,
)


class CompletionLike(Protocol):
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


class CompletionClient(Protocol):
    model: str

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        metadata: Dict[str, Any] | None = None,
        json_mode: bool = False,
        thinking: str | None = None,
    ) -> CompletionLike: ...


@dataclass
class CallAudit:
    calls: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    latency_ms: Dict[str, int] = field(default_factory=dict)

    def record(self, purpose: str, completion: CompletionLike) -> None:
        self.calls[purpose] = self.calls.get(purpose, 0) + 1
        self.tokens[purpose] = self.tokens.get(purpose, 0) + int(completion.total_tokens)
        self.latency_ms[purpose] = self.latency_ms.get(purpose, 0) + int(completion.latency_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "tokens": dict(self.tokens),
            "latency_ms": dict(self.latency_ms),
            "total_calls": sum(self.calls.values()),
            "total_tokens": sum(self.tokens.values()),
        }


class JsonLLM:
    def __init__(self, client: CompletionClient, audit: CallAudit) -> None:
        self.client = client
        self.audit = audit

    def call(
        self,
        purpose: str,
        system: str,
        payload: Mapping[str, Any],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        thinking: str | None = "disabled",
    ) -> Dict[str, Any]:
        completion = self.client.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            metadata={"purpose": purpose},
            json_mode=True,
            thinking=thinking,
        )
        self.audit.record(purpose, completion)
        if not completion.content.strip():
            raise ValueError(
                f"{purpose} returned empty content after {completion.completion_tokens} "
                "completion tokens; increase its budget or disable provider reasoning"
            )
        return parse_json_object(completion.content)


class WorkerModel:
    SYSTEM = """You are Roy's frozen Worker. Execute only the supplied local objective using the
agent's private state and visible context. Return one JSON object. Do not assume access to other
agents or memories unless their content is present. For MATH, candidate_answer must be a concise
answer containing only the final ``\\boxed{...}`` expression, with all derivation placed in the
other result fields; textual answers must use ``\\boxed{\\text{WORD}}``. For HumanEval,
candidate_answer must be complete
Python code including the requested function. Also propose at most the requested number of genuinely
different unresolved cognitive directions as a dependency graph. Proposals are semantic drafts, not
complete agent configurations. Hard dependencies mean a node cannot execute without its producer;
soft dependencies only indicate useful information flow. Candidate proposals are step-specific:
use the current matrix, current peers, private memory, inbound messages, active dependencies and
recent trajectory events. The candidate_answer must agree with the claims, evidence, and final
reasoning summary; perform a final consistency check before returning it. On the root's first round,
propose at least minimum_candidates independent verification directions whenever capacity permits,
even if the answer is high-confidence. A failed tool call always requires a verification direction.
Never repeat a gap that the current state or history already resolved."""

    RECONCILE_SYSTEM = """You are the final consistency pass inside Roy's frozen Worker harness.
Do not solve the task anew and do not add facts. Read the supplied structured result and return the
single MATH boxed answer actually entailed by its claims, evidence, and reasoning summary. If those
fields do not entail a unique answer, preserve the existing candidate_answer and mark ambiguity.
Return JSON only."""

    PROPOSE_SYSTEM = """You are Roy's frozen step-level candidate proposer. The supplied agent
state is already the committed result of the previous transition. Never solve the task again,
rewrite that result, or mutate memory. Inspect the current MAS, matrix, dependencies, and recent
trajectory, then propose only genuinely unresolved cognitive directions as a dependency graph.
Proposals are semantic drafts rather than complete agent configurations. Hard dependencies mean a
candidate cannot execute without its producer; soft dependencies only indicate useful information
flow. Never repeat a direction that the committed state or history already resolved. Return exactly
one JSON object."""

    def __init__(
        self,
        llm: JsonLLM,
        max_tokens: int = 2048,
        max_tool_rounds: int = 2,
        max_tool_calls: int = 3,
        result_reconciler_max_tokens: int = 256,
        harness_config: AgentHarnessConfig | None = None,
    ) -> None:
        self.llm, self.max_tokens = llm, max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls = max_tool_calls
        self.result_reconciler_max_tokens = result_reconciler_max_tokens
        self.harness_config = harness_config or AgentHarnessConfig()
        self.tools: TaskToolRegistry | None = None

    def configure_tools(self, tools: TaskToolRegistry) -> None:
        self.tools = tools

    def execute(
        self,
        agent: AgentState,
        benchmark: str,
        *,
        round_index: int,
        max_candidates: int,
        organization_context: Mapping[str, Any] | None = None,
    ) -> tuple[ResultState, CandidateGraph]:
        value = self._call_with_tools(
            "worker",
            {
                "benchmark": benchmark,
                "round_index": round_index,
                "candidate_id_prefix": f"r{round_index}_{agent.agent_id}_c",
                "max_candidates": max_candidates,
                "minimum_candidates": 1 if round_index == 0 and max_candidates > 0 else 0,
                "current_organization_context": dict(organization_context or {}),
                "required_schema": {
                    "result": {
                        "candidate_answer": "string", "claims": ["string"],
                        "evidence": ["string"], "assumptions": ["string"],
                        "unresolved": ["string"], "reasoning_summary": "string",
                        "confidence": "0..1",
                    },
                    "memory_entries": ["new private-memory entry grounded in this execution"],
                    "candidate_dependency_graph": {
                        "nodes": [{
                            "candidate_id": "unique string using candidate_id_prefix",
                            "parent_id": agent.agent_id,
                            "direction": "what to investigate, not a role label",
                            "why_needed": "specific unresolved issue",
                            "required_inputs": ["artifact references"],
                            "requested_tools": ["available tool id"],
                            "expected_output": "verifiable output",
                            "stop_condition": "observable completion condition",
                        }],
                        "dependencies": [{
                            "source": "agent/candidate id", "target": "candidate id",
                            "kind": "hard|soft", "artifact": "required information",
                        }],
                    },
                    "tool_requests": [{
                        "tool_name": "assigned tool name",
                        "arguments": {"tool-specific": "arguments"},
                        "reason": "what uncertainty this call resolves",
                    }],
                },
            },
            agent,
            max_tokens=self.max_tokens,
        )
        result = self._result_from_value(value, benchmark)
        graph = CandidateGraph.from_dict(value.get("candidate_dependency_graph", {}), agent.agent_id)
        if len(graph.nodes) > max_candidates:
            raise ValueError(f"Worker proposed {len(graph.nodes)} candidates; maximum is {max_candidates}")
        AgentHarness(agent, self.tools, self.harness_config).apply_model_update(
            result, value.get("memory_entries", []),
        )
        return result, graph

    def update(
        self,
        agent: AgentState,
        benchmark: str,
        inbound_messages: List[str],
        *,
        tool_scope: str = "counterfactual",
    ) -> AgentState:
        updated = copy.deepcopy(agent)
        harness = AgentHarness(updated, self.tools, self.harness_config)
        harness.receive_messages(inbound_messages)
        value = self._call_with_tools(
            "receiver_update",
            {
                "benchmark": benchmark,
                "inbound_messages": inbound_messages,
                "instruction": "Update Z and private M only. Return result and memory_entries.",
                "required_schema": {
                    "result": "ResultState", "memory_entries": ["string"],
                    "tool_requests": ["optional ToolRequest"],
                },
            },
            updated,
            max_tokens=self.max_tokens,
            tool_scope=tool_scope,
        )
        harness.apply_model_update(
            self._result_from_value(value, benchmark), value.get("memory_entries", []),
        )
        return updated

    def propose_candidates(
        self,
        agent: AgentState,
        *,
        round_index: int,
        max_candidates: int,
        organization_context: Mapping[str, Any] | None = None,
    ) -> CandidateGraph:
        """Propose children from committed X_t without re-executing or mutating the parent."""
        value = self.llm.call(
            "candidate_proposal",
            self.PROPOSE_SYSTEM,
            {
                "round_index": round_index,
                "candidate_id_prefix": f"r{round_index}_{agent.agent_id}_c",
                "max_candidates": max_candidates,
                "committed_agent": AgentHarness(
                    agent, self.tools, self.harness_config,
                ).execution_view(),
                "current_organization_context": dict(organization_context or {}),
                "required_schema": {
                    "candidate_dependency_graph": {
                        "nodes": [{
                            "candidate_id": "unique string using candidate_id_prefix",
                            "parent_id": agent.agent_id,
                            "direction": "what unresolved issue to investigate",
                            "why_needed": "specific unresolved issue",
                            "required_inputs": ["artifact references"],
                            "requested_tools": ["available tool id"],
                            "expected_output": "verifiable output",
                            "stop_condition": "observable completion condition",
                        }],
                        "dependencies": [{
                            "source": "agent/candidate id", "target": "candidate id",
                            "kind": "hard|soft", "artifact": "required information",
                        }],
                    },
                },
            },
            max_tokens=self.max_tokens,
        )
        graph = CandidateGraph.from_dict(
            value.get("candidate_dependency_graph", {}), agent.agent_id,
        )
        if len(graph.nodes) > max_candidates:
            raise ValueError(
                f"Worker proposed {len(graph.nodes)} candidates; maximum is {max_candidates}"
            )
        return graph

    def execute_local(
        self,
        agent: AgentState,
        benchmark: str,
        *,
        tool_scope: str = "counterfactual",
    ) -> AgentState:
        updated = copy.deepcopy(agent)
        value = self._call_with_tools(
            "provisional_worker",
            {
                "benchmark": benchmark,
                "instruction": (
                    "Execute the local objective now. Do not propose children. "
                    "Return the completed result and private memory updates."
                ),
                "required_schema": {
                    "result": "ResultState", "memory_entries": ["string"],
                    "tool_requests": ["optional ToolRequest"],
                },
            },
            updated,
            max_tokens=self.max_tokens,
            tool_scope=tool_scope,
        )
        AgentHarness(updated, self.tools, self.harness_config).apply_model_update(
            self._result_from_value(value, benchmark), value.get("memory_entries", []),
        )
        return updated

    def _result_from_value(self, value: Mapping[str, Any], benchmark: str) -> ResultState:
        raw_result = value.get("result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("Worker omitted the required top-level result object")
        result = ResultState.from_dict(raw_result)
        if not result.candidate_answer.strip():
            raise ValueError("Worker returned an empty candidate_answer")
        if benchmark != "MATH" or self.result_reconciler_max_tokens <= 0:
            return result
        reconciled = self.llm.call(
            "worker_result_reconciliation",
            self.RECONCILE_SYSTEM,
            {
                "result": asdict(result),
                "required_schema": {
                    "candidate_answer": "one exact boxed expression",
                    "ambiguous": "boolean",
                    "basis": "short pointer to the supplied derivation",
                },
            },
            max_tokens=self.result_reconciler_max_tokens,
        )
        candidate = str(reconciled.get("candidate_answer", "")).strip()
        if candidate and not bool(reconciled.get("ambiguous", False)):
            if "\n" in candidate or len(candidate) > 256:
                raise ValueError("Worker reconciler returned a non-concise MATH answer")
            result.candidate_answer = candidate
        return result

    def _call_with_tools(
        self,
        purpose: str,
        payload: Dict[str, Any],
        agent: AgentState,
        *,
        max_tokens: int,
        tool_scope: str = "committed",
    ) -> Dict[str, Any]:
        harness = AgentHarness(agent, self.tools, self.harness_config)
        working_payload = dict(payload)
        working_payload["agent"] = harness.execution_view()
        working_payload["available_tool_schemas"] = harness.tool_catalog
        observations: List[Dict[str, Any]] = []
        fingerprints: set[str] = set()
        calls = 0
        for tool_round in range(self.max_tool_rounds + 1):
            if observations:
                working_payload["tool_observations"] = observations
                working_payload["tool_instruction"] = (
                    "Revise the result using these observations. Request another tool only if "
                    "a specific unresolved issue remains. Never claim a failed tool succeeded."
                )
            value = self.llm.call(
                purpose,
                self.SYSTEM,
                working_payload,
                max_tokens=max_tokens,
            )
            raw_requests = value.get("tool_requests", [])
            if not raw_requests or tool_round == self.max_tool_rounds or calls >= self.max_tool_calls:
                return value
            if self.tools is None:
                raise ValueError("Worker requested tools but no task tool registry is configured")
            for raw_request in raw_requests:
                if calls >= self.max_tool_calls:
                    break
                calls += 1
                try:
                    request = ToolRequest.from_dict(raw_request)
                except (TypeError, ValueError) as error:
                    observations.append({
                        "tool_name": "INVALID_TOOL_REQUEST",
                        "success": False,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    continue
                fingerprint = json.dumps({
                    "tool_name": request.tool_name, "arguments": request.arguments,
                }, sort_keys=True, default=str)
                if fingerprint in fingerprints:
                    observations.append({
                        "tool_name": request.tool_name, "success": False,
                        "error": "duplicate tool request rejected",
                    })
                    continue
                fingerprints.add(fingerprint)
                result = harness.execute_tool(request, scope=tool_scope)
                observations.append({
                    "tool_name": request.tool_name,
                    "arguments": request.arguments,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                })
        return value


class GlobalSelector:
    SYSTEM = """You are Roy's frozen Global Semantic Searcher. Select a small set of distinct,
dependency-closed candidate subgraphs that target unresolved issues and are not redundant with the
current MAS. You do not execute candidates and do not assign numerical information-gain scores.
Return JSON only. Every selected subgraph must list candidate ids; hard predecessors will be closed
and validated by the runtime. Retain at least one non-redundant independent verifier when the root
has no successful external/tool verification; self-reported confidence is not verification."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 1024) -> None:
        self.llm, self.max_tokens = llm, max_tokens

    def select(
        self,
        graph: CandidateGraph,
        agents: Mapping[str, AgentState],
        *,
        maximum_subgraphs: int,
        maximum_nodes: int,
        organization_context: Mapping[str, Any] | None = None,
    ) -> List[List[str]]:
        value = self.llm.call(
            "global_selector",
            self.SYSTEM,
            {
                "current_agents": {
                    key: AgentHarness(agent).public_summary() for key, agent in agents.items()
                },
                "candidate_graph": graph.subset(graph.nodes),
                "current_organization_context": dict(organization_context or {}),
                "maximum_subgraphs": maximum_subgraphs,
                "maximum_nodes_per_subgraph": maximum_nodes,
                "required_schema": {
                    "selected_subgraphs": [{"candidate_ids": ["id"], "selection_reason": "string"}]
                },
            },
            max_tokens=self.max_tokens,
        )
        selected: List[List[str]] = []
        seen: set[frozenset[str]] = set()
        claimed_candidates: set[str] = set()
        for raw in value.get("selected_subgraphs", []):
            closure = graph.hard_closure(raw.get("candidate_ids", []), agents)
            if (
                not closure or len(closure) > maximum_nodes or closure in seen
                or claimed_candidates.intersection(closure)
            ):
                continue
            seen.add(closure)
            claimed_candidates.update(closure)
            selected.append(sorted(closure))
            if len(selected) == maximum_subgraphs:
                break
        return selected


class CandidateXRealizer:
    """The deliberately expensive semantic call that materializes final candidate X."""

    SYSTEM = """You are Roy's external Candidate X Realizer. This is the high-compute semantic
configuration stage. Carefully reason about the original task, current MAS, selected dependency
subgraph, benchmark contract, tool availability, and information-flow needs. Then configure every
candidate as a complete autonomous state X_i=(Q_i,R_i,C_i,M_i,T_i,Z_i,Sigma_i).

Q/objective must be narrow and executable. R/role is a capability boundary derived from Q, never a
role-pool label. C/context must explicitly name mandatory artifact inputs and useful weighted source
agents without copying hidden global memory. M/memory must be a private memory/<agent_id> namespace.
T/tools must be a subset of available_tools. Z/result is the candidate's initial epistemic state and
must preserve uncertainty rather than fabricate work not yet performed. Sigma/status must respect
hard dependencies. expected_output and stop_condition must be observable. Return a concise
configuration_reasoning_summary explaining the design tradeoffs, not private chain-of-thought.
The context original_task field must copy the supplied original_task exactly.
Return exactly one JSON object and configure all fields; the runtime will reject incomplete X."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 4096, thinking: str = "disabled") -> None:
        self.llm, self.max_tokens, self.thinking = llm, max_tokens, thinking

    def realize(
        self,
        subgraph_id: str,
        selected_ids: List[str],
        graph: CandidateGraph,
        agents: Mapping[str, AgentState],
        *,
        benchmark: str,
        original_task: str,
        public_tests: List[str],
        available_tools: List[str],
        organization_context: Mapping[str, Any] | None = None,
    ) -> RealizedSubgraph:
        value = self.llm.call(
            "candidate_x_realization",
            self.SYSTEM,
            {
                "subgraph_id": subgraph_id,
                "benchmark": benchmark,
                "original_task": original_task,
                "public_tests": public_tests,
                "available_tools": available_tools,
                "current_mas": {
                    key: AgentHarness(agent).public_summary() for key, agent in agents.items()
                },
                "current_organization_context": dict(organization_context or {}),
                "selected_candidate_subgraph": graph.subset(selected_ids),
                "required_schema": {
                    "configuration_reasoning_summary": "concise design rationale",
                    "risks": ["configuration risk"],
                    "agents": [{
                        "agent_id": "exact candidate_id", "parent_id": "existing agent id",
                        "objective": "Q_i", "role": "R_i",
                        "context": {
                            "original_task": "exact supplied original_task",
                            "mandatory_inputs": ["artifact"],
                            "weighted_inputs": {"source agent id": "0..1"},
                            "received_messages": [], "public_tests": ["allowed public test"],
                        },
                        "memory": {
                            "namespace": "memory/<agent_id>", "inherited_refs": ["reference"],
                            "entries": [],
                        },
                        "tools": ["available tool id"],
                        "result": {
                            "candidate_answer": "", "claims": [], "evidence": [],
                            "assumptions": [], "unresolved": ["local unresolved objective"],
                            "reasoning_summary": "configuration only; not execution", "confidence": 0,
                        },
                        "status": "ready|waiting_for_dependencies",
                        "expected_output": "contract", "stop_condition": "condition",
                    }],
                },
            },
            max_tokens=self.max_tokens,
            thinking=self.thinking,
        )
        raw_agents = value.get("agents", [])
        by_id = {str(raw.get("agent_id", "")): raw for raw in raw_agents}
        if set(by_id) != set(selected_ids):
            raise ValueError(
                f"X Realizer returned agents {sorted(by_id)}; expected {sorted(selected_ids)}"
            )
        realized = {
            candidate_id: AgentState.from_dict(
                {
                    **by_id[candidate_id],
                    # The dependency graph, not the external realizer, owns the
                    # immutable derivation parent relation.
                    "parent_id": graph.nodes[candidate_id].parent_id,
                },
                original_task=original_task,
                available_tools=available_tools, expected_agent_id=candidate_id,
            )
            for candidate_id in selected_ids
        }
        for agent in realized.values():
            AgentHarness(agent)
        allowed_sources = set(agents) | set(realized)
        for candidate_id, agent in realized.items():
            draft = graph.nodes[candidate_id]
            if agent.parent_id not in agents:
                raise ValueError(f"Agent {candidate_id} has invalid parent {agent.parent_id!r}")
            unknown_tests = set(agent.context.public_tests) - set(public_tests)
            if unknown_tests:
                raise ValueError(f"Agent {candidate_id} introduced non-public tests")
            unknown_sources = set(agent.context.weighted_inputs) - allowed_sources
            if unknown_sources or candidate_id in agent.context.weighted_inputs:
                raise ValueError(f"Agent {candidate_id} has invalid weighted input sources")
            missing_inputs = set(draft.required_inputs) - set(agent.context.mandatory_inputs)
            if missing_inputs:
                raise ValueError(f"Agent {candidate_id} omitted mandatory draft inputs: {sorted(missing_inputs)}")
        dependencies = [
            edge for edge in graph.dependencies
            if edge.target in realized and (edge.source in realized or edge.source in agents)
        ]
        for edge in dependencies:
            if edge.kind == "hard" and edge.source in realized:
                if realized[edge.target].status != AgentStatus.WAITING:
                    raise ValueError(
                        f"Agent {edge.target} must wait for hard predecessor {edge.source}"
                    )
        reasoning_summary = str(value.get("configuration_reasoning_summary", "")).strip()
        if not reasoning_summary:
            raise ValueError("X Realizer omitted configuration_reasoning_summary")
        return RealizedSubgraph(
            subgraph_id=subgraph_id,
            candidate_ids=list(selected_ids),
            agents=realized,
            dependencies=dependencies,
            configuration_reasoning_summary=reasoning_summary,
            risks=[str(item) for item in value.get("risks", [])],
        )


class SemanticInformationJudge:
    """Estimate one shared semantic landscape for a whole organization step."""

    SYSTEM = """You are Roy's frozen Semantic Information Judge. Inspect the supplied current
agent states once and estimate the task-conditioned semantic parameters used by the MIA matrix
functional. You do not choose an A2A matrix and do not simulate any candidate matrix.

G[source][receiver] is a directional potential in [0,1]: how much novel, usable, task-relevant
information the source currently has for the receiver, conditional on what the receiver already
knows. R[i][j] is symmetric information redundancy in [0,1]. Lambda[i] is the receiving Agent's
conversion fidelity in [0,1]. These are calibrated semantic estimates, never measured bits.
Distinguish repeated conclusions from complementary evidence, respect direction and unresolved
dependencies, and use the exact supplied agent order. Return JSON only."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 4096) -> None:
        self.llm, self.max_tokens = llm, max_tokens

    def estimate(
        self,
        agents: Mapping[str, AgentState],
        *,
        benchmark: str,
        root_id: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> SemanticInformationLandscape:
        agent_ids = list(agents)
        size = len(agent_ids)
        value = self.llm.call(
            "semantic_information_judge",
            self.SYSTEM,
            {
                "benchmark": benchmark,
                "root_id": root_id,
                "agent_ids": agent_ids,
                "current_agent_states": {
                    agent_id: AgentHarness(agent).execution_view()
                    for agent_id, agent in agents.items()
                },
                "current_state_context": dict(state_context or {}),
                "required_schema": {
                    "agent_ids": "exact supplied agent_ids in the same order",
                    "directional_potential": (
                        f"{size}x{size} G as arrays or agent-id map; row=source column=receiver"
                    ),
                    "redundancy": f"{size}x{size} symmetric R as arrays or agent-id map",
                    "conversion_fidelity": f"{size} Lambda values as array or agent-id map",
                    "root_uncertainty": "number in [0,1] grounded in the root state",
                    "calibration_summary": "concise evidence-grounded rationale",
                },
            },
            max_tokens=self.max_tokens,
        )
        return SemanticInformationLandscape.from_dict(
            value, expected_agent_ids=agent_ids, root_id=root_id,
        )


class ChannelizerModel:
    SYSTEM = """You are Roy's frozen A2A Channelizer. Compress only source information useful to
the receiver's objective. Capacity is a semantic budget: low capacity carries a conclusion and
critical evidence; high capacity may add derivation, assumptions, uncertainty, counterexamples and
provenance. Never invent facts or expose unrelated memory. Return JSON only."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 768) -> None:
        self.llm, self.max_tokens = llm, max_tokens

    def message(self, source: AgentState, receiver: AgentState, weight: float, token_budget: int) -> str:
        value = self.llm.call(
            "channelizer",
            self.SYSTEM,
            {
                "source": AgentHarness(source).channel_source_view(receiver.objective),
                "receiver": AgentHarness(receiver).public_summary(),
                "capacity_weight": weight, "maximum_tokens": max(1, round(weight * token_budget)),
                "required_schema": {"message": "task-conditioned semantic message"},
            },
            max_tokens=min(self.max_tokens, max(64, round(weight * token_budget))),
        )
        return str(value.get("message", ""))


class PosteriorProbeModel:
    SYSTEM = """You are Roy's frozen Posterior Probe, separate from the Worker. Map only the supplied
root epistemic state to a probability distribution over the exact shared hypothesis support. Do not
solve the task anew. Assign every probability, including OTHER, and make them sum to one. Return JSON."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 512) -> None:
        self.llm, self.max_tokens = llm, max_tokens

    def posterior(
        self,
        root: AgentState,
        support: List[str],
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        value = self.llm.call(
            "posterior_probe",
            self.SYSTEM,
            {
                "benchmark": benchmark, "root_state": AgentHarness(root).execution_view(),
                "current_state_context": dict(state_context or {}),
                "hypothesis_support": support,
                "required_schema": {"probabilities": {item: "0..1" for item in support}},
            },
            max_tokens=self.max_tokens,
        )
        raw = dict(value.get("probabilities", {}))
        probabilities = {item: max(0.0, float(raw.get(item, 0.0))) for item in support}
        total = sum(probabilities.values())
        if total <= 0:
            return {item: 1.0 / len(support) for item in support}
        return {item: probability / total for item, probability in probabilities.items()}


class PairwiseInformationProbeModel:
    SYSTEM = """You are Roy's frozen task-agnostic information probe, separate from the Worker.
Compare the supplied root state before and after an information-flow rollout. Measure only new,
task-relevant, usable information: verified evidence, resolved uncertainty, detected contradictions,
test behavior, or an actionable correction. Do not require an answer hypothesis set and do not solve
the task anew. Paraphrase, unsupported confidence, and repeated facts have zero gain. Return JSON."""

    def __init__(self, llm: JsonLLM, max_tokens: int = 512) -> None:
        self.llm, self.max_tokens = llm, max_tokens

    def compare(
        self,
        before: AgentState,
        after: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        value = self.llm.call(
            "pairwise_information_probe",
            self.SYSTEM,
            {
                "benchmark": benchmark,
                "before_root_state": AgentHarness(before).execution_view(),
                "after_root_state": AgentHarness(after).execution_view(),
                "current_state_context": dict(state_context or {}),
                "required_schema": {
                    "information_gain": "number in [0, 1]",
                    "before_uncertainty": "number in [0, 1]",
                    "after_uncertainty": "number in [0, 1]",
                    "new_information": ["concise task-relevant facts or behaviors"],
                    "rationale": "brief comparison grounded only in supplied states",
                },
            },
            max_tokens=self.max_tokens,
        )
        return {
            "information_gain": min(1.0, max(0.0, float(value.get("information_gain", 0.0)))),
            "before_uncertainty": min(
                1.0, max(0.0, float(value.get("before_uncertainty", 1.0)))
            ),
            "after_uncertainty": min(
                1.0, max(0.0, float(value.get("after_uncertainty", 1.0)))
            ),
            "new_information": [str(item) for item in value.get("new_information", [])],
            "rationale": str(value.get("rationale", "")),
        }


def parse_json_object(content: str) -> Dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start and (start != 0 or end != len(text) - 1):
        candidates.append(text[start : end + 1])
    first_error: json.JSONDecodeError | None = None
    value: Any = None
    for candidate in candidates:
        repaired = _escape_invalid_json_backslashes(candidate)
        attempts = (repaired, candidate) if repaired != candidate else (candidate,)
        for attempt in attempts:
            try:
                value = json.loads(attempt)
                break
            except json.JSONDecodeError as error:
                first_error = first_error or error
        else:
            continue
        break
    else:
        assert first_error is not None
        raise first_error
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _escape_invalid_json_backslashes(text: str) -> str:
    """Repair common LaTeX backslashes while leaving valid JSON escapes unchanged."""
    output: List[str] = []
    index = 0
    inside_string = False
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    latex_commands_with_json_escape_prefix = (
        "begin", "boxed", "boldsymbol", "frac", "neq", "not", "right", "text",
    )
    while index < len(text):
        character = text[index]
        if character == '"':
            inside_string = not inside_string
            output.append(character)
            index += 1
            continue
        if inside_string and character == "\\":
            following = text[index + 1] if index + 1 < len(text) else ""
            suffix = text[index + 1 :]
            if suffix.startswith(latex_commands_with_json_escape_prefix):
                output.append("\\\\")
                index += 1
                continue
            if following in '"\\/bfnrt':
                output.extend((character, following))
                index += 2
                continue
            if (
                following == "u"
                and index + 5 < len(text)
                and all(item in hexadecimal for item in text[index + 2 : index + 6])
            ):
                output.append(text[index : index + 6])
                index += 6
                continue
            output.append("\\\\")
            index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output)
