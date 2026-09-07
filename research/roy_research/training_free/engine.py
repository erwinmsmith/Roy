from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping

from .harness import AGENT_HARNESS_SCHEMA_VERSION, AgentHarness, AgentHarnessConfig
from .llm import (
    CallAudit,
    CandidateXRealizer,
    ChannelizerModel,
    CompletionClient,
    GlobalSelector,
    JsonLLM,
    SemanticInformationJudge,
    WorkerModel,
)
from .mia import (
    MIAObjectiveEvaluator,
    PrecisionLogDetObjectiveEvaluator,
    SemanticInformationLandscape,
)
from .matrix import (
    A2AExecutor,
    BeamCoordinateMatrixSearch,
    InformationMatrix,
    MatrixSearchResult,
    expand_matrix,
)
from .tools import TaskToolRegistry, ToolAudit
from .trajectory import (
    DependencyRecord,
    StateCheckpoint,
    TopologyDrift,
    TrajectoryEvent,
    committed_dependency_records,
    topology_drift,
)
from .types import (
    AgentState,
    AgentStatus,
    BenchmarkTask,
    CandidateDependency,
    CandidateGraph,
    CandidateNode,
    RealizedSubgraph,
    ResultState,
)


@dataclass(frozen=True)
class TrainingFreeConfig:
    maximum_agents: int = 5
    maximum_candidates: int = 10
    maximum_selected_subgraphs: int = 3
    maximum_nodes_per_subgraph: int = 2
    matrix_levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    matrix_beam_width: int = 3
    matrix_iterations: int = 3
    maximum_matrix_evaluations: int = 20
    communication_rounds: int = 2
    maximum_organization_rounds: int = 4
    inbound_token_budget: int = 4000
    information_gain_epsilon: float = 0.02
    finish_uncertainty: float = 0.15
    activation_threshold: float = 0.05
    worker_max_tokens: int = 8192
    candidate_worker_max_tokens: int = 16_384
    candidate_worker_thinking: str = "enabled"
    result_reconciler_max_tokens: int = 256
    selector_max_tokens: int = 1024
    candidate_realizer_max_tokens: int = 16_384
    channelizer_max_tokens: int = 768
    semantic_judge_max_tokens: int = 4096
    mia_path_horizon: int = 3
    matrix_objective: Literal["scalar_reach", "precision_logdet"] = "scalar_reach"
    precision_dimensions: int = 4
    precision_floor: float = 1e-3
    candidate_thinking: str = "enabled"
    available_tools: tuple[str, ...] = ("symbolic_math", "python", "public_tests")
    maximum_tool_rounds: int = 2
    maximum_tool_calls_per_worker_call: int = 3
    tool_timeout_seconds: int = 5
    hard_dependency_minimum: float = 0.5

    def __post_init__(self) -> None:
        if self.maximum_agents < 1 or self.maximum_nodes_per_subgraph < 1:
            raise ValueError("agent limits must be positive")
        if (
            self.communication_rounds < 1
            or self.mia_path_horizon < 1
            or self.maximum_organization_rounds < 1
        ):
            raise ValueError(
                "communication rounds, organization rounds, and MIA path horizon must be positive"
            )
        if self.maximum_matrix_evaluations < self.matrix_beam_width:
            raise ValueError("matrix evaluation budget must cover the beam")
        if self.maximum_tool_rounds < 0 or self.maximum_tool_calls_per_worker_call < 0:
            raise ValueError("tool loop limits cannot be negative")
        if not 0 <= self.hard_dependency_minimum <= 1:
            raise ValueError("hard dependency minimum must be in [0, 1]")
        if self.matrix_objective not in {"scalar_reach", "precision_logdet"}:
            raise ValueError("matrix_objective must be scalar_reach or precision_logdet")
        if self.precision_dimensions < 1:
            raise ValueError("precision_dimensions must be positive")
        if not 0.0 < self.precision_floor <= 1.0:
            raise ValueError("precision_floor must be in (0, 1]")


@dataclass
class RoundRecord:
    round_index: int
    state_before_checkpoint_id: str
    search_state_checkpoint_id: str
    committed_checkpoint_id: str | None
    information_measure: str
    information_state_before: Dict[str, Any]
    semantic_landscape: Dict[str, Any]
    reference_mia_objective: float
    candidate_graph: Dict[str, Any]
    candidate_calculation_list: List[List[str]]
    baseline: MatrixSearchResult
    frontiers: Dict[str, MatrixSearchResult]
    realized_candidates: Dict[str, RealizedSubgraph]
    provisional_agents: Dict[str, Dict[str, AgentState]]
    rejected_candidates: Dict[str, str]
    selected_subgraph_id: str | None
    expansion_gain: float
    selected_information_gain: float
    transition_kind: str
    topology_drift: TopologyDrift | None
    committed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "state_before_checkpoint_id": self.state_before_checkpoint_id,
            "search_state_checkpoint_id": self.search_state_checkpoint_id,
            "committed_checkpoint_id": self.committed_checkpoint_id,
            "information_measure": self.information_measure,
            "information_state_before": self.information_state_before,
            "semantic_landscape": self.semantic_landscape,
            "reference_mia_objective": self.reference_mia_objective,
            "candidate_graph": self.candidate_graph,
            "candidate_calculation_list": self.candidate_calculation_list,
            "baseline": self.baseline.to_dict(),
            "frontiers": {key: value.to_dict() for key, value in self.frontiers.items()},
            "realized_candidates": {
                key: value.to_dict() for key, value in self.realized_candidates.items()
            },
            "provisional_agents": {
                subgraph_id: {
                    agent_id: agent.to_dict() for agent_id, agent in agents.items()
                }
                for subgraph_id, agents in self.provisional_agents.items()
            },
            "rejected_candidates": self.rejected_candidates,
            "selected_subgraph_id": self.selected_subgraph_id,
            "expansion_gain": self.expansion_gain,
            "selected_information_gain": self.selected_information_gain,
            "predicted_mia_gain": self.selected_information_gain,
            "transition_kind": self.transition_kind,
            "topology_drift": self.topology_drift.to_dict() if self.topology_drift else None,
            "committed": self.committed,
        }


@dataclass(frozen=True)
class TransitionDecision:
    subgraph_id: str | None
    information_gain: float
    expansion_advantage: float
    kind: Literal["expand", "reorganize", "stop"]
    commit: bool
    structural_contraction: bool


def select_transition(
    baseline: MatrixSearchResult,
    frontiers: Mapping[str, MatrixSearchResult],
    epsilon: float,
) -> TransitionDecision:
    best_candidate_id = max(frontiers, key=lambda key: frontiers[key].score, default=None)
    best_candidate_score = (
        frontiers[best_candidate_id].score if best_candidate_id is not None else baseline.score
    )
    expansion_advantage = best_candidate_score - baseline.score
    choose_expansion = best_candidate_id is not None and expansion_advantage > epsilon
    selected_id = best_candidate_id if choose_expansion else None
    selected_score = frontiers[selected_id].score if selected_id is not None else baseline.score
    structural_contraction = (
        selected_id is None
        and selected_score >= -1e-12
        and (
            baseline.matrix.total_capacity()
            < baseline.initial_matrix.total_capacity() - 1e-12
            or baseline.matrix.positive_edge_count()
            < baseline.initial_matrix.positive_edge_count()
        )
    )
    if selected_score <= epsilon and not structural_contraction:
        kind: Literal["expand", "reorganize", "stop"] = "stop"
    else:
        kind = "expand" if selected_id is not None else "reorganize"
    return TransitionDecision(
        subgraph_id=selected_id,
        information_gain=selected_score,
        expansion_advantage=expansion_advantage,
        kind=kind,
        commit=kind != "stop",
        structural_contraction=structural_contraction,
    )


@dataclass
class TrainingFreeRun:
    task_id: str
    benchmark: str
    final_answer: str
    final_agents: Dict[str, AgentState]
    final_matrix: InformationMatrix
    rounds: List[RoundRecord]
    stop_reason: str
    call_audit: CallAudit
    tool_audit: ToolAudit
    checkpoints: List[StateCheckpoint]
    dependency_ledger: List[DependencyRecord]
    event_ledger: List[TrajectoryEvent]
    cumulative_information_gain: float
    harness_config: AgentHarnessConfig
    matrix_objective: str
    semantic_search_mode: str

    @property
    def initial_root_answer(self) -> str:
        checkpoint = next(item for item in self.checkpoints if item.phase == "search_state")
        return str(checkpoint.agents["A0"]["result"]["candidate_answer"])

    def organization_summary(self) -> Dict[str, int]:
        return {
            "rounds": len(self.rounds),
            "candidates_proposed": sum(
                len(record.candidate_graph.get("nodes", [])) for record in self.rounds
            ),
            "candidate_subgraphs_realized": sum(
                len(record.realized_candidates) for record in self.rounds
            ),
            "candidate_subgraphs_rejected": sum(
                len(record.rejected_candidates) for record in self.rounds
            ),
            "committed_expansions": sum(
                record.committed and record.transition_kind == "expand" for record in self.rounds
            ),
            "committed_reorganizations": sum(
                record.committed and record.transition_kind == "reorganize"
                for record in self.rounds
            ),
            "terminal_stops": sum(
                record.transition_kind == "stop" for record in self.rounds
            ),
            "committed_derivation_dependencies": sum(
                dependency.relation == "derivation" for dependency in self.dependency_ledger
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 5,
            "method": "training_free_information_flow_search",
            "search_architecture": {
                "semantic_judge": "once_per_organization_round",
                "candidate_selector_scope": "new_candidates_only",
                "candidate_calculation_policy": "selector_fixed_sparse_list",
                "judge_scope": "all_committed_agents_including_dormant_plus_selected_candidates",
                "semantic_search_mode": self.semantic_search_mode,
                "agent_x_policy": (
                    "configure_once_reuse_across_items"
                    if self.semantic_search_mode == "prospective_fixed_x"
                    else "current_item_configuration"
                ),
                "matrix_objective": self.matrix_objective,
                "candidate_matrix_llm_rollouts": 0,
                "execution_policy": "winner_only",
                "dormant_policy": "retained_judged_not_executed_until_matrix_reactivation",
                "round_policy": "matrix_gain_or_cost_contraction_with_hard_cap",
                "state_transition_policy": "initialize_once_then_winner_execution_only",
            },
            "agent_harness": {
                "schema_version": AGENT_HARNESS_SCHEMA_VERSION,
                "config": asdict(self.harness_config),
                "immutable_contract_fields": [
                    "agent_id", "parent_id", "objective", "role", "tools",
                    "expected_output", "stop_condition", "memory_namespace",
                    "inherited_memory_refs", "original_task_sha256", "static_context_sha256",
                ],
                "runtime_state_fields": [
                    "received_messages", "private_memory_entries", "result", "status",
                ],
            },
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "initial_root_answer": self.initial_root_answer,
            "final_answer": self.final_answer,
            "final_agents": {key: value.to_dict() for key, value in self.final_agents.items()},
            "final_matrix": self.final_matrix.to_dict(),
            "rounds": [record.to_dict() for record in self.rounds],
            "stop_reason": self.stop_reason,
            "call_audit": self.call_audit.to_dict(),
            "tool_audit": self.tool_audit.to_dict(),
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "dependency_ledger": [dependency.to_dict() for dependency in self.dependency_ledger],
            "event_ledger": [event.to_dict() for event in self.event_ledger],
            "organization_summary": self.organization_summary(),
            "cumulative_information_gain": self.cumulative_information_gain,
            "cumulative_predicted_mia_gain": self.cumulative_information_gain,
            "matrix_trajectory": [
                checkpoint.matrix for checkpoint in self.checkpoints
                if checkpoint.phase in ("initial", "committed", "terminal")
            ],
            "agent_basis_trajectory": [
                checkpoint.agents for checkpoint in self.checkpoints
                if checkpoint.phase in ("initial", "committed", "terminal")
            ],
        }


@dataclass
class SingleAgentRun:
    task_id: str
    benchmark: str
    final_agent: AgentState
    call_audit: CallAudit
    tool_audit: ToolAudit
    harness_config: AgentHarnessConfig

    @property
    def final_answer(self) -> str:
        return self.final_agent.result.candidate_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 5,
            "method": "single_agent_direct",
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "initial_root_answer": self.final_answer,
            "final_answer": self.final_answer,
            "final_agents": {"A0": self.final_agent.to_dict()},
            "final_matrix": InformationMatrix.zero(["A0"]).to_dict(),
            "rounds": [],
            "stop_reason": "single_execution_complete",
            "call_audit": self.call_audit.to_dict(),
            "tool_audit": self.tool_audit.to_dict(),
            "checkpoints": [],
            "dependency_ledger": [],
            "event_ledger": [],
            "organization_summary": {
                "rounds": 0,
                "candidates_proposed": 0,
                "candidate_subgraphs_realized": 0,
                "candidate_subgraphs_rejected": 0,
                "committed_expansions": 0,
                "committed_reorganizations": 0,
                "terminal_stops": 0,
                "committed_derivation_dependencies": 0,
            },
            "cumulative_information_gain": 0.0,
            "matrix_trajectory": [InformationMatrix.zero(["A0"]).to_dict()],
            "agent_basis_trajectory": [{"A0": self.final_agent.to_dict()}],
            "agent_harness": {
                "schema_version": AGENT_HARNESS_SCHEMA_VERSION,
                "config": asdict(self.harness_config),
            },
        }


@dataclass
class ContinualBenchmarkState:
    """Persistent organization carried between items of one ordered benchmark episode."""

    benchmark: str
    runtime_fingerprint: str
    agents: Dict[str, AgentState]
    matrix: InformationMatrix
    lineage_dependencies: List[DependencyRecord]
    next_round_index: int
    items_completed: int
    task_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "runtime_fingerprint": self.runtime_fingerprint,
            "agents": {key: value.to_dict() for key, value in self.agents.items()},
            "matrix": self.matrix.to_dict(),
            "lineage_dependencies": [item.to_dict() for item in self.lineage_dependencies],
            "next_round_index": self.next_round_index,
            "items_completed": self.items_completed,
            "task_ids": list(self.task_ids),
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        available_tools: List[str],
    ) -> "ContinualBenchmarkState":
        payload = {key: item for key, item in value.items() if key != "fingerprint"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if value.get("schema_version") != 1 or value.get("fingerprint") != expected:
            raise ValueError("invalid continual benchmark state fingerprint or schema")
        raw_agents = value.get("agents")
        raw_matrix = value.get("matrix")
        if not isinstance(raw_agents, Mapping) or not isinstance(raw_matrix, Mapping):
            raise ValueError("continual benchmark state is missing agents or matrix")
        agents: Dict[str, AgentState] = {}
        for agent_id, raw in raw_agents.items():
            if not isinstance(raw, Mapping) or not isinstance(raw.get("context"), Mapping):
                raise ValueError(f"invalid continual Agent state: {agent_id}")
            agents[str(agent_id)] = AgentState.from_dict(
                raw,
                original_task=str(raw["context"].get("original_task", "")),
                available_tools=available_tools,
                expected_agent_id=str(agent_id),
            )
        matrix = InformationMatrix(
            [str(item) for item in raw_matrix.get("agent_ids", [])],
            [[float(weight) for weight in row] for row in raw_matrix.get("values", [])],
        )
        matrix.validate()
        if set(matrix.agent_ids) != set(agents):
            raise ValueError("continual matrix basis does not match persistent Agents")
        dependencies = [
            DependencyRecord(**dict(item))
            for item in value.get("lineage_dependencies", [])
        ]
        if any(item.relation != "derivation" for item in dependencies):
            raise ValueError("continual state may retain only role-lineage dependencies")
        next_round_index = int(value.get("next_round_index", 0))
        items_completed = int(value.get("items_completed", 0))
        task_ids = [str(item) for item in value.get("task_ids", [])]
        if next_round_index < 0 or items_completed < 0 or len(task_ids) != items_completed:
            raise ValueError("continual state has invalid episode counters")
        return cls(
            benchmark=str(value.get("benchmark", "")),
            runtime_fingerprint=str(value.get("runtime_fingerprint", "")),
            agents=agents,
            matrix=matrix,
            lineage_dependencies=dependencies,
            next_round_index=next_round_index,
            items_completed=items_completed,
            task_ids=task_ids,
        )


class RoyTrainingFreeEngine:
    def __init__(
        self,
        worker_client: CompletionClient,
        candidate_client: CompletionClient | None = None,
        *,
        config: TrainingFreeConfig | None = None,
        code_sandbox_prefix: List[str] | None = None,
        semantic_landscape_factory: Callable[
            [str, Mapping[str, AgentState], Mapping[str, Any]], SemanticInformationLandscape
        ] | None = None,
    ) -> None:
        self.config = config or TrainingFreeConfig()
        self.worker_model = str(getattr(worker_client, "model", "unknown"))
        self.candidate_model = str(getattr(candidate_client or worker_client, "model", "unknown"))
        self.worker_endpoint = str(getattr(worker_client, "base_url", type(worker_client).__name__))
        self.candidate_endpoint = str(getattr(
            candidate_client or worker_client,
            "base_url",
            type(candidate_client or worker_client).__name__,
        ))
        self.audit = CallAudit()
        worker_llm = JsonLLM(worker_client, self.audit)
        candidate_llm = JsonLLM(candidate_client or worker_client, self.audit)
        self.worker = WorkerModel(
            worker_llm,
            self.config.worker_max_tokens,
            self.config.maximum_tool_rounds,
            self.config.maximum_tool_calls_per_worker_call,
            self.config.result_reconciler_max_tokens,
        )
        self.candidate_worker = WorkerModel(
            candidate_llm,
            self.config.candidate_worker_max_tokens,
            self.config.maximum_tool_rounds,
            self.config.maximum_tool_calls_per_worker_call,
            self.config.result_reconciler_max_tokens,
        )
        self.selector = GlobalSelector(worker_llm, self.config.selector_max_tokens)
        self.realizer = CandidateXRealizer(
            candidate_llm,
            self.config.candidate_realizer_max_tokens,
            self.config.candidate_thinking,
        )
        self.channelizer = ChannelizerModel(worker_llm, self.config.channelizer_max_tokens)
        self.semantic_judge = SemanticInformationJudge(
            worker_llm,
            self.config.semantic_judge_max_tokens,
            (
                self.config.precision_dimensions
                if self.config.matrix_objective == "precision_logdet" else 0
            ),
        )
        self.code_sandbox_prefix = list(code_sandbox_prefix or [])
        self.semantic_landscape_factory = semantic_landscape_factory
        self.tool_registry: TaskToolRegistry | None = None
        self.available_tools = list(self.config.available_tools)

    def run(
        self,
        task: BenchmarkTask,
        *,
        initial_agents: Mapping[str, AgentState] | None = None,
        initial_matrix: InformationMatrix | None = None,
        initial_dependencies: List[DependencyRecord] | None = None,
        round_index_offset: int = 0,
        prospective_semantic_search: bool = False,
    ) -> TrainingFreeRun:
        if round_index_offset < 0:
            raise ValueError("round_index_offset cannot be negative")
        self._configure_task(task)
        if initial_agents is None:
            root = self._root_agent(task)
            agents: Dict[str, AgentState] = {root.agent_id: root}
            matrix = InformationMatrix.zero(agents)
            dependencies: List[DependencyRecord] = []
            previously_dormant_agent_ids: List[str] = []
            initialization_kind = "state_initialized"
        else:
            if initial_matrix is None:
                raise ValueError("continual Agents require an initial information matrix")
            previously_dormant_agent_ids = sorted(
                agent_id for agent_id, agent in initial_agents.items()
                if agent.status == AgentStatus.DORMANT
            )
            agents = self._rebind_continual_agents(initial_agents, task)
            matrix = initial_matrix.clone()
            matrix.validate()
            if set(matrix.agent_ids) != set(agents):
                raise ValueError("continual matrix basis does not match rebound Agents")
            dependencies = copy.deepcopy(initial_dependencies or [])
            initialization_kind = "continual_state_rebound"
        records: List[RoundRecord] = []
        initial_round = round_index_offset - 1
        events = [TrajectoryEvent(
            "event-0", initial_round, initialization_kind,
            {
                "root_id": "A0",
                "reused_agent_ids": sorted(agents) if initial_agents is not None else [],
                "previously_dormant_agent_ids": previously_dormant_agent_ids,
                "dormant_agent_x_included_in_current_item_judge": bool(
                    previously_dormant_agent_ids
                ),
                "agent_x_rebuilt": False,
                "semantic_search_mode": (
                    "prospective_fixed_x" if prospective_semantic_search else "executed_state"
                ),
                "task_specific_results_reset": initial_agents is not None,
                "private_memory_retained": initial_agents is not None,
            },
        )]
        checkpoints = [StateCheckpoint.capture(
            f"checkpoint-{round_index_offset}-initial", None, initial_round, "initial", agents, matrix,
            dependencies, None, events,
        )]
        last_checkpoint_id = checkpoints[-1].checkpoint_id
        cumulative_information_gain = 0.0
        stop_reason = "maximum_organization_rounds"

        for local_round_index in range(self.config.maximum_organization_rounds):
            round_index = round_index_offset + local_round_index
            state_before_checkpoint_id = last_checkpoint_id
            candidate_graph, executed_parent_ids = self._prepare_parents_and_collect_candidates(
                agents, matrix, dependencies, events, task.benchmark, round_index,
                execute_parents=local_round_index == 0,
                execute_parent_ids=(
                    {"A0"} if prospective_semantic_search and local_round_index == 0 else None
                ),
                proposal_parent_ids=(
                    {"A0"}
                    if prospective_semantic_search and local_round_index == 0
                    else None
                ),
            )
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index,
                "parent_states_initialized" if local_round_index == 0 else "parent_states_reused",
                {
                    "executed_agent_ids": executed_parent_ids,
                    "reused_agent_ids": (
                        sorted(agents)
                        if initial_agents is not None or local_round_index > 0 else []
                    ),
                    "matrix": matrix.to_dict(),
                },
            ))
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index, "step_candidates_proposed",
                {"candidate_graph": candidate_graph.subset(candidate_graph.nodes)},
                scope="audit",
            ))

            room = self.config.maximum_agents - len(agents)
            maximum_nodes = min(self.config.maximum_nodes_per_subgraph, max(0, room))
            selected = [] if maximum_nodes == 0 or not candidate_graph.nodes else self.selector.select(
                candidate_graph,
                agents,
                maximum_subgraphs=self.config.maximum_selected_subgraphs,
                maximum_nodes=maximum_nodes,
                organization_context={
                    "current_matrix": matrix.to_dict(),
                    "dependency_state": [item.to_dict() for item in dependencies[-32:]],
                    "recent_history": [item.to_dict() for item in events[-16:]],
                },
            )
            selected_candidate_ids = sorted({
                candidate_id for group in selected for candidate_id in group
            })
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index, "candidate_calculation_list_fixed",
                {
                    "selected_subgraphs": selected,
                    "selected_candidate_ids": selected_candidate_ids,
                    "omitted_candidate_ids": sorted(
                        set(candidate_graph.nodes) - set(selected_candidate_ids)
                    ),
                    "selector_scope": "new_candidates_only",
                    "committed_agents_are_not_filtered": True,
                },
                scope="audit",
            ))

            realized: Dict[str, RealizedSubgraph] = {}
            provisional: Dict[str, Dict[str, AgentState]] = {}
            rejected: Dict[str, str] = {}
            for index, candidate_ids in enumerate(selected):
                subgraph_id = f"round-{round_index}-subgraph-{index}"
                try:
                    overlap = set(candidate_ids) & set(agents)
                    if overlap:
                        raise ValueError(f"candidate ids collide with committed agents: {sorted(overlap)}")
                    candidate = self.realizer.realize(
                        subgraph_id, candidate_ids, candidate_graph, agents,
                        benchmark=task.benchmark,
                        original_task=task.instruction,
                        public_tests=task.public_tests,
                        available_tools=self.available_tools,
                        persistent=prospective_semantic_search,
                        organization_context={
                            "current_matrix": matrix.to_dict(),
                            "dependency_state": [item.to_dict() for item in dependencies[-32:]],
                            "recent_history": [item.to_dict() for item in events[-16:]],
                        },
                    )
                    provisional_agents = (
                        copy.deepcopy(candidate.agents)
                        if prospective_semantic_search
                        else self._provisional_execute(
                            candidate, candidate_graph, agents, task.benchmark,
                        )
                    )
                    realized[subgraph_id] = candidate
                    provisional[subgraph_id] = provisional_agents
                    events.append(TrajectoryEvent(
                        f"event-{len(events)}", round_index, "candidate_x_realized",
                        {
                            "subgraph_id": subgraph_id,
                            "candidate_ids": candidate_ids,
                            "configuration_built_once": True,
                            "executed_before_judge": not prospective_semantic_search,
                        },
                        scope="counterfactual",
                    ))
                except (TypeError, ValueError, KeyError) as error:
                    rejected[subgraph_id] = f"{type(error).__name__}: {error}"
                    events.append(TrajectoryEvent(
                        f"event-{len(events)}", round_index, "candidate_x_rejected",
                        {"subgraph_id": subgraph_id, "error": rejected[subgraph_id]},
                        scope="counterfactual",
                    ))
            # Candidate X is counterfactual until selected by the matrix search.
            # If every selected proposal violates the harness contract, preserve
            # their rejection audit and continue with the committed agent basis.
            # The baseline frontier can then reorganize existing agents or emit
            # a normal terminal transition instead of discarding a valid Root
            # result as a task-execution failure.

            state_context = {
                "current_matrix": matrix.to_dict(),
                "dependency_state": [item.to_dict() for item in dependencies[-32:]],
                "recent_history": _recent_committed_history(events),
            }
            judge_agents = dict(agents)
            for candidate_agents in provisional.values():
                overlap = set(judge_agents) & set(candidate_agents)
                if overlap:
                    raise ValueError(
                        f"candidate subgraphs must be disjoint for one-shot Judge: {sorted(overlap)}"
                    )
                judge_agents.update(candidate_agents)
            landscape = self._estimate_semantic_landscape(
                task.benchmark, judge_agents, state_context,
                prospective=prospective_semantic_search,
            )
            information_state_before = {
                "estimator": "mia_semantic_landscape",
                "root_uncertainty": landscape.root_uncertainty,
                "semantic_landscape_revision": landscape.revision,
                "committed_agent_ids": sorted(agents),
                "previously_dormant_agent_ids": previously_dormant_agent_ids,
                "selected_candidate_ids": selected_candidate_ids,
                "semantic_search_mode": (
                    "prospective_fixed_x" if prospective_semantic_search else "executed_state"
                ),
            }
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index, "semantic_landscape_estimated",
                {
                    "revision": landscape.revision,
                    "agent_ids": landscape.agent_ids,
                    "committed_agent_ids": sorted(agents),
                    "previously_dormant_agent_ids": previously_dormant_agent_ids,
                    "selected_candidate_ids": selected_candidate_ids,
                    "pairwise_scope": (
                        "all_committed_agents_including_dormant_plus_selected_candidates"
                    ),
                    "semantic_search_mode": (
                        "prospective_fixed_x" if prospective_semantic_search else "executed_state"
                    ),
                    "judge_calls_this_round": 1,
                },
                scope="audit",
            ))
            search_state_checkpoint_id = f"checkpoint-{round_index}-search-state"
            checkpoints.append(StateCheckpoint.capture(
                search_state_checkpoint_id, state_before_checkpoint_id, round_index,
                "search_state", agents, matrix, dependencies,
                information_state_before, events,
            ))
            if self.config.matrix_objective == "precision_logdet":
                evaluator = PrecisionLogDetObjectiveEvaluator(
                    landscape,
                    path_horizon=self.config.mia_path_horizon,
                    reference_matrix=matrix,
                    precision_floor=self.config.precision_floor,
                )
            else:
                evaluator = MIAObjectiveEvaluator(
                    landscape,
                    path_horizon=self.config.mia_path_horizon,
                    reference_matrix=matrix,
                )
            optimizer = BeamCoordinateMatrixSearch(
                evaluator,
                levels=self.config.matrix_levels,
                beam_width=self.config.matrix_beam_width,
                iterations=self.config.matrix_iterations,
                hard_minimum=self.config.hard_dependency_minimum,
                maximum_evaluations=self.config.maximum_matrix_evaluations,
            )
            baseline = optimizer.optimize(agents, matrix)
            frontiers: Dict[str, MatrixSearchResult] = {}
            for subgraph_id, new_agents in provisional.items():
                basis = {**agents, **new_agents}
                candidate = realized[subgraph_id]
                initial = expand_matrix(matrix, new_agents, candidate.dependencies)
                frontiers[subgraph_id] = optimizer.optimize(
                    basis, initial, candidate.dependencies,
                )
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index, "mia_matrix_search_completed",
                {
                    "baseline_evaluations": baseline.evaluations,
                    "frontier_evaluations": {
                        key: result.evaluations for key, result in frontiers.items()
                    },
                    "candidate_matrix_llm_rollouts": 0,
                },
                scope="audit",
            ))

            decision = select_transition(
                baseline, frontiers, self.config.information_gain_epsilon,
            )
            best_candidate_score = max(
                (result.score for result in frontiers.values()), default=baseline.score,
            )
            expansion_advantage = decision.expansion_advantage
            selected_id = decision.subgraph_id
            selected_result = frontiers[selected_id] if selected_id is not None else baseline
            selected_information_gain = decision.information_gain
            transition_kind = decision.kind
            commit = decision.commit
            committed_checkpoint_id: str | None = None
            drift: TopologyDrift | None = None

            if not commit:
                stop_reason = (
                    "root_confident"
                    if landscape.root_uncertainty <= self.config.finish_uncertainty
                    else "structural_saturation"
                )
                events.append(TrajectoryEvent(
                    f"event-{len(events)}", round_index, "trajectory_stopped",
                    {
                        "reason": stop_reason,
                        "baseline_predicted_mia_gain": baseline.score,
                        "best_candidate_predicted_mia_gain": best_candidate_score,
                    },
                ))
                self._apply_activation(agents, matrix, dependencies)
                committed_checkpoint_id = f"checkpoint-{round_index}-terminal"
                checkpoints.append(StateCheckpoint.capture(
                    committed_checkpoint_id, search_state_checkpoint_id, round_index,
                    "terminal", agents, matrix, dependencies,
                    information_state_before, events,
                ))
                last_checkpoint_id = committed_checkpoint_id
            else:
                previous_matrix = matrix
                if selected_id is None:
                    winning_agents = agents
                    selected_dependencies: List[CandidateDependency] = []
                else:
                    winning_agents = {**agents, **provisional[selected_id]}
                    selected_dependencies = realized[selected_id].dependencies
                    dependencies.extend(committed_dependency_records(
                        round_index, selected_id, provisional[selected_id],
                        realized[selected_id].dependencies,
                    ))
                winning_matrix = selected_result.matrix
                prospective_executed_agent_ids: List[str] = []
                if prospective_semantic_search:
                    winning_agents, prospective_executed_agent_ids = self._execute_prospective_winner(
                        winning_agents,
                        winning_matrix,
                        task.benchmark,
                        selected_dependencies,
                    )
                executor = A2AExecutor(
                    self.worker, self.channelizer,
                    benchmark=task.benchmark,
                    inbound_token_budget=self.config.inbound_token_budget,
                    communication_rounds=self.config.communication_rounds,
                )
                next_agents = executor.realize_once(
                    winning_agents, winning_matrix, tool_scope="committed",
                )
                events.append(TrajectoryEvent(
                    f"event-{len(events)}", round_index, "winner_matrix_executed",
                    {
                        "matrix": winning_matrix.to_dict(),
                        "selected_subgraph_id": selected_id,
                        "task_local_executed_agent_ids": prospective_executed_agent_ids,
                        "a2a_active_agent_ids": sorted(
                            winning_matrix.active_agent_ids(
                                minimum_weight=self.config.activation_threshold,
                            )
                        ),
                        "a2a_dormant_agent_ids": sorted(
                            set(winning_agents) - winning_matrix.active_agent_ids(
                                minimum_weight=self.config.activation_threshold,
                            )
                        ),
                        "executed_matrix_count": 1,
                    },
                ))
                self._apply_activation(next_agents, winning_matrix, dependencies)
                agents = next_agents
                matrix = winning_matrix
                cumulative_information_gain += selected_information_gain
                drift = topology_drift(previous_matrix, matrix)
                events.append(TrajectoryEvent(
                    f"event-{len(events)}", round_index, "organization_transition_committed",
                    {
                        "transition_kind": transition_kind,
                        "subgraph_id": selected_id,
                        "predicted_mia_gain": selected_information_gain,
                        "expansion_advantage": expansion_advantage,
                        "structural_contraction": decision.structural_contraction,
                        "topology_drift": drift.to_dict(),
                    },
                ))
                committed_checkpoint_id = f"checkpoint-{round_index}-committed"
                checkpoint_information_state = {
                    **information_state_before,
                    "state_changed_after_winner_execution": True,
                }
                checkpoints.append(StateCheckpoint.capture(
                    committed_checkpoint_id, search_state_checkpoint_id, round_index,
                    "committed", agents, matrix, dependencies,
                    checkpoint_information_state, events,
                ))
                last_checkpoint_id = committed_checkpoint_id

            record = RoundRecord(
                round_index=round_index,
                state_before_checkpoint_id=state_before_checkpoint_id,
                search_state_checkpoint_id=search_state_checkpoint_id,
                committed_checkpoint_id=committed_checkpoint_id,
                information_measure=(
                    "mia_precision_logdet" if self.config.matrix_objective == "precision_logdet"
                    else "mia_semantic_landscape"
                ),
                information_state_before=information_state_before,
                semantic_landscape=landscape.to_dict(),
                reference_mia_objective=evaluator.reference,
                candidate_graph=candidate_graph.subset(candidate_graph.nodes),
                candidate_calculation_list=selected,
                baseline=baseline,
                frontiers=frontiers,
                realized_candidates=realized,
                provisional_agents=provisional,
                rejected_candidates=rejected,
                selected_subgraph_id=selected_id,
                expansion_gain=expansion_advantage,
                selected_information_gain=selected_information_gain,
                transition_kind=transition_kind,
                topology_drift=drift,
                committed=commit,
            )
            records.append(record)

            if not commit:
                break

        return TrainingFreeRun(
            task_id=task.task_id,
            benchmark=task.benchmark,
            final_answer=agents["A0"].result.candidate_answer,
            final_agents=agents,
            final_matrix=matrix,
            rounds=records,
            stop_reason=stop_reason,
            call_audit=self.audit,
            tool_audit=self.tool_registry.audit,
            checkpoints=checkpoints,
            dependency_ledger=dependencies,
            event_ledger=events,
            cumulative_information_gain=cumulative_information_gain,
            harness_config=self.worker.harness_config,
            matrix_objective=self.config.matrix_objective,
            semantic_search_mode=(
                "prospective_fixed_x" if prospective_semantic_search else "executed_state"
            ),
        )

    def run_continual(
        self,
        task: BenchmarkTask,
        prior: ContinualBenchmarkState | None,
    ) -> tuple[TrainingFreeRun, ContinualBenchmarkState]:
        if prior is not None and prior.benchmark != task.benchmark:
            raise ValueError(
                f"cannot carry {prior.benchmark} organization into {task.benchmark}"
            )
        runtime_fingerprint = self._continual_runtime_fingerprint()
        if prior is not None and prior.runtime_fingerprint != runtime_fingerprint:
            raise ValueError("continual runtime/model config changed during an episode")
        run = self.run(
            task,
            initial_agents=None if prior is None else prior.agents,
            initial_matrix=None if prior is None else prior.matrix,
            initial_dependencies=None if prior is None else prior.lineage_dependencies,
            round_index_offset=0 if prior is None else prior.next_round_index,
            prospective_semantic_search=True,
        )
        lineage = [
            copy.deepcopy(item) for item in run.dependency_ledger
            if item.relation == "derivation"
        ]
        next_round = max(
            (record.round_index for record in run.rounds),
            default=(-1 if prior is None else prior.next_round_index - 1),
        ) + 1
        state = ContinualBenchmarkState(
            benchmark=task.benchmark,
            runtime_fingerprint=runtime_fingerprint,
            agents=copy.deepcopy(run.final_agents),
            matrix=run.final_matrix.clone(),
            lineage_dependencies=lineage,
            next_round_index=next_round,
            items_completed=(0 if prior is None else prior.items_completed) + 1,
            task_ids=[*([] if prior is None else prior.task_ids), task.task_id],
        )
        return run, state

    def _continual_runtime_fingerprint(self) -> str:
        return hashlib.sha256(json.dumps({
            "config": asdict(self.config),
            "worker_model": self.worker_model,
            "candidate_model": self.candidate_model,
            "worker_endpoint": self.worker_endpoint,
            "candidate_endpoint": self.candidate_endpoint,
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def run_direct(self, task: BenchmarkTask) -> SingleAgentRun:
        """Run the matched one-Agent baseline without organization search."""
        self._configure_task(task)
        root = self.worker.execute_root(
            self._root_agent(task), task.benchmark, tool_scope="committed",
        )
        root.status = AgentStatus.DONE
        return SingleAgentRun(
            task_id=task.task_id,
            benchmark=task.benchmark,
            final_agent=root,
            call_audit=self.audit,
            tool_audit=self.tool_registry.audit,
            harness_config=self.worker.harness_config,
        )

    def _configure_task(self, task: BenchmarkTask) -> None:
        self.tool_registry = TaskToolRegistry(
            task,
            code_sandbox_prefix=self.code_sandbox_prefix,
            timeout_seconds=self.config.tool_timeout_seconds,
        )
        configured_names = {item["name"] for item in self.tool_registry.catalog}
        self.available_tools = [
            name for name in self.config.available_tools if name in configured_names
        ]
        self.worker.configure_tools(self.tool_registry)
        self.candidate_worker.configure_tools(self.tool_registry)

    def _prepare_parents_and_collect_candidates(
        self,
        agents: Dict[str, AgentState],
        matrix: InformationMatrix,
        dependencies: List[DependencyRecord],
        events: List[TrajectoryEvent],
        benchmark: str,
        round_index: int,
        *,
        execute_parents: bool = False,
        execute_parent_ids: set[str] | None = None,
        proposal_parent_ids: set[str] | None = None,
    ) -> tuple[CandidateGraph, List[str]]:
        nodes: Dict[str, CandidateNode] = {}
        state_dependencies = dependencies
        candidate_dependencies: List[CandidateDependency] = []
        state_snapshot = copy.deepcopy(agents)
        executed: Dict[str, AgentState] = {}
        for agent_id in sorted(state_snapshot):
            agent = state_snapshot[agent_id]
            if agent.status in (AgentStatus.DORMANT, AgentStatus.FAILED, AgentStatus.WAITING):
                continue
            organization_context = {
                "current_matrix": matrix.to_dict(),
                "other_agents": {
                    other_id: AgentHarness(other).public_summary()
                    for other_id, other in state_snapshot.items() if other_id != agent_id
                },
                "dependency_state": [item.to_dict() for item in state_dependencies[-32:]],
                "recent_history": _recent_committed_history(events),
                "recent_search_history": [
                    item.to_dict() for item in events if item.scope != "committed"
                ][-16:],
            }
            proposal_agent = agent
            if execute_parents and (
                execute_parent_ids is None or agent_id in execute_parent_ids
            ):
                proposal_agent = (
                    self.worker.execute_root(agent, benchmark, tool_scope="committed")
                    if agent_id == "A0"
                    else self.worker.execute_persistent(
                        agent, benchmark, tool_scope="committed",
                    )
                )
                executed[agent_id] = proposal_agent
            remaining_capacity = max(0, self.config.maximum_agents - len(agents))
            if (
                remaining_capacity == 0
                or proposal_parent_ids is not None and agent_id not in proposal_parent_ids
            ):
                continue
            minimum_candidates = 0
            if remaining_capacity and agent_id == "A0":
                minimum_candidates = min(
                    3 if round_index == 0 else 1,
                    remaining_capacity,
                    self.config.maximum_candidates,
                )
            graph = self.worker.propose_candidates(
                proposal_agent,
                round_index=round_index,
                max_candidates=min(self.config.maximum_candidates, remaining_capacity),
                minimum_candidates=minimum_candidates,
                organization_context=organization_context,
            )
            collisions = set(nodes) & set(graph.nodes)
            if collisions:
                raise ValueError(f"parents proposed duplicate candidate ids: {sorted(collisions)}")
            nodes.update(graph.nodes)
            candidate_dependencies.extend(graph.dependencies)
        agents.update(executed)
        combined = CandidateGraph("GLOBAL", nodes, candidate_dependencies)
        combined.topological_order(set(nodes), hard_only=True)
        return combined, sorted(executed)

    def _rebind_continual_agents(
        self,
        persistent_agents: Mapping[str, AgentState],
        task: BenchmarkTask,
    ) -> Dict[str, AgentState]:
        rebound: Dict[str, AgentState] = {}
        for agent_id, persistent in persistent_agents.items():
            agent = copy.deepcopy(persistent)
            # X_i is configured exactly once. A new item supplies only an
            # ephemeral task input and fresh runtime result/message buffers.
            # Role, objective, tools, dependency preferences, memory namespace,
            # output contract, and stop condition remain the committed X_i.
            agent.context.original_task = task.instruction
            agent.context.public_tests = list(task.public_tests)
            agent.context.received_messages = []
            agent.tools = [tool for tool in agent.tools if tool in self.available_tools]
            agent.result = ResultState(unresolved=[agent.objective])
            agent.status = AgentStatus.READY
            AgentHarness(agent, self.tool_registry, self.worker.harness_config).validate()
            rebound[agent_id] = agent
        if "A0" not in rebound:
            raise ValueError("continual organization has no A0 root")
        return rebound

    def _root_agent(self, task: BenchmarkTask) -> AgentState:
        objective = (
            "Solve the original mathematical problem and return the final answer."
            if task.benchmark == "MATH"
            else "Implement the requested Python function so it passes the allowed public tests and hidden tests."
        )
        role = (
            "General mathematical reasoning agent responsible for the final answer."
            if task.benchmark == "MATH"
            else "General Python program-synthesis agent responsible for the final implementation."
        )
        return AgentHarness.create_root(
            original_task=task.instruction,
            objective=objective,
            role=role,
            tools=self.available_tools,
            expected_output="A correct boxed answer" if task.benchmark == "MATH" else "Complete Python code",
            stop_condition="A verified final answer is ready",
            public_tests=task.public_tests,
            tool_registry=self.tool_registry,
        ).state

    def _provisional_execute(
        self,
        candidate: RealizedSubgraph,
        graph: CandidateGraph,
        existing: Mapping[str, AgentState],
        benchmark: str,
    ) -> Dict[str, AgentState]:
        states = copy.deepcopy(candidate.agents)
        order = graph.topological_order(set(candidate.candidate_ids), hard_only=True)
        for agent_id in order:
            agent = states[agent_id]
            for edge in candidate.dependencies:
                if edge.kind != "hard" or edge.target != agent_id:
                    continue
                source = states.get(edge.source) or existing.get(edge.source)
                if source is None:
                    raise ValueError(f"hard dependency source {edge.source} is unavailable")
                dependency_message = json.dumps({
                    "source": edge.source,
                    "artifact": edge.artifact,
                    "result": asdict(source.result),
                }, ensure_ascii=False, sort_keys=True)
                AgentHarness(agent, self.tool_registry).receive_messages([dependency_message])
            states[agent_id] = self.candidate_worker.execute_local(
                agent, benchmark, thinking=self.config.candidate_worker_thinking,
            )
        return states

    def _execute_prospective_winner(
        self,
        agents: Mapping[str, AgentState],
        matrix: InformationMatrix,
        benchmark: str,
        candidate_dependencies: List[CandidateDependency],
    ) -> tuple[Dict[str, AgentState], List[str]]:
        """Execute only fixed X_i selected by the winning nonzero matrix."""
        states = copy.deepcopy(dict(agents))
        active = matrix.active_agent_ids(
            minimum_weight=self.config.activation_threshold,
        )
        executed: List[str] = []
        completed = {
            agent_id for agent_id in active
            if states[agent_id].result.candidate_answer.strip()
        }
        pending = {
            agent_id for agent_id in active
            if agent_id != "A0" and agent_id not in completed
        }
        if "A0" not in completed:
            states["A0"] = self.worker.execute_root(
                states["A0"], benchmark, tool_scope="committed",
            )
            completed.add("A0")
            executed.append("A0")

        while pending:
            progressed = False
            for agent_id in sorted(pending):
                hard_sources = {
                    edge.source for edge in candidate_dependencies
                    if edge.kind == "hard" and edge.target == agent_id and edge.source in active
                }
                if not hard_sources.issubset(completed):
                    continue
                agent = states[agent_id]
                messages = [
                    json.dumps({
                        "source": source,
                        "artifact": next(
                            edge.artifact for edge in candidate_dependencies
                            if edge.kind == "hard"
                            and edge.source == source and edge.target == agent_id
                        ),
                        "result": asdict(states[source].result),
                    }, ensure_ascii=False, sort_keys=True)
                    for source in sorted(hard_sources)
                ]
                if messages:
                    AgentHarness(
                        agent, self.tool_registry, self.worker.harness_config,
                    ).receive_messages(messages)
                states[agent_id] = self.worker.execute_persistent(
                    agent, benchmark, tool_scope="committed",
                )
                completed.add(agent_id)
                executed.append(agent_id)
                pending.remove(agent_id)
                progressed = True
            if not progressed:
                raise ValueError(
                    "winner contains an unsatisfied hard-dependency cycle or inactive source"
                )
        return states, executed

    def _apply_activation(
        self,
        agents: Dict[str, AgentState],
        matrix: InformationMatrix,
        dependencies: List[DependencyRecord],
    ) -> None:
        active_agent_ids = matrix.active_agent_ids(
            minimum_weight=self.config.activation_threshold,
        )
        for agent_id, agent in agents.items():
            if agent_id == "A0":
                agent.status = AgentStatus.DONE
            elif agent_id not in active_agent_ids:
                agent.status = AgentStatus.DORMANT
            elif agent.status != AgentStatus.FAILED:
                agent.status = AgentStatus.DONE
        for dependency in dependencies:
            target = agents.get(dependency.target)
            if target is None or dependency.status == "satisfied":
                continue
            dependency.status = "dormant" if target.status == AgentStatus.DORMANT else "active"

    def _estimate_semantic_landscape(
        self,
        benchmark: str,
        agents: Mapping[str, AgentState],
        state_context: Mapping[str, Any],
        *,
        prospective: bool = False,
    ) -> SemanticInformationLandscape:
        if self.semantic_landscape_factory is not None:
            return self.semantic_landscape_factory(benchmark, agents, state_context)
        return self.semantic_judge.estimate(
            agents, benchmark=benchmark, root_id="A0", state_context=state_context,
            prospective=prospective,
        )


def _recent_committed_history(events: List[TrajectoryEvent]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in events if item.scope == "committed"][-16:]
