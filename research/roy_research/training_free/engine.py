from __future__ import annotations

import copy
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
from .mia import MIAObjectiveEvaluator, SemanticInformationLandscape
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
    worker_max_tokens: int = 2048
    candidate_worker_max_tokens: int = 8192
    candidate_worker_thinking: str = "enabled"
    result_reconciler_max_tokens: int = 256
    selector_max_tokens: int = 1024
    candidate_realizer_max_tokens: int = 16_384
    channelizer_max_tokens: int = 768
    semantic_judge_max_tokens: int = 4096
    mia_path_horizon: int = 3
    candidate_thinking: str = "enabled"
    available_tools: tuple[str, ...] = ("symbolic_math", "python", "public_tests")
    maximum_tool_rounds: int = 2
    maximum_tool_calls_per_worker_call: int = 3
    tool_timeout_seconds: int = 5
    hard_dependency_minimum: float = 0.5

    def __post_init__(self) -> None:
        if self.maximum_agents < 1 or self.maximum_nodes_per_subgraph < 1:
            raise ValueError("agent limits must be positive")
        if self.communication_rounds < 1 or self.mia_path_horizon < 1:
            raise ValueError("communication rounds and MIA path horizon must be positive")
        if self.maximum_matrix_evaluations < self.matrix_beam_width:
            raise ValueError("matrix evaluation budget must cover the beam")
        if self.maximum_tool_rounds < 0 or self.maximum_tool_calls_per_worker_call < 0:
            raise ValueError("tool loop limits cannot be negative")
        if not 0 <= self.hard_dependency_minimum <= 1:
            raise ValueError("hard dependency minimum must be in [0, 1]")


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
    if selected_score <= epsilon:
        kind: Literal["expand", "reorganize", "stop"] = "stop"
    else:
        kind = "expand" if selected_id is not None else "reorganize"
    return TransitionDecision(
        subgraph_id=selected_id,
        information_gain=selected_score,
        expansion_advantage=expansion_advantage,
        kind=kind,
        commit=kind != "stop",
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

    @property
    def initial_root_answer(self) -> str:
        checkpoint = next(
            item for item in self.checkpoints
            if item.phase == "search_state" and item.round_index == 0
        )
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
                "matrix_objective": "pure_mia_functional",
                "candidate_matrix_llm_rollouts": 0,
                "execution_policy": "winner_only",
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
            worker_llm, self.config.semantic_judge_max_tokens,
        )
        self.code_sandbox_prefix = list(code_sandbox_prefix or [])
        self.semantic_landscape_factory = semantic_landscape_factory
        self.tool_registry: TaskToolRegistry | None = None
        self.available_tools = list(self.config.available_tools)

    def run(self, task: BenchmarkTask) -> TrainingFreeRun:
        self._configure_task(task)
        root = self._root_agent(task)
        agents: Dict[str, AgentState] = {root.agent_id: root}
        matrix = InformationMatrix.zero(agents)
        records: List[RoundRecord] = []
        dependencies: List[DependencyRecord] = []
        events = [TrajectoryEvent("event-0", -1, "state_initialized", {"root_id": "A0"})]
        checkpoints = [StateCheckpoint.capture(
            "checkpoint-0-initial", None, -1, "initial", agents, matrix,
            dependencies, None, events,
        )]
        last_checkpoint_id = checkpoints[-1].checkpoint_id
        cumulative_information_gain = 0.0
        stop_reason = "maximum_organization_rounds"

        for round_index in range(self.config.maximum_organization_rounds):
            state_before_checkpoint_id = last_checkpoint_id
            candidate_graph, executed_parent_ids = self._prepare_parents_and_collect_candidates(
                agents, matrix, dependencies, events, task.benchmark, round_index,
            )
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index,
                "parent_states_initialized" if round_index == 0 else "parent_states_reused",
                {
                    "executed_agent_ids": executed_parent_ids,
                    "reused_agent_ids": [] if round_index == 0 else sorted(agents),
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
                        organization_context={
                            "current_matrix": matrix.to_dict(),
                            "dependency_state": [item.to_dict() for item in dependencies[-32:]],
                            "recent_history": [item.to_dict() for item in events[-16:]],
                        },
                    )
                    provisional_agents = self._provisional_execute(
                        candidate, candidate_graph, agents, task.benchmark,
                    )
                    realized[subgraph_id] = candidate
                    provisional[subgraph_id] = provisional_agents
                    events.append(TrajectoryEvent(
                        f"event-{len(events)}", round_index, "candidate_x_realized",
                        {"subgraph_id": subgraph_id, "candidate_ids": candidate_ids},
                        scope="counterfactual",
                    ))
                except (TypeError, ValueError, KeyError) as error:
                    rejected[subgraph_id] = f"{type(error).__name__}: {error}"
                    events.append(TrajectoryEvent(
                        f"event-{len(events)}", round_index, "candidate_x_rejected",
                        {"subgraph_id": subgraph_id, "error": rejected[subgraph_id]},
                        scope="counterfactual",
                    ))
            if rejected:
                raise RuntimeError(
                    "selected candidate portfolio was not fully realized: "
                    + "; ".join(f"{key}={value}" for key, value in sorted(rejected.items()))
                )

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
            )
            information_state_before = {
                "estimator": "mia_semantic_landscape",
                "root_uncertainty": landscape.root_uncertainty,
                "semantic_landscape_revision": landscape.revision,
            }
            events.append(TrajectoryEvent(
                f"event-{len(events)}", round_index, "semantic_landscape_estimated",
                {
                    "revision": landscape.revision,
                    "agent_ids": landscape.agent_ids,
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
                else:
                    winning_agents = {**agents, **provisional[selected_id]}
                    dependencies.extend(committed_dependency_records(
                        round_index, selected_id, provisional[selected_id],
                        realized[selected_id].dependencies,
                    ))
                winning_matrix = selected_result.matrix
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
                information_measure="mia_semantic_landscape",
                information_state_before=information_state_before,
                semantic_landscape=landscape.to_dict(),
                reference_mia_objective=evaluator.reference,
                candidate_graph=candidate_graph.subset(candidate_graph.nodes),
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
        )

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
            if round_index == 0:
                proposal_agent = self.worker.execute_root(
                    agent, benchmark, tool_scope="committed",
                )
                executed[agent_id] = proposal_agent
            remaining_capacity = max(0, self.config.maximum_agents - len(agents))
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
                max_candidates=self.config.maximum_candidates,
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

    def _apply_activation(
        self,
        agents: Dict[str, AgentState],
        matrix: InformationMatrix,
        dependencies: List[DependencyRecord],
    ) -> None:
        hard_path_nodes = {
            node for edge in dependencies if edge.relation == "hard" and edge.status == "active"
            for node in (edge.source, edge.target)
        }
        for agent_id, agent in agents.items():
            if agent_id == "A0":
                agent.status = AgentStatus.DONE
            elif matrix.activation(agent_id) < self.config.activation_threshold \
                    and agent_id not in hard_path_nodes and not agent.result.unresolved:
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
    ) -> SemanticInformationLandscape:
        if self.semantic_landscape_factory is not None:
            return self.semantic_landscape_factory(benchmark, agents, state_context)
        return self.semantic_judge.estimate(
            agents, benchmark=benchmark, root_id="A0", state_context=state_context,
        )


def _recent_committed_history(events: List[TrajectoryEvent]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in events if item.scope == "committed"][-16:]
