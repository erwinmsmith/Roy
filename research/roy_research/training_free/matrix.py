from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, Tuple

from .information import InformationMeasure
from .llm import ChannelizerModel, WorkerModel
from .types import AgentState, CandidateDependency


@dataclass
class InformationMatrix:
    agent_ids: List[str]
    values: List[List[float]]

    @classmethod
    def zero(cls, agent_ids: Iterable[str]) -> "InformationMatrix":
        ids = list(agent_ids)
        return cls(ids, [[0.0 for _ in ids] for _ in ids])

    def clone(self) -> "InformationMatrix":
        return InformationMatrix(list(self.agent_ids), [list(row) for row in self.values])

    def weight(self, source: str, target: str) -> float:
        return self.values[self.agent_ids.index(source)][self.agent_ids.index(target)]

    def set_weight(self, source: str, target: str, value: float) -> None:
        self.values[self.agent_ids.index(source)][self.agent_ids.index(target)] = float(value)

    def inbound(self, target: str) -> float:
        column = self.agent_ids.index(target)
        return sum(row[column] for row in self.values)

    def activation(self, agent_id: str) -> float:
        index = self.agent_ids.index(agent_id)
        return sum(self.values[index]) + sum(row[index] for row in self.values)

    def validate(
        self,
        *,
        hard_dependencies: Sequence[CandidateDependency] = (),
        hard_minimum: float = 0.0,
    ) -> None:
        size = len(self.agent_ids)
        if len(set(self.agent_ids)) != size or len(self.values) != size:
            raise ValueError("information matrix has inconsistent agent basis")
        for row_index, row in enumerate(self.values):
            if len(row) != size:
                raise ValueError("information matrix must be square")
            for column_index, weight in enumerate(row):
                if not 0 <= weight <= 1:
                    raise ValueError("matrix weights must be in [0, 1]")
                if row_index == column_index and weight != 0:
                    raise ValueError("matrix diagonal must be zero")
        for target in self.agent_ids:
            if self.inbound(target) > 1.0 + 1e-9:
                raise ValueError(f"inbound capacity exceeded for {target}")
        for edge in hard_dependencies:
            if edge.kind != "hard" or edge.source not in self.agent_ids or edge.target not in self.agent_ids:
                continue
            if hard_minimum > 0 and self.weight(edge.source, edge.target) + 1e-9 < hard_minimum:
                raise ValueError(f"hard dependency channel below minimum: {edge.source}->{edge.target}")

    def key(self) -> Tuple[float, ...]:
        return tuple(round(value, 6) for row in self.values for value in row)

    def to_dict(self) -> Dict[str, object]:
        return {"agent_ids": list(self.agent_ids), "values": [list(row) for row in self.values]}


def expand_matrix(
    current: InformationMatrix,
    new_agents: Mapping[str, AgentState],
    dependencies: Sequence[CandidateDependency],
    *,
    initial_weight: float = 0.5,
) -> InformationMatrix:
    new_ids = list(new_agents)
    result = InformationMatrix.zero([*current.agent_ids, *new_ids])
    for source_index, source in enumerate(current.agent_ids):
        for target_index, target in enumerate(current.agent_ids):
            result.set_weight(source, target, current.values[source_index][target_index])

    proposed: Dict[str, Dict[str, float]] = {agent_id: {} for agent_id in result.agent_ids}
    for agent_id, agent in new_agents.items():
        for source, weight in agent.context.weighted_inputs.items():
            if source in result.agent_ids and source != agent_id and weight > 0:
                proposed[agent_id][source] = weight
        if agent.parent_id in result.agent_ids and agent.parent_id != agent_id:
            proposed[agent_id].setdefault(agent.parent_id, initial_weight)
            proposed[agent.parent_id].setdefault(agent_id, initial_weight)
    for edge in dependencies:
        if edge.source in result.agent_ids and edge.target in result.agent_ids:
            minimum = initial_weight if edge.kind == "hard" else initial_weight
            proposed[edge.target][edge.source] = max(proposed[edge.target].get(edge.source, 0.0), minimum)

    for target, sources in proposed.items():
        # W_t is the immutable upper-left block of an expansion. New edges may
        # consume only unused inbound capacity; the coordinate search can later
        # lower an old edge and raise a new one in separate admissible steps.
        additions = {
            source: weight for source, weight in sources.items()
            if result.weight(source, target) <= 0
        }
        available = max(0.0, 1.0 - result.inbound(target))
        total = sum(additions.values())
        scale = 0.0 if total <= 0 else min(1.0, available / total)
        for source, weight in additions.items():
            result.set_weight(source, target, weight * scale)
    result.validate()
    return result

def matrix_neighbors(
    matrix: InformationMatrix,
    levels: Sequence[float],
    hard_dependencies: Sequence[CandidateDependency],
    hard_minimum: float,
) -> List[InformationMatrix]:
    ordered_levels = sorted(set(float(value) for value in levels))
    neighbors: List[InformationMatrix] = []
    size = len(matrix.agent_ids)
    for source in range(size):
        for target in range(size):
            if source == target:
                continue
            current = matrix.values[source][target]
            exact_index = next((
                index for index, level in enumerate(ordered_levels)
                if abs(level - current) <= 1e-9
            ), None)
            if exact_index is not None:
                candidate_levels = [
                    ordered_levels[index]
                    for index in (exact_index - 1, exact_index + 1)
                    if 0 <= index < len(ordered_levels)
                ]
            else:
                lower = [level for level in ordered_levels if level < current]
                upper = [level for level in ordered_levels if level > current]
                candidate_levels = ([max(lower)] if lower else []) + ([min(upper)] if upper else [])
            for level in candidate_levels:
                candidate = matrix.clone()
                candidate.values[source][target] = level
                try:
                    candidate.validate(
                        hard_dependencies=hard_dependencies,
                        hard_minimum=hard_minimum,
                    )
                except ValueError:
                    continue
                neighbors.append(candidate)
    return neighbors


@dataclass
class MatrixSearchResult:
    initial_matrix: InformationMatrix
    matrix: InformationMatrix
    score: float
    evaluations: int
    information_observations: List[Dict[str, Any]]
    evaluated_matrices: List[Dict[str, Any]] = field(default_factory=list)
    admissible_level_matrices: int = 0

    def to_dict(self) -> Dict[str, object]:
        top_matrices = sorted(
            self.evaluated_matrices,
            key=lambda item: float(item["predicted_information_gain"]),
            reverse=True,
        )[:10]
        return {
            "initial_matrix": self.initial_matrix.to_dict(),
            "matrix": self.matrix.to_dict(), "score": self.score,
            "predicted_mia_gain": self.score,
            "delta_matrix": matrix_difference(self.initial_matrix, self.matrix).to_dict(),
            "evaluations": self.evaluations,
            "information_observations": self.information_observations,
            "search_space": {
                "admissible_level_matrices": self.admissible_level_matrices,
                "evaluated_count": len(self.evaluated_matrices),
                "top_matrices": [
                    {
                        "evaluation_index": item["evaluation_index"],
                        "matrix": item["matrix"],
                        "predicted_mia_gain": item["predicted_information_gain"],
                    }
                    for item in top_matrices
                ],
                "evaluated_matrices": self.evaluated_matrices,
            },
        }


class MatrixEvaluator(Protocol):
    @property
    def revision(self) -> str: ...

    def evaluate(
        self, agents: Mapping[str, AgentState], matrix: InformationMatrix,
    ) -> tuple[float, List[Dict[str, Any]]]: ...


class A2AExecutor:
    def __init__(
        self,
        worker: WorkerModel,
        channelizer: ChannelizerModel,
        *,
        benchmark: str,
        inbound_token_budget: int,
        communication_rounds: int,
    ) -> None:
        self.worker = worker
        self.channelizer = channelizer
        self.benchmark = benchmark
        self.inbound_token_budget = inbound_token_budget
        self.communication_rounds = communication_rounds

    def realize_once(
        self,
        agents: Mapping[str, AgentState],
        matrix: InformationMatrix,
        *,
        tool_scope: str = "counterfactual",
    ) -> Dict[str, AgentState]:
        states = copy.deepcopy(dict(agents))
        for _ in range(self.communication_rounds):
            incoming: Dict[str, List[str]] = {agent_id: [] for agent_id in matrix.agent_ids}
            for source in matrix.agent_ids:
                for target in matrix.agent_ids:
                    weight = matrix.weight(source, target)
                    if source == target or weight <= 0:
                        continue
                    message = self.channelizer.message(
                        states[source], states[target], weight, self.inbound_token_budget,
                    )
                    if message:
                        incoming[target].append(f"From {source} (capacity={weight:.2f}): {message}")
            next_states = copy.deepcopy(states)
            for target, messages in incoming.items():
                if messages:
                    next_states[target] = self.worker.update(
                        states[target], self.benchmark, messages, tool_scope=tool_scope,
                    )
            states = next_states
        return states


class SemanticRolloutEvaluator(A2AExecutor):
    """Optional empirical validator; the production MIA search does not use it."""

    def __init__(
        self,
        worker: WorkerModel,
        channelizer: ChannelizerModel,
        information_measure: InformationMeasure,
        *,
        benchmark: str,
        root_id: str,
        inbound_token_budget: int,
        communication_rounds: int,
        information_rollouts: int,
        state_context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            worker, channelizer, benchmark=benchmark,
            inbound_token_budget=inbound_token_budget,
            communication_rounds=communication_rounds,
        )
        self.information_measure = information_measure
        self.root_id = root_id
        self.information_rollouts = information_rollouts
        self.state_context = dict(state_context or {})
        self._realization_cache: Dict[
            Tuple[str, Tuple[float, ...], int], Dict[str, AgentState]
        ] = {}

    @property
    def revision(self) -> str:
        return self.information_measure.revision

    def _realize_for_evaluation(
        self,
        agents: Mapping[str, AgentState],
        matrix: InformationMatrix,
        rollout_index: int,
    ) -> Dict[str, AgentState]:
        agents_fingerprint = json.dumps(
            {key: value.to_dict() for key, value in agents.items()},
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_key = (agents_fingerprint, matrix.key(), rollout_index)
        if cache_key not in self._realization_cache:
            self._realization_cache[cache_key] = self.realize_once(agents, matrix)
        return copy.deepcopy(self._realization_cache[cache_key])

    def evaluate(
        self,
        agents: Mapping[str, AgentState],
        matrix: InformationMatrix,
    ) -> tuple[float, List[Dict[str, Any]]]:
        observations: List[Dict[str, Any]] = []
        for rollout_index in range(self.information_rollouts):
            realized = self._realize_for_evaluation(agents, matrix, rollout_index)
            realized_root = realized[self.root_id]
            evaluated_context = {
                **self.state_context,
                "evaluated_matrix": matrix.to_dict(),
            }
            observation = self.information_measure.compare(
                agents[self.root_id], realized_root, self.benchmark, evaluated_context,
            )
            observations.append(observation.to_dict())
        score = sum(item["score"] for item in observations)
        return score / len(observations), observations


class BeamCoordinateMatrixSearch:
    def __init__(
        self,
        evaluator: MatrixEvaluator,
        *,
        levels: Sequence[float] = (0.0, 0.5, 1.0),
        beam_width: int = 3,
        iterations: int = 3,
        hard_minimum: float = 0.0,
        maximum_evaluations: int = 20,
    ) -> None:
        self.evaluator = evaluator
        self.levels = tuple(levels)
        self.beam_width = beam_width
        self.iterations = iterations
        self.hard_minimum = hard_minimum
        self.maximum_evaluations = maximum_evaluations

    def optimize(
        self,
        agents: Mapping[str, AgentState],
        initial: InformationMatrix,
        dependencies: Sequence[CandidateDependency] = (),
    ) -> MatrixSearchResult:
        hard = [edge for edge in dependencies if edge.kind == "hard"]
        cache: Dict[
            Tuple[str, Tuple[float, ...]],
            tuple[float, List[Dict[str, Any]]],
        ] = {}
        discovered_at = {initial.key(): 0}
        evaluated_matrices: List[Dict[str, Any]] = []

        def score(matrix: InformationMatrix) -> tuple[float, List[Dict[str, Any]]]:
            matrix_key = matrix.key()
            cache_key = (self.evaluator.revision, matrix_key)
            if cache_key not in cache:
                if len(evaluated_matrices) >= self.maximum_evaluations:
                    return float("-inf"), []
                evaluated = self.evaluator.evaluate(agents, matrix)
                cache_key = (self.evaluator.revision, matrix_key)
                cache[cache_key] = evaluated
                evaluated_matrices.append({
                    "evaluation_index": len(evaluated_matrices),
                    "discovery_iteration": discovered_at.get(matrix_key, 0),
                    "matrix": matrix.to_dict(),
                    "delta_from_initial": matrix_difference(initial, matrix).to_dict(),
                    "matrix_evaluator": type(self.evaluator).__name__,
                    "measure_revision": self.evaluator.revision,
                    "predicted_information_gain": evaluated[0],
                    "information_observations": evaluated[1],
                })
            return cache[cache_key]

        beam = [initial]
        score(initial)
        for iteration in range(1, self.iterations + 1):
            candidates = {matrix.key(): matrix for matrix in beam}
            for matrix in beam:
                for neighbor in matrix_neighbors(matrix, self.levels, hard, self.hard_minimum):
                    candidates.setdefault(neighbor.key(), neighbor)
                    discovered_at.setdefault(neighbor.key(), iteration)
            ranked = sorted(candidates.values(), key=lambda candidate: candidate.key())
            ranked = sorted(
                ranked, key=lambda candidate: score(candidate)[0], reverse=True,
            )
            beam = ranked[: self.beam_width]
        best = max(beam, key=lambda candidate: score(candidate)[0])
        best_score, observations = score(best)
        return MatrixSearchResult(
            initial.clone(), best, best_score, len(evaluated_matrices), observations,
            evaluated_matrices=evaluated_matrices,
            admissible_level_matrices=admissible_matrix_count(
                initial.agent_ids, self.levels, hard, self.hard_minimum,
            ),
        )


def admissible_matrix_count(
    agent_ids: Sequence[str],
    levels: Sequence[float],
    hard_dependencies: Sequence[CandidateDependency] = (),
    hard_minimum: float = 0.0,
) -> int:
    """Count the global discrete matrix space under inbound and hard-edge constraints."""
    ordered_levels = sorted(set(float(value) for value in levels))
    total = 1
    for target in agent_ids:
        sources = [source for source in agent_ids if source != target]
        required_sources = {
            edge.source for edge in hard_dependencies
            if edge.kind == "hard" and edge.target == target and edge.source in sources
        }
        valid_columns = 0
        for weights in itertools.product(ordered_levels, repeat=len(sources)):
            if sum(weights) > 1.0 + 1e-9:
                continue
            by_source = dict(zip(sources, weights))
            if any(by_source[source] + 1e-9 < hard_minimum for source in required_sources):
                continue
            valid_columns += 1
        total *= valid_columns
    return total


def matrix_difference(previous: InformationMatrix, current: InformationMatrix) -> InformationMatrix:
    agent_ids = list(previous.agent_ids)
    agent_ids.extend(agent_id for agent_id in current.agent_ids if agent_id not in agent_ids)
    result = InformationMatrix.zero(agent_ids)
    for source in agent_ids:
        for target in agent_ids:
            if source == target:
                continue
            old = previous.weight(source, target) \
                if source in previous.agent_ids and target in previous.agent_ids else 0.0
            new = current.weight(source, target) \
                if source in current.agent_ids and target in current.agent_ids else 0.0
            result.set_weight(source, target, new - old)
    return result
