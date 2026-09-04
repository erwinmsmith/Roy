from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

if TYPE_CHECKING:
    from .matrix import InformationMatrix


@dataclass(frozen=True)
class SemanticInformationLandscape:
    """One Judge snapshot of task-conditioned semantic MIA parameters.

    These values are calibrated semantic potentials, not claims about measured
    Shannon information in bits. Rows are sources and columns are receivers.
    """

    agent_ids: List[str]
    directional_potential: List[List[float]]
    redundancy: List[List[float]]
    conversion_fidelity: List[float]
    root_id: str
    root_uncertainty: float
    calibration_summary: str
    root_relations: Dict[str, str] = field(default_factory=dict)
    coherence_adjustments: List[str] = field(default_factory=list)
    task_dimensions: List[str] = field(default_factory=list)
    observation_vectors: Dict[str, List[float]] = field(default_factory=dict)
    observation_noise: Dict[str, float] = field(default_factory=dict)
    root_dimension_uncertainty: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_agent_ids: Sequence[str],
        root_id: str,
        precision_dimensions: int = 0,
    ) -> "SemanticInformationLandscape":
        supplied_ids = [str(item) for item in value.get("agent_ids", [])]
        expected = list(expected_agent_ids)
        if len(supplied_ids) != len(expected) or set(supplied_ids) != set(expected):
            raise ValueError("Semantic Judge must return every supplied agent exactly once")
        source_g = _square_matrix(
            value.get("directional_potential"), supplied_ids, "G",
        )
        source_r = _square_matrix(value.get("redundancy"), supplied_ids, "R")
        source_lambdas = _unit_vector(
            value.get("conversion_fidelity"), supplied_ids, "conversion_fidelity",
        )
        source_index = {agent_id: index for index, agent_id in enumerate(supplied_ids)}
        order = [source_index[agent_id] for agent_id in expected]
        g = [[source_g[row][column] for column in order] for row in order]
        raw_r = [[source_r[row][column] for column in order] for row in order]
        lambdas = [source_lambdas[index] for index in order]
        for index in range(len(expected)):
            g[index][index] = 0.0
        redundancy = [[0.0 for _ in expected] for _ in expected]
        for left in range(len(expected)):
            for right in range(left + 1, len(expected)):
                overlap = (raw_r[left][right] + raw_r[right][left]) / 2.0
                redundancy[left][right] = redundancy[right][left] = overlap
        if root_id not in expected:
            raise ValueError(f"Semantic Judge root {root_id!r} is absent")
        uncertainty = _unit(float(value.get("root_uncertainty", 1.0)), "root_uncertainty")
        raw_relations = value.get("root_relations")
        if not isinstance(raw_relations, Mapping) or set(map(str, raw_relations)) != set(expected):
            raise ValueError("Semantic Judge root_relations must cover every supplied agent")
        allowed_relations = {"supports", "contradicts", "complements", "unresolved"}
        root_relations = {agent_id: str(raw_relations[agent_id]) for agent_id in expected}
        invalid_relations = set(root_relations.values()) - allowed_relations
        if invalid_relations:
            raise ValueError(f"Semantic Judge returned invalid root relations: {sorted(invalid_relations)}")
        if root_relations[root_id] != "supports":
            raise ValueError("Semantic Judge must mark the root as supporting its own current answer")
        root_index = expected.index(root_id)
        adjustments: List[str] = []
        for source, relation in root_relations.items():
            if source == root_id:
                continue
            source_index_value = expected.index(source)
            novelty = max(0.0, 1.0 - redundancy[source_index_value][root_index])
            minimum = novelty if relation == "contradicts" else (
                0.25 * novelty if relation == "complements" else 0.0
            )
            if g[source_index_value][root_index] + 1e-9 < minimum:
                previous = g[source_index_value][root_index]
                g[source_index_value][root_index] = minimum
                adjustments.append(
                    f"G[{source}][{root_id}] raised from {previous:.6g} to {minimum:.6g} "
                    f"for relation={relation} and redundancy={redundancy[source_index_value][root_index]:.6g}"
                )
        summary = str(value.get("calibration_summary", "")).strip()
        if not summary:
            raise ValueError("Semantic Judge omitted calibration_summary")
        task_dimensions: List[str] = []
        observation_vectors: Dict[str, List[float]] = {}
        observation_noise: Dict[str, float] = {}
        root_dimension_uncertainty: List[float] = []
        if precision_dimensions:
            task_dimensions = [
                str(item).strip() for item in value.get("task_dimensions", [])
            ]
            if (
                len(task_dimensions) != precision_dimensions
                or len(set(task_dimensions)) != precision_dimensions
                or not all(task_dimensions)
            ):
                raise ValueError(
                    f"Semantic Judge task_dimensions must contain {precision_dimensions} "
                    "distinct non-empty dimensions"
                )
            raw_vectors = value.get("observation_vectors")
            if not isinstance(raw_vectors, Mapping) or set(map(str, raw_vectors)) != set(expected):
                raise ValueError("Semantic Judge observation_vectors must cover every supplied agent")
            for agent_id in expected:
                vector = list(raw_vectors[agent_id])
                if len(vector) != precision_dimensions:
                    raise ValueError(
                        f"Semantic Judge h[{agent_id}] must contain {precision_dimensions} values"
                    )
                observation_vectors[agent_id] = [
                    _unit(float(item), "observation_vectors") for item in vector
                ]
            raw_noise = value.get("observation_noise")
            if not isinstance(raw_noise, Mapping) or set(map(str, raw_noise)) != set(expected):
                raise ValueError("Semantic Judge observation_noise must cover every supplied agent")
            observation_noise = {
                agent_id: _unit(float(raw_noise[agent_id]), "observation_noise")
                for agent_id in expected
            }
            raw_uncertainty = list(value.get("root_dimension_uncertainty", []))
            if len(raw_uncertainty) != precision_dimensions:
                raise ValueError(
                    "Semantic Judge root_dimension_uncertainty must match task_dimensions"
                )
            root_dimension_uncertainty = [
                _unit(float(item), "root_dimension_uncertainty")
                for item in raw_uncertainty
            ]
        return cls(
            expected, g, redundancy, lambdas, root_id, uncertainty, summary,
            root_relations, adjustments, task_dimensions, observation_vectors,
            observation_noise, root_dimension_uncertainty,
        )

    @property
    def revision(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def subset(self, agent_ids: Iterable[str]) -> "SemanticInformationLandscape":
        selected = list(agent_ids)
        if self.root_id not in selected or not set(selected).issubset(self.agent_ids):
            raise ValueError("MIA landscape subset must retain the root and use known agents")
        indices = [self.agent_ids.index(agent_id) for agent_id in selected]
        return SemanticInformationLandscape(
            selected,
            [[self.directional_potential[row][column] for column in indices] for row in indices],
            [[self.redundancy[row][column] for column in indices] for row in indices],
            [self.conversion_fidelity[index] for index in indices],
            self.root_id,
            self.root_uncertainty,
            self.calibration_summary,
            {agent_id: self.root_relations.get(agent_id, "unresolved") for agent_id in selected},
            list(self.coherence_adjustments),
            list(self.task_dimensions),
            {
                agent_id: list(self.observation_vectors[agent_id])
                for agent_id in selected if agent_id in self.observation_vectors
            },
            {
                agent_id: self.observation_noise[agent_id]
                for agent_id in selected if agent_id in self.observation_noise
            },
            list(self.root_dimension_uncertainty),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "revision": self.revision, "quantity_kind": "semantic_estimate"}


@dataclass(frozen=True)
class MIAObjective:
    objective: float
    usable_delivery: float
    root_uncertainty: float
    direct_delivery: float
    multi_hop_delivery: float
    redundancy_correction: float
    root_reach: Dict[str, float]
    path_horizon: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MIAObjectiveEvaluator:
    """Pure numerical matrix evaluator; evaluating W never invokes an LLM."""

    name = "mia_matrix_functional_v2"

    def __init__(
        self,
        landscape: SemanticInformationLandscape,
        *,
        path_horizon: int = 3,
        reference_matrix: InformationMatrix | None = None,
    ) -> None:
        if path_horizon < 1:
            raise ValueError("MIA path horizon must be positive")
        self.landscape = landscape
        self.path_horizon = path_horizon
        self.reference = (
            self.objective(reference_matrix).objective if reference_matrix is not None else 0.0
        )

    @property
    def revision(self) -> str:
        return self.landscape.revision

    def objective(self, matrix: InformationMatrix) -> MIAObjective:
        landscape = self.landscape.subset(matrix.agent_ids)
        size = len(matrix.agent_ids)
        root = matrix.agent_ids.index(landscape.root_id)
        operator = [
            [
                matrix.values[source][receiver]
                * landscape.directional_potential[source][receiver]
                * landscape.conversion_fidelity[receiver]
                for receiver in range(size)
            ]
            for source in range(size)
        ]
        # Root is an absorbing destination for this root-conditioned objective.
        # This prevents cycles such as root->worker->root from being counted as
        # repeatedly new information while retaining all paths that end at root.
        propagation = [[0.0 for _ in range(size)] for _ in range(size)]
        power = [list(row) for row in operator]
        horizon = min(self.path_horizon, max(1, size - 1))
        for _ in range(horizon):
            for row in range(size):
                for column in range(size):
                    propagation[row][column] += power[row][column]
            absorb = [list(row) for row in operator]
            for receiver in range(size):
                absorb[root][receiver] = 0.0
            power = _matmul(power, absorb)
        reach = {
            agent_id: propagation[index][root]
            for index, agent_id in enumerate(matrix.agent_ids)
            if index != root
        }
        direct = sum(operator[index][root] for index in range(size) if index != root)
        total = sum(reach.values())
        correction = 0.0
        non_root = [index for index in range(size) if index != root]
        for position, left in enumerate(non_root):
            for right in non_root[position + 1:]:
                correction += (
                    landscape.redundancy[left][right]
                    * propagation[left][root]
                    * propagation[right][root]
                )
        usable_delivery = max(0.0, total - correction)
        # Information delivered to an already-certain root is not information
        # gain. Use a smooth intersection that stays below both normalized
        # quantities while remaining strictly monotone in delivery. A hard
        # minimum creates plateaus where weak redundant support ties decisive
        # contradictory evidence; a raw product over-attenuates modest values.
        uncertainty = landscape.root_uncertainty
        objective = (
            uncertainty * usable_delivery / (uncertainty + usable_delivery)
            if uncertainty > 0.0 and usable_delivery > 0.0 else 0.0
        )
        return MIAObjective(
            objective=objective,
            usable_delivery=usable_delivery,
            root_uncertainty=landscape.root_uncertainty,
            direct_delivery=direct,
            multi_hop_delivery=total - direct,
            redundancy_correction=correction,
            root_reach=reach,
            path_horizon=horizon,
        )

    def evaluate(
        self,
        agents: Mapping[str, object],
        matrix: InformationMatrix,
    ) -> tuple[float, List[Dict[str, Any]]]:
        del agents
        objective = self.objective(matrix)
        gain = objective.objective - self.reference
        observation = {
            "estimator": self.name,
            "score": gain,
            "reference_objective": self.reference,
            "objective": objective.to_dict(),
            "llm_rollouts": 0,
        }
        return gain, [observation]


@dataclass(frozen=True)
class PrecisionLogDetObjective:
    objective: float
    log_det_gain: float
    root_reach: Dict[str, float]
    observation_strength: Dict[str, float]
    root_precision: List[List[float]]
    realized_precision: List[List[float]]
    path_horizon: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrecisionLogDetObjectiveEvaluator:
    """D-optimal semantic information gain over the transfer network."""

    name = "mia_precision_logdet_v1"

    def __init__(
        self,
        landscape: SemanticInformationLandscape,
        *,
        path_horizon: int = 3,
        reference_matrix: InformationMatrix | None = None,
        precision_floor: float = 1e-3,
    ) -> None:
        if path_horizon < 1:
            raise ValueError("MIA path horizon must be positive")
        if not 0.0 < precision_floor <= 1.0:
            raise ValueError("precision_floor must be in (0, 1]")
        if not landscape.task_dimensions:
            raise ValueError("precision log-det objective requires Judge observation geometry")
        self.landscape = landscape
        self.path_horizon = path_horizon
        self.precision_floor = precision_floor
        self.reference = (
            self.objective(reference_matrix).objective if reference_matrix is not None else 0.0
        )

    @property
    def revision(self) -> str:
        return f"{self.name}:{self.landscape.revision}:{self.precision_floor:.12g}"

    def objective(self, matrix: InformationMatrix) -> PrecisionLogDetObjective:
        landscape = self.landscape.subset(matrix.agent_ids)
        size = len(matrix.agent_ids)
        root = matrix.agent_ids.index(landscape.root_id)
        operator = [
            [
                matrix.values[source][receiver]
                * landscape.directional_potential[source][receiver]
                * landscape.conversion_fidelity[receiver]
                for receiver in range(size)
            ]
            for source in range(size)
        ]
        propagation = [[0.0 for _ in range(size)] for _ in range(size)]
        power = [list(row) for row in operator]
        horizon = min(self.path_horizon, max(1, size - 1))
        for _ in range(horizon):
            for row in range(size):
                for column in range(size):
                    propagation[row][column] += power[row][column]
            absorb = [list(row) for row in operator]
            for receiver in range(size):
                absorb[root][receiver] = 0.0
            power = _matmul(power, absorb)

        reach = {
            agent_id: min(1.0, max(0.0, propagation[index][root]))
            for index, agent_id in enumerate(matrix.agent_ids)
            if index != root
        }
        dimension_count = len(landscape.task_dimensions)
        root_precision = [[0.0 for _ in range(dimension_count)] for _ in range(dimension_count)]
        for index, uncertainty in enumerate(landscape.root_dimension_uncertainty):
            root_precision[index][index] = 1.0 / max(uncertainty, self.precision_floor)
        realized = [list(row) for row in root_precision]
        strength: Dict[str, float] = {}
        for agent_id, amplitude in reach.items():
            noise = max(landscape.observation_noise[agent_id], self.precision_floor)
            contribution = amplitude * amplitude / noise
            strength[agent_id] = contribution
            vector = landscape.observation_vectors[agent_id]
            for row in range(dimension_count):
                for column in range(dimension_count):
                    realized[row][column] += contribution * vector[row] * vector[column]
        gain = 0.5 * (_logdet_spd(realized) - _logdet_spd(root_precision))
        return PrecisionLogDetObjective(
            objective=max(0.0, gain),
            log_det_gain=max(0.0, gain),
            root_reach=reach,
            observation_strength=strength,
            root_precision=root_precision,
            realized_precision=realized,
            path_horizon=horizon,
        )

    def evaluate(
        self,
        agents: Mapping[str, object],
        matrix: InformationMatrix,
    ) -> tuple[float, List[Dict[str, Any]]]:
        del agents
        objective = self.objective(matrix)
        gain = objective.objective - self.reference
        return gain, [{
            "estimator": self.name,
            "score": gain,
            "reference_objective": self.reference,
            "objective": objective.to_dict(),
            "llm_rollouts": 0,
        }]


def _matmul(left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
    size = len(left)
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(size))
         for column in range(size)]
        for row in range(size)
    ]


def _square_matrix(value: Any, agent_ids: Sequence[str], name: str) -> List[List[float]]:
    size = len(agent_ids)
    if isinstance(value, Mapping):
        if set(map(str, value)) != set(agent_ids):
            raise ValueError(f"Semantic Judge {name} mapping has incorrect row ids")
        rows = []
        for source in agent_ids:
            row = value[source]
            if isinstance(row, Mapping):
                if set(map(str, row)) != set(agent_ids):
                    raise ValueError(f"Semantic Judge {name} mapping has incorrect column ids")
                rows.append([row[target] for target in agent_ids])
            else:
                rows.append(list(row))
    else:
        rows = list(value or [])
    if len(rows) != size or any(len(list(row)) != size for row in rows):
        raise ValueError(f"Semantic Judge {name} must be {size}x{size}")
    return [[_unit(float(item), name) for item in row] for row in rows]


def _unit_vector(value: Any, agent_ids: Sequence[str], name: str) -> List[float]:
    size = len(agent_ids)
    if isinstance(value, Mapping):
        if set(map(str, value)) != set(agent_ids):
            raise ValueError(f"Semantic Judge {name} mapping has incorrect agent ids")
        items = [value[agent_id] for agent_id in agent_ids]
    else:
        items = list(value or [])
    if len(items) != size:
        raise ValueError(f"Semantic Judge {name} must contain {size} values")
    return [_unit(float(item), name) for item in items]


def _unit(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Semantic Judge {name} values must be in [0, 1]")
    return value


def _logdet_spd(matrix: List[List[float]]) -> float:
    """Return log(det(matrix)) using a small dependency-free Cholesky factorization."""
    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                lower[row][inner] * lower[column][inner] for inner in range(column)
            )
            if row == column:
                if residual <= 0.0:
                    raise ValueError("precision matrix must be positive definite")
                lower[row][column] = residual ** 0.5
            else:
                lower[row][column] = residual / lower[column][column]
    return 2.0 * sum(math.log(lower[index][index]) for index in range(size))
