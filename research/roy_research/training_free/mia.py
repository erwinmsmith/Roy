from __future__ import annotations

import hashlib
import json
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

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_agent_ids: Sequence[str],
        root_id: str,
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
        return cls(
            expected, g, redundancy, lambdas, root_id, uncertainty, summary,
            root_relations, adjustments,
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
        # gain. Both quantities use the same normalized semantic scale, so the
        # realizable gain is their intersection. Multiplication would attenuate
        # the estimate twice and make modest uncertainty mathematically unable
        # to cross the organization threshold even under decisive delivery.
        objective = min(landscape.root_uncertainty, usable_delivery)
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
