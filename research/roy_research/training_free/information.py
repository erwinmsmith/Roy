from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence

from .llm import PairwiseInformationProbeModel, PosteriorProbeModel
from .types import AgentState


@dataclass(frozen=True)
class InformationObservation:
    estimator: str
    score: float
    before: Dict[str, Any]
    after: Dict[str, Any]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InformationMeasure(Protocol):
    """Benchmark adapter consumed by search; no hypothesis space is required."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def state(
        self,
        root: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]: ...

    def compare(
        self,
        before: AgentState,
        after: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> InformationObservation: ...

    def uncertainty(self, state: Mapping[str, Any]) -> float | None: ...


class AnswerDistributionMeasure:
    """Optional answer-space MIA estimator, suitable for tasks such as MATH."""

    name = "answer_distribution"

    def __init__(
        self,
        probe: PosteriorProbeModel,
        agent_sets: Iterable[Mapping[str, AgentState]],
        *,
        maximum_hypotheses: int = 8,
    ) -> None:
        self.probe = probe
        self.maximum_hypotheses = maximum_hypotheses
        self.support = answer_support(agent_sets, maximum=maximum_hypotheses)
        self._state_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

    @property
    def revision(self) -> str:
        return json.dumps(self.support, ensure_ascii=False)

    def state(
        self,
        root: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        fingerprint = _agent_fingerprint(root)
        key = (self.revision, fingerprint)
        if key not in self._state_cache:
            distribution = self.probe.posterior(
                root, self.support, benchmark, state_context,
            )
            self._state_cache[key] = {
                "estimator": self.name,
                "support": list(self.support),
                "distribution": distribution,
                "uncertainty": entropy(distribution),
            }
        return dict(self._state_cache[key])

    def compare(
        self,
        before: AgentState,
        after: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> InformationObservation:
        _add_answer(self.support, after.result.candidate_answer, self.maximum_hypotheses)
        before_state = self.state(before, benchmark, state_context)
        if after.to_dict() == before.to_dict():
            after_state = before_state
            score = 0.0
        else:
            after_state = self.state(after, benchmark, state_context)
            score = kl_divergence(
                after_state["distribution"], before_state["distribution"], self.support,
            )
        return InformationObservation(
            estimator=self.name,
            score=score,
            before=before_state,
            after=after_state,
            details={"measure": "KL(after || before)"},
        )

    def uncertainty(self, state: Mapping[str, Any]) -> float | None:
        value = state.get("uncertainty")
        return float(value) if value is not None else None


class PairwiseStateMeasure:
    """Support-free estimator for code, tool, interactive, and open-ended tasks."""

    name = "pairwise_state"
    revision = "pairwise-state-v1"

    def __init__(self, probe: PairwiseInformationProbeModel) -> None:
        self.probe = probe
        self._comparison_cache: Dict[tuple[str, str], InformationObservation] = {}

    def state(
        self,
        root: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "estimator": self.name,
            "uncertainty": 1.0 - root.result.confidence,
            "candidate_present": bool(root.result.candidate_answer.strip()),
            "claim_count": len(root.result.claims),
            "evidence_count": len(root.result.evidence),
            "unresolved_count": len(root.result.unresolved),
        }

    def compare(
        self,
        before: AgentState,
        after: AgentState,
        benchmark: str,
        state_context: Mapping[str, Any] | None = None,
    ) -> InformationObservation:
        before_state = self.state(before, benchmark, state_context)
        if after.to_dict() == before.to_dict():
            return InformationObservation(
                self.name, 0.0, before_state, before_state,
                {"rationale": "root state is unchanged"},
            )
        key = (_agent_fingerprint(before), _agent_fingerprint(after))
        if key not in self._comparison_cache:
            result = self.probe.compare(before, after, benchmark, state_context)
            after_state = self.state(after, benchmark, state_context)
            after_state["uncertainty"] = result["after_uncertainty"]
            before_state["uncertainty"] = result["before_uncertainty"]
            self._comparison_cache[key] = InformationObservation(
                estimator=self.name,
                score=result["information_gain"],
                before=before_state,
                after=after_state,
                details={
                    "new_information": result["new_information"],
                    "rationale": result["rationale"],
                },
            )
        return self._comparison_cache[key]

    def uncertainty(self, state: Mapping[str, Any]) -> float | None:
        value = state.get("uncertainty")
        return float(value) if value is not None else None


def answer_support(
    agent_sets: Iterable[Mapping[str, AgentState]],
    *,
    maximum: int = 8,
) -> List[str]:
    answers: List[str] = []
    for agents in agent_sets:
        for agent in agents.values():
            answer = agent.result.candidate_answer.strip()
            if answer and answer not in answers:
                answers.append(answer)
    answers = answers[: max(1, maximum - 1)]
    return [*answers, "OTHER"] if answers else ["UNRESOLVED", "OTHER"]


def _add_answer(support: List[str], answer: str, maximum: int) -> bool:
    hypothesis = answer.strip()
    if not hypothesis or hypothesis in support or hypothesis == "OTHER":
        return False
    if sum(item != "OTHER" for item in support) >= maximum - 1:
        return False
    support.insert(support.index("OTHER") if "OTHER" in support else len(support), hypothesis)
    return True


def kl_divergence(
    posterior: Mapping[str, float],
    prior: Mapping[str, float],
    support: Sequence[str],
    epsilon: float = 1e-9,
) -> float:
    return sum(
        max(epsilon, posterior.get(item, 0.0))
        * math.log(
            max(epsilon, posterior.get(item, 0.0))
            / max(epsilon, prior.get(item, 0.0))
        )
        for item in support
    )


def entropy(distribution: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in distribution.values() if value > 0)


def _agent_fingerprint(agent: AgentState) -> str:
    return json.dumps(agent.to_dict(), sort_keys=True, ensure_ascii=False)
