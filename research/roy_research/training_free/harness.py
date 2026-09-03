from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping

from .tools import TaskToolRegistry, ToolRequest, ToolResult
from .types import (
    AgentState,
    AgentStatus,
    ContextState,
    MemoryState,
    ResultState,
)

AGENT_HARNESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentHarnessConfig:
    maximum_memory_entries: int = 64
    maximum_memory_entry_chars: int = 2_000
    retrieved_memory_entries: int = 12
    maximum_received_messages: int = 16
    maximum_context_chars: int = 24_000

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError("Agent harness limits must be positive")


class AgentHarness:
    """Fixed, benchmark-neutral execution harness around one autonomous Agent state."""

    def __init__(
        self,
        state: AgentState,
        tools: TaskToolRegistry | None = None,
        config: AgentHarnessConfig | None = None,
    ) -> None:
        self.state = state
        self.tools = tools
        self.config = config or AgentHarnessConfig()
        self.validate()
        self._contract_fingerprint = self.contract_fingerprint()

    @classmethod
    def create_root(
        cls,
        *,
        original_task: str,
        objective: str,
        role: str,
        tools: Iterable[str],
        expected_output: str,
        stop_condition: str,
        public_tests: Iterable[str] = (),
        tool_registry: TaskToolRegistry | None = None,
        config: AgentHarnessConfig | None = None,
    ) -> "AgentHarness":
        state = AgentState(
            agent_id="A0",
            parent_id=None,
            objective=objective,
            role=role,
            context=ContextState(original_task, public_tests=list(public_tests)),
            memory=MemoryState("memory/A0"),
            tools=list(tools),
            result=ResultState(unresolved=[objective]),
            status=AgentStatus.READY,
            expected_output=expected_output,
            stop_condition=stop_condition,
        )
        return cls(state, tool_registry, config)

    @property
    def contract(self) -> Dict[str, Any]:
        static_context = {
            "original_task": self.state.context.original_task,
            "mandatory_inputs": self.state.context.mandatory_inputs,
            "weighted_inputs": self.state.context.weighted_inputs,
            "public_tests": self.state.context.public_tests,
        }
        return {
            "agent_id": self.state.agent_id,
            "parent_id": self.state.parent_id,
            "objective": self.state.objective,
            "role": self.state.role,
            "tools": list(self.state.tools),
            "expected_output": self.state.expected_output,
            "stop_condition": self.state.stop_condition,
            "memory_namespace": self.state.memory.namespace,
            "inherited_memory_refs": list(self.state.memory.inherited_refs),
            "original_task_sha256": hashlib.sha256(
                self.state.context.original_task.encode("utf-8")
            ).hexdigest(),
            "static_context_sha256": hashlib.sha256(
                json.dumps(static_context, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }

    def contract_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def assert_contract_unchanged(self) -> None:
        if self.contract_fingerprint() != self._contract_fingerprint:
            raise ValueError(f"Agent {self.state.agent_id} immutable contract was modified")

    def validate(self) -> None:
        agent = self.state
        if not agent.agent_id or not agent.objective or not agent.role:
            raise ValueError("Agent id, objective, and role are required")
        if agent.memory.namespace != f"memory/{agent.agent_id}":
            raise ValueError(f"Agent {agent.agent_id} must have a private memory namespace")
        if not agent.context.original_task:
            raise ValueError(f"Agent {agent.agent_id} has no original task context")
        if len(agent.tools) != len(set(agent.tools)):
            raise ValueError(f"Agent {agent.agent_id} has duplicate tool capabilities")
        if not agent.expected_output or not agent.stop_condition:
            raise ValueError(f"Agent {agent.agent_id} has no output/termination contract")
        if self.tools is not None:
            registered = {item["name"] for item in self.tools.catalog}
            unavailable = set(agent.tools) - registered
            if unavailable:
                raise ValueError(
                    f"Agent {agent.agent_id} has unavailable tools: {sorted(unavailable)}"
                )

    def execution_view(self) -> Dict[str, Any]:
        self.assert_contract_unchanged()
        query = " ".join((
            self.state.objective,
            *self.state.result.unresolved,
            *self.state.context.mandatory_inputs,
        ))
        view = {
            "harness_schema_version": AGENT_HARNESS_SCHEMA_VERSION,
            "contract_fingerprint": self._contract_fingerprint,
            "contract": self.contract,
            "status": self.state.status.value,
            "context": {
                "original_task": self.state.context.original_task,
                "mandatory_inputs": list(self.state.context.mandatory_inputs),
                "weighted_inputs": dict(self.state.context.weighted_inputs),
                "received_messages": self.state.context.received_messages[
                    -self.config.maximum_received_messages:
                ],
                "public_tests": list(self.state.context.public_tests),
            },
            "private_memory": {
                "namespace": self.state.memory.namespace,
                "retrieved_entries": self.retrieve_memory(query),
            },
            "result": asdict(self.state.result),
        }
        return _truncate_view(view, self.config.maximum_context_chars)

    def public_summary(self) -> Dict[str, Any]:
        self.assert_contract_unchanged()
        return {
            "agent_id": self.state.agent_id,
            "parent_id": self.state.parent_id,
            "objective": self.state.objective,
            "role": self.state.role,
            "status": self.state.status.value,
            "tools": list(self.state.tools),
            "expected_output": self.state.expected_output,
            "stop_condition": self.state.stop_condition,
            "result": asdict(self.state.result),
            "memory_namespace": self.state.memory.namespace,
            "memory_entry_count": len(self.state.memory.entries),
        }

    def channel_source_view(self, receiver_objective: str) -> Dict[str, Any]:
        self.assert_contract_unchanged()
        return {
            **self.public_summary(),
            "relevant_private_memory": self.retrieve_memory(receiver_objective),
        }

    @property
    def tool_catalog(self) -> List[Dict[str, Any]]:
        if self.tools is None:
            return []
        assigned = set(self.state.tools)
        return [item for item in self.tools.catalog if item["name"] in assigned]

    def receive_messages(self, messages: Iterable[str]) -> None:
        for message in messages:
            text = str(message).strip()
            if text and text not in self.state.context.received_messages:
                self.state.context.received_messages.append(text)
        self.state.context.received_messages = self.state.context.received_messages[
            -self.config.maximum_received_messages:
        ]

    def remember(self, entries: Iterable[str]) -> None:
        for entry in entries:
            text = str(entry).strip()[: self.config.maximum_memory_entry_chars]
            if text and text not in self.state.memory.entries:
                self.state.memory.entries.append(text)
        self.state.memory.entries = self.state.memory.entries[-self.config.maximum_memory_entries:]

    def retrieve_memory(self, query: str) -> List[str]:
        entries = self.state.memory.entries
        if not entries:
            return []
        terms = set(_terms(query))
        ranked = sorted(
            enumerate(entries),
            key=lambda item: (
                len(terms & set(_terms(item[1]))),
                item[0],
            ),
            reverse=True,
        )
        selected = ranked[: self.config.retrieved_memory_entries]
        return [entry for _, entry in sorted(selected, key=lambda item: item[0])]

    def execute_tool(
        self,
        request: ToolRequest,
        *,
        scope: str,
    ) -> ToolResult:
        self.assert_contract_unchanged()
        if self.tools is None:
            return ToolResult(False, error="this Agent has no tool registry")
        if request.tool_name not in self.state.tools:
            return ToolResult(False, error="tool is outside this Agent's fixed capability set")
        return self.tools.execute(self.state, request, scope=scope)

    def apply_model_update(
        self,
        result: ResultState,
        memory_entries: Iterable[str],
        *,
        status: AgentStatus = AgentStatus.DONE,
    ) -> None:
        self.assert_contract_unchanged()
        self.state.result = result
        self.remember(memory_entries)
        self.state.status = status

    def snapshot(self) -> Dict[str, Any]:
        self.assert_contract_unchanged()
        return {
            "schema_version": AGENT_HARNESS_SCHEMA_VERSION,
            "contract_fingerprint": self._contract_fingerprint,
            "config": asdict(self.config),
            "state": self.state.to_dict(),
        }

    @classmethod
    def restore(
        cls,
        snapshot: Mapping[str, Any],
        *,
        available_tools: Iterable[str],
        tool_registry: TaskToolRegistry | None = None,
    ) -> "AgentHarness":
        if snapshot.get("schema_version") != AGENT_HARNESS_SCHEMA_VERSION:
            raise ValueError("unsupported Agent harness snapshot schema")
        state_value = snapshot.get("state")
        if not isinstance(state_value, Mapping):
            raise ValueError("Agent harness snapshot has no state")
        context = state_value.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("Agent harness snapshot has no context")
        state = AgentState.from_dict(
            state_value,
            original_task=str(context.get("original_task", "")),
            available_tools=available_tools,
        )
        config = AgentHarnessConfig(**dict(snapshot.get("config", {})))
        harness = cls(state, tool_registry, config)
        if harness.contract_fingerprint() != snapshot.get("contract_fingerprint"):
            raise ValueError("Agent harness snapshot contract fingerprint mismatch")
        return harness

    def clone(self) -> "AgentHarness":
        return AgentHarness(copy.deepcopy(self.state), self.tools, self.config)


def _terms(value: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", value.lower())


def _truncate_view(view: Dict[str, Any], maximum_chars: int) -> Dict[str, Any]:
    def size() -> int:
        return len(json.dumps(truncated, ensure_ascii=False))

    truncated = copy.deepcopy(view)
    if size() <= maximum_chars:
        return view
    truncated["private_memory"]["retrieved_entries"] = []
    messages = truncated["context"]["received_messages"]
    while messages and size() > maximum_chars:
        messages.pop(0)
    for key in ("evidence", "assumptions", "claims"):
        values = truncated["result"][key]
        while values and size() > maximum_chars:
            values.pop(0)
    for key in ("public_tests", "mandatory_inputs"):
        values = truncated["context"][key]
        while values and size() > maximum_chars:
            values.pop()
    unresolved = truncated["result"]["unresolved"]
    while len(unresolved) > 1 and size() > maximum_chars:
        unresolved.pop(0)
    for container, key in (
        (truncated["result"], "reasoning_summary"),
        (truncated["result"], "candidate_answer"),
        (truncated["context"], "original_task"),
    ):
        marker = "...[truncated]"
        while len(container[key]) > 256 and size() > maximum_chars:
            target_length = max(256, len(container[key]) // 2)
            container[key] = container[key][: target_length - len(marker)] + marker
    if size() > maximum_chars:
        raise ValueError("Agent immutable contract and minimum state exceed context budget")
    return truncated
