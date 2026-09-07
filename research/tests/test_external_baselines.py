from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from roy_research.external_baselines import (
    AuditedCaller,
    AutoAgentsBaseline,
    DyLANBaseline,
    EvoAgentBaseline,
    load_string_constants,
    verify_checkout,
)
from roy_research.training_free.types import BenchmarkTask


@dataclass
class Completion:
    content: str
    prompt_tokens: int = 1
    completion_tokens: int = 2
    total_tokens: int = 3


class QueueClient:
    model = "fake"

    def __init__(self, responses: List[str]) -> None:
        self.responses = iter(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return Completion(next(self.responses))


def math_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="MATH/test/0", benchmark="MATH", instruction="What is 1+1?",
        public_tests=[], evaluator_payload={"solution": "\\boxed{2}"},
    )


def test_literal_loader_does_not_import_upstream_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "prompts.py"
    source.write_text("TEXT = 'hello'\nMAPPING = {'a': 'b'}\nimport missing_package\n", encoding="utf-8")
    assert load_string_constants(source, ["TEXT", "MAPPING"]) == {
        "TEXT": "hello", "MAPPING": {"a": "b"},
    }


def test_checkout_verification_rejects_wrong_revision_and_tracked_edits(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    assert verify_checkout(tmp_path, revision) == revision
    with pytest.raises(ValueError, match="revision mismatch"):
        verify_checkout(tmp_path, "0" * 40)
    tracked.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked modifications"):
        verify_checkout(tmp_path, revision)


def test_dylan_math_preserves_early_consensus(tmp_path: Path) -> None:
    math_source = tmp_path / "code/MATH/llmlp_gen_math_listwise_cot.py"
    code_source = tmp_path / "code/HumanEval/prompt_lib.py"
    math_source.parent.mkdir(parents=True)
    code_source.parent.mkdir(parents=True)
    math_source.write_text("SYSTEM_PROMPT='debate'\nEXAMPLES='example'\n", encoding="utf-8")
    code_source.write_text(
        "ROLE_MAP={'PythonAssistant':'p','AlgorithmDeveloper':'a','ComputerScientist':'c','Programmer':'r'}\n"
        "ROLE_MAP_INIT={}\n", encoding="utf-8",
    )
    client = QueueClient(["reason \\boxed{2}"] * 3)
    caller = AuditedCaller(client, method="dylan", max_tokens=10, temperature=0)
    result = DyLANBaseline(tmp_path, caller).run(math_task())
    assert result.trace["early_stop"] == "round_0_consensus"
    assert len(client.calls) == 3
    assert result.prediction.endswith("\\boxed{2}")


def test_evoagent_runs_published_evolution_sequence(tmp_path: Path) -> None:
    prompt_source = tmp_path / "spp/agent_prompt_writing.py"
    control_source = tmp_path / "spp/util_func.py"
    prompt_source.parent.mkdir(parents=True)
    prompt_source.write_text(
        "INSTRUCTION_META='{question} {answer} {description}'\n"
        "INSTRUCTION_CHECK='{question} {description_ls} {description}'\n"
        "INSTRUCTION_MULTI='{question} {description}'\n"
        "INSTRUCTION_REFINE='{question} {old_answer} {description} {new_answer}'\n",
        encoding="utf-8",
    )
    control_source.write_text("def collaboration_func(): pass\n", encoding="utf-8")
    client = QueueClient(["initial", "expert one", "Retain", "sub", "refined"])
    caller = AuditedCaller(client, method="evoagent", max_tokens=10, temperature=0)
    result = EvoAgentBaseline(tmp_path, caller, iterations=1).run(math_task())
    assert result.prediction == "refined"
    assert result.trace["evolved_agents"][0]["description"] == "expert one"
    assert [call["metadata"]["stage"] for call in client.calls] == [
        "initial_answer", "iteration_0_meta", "iteration_0_check_0",
        "iteration_0_expert", "iteration_0_refine",
    ]


def _write_autoagents_sources(root: Path) -> None:
    action_root = root / "autoagents/actions"
    action_root.mkdir(parents=True)
    common = "PROMPT_TEMPLATE='{context} {format_example}'\nFORMAT_EXAMPLE='format'\nTOOLS='None'\n"
    (action_root / "create_roles.py").write_text(common, encoding="utf-8")
    (action_root / "check_roles.py").write_text(
        "PROMPT_TEMPLATE='{question} {existing_roles} {selected_roles} {created_roles} {history} {tools} {format_example}'\n"
        "FORMAT_EXAMPLE='format'\nTOOLS='None'\n", encoding="utf-8",
    )
    (action_root / "check_plans.py").write_text(
        "PROMPT_TEMPLATE='{context} {roles} {plan} {history} {tools} {format_example}'\n"
        "FORMAT_EXAMPLE='format'\nTOOLS='None'\n", encoding="utf-8",
    )
    (action_root / "custom_action.py").write_text(
        "PROMPT_TEMPLATE='{role} {context} {suggestions} {previous} {completed_steps} {tool} {format_example}'\n"
        "FORMAT_EXAMPLE='format'\n", encoding="utf-8",
    )


def test_autoagents_records_generated_roles_and_plan(tmp_path: Path) -> None:
    _write_autoagents_sources(tmp_path)
    plan = """## Created Roles List
```json
{"name":"Solver","description":"compute","tools":[],"suggestions":"check","prompt":"solve"}
```
## Execution Plan
1. Solver: solve carefully
"""
    client = QueueClient([plan, "expert result", "final \\boxed{2}"])
    caller = AuditedCaller(client, method="autoagents", max_tokens=10, temperature=0)
    result = AutoAgentsBaseline(tmp_path, caller, review_rounds=0).run(math_task())
    assert result.trace["generated_roles"][0]["name"] == "Solver"
    assert result.trace["execution_plan"] == ["Solver: solve carefully"]
    assert result.prediction == "final \\boxed{2}"
    assert len(client.calls) == 3
