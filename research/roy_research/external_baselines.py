"""Source-pinned adapters for external dynamic multi-agent baselines.

The upstream repositories remain outside Roy.  We read their prompt constants
at runtime and preserve their published orchestration, while replacing only
provider and benchmark I/O boundaries so all methods use the same model,
ledger, datasets, and final scorer.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .training_free.types import BenchmarkTask


def verify_checkout(root: Path, expected_revision: str) -> str:
    root = root.expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != expected_revision:
        raise ValueError(f"external baseline revision mismatch: {revision} != {expected_revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"external baseline has tracked modifications: {root}")
    return revision


def source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_string_constants(path: Path, names: Sequence[str]) -> Dict[str, Any]:
    """Read literal constants without importing an upstream dependency tree."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    values: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                literal = ast.literal_eval(value)
                values[target.id] = literal
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"missing upstream prompt constants in {path}: {sorted(missing)}")
    return values


@dataclass
class BaselineResult:
    prediction: str
    trace: Dict[str, Any]


class AuditedCaller:
    def __init__(
        self,
        client: Any,
        *,
        method: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.client = client
        self.method = method
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, messages: List[Dict[str, str]], stage: str) -> str:
        completion = self.client.complete(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            metadata={"purpose": f"external_baseline_{self.method}", "stage": stage},
        )
        self.calls.append({
            "stage": stage,
            "prompt_tokens": int(completion.prompt_tokens),
            "completion_tokens": int(completion.completion_tokens),
            "total_tokens": int(completion.total_tokens),
        })
        return str(completion.content)


def _task_contract(task: BenchmarkTask) -> str:
    if task.benchmark == "MATH":
        return (
            f"{task.instruction}\n\nShow the reasoning, then put the final answer in "
            "\\boxed{...}. Put only the requested mathematical value inside the box; "
            "do not append prose or physical units such as degrees."
        )
    tests = "\n".join(task.public_tests)
    public = f"\n\nPublic tests available to all methods:\n{tests}" if tests else ""
    return (
        f"{task.instruction}{public}\n\nReturn only a complete Python implementation of "
        f"{task.evaluator_payload['entry_point']}, preferably in one Python code block."
    )


def _last_boxed(text: str) -> str:
    marker = text.rfind("\\boxed")
    if marker < 0:
        matches = re.findall(r"(?:final answer|the answer is)\s*[:=]?\s*([^\n]+)", text, re.I)
        return re.sub(r"\s+", "", matches[-1]) if matches else ""
    start = text.find("{", marker)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", "", text[start + 1:index])
    return ""


def _consensus(responses: Sequence[str], benchmark: str) -> str | None:
    if benchmark != "MATH":
        return None
    answers = [_last_boxed(item) for item in responses]
    counts = Counter(item for item in answers if item)
    if not counts:
        return None
    answer, count = counts.most_common(1)[0]
    return next(item for item in responses if _last_boxed(item) == answer) if count > 2 else None


def _rank_indices(text: str, size: int) -> List[int]:
    pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not pairs:
        return list(range(min(2, size)))
    result: List[int] = []
    for raw in pairs[-1]:
        index = min(max(int(raw) - 1, 0), size - 1)
        if index not in result:
            result.append(index)
    return result or [0]


class DyLANBaseline:
    method = "dylan"

    def __init__(self, root: Path, caller: AuditedCaller) -> None:
        self.root = root.resolve()
        self.caller = caller
        self.math_source = self.root / "code/MATH/llmlp_gen_math_listwise_cot.py"
        self.code_source = self.root / "code/HumanEval/prompt_lib.py"
        self.sources = [
            self.math_source,
            self.root / "code/MATH/util.py",
            self.code_source,
            self.root / "code/HumanEval/CoLLMLP.py",
            self.root / "code/HumanEval/LLM_Neuron.py",
            self.root / "code/HumanEval/utils.py",
        ]
        self.math = load_string_constants(self.math_source, ["SYSTEM_PROMPT", "EXAMPLES"])
        self.code = load_string_constants(self.code_source, ["ROLE_MAP", "ROLE_MAP_INIT"])

    def run(self, task: BenchmarkTask) -> BaselineResult:
        return self._math(task) if task.benchmark == "MATH" else self._code(task)

    def _math(self, task: BenchmarkTask) -> BaselineResult:
        question = (
            self.math["EXAMPLES"] + "\n\nPlease solve the problem below.\nProblem: "
            + task.instruction + "\nAnswer:"
        )
        system = self.math["SYSTEM_PROMPT"]
        responses: List[str] = []
        for index in range(4):
            responses.append(self.caller([
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ], f"round_0_agent_{index}"))
            agreed = _consensus(responses, "MATH")
            if agreed:
                return BaselineResult(agreed, {"early_stop": "round_0_consensus", "active_agents": len(responses)})

        advice = _math_advice_message(task.instruction, responses)
        second = [self.caller([
            {"role": "system", "content": system}, {"role": "user", "content": advice},
        ], f"round_1_agent_{index}") for index in range(4)]
        agreed = _consensus(second, "MATH")
        if agreed:
            return BaselineResult(agreed, {"early_stop": "round_1_consensus", "active_agents": 4})

        ranking_prompt = _ranking_message(task, second)
        ranking = self.caller([{"role": "user", "content": ranking_prompt}], "listwise_ranker")
        selected = _rank_indices(ranking, len(second))
        finalists = [second[index] for index in selected]
        final_advice = _math_advice_message(task.instruction, finalists)
        final = [self.caller([
            {"role": "system", "content": system}, {"role": "user", "content": final_advice},
        ], f"round_2_agent_{index}") for index in range(len(finalists))]
        return BaselineResult(final[0], {
            "early_stop": None, "selected_agent_indices": selected,
            "candidate_final_answers": [_last_boxed(item) for item in final],
        })

    def _code(self, task: BenchmarkTask) -> BaselineResult:
        roles = ["PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer"]
        role_map = self.code["ROLE_MAP"]
        question = _task_contract(task)
        initial = [self.caller([
            {"role": "system", "content": role_map[role]},
            {"role": "user", "content": question},
        ], f"round_0_{role}") for role in roles]
        ranking = self.caller(
            [{"role": "user", "content": _ranking_message(task, initial)}],
            "round_1_ranker",
        )
        selected = _rank_indices(ranking, len(initial))
        reviews = self.caller([{"role": "user", "content": _code_review_message(task, initial)}], "round_1_reflector")
        finalists = [initial[index] for index in selected]
        refinement = _code_refinement_message(task, finalists, reviews)
        final = [self.caller([
            {"role": "system", "content": role_map[roles[index]]},
            {"role": "user", "content": refinement},
        ], f"round_2_{roles[index]}") for index in selected]
        final_ranking = self.caller(
            [{"role": "user", "content": _ranking_message(task, final, choose=1)}],
            "final_ranker",
        )
        winner = _rank_indices(final_ranking, len(final))[0]
        return BaselineResult(final[winner], {
            "roles": roles, "selected_agent_indices": selected, "winner": winner,
            "public_tests_exposed": bool(task.public_tests),
        })


def _math_advice_message(question: str, responses: Sequence[str]) -> str:
    joined = "".join(f"\n\nOne agent's solution: ```{item}```" for item in responses)
    return (
        "Follow the given examples and answer the mathematics problem.\n\n" + question
        + "\n\nThese are the solutions to the problem from other agents: " + joined
        + "\n\nUsing the reasoning from other agents as additional advice with critical "
          "thinking, give an updated answer. The former answers might all be wrong. "
          "Put the final answer in \\boxed{...}."
    )


def _ranking_message(task: BenchmarkTask, responses: Sequence[str], choose: int = 2) -> str:
    entries = "\n\n".join(
        f"Agent solution {index}: ```{item}```" for index, item in enumerate(responses, 1)
    )
    return (
        f"Here is the question:\n{_task_contract(task)}\n\nCandidates:\n{entries}\n\n"
        f"Choose the best {choose} solution(s) for correctness and edge cases. "
        "For compatibility, finish with two indices like [1,2] (repeat the winner if choosing one)."
    )


def _code_review_message(task: BenchmarkTask, responses: Sequence[str]) -> str:
    entries = "\n\n".join(
        f"[implementation {index}]\n```python\n{item}\n```" for index, item in enumerate(responses, 1)
    )
    return (
        f"Function task:\n{_task_contract(task)}\n\n{entries}\n\nReview each implementation "
        "for syntax, correctness, efficiency, and corner cases. Do not reveal or invent hidden tests."
    )


def _code_refinement_message(task: BenchmarkTask, responses: Sequence[str], review: str) -> str:
    entries = "\n\n".join(f"```python\n{item}\n```" for item in responses)
    return (
        f"Complete the function in this task:\n{_task_contract(task)}\n\nCandidate implementations:\n"
        f"{entries}\n\nReview:\n{review}\n\nReturn a corrected complete implementation only."
    )


class EvoAgentBaseline:
    method = "evoagent"

    def __init__(self, root: Path, caller: AuditedCaller, iterations: int = 3) -> None:
        self.root = root.resolve()
        self.caller = caller
        self.iterations = iterations
        self.prompt_source = self.root / "spp/agent_prompt_writing.py"
        self.control_source = self.root / "spp/util_func.py"
        self.sources = [self.prompt_source, self.control_source]
        self.prompts = load_string_constants(self.prompt_source, [
            "INSTRUCTION_META", "INSTRUCTION_CHECK", "INSTRUCTION_MULTI", "INSTRUCTION_REFINE",
        ])

    def run(self, task: BenchmarkTask) -> BaselineResult:
        question = _task_contract(task)
        answer = self.caller([{"role": "user", "content": question}], "initial_answer")
        descriptions: List[str] = []
        history: List[Dict[str, Any]] = []
        for iteration in range(self.iterations):
            attempts = 0
            while True:
                description = self.caller([{"role": "user", "content": self.prompts["INSTRUCTION_META"].format(
                    question=question, answer=answer, description="\n-".join(descriptions),
                )}], f"iteration_{iteration}_meta")
                check = self.caller([{"role": "user", "content": self.prompts["INSTRUCTION_CHECK"].format(
                    question=question, description_ls="\n-".join(descriptions), description=description,
                )}], f"iteration_{iteration}_check_{attempts}")
                if "discard" not in check.lower() or attempts > 3:
                    descriptions.append(description)
                    break
                attempts += 1
            sub_answer = self.caller([{"role": "user", "content": self.prompts["INSTRUCTION_MULTI"].format(
                question=question, description=description,
            )}], f"iteration_{iteration}_expert")
            new_answer = self.caller([{"role": "user", "content": self.prompts["INSTRUCTION_REFINE"].format(
                question=question, old_answer=answer, description=description, new_answer=sub_answer,
            )}], f"iteration_{iteration}_refine")
            history.append({
                "iteration": iteration, "description": description, "check": check,
                "discard_attempts": attempts,
            })
            answer = new_answer
        return BaselineResult(answer, {"evolved_agents": history, "iterations": self.iterations})


class AutoAgentsBaseline:
    method = "autoagents"

    def __init__(self, root: Path, caller: AuditedCaller, review_rounds: int = 1) -> None:
        self.root = root.resolve()
        self.caller = caller
        self.review_rounds = review_rounds
        action_root = self.root / "autoagents/actions"
        self.source_paths = {
            "create": action_root / "create_roles.py",
            "roles": action_root / "check_roles.py",
            "plans": action_root / "check_plans.py",
            "custom": action_root / "custom_action.py",
        }
        self.sources = [
            *self.source_paths.values(),
            self.root / "autoagents/roles/manager.py",
            self.root / "autoagents/roles/group.py",
        ]
        self.create = load_string_constants(self.source_paths["create"], ["PROMPT_TEMPLATE", "FORMAT_EXAMPLE", "TOOLS"])
        self.check_roles = load_string_constants(self.source_paths["roles"], ["PROMPT_TEMPLATE", "FORMAT_EXAMPLE", "TOOLS"])
        self.check_plans = load_string_constants(self.source_paths["plans"], ["PROMPT_TEMPLATE", "FORMAT_EXAMPLE", "TOOLS"])
        self.custom = load_string_constants(self.source_paths["custom"], ["PROMPT_TEMPLATE", "FORMAT_EXAMPLE"])

    def run(self, task: BenchmarkTask) -> BaselineResult:
        question = _task_contract(task)
        roles_plan = self._plan(question, "", "")
        reviews: List[Dict[str, str]] = []
        for iteration in range(self.review_rounds):
            sections = _sections(roles_plan)
            role_review = self.caller([{"role": "user", "content": self.check_roles["PROMPT_TEMPLATE"].format(
                question=question,
                existing_roles="[]",
                selected_roles=sections.get("Selected Roles List", ""),
                created_roles=sections.get("Created Roles List", ""),
                history="",
                tools=self.check_roles["TOOLS"],
                format_example=self.check_roles["FORMAT_EXAMPLE"],
            )}], f"review_{iteration}_roles")
            plan_review = self.caller([{"role": "user", "content": self.check_plans["PROMPT_TEMPLATE"].format(
                context=question,
                roles=sections.get("Selected Roles List", "") + sections.get("Created Roles List", ""),
                plan=sections.get("Execution Plan", ""),
                history="",
                tools=self.check_plans["TOOLS"],
                format_example=self.check_plans["FORMAT_EXAMPLE"],
            )}], f"review_{iteration}_plan")
            reviews.append({"role_review": role_review, "plan_review": plan_review})
            roles_plan = self._plan(question, roles_plan, f"{role_review}\n{plan_review}")

        sections = _sections(roles_plan)
        roles = _json_objects(
            sections.get("Selected Roles List", "") + "\n" + sections.get("Created Roles List", "")
        )
        if not roles:
            roles = [{
                "name": "Language Expert", "description": "Solve and synthesize the task.",
                "prompt": "You are the final language expert.", "suggestions": "Be accurate.", "tools": [],
            }]
        steps = _plan_steps(sections.get("Execution Plan", "")) or ["Solve the task", "Synthesize the final answer"]
        completed = ""
        executed: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            chosen = [role for role in roles if str(role.get("name", "")).lower() in step.lower()]
            if not chosen:
                chosen = [roles[-1] if index == len(steps) - 1 else roles[min(index, len(roles) - 1)]]
            for role in chosen[:4]:
                prompt = self.custom["PROMPT_TEMPLATE"].format(
                    role=role.get("prompt") or role.get("description") or role.get("name"),
                    context=f"{question}\n\nCurrent planned step: {step}",
                    suggestions=role.get("suggestions", ""),
                    previous=completed,
                    completed_steps=completed,
                    tool="['Final Output']",
                    format_example=self.custom["FORMAT_EXAMPLE"],
                )
                response = self.caller([{"role": "user", "content": prompt}], f"step_{index}_{_slug(str(role.get('name', 'role')))}")
                completed += f"\n\n[{role.get('name', 'Role')} / {step}]\n{response}"
                executed.append({"step": step, "role": role.get("name", "Role")})
        final = self.caller([{"role": "user", "content": (
            f"Task:\n{question}\n\nExpert work:\n{completed}\n\nSynthesize the final answer now. "
            "Obey the task's output format and do not mention this collaboration."
        )}], "final_synthesis")
        return BaselineResult(final, {
            "generated_roles": [{"name": item.get("name"), "description": item.get("description")} for item in roles],
            "execution_plan": steps, "executed": executed, "observer_reviews": reviews,
        })

    def _plan(self, question: str, history: str, suggestions: str) -> str:
        prompt = self.create["PROMPT_TEMPLATE"].format(
            context=question, existing_roles="[]", history=history, suggestions=suggestions,
            tools="None", format_example=self.create["FORMAT_EXAMPLE"],
        )
        return self.caller([{"role": "user", "content": prompt}], "planner")


def _sections(text: str) -> Dict[str, str]:
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    result: Dict[str, str] = {}
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        result[title.strip().rstrip(":")] = body.strip()
    return result


def _json_objects(text: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("name") and value not in objects:
            objects.append(value)
    return objects


def _plan_steps(text: str) -> List[str]:
    return [match.group(1).strip() for match in re.finditer(r"^\s*\d+[.)]\s*(.+)$", text, re.MULTILINE)]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "role"


def build_baseline(
    method: str,
    root: Path,
    caller: AuditedCaller,
    *,
    evoagent_iterations: int = 3,
    autoagents_review_rounds: int = 1,
) -> Any:
    if method == "dylan":
        return DyLANBaseline(root, caller)
    if method == "evoagent":
        return EvoAgentBaseline(root, caller, evoagent_iterations)
    if method == "autoagents":
        return AutoAgentsBaseline(root, caller, autoagents_review_rounds)
    raise ValueError(f"unknown external baseline: {method}")
