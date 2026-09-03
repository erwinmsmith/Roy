from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from .types import AgentState, BenchmarkTask


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    arguments: Dict[str, Any]
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolRequest":
        name = str(value.get("tool_name", "")).strip()
        arguments = value.get("arguments", {})
        reason = str(value.get("reason", "")).strip()
        if not name or not isinstance(arguments, Mapping):
            raise ValueError("tool request requires tool_name and object arguments")
        return cls(name, dict(arguments), reason)


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""


@dataclass(frozen=True)
class ToolCallRecord:
    agent_id: str
    tool_name: str
    arguments: Dict[str, Any]
    reason: str
    success: bool
    output: str
    error: str
    duration_ms: int
    sandbox_backend: str
    scope: str


@dataclass
class ToolAudit:
    records: List[ToolCallRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": [asdict(record) for record in self.records],
            "total_calls": len(self.records),
            "successful_calls": sum(record.success for record in self.records),
            "failed_calls": sum(not record.success for record in self.records),
        }


class TaskToolRegistry:
    """Task-scoped, capability-checked tools with no hidden-test access."""

    def __init__(
        self,
        task: BenchmarkTask,
        *,
        code_sandbox_prefix: Sequence[str] = (),
        timeout_seconds: int = 5,
        maximum_output_chars: int = 20_000,
    ) -> None:
        self.task = task
        self.code_sandbox_prefix = list(code_sandbox_prefix)
        self.timeout_seconds = timeout_seconds
        self.maximum_output_chars = maximum_output_chars
        self.audit = ToolAudit()

    @property
    def catalog(self) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = [{
            "name": "symbolic_math",
            "description": "Safely simplify, factor, expand, solve, or evaluate one math expression.",
            "arguments": {
                "operation": "simplify|factor|expand|solve|evaluate",
                "expression": "arithmetic/symbolic expression or equation",
                "variable": "required for solve; optional otherwise",
                "substitutions": "optional object of symbol to numeric value",
            },
        }]
        if self.code_sandbox_prefix:
            values.append({
                "name": "python",
                "description": "Run a bounded Python snippet in the configured process sandbox.",
                "arguments": {"code": "Python source"},
            })
            if self.task.benchmark == "HumanEval" and self.task.public_tests:
                values.append({
                    "name": "public_tests",
                    "description": (
                        "Run a complete HumanEval candidate against only the task's allowed public tests."
                    ),
                    "arguments": {"code": "complete Python solution with the entry-point function"},
                })
        return values

    def execute(
        self,
        agent: AgentState,
        request: ToolRequest,
        *,
        scope: str = "committed",
    ) -> ToolResult:
        if scope not in ("committed", "counterfactual"):
            raise ValueError(f"invalid tool-call scope: {scope}")
        started = time.monotonic()
        backend = "safe_symbolic_ast"
        try:
            if request.tool_name not in agent.tools:
                result = ToolResult(False, error="tool is not assigned to this agent")
            elif request.tool_name == "symbolic_math":
                result = self._symbolic_math(request.arguments)
            elif request.tool_name == "python":
                backend = self._sandbox_backend()
                result = self._python(request.arguments)
            elif request.tool_name == "public_tests":
                backend = self._sandbox_backend()
                result = self._public_tests(agent, request.arguments)
            else:
                result = ToolResult(False, error=f"unknown tool: {request.tool_name}")
        except (TypeError, ValueError, KeyError) as error:
            result = ToolResult(False, error=f"{type(error).__name__}: {error}")
        duration_ms = int((time.monotonic() - started) * 1000)
        self.audit.records.append(ToolCallRecord(
            agent_id=agent.agent_id,
            tool_name=request.tool_name,
            arguments=_bounded_arguments(request.arguments),
            reason=request.reason,
            success=result.success,
            output=result.output[: self.maximum_output_chars],
            error=result.error[: self.maximum_output_chars],
            duration_ms=duration_ms,
            sandbox_backend=backend,
            scope=scope,
        ))
        return result

    def _symbolic_math(self, arguments: Mapping[str, Any]) -> ToolResult:
        operation = str(arguments.get("operation", "")).lower()
        if operation not in ("simplify", "factor", "expand", "solve", "evaluate"):
            raise ValueError("unsupported symbolic_math operation")
        expression = str(arguments.get("expression", ""))
        variable = str(arguments.get("variable", ""))
        substitutions = arguments.get("substitutions", {})
        if not expression or len(expression) > 2_000:
            raise ValueError("expression must contain 1..2000 characters")
        if not isinstance(substitutions, Mapping):
            raise TypeError("substitutions must be an object")
        return self._run_python(
            _SYMBOLIC_RUNNER,
            {"operation": operation, "expression": expression,
             "variable": variable, "substitutions": dict(substitutions)},
            prefix=(),
            isolated=False,
        )

    def _python(self, arguments: Mapping[str, Any]) -> ToolResult:
        self._require_code_sandbox()
        code = str(arguments.get("code", ""))
        if not code or len(code) > 20_000:
            raise ValueError("code must contain 1..20000 characters")
        return self._run_python(code, None, prefix=self.code_sandbox_prefix, isolated=True)

    def _public_tests(self, agent: AgentState, arguments: Mapping[str, Any]) -> ToolResult:
        self._require_code_sandbox()
        if self.task.benchmark != "HumanEval":
            raise ValueError("public_tests is only available for HumanEval")
        code = str(arguments.get("code", ""))
        entry_point = str(self.task.evaluator_payload.get("entry_point", ""))
        allowed_tests = list(agent.context.public_tests)
        if not code or not entry_point:
            raise ValueError("code and HumanEval entry point are required")
        if not allowed_tests:
            raise ValueError("this agent has no allowed public tests")
        return self._run_python(
            _PUBLIC_TEST_RUNNER,
            {"code": code, "entry_point": entry_point, "tests": allowed_tests},
            prefix=self.code_sandbox_prefix,
            isolated=True,
        )

    def _run_python(
        self,
        code: str,
        payload: Dict[str, Any] | None,
        *,
        prefix: Sequence[str],
        isolated: bool,
    ) -> ToolResult:
        interpreter = os.path.realpath(sys.executable) if isolated else sys.executable
        command = [*prefix, interpreter]
        if isolated:
            command.append("-I")
        command.extend(["-c", code])
        try:
            completed = subprocess.run(
                command,
                cwd="/" if isolated else None,
                input="" if payload is None else json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                preexec_fn=_resource_limits(self.timeout_seconds) if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, error=f"tool timed out after {self.timeout_seconds}s")
        stdout = completed.stdout[: self.maximum_output_chars]
        stderr = completed.stderr[: self.maximum_output_chars]
        if completed.returncode != 0:
            return ToolResult(False, output=stdout, error=stderr or f"exit code {completed.returncode}")
        return ToolResult(True, output=stdout.strip())

    def _require_code_sandbox(self) -> None:
        if not self.code_sandbox_prefix:
            raise ValueError("code execution is disabled until a reviewed sandbox prefix is configured")

    def _sandbox_backend(self) -> str:
        return " ".join(self.code_sandbox_prefix[:2]) if self.code_sandbox_prefix else "disabled"


def macos_readonly_sandbox_prefix() -> List[str]:
    if platform.system() != "Darwin" or not os.path.exists("/usr/bin/sandbox-exec"):
        return []
    runtime_paths = sorted({os.path.realpath(sys.executable), os.path.realpath(sys.base_prefix)})
    ancestors = {"/"}
    for path in runtime_paths:
        parent = os.path.dirname(path)
        while parent and parent != "/":
            ancestors.add(parent)
            parent = os.path.dirname(parent)
    ancestor_rules = " ".join(
        f'(literal "{path.replace(chr(34), "")}")' for path in sorted(ancestors)
    )
    runtime_rules = " ".join(
        f'(subpath "{path.replace(chr(34), "")}")' for path in runtime_paths
    )
    profile = " ".join((
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        f'(allow file-read* {ancestor_rules} (subpath "/usr") (subpath "/System")',
        f'(subpath "/Library") (subpath "/Applications") {runtime_rules})',
    ))
    return ["/usr/bin/sandbox-exec", "-p", profile]


def _resource_limits(timeout_seconds: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
            except (OSError, ValueError):
                pass
    return apply


def _bounded_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    value = json.loads(json.dumps(dict(arguments), default=str))
    for key, item in list(value.items()):
        if isinstance(item, str) and len(item) > 20_000:
            value[key] = item[:20_000] + "...[truncated]"
    return value


_SYMBOLIC_RUNNER = r'''
import ast
import json
import sys
import sympy as sp

payload = json.load(sys.stdin)
allowed_calls = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
    "log": sp.log, "exp": sp.exp, "Abs": sp.Abs,
}

def convert(node):
    if isinstance(node, ast.Expression):
        return convert(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
    if isinstance(node, ast.Name):
        if node.id in ("pi", "E"):
            return {"pi": sp.pi, "E": sp.E}[node.id]
        if not node.id.isidentifier() or node.id.startswith("_"):
            raise ValueError("invalid symbol")
        return sp.Symbol(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = convert(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = convert(node.left), convert(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow):
            if getattr(right, "is_number", False) and abs(float(right)) > 100:
                raise ValueError("exponent is too large")
            return left ** right
        raise ValueError("operator is not allowed")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = allowed_calls.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("function is not allowed")
        return function(*(convert(arg) for arg in node.args))
    raise ValueError(f"syntax is not allowed: {type(node).__name__}")

def parse(text):
    return convert(ast.parse(text.replace("^", "**"), mode="eval"))

text = payload["expression"]
if "=" in text:
    left, right = text.split("=", 1)
    expression = sp.Eq(parse(left), parse(right))
else:
    expression = parse(text)
subs = {sp.Symbol(str(key)): parse(str(value)) for key, value in payload["substitutions"].items()}
operation = payload["operation"]
if operation == "solve":
    variable = payload["variable"]
    if not variable.isidentifier() or variable.startswith("_"):
        raise ValueError("solve requires a valid variable")
    result = sp.solve(expression, sp.Symbol(variable))
elif operation == "simplify": result = sp.simplify(expression)
elif operation == "factor": result = sp.factor(expression)
elif operation == "expand": result = sp.expand(expression)
elif operation == "evaluate": result = sp.N(expression.subs(subs))
else: raise ValueError("unsupported operation")
print(json.dumps({"result": str(result)}, ensure_ascii=False))
'''


_PUBLIC_TEST_RUNNER = r'''
import json
import sys

payload = json.load(sys.stdin)
namespace = {}
exec(payload["code"], namespace)
entry_point = payload["entry_point"]
if entry_point not in namespace or not callable(namespace[entry_point]):
    raise ValueError(f"entry point {entry_point!r} is not defined")
namespace["candidate"] = namespace[entry_point]
passed = 0
for index, test in enumerate(payload["tests"]):
    try:
        exec(test, namespace)
        passed += 1
    except Exception as error:
        print(json.dumps({
            "passed": passed, "total": len(payload["tests"]),
            "failed_index": index, "error": f"{type(error).__name__}: {error}",
        }))
        raise SystemExit(1)
print(json.dumps({"passed": passed, "total": len(payload["tests"]), "all_passed": True}))
'''
