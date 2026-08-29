from __future__ import annotations

import asyncio
import atexit
import json
import os
import queue
import shlex
import subprocess
import threading
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


class JSONRPCError(RuntimeError):
    pass


class RolloutDeadlineReached(RuntimeError):
    pass


class PersistentNodeRPC:
    """Timeout-aware JSONL RPC client with one automatic restore after restart."""

    def __init__(self, command: Sequence[str], timeout: float = 120.0,
                 environment: Mapping[str, str] | None = None,
                 stderr_path: Path | None = None) -> None:
        self.command = list(command)
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[str] = queue.Queue()
        self.next_id = 0
        self.last_snapshot: Mapping[str, Any] | None = None
        self.environment = dict(environment or os.environ)
        self.stderr_path = stderr_path

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=self.environment,
        )
        assert self.process.stdout is not None
        threading.Thread(target=self._read_stdout, args=(self.process.stdout,), daemon=True).start()
        assert self.process.stderr is not None
        threading.Thread(target=self._read_stderr, args=(self.process.stderr,), daemon=True).start()

    def request(self, method: str, params: Mapping[str, Any], retry: bool = True,
                timeout: float | None = None) -> Mapping[str, Any]:
        self.start()
        self.next_id += 1
        request_id = self.next_id
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps({
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
            }) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Roy JSON-RPC {method} timed out")
                response = json.loads(self.responses.get(timeout=remaining))
                if response.get("id") != request_id:
                    continue
                if response.get("error"):
                    raise JSONRPCError(str(response["error"].get("message")))
                result = dict(response.get("result") or {})
                if isinstance(result.get("snapshot"), Mapping):
                    self.last_snapshot = result["snapshot"]
                return result
        except (BrokenPipeError, EOFError, queue.Empty, TimeoutError):
            if not retry:
                raise
            self.restart()
            return self.request(method, params, retry=False, timeout=timeout)

    def restart(self, timeout: float | None = None) -> None:
        self.close()
        self.responses = queue.Queue()
        self.start()
        if self.last_snapshot is not None:
            self.request("restore", {"snapshot": self.last_snapshot}, retry=False,
                         timeout=timeout)

    def close(self) -> None:
        if not self.process:
            return
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self.process = None

    def _read_stdout(self, stream: Any) -> None:
        for line in stream:
            if line.strip():
                self.responses.put(line)

    def _read_stderr(self, stream: Any) -> None:
        for line in stream:
            if self.stderr_path is not None:
                self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
                with self.stderr_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)


try:
    from harbor.agents.base import BaseAgent
except ImportError:  # Harbor remains an optional dependency outside the pinned checkout.
    class BaseAgent:  # type: ignore[no-redef]
        def __init__(self, logs_dir: Path, model_name: str | None = None, **_: Any) -> None:
            self.logs_dir = logs_dir
            self.model_name = model_name


class RoyHarborAgent(BaseAgent):
    """Thin Harbor lifecycle adapter; all organization state lives in Node Roy."""

    SUPPORTS_ATIF = False

    def __init__(self, *args: Any, node_command: str | None = None,
                 rpc_timeout: float = 120.0, track_file_changes: bool = True,
                 rollout_timeout_sec: float | None = None,
                 extra_env: Mapping[str, str] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configured = node_command or os.environ.get(
            "ROY_LHTB_NODE_COMMAND", "node dist/cli/LhtbAgent.js"
        )
        self.semantic_root = Path(self.logs_dir) / "semantic"
        self.runtime_audit_root = Path(self.logs_dir) / "runtime-audit"
        child_environment = dict(os.environ)
        child_environment.update({str(key): str(value)
                                  for key, value in (extra_env or {}).items()})
        child_environment["ROY_LHTB_SEMANTIC_ROOT"] = str(self.semantic_root)
        child_environment["ROY_LHTB_AUDIT_ROOT"] = str(self.runtime_audit_root)
        self.agent_environment = child_environment
        if child_environment.get("HB_CONTINUE_MODE"):
            os.environ["HB_CONTINUE_MODE"] = child_environment["HB_CONTINUE_MODE"]
        self.rpc = PersistentNodeRPC(
            shlex.split(configured), timeout=rpc_timeout, environment=child_environment,
            stderr_path=self.runtime_audit_root / "child-stderr.log",
        )
        atexit.register(self.close)
        self.partial_path = Path(self.logs_dir) / "roy-partial-trajectory.json"
        self.track_file_changes = track_file_changes
        self._environment: Any | None = None
        self._trajectory_id: str | None = None
        self._continuation_count = 0
        self.rollout_timeout_sec = rollout_timeout_sec
        # One Harbor trial calls ``run`` once and may then call
        # ``resume_after_verifier_rejection`` many times. Keep one deadline for
        # that complete lifecycle so the outer Harbor timeout never wins.
        self._rollout_deadline: float | None = None
        self.partial_save_interval_sec = max(0.0, float(child_environment.get(
            "ROY_LHTB_PARTIAL_SAVE_INTERVAL_SEC", "30"
        )))
        self._last_partial_save = 0.0

    @staticmethod
    def name() -> str:
        return "roy-lhtb-agent"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: Any) -> None:
        self._environment = environment
        self.rpc.start()

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        self._rollout_deadline = (
            time.monotonic() + self.rollout_timeout_sec
            if self.rollout_timeout_sec is not None else None
        )
        deadline = self._rollout_deadline
        try:
            trajectory_id = str(uuid.uuid4())
            self._environment = environment
            self._trajectory_id = trajectory_id
            base_seed = int(self.agent_environment.get(
                "ROY_LHTB_ORGANIZATION_SEED", "20260820"
            ))
            session_seed = int(hashlib.sha256(
                str(getattr(environment, "session_id", trajectory_id)).encode("utf-8")
            ).hexdigest()[:8], 16)
            response = await self._rpc_before_deadline("run", {
                "trajectoryId": trajectory_id,
                "taskId": getattr(environment, "environment_name", "unknown"),
                "instruction": instruction,
                "environmentRevision": self.agent_environment.get("LHTB_COMMIT", "pinned"),
                "organizationMode": self.agent_environment.get(
                    "ROY_LHTB_ARM", "learned_information_realization"
                ),
                "organizationSeed": base_seed ^ session_seed,
                "initialSnapshotFingerprint": self.agent_environment.get(
                    "ROY_LHTB_INITIAL_FINGERPRINT", ""
                ),
            }, deadline)
            self._save_partial()
            await self._drive(response, environment, deadline)
            self._save_partial(force=True)
            self._update_context(context, termination_reason="confirmed_task_complete")
        except RolloutDeadlineReached:
            await self._finalize_rollout_deadline()
            self._save_partial(force=True)
            self._update_context(context, termination_reason="rollout_deadline")
        except BaseException:
            self._save_partial(force=True)
            self.close()
            raise

    async def resume_after_verifier_rejection(
        self, user_prompt: str, context: Any
    ) -> None:
        """Compatible with official same-conversation mode when a provider supports it."""
        if self.rpc.last_snapshot is None or self._environment is None:
            raise RuntimeError("Roy has no session to resume")
        try:
            self._continuation_count += 1
            response = await self._rpc_before_deadline("verifier_rejection", {
                "feedback": user_prompt,
            }, self._rollout_deadline)
            self._save_partial()
            await self._drive(response, self._environment, self._rollout_deadline)
            self._save_partial(force=True)
            self._update_context(context, termination_reason="confirmed_task_complete")
        except RolloutDeadlineReached:
            await self._finalize_rollout_deadline()
            self._save_partial(force=True)
            self._update_context(context, termination_reason="rollout_deadline")
        except BaseException:
            self._save_partial(force=True)
            self.close()
            raise

    def _update_context(self, context: Any, termination_reason: str) -> None:
        context.metadata = {
            **(context.metadata or {}),
            "roy_trajectory_id": self._trajectory_id,
            "roy_partial_trajectory": str(self.partial_path),
            "roy_semantic_audit_root": str(self.semantic_root),
            "roy_runtime_audit_root": str(self.runtime_audit_root),
            "same_conversation_continuations": self._continuation_count,
            "termination_reason": termination_reason,
        }
        snapshot = self.rpc.last_snapshot
        if snapshot:
            states = list(snapshot.get("processStates", []))
            usage = dict(states[-1].get("usage", {})) if states else {}
            context.n_input_tokens = int(usage.get("inputTokens", 0))
            context.n_output_tokens = int(usage.get("outputTokens", 0))

    async def _drive(self, response: Mapping[str, Any], environment: Any,
                     deadline: float | None = None) -> None:
        while response.get("status") in ("terminal_request", "continue"):
            self._remaining(deadline)
            self._save_partial()
            if response.get("status") == "continue":
                response = await self._rpc_before_deadline("advance", {}, deadline)
                self._save_partial()
                continue
            request = dict(response["request"])
            started = time.monotonic()
            before_files = await self._await_before_deadline(
                self._file_snapshot(environment, request.get("cwd")), deadline
            )
            requested_timeout = max(
                1, int(float(request.get("timeoutMs", 120000)) / 1000)
            )
            remaining = self._remaining(deadline)
            command_timeout = requested_timeout if remaining is None else max(
                1, min(requested_timeout, int(remaining))
            )
            result = await self._await_before_deadline(environment.exec(
                str(request["command"]), cwd=request.get("cwd"),
                timeout_sec=command_timeout,
            ), deadline)
            after_files = await self._await_before_deadline(
                self._file_snapshot(environment, request.get("cwd")), deadline
            )
            file_changes = sorted(
                path for path in set(before_files).union(after_files)
                if before_files.get(path) != after_files.get(path)
            )
            response = await self._rpc_before_deadline("resume", {"result": {
                "requestId": request["id"], "exitCode": result.return_code,
                "stdout": result.stdout or "", "stderr": result.stderr or "",
                "durationMs": int((time.monotonic() - started) * 1000),
                "fileChanges": file_changes,
            }}, deadline)
            self._save_partial()

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RolloutDeadlineReached("Roy reached its rollout deadline")
        return remaining

    async def _rpc_before_deadline(
        self, method: str, params: Mapping[str, Any], deadline: float | None
    ) -> Mapping[str, Any]:
        remaining = self._remaining(deadline)
        if remaining is None:
            return await asyncio.to_thread(self.rpc.request, method, params)
        request_timeout = max(0.01, min(self.rpc.timeout, remaining))
        retry = remaining > (2 * self.rpc.timeout + 10)
        try:
            return await asyncio.to_thread(
                self.rpc.request, method, params, retry, request_timeout
            )
        except (TimeoutError, queue.Empty) as exc:
            if deadline - time.monotonic() <= 0.1:
                raise RolloutDeadlineReached(
                    f"Roy reached its rollout deadline during {method}"
                ) from exc
            raise

    async def _await_before_deadline(self, awaitable: Any,
                                     deadline: float | None) -> Any:
        remaining = self._remaining(deadline)
        if remaining is None:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise RolloutDeadlineReached("Roy reached its rollout deadline") from exc

    async def _finalize_rollout_deadline(self) -> None:
        if self.rpc.last_snapshot is None:
            return
        try:
            await asyncio.to_thread(self.rpc.restart, 10.0)
            await asyncio.to_thread(
                self.rpc.request, "rollout_deadline",
                {"reason": "training_rollout_deadline"}, False, 10.0,
            )
        except BaseException:
            # The last append-only snapshot is still retained and saved below. Harbor must
            # return normally so its official verifier can score the partial environment.
            self.rpc.close()

    def _save_partial(self, force: bool = False) -> None:
        snapshot = self.rpc.last_snapshot
        if snapshot is None:
            return
        now = time.monotonic()
        if not force and self._last_partial_save > 0 \
                and now - self._last_partial_save < self.partial_save_interval_sec:
            return
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.partial_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False,
                                        separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, self.partial_path)
        self._last_partial_save = now

    async def _file_snapshot(self, environment: Any, cwd: str | None) -> Dict[str, str]:
        if not self.track_file_changes:
            return {}
        root = cwd or "/workspace"
        command = (f"find {shlex.quote(root)} -xdev -type f "
                   "-printf '%P\\t%s\\t%T@\\n' 2>/dev/null | sort")
        result = await environment.exec(command, timeout_sec=30)
        if result.return_code != 0:
            return {}
        values: Dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                values[parts[0]] = parts[1]
        return values

    def close(self) -> None:
        self.rpc.close()


class FrozenFinalizeNowAgent(BaseAgent):
    """Fixed A0 readout for artifact-state value labels.

    The agent performs no terminal command and no Roy structural action. Harbor
    immediately invokes the task's original verifier on the current environment,
    so the resulting score measures the task value already realized in that
    checkpoint. Checkpoint restoration, when used, is owned and audited by the
    environment backend before this agent runs.
    """

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "roy-frozen-finalize-now"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: Any) -> None:
        restore_path = os.environ.get("ROY_LHTB_FINALIZE_CHECKPOINT")
        if restore_path:
            restore = getattr(environment, "restore_training_checkpoint", None)
            if not callable(restore):
                raise RuntimeError(
                    "forced-finalize checkpoint requires a clonable environment backend"
                )
            await restore(Path(restore_path))

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        context.metadata = {
            **(context.metadata or {}),
            "finalizer_policy": "frozen_finalize_now_no_structural_actions",
            "finalizer_revision": "artifact-identity-a0-v1",
            "structural_actions": 0,
            "terminal_commands": 0,
        }


class NodewiseCheckpointFinalizeAgent(BaseAgent):
    """Capture S_t or execute exactly one node macro-action before verification.

    A zero-step run materializes an auditable base checkpoint.  A one-step run
    restores that exact checkpoint and Roy session, samples one action for the
    scheduler-selected context node, completes only that action's terminal
    side effect (if any), then stops.  Harbor's verifier supplies U_T for the
    resulting state; it does not supply the derived reward R_t.
    """

    SUPPORTS_ATIF = False

    def __init__(
        self, *args: Any, macro_steps: int,
        output_snapshot_path: str, output_state_path: str,
        output_checkpoint_path: str,
        organization_seed: int = 20260820,
        initial_snapshot_fingerprint: str = "",
        environment_revision: str = "lhtb-pinned",
        source_snapshot_path: str | None = None,
        source_checkpoint_path: str | None = None,
        node_command: str | None = None, rpc_timeout: float = 720.0,
        extra_env: Mapping[str, str] | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if macro_steps not in (0, 1):
            raise ValueError("node-wise finalize permits exactly zero or one macro-action")
        if bool(source_snapshot_path) != bool(source_checkpoint_path):
            raise ValueError("source Roy snapshot and environment checkpoint must be paired")
        if macro_steps == 1 and not source_snapshot_path:
            raise ValueError("one-step node-wise finalize requires one exact source checkpoint")
        self.macro_steps = macro_steps
        self.organization_seed = int(organization_seed)
        self.initial_snapshot_fingerprint = initial_snapshot_fingerprint
        self.environment_revision = environment_revision
        self.output_snapshot_path = Path(output_snapshot_path).expanduser().resolve()
        self.output_state_path = Path(output_state_path).expanduser().resolve()
        self.output_checkpoint_path = Path(output_checkpoint_path).expanduser().resolve()
        self.source_snapshot_path = (
            Path(source_snapshot_path).expanduser().resolve() if source_snapshot_path else None
        )
        self.source_checkpoint_path = (
            Path(source_checkpoint_path).expanduser().resolve()
            if source_checkpoint_path else None
        )
        configured = node_command or os.environ.get(
            "ROY_LHTB_NODE_COMMAND", "node dist/cli/LhtbAgent.js"
        )
        child_environment = dict(os.environ)
        child_environment.update({str(key): str(value)
                                  for key, value in (extra_env or {}).items()})
        self.rpc = PersistentNodeRPC(
            shlex.split(configured), timeout=rpc_timeout, environment=child_environment,
            stderr_path=Path(self.logs_dir) / "nodewise-child-stderr.log",
        )
        self.environment: Any | None = None
        atexit.register(self.close)

    @staticmethod
    def name() -> str:
        return "roy-nodewise-checkpoint-finalize"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: Any) -> None:
        self.environment = environment
        if self.source_checkpoint_path is not None:
            restore = getattr(environment, "restore_training_checkpoint", None)
            if not callable(restore):
                raise RuntimeError("node-wise source requires a clonable environment backend")
            await restore(self.source_checkpoint_path)
        self.rpc.start()

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        self.environment = environment
        if self.source_snapshot_path is not None:
            source_snapshot = json.loads(
                self.source_snapshot_path.read_text(encoding="utf-8")
            )
            response = await asyncio.to_thread(
                self.rpc.request, "restore", {
                    "snapshot": source_snapshot,
                    "organizationSeed": self.organization_seed,
                }
            )
            # A saved macro successor is a valid SMDP boundary, but terminal/tool
            # events produced by that macro may not yet have been projected into
            # the next node's semantic state. Materialize that deterministic
            # boundary before fingerprinting or sampling the next action.
            response = await asyncio.to_thread(
                self.rpc.request, "prepare_boundary", {}
            )
        else:
            trajectory_id = str(uuid.uuid4())
            response = await asyncio.to_thread(self.rpc.request, "initialize", {
                "trajectoryId": trajectory_id,
                "taskId": getattr(environment, "environment_name", "unknown"),
                "instruction": instruction,
                "environmentRevision": self.environment_revision,
                "organizationMode": "learned_information_realization",
                "organizationSeed": self.organization_seed,
                "initialSnapshotFingerprint": self.initial_snapshot_fingerprint,
            })
            response = await asyncio.to_thread(
                self.rpc.request, "prepare_boundary", {}
            )
        base_snapshot = dict(response.get("snapshot") or {})
        base_policy_count = len(base_snapshot.get("policyRecords") or [])
        base_states = list(base_snapshot.get("processStates") or [])
        if not base_states or not str(base_states[-1].get("fingerprint") or ""):
            raise RuntimeError("node-wise base is not a fingerprinted decision boundary")
        base_fingerprint = str(base_states[-1]["fingerprint"])
        macro_terminal_commands = 0
        if self.macro_steps == 1:
            response = await asyncio.to_thread(self.rpc.request, "advance_one", {})
            if response.get("status") == "terminal_request":
                request = dict(response["request"])
                started = time.monotonic()
                before_files = await self._file_snapshot(environment, request.get("cwd"))
                result = await environment.exec(
                    str(request["command"]), cwd=request.get("cwd"),
                    timeout_sec=max(1, int(float(request.get("timeoutMs", 120_000)) / 1000)),
                )
                after_files = await self._file_snapshot(environment, request.get("cwd"))
                response = await asyncio.to_thread(self.rpc.request, "resume_boundary", {
                    "result": {
                        "requestId": request["id"], "exitCode": result.return_code,
                        "stdout": result.stdout or "", "stderr": result.stderr or "",
                        "durationMs": int((time.monotonic() - started) * 1000),
                        "fileChanges": sorted(
                            path for path in set(before_files).union(after_files)
                            if before_files.get(path) != after_files.get(path)
                        ),
                    },
                })
                macro_terminal_commands += 1
        snapshot = dict(response.get("snapshot") or self.rpc.last_snapshot or {})
        states = list(snapshot.get("processStates") or [])
        if not states:
            raise RuntimeError("node-wise checkpoint has no process state")
        policy_count = len(snapshot.get("policyRecords") or [])
        if policy_count - base_policy_count != self.macro_steps:
            raise RuntimeError("node-wise checkpoint did not execute exactly one policy action")
        if self.macro_steps == 1:
            new_record = list(snapshot.get("policyRecords") or [])[-1]
            if str(new_record.get("stateFingerprint") or "") != base_fingerprint:
                raise RuntimeError(
                    "node-wise actor fingerprint differs from restored decision boundary"
                )
        state = dict(states[-1])
        fingerprint = str(state.get("fingerprint") or "")
        if not fingerprint:
            raise RuntimeError("node-wise process state has no immutable fingerprint")
        self._write_json(self.output_snapshot_path, snapshot)
        self._write_json(self.output_state_path, state)
        checkpoint = getattr(environment, "create_training_checkpoint", None)
        if not callable(checkpoint):
            raise RuntimeError("node-wise output requires a clonable environment backend")
        checkpoint_audit = await checkpoint(self.output_checkpoint_path, fingerprint)
        finalizer_steps = 0
        finalizer_terminal_commands = 0
        while finalizer_steps < 16:
            finalizer = await asyncio.to_thread(self.rpc.request, "finalize_now", {})
            finalizer_steps += 1
            if finalizer.get("status") == "terminal_request":
                request = dict(finalizer["request"])
                started = time.monotonic()
                result = await environment.exec(
                    str(request["command"]), cwd=request.get("cwd"),
                    timeout_sec=max(
                        1, int(float(request.get("timeoutMs", 120_000)) / 1000)
                    ),
                )
                await asyncio.to_thread(self.rpc.request, "resume_boundary", {
                    "result": {
                        "requestId": request["id"], "exitCode": result.return_code,
                        "stdout": result.stdout or "", "stderr": result.stderr or "",
                        "durationMs": int((time.monotonic() - started) * 1000),
                        "fileChanges": [],
                    },
                })
                finalizer_terminal_commands += 1
                break
            if finalizer.get("status") == "completed":
                break
            if finalizer.get("status") != "continue":
                raise RuntimeError("frozen finalize-now returned an invalid boundary")
        else:
            raise RuntimeError("frozen finalize-now exceeded its integration step limit")
        latest_states = list((self.rpc.last_snapshot or {}).get("processStates") or [])
        usage = dict((latest_states[-1] if latest_states else state).get("usage") or {})
        context.n_input_tokens = int(usage.get("inputTokens", 0))
        context.n_output_tokens = int(usage.get("outputTokens", 0))
        context.metadata = {
            **(context.metadata or {}),
            "nodewise_protocol": "same_checkpoint_single_node_macro_action_v1",
            "macro_steps": self.macro_steps,
            "derived_reward_emitted": False,
            "task_utility_role": "value_supervision_only",
            "state_fingerprint": fingerprint,
            "base_decision_fingerprint": base_fingerprint,
            "organization_seed": self.organization_seed,
            "session_snapshot_path": str(self.output_snapshot_path),
            "process_state_path": str(self.output_state_path),
            "environment_checkpoint_path": str(self.output_checkpoint_path),
            "checkpoint_audit": dict(checkpoint_audit),
            "macro_terminal_commands": macro_terminal_commands,
            "finalizer_policy": "frozen_finalize_now_no_structural_actions",
            "finalizer_revision": "frozen-one-root-conversion-a0-v2",
            "finalizer_worker_steps": finalizer_steps,
            "finalizer_terminal_commands": finalizer_terminal_commands,
        }

    @staticmethod
    async def _file_snapshot(environment: Any, cwd: str | None) -> Dict[str, str]:
        snapshot = getattr(environment, "snapshot_workspace_files", None)
        if callable(snapshot):
            return dict(await snapshot(cwd))
        return {}

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def close(self) -> None:
        self.rpc.close()
