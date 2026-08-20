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

    def request(self, method: str, params: Mapping[str, Any], retry: bool = True) -> Mapping[str, Any]:
        self.start()
        self.next_id += 1
        request_id = self.next_id
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps({
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
            }) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
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
            return self.request(method, params, retry=False)

    def restart(self) -> None:
        self.close()
        self.responses = queue.Queue()
        self.start()
        if self.last_snapshot is not None:
            self.request("restore", {"snapshot": self.last_snapshot}, retry=False)

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
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        configured = node_command or os.environ.get(
            "ROY_LHTB_NODE_COMMAND", "node dist/cli/LhtbAgent.js"
        )
        self.semantic_root = Path(self.logs_dir) / "semantic"
        self.runtime_audit_root = Path(self.logs_dir) / "runtime-audit"
        child_environment = dict(os.environ)
        child_environment["ROY_LHTB_SEMANTIC_ROOT"] = str(self.semantic_root)
        child_environment["ROY_LHTB_AUDIT_ROOT"] = str(self.runtime_audit_root)
        self.rpc = PersistentNodeRPC(
            shlex.split(configured), timeout=rpc_timeout, environment=child_environment,
            stderr_path=self.runtime_audit_root / "child-stderr.log",
        )
        atexit.register(self.close)
        self.partial_path = Path(self.logs_dir) / "roy-partial-trajectory.json"
        self.track_file_changes = track_file_changes

    @staticmethod
    def name() -> str:
        return "roy-lhtb-agent"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: Any) -> None:
        self.rpc.start()

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            trajectory_id = str(uuid.uuid4())
            base_seed = int(os.environ.get("ROY_LHTB_ORGANIZATION_SEED", "20260820"))
            session_seed = int(hashlib.sha256(
                str(getattr(environment, "session_id", trajectory_id)).encode("utf-8")
            ).hexdigest()[:8], 16)
            response = await asyncio.to_thread(self.rpc.request, "run", {
                "trajectoryId": trajectory_id,
                "taskId": getattr(environment, "environment_name", "unknown"),
                "instruction": instruction,
                "environmentRevision": os.environ.get("LHTB_COMMIT", "pinned"),
                "organizationMode": os.environ.get(
                    "ROY_LHTB_ARM", "learned_information_realization"
                ),
                "organizationSeed": base_seed ^ session_seed,
                "initialSnapshotFingerprint": os.environ.get(
                    "ROY_LHTB_INITIAL_FINGERPRINT", ""
                ),
            })
            self._save_partial()
            await self._drive(response, environment)
            snapshot = self.rpc.last_snapshot
            self._save_partial()
            context.metadata = {
                **(context.metadata or {}), "roy_trajectory_id": trajectory_id,
                "roy_partial_trajectory": str(self.partial_path),
                "roy_semantic_audit_root": str(self.semantic_root),
                "roy_runtime_audit_root": str(self.runtime_audit_root),
            }
            if snapshot:
                states = list(snapshot.get("processStates", []))
                usage = dict(states[-1].get("usage", {})) if states else {}
                context.n_input_tokens = int(usage.get("inputTokens", 0))
                context.n_output_tokens = int(usage.get("outputTokens", 0))
        except BaseException:
            self._save_partial()
            self.close()
            raise

    async def resume_after_verifier_rejection(self, environment: Any, context: Any) -> None:
        """Compatible with official same-conversation mode when a provider supports it."""
        if self.rpc.last_snapshot is None:
            raise RuntimeError("Roy has no session to resume")
        response = await asyncio.to_thread(self.rpc.request, "verifier_rejection", {})
        await self._drive(response, environment)

    async def _drive(self, response: Mapping[str, Any], environment: Any) -> None:
        while response.get("status") in ("terminal_request", "continue"):
            if response.get("status") == "continue":
                response = await asyncio.to_thread(self.rpc.request, "advance", {})
                self._save_partial()
                continue
            request = dict(response["request"])
            started = time.monotonic()
            before_files = await self._file_snapshot(environment, request.get("cwd"))
            result = await environment.exec(
                str(request["command"]), cwd=request.get("cwd"),
                timeout_sec=max(1, int(float(request.get("timeoutMs", 120000)) / 1000)),
            )
            after_files = await self._file_snapshot(environment, request.get("cwd"))
            file_changes = sorted(
                path for path in set(before_files).union(after_files)
                if before_files.get(path) != after_files.get(path)
            )
            response = await asyncio.to_thread(self.rpc.request, "resume", {"result": {
                "requestId": request["id"], "exitCode": result.return_code,
                "stdout": result.stdout or "", "stderr": result.stderr or "",
                "durationMs": int((time.monotonic() - started) * 1000),
                "fileChanges": file_changes,
            }})
            self._save_partial()

    def _save_partial(self) -> None:
        snapshot = self.rpc.last_snapshot
        if snapshot is None:
            return
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.partial_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.partial_path)

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
