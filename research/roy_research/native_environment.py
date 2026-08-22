from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Mapping

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .lhtb_native import (
    NATIVE_BACKEND_ID,
    load_task_native_manifest,
    native_preflight,
    native_session_uids,
    normalize_native_task_id,
    tree_digest,
)


class NativeProcessEnvironment(BaseEnvironment):
    """GPUHome-compatible PRoot environment for the non-official LHTB adaptation.

    This backend intentionally does not claim namespace or network isolation. Every
    command runs under a dedicated unprivileged UID, receives a minimal environment,
    and sees per-trial /app, /workspace, /tests, /solution, /tmp and /opt paths through
    PRoot. It is suitable for Roy training, but not for LHTB leaderboard submission.
    """

    def __init__(
        self,
        *args: Any,
        runtime_root: str,
        template_root: str,
        uid_base: int = 210000,
        uid_slots: int = 20000,
        task_gid: int = 210000,
        allow_network_degraded: bool = False,
        **kwargs: Any,
    ) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.template_root = Path(template_root).expanduser().resolve()
        self.uid_base = int(uid_base)
        self.uid_slots = int(uid_slots)
        self.task_gid = int(task_gid)
        if self.uid_slots < 1:
            raise ValueError("uid_slots must be positive")
        self.allow_network_degraded = bool(allow_network_degraded)
        environment_name = str(kwargs.get("environment_name", ""))
        if not environment_name and len(args) > 1:
            environment_name = str(args[1])
        self.native_task_id = normalize_native_task_id(environment_name)
        self.session_root: Path | None = None
        self._manifest: Mapping[str, Any] | None = None
        self._process_groups: set[int] = set()
        self._exec_index = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return NATIVE_BACKEND_ID

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=False,
            disable_internet=False,
            windows=False,
            mounted=False,
        )

    @classmethod
    def preflight(cls) -> None:
        runtime = Path(os.environ.get("ROY_LHTB_NATIVE_ROOT", "/tmp/roy-lhtb-native"))
        native_preflight(runtime)

    def _validate_definition(self) -> None:
        if not (self.environment_dir / "Dockerfile").exists():
            raise FileNotFoundError(f"native task has no Dockerfile context: {self.environment_dir}")
        load_task_native_manifest(self.template_root, self.native_task_id)

    def _validate_internet_config(self) -> None:
        if not self.task_env_config.allow_internet and not self.allow_network_degraded:
            raise ValueError(
                "LHTB-native cannot enforce allow_internet=false on GPUHome; "
                "set allow_network_degraded only for explicitly non-comparable runs"
            )
        if not self.task_env_config.allow_internet:
            self.logger.warning(
                "Running %s without required network isolation; result is degraded",
                self.native_task_id,
            )

    @property
    def environment_digest(self) -> str:
        if self._manifest is None:
            self._manifest = load_task_native_manifest(
                self.template_root, self.native_task_id
            )
        return str(self._manifest["environment_digest"])

    @property
    def _uid(self) -> int:
        return native_session_uids(self.session_id, self.uid_base, self.uid_slots)[0]

    @property
    def _service_uid(self) -> int:
        """A distinct kernel UID for hidden task services and official verification."""
        return native_session_uids(self.session_id, self.uid_base, self.uid_slots)[1]

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("LHTB-native templates must be provisioned before Harbor starts")
        native_preflight(self.runtime_root)
        manifest = load_task_native_manifest(self.template_root, self.native_task_id)
        task_digest = tree_digest(self.environment_dir.parent)
        if manifest.get("task_digest") != task_digest:
            raise RuntimeError("native task template is stale relative to the pinned task")
        self._manifest = manifest
        safe_session = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.session_id
        )
        self.session_root = self.runtime_root / "sessions" / safe_session
        if self.session_root.exists():
            raise RuntimeError(f"native session already exists: {self.session_root}")
        for name in ("app", "opt", "tests", "solution", "tmp", "home", "logs"):
            (self.session_root / name).mkdir(parents=True, exist_ok=True)
        for name in ("agent", "verifier", "artifacts"):
            (self.session_root / "logs" / name).mkdir(parents=True, exist_ok=True)
        self.session_root.chmod(0o700)
        template = self.template_root / self.native_task_id
        for name in ("app", "opt"):
            source = template / name
            if source.exists():
                shutil.copytree(source, self.session_root / name, dirs_exist_ok=True, symlinks=True)
        if (template / "venv").exists():
            (self.session_root / "opt" / "roy-native" / "venv").mkdir(parents=True)
        if (template / "bin").exists():
            (self.session_root / "opt" / "roy-native" / "bin").mkdir(parents=True)
        await asyncio.to_thread(self._chown_session)
        await asyncio.to_thread(self._apply_path_permissions)
        audit = {
            "schema_version": 1,
            "backend": NATIVE_BACKEND_ID,
            "environment_digest": self.environment_digest,
            "task_id": self.native_task_id,
            "task_digest": task_digest,
            "session_id": self.session_id,
            "uid": self._uid,
            "service_uid": self._service_uid,
            "gid": self.task_gid,
            "network_isolation": False,
            "pid_namespace": False,
            "mount_namespace": False,
            "official_leaderboard_comparable": False,
        }
        audit_path = self.trial_paths.trial_dir / "native-environment.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _chown_session(self) -> None:
        assert self.session_root is not None
        for root, dirs, files in os.walk(self.session_root):
            os.chown(root, self._uid, self.task_gid)
            for name in dirs:
                os.chown(Path(root) / name, self._uid, self.task_gid)
            for name in files:
                os.chown(Path(root) / name, self._uid, self.task_gid)
        # The Agent owns the trial root; the shared task GID gives the distinct
        # service/verifier UID traverse-only access to its protected descendants.
        self.session_root.chmod(0o710)

    def _apply_path_permissions(self) -> None:
        for value in (self._manifest or {}).get("path_permissions", []):
            if not isinstance(value, Mapping):
                raise ValueError("native path permission must be an object")
            virtual_path = str(value.get("path") or "")
            target = self._mounted_path(virtual_path)
            if not target.exists():
                raise FileNotFoundError(
                    f"native protected path does not exist: {virtual_path}"
                )
            owner = str(value.get("owner", "service"))
            if owner not in ("agent", "service"):
                raise ValueError(f"unsupported native path owner: {owner}")
            uid = self._service_uid if owner == "service" else self._uid
            recursive = bool(value.get("recursive", True))
            os.chown(target, uid, self.task_gid)
            if recursive and target.is_dir():
                for root, dirs, files in os.walk(target):
                    os.chown(root, uid, self.task_gid)
                    for name in dirs:
                        os.chown(Path(root) / name, uid, self.task_gid)
                    for name in files:
                        os.chown(Path(root) / name, uid, self.task_gid)
            mode = value.get("mode")
            if mode is not None:
                parsed = int(str(mode), 8) if isinstance(mode, str) else int(mode)
                os.chmod(target, parsed)

    async def stop(self, delete: bool) -> None:
        for process_group in tuple(self._process_groups):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                continue
        await asyncio.sleep(0)
        for process_group in tuple(self._process_groups):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._process_groups.clear()
        # Daemon-style task CLIs can detach from the command process group. Each
        # trial has dedicated real UIDs, so clean every remaining process owned by
        # either identity before deleting its filesystem.
        await asyncio.to_thread(self._kill_uid_processes, signal.SIGTERM)
        await asyncio.sleep(0.2)
        await asyncio.to_thread(self._kill_uid_processes, signal.SIGKILL)
        if delete and self.session_root is not None:
            await asyncio.to_thread(shutil.rmtree, self.session_root, True)

    def _kill_uid_processes(self, requested_signal: signal.Signals) -> None:
        own_pid = os.getpid()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == own_pid:
                continue
            try:
                status = (entry / "status").read_text(encoding="utf-8")
                uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
                effective_uid = int(uid_line.split()[2])
                if effective_uid in (self._uid, self._service_uid):
                    os.kill(int(entry.name), requested_signal)
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration,
                    ValueError):
                continue

    def _mounted_path(self, path: str) -> Path:
        if not path.startswith("/"):
            working_directory = str(
                (self._manifest or {}).get("working_directory", "/app")
            )
            path = str(Path(working_directory) / path)
        normalized = Path(path)
        if ".." in normalized.parts:
            raise ValueError(f"native environment path traversal is forbidden: {path}")
        assert self.session_root is not None
        mappings = {
            "/workspace": "app",
            "/app": "app",
            "/tests": "tests",
            "/solution": "solution",
            "/tmp": "tmp",
            "/opt": "opt",
            "/root": "home",
            "/logs": "logs",
        }
        for target, relative in mappings.items():
            if path == target or path.startswith(target + "/"):
                suffix = path[len(target):].lstrip("/")
                return self.session_root / relative / suffix
        raise ValueError(f"native backend does not expose host path {path}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        destination = self._mounted_path(target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source_path, destination)
        hidden = target_path == "/tests" or target_path.startswith("/tests/") \
            or target_path == "/solution" or target_path.startswith("/solution/")
        os.chown(destination, self._service_uid if hidden else self._uid, self.task_gid)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        destination = self._mounted_path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree, source_dir, destination,
            dirs_exist_ok=True, symlinks=True,
        )
        hidden = target_dir == "/tests" or target_dir.startswith("/tests/") \
            or target_dir == "/solution" or target_dir.startswith("/solution/")
        await asyncio.to_thread(
            self._chown_path, destination,
            self._service_uid if hidden else self._uid,
        )
        if hidden:
            destination.chmod(0o700)

    def _chown_path(self, path: Path, uid: int | None = None) -> None:
        owner = self._uid if uid is None else uid
        os.chown(path, owner, self.task_gid)
        for root, dirs, files in os.walk(path):
            for name in dirs:
                os.chown(Path(root) / name, owner, self.task_gid)
            for name in files:
                os.chown(Path(root) / name, owner, self.task_gid)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._mounted_path(source_path)
        destination = Path(target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, destination)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._mounted_path(source_dir)
        destination = Path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree, source, destination,
            dirs_exist_ok=True, symlinks=True,
        )

    def _command(self, command: str, cwd: str | None, env: Mapping[str, str],
                 execution_uid: int | None = None) -> list[str]:
        assert self.session_root is not None
        template = self.template_root / self.native_task_id
        default_working_directory = str(
            self._manifest.get("working_directory", "/app")  # type: ignore[union-attr]
        )
        working_directory = cwd or default_working_directory
        if not working_directory.startswith("/"):
            working_directory = str(Path(default_working_directory) / working_directory)
        self._mounted_path(working_directory)
        binds: list[tuple[Path, str]] = [
            (self.session_root / "home", "/root"),
            (self.session_root / "tmp", "/tmp"),
            (self.session_root / "app", "/app"),
            (self.session_root / "app", "/workspace"),
            (self.session_root / "tests", "/tests"),
            (self.session_root / "solution", "/solution"),
            (self.session_root / "opt", "/opt"),
            (self.session_root / "logs", "/logs"),
        ]
        if (template / "venv").exists():
            binds.append((template / "venv", "/opt/roy-native/venv"))
        if (template / "bin").exists():
            binds.append((template / "bin", "/opt/roy-native/bin"))
        proot = ["proot", "-0", "-R", "/", "-w", working_directory]
        for source, target in binds:
            proot.extend(["-b", f"{source}:{target}"])
        shell = (
            f"cd -- {shlex.quote(working_directory)} && "
            f"exec /bin/bash -lc {shlex.quote(command)}"
        )
        exported = [f"{key}={value}" for key, value in env.items()]
        uid = self._uid if execution_uid is None else execution_uid
        return [
            "setpriv", f"--reuid={uid}", f"--regid={self.task_gid}",
            "--clear-groups", "--no-new-privs", "env", "-i", *exported,
            *proot, "/bin/bash", "-c", shell,
        ]

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.session_root is None or self._manifest is None:
            raise RuntimeError("native environment is not started")
        if user not in (None, "root", "agent", "verifier") and not isinstance(user, int):
            raise ValueError(f"unsupported native environment user {user!r}")
        merged = {
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "xterm-256color",
            **dict(self._manifest.get("environment", {})),
            **(self._merge_env(env) or {}),
        }
        started = time.monotonic()
        execution_uid = self._service_uid if user in ("root", "verifier", 0) else self._uid
        process = await asyncio.create_subprocess_exec(
            *self._command(command, cwd, merged, execution_uid),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._process_groups.add(process.pid)
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            ) if timeout_sec else await process.communicate()
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            stdout = b""
            stderr = f"command timed out after {timeout_sec} seconds".encode()
        finally:
            self._process_groups.discard(process.pid)
        result = ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=124 if timed_out else int(process.returncode or 0),
        )
        self._exec_index += 1
        audit = {
            "schema_version": 1,
            "index": self._exec_index,
            "command": command,
            "cwd": cwd,
            "timeout_sec": timeout_sec,
            "timed_out": timed_out,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        audit_path = self.trial_paths.trial_dir / "native-exec.jsonl"
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, sort_keys=True) + "\n")
        return result
