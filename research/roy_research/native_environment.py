from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import signal
from pathlib import Path
from typing import Any, Mapping

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .lhtb_native import (
    NATIVE_BACKEND_ID,
    load_task_native_manifest,
    native_preflight,
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
        allow_network_degraded: bool = False,
        **kwargs: Any,
    ) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.template_root = Path(template_root).expanduser().resolve()
        self.uid_base = int(uid_base)
        self.uid_slots = int(uid_slots)
        if self.uid_slots < 1:
            raise ValueError("uid_slots must be positive")
        self.allow_network_degraded = bool(allow_network_degraded)
        self.session_root: Path | None = None
        self._manifest: Mapping[str, Any] | None = None
        self._process_groups: set[int] = set()
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
        load_task_native_manifest(self.template_root, self.environment_name)

    def _validate_internet_config(self) -> None:
        if not self.task_env_config.allow_internet and not self.allow_network_degraded:
            raise ValueError(
                "LHTB-native cannot enforce allow_internet=false on GPUHome; "
                "set allow_network_degraded only for explicitly non-comparable runs"
            )
        if not self.task_env_config.allow_internet:
            self.logger.warning(
                "Running %s without required network isolation; result is degraded",
                self.environment_name,
            )

    @property
    def environment_digest(self) -> str:
        if self._manifest is None:
            self._manifest = load_task_native_manifest(
                self.template_root, self.environment_name
            )
        return str(self._manifest["environment_digest"])

    @property
    def _uid(self) -> int:
        value = int(hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:8], 16)
        value %= self.uid_slots
        return self.uid_base + value

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("LHTB-native templates must be provisioned before Harbor starts")
        native_preflight(self.runtime_root)
        manifest = load_task_native_manifest(self.template_root, self.environment_name)
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
        template = self.template_root / self.environment_name
        for name in ("app", "opt"):
            source = template / name
            if source.exists():
                shutil.copytree(source, self.session_root / name, dirs_exist_ok=True, symlinks=True)
        if (template / "venv").exists():
            (self.session_root / "opt" / "roy-native" / "venv").mkdir(parents=True)
        if (template / "bin").exists():
            (self.session_root / "opt" / "roy-native" / "bin").mkdir(parents=True)
        await asyncio.to_thread(self._chown_session)
        audit = {
            "schema_version": 1,
            "backend": NATIVE_BACKEND_ID,
            "environment_digest": self.environment_digest,
            "task_id": self.environment_name,
            "task_digest": task_digest,
            "session_id": self.session_id,
            "uid": self._uid,
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
            os.chown(root, self._uid, self._uid)
            for name in dirs:
                os.chown(Path(root) / name, self._uid, self._uid)
            for name in files:
                os.chown(Path(root) / name, self._uid, self._uid)

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
        if delete and self.session_root is not None:
            await asyncio.to_thread(shutil.rmtree, self.session_root, True)

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
        os.chown(destination, self._uid, self._uid)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        destination = self._mounted_path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree, source_dir, destination,
            dirs_exist_ok=True, symlinks=True,
        )
        await asyncio.to_thread(self._chown_path, destination)

    def _chown_path(self, path: Path) -> None:
        os.chown(path, self._uid, self._uid)
        for root, dirs, files in os.walk(path):
            for name in dirs:
                os.chown(Path(root) / name, self._uid, self._uid)
            for name in files:
                os.chown(Path(root) / name, self._uid, self._uid)

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

    def _command(self, command: str, cwd: str | None, env: Mapping[str, str]) -> list[str]:
        assert self.session_root is not None
        template = self.template_root / self.environment_name
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
        proot = ["proot", "-0", "-R", "/"]
        for source, target in binds:
            proot.extend(["-b", f"{source}:{target}"])
        working_directory = cwd or str(self._manifest.get("working_directory", "/app"))  # type: ignore[union-attr]
        shell = (
            f"cd -- {shlex.quote(working_directory)} && "
            f"exec /bin/bash -lc {shlex.quote(command)}"
        )
        exported = [f"{key}={value}" for key, value in env.items()]
        return [
            "setpriv", f"--reuid={self._uid}", f"--regid={self._uid}",
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
        if user not in (None, "root", "agent") and not isinstance(user, int):
            raise ValueError(f"unsupported native environment user {user!r}")
        merged = {
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "xterm-256color",
            **dict(self._manifest.get("environment", {})),
            **(self._merge_env(env) or {}),
        }
        process = await asyncio.create_subprocess_exec(
            *self._command(command, cwd, merged),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._process_groups.add(process.pid)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            ) if timeout_sec else await process.communicate()
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            return ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout_sec} seconds",
                return_code=124,
            )
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=int(process.returncode or 0),
        )
