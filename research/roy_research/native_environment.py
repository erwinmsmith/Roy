from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .lhtb_native import (
    NATIVE_BACKEND_ID,
    copy_checkpoint_tree,
    load_task_native_manifest,
    native_task_id_from_harbor,
    native_preflight,
    native_proot_launcher_environment,
    native_session_uids,
    resolve_native_task_source,
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
        proot_executable = shutil.which("proot")
        if proot_executable is None:
            raise RuntimeError("native environment requires PRoot")
        self.proot_executable = str(Path(proot_executable).resolve())
        if self.uid_slots < 1:
            raise ValueError("uid_slots must be positive")
        self.allow_network_degraded = bool(allow_network_degraded)
        environment_dir = kwargs.get("environment_dir")
        if environment_dir is None and args:
            environment_dir = args[0]
        environment_name = str(kwargs.get("environment_name", ""))
        if not environment_name and len(args) > 1:
            environment_name = str(args[1])
        # Harbor's task.name is a display identifier and is not required to
        # match the pinned dataset directory (for example
        # custom/snake-obstacle-campaign lives in snake_maze_campaign).
        # The task definition path is therefore the canonical native key.
        self.native_task_id = native_task_id_from_harbor(
            environment_dir, environment_name
        )
        self.session_root: Path | None = None
        self._rootfs: Path | None = None
        self._manifest: Mapping[str, Any] | None = None
        self._process_groups: set[int] = set()
        self._service_audit: list[dict[str, Any]] = []
        self._stopped_agent_pids: set[int] = set()
        self._protected_material_roots: set[str] = set()
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
        source_task_root = resolve_native_task_source(
            self.environment_dir.parent, self.native_task_id
        )
        task_digest = tree_digest(source_task_root)
        if manifest.get("task_digest") != task_digest:
            raise RuntimeError("native task template is stale relative to the pinned task")
        self._manifest = manifest
        rootfs_relative = manifest.get("rootfs")
        if rootfs_relative:
            rootfs = (self.template_root / self.native_task_id / str(rootfs_relative)).resolve()
            template = (self.template_root / self.native_task_id).resolve()
            if not rootfs.is_relative_to(template) or not (rootfs / "bin" / "bash").exists():
                raise RuntimeError(f"native task rootfs is invalid: {rootfs}")
            self._rootfs = rootfs
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
        await asyncio.to_thread(self._start_services, template)
        audit = {
            "schema_version": 1,
            "backend": NATIVE_BACKEND_ID,
            "environment_digest": self.environment_digest,
            "task_id": self.native_task_id,
            "task_digest": task_digest,
            "task_source_root": str(source_task_root),
            "task_control_overlay": source_task_root != self.environment_dir.parent.resolve(),
            "session_id": self.session_id,
            "uid": self._uid,
            "service_uid": self._service_uid,
            "gid": self.task_gid,
            "network_isolation": False,
            "pid_namespace": False,
            "mount_namespace": False,
            "official_leaderboard_comparable": False,
            "proot_executable": self.proot_executable,
            "rootfs": str(self._rootfs) if self._rootfs else None,
            "oci_image": manifest.get("oci_image"),
            "oci_digest": manifest.get("oci_digest"),
            "services": self._service_audit,
        }
        audit_path = self.trial_paths.trial_dir / "native-environment.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    async def create_training_checkpoint(
        self, destination: Path | str, source_state_fingerprint: str
    ) -> Mapping[str, Any]:
        """Record a restorable clone or an isolated non-restorable observation.

        Reviewed persistent services can be replayed only while still in their
        deterministic initial state. After any task command may have mutated a
        service, the filesystem and complete observation are retained for value
        supervision, but the artifact is explicitly non-restorable.
        """
        if self.session_root is None or self._manifest is None:
            raise RuntimeError("native environment is not started")
        if self._process_groups:
            service_pids = {int(value.get("pid", -1)) for value in self._service_audit}
            unexpected = self._process_groups - service_pids
            if unexpected:
                raise RuntimeError("native checkpoint requires an idle command boundary")
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise RuntimeError(f"refusing to overwrite native checkpoint {target}")
        if target == self.session_root or target.is_relative_to(self.session_root):
            raise ValueError("native checkpoint must be outside its source session")
        payload = target / "payload"
        payload.mkdir(parents=True)
        names = ("app", "opt", "tests", "solution", "tmp", "home", "logs")
        excluded_special_files: list[dict[str, str]] = []
        for name in names:
            source = self.session_root / name
            if source.exists():
                excluded = await asyncio.to_thread(
                    copy_checkpoint_tree, source, payload / name,
                )
                excluded_special_files.extend({
                    **value,
                    "path": f"{name}/{value['path']}",
                } for value in excluded)
        payload_digest = tree_digest(payload)
        services = list(self._manifest.get("services") or [])
        service_digest = hashlib.sha256(json.dumps(
            services, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest() if services else None
        service_state_touched = bool(services and self._exec_index > 0)
        mode = (
            "isolated_instance_observation" if service_state_touched
            else "deterministic_replay" if services
            else "full_clone"
        )
        audit = {
            "schema_version": 1,
            "mode": mode,
            "complete": True,
            "restorable": not service_state_touched,
            "task_id": self.native_task_id,
            "source_session_id": self.session_id,
            "source_state_fingerprint": source_state_fingerprint,
            "source_environment_digest": self.environment_digest,
            "payload_digest": payload_digest,
            "directories": list(names),
            "services": [str(value.get("name")) for value in services],
            "service_replay_digest": service_digest,
            "service_state": (
                "mutated_isolated_observation" if service_state_touched
                else "initial_deterministic_replay" if services
                else "absent"
            ),
            "exec_count_at_capture": self._exec_index,
            "excluded_special_files": excluded_special_files,
        }
        (target / "checkpoint.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return audit

    async def restore_training_checkpoint(self, source: Path | str) -> Mapping[str, Any]:
        if self.session_root is None or self._manifest is None:
            raise RuntimeError("native environment is not started")
        service_pids = {int(value.get("pid", -1)) for value in self._service_audit}
        if self._process_groups - service_pids or self._exec_index != 0:
            raise RuntimeError("native checkpoint restore requires a fresh idle environment")
        root = Path(source).expanduser().resolve(strict=True)
        manifest_path = root / "checkpoint.json"
        payload = root / "payload"
        if not manifest_path.is_file() or not payload.is_dir():
            raise ValueError("native training checkpoint is incomplete")
        audit = json.loads(manifest_path.read_text(encoding="utf-8"))
        if audit.get("schema_version") != 1 or audit.get("complete") is not True:
            raise ValueError("unsupported or incomplete native training checkpoint")
        restorable = audit.get("restorable", audit.get("mode") == "full_clone")
        if audit.get("mode") not in ("full_clone", "deterministic_replay") \
                or restorable is not True:
            raise ValueError("native backend accepts only restorable checkpoint states")
        if audit.get("task_id") != self.native_task_id:
            raise ValueError("native checkpoint belongs to another LHTB task")
        if audit.get("source_environment_digest") != self.environment_digest:
            raise ValueError("native checkpoint environment digest mismatch")
        services = list(self._manifest.get("services") or [])
        service_digest = hashlib.sha256(json.dumps(
            services, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest() if services else None
        if audit.get("service_replay_digest") != service_digest:
            raise ValueError("native checkpoint service replay configuration mismatch")
        if services and audit.get("service_state") != "initial_deterministic_replay":
            raise ValueError("native checkpoint does not contain replayable service state")
        if audit.get("payload_digest") != tree_digest(payload):
            raise ValueError("native checkpoint payload digest mismatch")
        for name in audit.get("directories", []):
            if name not in ("app", "opt", "tests", "solution", "tmp", "home", "logs"):
                raise ValueError(f"native checkpoint contains an invalid directory {name}")
            target = self.session_root / name
            await asyncio.to_thread(shutil.rmtree, target, True)
            await asyncio.to_thread(
                shutil.copytree, payload / name, target,
                dirs_exist_ok=True, symlinks=True,
            )
        await asyncio.to_thread(self._chown_session)
        await asyncio.to_thread(self._apply_path_permissions)
        restore_audit = {
            **audit,
            "restored_session_id": self.session_id,
            "restore_verified": True,
        }
        (self.trial_paths.trial_dir / "native-checkpoint-restore.json").write_text(
            json.dumps(restore_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return restore_audit

    @staticmethod
    def _allocate_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def _start_services(self, template: Path) -> None:
        services = list((self._manifest or {}).get("services", []))
        if not services:
            return
        agent_environment = dict((self._manifest or {}).get("environment", {}))
        for value in services:
            if not isinstance(value, Mapping):
                raise ValueError("native service must be an object")
            name = str(value.get("name") or "")
            if not name or not all(character.isalnum() or character in "-_" for character in name):
                raise ValueError(f"invalid native service name: {name!r}")
            port = self._allocate_loopback_port()
            substitutions = {
                "template": str(template),
                "port": str(port),
            }
            command = [
                str(item).format_map(substitutions)
                for item in value.get("command", [])
            ]
            if not command:
                raise ValueError(f"native service {name} has no command")
            working_directory = Path(
                str(value.get("working_directory", "{template}"))
                .format_map(substitutions)
            ).resolve()
            if not working_directory.is_relative_to(template.resolve()):
                raise ValueError(f"native service {name} cwd escapes its template")
            environment = {
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/games",
                **{
                    str(key): str(item).format_map(substitutions)
                    for key, item in dict(value.get("environment", {})).items()
                },
            }
            port_environment = str(value.get("port_environment") or "")
            if port_environment:
                environment[port_environment] = str(port)
            agent_url_environment = str(value.get("agent_url_environment") or "")
            if agent_url_environment:
                agent_environment[agent_url_environment] = f"http://127.0.0.1:{port}"
            log_root = self.trial_paths.trial_dir / "services"
            log_root.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{name}.log"
            with log_path.open("ab") as service_log:
                process = subprocess.Popen(
                    [
                        "setpriv", f"--reuid={self._service_uid}",
                        f"--regid={self.task_gid}", "--clear-groups",
                        "--no-new-privs", "env", "-i",
                        *[f"{key}={item}" for key, item in environment.items()],
                        *command,
                    ],
                    cwd=working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=service_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self._process_groups.add(process.pid)
            health_path = str(value.get("health_path", "/"))
            health_timeout = float(value.get("health_timeout_seconds", 60))
            health_url = f"http://127.0.0.1:{port}{health_path}"
            deadline = time.monotonic() + health_timeout
            last_error = "service did not become healthy"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    last_error = f"service exited with code {process.returncode}"
                    break
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as response:
                        if 200 <= response.status < 300:
                            self._service_audit.append({
                                "name": name,
                                "pid": process.pid,
                                "port": port,
                                "health_url": health_url,
                            })
                            break
                except (OSError, urllib.error.URLError) as error:
                    last_error = str(error)
                time.sleep(0.25)
            else:
                pass
            if not any(item["name"] == name for item in self._service_audit):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process_groups.discard(process.pid)
                raise RuntimeError(
                    f"native service {name} failed healthcheck: {last_error}; "
                    f"see {log_path}"
                )
        updated_manifest = dict(self._manifest or {})
        updated_manifest["environment"] = agent_environment
        self._manifest = updated_manifest

    def _chown_session(self) -> None:
        assert self.session_root is not None
        for root, dirs, files in os.walk(self.session_root):
            os.chown(root, self._uid, self.task_gid)
            os.chmod(root, os.stat(root).st_mode | 0o070)
            for name in dirs:
                path = Path(root) / name
                os.chown(path, self._uid, self.task_gid)
                os.chmod(path, path.stat().st_mode | 0o070)
            for name in files:
                path = Path(root) / name
                os.chown(path, self._uid, self.task_gid)
                os.chmod(path, path.stat().st_mode | 0o060)
        # The Agent owns the trial root; the shared task GID gives the distinct
        # service/verifier UID traverse-only access to its protected descendants.
        self.session_root.chmod(0o710)
        app_root = self.session_root / "app"
        subprocess.run(
            ["setfacl", "-R", "-m", f"u:{self._service_uid}:rwx", str(app_root)],
            check=True,
        )
        for root, _, _ in os.walk(app_root):
            subprocess.run(
                ["setfacl", "-m", f"d:u:{self._service_uid}:rwx", root],
                check=True,
            )

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
                    if owner == "service":
                        os.chmod(root, os.stat(root).st_mode & ~0o022)
                    for name in dirs:
                        path = Path(root) / name
                        os.chown(path, uid, self.task_gid)
                        if owner == "service":
                            os.chmod(path, path.stat().st_mode & ~0o022)
                    for name in files:
                        path = Path(root) / name
                        os.chown(path, uid, self.task_gid)
                        if owner == "service":
                            os.chmod(path, path.stat().st_mode & ~0o022)
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
                uid_line = next(
                    line for line in status.splitlines() if line.startswith("Uid:")
                )
                effective_uid = int(uid_line.split()[2])
                if effective_uid in (self._uid, self._service_uid):
                    os.kill(int(entry.name), requested_signal)
            except (
                FileNotFoundError,
                ProcessLookupError,
                PermissionError,
                StopIteration,
                ValueError,
            ):
                continue

    @staticmethod
    def _harbor_agent_tree_signal(command: str) -> signal.Signals | None:
        """Recognize Harbor's pinned shared-verifier freeze/thaw command.

        Running that command inside PRoot is unsafe because PRoot exposes the
        host PID namespace. The script consequently discovers the outer Harbor
        tmux session and can stop the PRoot process executing the script itself.
        Native trials instead signal only processes owned by their dedicated
        Agent UID from the host side.
        """
        required = ("/proc/[0-9]*", "tmux*)", 'case " $marked "', 'kill -')
        if not all(marker in command for marker in required):
            return None
        if 'kill -STOP "$pid"' in command:
            return signal.SIGSTOP
        if 'kill -CONT "$pid"' in command:
            return signal.SIGCONT
        return None

    def _signal_agent_uid_processes(self, requested_signal: signal.Signals) -> int:
        if requested_signal == signal.SIGCONT:
            targets = tuple(self._stopped_agent_pids)
            self._stopped_agent_pids.clear()
        else:
            targets = tuple(self._uid_processes(self._uid))
        signalled = 0
        for pid in targets:
            try:
                os.kill(pid, requested_signal)
                signalled += 1
                if requested_signal == signal.SIGSTOP:
                    self._stopped_agent_pids.add(pid)
            except (ProcessLookupError, PermissionError):
                continue
        return signalled

    @staticmethod
    def _uid_processes(uid: int) -> list[int]:
        own_pid = os.getpid()
        matches: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == own_pid:
                continue
            try:
                status = (entry / "status").read_text(encoding="utf-8")
                uid_line = next(
                    line for line in status.splitlines() if line.startswith("Uid:")
                )
                if int(uid_line.split()[2]) == uid:
                    matches.append(int(entry.name))
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                continue
        return matches

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
        if target_path == "/tests" or target_path.startswith("/tests/"):
            await asyncio.to_thread(self._make_app_group_writable)
        destination = self._mounted_path(target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source_path, destination)
        hidden = target_path == "/tests" or target_path.startswith("/tests/") \
            or target_path == "/solution" or target_path.startswith("/solution/")
        os.chown(destination, self._service_uid if hidden else self._uid, self.task_gid)
        if hidden:
            protected_root = "/tests" if target_path.startswith("/tests") else "/solution"
            mounted_root = self._mounted_path(protected_root)
            os.chown(mounted_root, self._service_uid, self.task_gid)
            mounted_root.chmod(0o700 if protected_root == "/tests" else 0o750)
            self._protected_material_roots.add(protected_root)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if target_dir == "/tests" or target_dir.startswith("/tests/"):
            await asyncio.to_thread(self._make_app_group_writable)
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
            protected_root = "/tests" if target_dir.startswith("/tests") else "/solution"
            destination.chmod(0o700 if protected_root == "/tests" else 0o750)
            self._protected_material_roots.add(protected_root)

    async def reset_dirs(
        self,
        *,
        remove_dirs: Any,
        create_dirs: Any,
        chmod_dirs: Any = None,
    ) -> ExecResult:
        """Reset bind sources on the host instead of unlinking PRoot mountpoints."""
        started = time.monotonic()
        remove_values = [str(path) for path in remove_dirs]
        create_values = [str(path) for path in create_dirs]
        chmod_values = [str(path) for path in (chmod_dirs or [])]
        await asyncio.to_thread(
            self._reset_native_dirs,
            remove_values,
            create_values,
            chmod_values,
        )
        result = ExecResult(stdout="", stderr="", return_code=0)
        self._append_exec_audit(
            command=json.dumps({
                "operation": "native_reset_dirs",
                "remove": remove_values,
                "create": create_values,
                "chmod": chmod_values,
            }, sort_keys=True),
            cwd=None,
            timeout_sec=None,
            timed_out=False,
            execution_role="service",
            host_side_process_signal=None,
            result=result,
            started=started,
        )
        return result

    def _reset_native_dirs(
        self,
        remove_dirs: list[str],
        create_dirs: list[str],
        chmod_dirs: list[str],
    ) -> None:
        assert self.session_root is not None
        protected = ("/tests", "/solution")
        for virtual_path in remove_dirs:
            target = self._mounted_path(virtual_path)
            is_bind_root = target.parent == self.session_root
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.exists() and is_bind_root:
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
            elif target.exists():
                shutil.rmtree(target)
            for root in protected:
                if virtual_path == root or virtual_path.startswith(root + "/"):
                    self._protected_material_roots.discard(root)
            if virtual_path == "/tests" or virtual_path.startswith("/tests/"):
                self._make_app_group_writable()
        for virtual_path in create_dirs:
            target = self._mounted_path(virtual_path)
            target.mkdir(parents=True, exist_ok=True)
            root = next(
                (value for value in protected
                 if virtual_path == value or virtual_path.startswith(value + "/")),
                None,
            )
            self._chown_path(target, self._service_uid if root else self._uid)
            if root:
                self._mounted_path(root).chmod(0o700 if root == "/tests" else 0o750)
        for virtual_path in chmod_dirs:
            target = self._mounted_path(virtual_path)
            if target.exists():
                target.chmod(0o777)

    def _make_app_group_writable(self) -> None:
        """Match Docker-root workspace mutation through the shared trial GID."""
        assert self.session_root is not None
        app_root = self.session_root / "app"
        for root, dirs, files in os.walk(app_root):
            root_path = Path(root)
            os.chmod(root_path, root_path.stat().st_mode | 0o070)
            for name in dirs:
                path = root_path / name
                os.chmod(path, path.stat().st_mode | 0o070)
            for name in files:
                path = root_path / name
                os.chmod(path, path.stat().st_mode | 0o060)

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
            (self.session_root / "logs", "/logs"),
        ]
        if self._rootfs is None or bool(self._manifest.get("bind_opt", False)):
            binds.append((self.session_root / "opt", "/opt"))
        if (template / "venv").exists():
            binds.append((template / "venv", "/opt/roy-native/venv"))
        if (template / "bin").exists():
            binds.append((template / "bin", "/opt/roy-native/bin"))
        proot_root = str(self._rootfs) if self._rootfs else "/"
        proot = [
            self.proot_executable, "-0", "-R", proot_root,
            "-w", working_directory,
        ]
        for source, target in binds:
            proot.extend(["-b", f"{source}:{target}"])
        shell = (
            f"cd -- {shlex.quote(working_directory)} && "
            f"exec /bin/bash -c {shlex.quote(command)}"
        )
        # PRoot resolves its temporary directory before guest bind mounts exist. Under
        # setpriv the host default selected by PRoot is not reliably writable on
        # GPUHome, so point it at the per-session directory already owned by this UID.
        # This is launcher state, not a command-level environment workaround.
        exported = native_proot_launcher_environment(env, self.session_root / "tmp")
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
        effective_user = self._resolve_user(user)
        service_execution = effective_user in ("root", "verifier", 0)
        if command in set(self._manifest.get("service_commands", [])):
            service_execution = True
        # LHTB tasks omit both Agent and verifier users because Docker normally
        # runs both as root. Tests are uploaded immediately before verification,
        # which gives the native backend an explicit, auditable verifier phase
        # boundary even when default_user remains None. Oracle solutions execute
        # as the Agent UID so ownership-sensitive copies into /app remain valid.
        if effective_user is None and "/tests" in self._protected_material_roots:
            service_execution = True
        execution_uid = self._service_uid if service_execution else self._uid
        tree_signal = self._harbor_agent_tree_signal(command)
        timed_out = False
        if tree_signal is not None:
            count = await asyncio.to_thread(
                self._signal_agent_uid_processes, tree_signal
            )
            result = ExecResult(
                stdout=(
                    f"native host-side SIG{tree_signal.name.removeprefix('SIG')} "
                    f"sent to {count} agent processes\n"
                ),
                stderr="",
                return_code=0,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *self._command(command, cwd, merged, execution_uid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/",
                start_new_session=True,
            )
            self._process_groups.add(process.pid)
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
        self._append_exec_audit(
            command=command,
            cwd=cwd,
            timeout_sec=timeout_sec,
            timed_out=timed_out,
            execution_role="service" if service_execution else "agent",
            host_side_process_signal=tree_signal.name if tree_signal else None,
            result=result,
            started=started,
        )
        return result

    def _append_exec_audit(
        self,
        *,
        command: str,
        cwd: str | None,
        timeout_sec: int | None,
        timed_out: bool,
        execution_role: str,
        host_side_process_signal: str | None,
        result: ExecResult,
        started: float,
    ) -> None:
        self._exec_index += 1
        audit = {
            "schema_version": 1,
            "index": self._exec_index,
            "command": command,
            "cwd": cwd,
            "timeout_sec": timeout_sec,
            "timed_out": timed_out,
            "execution_role": execution_role,
            "host_side_process_signal": host_side_process_signal,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        audit_path = self.trial_paths.trial_dir / "native-exec.jsonl"
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, sort_keys=True) + "\n")
