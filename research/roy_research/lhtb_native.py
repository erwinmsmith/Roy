from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - formal runner uses Python 3.12.
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .lhtb import LHTB_COMMIT, load_lhtb_manifest


NATIVE_SCHEMA_VERSION = 1
NATIVE_BACKEND_ID = "lhtb-native-process-v1"
NATIVE_SOURCE_TASK_MARKER = ".roy-native-source-task"


def native_session_uids(session_id: str, uid_base: int, uid_slots: int) -> tuple[int, int]:
    if uid_slots < 1:
        raise ValueError("uid_slots must be positive")
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    agent_uid = uid_base + (int(digest[:8], 16) % uid_slots)
    service_uid = uid_base + uid_slots + (int(digest[8:16], 16) % uid_slots)
    return agent_uid, service_uid


def normalize_native_task_id(environment_name: str) -> str:
    task_id = environment_name.rstrip("/").rsplit("/", 1)[-1]
    if not task_id or task_id in (".", ".."):
        raise ValueError(f"invalid native environment name: {environment_name!r}")
    return task_id


def resolve_native_task_source(task_root: Path, task_id: str) -> Path:
    """Resolve an immutable official task behind a native control-only overlay."""
    marker = task_root / NATIVE_SOURCE_TASK_MARKER
    if not marker.exists():
        return task_root.resolve()
    source = Path(marker.read_text(encoding="utf-8").strip()).resolve(strict=True)
    if source.name != task_id or source.parent.name != "tasks":
        raise ValueError(f"invalid native source task marker for {task_id}: {source}")
    if not (source / "task.toml").is_file() or not (source / "environment").is_dir():
        raise ValueError(f"native source task is incomplete: {source}")
    return source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path, *, excluded: Iterable[str] = ()) -> str:
    excluded_names = set(excluded)
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_names:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeAuditRecord:
    task_id: str
    category: str
    split: str
    status: str
    reasons: tuple[str, ...]
    task_digest: str
    environment_digest: str | None
    allow_internet: bool
    has_compose: bool
    provisioned: bool

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def _task_config(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _load_native_manifest(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != NATIVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported native manifest schema: {path}")
    return value


def audit_native_tasks(
    lhtb_root: Path,
    split_manifest: Path,
    template_root: Path,
    *,
    allow_network_degraded: bool = False,
) -> Dict[str, Any]:
    split = {str(value["task_id"]): value for value in load_lhtb_manifest(split_manifest)}
    records: list[NativeAuditRecord] = []
    for task_id, split_value in sorted(split.items()):
        task_root = lhtb_root / "tasks" / task_id
        if not task_root.is_dir():
            raise FileNotFoundError(f"missing pinned LHTB task {task_id}")
        config = _task_config(task_root / "task.toml")
        environment = config.get("environment") or {}
        allow_internet = bool(environment.get("allow_internet", True))
        task_digest = tree_digest(task_root)
        has_compose = (task_root / "environment" / "docker-compose.yaml").exists()
        manifest_path = template_root / task_id / "native-manifest.json"
        native_manifest = _load_native_manifest(manifest_path)
        reasons: list[str] = []
        if has_compose:
            reasons.append("docker_compose_requires_multiple_services")
        if str(environment.get("os", "linux")).lower() not in ("linux", "taskos.linux"):
            reasons.append("non_linux_task")
        if not allow_internet:
            reasons.append("network_isolation_unavailable_on_gpuhome_container")
        if native_manifest is None:
            reasons.append("native_template_not_provisioned")
        elif native_manifest.get("task_digest") != task_digest:
            reasons.append("native_template_task_digest_mismatch")
        elif native_manifest.get("lhtb_commit") != LHTB_COMMIT:
            reasons.append("native_template_lhtb_commit_mismatch")

        hard = {"docker_compose_requires_multiple_services", "non_linux_task"}
        stale = {"native_template_task_digest_mismatch", "native_template_lhtb_commit_mismatch"}
        if hard.intersection(reasons) or stale.intersection(reasons):
            status = "incompatible"
        elif native_manifest is None:
            status = "needs_provisioning"
        elif not allow_internet and not allow_network_degraded:
            status = "incompatible"
        elif not allow_internet:
            status = "degraded"
        else:
            status = "compatible"
        records.append(NativeAuditRecord(
            task_id=task_id,
            category=str(split_value["category"]),
            split=str(split_value["split"]),
            status=status,
            reasons=tuple(reasons),
            task_digest=task_digest,
            environment_digest=(str(native_manifest.get("environment_digest"))
                                if native_manifest else None),
            allow_internet=allow_internet,
            has_compose=has_compose,
            provisioned=native_manifest is not None,
        ))
    counts: Dict[str, int] = {}
    for value in records:
        counts[value.status] = counts.get(value.status, 0) + 1
    return {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "backend": NATIVE_BACKEND_ID,
        "lhtb_commit": LHTB_COMMIT,
        "allow_network_degraded": allow_network_degraded,
        "counts": counts,
        "tasks": [value.to_dict() for value in records],
    }


def write_native_audit(
    output: Path,
    lhtb_root: Path,
    split_manifest: Path,
    template_root: Path,
    *,
    allow_network_degraded: bool = False,
) -> Mapping[str, Any]:
    value = audit_native_tasks(
        lhtb_root, split_manifest, template_root,
        allow_network_degraded=allow_network_degraded,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def native_environment_digest(audit: Mapping[str, Any], task_id: str) -> str:
    matches = [value for value in audit.get("tasks", []) if value.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"native audit does not contain exactly one {task_id}")
    record = matches[0]
    if record.get("status") not in ("compatible", "degraded"):
        raise ValueError(f"native task {task_id} is not runnable: {record.get('status')}")
    digest = str(record.get("environment_digest") or "")
    if not digest.startswith("sha256:"):
        raise ValueError(f"native task {task_id} has no immutable environment digest")
    return digest


def native_preflight(runtime_root: Path, *, task_gid: int = 210000) -> Dict[str, Any]:
    required = ("proot", "setpriv", "timeout", "cp", "getfacl", "setfacl")
    missing = [name for name in required if shutil.which(name) is None]
    result = {
        "backend": NATIVE_BACKEND_ID,
        "architecture": platform.machine(),
        "platform": sys.platform,
        "runtime_root": str(runtime_root.resolve()),
        "required_commands": list(required),
        "missing_commands": missing,
        "network_isolation": False,
        "pid_namespace": False,
        "mount_namespace": False,
        "task_gid": task_gid,
        "official_leaderboard_comparable": False,
    }
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("LHTB-native requires x86_64 Linux")
    if missing:
        raise RuntimeError(f"LHTB-native is missing commands: {', '.join(missing)}")
    mountpoints = (
        Path("/app"), Path("/tests"), Path("/solution"), Path("/logs"),
        Path("/opt/roy-native/venv"), Path("/opt/roy-native/bin"),
    )
    for mountpoint in mountpoints:
        if mountpoint.exists() and not mountpoint.is_dir():
            raise RuntimeError(f"native PRoot mountpoint is not a directory: {mountpoint}")
        mountpoint.mkdir(parents=True, exist_ok=True)
        mountpoint.chmod(0o755)
    result["proot_mountpoints"] = [str(value) for value in mountpoints]
    traversal = subprocess.run([
        "setpriv", "--reuid=229999", f"--regid={task_gid}", "--clear-groups",
        "--no-new-privs", "/usr/bin/true",
    ], capture_output=True, text=True)
    if traversal.returncode != 0:
        raise RuntimeError(
            "LHTB-native task GID cannot traverse the GPUHome host root; "
            "run prepare_lhtb_native.sh prepare to install the execute-only ACL: "
            + traversal.stderr.strip()
        )
    result["root_traversal_acl"] = "execute-only"
    runtime_root.mkdir(parents=True, exist_ok=True)
    probe = runtime_root / ".write-probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return result


def _load_provisioning_specs(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != NATIVE_SCHEMA_VERSION:
        raise ValueError("unsupported native provisioning schema")
    return value.get("tasks") or {}


def _copy_entry(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def provision_native_task(
    lhtb_root: Path,
    template_root: Path,
    specs_path: Path,
    task_id: str,
    *,
    python_executable: str = sys.executable,
) -> Mapping[str, Any]:
    specs = _load_provisioning_specs(specs_path)
    if task_id not in specs:
        raise ValueError(f"no reviewed native provisioning spec for {task_id}")
    spec = specs[task_id]
    task_root = lhtb_root / "tasks" / task_id
    environment_root = task_root / "environment"
    target = template_root / task_id
    if target.exists():
        manifest = _load_native_manifest(target / "native-manifest.json")
        if manifest and manifest.get("task_digest") == tree_digest(task_root):
            return manifest
        raise RuntimeError(f"refusing to overwrite stale native template {target}")
    target.mkdir(parents=True)
    try:
        for copy in spec.get("copies", []):
            source = environment_root / str(copy["source"])
            destination = target / str(copy["target"])
            _copy_entry(source, destination)
        required = list(spec.get("required_commands", []))
        missing = [name for name in required if shutil.which(name) is None]
        if missing:
            raise RuntimeError(f"task {task_id} is missing host commands: {', '.join(missing)}")

        dependencies = list(spec.get("python_dependencies", []))
        if spec.get("python_project"):
            project = environment_root / str(spec["python_project"])
            config = _task_config(project / "pyproject.toml")
            dependencies.extend(config.get("project", {}).get("dependencies", []))
        if dependencies:
            venv = target / "venv"
            subprocess.run(["uv", "venv", "--python", python_executable, str(venv)], check=True)
            subprocess.run([
                "uv", "pip", "install", "--python", str(venv / "bin" / "python"),
                *dependencies,
            ], check=True)
        for command in spec.get("build_commands", []):
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", str(command)],
                check=True, cwd=target,
            )
        for wrapper in spec.get("wrappers", []):
            path = target / "bin" / str(wrapper["name"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(wrapper["content"]), encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        fingerprint_payload = {
            "backend": NATIVE_BACKEND_ID,
            "lhtb_commit": LHTB_COMMIT,
            "task_id": task_id,
            "task_digest": tree_digest(task_root),
            "spec": spec,
            "template_digest": tree_digest(target),
        }
        environment_digest = "sha256:" + hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": NATIVE_SCHEMA_VERSION,
            **fingerprint_payload,
            "environment_digest": environment_digest,
            "environment": dict(spec.get("environment", {})),
            "path_permissions": list(spec.get("path_permissions", [])),
            "working_directory": str(spec.get("working_directory", "/app")),
            "network_isolation": False,
            "official_leaderboard_comparable": False,
        }
        (target / "native-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~stat.S_IWOTH)
        return manifest
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def load_task_native_manifest(template_root: Path, task_id: str) -> Mapping[str, Any]:
    path = template_root / task_id / "native-manifest.json"
    value = _load_native_manifest(path)
    if value is None:
        raise FileNotFoundError(f"native task template is not provisioned: {task_id}")
    return value
