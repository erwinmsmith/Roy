from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .analysis import paired_bootstrap_interval, summarize_groups
from .baselines import evaluate_controlled_arms
from .benchmarks import build_remote_manifest
from .controlled import collect_group, generate_tasks, mechanism_diagnostics
from .controlled import TERMINAL_SUCCESS_THRESHOLD
from .io import atomic_json, read_jsonl, write_jsonl
from .live_controlled import collect_forced_full_mas, collect_live_group
from .providers import DeepSeekClient
from .reporting import write_utility_svg
from .schema import TraceRecord
from .token_ledger import PersistentTokenLedger
from .tau3 import build_tau3_manifest, manifest_summary, verify_tau3_root
from .organization import RuntimeBudget
from .tau3_runner import evaluate_tau3_against_direct, train_tau3_on_policy
from .lhtb import build_lhtb_split, verify_lhtb_checkout, write_lhtb_manifest, load_lhtb_manifest
from .lhtb_experiment import build_training_schedule, disk_preflight, select_dev_checkpoint, summarize_test, write_json, write_harbor_group_config, write_lhtb_svg
from .lhtb_training import LHTBProcessGRPOTrainer
from .lhtb_results import import_harbor_group, validate_smoke
from .lhtb_value_metrics import value_metrics, annotate_value_traces
from .lhtb_native import (
    native_environment_digest,
    native_preflight,
    provision_native_task,
    write_native_audit,
)
from .training import TRAINING_VARIANTS, evaluate_groups, train_groups


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="roy-research")
    commands = root.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate the 180-task controlled benchmark")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=20260815)

    collect = commands.add_parser("collect", help="Collect deterministic counterfactual groups")
    collect.add_argument("--tasks", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--repeats", type=int, default=2)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--traces", type=Path, help="Optional flat trajectory JSONL output")

    collect_live = commands.add_parser("collect-live", help="Collect real DeepSeek counterfactual rollout groups")
    collect_live.add_argument("--tasks", type=Path, required=True)
    collect_live.add_argument("--output", type=Path, required=True)
    collect_live.add_argument("--ledger", type=Path, required=True)
    collect_live.add_argument("--events", type=Path, required=True, help="Append-only request/response event log")
    collect_live.add_argument("--traces", type=Path, help="Optional flat trajectory JSONL output")
    collect_live.add_argument("--task-ids", nargs="*")
    collect_live.add_argument("--split", choices=("train", "validation", "test"))
    collect_live.add_argument("--limit", type=int)
    collect_live.add_argument("--repeats", type=int, default=2)
    collect_live.add_argument("--max-tokens", type=int, default=384)
    collect_live.add_argument("--temperature", type=float, default=0.0)
    collect_live.add_argument("--problem-version", choices=("v1", "v2"), default="v1")
    collect_live.add_argument("--token-limit", type=int, default=10_000_000)
    collect_live.add_argument("--timeout", type=float, default=90.0)
    collect_live.add_argument("--workers", type=int, default=1)
    collect_live.add_argument("--resume", action="store_true")

    collect_mas = commands.add_parser(
        "collect-live-mas", help="Aggregate all three matched child agents as a forced complete MAS"
    )
    collect_mas.add_argument("--groups", type=Path, required=True)
    collect_mas.add_argument("--source-events", type=Path, required=True)
    collect_mas.add_argument("--output", type=Path, required=True)
    collect_mas.add_argument("--events", type=Path, required=True)
    collect_mas.add_argument("--ledger", type=Path, required=True)
    collect_mas.add_argument("--max-tokens", type=int, default=384)
    collect_mas.add_argument("--temperature", type=float, default=0.0)
    collect_mas.add_argument("--token-limit", type=int, default=10_000_000)
    collect_mas.add_argument("--timeout", type=float, default=90.0)
    collect_mas.add_argument("--resume", action="store_true")

    compare_live = commands.add_parser(
        "compare-live-arms", help="Compare single-agent direct, forced full MAS and learned full policy"
    )
    compare_live.add_argument("--groups", type=Path, required=True)
    compare_live.add_argument("--evaluation", type=Path, required=True)
    compare_live.add_argument("--mas", type=Path, required=True)
    compare_live.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train", help="Train Node CS-GRPO and derivation heads")
    train.add_argument("--groups", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--device")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--variant", choices=TRAINING_VARIANTS, default="full")

    evaluate = commands.add_parser("evaluate", help="Evaluate structural decision regret")
    evaluate.add_argument("--groups", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--device")

    report = commands.add_parser("report", help="Write a lightweight Markdown report")
    report.add_argument("--groups", type=Path, required=True)
    report.add_argument("--evaluation", type=Path)
    report.add_argument("--experiment", type=Path)
    report.add_argument("--output", type=Path, required=True)

    baselines = commands.add_parser("baselines", help="Evaluate controlled non-parametric arms")
    baselines.add_argument("--groups", type=Path, required=True)
    baselines.add_argument("--output", type=Path, required=True)
    baselines.add_argument("--split")

    package = commands.add_parser("package-remote", help="Create a pinned remote benchmark bundle")
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--token-limit", type=int, default=10_000_000)

    imported = commands.add_parser("import-results", help="Import remote JSONL results without benchmark assets")
    imported.add_argument("--source", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)

    smoke = commands.add_parser("smoke", help="Run generation and deterministic rollout smoke")
    smoke.add_argument("--output", type=Path, required=True)

    api_smoke = commands.add_parser("api-smoke", help="Run one budgeted live DeepSeek request")
    api_smoke.add_argument("--output", type=Path, required=True)
    api_smoke.add_argument("--ledger", type=Path, required=True)
    api_smoke.add_argument("--max-tokens", type=int, default=256)
    api_smoke.add_argument("--token-limit", type=int, default=10_000_000)

    ledger_limit = commands.add_parser(
        "raise-ledger-limit", help="Increase an existing persistent token ledger hard limit"
    )
    ledger_limit.add_argument("--ledger", type=Path, required=True)
    ledger_limit.add_argument("--token-limit", type=int, required=True)

    experiment = commands.add_parser("experiment", help="Train and compare all controlled CS-GRPO ablations")
    experiment.add_argument("--groups", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--epochs", type=int, default=3)
    experiment.add_argument("--device")
    experiment.add_argument("--resume", action="store_true")

    tau3_manifest = commands.add_parser(
        "tau3-manifest", help="Verify the pinned tau3 checkout and write leakage-safe splits"
    )
    tau3_manifest.add_argument("--tau3-root", type=Path, required=True)
    tau3_manifest.add_argument("--output", type=Path, required=True)
    tau3_manifest.add_argument("--validation-modulus", type=int, default=10)

    tau3_train = commands.add_parser(
        "tau3-train", help="Sample tau3 groups and update the organization policy on-policy"
    )
    tau3_train.add_argument("--manifest", type=Path, required=True)
    tau3_train.add_argument("--trajectories", type=Path, required=True)
    tau3_train.add_argument("--model", type=Path, required=True)
    tau3_train.add_argument("--agent-llm", default="deepseek/deepseek-v4-flash")
    tau3_train.add_argument("--user-llm", default="deepseek/deepseek-v4-flash")
    tau3_train.add_argument(
        "--task-keys",
        nargs="*",
        help="Optional domain:task_id train records for a leakage-safe focused run",
    )
    tau3_train.add_argument("--limit", type=int)
    tau3_train.add_argument("--max-steps", type=int, default=1000)
    tau3_train.add_argument("--max-tokens", type=int, default=50000)
    tau3_train.add_argument("--temperature", type=float, default=0.0)
    tau3_train.add_argument("--epochs", type=int, default=4)
    tau3_train.add_argument("--organization-temperature", type=float, default=2.0)
    tau3_train.add_argument(
        "--max-rollout-attempts",
        type=int,
        default=3,
        help="Retries per rollout slot when an episode is censored; every attempt is saved",
    )
    _add_tau3_runtime_budget_arguments(tau3_train)
    tau3_train.add_argument("--seed", type=int, default=20260818)
    tau3_train.add_argument("--resume", action="store_true")

    tau3_evaluate = commands.add_parser(
        "tau3-evaluate", help="Compare single-agent direct with one learned organization"
    )
    tau3_evaluate.add_argument("--manifest", type=Path, required=True)
    tau3_evaluate.add_argument("--model", type=Path, required=True)
    tau3_evaluate.add_argument("--output", type=Path, required=True)
    tau3_evaluate.add_argument("--summary", type=Path, required=True)
    tau3_evaluate.add_argument("--agent-llm", default="deepseek/deepseek-v4-flash")
    tau3_evaluate.add_argument("--user-llm", default="deepseek/deepseek-v4-flash")
    tau3_evaluate.add_argument("--split", choices=("validation", "test", "heldout"), default="test")
    tau3_evaluate.add_argument("--limit", type=int)
    tau3_evaluate.add_argument("--max-steps", type=int, default=1000)
    tau3_evaluate.add_argument("--max-tokens", type=int, default=50000)
    tau3_evaluate.add_argument("--temperature", type=float, default=0.0)
    tau3_evaluate.add_argument("--organization-temperature", type=float, default=1.0)
    _add_tau3_runtime_budget_arguments(tau3_evaluate)
    tau3_evaluate.add_argument("--seed", type=int, default=20260818)

    lhtb_manifest = commands.add_parser(
        "lhtb-manifest", help="Write the pinned leakage-safe LHTB 30/8/8 split"
    )
    lhtb_manifest.add_argument("--output", type=Path, required=True)
    lhtb_manifest.add_argument("--lhtb-root", type=Path)

    lhtb_preflight = commands.add_parser(
        "lhtb-preflight", help="Fail-closed VM and disk preflight for formal LHTB runs"
    )
    lhtb_preflight.add_argument("--path", type=Path, required=True)
    lhtb_preflight.add_argument("--output", type=Path, required=True)

    native_preflight_parser = commands.add_parser(
        "lhtb-native-preflight", help="Validate the non-container GPUHome native backend"
    )
    native_preflight_parser.add_argument("--runtime-root", type=Path, required=True)
    native_preflight_parser.add_argument("--output", type=Path, required=True)

    native_audit = commands.add_parser(
        "lhtb-native-audit", help="Audit pinned tasks for native-process compatibility"
    )
    native_audit.add_argument("--lhtb-root", type=Path, required=True)
    native_audit.add_argument("--manifest", type=Path, required=True)
    native_audit.add_argument("--template-root", type=Path, required=True)
    native_audit.add_argument("--output", type=Path, required=True)
    native_audit.add_argument("--allow-network-degraded", action="store_true")

    native_provision = commands.add_parser(
        "lhtb-native-provision", help="Provision one reviewed native task template"
    )
    native_provision.add_argument("--lhtb-root", type=Path, required=True)
    native_provision.add_argument("--template-root", type=Path, required=True)
    native_provision.add_argument("--specs", type=Path, required=True)
    native_provision.add_argument("--task-id", required=True)
    native_provision.add_argument("--python", default=sys.executable)

    native_digest = commands.add_parser(
        "lhtb-native-digest", help="Resolve an audited immutable native environment digest"
    )
    native_digest.add_argument("--audit", type=Path, required=True)
    native_digest.add_argument("--task-id", required=True)

    lhtb_schedule = commands.add_parser(
        "lhtb-schedule", help="Create the formal four-epoch, 960-rollout schedule"
    )
    lhtb_schedule.add_argument("--manifest", type=Path, required=True)
    lhtb_schedule.add_argument("--output", type=Path, required=True)

    lhtb_select = commands.add_parser(
        "lhtb-select", help="Select one checkpoint from complete dev metrics"
    )
    lhtb_select.add_argument("--metrics", type=Path, required=True)
    lhtb_select.add_argument("--output", type=Path, required=True)

    lhtb_report = commands.add_parser(
        "lhtb-report", help="Summarize the one-shot three-arm LHTB test"
    )
    lhtb_report.add_argument("--results", type=Path, required=True)
    lhtb_report.add_argument("--output", type=Path, required=True)
    lhtb_report.add_argument("--checkpoint", type=Path)

    lhtb_update = commands.add_parser(
        "lhtb-update", help="Apply current-policy LHTB G=8 groups with value/EMA process credit"
    )
    lhtb_update.add_argument("--manifest", type=Path, required=True)
    lhtb_update.add_argument("--trajectories", type=Path, required=True)
    lhtb_update.add_argument("--model", type=Path, required=True)
    lhtb_update.add_argument("--updates", type=Path, required=True)
    lhtb_update.add_argument("--device", default="cpu")
    lhtb_update.add_argument("--resume", action="store_true")

    lhtb_init = commands.add_parser(
        "lhtb-init", help="Initialize independent actor/value/EMA state for LHTB"
    )
    lhtb_init.add_argument("--manifest", type=Path, required=True)
    lhtb_init.add_argument("--model", type=Path, required=True)
    lhtb_init.add_argument("--device", default="cpu")

    lhtb_import = commands.add_parser(
        "lhtb-import-group", help="Import one Harbor G=8 job into append-only trajectories"
    )
    lhtb_import.add_argument("--job-dir", type=Path, required=True)
    lhtb_import.add_argument("--output", type=Path, required=True)
    lhtb_import.add_argument("--group-id", required=True)
    lhtb_import.add_argument("--task-id", required=True)
    lhtb_import.add_argument("--category", required=True)
    lhtb_import.add_argument("--split", choices=("train", "dev", "test"), required=True)
    lhtb_import.add_argument("--epoch", type=int, required=True)
    lhtb_import.add_argument("--policy-revision", type=int, required=True)
    lhtb_import.add_argument("--environment-digest")
    lhtb_import.add_argument("--docker-digest", help="Deprecated Docker-only alias")
    lhtb_import.add_argument("--environment-backend", choices=("docker", "native"),
                             default="docker")
    lhtb_import.add_argument("--expected", type=int, default=8)
    lhtb_import.add_argument("--arm", choices=("single_agent_direct", "roy_runtime_heuristic",
                                               "learned_information_realization"),
                             default="learned_information_realization")

    lhtb_config = commands.add_parser(
        "lhtb-group-config", help="Write one pinned Harbor task/group config"
    )
    lhtb_config.add_argument("--output", type=Path, required=True)
    lhtb_config.add_argument("--jobs-dir", type=Path, required=True)
    lhtb_config.add_argument("--task-id", required=True)
    lhtb_config.add_argument("--arm", choices=("single_agent_direct", "roy_runtime_heuristic",
                                                "learned_information_realization"), required=True)
    lhtb_config.add_argument("--initial-fingerprint", required=True)
    lhtb_config.add_argument("--organization-seed", type=int, required=True)
    lhtb_config.add_argument("--attempts", type=int, default=8)
    lhtb_config.add_argument("--official-timeout", action="store_true")
    lhtb_config.add_argument("--environment-backend", choices=("docker", "native"),
                             default="docker")
    lhtb_config.add_argument("--native-runtime-root", type=Path)
    lhtb_config.add_argument("--native-template-root", type=Path)
    lhtb_config.add_argument("--allow-network-degraded", action="store_true")
    lhtb_config.add_argument("--max-retries", type=int, default=2)

    lhtb_dev = commands.add_parser(
        "lhtb-dev-metrics", help="Append checkpoint-selection metrics for one dev epoch"
    )
    lhtb_dev.add_argument("--trajectories", type=Path, required=True)
    lhtb_dev.add_argument("--checkpoint", type=Path, required=True)
    lhtb_dev.add_argument("--epoch", type=int, required=True)
    lhtb_dev.add_argument("--output", type=Path, required=True)
    lhtb_dev.add_argument("--device", default="cpu")

    lhtb_smoke_validate = commands.add_parser(
        "lhtb-smoke-validate", help="Validate matched G=8 Harbor smoke groups"
    )
    lhtb_smoke_validate.add_argument("--jobs-dir", type=Path, required=True)
    lhtb_smoke_validate.add_argument("--output", type=Path, required=True)
    return root


def _add_tau3_runtime_budget_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--max-llm-calls", type=int)
    command.add_argument("--max-tool-calls", type=int)
    command.add_argument("--max-nodes", type=int)
    command.add_argument("--max-depth", type=int)
    command.add_argument("--max-decisions", type=int)


def _tau3_runtime_budget(args: argparse.Namespace) -> RuntimeBudget:
    return RuntimeBudget(
        maximum_llm_calls=args.max_llm_calls,
        maximum_tool_calls=args.max_tool_calls,
        maximum_nodes=args.max_nodes,
        maximum_depth=args.max_depth,
        maximum_decisions=args.max_decisions,
    )


def main(argv: List[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "generate":
        tasks = [task.to_dict() for task in generate_tasks(args.seed)]
        write_jsonl(args.output, tasks)
        print(json.dumps({"tasks": len(tasks), "output": str(args.output)}))
    elif args.command == "lhtb-manifest":
        records = build_lhtb_split()
        if args.lhtb_root:
            verify_lhtb_checkout(args.lhtb_root, records)
        write_lhtb_manifest(args.output, records)
        print(json.dumps({"tasks": 46, "train": 30, "dev": 8, "test": 8,
                          "output": str(args.output)}))
    elif args.command == "lhtb-preflight":
        result = disk_preflight(args.path)
        write_json(args.output, result)
        print(json.dumps(result))
    elif args.command == "lhtb-native-preflight":
        result = native_preflight(args.runtime_root)
        write_json(args.output, result)
        print(json.dumps(result))
    elif args.command == "lhtb-native-audit":
        result = write_native_audit(
            args.output, args.lhtb_root, args.manifest, args.template_root,
            allow_network_degraded=args.allow_network_degraded,
        )
        print(json.dumps({"output": str(args.output), **result["counts"]}))
    elif args.command == "lhtb-native-provision":
        result = provision_native_task(
            args.lhtb_root, args.template_root, args.specs, args.task_id,
            python_executable=args.python,
        )
        print(json.dumps({
            "task_id": args.task_id,
            "environment_digest": result["environment_digest"],
            "template_root": str(args.template_root),
        }))
    elif args.command == "lhtb-native-digest":
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        print(native_environment_digest(audit, args.task_id))
    elif args.command == "lhtb-schedule":
        schedule = build_training_schedule(load_lhtb_manifest(args.manifest))
        write_json(args.output, {"schema_version": 1, "epochs": 4, "group_size": 8,
            "rollouts": 960, "groups": [{"epoch": value.epoch, "task_id": value.task_id,
                "group_id": value.group_id, "organization_seeds": value.organization_seeds}
                for value in schedule]})
        print(json.dumps({"groups": len(schedule), "rollouts": 960}))
    elif args.command == "lhtb-select":
        selected = select_dev_checkpoint(list(read_jsonl(args.metrics)))
        write_json(args.output, dict(selected))
        print(json.dumps(selected))
    elif args.command == "lhtb-report":
        records = list(read_jsonl(args.results))
        calibration = None
        if args.checkpoint:
            calibration = value_metrics(records, str(args.checkpoint))
            records = list(annotate_value_traces(records, str(args.checkpoint)))
            write_jsonl(args.output.with_suffix(".value-traces.jsonl"), records)
        summary = summarize_test(records)
        if calibration is not None:
            summary["value_calibration"] = calibration
        write_json(args.output, summary)
        write_lhtb_svg(args.output.with_suffix(".svg"), summary)
        print(json.dumps(summary))
    elif args.command == "lhtb-update":
        manifest = load_lhtb_manifest(args.manifest)
        trainer = LHTBProcessGRPOTrainer(
            args.model, manifest, device_name=args.device, resume=args.resume
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in read_jsonl(args.trajectories):
            if not record.get("accepted_for_training", True):
                continue
            grouped.setdefault(str(record.get("group_id", "")), []).append(record)
        updates = []
        for group_id, records in grouped.items():
            if group_id in trainer.updated_group_ids:
                continue
            update = trainer.update_group(records)
            updates.append(update)
            write_jsonl(args.updates, [update], append=args.updates.exists())
        print(json.dumps({"new_updates": len(updates), **trainer.metadata()}))
    elif args.command == "lhtb-init":
        trainer = LHTBProcessGRPOTrainer(
            args.model, load_lhtb_manifest(args.manifest), device_name=args.device
        )
        trainer.save()
        print(json.dumps({"model": str(args.model), **trainer.metadata()}))
    elif args.command == "lhtb-import-group":
        environment_digest = args.environment_digest or args.docker_digest
        if not environment_digest:
            raise ValueError("an immutable environment digest is required")
        records = import_harbor_group(
            args.job_dir, args.output, args.group_id, args.task_id, args.category,
            args.split, args.epoch, args.policy_revision, environment_digest,
            {"maximum_rollout_seconds": 3600, "max_response_tokens": 32768,
             "concurrency": 4},
            expected=args.expected,
            arm=args.arm,
            environment_backend=args.environment_backend,
        )
        print(json.dumps({"imported": len(records), "output": str(args.output)}))
    elif args.command == "lhtb-group-config":
        write_harbor_group_config(args.output, args.task_id, args.jobs_dir, args.arm,
                                  args.initial_fingerprint, args.organization_seed, args.attempts,
                                  args.official_timeout, args.environment_backend,
                                  args.native_runtime_root, args.native_template_root,
                                  args.allow_network_degraded, args.max_retries)
        print(json.dumps({"output": str(args.output), "task_id": args.task_id}))
    elif args.command == "lhtb-dev-metrics":
        records = [value for value in read_jsonl(args.trajectories)
                   if value.get("split") == "dev" and int(value.get("epoch", -1)) == args.epoch]
        metrics = value_metrics(records, str(args.checkpoint), args.device)
        rows = [{"split": "dev", "epoch": args.epoch, "task_id": value["task_id"],
                 "checkpoint": str(args.checkpoint), "reward": value["terminal_reward"],
                 "tokens": value.get("tokens", 0), **metrics} for value in records]
        write_jsonl(args.output, rows, append=args.output.exists())
        print(json.dumps({"records": len(rows), **metrics}))
    elif args.command == "lhtb-smoke-validate":
        result = validate_smoke(args.jobs_dir)
        write_json(args.output, result)
        print(json.dumps(result))
    elif args.command == "raise-ledger-limit":
        state = PersistentTokenLedger.raise_existing_limit(args.ledger, args.token_limit)
        print(json.dumps({"ledger": str(args.ledger), **state}))
    elif args.command == "collect":
        tasks = list(read_jsonl(args.tasks))
        if args.limit is not None:
            tasks = tasks[: args.limit]
        completed = set()
        if args.resume and args.output.exists():
            completed = {record["task"]["id"] for record in read_jsonl(args.output)}
        groups = []
        traces = []
        for task_value in tasks:
            task = _task_from_dict(task_value)
            if task.id in completed:
                continue
            group = collect_group(task, args.repeats)
            group["mechanism_diagnostics"] = mechanism_diagnostics(task)
            groups.append(group)
            traces.extend(_group_traces(group))
        count = write_jsonl(args.output, groups, append=args.resume and args.output.exists())
        if args.traces:
            write_jsonl(args.traces, traces, append=args.resume and args.traces.exists())
        print(json.dumps({"collected": count, "traces": len(traces), "output": str(args.output)}))
    elif args.command == "collect-live":
        tasks = list(read_jsonl(args.tasks))
        if args.task_ids:
            selected_ids = set(args.task_ids)
            tasks = [task for task in tasks if str(task["id"]) in selected_ids]
            missing = selected_ids - {str(task["id"]) for task in tasks}
            if missing:
                raise ValueError(f"unknown task ids: {sorted(missing)}")
        if args.split:
            tasks = [task for task in tasks if str(task.get("split")) == args.split]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        completed = set()
        if args.resume and args.output.exists():
            completed = {record["task"]["id"] for record in read_jsonl(args.output)}
        ledger = PersistentTokenLedger(args.ledger, args.token_limit)
        client = DeepSeekClient(ledger, timeout=args.timeout, event_log=args.events)
        if args.workers < 1:
            raise ValueError("workers must be positive")
        collected = 0
        trace_count = 0
        trace_batch = 0

        pending_tasks = [
            _task_from_dict(task_value) for task_value in tasks
            if str(task_value["id"]) not in completed
        ]

        def collect_task(task):
            return collect_live_group(
                task, client, repeats=args.repeats,
                max_tokens=args.max_tokens, temperature=args.temperature,
                problem_version=args.problem_version,
            )

        def persist_task(task, group):
            nonlocal collected, trace_count, trace_batch
            write_jsonl(
                args.output, [group],
                append=args.output.exists() and (args.resume or collected > 0),
            )
            if args.traces:
                traces = _group_traces(group)
                write_jsonl(
                    args.traces, traces,
                    append=(args.resume and args.traces.exists()) or trace_batch > 0,
                )
                trace_count += len(traces)
                trace_batch += 1
            collected += 1
            print(json.dumps({
                "task_id": task.id,
                "collected": collected,
                "ledger": ledger.snapshot(),
            }), flush=True)

        if args.workers == 1:
            for task in pending_tasks:
                persist_task(task, collect_task(task))
        else:
            task_iterator = iter(pending_tasks)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                active: Dict[Future, Any] = {}
                for _ in range(args.workers):
                    task = next(task_iterator, None)
                    if task is not None:
                        active[executor.submit(collect_task, task)] = task
                while active:
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = active.pop(future)
                        try:
                            persist_task(task, future.result())
                        except Exception:
                            for remaining in active:
                                remaining.cancel()
                            raise
                        next_task = next(task_iterator, None)
                        if next_task is not None:
                            active[executor.submit(collect_task, next_task)] = next_task
        print(json.dumps({
            "collected": collected, "traces": trace_count,
            "output": str(args.output), "ledger": ledger.snapshot(),
        }))
    elif args.command == "collect-live-mas":
        groups = list(read_jsonl(args.groups))
        source_events = list(read_jsonl(args.source_events))
        completed = set()
        if args.resume and args.output.exists():
            completed = {str(record["task_id"]) for record in read_jsonl(args.output)}
        ledger = PersistentTokenLedger(args.ledger, args.token_limit)
        client = DeepSeekClient(ledger, timeout=args.timeout, event_log=args.events)
        collected = 0
        for group in groups:
            task_id = str(group["task"]["id"])
            if task_id in completed:
                continue
            result = collect_forced_full_mas(
                group, source_events, client,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            write_jsonl(
                args.output, [result],
                append=args.output.exists() and (args.resume or collected > 0),
            )
            collected += 1
            print(json.dumps({
                "task_id": task_id, "collected": collected,
                "utility": result["utility"], "ledger": ledger.snapshot(),
            }), flush=True)
        print(json.dumps({
            "collected": collected, "output": str(args.output), "ledger": ledger.snapshot(),
        }))
    elif args.command == "compare-live-arms":
        result = _compare_live_arms(args.groups, args.evaluation, args.mas)
        atomic_json(args.output, result)
        report_path = args.output.with_suffix(".md")
        _write_live_arm_report(result, report_path)
        print(json.dumps({"output": str(args.output), "report": str(report_path)}))
    elif args.command == "train":
        result = train_groups(
            args.groups, args.output, epochs=args.epochs,
            device_name=args.device, resume=args.resume, variant=args.variant,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "evaluate":
        result = evaluate_groups(args.groups, args.model, split=args.split, device_name=args.device)
        atomic_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "report":
        _write_report(args.groups, args.evaluation, args.output, args.experiment)
        print(json.dumps({"output": str(args.output)}))
    elif args.command == "baselines":
        result = evaluate_controlled_arms(read_jsonl(args.groups), split=args.split)
        atomic_json(args.output, result)
        print(json.dumps({"output": str(args.output), "arms": len(result["arms"])}))
    elif args.command == "package-remote":
        _package_remote(args.output, args.token_limit)
        print(json.dumps({"output": str(args.output)}))
    elif args.command == "import-results":
        records = list(read_jsonl(args.source))
        for record in records:
            record["imported"] = True
        write_jsonl(args.output, records)
        print(json.dumps({"imported": len(records), "output": str(args.output)}))
    elif args.command == "smoke":
        args.output.mkdir(parents=True, exist_ok=True)
        tasks_path = args.output / "tasks.jsonl"
        groups_path = args.output / "groups.jsonl"
        tasks = [task.to_dict() for task in generate_tasks()]
        write_jsonl(tasks_path, tasks)
        selected = [_task_from_dict(tasks[index]) for index in (0, 1, 60, 61, 120, 121)]
        groups = []
        for task in selected:
            group = collect_group(task, 2)
            group["mechanism_diagnostics"] = mechanism_diagnostics(task)
            groups.append(group)
        write_jsonl(groups_path, groups)
        summary = summarize_groups(groups)
        atomic_json(args.output / "baselines.json", evaluate_controlled_arms(groups))
        atomic_json(args.output / "summary.json", summary)
        _write_report(groups_path, None, args.output / "report.md", None)
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "api-smoke":
        ledger = PersistentTokenLedger(args.ledger, args.token_limit)
        client = DeepSeekClient(ledger)
        completion = client.complete([
            {"role": "system", "content": "Return exactly one JSON object and no prose."},
            {"role": "user", "content": "Choose one structural action for a solved task: CONTINUE, BRANCH, or RETURN."},
        ], max_tokens=args.max_tokens, temperature=0.0)
        atomic_json(args.output, {
            "provider": "deepseek",
            "model": client.model,
            "content": completion.content,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "latency_ms": completion.latency_ms,
            "ledger": ledger.snapshot(),
        })
        print(json.dumps({"output": str(args.output), "tokens": completion.total_tokens}))
    elif args.command == "experiment":
        args.output.mkdir(parents=True, exist_ok=True)
        learned: Dict[str, Any] = {}
        for variant in TRAINING_VARIANTS:
            model_path = args.output / f"{variant}.pt"
            metadata = train_groups(
                args.groups, model_path, epochs=args.epochs, device_name=args.device,
                resume=args.resume, variant=variant,
            )
            learned[variant] = {
                "metadata": metadata,
                "validation": evaluate_groups(args.groups, model_path, "validation", args.device),
                "test": evaluate_groups(args.groups, model_path, "test", args.device),
                "ood": evaluate_groups(args.groups, model_path, "ood", args.device),
            }
        rules = evaluate_controlled_arms(read_jsonl(args.groups), split="test")
        no_derivation = {
            group["task"]["id"]: float(group["action_values"]["CONTINUE"])
            for group in read_jsonl(args.groups) if group["task"]["split"] == "test"
        }
        for values in learned.values():
            task_results = values["test"]["task_results"]
            values["test"]["paired_vs_no_derivation"] = paired_bootstrap_interval(
                [float(item["utility"]) for item in task_results],
                [no_derivation[item["task_id"]] for item in task_results],
            )
            values["test"]["paired_success_vs_direct"] = paired_bootstrap_interval(
                [float(item["success"]) for item in task_results],
                [float(no_derivation[item["task_id"]] >= TERMINAL_SUCCESS_THRESHOLD) for item in task_results],
            )
        full_by_task = {
            item["task_id"]: float(item["utility"])
            for item in learned["full"]["test"]["task_results"]
        }
        for values in learned.values():
            task_results = values["test"]["task_results"]
            values["test"]["paired_vs_full"] = paired_bootstrap_interval(
                [float(item["utility"]) for item in task_results],
                [full_by_task[item["task_id"]] for item in task_results],
            )
        result = {"schema_version": 1, "rule_arms": rules["arms"], "learned_arms": learned}
        atomic_json(args.output / "experiment.json", result)
        print(json.dumps({"output": str(args.output / "experiment.json"), "learned_arms": len(learned)}))
    elif args.command == "tau3-manifest":
        revision = verify_tau3_root(args.tau3_root)
        manifest = build_tau3_manifest(args.tau3_root, args.validation_modulus)
        write_jsonl(args.output, [value.to_dict() for value in manifest])
        print(json.dumps({
            "output": str(args.output),
            "revision": revision,
            **manifest_summary(manifest),
        }, sort_keys=True))
    elif args.command == "tau3-train":
        metadata = train_tau3_on_policy(
            args.manifest,
            args.trajectories,
            args.model,
            args.agent_llm,
            args.user_llm,
            task_keys=args.task_keys,
            limit=args.limit,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            epochs=args.epochs,
            organization_temperature=args.organization_temperature,
            runtime_budget=_tau3_runtime_budget(args),
            max_rollout_attempts=args.max_rollout_attempts,
            seed=args.seed,
            resume=args.resume,
        )
        print(json.dumps(metadata, sort_keys=True))
    elif args.command == "tau3-evaluate":
        result = evaluate_tau3_against_direct(
            args.manifest,
            args.model,
            args.output,
            args.agent_llm,
            args.user_llm,
            split=args.split,
            limit=args.limit,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            organization_temperature=args.organization_temperature,
            runtime_budget=_tau3_runtime_budget(args),
            seed=args.seed,
        )
        atomic_json(args.summary, result)
        print(json.dumps({"output": str(args.output), "summary": str(args.summary), **result}, sort_keys=True))


def _compare_live_arms(groups_path: Path, evaluation_path: Path, mas_path: Path) -> Dict[str, Any]:
    groups = {str(group["task"]["id"]): group for group in read_jsonl(groups_path)}
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    learned = {str(row["task_id"]): row for row in evaluation["task_results"]}
    mas = {str(row["task_id"]): row for row in read_jsonl(mas_path)}
    expected = set(groups)
    if set(learned) != expected or set(mas) != expected:
        raise ValueError(
            "arm task mismatch: "
            f"groups={sorted(expected)} learned={sorted(learned)} mas={sorted(mas)}"
        )

    arm_rows: Dict[str, List[Dict[str, Any]]] = {
        "single_agent_direct": [],
        "forced_full_mas": [],
        "learned_full_policy": [],
    }
    for task_id in sorted(expected):
        group = groups[task_id]
        direct_results = [
            result for result in group["results"] if result["action"] == "CONTINUE"
        ]
        direct_utility = float(group["action_values"]["CONTINUE"])
        arm_rows["single_agent_direct"].append({
            "task_id": task_id,
            "utility": direct_utility,
            "success": direct_utility >= TERMINAL_SUCCESS_THRESHOLD,
            "tokens": round(sum(int(row.get("token_usage", 0)) for row in direct_results) / len(direct_results)),
            "latency_ms": round(sum(int(row.get("duration_ms", 0)) for row in direct_results) / len(direct_results)),
            "rollout_repeats": len(direct_results),
        })

        mas_result = mas[task_id]
        arm_rows["forced_full_mas"].append({
            "task_id": task_id,
            "utility": float(mas_result["utility"]),
            "success": bool(mas_result["success"]),
            "tokens": int(mas_result["total_tokens"]),
            "latency_ms": int(mas_result["parallel_span_ms"]),
            "work_latency_ms": int(mas_result["work_latency_ms"]),
            "child_agent_count": int(mas_result["child_agent_count"]),
        })

        learned_result = learned[task_id]
        selected_action = str(learned_result["selected_action"])
        selected_child = learned_result.get("selected_child_specification")
        candidates = [result for result in group["results"] if result["action"] == selected_action]
        if selected_action == "BRANCH" and selected_child:
            candidates = [
                result for result in candidates
                if (result.get("child_specification") or {}).get("id") == selected_child
            ]
        arm_rows["learned_full_policy"].append({
            "task_id": task_id,
            "selected_action": selected_action,
            "selected_child_specification": selected_child,
            "utility": float(learned_result["utility"]),
            "success": bool(learned_result["success"]),
            "tokens": round(sum(int(row.get("token_usage", 0)) for row in candidates) / len(candidates)),
            "latency_ms": round(sum(int(row.get("duration_ms", 0)) for row in candidates) / len(candidates)),
            "rollout_repeats": len(candidates),
        })

    direct_rows = arm_rows["single_agent_direct"]
    direct_utility = [float(row["utility"]) for row in direct_rows]
    direct_success = [float(row["success"]) for row in direct_rows]
    arms: Dict[str, Any] = {}
    for name, rows in arm_rows.items():
        utilities = [float(row["utility"]) for row in rows]
        successes = [float(row["success"]) for row in rows]
        arm = {
            "episodes": len(rows),
            "successes": int(sum(successes)),
            "success_rate": sum(successes) / len(rows),
            "mean_utility": sum(utilities) / len(rows),
            "total_tokens": sum(int(row["tokens"]) for row in rows),
            "mean_tokens": sum(int(row["tokens"]) for row in rows) / len(rows),
            "mean_latency_ms": sum(int(row["latency_ms"]) for row in rows) / len(rows),
            "paired_vs_single_agent_direct": None if name == "single_agent_direct" else {
                "success": paired_bootstrap_interval(successes, direct_success),
                "utility": paired_bootstrap_interval(utilities, direct_utility),
            },
            "task_results": rows,
        }
        if name == "forced_full_mas":
            arm["mean_work_latency_ms"] = sum(
                int(row["work_latency_ms"]) for row in rows
            ) / len(rows)
            arm["child_agents_per_episode"] = 3
        arms[name] = arm
    return {
        "schema_version": 1,
        "split": str(evaluation["split"]),
        "success_threshold": TERMINAL_SUCCESS_THRESHOLD,
        "task_ids": sorted(expected),
        "arm_definitions": {
            "single_agent_direct": "Parent CONTINUE; task utility and cost are estimated over matched rollout repeats.",
            "forced_full_mas": "All three child agents execute, then one parent aggregates all proposals.",
            "learned_full_policy": "Trained structural policy chooses CONTINUE, RETURN, or one BRANCH specification; task utility and cost use the same matched-repeat estimator as direct.",
        },
        "arms": arms,
    }


def _write_live_arm_report(result: Dict[str, Any], output: Path) -> None:
    lines = [
        "# Live DeepSeek Single-Agent vs MAS Pilot",
        "",
        f"Split: `{result['split']}`; tasks: `{len(result['task_ids'])}`; "
        f"success threshold: `{result['success_threshold']}`.",
        "",
        "| Arm | E2E success | Utility | Mean tokens | Mean latency | Delta vs single-agent direct |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("single_agent_direct", "forced_full_mas", "learned_full_policy"):
        arm = result["arms"][name]
        interval = arm["paired_vs_single_agent_direct"]
        if interval is None:
            delta = "baseline"
        else:
            success = interval["success"]
            utility = interval["utility"]
            delta = (
                f"success {success['mean_difference']:+.4f} "
                f"[{success['ci95_low']:+.4f}, {success['ci95_high']:+.4f}]; "
                f"utility {utility['mean_difference']:+.4f} "
                f"[{utility['ci95_low']:+.4f}, {utility['ci95_high']:+.4f}]"
            )
        lines.append(
            f"| `{name}` | {arm['success_rate']:.2%} ({arm['successes']}/{arm['episodes']}) | "
            f"{arm['mean_utility']:.4f} | {arm['mean_tokens']:.1f} | "
            f"{arm['mean_latency_ms']:.1f} ms | {delta} |"
        )
    lines.extend([
        "",
        "`forced_full_mas` latency is ideal parallel span: slowest child plus parent aggregation. "
        "Its total token count always includes all three child calls and the parent aggregation.",
        "",
        "A confidence interval containing zero is inconclusive; this pilot is not an external benchmark claim.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _task_from_dict(value: Dict[str, Any]):
    from .controlled import ControlledTask
    return ControlledTask(
        id=str(value["id"]), family=str(value["family"]), split=str(value["split"]),
        ood=bool(value["ood"]), seed=int(value["seed"]),
        hidden_evidence_value=float(value["hidden_evidence_value"]),
        direct_value=float(value["direct_value"]),
        return_value=float(value["return_value"]),
        branch_values=tuple(float(item) for item in value["branch_values"]),
        required_depth=int(value["required_depth"]), unseen_tool=bool(value["unseen_tool"]),
        expected_action=str(value["expected_action"]),
    )


def _write_report(
    groups_path: Path,
    evaluation_path: Path | None,
    output: Path,
    experiment_path: Path | None,
) -> None:
    summary = summarize_groups(read_jsonl(groups_path))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path else None
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path else None
    lines = [
        "# Roy CS-GRPO Pilot Report",
        "",
        "## Dataset",
        "",
        f"- Counterfactual groups: {summary['groups']}",
        f"- Families: `{json.dumps(summary['by_family'], sort_keys=True)}`",
        f"- Maximum acquisition/activation reconstruction error: `{summary['max_reconstruction_error']}`",
        f"- Mean G_acq: `{summary['mean_g_acq']:.4f}`",
        f"- Mean G_act: `{summary['mean_g_act']:.4f}`",
        f"- Mean work/span: `{summary['mean_work']:.2f}` / `{summary['mean_span']:.2f}`",
        f"- Mean graph nodes/communication edges: `{summary['mean_nodes']:.2f}` / `{summary['mean_communication_edges']:.2f}`",
        "",
        "## Structural policy",
        "",
    ]
    if evaluation:
        direct = evaluation.get("direct")
        lines.extend([
            f"- Split: `{evaluation['split']}`",
            f"- Decision accuracy: `{evaluation['accuracy']:.4f}`",
            f"- End-to-end success: `{evaluation.get('success_rate', 0.0):.2%}`",
            f"- Mean utility: `{evaluation['mean_utility']:.4f}`",
            f"- Mean structural regret: `{evaluation['mean_regret']:.4f}`",
        ])
        if direct:
            success_interval = evaluation["paired_vs_direct"]["success"]
            utility_interval = evaluation["paired_vs_direct"]["utility"]
            lines.extend([
                f"- Direct success/utility: `{direct['success_rate']:.2%}` / `{direct['mean_utility']:.4f}`",
                f"- Success delta vs direct (paired 95% CI): `{success_interval['mean_difference']:.4f}` "
                f"`[{success_interval['ci95_low']:.4f}, {success_interval['ci95_high']:.4f}]` "
                f"(`{success_interval['conclusion']}`)",
                f"- Utility delta vs direct (paired 95% CI): `{utility_interval['mean_difference']:.4f}` "
                f"`[{utility_interval['ci95_low']:.4f}, {utility_interval['ci95_high']:.4f}]` "
                f"(`{utility_interval['conclusion']}`)",
            ])
    elif experiment:
        full_test = experiment["learned_arms"]["full"]["test"]
        lines.extend([
            "- Split: `test`",
            f"- Decision accuracy: `{full_test['accuracy']:.4f}`",
            f"- End-to-end success: `{full_test['success_rate']:.2%}`",
            f"- Mean utility: `{full_test['mean_utility']:.4f}`",
            f"- Mean structural regret: `{full_test['mean_regret']:.4f}`",
        ])
    else:
        lines.append("- Training/evaluation was not part of this smoke run.")
    if experiment:
        chart_path = output.with_suffix(".svg")
        write_utility_svg(chart_path, experiment)
        lines.extend(["", "## Controlled arms", "", f"End-to-end success means terminal utility >= `{TERMINAL_SUCCESS_THRESHOLD:.1f}`; `direct` is the existing `no_derivation` arm (`CONTINUE` only).", "", "| Arm | E2E success | Test utility | Regret | Success delta 95% CI vs direct | Utility delta 95% CI vs direct | Utility delta 95% CI vs full |", "| --- | ---: | ---: | ---: | --- | --- | --- |"])
        for name, value in experiment["rule_arms"].items():
            interval = value.get("paired_vs_no_derivation")
            interval_text = "baseline" if interval is None else f"[{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}] ({interval['conclusion']})"
            success_interval = value.get("paired_success_vs_direct")
            success_interval_text = "baseline" if success_interval is None else f"[{success_interval['ci95_low']:.4f}, {success_interval['ci95_high']:.4f}] ({success_interval['conclusion']})"
            display_name = value.get("display_name", name)
            lines.append(f"| `{display_name}` | {value['success_rate']:.2%} ({value['successes']}/{value['episodes']}) | {value['mean_utility']:.4f} | {value['mean_rollout_policy_regret']:.4f} | {success_interval_text} | {interval_text} | n/a |")
        for name, value in experiment["learned_arms"].items():
            test = value["test"]
            interval = test["paired_vs_no_derivation"]
            success_interval = test["paired_success_vs_direct"]
            versus_full = test["paired_vs_full"]
            lines.append(f"| `{name}` | {test['success_rate']:.2%} ({test['successes']}/{test['count']}) | {test['mean_utility']:.4f} | {test['mean_regret']:.4f} | [{success_interval['ci95_low']:.4f}, {success_interval['ci95_high']:.4f}] ({success_interval['conclusion']}) | [{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}] ({interval['conclusion']}) | [{versus_full['ci95_low']:.4f}, {versus_full['ci95_high']:.4f}] ({versus_full['conclusion']}) |")
        full_ood = experiment["learned_arms"]["full"]["ood"]
        lines.extend([
            "",
            f"Full-policy OOD utility/regret: `{full_ood['mean_utility']:.4f}` / `{full_ood['mean_regret']:.4f}` over `{full_ood['count']}` instances.",
        ])
        lines.extend(["", f"![Controlled utility arms]({chart_path.name})"])
    if evaluation:
        failures = [item for item in sorted(
            evaluation.get("task_results", []), key=lambda item: item["regret"], reverse=True
        ) if item["regret"] > 0][:5]
        lines.extend(["", "## Failure cases", ""])
        if not failures:
            lines.append("- None on this evaluation split.")
        else:
            for failure in failures:
                child_suffix = (
                    f" / `{failure['selected_child_specification']}`"
                    if failure.get("selected_child_specification") else ""
                )
                lines.append(
                    f"- `{failure['task_id']}`: selected `{failure['selected_action']}`{child_suffix}, "
                    f"oracle under fixed rollout policy `{failure['oracle_action']}`, regret `{failure['regret']:.4f}`."
                )
    if experiment:
        reproduction = (
            "PYTHONPATH=research python3 -m roy_research experiment "
            f"--groups {groups_path} --output research/output/experiment "
            f"--epochs {experiment['learned_arms']['full']['metadata']['epochs']} --device cpu"
        )
    elif evaluation_path:
        reproduction = (
            "PYTHONPATH=research python3 -m roy_research report "
            f"--groups {groups_path} --evaluation {evaluation_path} --output {output}"
        )
    else:
        reproduction = f"PYTHONPATH=research python3 -m roy_research smoke --output {output.parent}"
    lines.extend([
        "",
        "## Interpretation contract",
        "",
        "This report is descriptive. A result whose paired 95% confidence interval crosses zero is reported as inconclusive.",
        "External benchmark claims require the pinned remote runner and are not inferred from controlled fixtures.",
        "The bounded local pilot is a pipeline validation; convergence and external-benchmark superiority remain unestablished.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "npm run research:test",
        reproduction,
        "```",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _group_traces(group: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoint = group["checkpoint"]
    task = group["task"]
    return [TraceRecord(
        task_id=str(task["id"]),
        checkpoint_id=str(checkpoint["id"]),
        checkpoint_fingerprint=str(checkpoint["fingerprint"]),
        parent_id=str(checkpoint["parent_id"]),
        event_graph=dict(checkpoint["event_graph"]),
        legal_actions=list(checkpoint["legal_actions"]),
        action=result["action"],
        child_specification=result.get("child_specification"),
        resources_before=dict(result["resources_before"]),
        resources_after=dict(result["resources_after"]),
        utility=float(result["utility"]),
        provider=str(result.get("provider", group.get("provider", "controlled"))),
        model=str(group.get("model", "deterministic-fixture-v1")),
        token_usage=int(result.get("token_usage", 0)),
        latency_ms=int(result["duration_ms"]),
        repeat=int(result["repeat"]),
        environment_revision=str(checkpoint["environment_revision"]),
        error=result.get("error"),
    ).to_dict() for result in group["results"]]


def _package_remote(output: Path, token_limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", build_remote_manifest(token_limit))
    source = Path(__file__).resolve().parents[1] / "remote" / "run_remote.sh"
    if source.exists():
        shutil.copy2(source, output / "run_remote.sh")
        (output / "run_remote.sh").chmod(0o755)
    proxy = Path(__file__).resolve().parents[1] / "remote" / "token_proxy.py"
    if proxy.exists():
        shutil.copy2(proxy, output / "token_proxy.py")
    package = Path(__file__).resolve().parent
    shutil.copytree(package, output / "roy_research", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
