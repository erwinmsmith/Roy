from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .analysis import paired_bootstrap_interval, summarize_groups
from .baselines import evaluate_controlled_arms
from .benchmarks import build_remote_manifest
from .controlled import collect_group, generate_tasks, mechanism_diagnostics
from .controlled import TERMINAL_SUCCESS_THRESHOLD
from .io import atomic_json, read_jsonl, write_jsonl
from .live_controlled import collect_live_group
from .providers import DeepSeekClient
from .reporting import write_utility_svg
from .schema import TraceRecord
from .token_ledger import PersistentTokenLedger
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
    collect_live.add_argument("--limit", type=int)
    collect_live.add_argument("--repeats", type=int, default=2)
    collect_live.add_argument("--max-tokens", type=int, default=384)
    collect_live.add_argument("--temperature", type=float, default=0.0)
    collect_live.add_argument("--token-limit", type=int, default=10_000_000)
    collect_live.add_argument("--timeout", type=float, default=90.0)
    collect_live.add_argument("--resume", action="store_true")

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

    experiment = commands.add_parser("experiment", help="Train and compare all controlled CS-GRPO ablations")
    experiment.add_argument("--groups", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--epochs", type=int, default=3)
    experiment.add_argument("--device")
    experiment.add_argument("--resume", action="store_true")
    return root


def main(argv: List[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "generate":
        tasks = [task.to_dict() for task in generate_tasks(args.seed)]
        write_jsonl(args.output, tasks)
        print(json.dumps({"tasks": len(tasks), "output": str(args.output)}))
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
        if args.limit is not None:
            tasks = tasks[: args.limit]
        completed = set()
        if args.resume and args.output.exists():
            completed = {record["task"]["id"] for record in read_jsonl(args.output)}
        ledger = PersistentTokenLedger(args.ledger, args.token_limit)
        client = DeepSeekClient(ledger, timeout=args.timeout, event_log=args.events)
        collected = 0
        trace_count = 0
        trace_batch = 0
        for task_value in tasks:
            task = _task_from_dict(task_value)
            if task.id in completed:
                continue
            group = collect_live_group(
                task, client, repeats=args.repeats,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            append_group = args.output.exists() and (args.resume or collected > 0)
            write_jsonl(args.output, [group], append=append_group)
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
        print(json.dumps({
            "collected": collected, "traces": trace_count,
            "output": str(args.output), "ledger": ledger.snapshot(),
        }))
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
        lines.extend([
            f"- Split: `{evaluation['split']}`",
            f"- Decision accuracy: `{evaluation['accuracy']:.4f}`",
            f"- Mean utility: `{evaluation['mean_utility']:.4f}`",
            f"- Mean structural regret: `{evaluation['mean_regret']:.4f}`",
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
        for failure in failures:
            child_suffix = (
                f" / `{failure['selected_child_specification']}`"
                if failure.get("selected_child_specification") else ""
            )
            lines.append(
                f"- `{failure['task_id']}`: selected `{failure['selected_action']}`{child_suffix}, "
                f"oracle under fixed rollout policy `{failure['oracle_action']}`, regret `{failure['regret']:.4f}`."
            )
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
        f"PYTHONPATH=research python3 -m roy_research experiment --groups research/output/full/groups.jsonl --output research/output/full/experiment --epochs {experiment['learned_arms']['full']['metadata']['epochs'] if experiment else 1} --device cpu",
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
