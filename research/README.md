# Roy structural-learning research

This directory contains the canonical theory, controlled benchmark, CS-GRPO policy code and pinned external-pilot packaging. Generated trajectories, model weights, embedding caches and external benchmark assets are intentionally ignored by Git.

## Local quick start

```bash
npm run research:test
npm run research:smoke

PYTHONPATH=research python3 -m roy_research generate --output research/output/tasks.jsonl
PYTHONPATH=research python3 -m roy_research collect \
  --tasks research/output/tasks.jsonl \
  --output research/output/groups.jsonl \
  --traces research/output/traces.jsonl \
  --resume
PYTHONPATH=research python3 -m roy_research train \
  --groups research/output/groups.jsonl \
  --output research/output/cs_grpo.pt
PYTHONPATH=research python3 -m roy_research evaluate \
  --groups research/output/groups.jsonl \
  --model research/output/cs_grpo.pt \
  --output research/output/evaluation.json
PYTHONPATH=research python3 -m roy_research report \
  --groups research/output/groups.jsonl \
  --evaluation research/output/evaluation.json \
  --output research/output/report.md

PYTHONPATH=research python3 -m roy_research experiment \
  --groups research/output/groups.jsonl \
  --output research/output/experiment \
  --epochs 3 --resume
```

The encoder is loaded from the local Hugging Face cache at the pinned revision. To use the learned policy from TypeScript, launch `python3 -m roy_research.policy_server` through `PythonStructuralPolicyClient` and set `ROY_STRUCTURAL_MODEL` to a local checkpoint.

## Live and remote execution

Live API execution is opt-in:

```bash
PYTHONPATH=research python3 -m roy_research api-smoke \
  --ledger research/output/token-ledger.json \
  --output research/output/api-smoke.json
```

To collect counterfactual groups from the real DeepSeek downstream rollout policy, select a bounded task slice explicitly. Each completed task is checkpointed to JSONL immediately, and `events.jsonl` is an append-only request/response audit log. The API key is read only from `DEEPSEEK_API_KEY`.

```bash
PYTHONPATH=research python3 -m roy_research collect-live \
  --tasks research/output/tasks.jsonl \
  --task-ids activation-000 acquisition-000 mixed-000 \
  --output research/output/live/groups.jsonl \
  --traces research/output/live/traces.jsonl \
  --events research/output/live/events.jsonl \
  --ledger research/output/live/token-ledger.json \
  --repeats 1 --max-tokens 384 --resume

PYTHONPATH=research python3 -m roy_research train \
  --groups research/output/live/groups.jsonl \
  --output research/output/live/cs-grpo.pt \
  --epochs 3 --device cpu
```

Transport failures are fail-closed: because a timed-out provider request may still have consumed tokens, its full reservation is charged to the ledger. Resume skips only complete groups; it never fills a missing counterfactual with fixture utility.

The ledger defaults to a hard 10,000,000-token limit and reserves budget before every request. Package the external runner with:

```bash
PYTHONPATH=research python3 -m roy_research package-remote \
  --output research/output/remote-bundle

research/output/remote-bundle/run_remote.sh audit-images
```

The bundle pins benchmark revisions and contains no benchmark assets. A Docker-capable host with adequate disk and benchmark-specific credentials is required for the complete tau Knowledge and TUA-Bench pilot.
On a CPU host without container privileges, `run_remote.sh prepare-tau` prepares only the pinned tau Knowledge environment and writes `prepared-tau.json`; it does not claim that the complete external pilot is ready. The full `prepare` and `run` modes continue to require the TUA container environment.
The `run` mode fails closed unless `ROY_EXTERNAL_ADAPTER_READY=1` is set after verifying that both host-side benchmark agents consume `ROY_STRUCTURAL_ARM` and `ROY_STRUCTURAL_POLICY_COMMAND`; this prevents mislabeled identical-arm comparisons.

TUA-Bench does not publish one prebuilt benchmark image. Its 120 task images are built from the Dockerfiles under `tasks/`; `uv run setup-env` first downloads the uncommitted benchmark assets. A local image audit should therefore check both the five referenced base-image manifests and the setup asset endpoints before scheduling a run. Scientific tasks based on CellProfiler/OpenFOAM may require Linux/amd64 emulation on Apple Silicon, so the full pilot remains a server workload.
