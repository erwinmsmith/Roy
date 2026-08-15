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

The ledger defaults to a hard 10,000,000-token limit and reserves budget before every request. Package the external runner with:

```bash
PYTHONPATH=research python3 -m roy_research package-remote \
  --output research/output/remote-bundle
```

The bundle pins benchmark revisions and contains no benchmark assets. A Docker-capable host with adequate disk and benchmark-specific credentials is required for the complete tau Knowledge and TUA-Bench pilot.
The `run` mode fails closed unless `ROY_EXTERNAL_ADAPTER_READY=1` is set after verifying that both host-side benchmark agents consume `ROY_STRUCTURAL_ARM` and `ROY_STRUCTURAL_POLICY_COMMAND`; this prevents mislabeled identical-arm comparisons.
