# Roy real-DeepSeek structural rollout pilot

Date: 2026-08-17. This is a bounded implementation/pipeline pilot, not an external benchmark result.

## What ran

- Provider/model: real DeepSeek Chat Completions API, `deepseek-v4-flash`.
- Nine matched structural checkpoints: six train and three test; Activation, Acquisition and Mixed are represented in both splits.
- For each checkpoint, the fixed downstream rollout policy evaluated `CONTINUE`, three conditional child specifications under `BRANCH`, and `RETURN` when the action mask allowed it.
- Branch rollouts made a child call followed by a parent synthesis call. Utilities were calculated locally against a hidden deterministic answer and output contract; no LLM judge was used.
- The final pilot contains 66 provider requests and 25,880 billed tokens. All requests completed and all accepted outputs used a 384-token completion cap.
- A rejected 128-token configuration used another 22 requests and 7,013 tokens; reasoning tokens exhausted the cap and truncated JSON, so those groups were excluded from training.
- A 25,666-token preliminary 384-token run was used to validate scoring before checkpoint resources were added to the immutable fingerprint; it was excluded from the final training result.
- Including the earlier 175-token connectivity smoke, the persistent global ledger now records 58,734 / 10,000,000 tokens used and zero reserved.

Generated requests, responses, groups, traces and weights remain under `research/output/live/` and are excluded from Git.

## Pilot results

Four learned variants were trained for 20 CPU epochs on the six real-rollout train groups. On the three held-out groups, all four selected `CONTINUE` and obtained the same result:

| Learned arm | Test utility | Structural regret | Utility-optimal decision rate |
| --- | ---: | ---: | ---: |
| Full-trajectory GRPO | 0.9833 | 0.0000 | 1.0000 |
| CS-GRPO without event graph | 0.9833 | 0.0000 | 1.0000 |
| Node-only CS-GRPO | 0.9833 | 0.0000 | 1.0000 |
| Full hierarchical CS-GRPO | 0.9833 | 0.0000 | 1.0000 |

The no-derivation baseline and fixed-rollout joint oracle also obtained 0.9833 because every structural action tied on the three final test checkpoints. Paired differences between the learned arms, no derivation and full were exactly zero and are inconclusive. This pilot establishes that real provider rollouts now create counterfactual utilities and can train the CS-GRPO code path. It does not establish that full CS-GRPO is better: the sample is intentionally too small and the arithmetic tasks are at ceiling.

One useful configuration finding was that provider reasoning tokens count against `max_tokens`. A 128-token limit produced 20 length terminations among the first 22 calls. Raising both child and synthesis limits to 384 produced zero length terminations in the accepted 66-call pilot.

## TUA-Bench image audit

The pinned TUA-Bench revision `3497fd320abcafaf4797424192c891a593fd7964` is still the upstream `main` revision. The repository has 120 task Dockerfiles and does not specify prebuilt task images; Harbor builds each task image locally after `uv run setup-env` restores uncommitted assets.

No TUA-Bench task image is currently present in the local Docker image store. All five referenced base tags still resolve:

| Base image | Manifest digest | Platform note |
| --- | --- | --- |
| `debian:bookworm` | `sha256:813017f3d62be4b5891a7acca6a01bdcd4b8513daa81b1ab99d3a50385b26931` | multi-architecture |
| `ubuntu:24.04` | `sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea` | multi-architecture |
| `ubuntu:20.04` | `sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214` | multi-architecture |
| `cellprofiler/cellprofiler:4.2.8` | `sha256:fec440caa2b44edf80f9bd440a3d8c6b214d72437332ad2586a32d79e2eae2a4` | Linux/amd64 only |
| `openfoam/openfoam11-paraview510` | `sha256:fd10956e0b1eb70f9808baf2857e4baf846a0f6f272f73b6d00546eae96be181` | Linux/amd64 only |

HEAD checks for the upstream draw.io, Google Drive MRI, PE-Video, floor-plan, NREL ComStock and CellProfiler asset endpoints returned HTTP 200. The assets were not downloaded and task images were not built in this audit. Because two large scientific bases are amd64-only, a Linux/amd64 server remains the recommended target for the full TUA-Bench pilot.

## Reproduce

```bash
PYTHONPATH=research python3 -m roy_research collect-live \
  --tasks research/output/full/tasks.jsonl \
  --task-ids activation-000 activation-001 activation-040 \
             acquisition-000 acquisition-001 acquisition-040 \
             mixed-000 mixed-001 mixed-040 \
  --output research/output/live/groups-final.jsonl \
  --traces research/output/live/traces-final.jsonl \
  --events research/output/live/events.jsonl \
  --ledger research/output/live/token-ledger.json \
  --repeats 1 --max-tokens 384 --temperature 0 --resume

PYTHONPATH=research python3 -m roy_research experiment \
  --groups research/output/live/groups-final.jsonl \
  --output research/output/live/experiment-final \
  --epochs 20 --device cpu
```
