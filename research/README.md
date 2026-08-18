# Roy information-realization research

This directory contains the canonical theory, recursive epistemic runtime, autonomous organization policy, organization-GRPO training code and τ³ integration. Generated trajectories, model weights, embedding caches and benchmark assets are intentionally ignored by Git.

The current research method uses one policy, one on-policy training process and one optimization signal: terminal τ³ task utility. It has no teacher policy, predefined agent-role catalog, imitation warm start or weighted objective sum. Observable LLM-call, tool-call, node, depth and decision budgets are enforced through the legal-action mask; resource use is never subtracted from reward.

## τ³ preparation and training

```bash
npm run research:test
npm run research:smoke

PYTHONPATH=research python3 -m roy_research tau3-manifest \
  --tau3-root ~/rivermind-data/benchmarks/tau3-bench-v1.0.1 \
  --output research/output/tau3/manifest.jsonl

PYTHONPATH=research python3 -m roy_research tau3-train \
  --manifest research/output/tau3/manifest.jsonl \
  --trajectories research/output/tau3/train-trajectories.jsonl \
  --model research/output/tau3/organization-policy.pt \
  --epochs 4 --organization-temperature 2.0 --max-tokens 50000 --resume

PYTHONPATH=research python3 -m roy_research tau3-evaluate \
  --manifest research/output/tau3/manifest.jsonl \
  --model research/output/tau3/organization-policy.pt \
  --output research/output/tau3/test-results.jsonl \
  --summary research/output/tau3/test-summary.json --split test
```

On the configured server, Roy and τ³ use separate Python environments. Use the checked-in launcher so Roy's PyTorch/encoder dependencies and τ³'s benchmark dependencies are resolved in a stable order:

```bash
research/remote/run_tau3.sh tau3-train \
  --manifest research/output/tau3/manifest.jsonl \
  --trajectories research/output/tau3/train-trajectories.jsonl \
  --model research/output/tau3/organization-policy.pt \
  --epochs 4 --max-tokens 50000 --resume
```

Set `TAU3_ROOT` only when the pinned checkout is not at the sibling `benchmarks/tau3-bench-v1.0.1` path.

The τ³ checkout is pinned to commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`. Airline, retail and telecom preserve their official train/test split; validation is reserved deterministically from official training tasks. Banking knowledge is held out because the pinned benchmark does not define an official training split for it.

Each task/epoch provides exactly eight complete organization trajectories under one shared exploration envelope, runtime budget and environment seed. Only the organization sampling seed differs. Every tool exposed by the selected τ³ domain becomes its own legal `ACQUIRE` candidate. DeepSeek generates only a JSON argument object from the selected tool's official schema; Roy deterministically binds the policy-selected tool name and constructs the τ³ ToolCall, so the provider cannot substitute another tool. The official environment remains responsible for argument validation and execution. Roy's report, argument and final-synthesis calls use DeepSeek non-thinking mode because the pinned τ³ message adapter does not preserve the `reasoning_content` required after tool calls; the direct baseline and user simulator retain their configured mode. Roy marks policy `STOP` and resource truncation with the official Agent stop protocol so the orchestrator terminates and scores the episode immediately. Every LLM request and response is written under the run's `llm-calls/` directory. Airline, retail and telecom expose 14, 16 and 13 tools respectively. Banking uses the official AllTools construction with the pinned local MiniLM encoder, BM25, dense retrieval, shell and dynamically discovered tools; it does not require a remote embedding service.

The exact masked old-policy probability is recorded for every `active node → conditional candidate` decision and replay uses the identical mask and organization temperature. Training defaults to temperature `2.0` to broaden genuine on-policy exploration without mixing in a separate random behavior policy; evaluation defaults to `1.0`. One GRPO update follows each freshly sampled group. The default four epochs anneal conditional node/depth floors as `(6,3) → (4,2) → (2,1) → (0,0)`. While a floor is unmet, sampling is directed among legal `DERIVE` actions only when the LLM report contains a genuine residual requirement; otherwise reasoning and acquisition remain available. A report residual without an explicit child proposal becomes an open child specification grounded in that residual, rather than a predefined role or synthetic task. Evaluation removes minimum node/depth floors and runs one organization per episode.

Saved schema-v3 trajectories include every selected child specification plus a topology summary (`single`, `chain`, `fan_out_tree`, `dependency_dag`, `communication_dag`, or `hybrid_dag`). Dependency children wait for required producer reports and wake on producer return. Optional communication is sparse and forward-only, preserving a DAG while allowing the sampled policy to realize different cross-branch structures. User-information residuals are handled as resumable `ACQUIRE` interactions with the τ³ user simulator, while tool-only residuals remain tool acquisitions; neither is recursively copied into child agents. During the conditional structural floor, exact tool hints from the model report select matching official tool candidates before free tool exploration resumes. The first training envelope targets six total nodes when genuine substantive gaps support them; it does not add node count, topology, communication, or resource terms to terminal utility.

The primary baseline is `single_agent_direct`. Reports also distinguish `fixed_complete_mas`, `roy_runtime_heuristic` and `learned_information_realization`; end-to-end success and official τ³ reward are reported separately from organization diagnostics.

## Legacy controlled fixtures

The older deterministic controlled commands remain available as regression fixtures for runtime and math tests. They are not the benchmark or training protocol for the current theory.

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

# Full versioned training collection: exactly the 90 train tasks, K=3 and M=2.
PYTHONPATH=research python3 -m roy_research collect-live \
  --tasks research/output/tasks.jsonl --split train --problem-version v2 \
  --output research/output/live-v2/train/groups.jsonl \
  --traces research/output/live-v2/train/traces.jsonl \
  --events research/output/live-v2/train/events.jsonl \
  --ledger research/output/token-ledger.json \
  --repeats 2 --max-tokens 384 --resume

PYTHONPATH=research python3 -m roy_research train \
  --groups research/output/live/groups.jsonl \
  --output research/output/live/cs-grpo.pt \
  --epochs 3 --device cpu
```

The v2 live suite replaces the ceiling-prone arithmetic fixture with modular recurrences, hidden-coefficient polynomial evaluation and evidence-updated shortest paths. Keep train, validation and test in separate output directories; the `--split` filter is applied before collection, training reads only `train`, and evaluation never updates model weights.

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
