# Roy information-realization research

This directory contains the canonical theory, recursive epistemic runtime, autonomous organization policy, LHTB/Harbor adapter, continuous process-credit GRPO code and τ³ transfer integration. Generated trajectories, model weights, embedding caches, Docker images and benchmark assets are intentionally ignored by Git.

The current research method uses one policy, one on-policy training process and one environment objective: official LHTB final reward. A separately trained value model turns that target into telescoping `Delta V` decision credit; it does not add a second reward. There is no teacher policy, predefined agent-role catalog, imitation warm start, weighted objective sum, keyword scoring, node target, depth target or token penalty.

## Primary LHTB workflow

LHTB and its bundled Harbor are pinned to commit `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`. The checked [split manifest](config/lhtb_split.json) fixes all 46 official tasks to 30 train, 8 dev and 8 test records by per-category SHA-256 ordering. The trainer rejects dev/test IDs and stale policy revisions.

```bash
PYTHONPATH=research python3 -m roy_research lhtb-manifest \
  --lhtb-root /path/to/LHTB --output research/output/lhtb/manifest.json

PYTHONPATH=research python3 -m roy_research lhtb-schedule \
  --manifest research/output/lhtb/manifest.json \
  --output research/output/lhtb/schedule.json

# Apply one freshly sampled current-policy G=8 group at a time. This updates
# actor -> value -> EMA and refuses a repeated group optimizer step.
PYTHONPATH=research python3 -m roy_research lhtb-update \
  --manifest research/output/lhtb/manifest.json \
  --trajectories research/output/lhtb/train-trajectories.jsonl \
  --model research/output/lhtb/checkpoints/current.pt \
  --updates research/output/lhtb/update-audit.jsonl --resume
```

The formal schedule is four epochs over all 30 train tasks, `G=8`, for 960 rollouts. Every group uses fresh matched environments with the same task checksum, immutable environment digest, initial fingerprint and runtime config. Each rollout has a 60-minute training deadline, concurrency is four, and one DeepSeek response is capped at 32,768 tokens. These are execution settings, not reward terms and not forced node/depth limits.

Each organization action and terminal result appends an immutable `GlobalEpistemicState` `M_t`. Frozen DeepSeek prompts separately extract entities and verify `entail / contradict / unknown`; pinned MiniLM only recalls top-eight candidate pairs. All requests, responses, cache keys and model revisions are retained. No benchmark keyword field, lexical rule, regex, frequency score or embedding threshold labels meaning.

The append-only ledger remains complete, but the proposer and actor receive a bounded epistemic working state: unresolved requirements, active nodes, high-relevance entities, graph summaries and references into recent raw events. Semantic extraction is incremental, and the organization actor is invoked only on an initial decision or a real event boundary such as a new gap, failure, file change, contradiction, completed node, structural action, or a configurable terminal-result interval. Every sampled organization decision records raw and masked action probabilities, the selected action, exact old log-probability, real residual-gap count and STOP-mask reason.

The critic is independent of the actor except for the frozen encoder. For non-final decisions `r_proc = V_bar(M_t+1)-V_bar(M_t)` and the final credit is `R-V_bar(M_T-1)`, so returns telescope to `R-V_bar(M_t)`. Every trajectory has total group-statistic weight one. Equal terminal rewards still train value; actor updates only when shaped returns vary.

Use a dedicated x86_64 Docker VM (16 vCPU, 64 GB RAM, at least 200 GB and preferably 300 GB SSD):

```bash
ssh exp-roy-lhtb
cd ~/rivermind-data/roy
research/remote/prepare_lhtb.sh check
research/remote/prepare_lhtb.sh prepare
research/remote/prepare_lhtb.sh oracle-smoke
research/remote/prepare_lhtb.sh roy-smoke

# Formal on-policy training, per-epoch dev selection, then frozen three-arm test.
research/remote/run_lhtb_training.sh
research/remote/run_lhtb_test.sh
```

The preflight stops below 15% free disk and never runs destructive Docker prune. Every optimizer group and Harbor trial is persisted before proceeding. The current `exp-roy` host is a container without a Docker socket, so it is not a valid formal LHTB runner; use the planned `exp-roy-lhtb` VM.

### GPUHome native-process track

GPUHome container instances do not expose the kernel capabilities needed for Docker, containerd, Podman, Buildah, Apptainer or a nested VM. For this host, Roy provides a non-official Harbor `BaseEnvironment` implementation backed by native Linux processes:

```bash
ssh exp-roy
cd ~/rivermind-data/roy
research/remote/prepare_lhtb_native.sh check
research/remote/prepare_lhtb_native.sh prepare
research/remote/prepare_lhtb_native.sh provision
research/remote/prepare_lhtb_native.sh oracle-smoke
research/remote/prepare_lhtb_native.sh roy-smoke

# These fail closed until every task in the selected split has a reviewed,
# digest-matched native provisioning manifest.
research/remote/run_lhtb_native_training.sh
research/remote/run_lhtb_native_test.sh
```

Each trial gets a copied workspace and a distinct unprivileged UID. GPUHome sets `/` to mode `0700`, so preparation grants only execute/traverse access on `/` to dedicated numeric GID `210000`; it does not grant directory listing or file read access. Every task UID uses that GID while its session root is mode `0700`, preventing sibling rollouts from reading one another. PRoot supplies stable `/app`, `/workspace`, `/tests`, `/solution`, `/tmp`, `/opt`, `/root` and `/logs` paths; process groups are terminated on cleanup, the task/template digest is checked before execution, and provider secrets are not inherited by task commands. The checked provisioning catalog enumerates all 46 tasks, but server-side template conversion and oracle validation are still incomplete. Every selected task must be provisioned and pass the audit before the 30-task training schedule can start.

OCI-backed native templates preserve two different identities: the pinned source-registry digest and the digest produced by the local OCI layout conversion. They are verified and recorded separately. Environment-invalid attempts such as an inactive actor are retained as audit records and retried by Harbor; they never receive a synthetic zero reward or enter actor/value updates. The fresh-sampling gate requires exactly eight valid rewards, distinct seeds, one matched initial fingerprint, complete process-state chains, nonzero reward variance, current compact-policy interface records and no more than 15 million aggregate input tokens per `G=8` group.

The topology-rich sampling profile defaults to a six-node exploration target and presents 6–8 nodes as a proposer preference, not a utility term or hard maximum. Before the first decision, the frozen semantic extractor decomposes only the explicit, independently verifiable requirements in the benchmark instruction; this replaces the single aggregate root gap without role templates, keywords or invented work. While the target is unmet, every real open parent-local gap must remain represented by enough distinct legal `DERIVE` candidates (up to three per decision), and states with a shallow candidate interface are saved as `sampling_invalid` and resampled. No gap is synthesized to meet the target. At three or more active nodes the proposer must also expose a novel communication edge when one is available; duplicate active edges are rejected. Dependency edges are allowed only when a child genuinely requires an existing producer's `report:<nodeId>` artifact, and returning that report resolves the edge and wakes the consumer.

GPUHome currently cannot reach Docker Hub directly. Native preparation therefore defaults to the `dockerproxy.net` transport mirror, which was verified to serve the exact configured source manifest digest. Set `ROY_LHTB_OCI_MIRROR` to override it. The mirror is never trusted by tag alone: provisioning pulls `repository@digest`, verifies the pinned source digest, selects only the explicit `linux/amd64` image when that source is a multi-architecture list, and then separately verifies the converted local OCI-layout digest. Runtime, templates and a resumable OCI blob cache default to the expanded `~/rivermind-data/lhtb-native` volume; failed conversion attempts remove task half-products but retain verified downloaded layers for the next attempt.

Native task virtual environments default to the host's Python 3.12 and are included in the content-derived template digest. Set `ROY_LHTB_NATIVE_TASK_PYTHON` only when deliberately evaluating another compatible local interpreter; this is another reason native results are not byte-equivalent to the original Python 3.11 Docker images.

This backend intentionally records `network_isolation=false` and the absence of PID/mount namespaces. A task declaring `allow_internet=false` is rejected unless `ROY_LHTB_ALLOW_NETWORK_DEGRADED=true` is explicitly set, and such a run is labeled degraded. Native results use a content-derived environment digest and the original final verifier, but they are not Docker-equivalent or official leaderboard-comparable. Reports must keep them separate from the official Docker protocol.

Dev checkpoint selection uses highest mean reward, then lower value MAE, fewer tokens and earlier epoch. Final test compares `single_agent_direct`, `roy_runtime_heuristic` and `learned_information_realization` for three repetitions. Direct uses the same model/executor/runtime while enforcing one root node and no communication. Report mean reward, success at `R >= 0.95`, paired bootstrap 95% CI, tokens, time, topology, `Delta V`, value calibration and failures.

See [the implementation note](reports/lhtb-process-reward-implementation.md) for the delivered components, validation boundary and formal-run outputs.

## τ³ transfer compatibility (not primary training)

The commands below preserve the earlier τ³ adapter and its historical exploration-envelope behavior for regression and transfer work. Its envelope is separate from LHTB's real-gap-only topology-rich sampling profile; neither is part of the task utility or a theoretical structural objective, and τ³ must not be reported as the primary training method.

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
  --epochs 4 --organization-temperature 2.0 --max-tokens 50000 \
  --max-steps 1000 --max-rollout-attempts 3 --resume

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
  --epochs 4 --max-tokens 50000 --max-steps 1000 \
  --max-rollout-attempts 3 --resume
```

Set `TAU3_ROOT` only when the pinned checkout is not at the sibling `benchmarks/tau3-bench-v1.0.1` path.

The τ³ checkout is pinned to commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`. Airline, retail and telecom preserve their official train/test split; validation is reserved deterministically from official training tasks. Banking knowledge is held out because the pinned benchmark does not define an official training split for it.

Each task/epoch provides exactly eight complete organization trajectories under one shared exploration envelope, runtime configuration and environment seed. Only the organization sampling seed differs. Every tool exposed by the selected τ³ domain is available, but it becomes a legal `ACQUIRE` candidate only when the current report identifies a tool-access residual. Exact tool hints narrow the candidates; a generic tool residual keeps all domain tools open. A stable `node + tool` identity prevents one node from looping on the same acquisition; genuinely distinct repeated uses of a tool are delegated to separate locally grounded nodes. DeepSeek generates only a JSON argument object from the selected tool's official schema; Roy deterministically binds the policy-selected tool name and constructs the τ³ ToolCall, so the provider cannot substitute another tool. The official environment remains responsible for argument validation and execution. Roy's report, argument and final-synthesis calls use DeepSeek non-thinking mode because the pinned τ³ message adapter does not preserve the `reasoning_content` required after tool calls; the direct baseline and user simulator retain their configured mode. Every LLM request and response is written under the run's `llm-calls/` directory. Airline, retail and telecom expose 14, 16 and 13 tools respectively. Banking uses the official AllTools construction with the pinned local MiniLM encoder, BM25, dense retrieval, shell and dynamically discovered tools; it does not require a remote embedding service.

Every rollout attempt is appended to the trajectory JSONL immediately. A censored attempt is marked `accepted_for_training=false` and retried with the same environment seed but a fresh organization seed; it is never passed to GRPO. Complete zero-variance groups remain in the sample record but do not execute an optimizer step, because terminal utility supplies no preference signal. Training continues over the complete train split so later groups can provide nonzero within-group utility variance.

The τ³ orchestrator error allowance is tied to `--max-steps` rather than a separate small constant, so repeated recoverable tool errors do not silently reintroduce a ten-error trajectory ceiling.

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
