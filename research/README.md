# Roy information-realization research

This directory contains the canonical theory, recursive epistemic runtime, autonomous node-level organization policy, LHTB/Harbor adapter, forced-finalize state-value/node-wise GRPO experiment, terminal-reward baseline and τ³ transfer integration. Generated trajectories, model weights, embedding caches, Docker images and benchmark assets are intentionally ignored by Git.

The current method uses one shared six-action node-level Controller and a frozen semantic Worker. The Controller is invoked separately whenever root, child or deeper descendant owns execution; each invocation consumes that node's local context/ancestry plus the current global `M_t` graph. It chooses only `CONTINUE / DERIVE_INFO / DERIVE_ORG / PRUNE / RETURN / FINISH`. It never chooses commands, child descriptions, reports, connection/reuse targets or prune targets.

Structural training compares `G=8` macro-actions cloned from the same node checkpoint. An independently trained `V_psi(S)` is supervised only by a fixed finalize-now readout scored by the official LHTB verifier. One frozen value revision supplies `Delta V = V_psi(S_next) - V_psi(S)` for the complete group, followed by group-relative clipped actor learning. This is counterfactual training data, not MCTS: inference directly samples one action per real node and does not use the value model or sibling outcomes. The earlier complete-trajectory terminal-score updater remains available as an explicit baseline and never shares groups with the node-wise updater.

The implementation names the benchmark output `u_env`/`environment_utility`, never node reward (the historical `official_lhtb_task_utility` field remains an audit alias). `nodewise_checkpoint_finalize` first captures an idle, service-free Roy session and native filesystem checkpoint. Each successor trial restores both identities, invokes the shared actor for exactly one scheduler-selected node, completes at most that macro-action's terminal side effect, stops before another policy decision, and lets the unchanged official verifier score the resulting artifact state. Only after `V_psi` is fitted from these forced-finalize utilities does Roy emit the theoretically derived `R_t^MIA = V_psi(S_t+1)-V_psi(S_t)`; no environment utility is copied into a step reward.

## Primary LHTB workflow

LHTB and its bundled Harbor are pinned to commit `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`. The checked [split manifest](config/lhtb_split.json) fixes all 46 official tasks to 30 train, 8 dev and 8 test records by per-category SHA-256 ordering. The trainer rejects dev/test IDs and stale policy revisions.

```bash
PYTHONPATH=research python3 -m roy_research lhtb-manifest \
  --lhtb-root /path/to/LHTB --output research/output/lhtb/manifest.json

PYTHONPATH=research python3 -m roy_research lhtb-schedule \
  --manifest research/output/lhtb/manifest.json \
  --output research/output/lhtb/schedule.json

# Initialize the main node-wise actor/value checkpoint.
PYTHONPATH=research python3 -m roy_research lhtb-nodewise-init \
  --manifest research/output/lhtb/manifest.json \
  --model research/output/lhtb/checkpoints/nodewise-current.pt

# Record each snapshot's K official frozen-finalize probes as V(S) labels,
# then update only the value estimator.
PYTHONPATH=research python3 -m roy_research lhtb-finalize-label \
  --state research/output/lhtb/probes/state.json \
  --output research/output/lhtb/value-labels.jsonl \
  --label-id TASK:CHECKPOINT --task-id TASK --split train \
  --checkpoint-id CHECKPOINT --finalizer-revision frozen-a0 \
  --task-checksum TASK_SHA --environment-digest sha256:ENV \
  --clone-mode full_clone --clone-audit-id CLONE_AUDIT \
  --harbor-result /path/to/forced-finalize/result.json

PYTHONPATH=research python3 -m roy_research lhtb-value-update \
  --manifest research/output/lhtb/manifest.json \
  --labels research/output/lhtb/value-labels.jsonl \
  --model research/output/lhtb/checkpoints/nodewise-current.pt \
  --updates research/output/lhtb/value-update-audit.jsonl --resume

# Apply one same-state G=8 macro-action group using one frozen value revision.
PYTHONPATH=research python3 -m roy_research lhtb-node-update \
  --manifest research/output/lhtb/manifest.json \
  --groups research/output/lhtb/node-macro-groups.jsonl \
  --model research/output/lhtb/checkpoints/nodewise-current.pt \
  --updates research/output/lhtb/node-update-audit.jsonl --resume

# Terminal-reward completion-level GRPO baseline only.
PYTHONPATH=research python3 -m roy_research lhtb-update \
  --manifest research/output/lhtb/manifest.json \
  --trajectories research/output/lhtb/train-trajectories.jsonl \
  --model research/output/lhtb/checkpoints/current.pt \
  --updates research/output/lhtb/update-audit.jsonl --resume
```

The node-wise formal schedule is four epochs over all 30 train tasks, with two sequential decision rounds per task/epoch and `G=8`, for 240 groups and 1,920 rollouts. The round count is configurable with `ROY_LHTB_DECISION_ROUNDS_PER_TASK_PER_EPOCH`; two rounds provide eight decision opportunities per task across training, so a trajectory can grow from one root to as many as nine nodes when the learned actor repeatedly derives. Nothing enforces a minimum node count, depth, edge count or topology: `CONTINUE`, `RETURN`, `PRUNE` and `FINISH` remain legal when state and action masks permit them, so single-agent and shallow outcomes remain part of the same distribution. Every group uses fresh matched environments with the same task checksum, immutable environment digest, initial fingerprint and runtime config. Each rollout has a six-hour training deadline, concurrency is four, and one DeepSeek response is capped at 32,768 tokens. Native runs reserve the final 30 seconds for append-only deadline finalization, return normally to Harbor, and then invoke the official verifier on the partial environment. These are execution settings, not reward terms.

For controlled deeper-topology experiments, use a separate round-count schedule and filter execution without changing or replaying completed groups. For example, the following resumes only epoch 0 of `riscv-core-debug` with six rounds per task/epoch; completed rounds 0 and 1 are skipped, while rounds 2 through 5 extend the same append-only task trajectory. Six rounds allow at most seven nodes within the epoch because every node-level macro action adds at most one node, but `FINISH` remains legal and no node count is forced.

```bash
ROY_LHTB_DECISION_ROUNDS_PER_TASK_PER_EPOCH=6 \
ROY_LHTB_SCHEDULE="$ROY_LHTB_RUN_ROOT/schedule-rounds-6.json" \
ROY_LHTB_TASK_FILTER=riscv-core-debug \
ROY_LHTB_EPOCH_FILTER=0 \
bash research/remote/run_lhtb_native_training.sh
```

Each organization action and terminal result appends an immutable `GlobalEpistemicState` `M_t`. Frozen DeepSeek prompts separately extract entities and verify `entail / contradict / unknown`; pinned MiniLM only recalls top-eight candidate pairs. All requests, responses, cache keys and model revisions are retained. No benchmark keyword field, lexical rule, regex, frequency score or embedding threshold labels meaning.

The append-only ledger remains complete. On every organization decision, the actor receives the acting node's local objective, parent/depth/status, assigned requirements, open child specification, tool access, termination condition and recent node-local events, together with a typed relational projection of the complete current `M_t` organization/epistemic graph. The same shared actor parameters are used separately for every node. Every sampled decision records the exact node context, candidates, raw and masked probabilities, selected action and exact old log-probability.

For the main node-wise algorithm, each G=8 group shares one exact node checkpoint and reaches eight meaningful macro-action boundaries. The frozen value revision scores the base and successors, and GRPO normalizes the eight `Delta V` values. Forced-finalize labels retain every official score and finalizer seed. RMSE, Spearman and within-task pairwise ranking accuracy are required value diagnostics. Ordinary adjacent `M_t -> M_t+1` transitions without a forced-finalize-trained value revision remain audit-only.

Actor optimization is rejected at value revision 0. The initial constant-output value model may verify collection plumbing, but it cannot define a trainable MIA reward group. At least one successful forced-finalize value update is required, after which actor groups must be freshly sampled with that exact frozen value revision.

For the terminal baseline, GRPO instead normalizes eight official full-trajectory final scores and assigns one trajectory advantage to all decisions. Equal terminal scores skip that baseline actor update. Baseline and node-wise JSONL/checkpoints are deliberately incompatible.

Use a dedicated x86_64 Docker VM (16 vCPU, 64 GB RAM, at least 200 GB and preferably 300 GB SSD):

```bash
ssh exp-roy-lhtb
cd ~/rivermind-data/roy
research/remote/prepare_lhtb.sh check
research/remote/prepare_lhtb.sh prepare
research/remote/prepare_lhtb.sh oracle-smoke
research/remote/prepare_lhtb.sh roy-smoke

# Formal node-wise Delta-V training. The terminal-score trajectory runner is
# retained separately as an explicit baseline.
research/remote/run_lhtb_nodewise_training.sh
research/remote/run_lhtb_training.sh  # terminal-GRPO baseline only
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

OCI-backed native templates preserve two different identities: the pinned source-registry digest and the digest produced by the local OCI layout conversion. They are verified and recorded separately. Environment-invalid attempts such as an inactive actor are retained as audit records and retried by Harbor; they never receive a synthetic zero utility or enter updates. The node-wise gate requires eight complete macro-action outcomes from one base fingerprint, actor revision and frozen value revision. Forced-finalize labels require official verifier task utility plus clone/result provenance. `Delta V` variance controls the node-wise actor update; official-utility variance applies only to the terminal baseline. Aggregate tokens remain a diagnostic, not a reward.

Every formal rollout directly samples from the current Controller with a distinct organization seed. Runtime first builds a payload-free six-way mask from hard legality only; frozen-Worker preference cannot suppress a legal action. After the Controller samples, the frozen Worker receives that category and materializes the command, child specification, report, reuse/connection or prune target. The Worker receives no node-count, depth or topology label. A legal root `FINISH` option permits genuine single-node trajectories; repeated real `DERIVE_INFO` and `DERIVE_ORG` choices produce branching and recursion one child at a time, while Worker/Runtime reuse and communication preserve DAG structure. `ROY_LHTB_TOPOLOGY_PROFILE` remains diagnostic-only and is rejected by the trainer. Every derivation consumes a real parent-local gap, duplicate objectives/communication edges are rejected, and producer dependencies refer to genuine artifacts.

Training saves every adjacent `M_t -> M_t+1` record and decision span, including topology snapshots, exact node/edge/status deltas, acting node and selected action. Ordinary transitions are structural audit samples. A transition becomes a node-wise reward sample only when it belongs to a same-base macro group and records the frozen forced-finalize value revision used to compute `Delta V`.

No MCTS occurs during collection, update or inference. Main training does create eight one-macro-action counterfactuals from a clonable checkpoint, solely to form a same-state GRPO group; it never searches deeper, backs up values or selects the live action. Inference samples and executes one real action, then later invokes the same actor for whichever node owns execution. The terminal baseline instead uses eight complete current-policy trajectories. The two protocols use separate schemas and update commands.

The actor uses relational message passing plus attention/mean/max graph pooling and a separately indexed representation for the acting node. The independent value estimator uses a typed relational graph and is absent from inference. Legacy MCTS/routing checkpoints and the old trajectory-shaped value checkpoints remain audit-only; node-wise training initializes a fresh `lhtb-nodewise-init` checkpoint.

Sample audit separates protocol trainability from optimization signal. `preconditions_for_training` requires eight complete matched direct-actor trajectories; `actor_terminal_signal_available` means official final scores have nonzero within-group variance. A legal zero-variance group is saved but correctly skips the actor update.

GPUHome currently cannot reach Docker Hub directly. Native preparation therefore defaults to the `dockerproxy.net` transport mirror, which was verified to serve the exact configured source manifest digest. Set `ROY_LHTB_OCI_MIRROR` to override it. The mirror is never trusted by tag alone: provisioning pulls `repository@digest`, verifies the pinned source digest, selects only the explicit `linux/amd64` image when that source is a multi-architecture list, and then separately verifies the converted local OCI-layout digest. Runtime, templates and a resumable OCI blob cache default to the expanded `~/rivermind-data/lhtb-native` volume; failed conversion attempts remove task half-products but retain verified downloaded layers for the next attempt.

Native task virtual environments default to the host's Python 3.12 and are included in the content-derived template digest. Set `ROY_LHTB_NATIVE_TASK_PYTHON` only when deliberately evaluating another compatible local interpreter; this is another reason native results are not byte-equivalent to the original Python 3.11 Docker images.

This backend intentionally records `network_isolation=false` and the absence of PID/mount namespaces. A task declaring `allow_internet=false` is rejected unless `ROY_LHTB_ALLOW_NETWORK_DEGRADED=true` is explicitly set, and such a run is labeled degraded. Native results use a content-derived environment digest and the original final verifier, but they are not Docker-equivalent or official leaderboard-comparable. Reports must keep them separate from the official Docker protocol.

Dev checkpoint selection uses highest mean official environment utility, then fewer tokens and earlier epoch. Final test compares `single_agent_direct`, `roy_runtime_heuristic` and `learned_information_realization` for three repetitions. Direct uses the same model/executor/runtime while enforcing one root node and no communication. Report mean `u_env`, success at `u_env >= 0.95`, paired bootstrap 95% CI, tokens, time, topology, per-node actions and failures. Node-wise `R_t^MIA` is reported separately as a training-credit diagnostic and never thresholded as task success.

See [the implementation note](reports/lhtb-process-reward-implementation.md) for the delivered components, validation boundary and formal-run outputs.

## AFlow benchmark and baseline compatibility

The AFlow comparison checkout is external to Roy and pinned in
[`config/aflow_benchmarks.json`](config/aflow_benchmarks.json). Raw datasets,
generated workflows and evaluation outputs remain outside Git. The official
download bundle contains the six paper benchmarks: HumanEval, MBPP, GSM8K,
MATH, HotpotQA and DROP. Although the current AFlow source also has a
LiveCodeBench evaluator, that dataset is not part of the downloaded paper
bundle and is excluded from the pinned comparison suite.

AFlow names its optimization files `*_validate.jsonl`; it does not provide a
separate train split in this bundle. For a leakage-safe comparison, Roy and
AFlow both treat those records as the optimization/training set. The
`*_test.jsonl` files are read only after the Roy actor or AFlow workflow is
frozen. HumanEval and MBPP public tests may be exposed to the workflow during
optimization, but their final tests remain evaluator-only.

```bash
# Clone next to Roy and pin the exact reviewed source revision.
cd ~/rivermind-data/benchmarks
git clone https://github.com/FoundationAgents/AFlow.git
git -C AFlow checkout --detach 3f457218fc716093fe53f6df8a5d5e6379d66346

# AFlow's documented environment is Python 3.9.
cd AFlow
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c 'from data.download_data import download; download(["datasets"])'

# Configure the optimizer and executor in the ignored config/config2.yaml,
# then optimize only on the AFlow validation split.
.venv/bin/python run.py --dataset GSM8K --sample 4 \
  --optimized_path workspace --initial_round 1 --max_rounds 20 \
  --validation_rounds 5 --opt_model_name MODEL --exec_model_name MODEL
```

Roy also provides a fail-closed preparation/audit wrapper. On GPUHome, where
the managed Python 3.9 download may be unavailable, Python 3.12 has passed the
four non-code scorer smoke tests and can be selected explicitly. This is a
host compatibility mode, not the upstream documented interpreter.

```bash
research/remote/prepare_aflow.sh check

AFLOW_VENV="$HOME/rivermind-data/benchmarks/AFlow/.venv312" \
AFLOW_PYTHON_SPEC=/usr/bin/python3.12 \
research/remote/prepare_aflow.sh env

AFLOW_VENV="$HOME/rivermind-data/benchmarks/AFlow/.venv312" \
research/remote/prepare_aflow.sh smoke
```

The upstream method uses an LLM-guided Monte Carlo-tree-search variant over
Python-represented workflows. It selects previously scored workflow rounds,
asks the optimizer model to mutate their graph/prompt using the dataset's
allowed operators, evaluates each mutation on the validation data, and keeps
the score history for later selection. This is an external `AFlow` baseline;
its search decisions and generated workflow code are not imported as Roy
GRPO labels and MCTS is not added to Roy training or inference.

For a controlled comparison, use the same execution model, temperature,
dataset records and scorer for `single_agent_direct`, frozen AFlow and frozen
Roy. Report the upstream score, model calls/tokens and wall time. AFlow's
bundled cost table does not price DeepSeek, so token counts are the comparable
cost measure unless a pinned price table is supplied. Its HumanEval and MBPP
evaluators execute generated Python with `exec`; run these two benchmarks in
Roy's reviewed PRoot isolation rather than in the host research process.

### Training-free variable-dimensional MAS (MATH/HumanEval V1)

The `training-free` branch also contains a separate inference-only Roy path in
`roy_research.training_free`. It starts with `A0`, lets the Worker propose a
dependency graph, uses the Global Semantic Searcher to retain at most three
dependency-closed subgraphs, and compares every realized expansion against an
optimized no-expansion baseline. The inner search evaluates complete weighted
A2A matrices through semantic channelization and a frozen posterior probe; it
does not add edge scores.

Candidate proposal and candidate realization are deliberately different calls.
The proposal is a cheap semantic direction. Only after global selection does an
independently configurable external candidate model perform the high-compute
`XRealizer` call. That call must explicitly configure the complete
`X_i=(Q_i,R_i,C_i,M_i,T_i,Z_i,Sigma_i)` for every candidate. Incomplete states,
unknown tools, shared-memory namespaces, invalid dependencies, and wrong agent
ids fail closed. The same realized `X` is cached and reused across all matrix
rollouts, so matrix search never silently redesigns the candidate agents.

Every realized state runs through the same versioned single-Agent harness. Its
immutable contract is `(agent_id, parent_id, Q/objective, R/role, T/tools,
expected_output, stop_condition, static-context hash, memory namespace and
inherited-memory references)`; only received messages, private-memory entries,
`Z/result`, and `Sigma/status` evolve during execution. The
harness supplies a bounded execution view, deterministic private-memory
retrieval, deduplicated bounded inbound messages, capability-filtered tool
schemas and calls, model-update application, cloning, and versioned snapshots.
Contract mutation fails closed through a fingerprint check.

Private memory is namespaced exactly as `memory/<agent_id>` and is never placed
in global-search, candidate-realization, or peer-summary payloads. A peer sees
only the source's public epistemic result. The Channelizer is the sole boundary
allowed to retrieve task-relevant entries from the source's private memory; it
compresses those entries into a receiver-specific message, and only that
message enters the receiver context. Tool membership is also part of the
immutable contract: Candidate X must choose a subset of the task registry, the
model sees only that subset's schemas, and out-of-capability calls are rejected
before registry execution. Each result records the harness schema and limits.

The runtime is path-dependent rather than a sequence of independent graph
searches. Its committed state is `S_t=(X_t,W_t,D_t,H_t)`: private Agent state,
the current information matrix, dependency/derivation state, and an append-only
event ledger. Every Parent proposal receives its current matrix, peer summaries,
private memory, active dependencies, committed history, and a separately marked
counterfactual search history. Posterior probes receive committed history only;
provisional candidate events cannot leak into `q_t`.

For no expansion, matrix search starts at `W_t`; for expansion it starts at
`[[W_t,B_t],[C_t,D_t]]`. A coordinate neighbor changes one edge by only one
adjacent configured weight level, so repeated beam iterations optimize a
reachable `Delta W_t` instead of rebuilding an unrelated graph. A positive
no-expansion frontier is committed as a `reorganize` transition and the system
continues; it stops only when both reorganization and admissible expansion fail
to exceed the conditional-information threshold. Selected dependency subgraphs
of up to two nodes provide the V1 short-horizon lookahead.

Every output now includes schema-v2 `checkpoints`, `event_ledger`,
`dependency_ledger`, `matrix_trajectory`, `agent_basis_trajectory`, cumulative
conditional information gain, per-round initial/final/delta matrices, and
topology drift (`Delta N`, edge L1/Frobenius drift, and edge additions/removals).

The checked V1 budget is in [`config/training_free_v1.json`](config/training_free_v1.json).
For a small MATH validation run:

```bash
PYTHONPATH=research python3 -m roy_research training-free-run \
  --aflow-root /path/to/AFlow \
  --benchmark MATH --split optimization --limit 2 \
  --worker-model deepseek-v4-flash \
  --candidate-model YOUR_REASONING_MODEL \
  --ledger research/output/training-free/token-ledger.json \
  --events research/output/training-free/events.jsonl \
  --output research/output/training-free/math.jsonl
```

Add `--score --aflow-python /path/to/AFlow/.venv/bin/python` to invoke the pinned
AFlow MATH scorer. HumanEval loads only its public tests into agent context and
keeps hidden tests in the evaluator payload. Its scorer refuses to run unless
`--human-eval-sandbox-command` supplies a reviewed isolation prefix.

Agents can make bounded structured tool requests during Worker and provisional
execution. `symbolic_math` uses an AST allowlist before calling SymPy and is
available by default. `python` and `public_tests` appear in the model's tool
catalog only when a code sandbox is configured. On macOS, add
`--macos-readonly-tool-sandbox`; other hosts should pass a reviewed PRoot,
container, or equivalent prefix with `--tool-sandbox-command`. The public-test
tool receives only tests already present in that Agent's `C_i`, never hidden
evaluator tests. Tool calls, arguments, results, failures, latency, and sandbox
backend are written under `tool_audit` in every result record.

## τ³ transfer compatibility (not primary training)

The commands below preserve the earlier τ³ adapter and its historical exploration-envelope behavior for regression and transfer work. Its envelope is separate from LHTB's direct node-actor rollout; neither is part of task utility or a theoretical structural objective, and τ³ must not be reported as the primary training method.

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

The Runtime scheduler deterministically supplies one context node; it is observed, not a routing action. `DERIVE_INFO` or `DERIVE_ORG` transfers the next decision to the new child, so child-local work or another child-local derivation is explicit; `RETURN` transfers control to the parent. The exact masked probability is recorded for the categorical structural action. The shared actor combines relational message passing with attention/mean/max pooling, then scores the six generic action tokens against the full graph and acting node context. Collection directly samples and executes one real action without MCTS or value backup. Training and inference use the identical interface.

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
