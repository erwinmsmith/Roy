# Roy × LHTB process-credit implementation note

Date: 2026-08-20

Branch: `exp`

Benchmark revision: `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`

## Implemented protocol

The primary training protocol now uses the fixed 30 train / 8 dev / 8 test LHTB manifest. Formal training contains 120 current-policy groups and 960 rollouts. A group is accepted only when all eight records share task checksum, immutable environment digest, runtime configuration, non-empty initial fingerprint and actor revision while using distinct organization seeds.

Roy's TypeScript process owns the seven-action recursive runtime, DAG state, immutable process-state chain and DeepSeek candidate proposer. Harbor interaction is a thin Python adapter over persistent JSON-RPC and `BaseEnvironment.exec`. Terminal results include command, cwd, timeout, exit code, output, duration and a before/after file inventory. Partial snapshots are atomically saved after every protocol round trip.

Semantic state construction uses two frozen DeepSeek prompts. The extractor returns typed entities; the verifier returns `entail / contradict / unknown` probabilities. Pinned MiniLM retrieves at most eight same-type candidate pairs and cannot label a relation. Runtime events known to the program bypass semantic inference. Requests, responses, provenance, cache records and proposer audits are append-only.

The only environment target is official final LHTB reward. The independent relational critic is trained with equal trajectory weight. Frozen EMA potential differences give decision credit and telescope to the final reward residual. The update order is actor, value, EMA. Initial constant value recovers terminal-GRPO ordering; a zero-variance terminal group still updates the critic, and the actor updates only when shaped returns vary.

## Formal execution and outputs

`prepare_lhtb.sh` checks x86_64, 16 CPUs, 64 GB RAM, at least 200 GB disk, 15% free space, Docker/buildx, Node 22 and Python 3.12. It pins LHTB and bundled Harbor, downloads the pinned MiniLM revision and runs the official oracle smoke. It never prunes Docker automatically.

`run_lhtb_training.sh` samples and immediately updates each group, runs all eight dev tasks after every epoch, checkpoints every epoch and applies the locked dev selection rule. `run_lhtb_test.sh` restores official task timeouts and compares true single-agent direct, Roy heuristic and frozen learned Roy with three repeats. The report contains reward, success at 0.95, paired bootstrap intervals, token/time/topology/process-credit/value metrics and failure records.

Generated trajectories, process states, semantic/model audits, Harbor results, value traces, checkpoints, Docker data and benchmark assets remain under ignored `research/output/` or the external LHTB checkout.

## Native GPUHome execution backend

The `exp-roy` GPUHome instance cannot launch a nested container runtime and the user chose not to move execution to a cloud API. A non-official `NativeProcessEnvironment` therefore implements Harbor's environment interface with a dedicated unprivileged UID, per-trial copied workspace, PRoot path virtualization, process-group cleanup, command timeouts and immutable task/template digests. Task commands receive a scrubbed environment and never inherit the DeepSeek credential.

Native task conversion is fail-closed. `lhtb-native-audit` checks all pinned split records against `lhtb_native_provisioning.json`; provisioning refuses unknown tasks, stale LHTB revisions and changed task checksums. The initial reviewed catalog deliberately contains only three smoke tasks. Formal native training remains unavailable until every selected train/dev/test task is reviewed and its dependencies and verifier behavior pass oracle smoke.

This is a portability adaptation, not an isolation-equivalent reimplementation of LHTB Docker images. GPUHome does not permit network namespaces, PID namespaces, mount namespaces or cgroup enforcement. The backend records those missing capabilities, rejects `allow_internet=false` tasks by default, and labels explicit exceptions as degraded. It runs the original final verifier, but native scores must be reported separately and cannot be presented as official leaderboard-comparable results.

## Current evidence boundary

Local validation covers the full TypeScript suite and 53 Python research tests, including fake Harbor, mock DeepSeek and native audit/provisioning checks. No formal LHTB reward is reported here. The official protocol still requires the dedicated `exp-roy-lhtb` Docker VM; the native GPUHome track may be used for separately labeled research runs after its selected task set passes provisioning and oracle smoke. A 24-trajectory Roy smoke is accepted only if all three train-task groups have complete process chains, matched initial state, no benchmark semantic-field leakage and at least one group with continuous final-reward variance.

The implementation does not claim that local `Delta V` is Harbor's official timed intermediate verifier reward. It is a value potential learned solely from the official final score. DeepSeek's execution, extraction and verification channels remain correlated, and the public API model alias does not provide a content-addressed weight revision; both limitations must remain visible in the final paper report.
