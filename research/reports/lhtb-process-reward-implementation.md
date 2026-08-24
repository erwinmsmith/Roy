# Roy × LHTB process-credit implementation note

Date: 2026-08-24

Branch: `exp`

Benchmark revision: `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`

## Implemented protocol

The primary training protocol now uses the fixed 30 train / 8 dev / 8 test LHTB manifest. Formal training contains 120 current-policy groups and 960 rollouts. A group is accepted only when all eight records share task checksum, immutable environment digest, runtime configuration, non-empty initial fingerprint and actor revision while using distinct organization seeds.

Roy's TypeScript process owns the seven-action recursive runtime, DAG state, immutable process-state chain and DeepSeek candidate proposer. Harbor interaction is a thin Python adapter over persistent JSON-RPC and `BaseEnvironment.exec`. Terminal results include command, cwd, timeout, exit code, output, duration and a before/after file inventory. Partial snapshots are atomically saved after every protocol round trip.

Semantic state construction uses two frozen DeepSeek prompts. The extractor returns typed entities; the verifier returns `entail / contradict / unknown` probabilities. Pinned MiniLM retrieves at most eight same-type candidate pairs and cannot label a relation. Runtime events known to the program bypass semantic inference. Requests, responses, provenance, cache records and proposer audits are append-only.

The raw process ledger remains complete. A separate compact working projection bounds proposer/actor context to active structure, unresolved gaps, selected entities, recent deltas and immutable raw-event references. Extraction remains incremental. Organization-policy inference is event-driven rather than attached to every terminal step. Policy records include raw and legally masked probabilities, selected actions, exact old log-probabilities, real-gap counts, child-proposal counts and explicit STOP-mask reasons.

Topology-rich sampling now defaults to a six-node exploration target with a non-binding 6–8-node proposer preference. It changes candidate coverage only: each `DERIVE` still requires an actual parent-local open requirement, no node-count or topology reward is added, and the runtime has no eight-node ceiling. Missing real-gap `DERIVE` or novel `CONNECT` candidates make an attempt `sampling_invalid` and therefore ineligible for training. Semantic requirements inherit the node that produced the source terminal event; semantically redundant requirements are suppressed only when the frozen verifier judges that an existing requirement entails the new one. Producer reports automatically resolve declared `report:<producer>` dependencies and wake waiting consumers.

The only environment target is official final LHTB reward. The independent relational critic is trained with equal trajectory weight. Frozen EMA potential differences give decision credit and telescope to the final reward residual. The update order is actor, value, EMA. Initial constant value recovers terminal-GRPO ordering; a zero-variance terminal group still updates the critic, and the actor updates only when shaped returns vary.

## Formal execution and outputs

`prepare_lhtb.sh` checks x86_64, 16 CPUs, 64 GB RAM, at least 200 GB disk, 15% free space, Docker/buildx, Node 22 and Python 3.12. It pins LHTB and bundled Harbor, downloads the pinned MiniLM revision and runs the official oracle smoke. It never prunes Docker automatically.

`run_lhtb_training.sh` samples and immediately updates each group, runs all eight dev tasks after every epoch, checkpoints every epoch and applies the locked dev selection rule. `run_lhtb_test.sh` restores official task timeouts and compares true single-agent direct, Roy heuristic and frozen learned Roy with three repeats. The report contains reward, success at 0.95, paired bootstrap intervals, token/time/topology/process-credit/value metrics and failure records.

Generated trajectories, process states, semantic/model audits, Harbor results, value traces, checkpoints, Docker data and benchmark assets remain under ignored `research/output/` or the external LHTB checkout.

## Native GPUHome execution backend

The `exp-roy` GPUHome instance cannot launch a nested container runtime and the user chose not to move execution to a cloud API. A non-official `NativeProcessEnvironment` therefore implements Harbor's environment interface with a dedicated unprivileged UID, per-trial copied workspace, PRoot path virtualization, process-group cleanup, command timeouts and immutable task/template digests. Task commands receive a scrubbed environment and never inherit the DeepSeek credential.

Native task conversion is fail-closed. `lhtb-native-audit` checks all pinned split records against `lhtb_native_provisioning.json`; provisioning refuses unknown tasks, stale LHTB revisions and changed task checksums. The catalog now enumerates all 46 tasks, while server-side conversion and oracle validation remain incomplete. OCI tasks separately pin the source registry digest and record the converted OCI-layout digest; conversion no longer assumes these different representations have the same digest. Formal native training remains unavailable until every selected train/dev/test task and its dependencies pass provisioning and oracle smoke.

This is a portability adaptation, not an isolation-equivalent reimplementation of LHTB Docker images. GPUHome does not permit network namespaces, PID namespaces, mount namespaces or cgroup enforcement. The backend records those missing capabilities, rejects `allow_internet=false` tasks by default, and labels explicit exceptions as degraded. It runs the original final verifier, but native scores must be reported separately and cannot be presented as official leaderboard-comparable results.

## Current evidence boundary

Local validation covers 555 TypeScript tests and 63 Python research tests, including fake Harbor, mock DeepSeek and native audit/provisioning checks. The earlier native diagnostic smoke produced seven valid official rewards `[0, 0, 0, 0, 0.2727, 0.2727, 0.2727]` and one `inactive_actor` RPC failure. It demonstrated continuous reward variance, but consumed about 75.2 million input tokens and produced no optimizer step; those seven trajectories are permanently diagnostic-only because they use the obsolete policy interface. A later shallow-topology diagnostic was stopped after confirming that 34–47 real open requirements were present while the proposer exposed no continuing `DERIVE` candidates; its partial trajectories remain saved and are also excluded from training.

The replacement gate runs one fresh train task with `G=8`. `inactive_actor` is classified as environment-invalid, retained for audit and retried rather than scored as zero. The group is accepted only with eight valid verifier rewards, distinct seeds, one matched initial fingerprint, complete raw `M_0...M_T`, the compact event-driven policy interface, full probability diagnostics, nonzero reward variance and at most 15 million aggregate input tokens. Only a group passing this gate can trigger the first GRPO update. No formal learned-policy performance claim is made before that update and subsequent held-out evaluation.

The implementation does not claim that local `Delta V` is Harbor's official timed intermediate verifier reward. It is a value potential learned solely from the official final score. DeepSeek's execution, extraction and verification channels remain correlated, and the public API model alias does not provide a content-addressed weight revision; both limitations must remain visible in the final paper report.
