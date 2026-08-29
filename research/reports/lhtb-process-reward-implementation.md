# Roy × LHTB direct node-actor GRPO implementation note

Date: 2026-08-24

Branch: `exp`

Benchmark revision: `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`

## Implemented protocol

The primary training protocol now uses the fixed 30 train / 8 dev / 8 test LHTB manifest. Formal training contains 120 current-policy groups and 960 rollouts. A group is accepted only when all eight records share task checksum, immutable environment digest, runtime configuration, non-empty initial fingerprint and actor revision while using distinct organization seeds.

Roy's TypeScript process owns the six-action shared recursive Controller, the lower-level Runtime execution primitives, DAG state, immutable process-state chain and frozen DeepSeek Worker. Harbor interaction is a thin Python adapter over persistent JSON-RPC and `BaseEnvironment.exec`. Terminal results include command, cwd, timeout, exit code, output, duration and a before/after file inventory. Partial snapshots are atomically saved after every protocol round trip. The six-hour native training envelope uses one absolute deadline across the initial call and every same-conversation verifier continuation, reserving ten minutes to finalize any pending command as a partial-side-effect event and run the official final verifier instead of emitting an unscored `AgentTimeoutError`.

Semantic state construction uses two frozen DeepSeek prompts. The extractor returns typed entities; the verifier returns `entail / contradict / unknown` probabilities. Pinned MiniLM retrieves at most eight same-type candidate pairs and cannot label a relation. Runtime events known to the program bypass semantic inference. Requests, responses, provenance, cache records and proposer audits are append-only.

The raw process ledger remains complete. At every organization decision, policy input explicitly contains the acting node's local objective, lineage, status, requirements, specification, tool access, termination condition and recent local events, plus the current bounded relational projection of global `M_t`. The same actor parameters are invoked separately for root, child and deeper nodes. Policy records include the exact node context, raw and legally masked probabilities, selected action and exact old log-probability.

Every formal rollout directly samples one of `CONTINUE / DERIVE_INFO / DERIVE_ORG / PRUNE / RETURN / FINISH` from a hard Runtime legality mask without MCTS, PUCT, value rollout or hypothetical-state expansion. Frozen-Worker preference cannot alter that support. After sampling, the Worker receives the selected category and produces its semantic payload; it receives no node range, minimum depth or topology label. Legal root `FINISH` permits single-node trajectories. Repeated real decisions yield wider branching, recursive sub-sub derivation and communication/dependency DAGs one graph edit at a time. The actor receives one generic token per legal action and never scores semantic payloads; commands, child descriptions, reports, reuse connections and prune targets remain Worker outputs. Explicit profile overrides remain diagnostic-only and are rejected by the trainer.

The importer saves the complete adjacent state-transition chain with before/after topology, exact deltas and official terminal-score provenance. These records are appended to `transition-audit-samples.jsonl` for structural audit, but no adjacent transition is assigned a synthetic reward.

The only reward is official final LHTB `R`. GRPO standardizes the eight scores in a matched group, assigns each trajectory-level advantage to all node decisions in that trajectory, and uses exact old-policy ratios with trajectory-length normalization. A zero-variance group is preserved but performs no optimizer step.

## Formal execution and outputs

`prepare_lhtb.sh` checks x86_64, 16 CPUs, 64 GB RAM, at least 200 GB disk, 15% free space, Docker/buildx, Node 22 and Python 3.12. It pins LHTB and bundled Harbor, downloads the pinned MiniLM revision and runs the official oracle smoke. It never prunes Docker automatically.

`run_lhtb_training.sh` samples and immediately updates each group, runs all eight dev tasks after every epoch, checkpoints every epoch and applies the locked dev selection rule. `run_lhtb_test.sh` restores official task timeouts and compares true single-agent direct, Roy heuristic and frozen learned Roy with three repeats. The report contains reward, success at 0.95, paired bootstrap intervals, token/time/topology/per-node-action metrics and failure records.

Generated trajectories, process states, semantic/model audits, Harbor results, checkpoints, Docker data and benchmark assets remain under ignored `research/output/` or the external LHTB checkout.

## Native GPUHome execution backend

The `exp-roy` GPUHome instance cannot launch a nested container runtime and the user chose not to move execution to a cloud API. A non-official `NativeProcessEnvironment` therefore implements Harbor's environment interface with a dedicated unprivileged UID, per-trial copied workspace, PRoot path virtualization, process-group cleanup, command timeouts and immutable task/template digests. Task commands receive a scrubbed environment and never inherit the DeepSeek credential.

Native task conversion is fail-closed. `lhtb-native-audit` checks all pinned split records against `lhtb_native_provisioning.json`; provisioning refuses unknown tasks, stale LHTB revisions and changed task checksums. The catalog now enumerates all 46 tasks, while server-side conversion and oracle validation remain incomplete. OCI tasks separately pin the source registry digest and record the converted OCI-layout digest; conversion no longer assumes these different representations have the same digest. Formal native training remains unavailable until every selected train/dev/test task and its dependencies pass provisioning and oracle smoke.

This is a portability adaptation, not an isolation-equivalent reimplementation of LHTB Docker images. GPUHome does not permit network namespaces, PID namespaces, mount namespaces or cgroup enforcement. The backend records those missing capabilities, rejects `allow_internet=false` tasks by default, and labels explicit exceptions as degraded. It runs the original final verifier, but native scores must be reported separately and cannot be presented as official leaderboard-comparable results.

## Current evidence boundary

Earlier MCTS/value-shaped native trajectories are permanently diagnostic-only because they use an obsolete behavior policy and interface. The replacement direct actor rejects such records at import/training gates. Historical runs remain useful for identifying inactive actors, missing DERIVE coverage and payload-count bias, but cannot update the current model.

The replacement gate runs one fresh train task with `G=8`. `inactive_actor` is classified as environment-invalid, retained for audit and retried rather than scored as zero. The group is accepted only with eight valid verifier rewards, distinct seeds, one matched initial fingerprint, complete raw `M_0...M_T`, the compact event-driven policy interface, full probability diagnostics and nonzero reward variance. Aggregate input tokens and topology diversity are reported diagnostics rather than legality filters, avoiding post-selection of favorable topology outcomes. Only a group passing this gate can trigger the first GRPO update. No formal learned-policy performance claim is made before that update and subsequent held-out evaluation.

The implementation does not claim an intermediate Harbor reward. `M_t -> M_t+1` is recorded for audit and actor input only; official terminal `R` is the sole GRPO reward. DeepSeek's execution, extraction and verification channels remain correlated, and the public API model alias does not provide a content-addressed weight revision; both limitations must remain visible in the final paper report.
