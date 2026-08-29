# Roy Information Realization and Autonomous Organization GRPO

Status: canonical research specification for the `exp` branch (2026-08-29).

This document is the normative bridge from MIA to Roy's recursive organization policy. It replaces the earlier staged structural-learning formulation. The implementation must not introduce predefined agent roles, teacher systems, imitation learning, or a weighted sum of objectives.

## 1. Information-realization objective

Let `Y` denote the task-relevant latent answer, `C_t` the information available to the organization at time `t`, and `Y_hat_t` its current output. Roy separates three factors:

```text
A_t  available task-relevant information,
S_t  fidelity of the distributed representation of that information,
G_t  conversion of represented information into a useful output.
```

The operational target is terminal task utility, used as the benchmark estimator of information realization:

```text
J(tau) = U_T(Y, Y_hat_terminal)

maximize_pi  E[tau ~ pi][J(tau)]
```

Agent count, depth, latency, token use, tool calls, topology complexity, communication, rationality and redundancy are never added to `J`. Roy does not impose a learned node-count, depth, or total-token preference. Benchmark timeouts, unavailable actions, Docker isolation and provider safety controls remain operational execution conditions, not theoretical reward terms or artificial structural ceilings.

MIA supplies the information-processing interpretation and its data-processing/information-preservation bounds; it does not license an independently scaled proxy reward. Adjacent changes in acquisition, representation, uncertainty, coverage or conversion may be logged and estimated for mechanism analysis, but an estimated `Delta Phi` is neither added to the LHTB score nor substituted for it in the main GRPO objective. This avoids the dimensionally invalid weighted objective rejected by the original derivation. Any future information estimator is evaluated by whether it predicts or explains terminal utility, not treated as ground-truth information creation.

## 2. State and information flow

The organization state contains:

- the current task and dynamic context;
- a task-derivation tree with immutable parent lineage;
- a separate dependency DAG;
- a separate communication graph;
- claims, evidence, external observations, assumptions, conflicts and residual requirements;
- node-local epistemic reports, coverage, uncertainty and blind spots;
- model, environment and randomness metadata needed to replay a trajectory;
- realized resource measurements, official benchmark timeout and provider provenance.

Derivation and communication have different meanings. A derivation edge records why a child exists. A communication edge records which nodes may exchange information. A dependency edge records which artifact or claim must be produced before another node can proceed. Combining these edge types into a single tree is invalid.

External observations must be stored separately from model inference. Tool, API, database, KB, code and environment results require provenance and their claim-support links. This preserves the distinction between information acquisition and reasoning over already available information.

## 3. Semantic--structural--execution separation

Roy has three non-overlapping layers:

```text
frozen Worker pi_W: semantic reasoning, tools and payload construction;
shared Controller pi_theta: one categorical structural decision for the current node;
Runtime: deterministic execution, routing, lifecycle and audit.
```

The learned Controller vocabulary is fixed:

```text
CONTINUE, DERIVE_INFO, DERIVE_ORG, PRUNE, RETURN, FINISH
```

It contains no natural-language action and no node-routing action. `CONTINUE` means that the current Worker continues its semantic work. `DERIVE_INFO` asks the current Worker to specify one child for external information acquisition. `DERIVE_ORG` asks it to specify one child for organization, comparison, verification or integration of information already represented in Roy. `PRUNE` asks the Worker to identify a legal low-value branch. `RETURN` ends the current non-root node and propagates its Worker-authored report to its parent. `FINISH` is root-only and submits the task to the official verifier.

The Runtime may record lower-level execution primitives `DERIVE`, `ACQUIRE`, `CONNECT`, `EXECUTE`, `RETURN`, `PRUNE` and `STOP`. Those primitives are not the learned action vocabulary. In particular, concrete terminal commands, tool access, report text, connection/reuse endpoints, child prompts and prune targets are frozen-Worker semantic payloads. They never become categorical Controller candidates and their text is not encoded by the actor.

The grammar does not define roles. For either derivation action, the frozen Worker emits an open agent specification tied to one residual requirement. The learned claim is whether and which information-realization mode to derive, not RL generation or selection of arbitrary child descriptions. A valid child:

- is a strict, narrower refinement of the parent objective;
- identifies the parent gap that caused its creation;
- states the new information or transformation required;
- receives only relevant claims, evidence and reports;
- states permitted external access without receiving a hidden answer;
- has a verifiable output contract and termination condition;
- is not a duplicate of an active or completed node.

Every node returns a structured epistemic report containing concise reasoning summaries rather than hidden chain-of-thought. The report includes claims, evidence, external observations, assumptions, uncertainty, conflicts, coverage, blind spots, residual requirements, optional open child proposals, whether the parent gap was resolved, and information to propagate.

Worker-selected `ACQUIRE` performs admissible external information access, `CONNECT` adds or reuses a legal communication route, and `EXECUTE` changes or verifies the task environment. They realize a selected `CONTINUE` or derivation payload but are not optimized as separate actor labels. Runtime `PRUNE` removes a Worker-selected node only when no unresolved dependency requires it. Runtime `STOP` realizes Controller `FINISH` and is illegal while required dependencies remain unresolved.

## 4. One autonomous policy

Roy learns one organization policy shared by every node. A deterministic dependency/event-locality scheduler first supplies the current execution owner `n_t`; this observed context is not an RL action. The policy makes one masked categorical choice:

```text
a_i,t ~ pi_theta(a | M_t, L_i,t, H_i)
```

Here `M_t` is the current global epistemic/organization graph, `L_i,t` is the exact current node context and `H_i` is its ancestry. The six action logits are independent of how many semantic payloads the Worker could construct. The actor receives one generic token for each legal action kind and never receives candidate descriptions or payloads. After the categorical choice, the frozen Worker supplies any required open payload. No fixed role ID, fixed child catalog, trainable payload selector, teacher trajectory, or teacher score is part of the model.

The policy encoder is a typed relational network over derivation, dependency, communication, tool-use, evidence and return edges. Learned-attention, mean and max pooling retain complementary full-graph summaries, while the relationally updated scheduler-supplied context-node embedding preserves the current agent's local task, progress and ancestry. Other nodes remain visible as global structure, but the actor never routes execution to one of them. `DERIVE_INFO` or `DERIVE_ORG` creates exactly one Worker-specified child and transfers the following decision to that child, which permits genuine child-to-grandchild recursion. `RETURN` transfers ownership back to the parent. Training replay reconstructs the exact masked six-way probability.

Environment and graph validity are enforced before sampling by the action mask:

```text
A_legal(s) = {a in A(s): graph, dependency and environment preconditions hold}
pi_theta(a | s) = softmax(mask(logits_theta(s), A_legal(s))).
```

The same mask semantics are reconstructed during replay. Runtime usage is absent from terminal utility, process credit, advantage and the GRPO loss. The implementation does not assign theoretical costs such as `DERIVE = 2`.

Formal sampling does not assign a topology profile, minimum node count or minimum depth, and does not perform MCTS or another look-ahead search. Whenever a node owns execution, the current policy directly samples one legal six-way structural action for that node. Repeated real decisions make node count, derivation depth and connectivity trajectory outcomes. `FINISH` remains in support whenever required report dependencies permit it, including at the root-only initial state, so both early single-agent termination and deeper recursive organizations are ordinary on-policy outcomes. Explicit topology profiles remain available only for controlled diagnostics and cannot enter formal actor updates.

## 5. LHTB state and semantic construction

LHTB is the primary training environment. After every organization decision and every terminal/tool result Roy appends an immutable global epistemic state `M_t`. It contains requirements, claims, assumptions, evidence, external observations, semantic relations, blind spots, dependencies, organization nodes, DAG edges, active subtree, commands, exit codes, file changes, failures, tokens and wall time. Programmatically known events are recorded directly.

A frozen DeepSeek extractor creates typed entities. A separate frozen DeepSeek semantic verifier assigns `entail`, `contradict`, or `unknown` probabilities. Frozen MiniLM embeddings retrieve at most eight candidate pairs per entity type, but similarity cannot create or override a semantic relation. Benchmark keyword fields, lexical rules, regular expressions and word-frequency thresholds are excluded from state construction and decisions. Every extraction, candidate source, verification request, response, model revision and cache key is retained.

Coverage, assumption closure, conflicts and blind spots are structural inputs to the policy. They are not reward terms.

The immutable ledger and the policy input are intentionally distinct. The ledger retains every event and full text needed for audit or later retrieval. The proposer and actor receive a bounded working projection containing active structure, unresolved gaps, selected epistemic entities, recent event deltas and immutable references back to the raw ledger. This projection is not a lossy replacement for `M_t`. Organization-policy decisions are event-driven: they occur initially and when a real state change exposes a gap, failure, file mutation, contradiction, completed node, structural transition or a bounded accumulation of terminal results. Local execution may continue between organization decisions.

## 6. On-policy groups and terminal-reward GRPO

Training samples fresh complete trajectories from the current masked policy. Each LHTB task and epoch produces `G=8` trajectories from eight fresh matched execution environments with the same task checksum, immutable environment digest, initial-state fingerprint, environment configuration and actor revision. Only organization sampling seeds differ. The official protocol realizes these environments as fresh Docker containers. A native portability backend may be used only as a separately labeled experimental condition and does not imply Docker-equivalent isolation. Every decision stores its exact masked old-policy joint log-probability.

The sole reward is the official LHTB final score `R_i` in `[0,1]`. For a matched group, GRPO computes one trajectory-level normalized advantage:

```text
A_i = (R_i - mean_j R_j) / (std_j R_j + epsilon).
```

Every sampled node decision in trajectory `i` receives the same `A_i`; the loss normalizes by that trajectory's number of organization decisions:

```text
L_actor = -(1/G) sum_i (1/T_i) sum_t
  min(rho_i,t A_i, clip(rho_i,t, 1-epsilon, 1+epsilon) A_i),
rho_i,t = pi_theta(a_i,t | M_i,t, n_i,t) / pi_old(a_i,t | M_i,t, n_i,t).
```

The exact masked old-policy probability is saved at collection time and reconstructed during replay. There is no learned value model, EMA target, search backup, intermediate shaped reward, topology bonus or synthetic per-step `R`. When all eight official scores are equal, the group has no preference signal and the actor update is skipped.

The immutable dataset still retains every adjacent `M_t -> M_t+1` transition and SMDP decision span for audit. Each record contains fingerprints, the acting node, node-local context, legal candidates, selected action and exact topology delta, but these fields are not rewards. The full raw runtime and semantic ledgers are stored once; each fingerprinted `M_t` contains a deterministic bounded relational projection with active requirements, recent typed entities, blind spots, usage and immutable ledger references. Node count and topology complexity do not enter `R_i` or the advantage.

Collection is ordinary current-policy rollout. The frozen Worker observes the real current state and acting node and materializes valid semantic payloads only to establish legality and execute the selected category. The actor receives a payload-free six-action mask and samples directly from it. Only the selected real action is executed; hypothetical sibling states are neither rolled out nor assigned rewards. Different organization seeds across the matched group provide exploration, while later groups are sampled from the newly updated actor.

There is no teacher, imitation, predefined role pool, staged objective, entropy bonus, cost penalty or weighted reward sum.

## 7. LHTB protocol and evaluation

The benchmark is `zli12321/LHTB` pinned with its bundled Harbor to commit `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`. Its 46 tasks follow the pinned README's eight-category taxonomy. Within each category, fixed SHA-256 ordering selects one dev task and one test task; the remaining tasks form the 30-task train split. The checked manifest is `30 train / 8 dev / 8 test`, and the trainer rejects dev or test task IDs.

Formal training uses four epochs, eight trajectories per train task and 960 rollouts total. Each training rollout may run for up to six hours, with concurrency four and a DeepSeek response ceiling of 32,768 tokens. These execution settings do not add structural reward or force an agent count. At each epoch boundary, learned Roy runs once on every dev task. Checkpoint selection maximizes dev mean reward, breaking ties by fewer tokens and earlier epoch.

The selected checkpoint is tested once, with three repetitions, against:

- `single_agent_direct`: the same DeepSeek and terminal executor in the same runtime, restricted to the root node with no derivation or communication;
- `roy_runtime_heuristic`: the same recursive runtime controlled by the compatibility heuristic;
- `learned_information_realization`: the learned six-action shared recursive Controller.

Report mean reward, success rate at `R >= 0.95`, paired bootstrap 95% confidence intervals, tokens, time, nodes, DAG structure, waits, communication edges, action distributions and failures. An interval crossing zero is reported as inconclusive. After freezing the selected model, τ³ is used only for smoke and held-out zero-shot transfer, with no benchmark-specific update.

## 8. Training and recovery invariants

- Every actor update consumes exactly eight complete current-policy train trajectories.
- Runtime crashes, environment-invalid attempts, environment failures and incomplete trajectories are preserved for audit but excluded from actor updates. Environment-invalid attempts are resampled rather than assigned reward zero.
- A normal six-hour training deadline triggers the official verifier; its partial score is a valid terminal label.
- Every trajectory preserves `M_0...M_T`, runtime events, semantic audits, raw and masked action probabilities, selected actions, exact old log-probabilities, real-gap diagnostics, task checksum, source and converted environment digests, backend capabilities, model revisions and final Harbor result.
- Actor, its optimizer and updated group IDs are restored together; a group ID can be optimized only once.
- Dev selects checkpoints but never updates them. Test runs only after selection and never updates weights.
- Neither the local Docker protocol nor the native portability backend invents a timed process reward. Adjacent state transitions are audit records; the official final verifier score is the only GRPO reward.

## 9. Limits

MIA's data-processing bound still applies: communication and recursion cannot create information unavailable from context, tools or environment. DeepSeek participates in execution, extraction and semantic verification, creating correlated errors despite separate frozen prompts and provenance. Utility judges can be biased, external environments can drift, tool side effects may be non-clonable, API sampling is not guaranteed deterministic, and snapshot replay may be incomplete. Claims therefore concern measured LHTB performance under the pinned protocol, not global optimality or universal information gain.
