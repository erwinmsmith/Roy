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

## 3. Fixed grammar, open agent space

The organization grammar is fixed:

```text
DERIVE, ACQUIRE, CONNECT, EXECUTE, RETURN, PRUNE, STOP
```

The grammar does not define roles. `DERIVE` takes an open agent specification tied to one residual requirement. In the current implementation a frozen LLM proposer emits those specifications inside the node report, while the trainable organization policy selects among the current open proposals. Therefore the learned claim is proposal selection and organization control, not RL generation of arbitrary specifications. A valid child:

- is a strict, narrower refinement of the parent objective;
- identifies the parent gap that caused its creation;
- states the new information or transformation required;
- receives only relevant claims, evidence and reports;
- states permitted external access without receiving a hidden answer;
- has a verifiable output contract and termination condition;
- is not a duplicate of an active or completed node.

Every node returns a structured epistemic report containing concise reasoning summaries rather than hidden chain-of-thought. The report includes claims, evidence, external observations, assumptions, uncertainty, conflicts, coverage, blind spots, residual requirements, optional open child proposals, whether the parent gap was resolved, and information to propagate.

`ACQUIRE` performs admissible external information access. `CONNECT` adds a legal communication route. `EXECUTE` updates a node's report. `RETURN` propagates a report to the parent. `PRUNE` removes a node only when no unresolved dependency requires it. `STOP` is a root action and is illegal while required dependencies remain unresolved.

## 4. One autonomous policy

Roy learns one organization policy. A deterministic dependency/event-locality scheduler first supplies the current execution owner `n_t`; this observed context is not an RL action. The policy selects an outer grammar action and then an open payload conditioned on that action:

```text
pi_theta(a, z | s, n_t)
  = pi_action(a | s, n_t)
    pi_payload(z | s, n_t, a).
```

Here `a` is one grammar action and `z` is its open payload, such as a proposed child specification, acquisition requirement, connection, report, or prune target. A `DERIVE` payload explicitly identifies whether the new child realizes information by `acquire_external` or `organize_knowledge`; the specification remains open rather than selecting from a role catalog. Outer action mass is computed independently of the number of payload candidates; multiple DERIVE specifications therefore compete only inside `pi_payload` and cannot multiply DERIVE's outer probability. Candidate count and node count are variable. No fixed role ID, fixed child catalog, teacher trajectory, or teacher score is part of the model.

The policy encoder is a typed relational network over derivation, dependency, communication, tool-use, evidence and return edges. Learned-attention, mean and max pooling retain complementary full-graph summaries, while the relationally updated scheduler-supplied context-node embedding preserves the current agent's local position and task. The actor scores only legal open candidates owned by that node against both representations. Other nodes remain visible for dependency, reuse and connection payloads, but the actor never routes execution to them. `DERIVE` deterministically transfers the following decision to the child, which permits an observed child-local action or recursive child-to-grandchild derivation; `RETURN` transfers it back to the parent. After a child tool result, a parent that still owns an open requirement receives a deterministic reuse/connection turn. Training replay reconstructs every such conditional probability exactly.

Environment and graph validity are enforced before sampling by the action mask:

```text
A_legal(s) = {a in A(s): graph, dependency and environment preconditions hold}
pi_theta(a | s) = softmax(mask(logits_theta(s), A_legal(s))).
```

The same mask semantics are reconstructed during replay. Runtime usage is absent from terminal utility, process credit, advantage and the GRPO loss. The implementation does not assign theoretical costs such as `DERIVE = 2`.

Formal sampling does not assign a topology profile, minimum node count or minimum depth. MCTS compares legal `DERIVE`, `CONNECT`, local work, return, prune and `STOP` candidates from the current epistemic state; node count, derivation depth and connectivity are therefore trajectory outcomes. `STOP` remains in support whenever required report dependencies permit it, including at the root-only initial state, so both early single-agent termination and deeper recursive organizations are genuine search outcomes. Explicit topology profiles remain available only for controlled diagnostics and cannot enter formal actor updates.

## 5. LHTB state and semantic construction

LHTB is the primary training environment. After every organization decision and every terminal/tool result Roy appends an immutable global epistemic state `M_t`. It contains requirements, claims, assumptions, evidence, external observations, semantic relations, blind spots, dependencies, organization nodes, DAG edges, active subtree, commands, exit codes, file changes, failures, tokens and wall time. Programmatically known events are recorded directly.

A frozen DeepSeek extractor creates typed entities. A separate frozen DeepSeek semantic verifier assigns `entail`, `contradict`, or `unknown` probabilities. Frozen MiniLM embeddings retrieve at most eight candidate pairs per entity type, but similarity cannot create or override a semantic relation. Benchmark keyword fields, lexical rules, regular expressions and word-frequency thresholds are excluded from state construction and decisions. Every extraction, candidate source, verification request, response, model revision and cache key is retained.

Coverage, assumption closure, conflicts and blind spots are structural inputs to the policy. They are not reward terms.

The immutable ledger and the policy input are intentionally distinct. The ledger retains every event and full text needed for audit or later retrieval. The proposer and actor receive a bounded working projection containing active structure, unresolved gaps, selected epistemic entities, recent event deltas and immutable references back to the raw ledger. This projection is not a lossy replacement for `M_t`. Organization-policy decisions are event-driven: they occur initially and when a real state change exposes a gap, failure, file mutation, contradiction, completed node, structural transition or a bounded accumulation of terminal results. Local execution may continue between organization decisions.

## 6. On-policy groups and continuous process credit

Training samples fresh complete trajectories from the current masked policy. Each LHTB task and epoch produces `G=8` trajectories from eight fresh matched execution environments with the same task checksum, immutable environment digest, initial-state fingerprint, environment configuration and actor revision. Only organization sampling seeds differ. The official protocol realizes these environments as fresh Docker containers. A native portability backend may be used only as a separately labeled experimental condition and does not imply Docker-equivalent isolation. Every decision stores its exact masked old-policy joint log-probability.

The sole environment target is the official LHTB final score `R_i` in `[0,1]`. An independent relational value model `V_psi(M_t)` shares only the frozen MiniLM encoder with the actor; it shares no trainable parameters. It is fitted with equal trajectory weight:

```text
L_V = (1/G) sum_i (1/T_i) sum_t Huber(V_psi(M_i,t), R_i).
```

An EMA target network `V_bar` is frozen while a group is sampled and updated. Its potential difference gives process credit:

```text
r_proc(i,t) = V_bar(M_i,t+1) - V_bar(M_i,t)       for non-final decisions
r_proc(i,T-1) = R_i - V_bar(M_i,T-1)             for the final decision
G_i,t = R_i - V_bar(M_i,t).
```

Thus the process rewards telescope exactly to `R_i - V_bar(M_i,0)` and do not introduce another objective. Across all decisions in the group, mean and variance use step weight `1/T_i`; every trajectory therefore has total statistical weight one. The resulting single advantage enters the length-normalized clipped surrogate with exact old-policy ratios. Update order is actor, value, then EMA. The value head starts at constant `0.5`, so the first group preserves terminal-GRPO ordering. When final scores are equal, the value model still learns; the actor updates only if shaped returns have variance.

The immutable dataset retains both adjacent event transitions and SMDP decision spans. Each adjacent sample contains the exact `M_t` and `M_t+1` fingerprints, one-step topology delta, target values and signed process reward. Decision credit is the telescoping sum over all adjacent transitions until the next organization decision. The full raw runtime and semantic audit ledgers are stored once; each fingerprinted `M_t` contains a deterministic bounded relational projection with active requirements, recent typed entities, blind spots, usage and immutable ledger references. This representation prevents quadratic serialization without changing reward. Node count and topology complexity never determine reward sign; only the final-score-trained frozen value potential does.

During collection, PUCT may use the actor as a prior and the same frozen potential as its search score. For an edge from state `M` to a searched leaf `M'`, backup is `V_bar(M') - V_bar(M)`; with discount one this is exactly the telescoping sum of the intervening process rewards. Every expanded sibling edge—not only the ultimately executed edge—is persisted with its deduplicated policy state, context node, conditional structural payload, actor prior, visits, target revision, backed-up utility and signed potential difference. The actually executed edge is later anchored by its official-verifier return; unexecuted edges remain explicitly labelled frozen-value bootstrap and are never represented as official rewards. These saved local counterfactual groups provide positive, negative and zero policy samples for replay. MCTS runs only while collecting data: optimizer updates read the saved samples and never invoke search. Search therefore changes exploration and supplies counterfactual estimates without introducing another task objective. External tool actions are treated as leaves whenever their side effects cannot be cloned; claims about MCTS coverage are restricted to the cloneable organization-state candidate space.

There is no teacher, imitation, predefined role pool, staged objective, entropy bonus, cost penalty or weighted reward sum.

## 7. LHTB protocol and evaluation

The benchmark is `zli12321/LHTB` pinned with its bundled Harbor to commit `84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54`. Its 46 tasks follow the pinned README's eight-category taxonomy. Within each category, fixed SHA-256 ordering selects one dev task and one test task; the remaining tasks form the 30-task train split. The checked manifest is `30 train / 8 dev / 8 test`, and the trainer rejects dev or test task IDs.

Formal training uses four epochs, eight trajectories per train task and 960 rollouts total. Each training rollout may run for up to six hours, with concurrency four and a DeepSeek response ceiling of 32,768 tokens. These execution settings do not add structural reward or force an agent count. At each epoch boundary, learned Roy runs once on every dev task. Checkpoint selection maximizes dev mean reward, breaking ties by lower value MAE, fewer tokens and earlier epoch.

The selected checkpoint is tested once, with three repetitions, against:

- `single_agent_direct`: the same DeepSeek and terminal executor in the same runtime, restricted to the root node with no derivation or communication;
- `roy_runtime_heuristic`: the same recursive runtime controlled by the compatibility heuristic;
- `learned_information_realization`: the learned seven-action policy.

Report mean reward, success rate at `R >= 0.95`, paired bootstrap 95% confidence intervals, tokens, time, nodes, DAG structure, waits, communication edges, process-value traces, value MAE/Spearman and failures. An interval crossing zero is reported as inconclusive. After freezing the selected model, τ³ is used only for smoke and held-out zero-shot transfer, with no benchmark-specific update.

## 8. Training and recovery invariants

- Every actor/value update consumes exactly eight complete current-policy train trajectories.
- Runtime crashes, environment-invalid attempts, environment failures and incomplete trajectories are preserved for audit but excluded from actor and value updates. Environment-invalid attempts are resampled rather than assigned reward zero.
- A normal six-hour training deadline triggers the official verifier; its partial score is a valid terminal label.
- Every trajectory preserves `M_0...M_T`, runtime events, semantic audits, raw and masked action probabilities, selected actions, exact old log-probabilities, real-gap diagnostics, task checksum, source and converted environment digests, backend capabilities, model revisions and final Harbor result.
- Actor, value, EMA, both optimizers and updated group IDs are restored together; a group ID can be optimized only once.
- Dev selects checkpoints but never updates them. Test runs only after selection and never updates weights.
- Neither the local Docker protocol nor the native portability backend claims Harbor's timed process verifier. “Continuous process reward” means final-score-supervised `V_psi` and `Delta V` credit.

## 9. Limits

MIA's data-processing bound still applies: communication and recursion cannot create information unavailable from context, tools or environment. DeepSeek participates in execution, extraction and semantic verification, creating correlated errors despite separate frozen prompts and provenance. Utility judges can be biased, external environments can drift, tool side effects may be non-clonable, API sampling is not guaranteed deterministic, and snapshot replay may be incomplete. Claims therefore concern measured LHTB performance under the pinned protocol, not global optimality or universal information gain.
