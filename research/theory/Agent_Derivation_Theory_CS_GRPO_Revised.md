# Roy Information Realization and Autonomous Organization GRPO

Status: canonical research specification for the `exp` branch (2026-08-18).

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
subject to   C(tau) <= B almost surely.
```

The budget is a runtime feasibility boundary, not a reward component. The implementation masks actions from observable remaining LLM-call, tool-call, node, depth and organization-decision budgets. Actual token use and latency are recorded after execution. Agent count, depth, latency, token use, tool calls, topology complexity, communication, rationality and redundancy are never added to `J`.

The implementation may retain a separate provider safety ledger to prevent accidental API overspend. That operational fail-safe is not the theoretical constraint and must not alter utility or advantage.

## 2. State and information flow

The organization state contains:

- the current task and dynamic context;
- a task-derivation tree with immutable parent lineage;
- a separate dependency DAG;
- a separate communication graph;
- claims, evidence, external observations, assumptions, conflicts and residual requirements;
- node-local epistemic reports, coverage, uncertainty and blind spots;
- model, environment and randomness metadata needed to replay a trajectory;
- realized resource measurements, hard runtime limits and remaining fractions.

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

Roy learns one organization policy. At each organization decision it first selects an active node and then selects an open candidate conditioned on that node:

```text
pi_theta(n, a, z | s)
  = pi_active(n | s)
    pi_candidate(a, z | s, n).
```

Here `a` is one grammar action and `z` is its open payload, such as a proposed child specification, acquisition requirement, connection, report, or prune target. Candidate count and node count are variable. No fixed role ID, fixed child catalog, teacher trajectory, or teacher score is part of the model.

The policy encoder is a typed relational network over derivation, dependency, communication, tool-use, evidence and return edges. It scores all currently active nodes, then embeds and scores open candidates conditioned on the sampled node and the full graph representation. Training replay must reconstruct this same joint conditional probability exactly.

Runtime feasibility is enforced before sampling by the action mask:

```text
A_B(s) = {a in A(s): executing a leaves the runtime within B}
pi_theta^B(a | s) = softmax(mask(logits_theta(s), A_B(s))).
```

The same mask semantics are reconstructed during replay. Runtime usage is absent from terminal utility, group advantage and the GRPO loss. The implementation does not assign theoretical costs such as `DERIVE = 2`; non-budget complexity features may be scheduler inputs only.

## 5. On-policy exploration and group sampling

Training samples complete organization trajectories directly from the masked old policy:

```text
tau_i ~ pi_theta_old^B(tau | x, e)
```

Every decision stores the exact masked old-policy joint log-probability. Replay computes the current probability with identical active-node and candidate masks, and the ratio is `exp(log pi_theta^B - log pi_theta_old^B)`. Exploration comes from stochastic policy sampling and an epoch-specific envelope, not a uniform-policy mixture. There is no imitation warm start or separate optimization phase.

Each τ³ task and epoch produces a group of eight complete trajectories. All eight share the same task, benchmark revision, isolated initial snapshot fingerprint, environment seed, executor configuration, runtime budget and exploration envelope. Only the organization-policy sampling seed differs. This makes the within-group utility difference interpretable as organization-policy variation rather than a confound from different structural constraints.

Fresh trajectories are collected on every epoch. For the default four-epoch run, conditional node/depth floors anneal as `(6,3) → (4,2) → (2,1) → (0,0)`. A floor delays `STOP` only while the model has reported a genuine residual gap with an available `DERIVE` or `ACQUIRE` action. It never inserts a synthetic gap or child. Ceilings remain hard runtime feasibility limits. Evaluation removes all minimum floors and samples one autonomous organization; it does not use best-of-eight test-time search.

## 6. Single-objective organization GRPO

For a group of terminal utilities `J_1 ... J_8`, compute only group-relative advantages:

```text
A_i = (J_i - mean(J)) / (std(J) + epsilon).
```

The policy update is the length-normalized clipped surrogate over the stored joint decision probabilities:

```text
L(theta) = -mean_i [
  (1 / |tau_i|) sum_t
  min(r_i,t A_i, clip(r_i,t, 1-epsilon, 1+epsilon) A_i)
]
```

There is one optimization target: τ³ terminal task utility. There are no cost, node-count, depth, communication, entropy, rationality, redundancy, teacher or auxiliary reward terms. A zero-variance group contributes zero preference gradient. Node necessity and counterfactual pruning may be reported as diagnostics but are not separate optimization objectives.

## 7. τ³ benchmark protocol

The benchmark is the official `sierra-research/tau2-bench` τ³ implementation pinned to commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`.

- Airline, retail and telecom retain the official train/test boundary.
- A deterministic subset of each official training split is held out for validation.
- Official test tasks are never used for updates or model selection.
- Banking knowledge tasks have no official training split in the pinned checkout, so they remain held out and are never training trajectories.
- All eight trajectories for one task/epoch use the same task identity, benchmark revision, environment seed, runtime budget and envelope; only organization sampling seeds differ.

Primary evaluation compares:

- `single_agent_direct`: one τ³ LLM agent without recursive derivation;
- `fixed_complete_mas`: a non-learned complete multi-agent organization with matched reporting;
- `roy_runtime_heuristic`: the current Roy delegation behavior;
- `learned_information_realization`: one organization sampled from the trained policy.

Report official τ³ reward and end-to-end success, paired task-level differences against `single_agent_direct`, bootstrap 95% confidence intervals, tokens, LLM calls, wall-clock, realized nodes/depth, dependencies, waits, communication edges, redundant branches and truncation rate. Results whose interval crosses zero are inconclusive.

## 8. Training and evaluation invariants

- Training consumes only manifest entries labeled `train`.
- Every GRPO group contains exactly eight complete trajectories from one task.
- Every policy record stores state fingerprint, active node, open candidate, exact masked old-policy joint log-probability, envelope, runtime state and the replayable policy state.
- The final utility is the only value used to compute advantage.
- No teacher names, teacher outputs or teacher scores may appear in a training trajectory.
- Resume restores model, optimizer and completed trajectory/group identifiers without changing the train/test boundary.
- A policy-selected `STOP` is `terminated`; decision, resource or environment limits are `truncated`. Both receive the benchmark terminal utility, but truncation never creates a fake `STOP` log-probability.
- Evaluation never updates model weights and never applies training minimum node/depth envelopes.
- API nondeterminism is addressed by request/response capture, fixed configuration and repeated paired evaluation, not by claiming exact deterministic counterfactuals.

## 9. Limits

MIA's data-processing bound still applies: communication and recursion cannot create information unavailable from context, tools or environment. The organization can improve acquisition, representation and conversion, but shared foundation models create correlated errors. Utility judges can be biased, external environments can drift, tool side effects may be non-clonable, and snapshot replay may be incomplete. Claims therefore concern measured τ³ performance under the pinned protocol, not global optimality or universal information gain.
