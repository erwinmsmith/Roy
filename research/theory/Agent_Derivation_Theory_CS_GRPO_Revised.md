# Roy Information Realization and Autonomous Organization GRPO

Status: canonical research specification for the `exp` branch (2026-08-18).

This document is the normative bridge from MIA to Roy's recursive organization policy. It replaces the earlier staged structural-learning formulation. The implementation must not introduce predefined agent roles, teacher systems, imitation learning, a weighted sum of objectives, or per-trajectory resource caps as the mathematical resource constraint.

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
subject to   E[tau ~ pi][C(tau)] <= B.
```

Cost is a distribution-level feasibility constraint, not a reward component. Agent count, depth, latency, token use, tool calls, topology complexity, communication, rationality and redundancy are never added to `J`. An individual sampled trajectory may consume more than `B`; feasibility is defined by expected consumption under the sampling distribution.

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
- realized resource measurements and the current expected-resource distribution.

Derivation and communication have different meanings. A derivation edge records why a child exists. A communication edge records which nodes may exchange information. A dependency edge records which artifact or claim must be produced before another node can proceed. Combining these edge types into a single tree is invalid.

External observations must be stored separately from model inference. Tool, API, database, KB, code and environment results require provenance and their claim-support links. This preserves the distinction between information acquisition and reasoning over already available information.

## 3. Fixed grammar, open agent space

The organization grammar is fixed:

```text
DERIVE, ACQUIRE, CONNECT, EXECUTE, RETURN, PRUNE, STOP
```

The grammar does not define roles. `DERIVE` takes a freely generated agent specification tied to one residual requirement. A valid child:

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

Here `a` is one grammar action and `z` is its open payload, such as a generated child specification, acquisition requirement, connection, report, or prune target. Candidate count and node count are variable. No fixed role ID, fixed child catalog, teacher trajectory, or teacher score is part of the model.

The policy encoder is a typed relational network over derivation, dependency, communication, tool-use, evidence and return edges. It scores all currently active nodes, then embeds and scores open candidates conditioned on the sampled node and the full graph representation. Training replay must reconstruct this same joint conditional probability exactly.

The expected resource constraint is enforced by projecting the legal candidate distribution:

```text
min_q  KL(q || pi_theta)
subject to  E[a ~ q][c(s,a)] <= b(s).
```

This projection changes the action distribution only. Resource cost is absent from terminal utility, group advantage and the GRPO loss. Legal high-cost actions retain non-zero probability whenever the expected constraint is feasible.

## 5. On-policy exploration and group sampling

Training samples complete organization trajectories from exactly one behavior distribution:

```text
q_theta(tau | x)
  = (1 - alpha) q_explore(tau | x)
    + alpha pi_theta(tau | x).
```

The behavior log-probability stored for every decision is the exact mixture probability, not the probability from only one component. The sampled old-policy probability is stored separately for the GRPO ratio. `alpha` may be annealed as part of this same training process; there is no imitation warm start or separate optimization phase.

Each τ³ training query produces a group of eight complete trajectories:

| Group member | Node exploration envelope | Depth exploration envelope | Mode |
| --- | ---: | ---: | --- |
| 1 | 1–4 | 1–2 | shallow |
| 2 | 2–5 | 1–3 | shallow |
| 3 | 4–7 | 2–3 | medium |
| 4 | 4–7 | 2–4 | medium |
| 5 | 6–9 | 3–4 | deep |
| 6 | 6–9 | 3–4 | deep |
| 7 | 6–12 | 3–5 | expansive |
| 8 | 6–12 | 4–5 | expansive |

These envelopes define exploration support during training. They do not prescribe roles, tasks, topology or communication, and they are not resource budgets or reward terms. Evaluation removes the minimum node/depth requirements and samples one autonomous organization from the learned policy; it does not use best-of-eight test-time search.

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
- All eight trajectories for one query use the same task identity and benchmark revision, while their exploration envelopes and random streams are recorded explicitly.

Primary evaluation compares:

- `single_agent_direct`: one τ³ LLM agent without recursive derivation;
- `fixed_complete_mas`: a non-learned complete multi-agent organization with matched reporting;
- `roy_runtime_heuristic`: the current Roy delegation behavior;
- `learned_information_realization`: one organization sampled from the trained policy.

Report official τ³ reward and end-to-end success, paired task-level differences against `single_agent_direct`, bootstrap 95% confidence intervals, tokens, LLM calls, wall-clock, realized nodes/depth, dependencies, waits, communication edges and redundant branches. Results whose interval crosses zero are inconclusive.

## 8. Training and evaluation invariants

- Training consumes only manifest entries labeled `train`.
- Every GRPO group contains exactly eight complete trajectories from one task.
- Every policy record stores state fingerprint, active node, open candidate, exact behavior log-probability, sampled old-policy log-probability, exploration weight, envelope and projected expected resource.
- The final utility is the only value used to compute advantage.
- No teacher names, teacher outputs or teacher scores may appear in a training trajectory.
- Resume restores model, optimizer and completed trajectory/group identifiers without changing the train/test boundary.
- Evaluation never updates model weights and never applies training minimum node/depth envelopes.
- API nondeterminism is addressed by request/response capture, fixed configuration and repeated paired evaluation, not by claiming exact deterministic counterfactuals.

## 9. Limits

MIA's data-processing bound still applies: communication and recursion cannot create information unavailable from context, tools or environment. The organization can improve acquisition, representation and conversion, but shared foundation models create correlated errors. Utility judges can be biased, external environments can drift, tool side effects may be non-clonable, and snapshot replay may be incomplete. Claims therefore concern measured τ³ performance under the pinned protocol, not global optimality or universal information gain.
