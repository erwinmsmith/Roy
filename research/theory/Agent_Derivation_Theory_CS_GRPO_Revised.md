# Agent Derivation Theory and CS-GRPO

Status: canonical Roy research theory, revision 1 (2026-08-15).

This document is the repository authority for the first Roy structural-learning experiments. It revises `Agent_Derivation_Theory_CS_GRPO_Revised.md` from the research input without modifying that source file. It adopts the information-preservation and bounded-conversion distinctions in *Meta-Information Activation* (MIA), while replacing dimensionally mixed optimization claims with an operational task-utility objective.

Reviewed source provenance:

- Input Markdown SHA-256: `57f9abb3c499b2eb96a6f78a4865ee8230acdc118b85de2009d208ee6c3d46e6`.
- MIA PDF SHA-256: `391cec40d5d7f01414a2ffec202c88322dabf8ea99f0daf8fbbae120bc59dd8d`.

The Downloads copies are the immutable research inputs. This repository document is the consolidated normative revision used by code, tests and experiments; details from the input remain informative unless they conflict with an explicit rule below.

## 1. Scope and claims

An agent derivation policy decides whether a running parent should continue locally, derive a child, or return. The policy may also choose the child specification and optional communication shortcuts. Its purpose is not to maximize the number of agents. It should improve terminal task utility under a finite resource envelope.

MIA supplies two constraints that remain intact:

1. For a Markov chain from complete context through a representation to an answer, the Data Processing Inequality bounds the information retained downstream. Agent communication does not violate this bound.
2. Preserving task-relevant information and converting preserved information into a calibrated final output are distinct. A bounded architecture can retain useful evidence and still fail at aggregation.

Task utility, best-trace retention, and conversion deficit are operational diagnostics. They are not assumed to be unbiased estimators of mutual information.

## 2. Objective

The former scalar objective `IL + EL + DL - beta GL` adds quantities without a shared measurement scale and is withdrawn. Let `U_T(y, o) in [0, 1]` be the terminal utility for task `T`, latent or reference answer `y`, and terminal output `o`. Let `B` be a finite resource envelope. The structural objective is

```text
maximize_pi  E_pi[U_T(Y, O_terminal)]
subject to   cumulative resource use <= B
             terminal constraints are satisfied.
```

Intrinsic load, extraneous load, depth load, and germane conversion remain explanatory variables and stratification axes. They are not summed into the training reward unless separately normalized and empirically validated.

## 3. Finite-resource terminating SMDP

The process is a stochastic semi-Markov decision process

```text
M = (S, A, A_legal, P, D, R, gamma, B, S_terminal).
```

- `S` contains the Parent-local observable event graph, local projections of the derivation tree, dependency DAG and communication graph, resource state, environment revision, model metadata, and randomness metadata.
- `A = {CONTINUE, BRANCH, RETURN}` at the node level. `BRANCH` additionally samples a `ChildSpecification`.
- `A_legal(s)` is an action mask. `BRANCH` requires remaining branch capacity. `RETURN` is illegal while a required dependency is unresolved or the output contract is unsatisfied.
- `P(s', delta, r | s, a)` is the transition kernel over state, duration and reward.
- `D` records action duration and resource consumption.
- Reward is zero or shaped during execution and equals terminal task utility at termination. Evaluation always reports unshaped terminal utility.
- The resource envelope and termination rule make the process finite. An implementation must stop on terminal output, exhaustion, explicit horizon, or unrecoverable environment failure.

The Parent observes only events addressed to it, events emitted by it, public events, and results legally returned along derivation/dependency/communication edges. Sibling private traces are not visible by default.

## 4. Checkpoints and counterfactual validity

A `StructuralCheckpoint` is immutable and versioned. It contains:

- Parent-local events and graph edges;
- local tree, dependency and communication projections;
- remaining resources and legal actions;
- environment revision and restorable environment state reference;
- provider/model parameters and randomness metadata;
- an immutable fingerprint over all semantically relevant fields.

Controlled environments must support full clone and restore. External environments use deterministic fixtures/replay when possible, otherwise isolated instances. Counterfactuals within one group must start from the same checkpoint fingerprint, total resources, termination condition, environment revision and common-random-number seed. The intervened structural action is the only intended difference.

When a snapshot cannot be cloned, the experiment reports that limitation and does not describe the comparison as an exact counterfactual.

## 5. Theoretical and estimated values

The theoretical action value is

```text
Q*(s, a) = sup_pi E[U_terminal | s, a, pi].
```

The optimal branch gap is the difference between the best legal structural action and a selected action under `Q*`. Finite experiments do not observe `Q*`. They estimate

```text
Q^mu(s, a) = E[U_terminal | s, a, downstream rollout policy mu],
```

where `mu` is fixed within a comparison. `Q^mu` approaches `Q*` only under a stated consistency condition, such as increasingly capable rollout policies with sufficient coverage and convergent value estimates. Reports must label measured regret as rollout-policy structural regret unless such a condition is justified.

## 6. Joint structural policy and hierarchical credit

The joint policy factorizes as

```text
pi(a, c | s) = pi_node(a | s) p_derive(c | s, a = BRANCH).
```

The outer group compares three node-level quantities:

```text
V_CONTINUE, V_RETURN, aggregate_c(V_BRANCH,c).
```

The default branch aggregate is the arithmetic mean over the fixed candidate set. The inner group compares child specifications only after conditioning on `BRANCH`. Outer standardized advantages update `pi_node`; inner standardized advantages update `p_derive`. The `BRANCH` node log-probability appears once per checkpoint group. It is not duplicated for every child, and it never receives mutually conflicting child-level advantages.

For either level, CS-GRPO applies an action mask and the clipped surrogate

```text
L = -E[min(r(theta) A, clip(r(theta), 1-epsilon, 1+epsilon) A)].
```

A zero-variance group has zero standardized advantage and therefore contributes no preference gradient.

## 7. Acquisition and activation

For a clonable deterministic environment, define matched runs:

```text
U_0 = utility without child-acquired evidence,
U_E = utility after evidence acquisition with the acquisition cost charged,
U_F = utility after evidence acquisition and downstream activation.

G_acq = U_E - U_0
G_act = U_F - U_E
U_F - U_0 = G_acq + G_act.
```

All three runs share the initial checkpoint, total resource budget and terminal condition. `U_E` explicitly records the budget remaining after evidence acquisition. The deterministic reconstruction error must be at most `1e-6`.

If obtaining tool evidence changes external state in a way that cannot be replayed or isolated, acquisition and activation are not separately identified. The report omits the decomposition rather than assigning an artificial value.

## 8. Structural runtime versions

- **V0 - Tree runtime:** immutable lineage, restorable controlled state, resource accounting and exact checkpoint fingerprints.
- **V1 - Node policy:** masked `CONTINUE`, `BRANCH`, `RETURN` decisions over Parent-local state.
- **V2 - Open derivation:** `BRANCH` creates a typed child specification with task, context, tools, resource slice, output contract and dependencies.
- **V3 - Dependency execution:** artifact/subgoal dependency DAG, automatic `WAIT`, and wake-up when all required producers resolve.
- **V4 - Communication:** dependency-required routes cannot be removed; optional communication shortcuts may be learned and closed.
- **V5 - CS-GRPO:** relational event-graph policy, hierarchical node/derivation credit, trajectory collection, checkpointed training and evaluation.

The TypeScript runtime keeps structural learning behind `structuralLearning.enabled`. When disabled, a compatibility adapter preserves existing Roy delegation behavior. A Python/PyTorch sidecar communicates over versioned JSONL on stdio, with bounded timeout, restart and fallback.

## 9. Controlled benchmark

Controlled Derivation Benchmark v1 contains 180 deterministically generated English instances: 90 train, 30 validation and 60 test. It balances Activation, Acquisition and Mixed families. The final half of each family's test slice is OOD in depth and, where applicable, tool and branch pattern.

Each instance exposes one checkpoint, three child specifications and two repeated rollouts per action/specification with a common environment seed. Baselines are no derivation, random derivation, fixed branch count, current Roy heuristic, full-trajectory GRPO, graph-ablated CS-GRPO, Node-only CS-GRPO and full V0-V4.

## 10. External pilot and statistics

The first external pilot is text/tool based: tau Knowledge and TUA-Bench, each with five fixed tasks and three repeats, across no derivation, current heuristic, Node-only and full policies. This is 120 episodes. Voice is future work.

All DeepSeek runs share one persistent 10,000,000-token ledger. A request reserves its maximum possible use before dispatch. If reservation would cross the cap, the runner saves current state and stops; it never automatically exceeds the cap.

Reports include terminal utility, rollout-policy structural regret, action confusion matrix, `G_acq`, `G_act`, work, span, nodes, wait time, redundant branches, communication edges, token use and wall time. Paired bootstrap 95% confidence intervals and multiple random repeats are required. If an interval crosses zero, the result is reported as inconclusive.

## 11. Limits and threats to validity

- External API sampling may remain nondeterministic despite a seed. Reproducibility relies on complete request/response capture, pinned configuration and paired repeats.
- Shared foundation models induce correlated errors across agents, violating naive independence assumptions.
- Snapshot cloning may be incomplete for external services; fixture/replay changes ecological validity.
- Utility judges may be biased, noisy or contaminated by model-family preference.
- Tool and benchmark environments may drift after the pinned revision.
- The first encoder is the frozen English `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, with 384 dimensions. No Chinese structural-decision generalization claim follows.
- A high `Q^mu` value demonstrates benefit under `mu`, not global optimality.

## 12. Falsification criteria

The central empirical claim is weakened if full CS-GRPO fails to improve paired task utility or structural regret over the Roy heuristic, if gains disappear under matched resource accounting, if event-graph ablations perform equivalently, or if acquisition/activation effects do not reconstruct under deterministic controls. Such outcomes remain reportable results and must not be filtered by expected direction.
