# Roy

Roy is a TypeScript runtime for building observable, budget-aware, Theory-of-Mind multi-agent systems. A session starts with Roy as the root agent. Roy can solve a task directly, request clarification, or create a bounded tree of specialized child agents and subteams.

Every agent is a runtime actor with an identity, strict finite-state lifecycle, approved skills and tools, a private memory boundary, token accounting, parent-child relationships, and persisted message/event traces. CLI and HTTP interfaces are adapters over the same runtime.

## Runtime Model

```text
User input
  -> CLI or HTTP adapter
  -> ContextWindowManager
  -> Roy delegation assessment
  -> candidate propose / evaluate / select
  -> solve directly or create child agents/subteam
  -> message-mediated execution
  -> tool policy and approval
  -> parent-level synthesis
  -> final response
  -> trace, cache, memory proposal, and budget settlement
```

The runtime supports recursive delegation:

```text
Roy [root]
├── Researcher-1
│   └── Critic-1
└── Planner-1
    └── Tester-1
```

Each parent owns its direct children and synthesizes their results before returning a result upward.

When a delegation needs multiple cooperating actors, the runtime creates a formal subteam rather than treating the agents as a flat list:

```text
Roy [root]
└── AnalysisTeam [subteam, ToM-2]
    ├── Researcher-1
    ├── Critic-1
    └── Summarizer-1
```

A subteam has its own identity, FSM, task/result boundary, synthesis call, token usage, message flow, memory directory, topology snapshot, session log, and reusable cache pattern. Member tasks flow `parent -> team -> member`; member results flow `member -> team -> parent`.

Team execution is policy controlled. A team can run sequentially or with bounded concurrency, stop on the first failure or continue in best-effort mode, and require a minimum number of successful members before synthesis. Cached team patterns restore member tasks, tools, skills, lead assignment, full ToM profiles, cognitive-gap assignments, and execution policy into a new runtime team instance.

## ToM-Aware Delegation

ToM is the default agent communication protocol, not a mandatory wire format. The communication layer is replaceable: `tom` renders beliefs, goals, uncertainty, perspectives, and participant models; `structured` renders a simpler provider-neutral message envelope. Applications can register additional `AgentCommunicationProtocol` implementations without changing the queue, runtime actor, or agent implementation.

This communication protocol is separate from the ToM delegation planner. The planner may use ToM semantics for cognitive-gap analysis, while a message can still be delivered through `structured` or another registered protocol.

Delegation is driven by explicit cognitive gaps rather than role labels alone. Before candidate selection, `ToMDelegationPlanner` models the parent actor's current beliefs, goals, and uncertainty, then derives bounded gaps such as missing evidence, adversarial perspective, planning, implementation, verification, or belief reconciliation.

Each delegated agent receives a full `ToMProfile`:

```text
beliefScope       facts or hypotheses the actor owns
goalModel         the cognitive result it must produce
uncertainty       unresolved questions it must reduce or preserve
perspective       the distinct viewpoint that justifies the actor
observesAgents    actors whose outputs it can observe
modelsAgents      actors whose beliefs/goals it explicitly models
cognitiveGaps     gap IDs that explain why the actor exists
```

When multiple perspectives are required, the runtime can complete a partial delegation plan with missing specialists and a higher-order synthesizer. Candidate scoring measures weighted gap coverage, perspective diversity, higher-order fit, cost, cache reuse, and task utility. Policy and budget limits can still select a deliberately partial plan; the uncovered gaps remain observable instead of being hidden.

Use `/agents --tree --tom`, `/teams --tree --tom`, `/tom`, or `GET /v1/tom` to inspect the resulting epistemic structure. Events include `tom.task.analyzed`, `tom.gap.identified`, `tom.higher_order.required`, `tom.profile.assigned`, `tom.team.profile.created`, and `tom.delegation.coverage.evaluated`.

Workspace policy is stored in `.roy/config.json`:

```json
{
  "tom": {
    "enabled": true,
    "autoCompleteGaps": true,
    "maxAgentsPerDecision": 3,
    "minimumCoverage": 0.6,
    "requireExistenceReason": true,
    "higherOrderForMultiplePerspectives": true
  },
  "communication": {
    "defaultProtocol": "tom",
    "allowMessageOverride": true,
    "traceWindowSize": 200,
    "includeCompletedMessages": true
  }
}
```

Existing workspace configs are migrated to schema version 5 without discarding user overrides.

## Core Architecture

```text
src/core/
  agent/        Agent identity, state, usage, and execution
  runtime/      Runtime orchestration and actor registry
  context/      Bounded public/private/session context construction
  executor/     Strict FSM and signal control
  delegation/   Candidate generation and pluggable scorers
  tom/          Cognitive-gap analysis, ToM profiles, and coverage evaluation
  communication/ Replaceable message protocols and multi-party trace delivery
  evolution/    Team-first genomes, lifecycle FSM, operators, evaluation, and selection
  budget/       Token allocation market and settlement
  team/         Formal subteam actor registry
  queue/        Runtime message queue and scheduler
  memory/       Workspace memory, sessions, traces, and caches
  skills/       Agent-facing composed capabilities
  tools/        Tool registry, planning, policy, and approvals
  prompts/      Prompt templates and slot rendering
  llm/          Anthropic and OpenAI-compatible providers

src/cli/        Terminal adapter
src/server/     Express and Socket.IO adapter
```

## Delegation Scoring

Delegation candidates are evaluated by replaceable scorers:

- task/archetype fit
- expected token cost and remaining budget
- weighted cognitive-gap coverage and perspective diversity
- cache similarity using deterministic task embeddings
- cache reuse and mutation lineage
- LLM-based candidate evaluation

Delegation candidate scoring remains a lightweight control-plane step. Phase 6 evolution is a separate, full runtime lifecycle:

```text
propose -> instantiate -> execute -> evaluate -> select -> mutate -> integrate -> done
```

Every evolution candidate is represented as a `TeamGenome`. A one-member genome is compiled directly to an agent; larger genomes are compiled to formal subteams. Candidate actors run through the existing message queue, strict agent/team FSMs, tool policies, ToM profiles, budget market, memory, and traces. Rejected candidates are retained in history but are not integrated into pattern memory.

Authorized capability execution is always followed by an agent synthesis step. Raw command results and unresolved tool-call markup are not valid candidate answers; unresolved tool intents fail evaluation and cannot enter evolution pattern memory.

Grounding-required genomes must bind an approved filesystem evidence tool before instantiation. Runtime tool evidence is carried as structured paths and summaries, and is attached to an agent result when the model omits the concrete observations it used. Provider/model-aware token estimators drive budget requests; reasoning models receive a separate synthesis reserve so thinking does not consume the entire visible-answer allowance.

Cached genomes are validated before population admission. Structurally stale patterns are marked `deprecated`, while an invalid individual candidate is rejected without consuming an execution or agent slot. A deterministic team fallback can preserve member diagnostics when a model returns no visible synthesis, but that fallback is explicitly ineligible for evolution selection or pattern integration.

Selected genomes are stored in `.roy/cache/evolution-patterns.json` and linked back to their concrete agent/team patterns. Full runs and metrics are appended to `.roy/cache/evolution-history.jsonl`. Evolution defaults to `manual` mode so normal chat does not unexpectedly execute a candidate population. Set `/evo mode auto` to route complex delegated turns through evolution.

The workspace defaults can be changed in `.roy/config.json` or through `/evo` and the evolution config API:

```json
{
  "evolution": {
    "enabled": true,
    "mode": "manual",
    "profile": "evo_team",
    "populationSize": 3,
    "generations": 1,
    "topK": 1,
    "maxExecutedCandidates": 3,
    "integrationMinimumScore": 0.55,
    "patternSimilarityThreshold": 0.35,
    "useLlmJudge": false,
    "ablations": {
      "withoutSubagents": false,
      "withoutToMProfile": false,
      "withoutBudgetMarket": false,
      "withoutEvoMutation": false,
      "withoutPatternMemory": false
    }
  }
}
```

The LLM scorer is also a metered control-plane call. Before scoring, the runtime either reserves a dedicated root allocation or reuses the active parent-agent allocation. The provider response is attributed to that parent actor, including reported thinking/cache tokens, and the allocation is settled or released on failure. Custom planners and scorer hooks are available through the `roy/delegation` package export.

## Context And Memory

Roy initializes a project-local `.roy/` workspace:

```text
.roy/
  public/       Shared project, context, decision, and constraint memory
  agents/       Private identity, prompt, context, memory, state, and sessions
  teams/        Team memory and topology
  cache/        Agent, delegation, team, proposal, and evolution records
  queue/        Queue state location
  sessions/     Complete conversation JSONL files
  traces/       Runtime event traces
  config.json   Workspace runtime policy
```

`ContextWindowManager` loads bounded public memory, the current agent's private memory, approved parent context, and a compact recent-session window. It never injects another agent's private memory by default.

Agent prompt templates support these runtime slots:

```text
{{public_context}}
{{agent_private_memory}}
{{agent_identity}}
{{tom_profile}}
{{communication_context}}
{{multi_party_traces}}
{{available_skills}}
{{available_tools}}
{{parent_context}}
{{task}}
```

Every `BaseAgent` implements the same trace receiver contract: `receiveSystemTrace`, `receiveSystemTraces`, `getSystemTraces`, and `receiveCommunicationContext`. Traces are observable runtime facts such as messages, tools, results, and state transitions; hidden model chain-of-thought is never treated as a trace. `ContextWindowManager` accepts the same protocol context and trace records with bounded token allocation.

Protocol extensions implement `AgentCommunicationProtocol` and can be registered through `runtime.registerCommunicationProtocol(...)`. The runtime supports a global default, per-agent selection during creation, and explicit per-message override.

## Tools And Approvals

Built-in tools include:

- `fs.list`
- `fs.read`
- `shell.exec`

Agents plan tool use only when the task needs external evidence or execution. Tool availability comes from parent-approved bindings. Read-only tools can run automatically by default; write and execute tools require an approval unless workspace policy overrides them. `shell.exec` also applies its own command allowlist.

## Budget Control

Token metering is always enabled. The default allocation policy is `market`; the session limit may still be unlimited. Agents and teams submit requests with purpose, priority, resource estimates, requested tokens, and a minimum viable grant. A replaceable `ReasoningInvestmentModel` combines root/parent utility, historical outcomes, evidence gain, uncertainty reduction, conflict resolution, verification gain, cache confidence, execution risk, token/context cost, tool calls, and latency into a structured expected-return estimate.

Three replaceable policies are included:

- `unlimited`: grants every valid request while retaining the full ledger.
- `fixed`: grants in request order from the remaining session supply.
- `market`: scores competing requests by priority, risk-adjusted utility, expected return, confidence, and cost. It performs minimum-viable admission followed by iterative clearing, supports partial grants, and can rebalance active reservations.

Execution can attach a realized outcome to an allocation. Success, quality, evidence gain, uncertainty reduction, conflict resolution, and verification gain update allocation efficiency and purpose-level outcome history. Later requests reuse this history instead of relying only on static archetype values.

Usage is normalized into input, output, total, thinking, cached-input, and cache-creation tokens. OpenAI-compatible providers read reasoning and prompt-cache detail fields when present. Anthropic providers read input/output and cache read/creation fields. A missing thinking count is stored as `null`, never guessed. If a provider returns no usage, a provider/model-family estimator is used and marked `estimated`.

The market is exposed through `GET /v1/budget/market`. External controllers can record observed value with `POST /v1/budget/allocations/:id/outcome`.

Workspace policy is configured in `.roy/config.json`:

```json
{
  "budgetMarket": {
    "enabled": true,
    "mode": "market",
    "minimumGrantTokens": 256,
    "accountingDimension": "total_tokens",
    "rebalanceOnRequest": false,
    "defaultPriority": "medium",
    "priorityWeights": {
      "low": 0.6,
      "medium": 1,
      "normal": 1,
      "high": 1.5,
      "critical": 2.2
    }
  }
}
```

`accountingDimension` can be `total_tokens`, `output_tokens`, or `thinking_tokens`. When a model does not report thinking tokens, thinking-based accounting falls back to total tokens rather than fabricating a reasoning count.

The selected accounting dimension controls reservation consumption, overrun detection, remaining supply, and rebalancing. Each allocation still retains the full normalized model usage so input, output, total, thinking, and cache metrics remain inspectable independently of the market dimension.

Under `total_tokens` accounting, Roy reserves completion capacity before each call and deterministically truncates oversized agent prompts, observations, communication context, and recursive synthesis input. Capability selection uses a separate compact control-plane prompt containing only the task and parent-approved capabilities, so tool routing does not duplicate the full agent memory/context charge.

The runtime enforces:

- maximum children per parent
- maximum agent-tree depth
- maximum total agents per turn
- budget-aware candidate reduction
- per-agent tool-call limits
- strict FSM states for child creation

## Installation

```bash
npm install
cp .env.example .env
```

Configure one provider in `.env`. DeepSeek uses the OpenAI-compatible API:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEFAULT_MODEL=deepseek-v4-flash
```

Roy also supports Anthropic and OpenAI-compatible providers through `roy.config.yaml` and environment variables.

## Run

Start the CLI:

```bash
npm run dev:cli
```

Start the HTTP and Socket.IO server:

```bash
npm run dev:server
```

Build and test:

```bash
npm run build
npm test
```

Use Roy as a library:

```ts
import { Runtime } from 'roy';

const runtime = new Runtime();
await runtime.initialize({ sessionId: 'my-session', workspaceCwd: process.cwd() });
```

The package also exposes `roy/runtime`, `roy/team`, `roy/tom`, `roy/communication`, `roy/queue`, and `roy/memory` subpaths. Installed binaries are `roy` and `roy-server`.

## CLI Commands

```text
/status
/agents --tree --tom
/agents archetypes
/agents policy <agentId>
/spawn researcher "Inspect the project structure"
/spawn researcher --protocol structured "Inspect the project structure"
/spawn critic --parent <agentId> "Review the parent result"
/teams
/teams --tree
/team <teamId>
/team create --name "AnalysisTeam" --description "Inspect architecture and risks" --failure best_effort --concurrency 2
/team add <teamId> researcher "Inspect the project structure"
/team run <teamId> "Analyze the project architecture"
/budget
/budget --market
/budget rebalance
/events --latest 50
/messages --correlation <id>
/context render researcher --task "Inspect the repository"
/prompt render researcher --task "Inspect the repository"
/tools approvals
/tools approve <approvalId>
/tools deny <approvalId>
/memory
/cache agents
/cache delegations
/cache teams
/cache evolution
/evo
/evo run --profile evo_team "Analyze this repository architecture and identify risks"
/evo benchmark "Analyze this repository architecture and identify risks"
/evo patterns
/evo history
/evo mode manual
/evo mode auto
/evo ablate tom on
/evo ablate tom off
/traces latest
/communication
/communication use structured
/communication traces 20
```

Normal chat input uses root-controlled delegation. `/spawn` remains available for controlled testing and explicit actor creation.

## HTTP API

Primary endpoints:

```text
POST /v1/chat
GET  /v1/status
GET  /v1/agents
GET  /v1/agents/tree
POST /v1/agents
POST /v1/agents/:id/run
GET  /v1/teams
GET  /v1/teams/tree
GET  /v1/teams/:id
POST /v1/teams
POST /v1/teams/:id/agents
POST /v1/teams/:id/run
GET  /v1/evolution
GET  /v1/evolution/patterns
GET  /v1/evolution/history
PATCH /v1/evolution/config
POST /v1/evolution/run
POST /v1/evolution/benchmark
GET  /v1/runtime/sessions
DELETE /v1/runtime/session
GET  /v1/budget
GET  /v1/budget/market
GET  /v1/budget/allocations/:id
POST /v1/budget/allocations
POST /v1/budget/allocations/batch
POST /v1/budget/allocations/:id/consume
POST /v1/budget/allocations/:id/settle
POST /v1/budget/allocations/:id/release
POST /v1/budget/rebalance
GET  /v1/cache/:kind
GET  /v1/events
GET  /v1/messages
GET  /v1/communication
GET  /v1/communication/traces
POST /v1/communication/traces
POST /v1/communication/default
GET  /v1/queue
POST /v1/context/render
GET  /v1/tools/approvals
POST /v1/tools/approvals/:id
POST /v1/tools/:name/execute
GET  /v1/memory
GET  /v1/traces
```

## Validation

The test suite covers root-controlled and recursive delegation, strict nested FSM transitions, context boundaries, subteam lifecycle, candidate scoring, cache mutation, competitive budget allocation and rebalancing, provider-specific token normalization, tool approval, memory persistence, queue transitions, and CLI-facing runtime behavior.

## Contact

Use [GitHub Issues](https://github.com/erwinmsmith/Roy/issues) for bug reports and engineering discussions.
