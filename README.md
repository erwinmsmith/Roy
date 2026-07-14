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
└── ReviewTeam [subteam, ToM-2]
    ├── Researcher-1
    ├── Critic-1
    └── Summarizer-1
```

A subteam has its own identity, FSM, task/result boundary, synthesis call, token usage, message flow, memory directory, topology snapshot, session log, and reusable cache pattern. Member tasks flow `parent -> team -> member`; member results flow `member -> team -> parent`.

## Core Architecture

```text
src/core/
  agent/        Agent identity, state, usage, and execution
  runtime/      Runtime orchestration and actor registry
  context/      Bounded public/private/session context construction
  executor/     Strict FSM and signal control
  delegation/   Candidate generation and pluggable scorers
  evolution/    Propose/evaluate/select execution pipeline
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
- ToM role complementarity
- cache similarity using deterministic task embeddings
- cache reuse and mutation lineage
- LLM-based candidate evaluation

Candidate generation and selection run through an explicit `propose -> evaluate -> select` pipeline. Evaluations are written to `.roy/cache/evolution-history.jsonl` for inspection and later reuse.

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
{{available_skills}}
{{available_tools}}
{{parent_context}}
{{task}}
```

## Tools And Approvals

Built-in tools include:

- `fs.list`
- `fs.read`
- `shell.exec`

Agents plan tool use only when the task needs external evidence or execution. Tool availability comes from parent-approved bindings. Read-only tools can run automatically by default; write and execute tools require an approval unless workspace policy overrides them. `shell.exec` also applies its own command allowlist.

## Budget Control

Token metering is always enabled. Without a configured limit, the budget is unlimited. With a limit, the budget market reserves tokens before agent creation, grants or denies requests, and settles allocations against actual usage.

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

## CLI Commands

```text
/status
/agents --tree --tom
/agents archetypes
/agents policy <agentId>
/spawn researcher "Inspect the project structure"
/spawn critic --parent <agentId> "Review the parent result"
/teams
/teams --tree
/team <teamId>
/team create --name "AnalysisTeam" --description "Inspect architecture and risks"
/team add <teamId> researcher "Inspect the project structure"
/team run <teamId> "Analyze the project architecture"
/budget
/budget market
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
/traces latest
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
GET  /v1/budget
GET  /v1/budget/market
GET  /v1/cache/:kind
GET  /v1/events
GET  /v1/messages
GET  /v1/queue
POST /v1/context/render
GET  /v1/tools/approvals
POST /v1/tools/approvals/:id
POST /v1/tools/:name/execute
GET  /v1/memory
GET  /v1/traces
```

## Validation

The test suite covers root-controlled and recursive delegation, strict nested FSM transitions, context boundaries, subteam lifecycle, candidate scoring, cache mutation, budget allocation, tool approval, memory persistence, queue transitions, and CLI-facing runtime behavior.

## Contact

Use [GitHub Issues](https://github.com/erwinmsmith/Roy/issues) for bug reports and engineering discussions.
