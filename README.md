# Roy

Roy is an experimental TypeScript framework for building dynamically expanding multi-agent systems. It starts from a single root agent and grows a Theory-of-Mind-aware reasoning structure only when the current reasoning trace shows that more perspective, verification, evidence, or decomposition is worth the cost.

The core design is based on FSM-controlled Evo-ToM expansion: finite-state control governs when the system can expand, ToM/MIA diagnosis explains why expansion may be needed, a market-style allocator estimates whether the next thinking investment is worth paying for, and EvoAgent-style derivation creates or reuses specialized agents and subteams.

## Core Idea

Most multi-agent systems begin with a predefined team. Roy takes the opposite approach. A task starts with one first-order root agent. As the trace develops, the system diagnoses uncertainty, disagreement, missing evidence, blind spots, and reliability gaps. Only then can the finite-state controller decide whether to continue solo reasoning, reuse cached reasoning structures, derive a new agent, derive a ToM-aware subteam, verify, backtrack, merge results, or finalize.

This keeps the multi-agent structure adaptive instead of fixed:

```text
root agent
  -> diagnose reasoning bottleneck
  -> estimate expected reasoning return
  -> derive or reuse agent/subteam when useful
  -> execute ToM-aware inference
  -> merge explicit outputs and meta-traces
  -> verify, backtrack, or finalize
```

## Architecture

Roy combines several layers:

- **FSM control**: explicit runtime states decide when the system should continue, diagnose, derive, reuse, execute, merge, verify, backtrack, or finish.
- **ToM/MIA diagnosis**: reasoning traces are inspected for beliefs, uncertainty, reliability, evidence coverage, disagreement, and blind spots.
- **Market-based thinking allocation**: candidate reasoning investments are scored by expected gain, cost, risk, budget pressure, and relevance to the user's objective.
- **Evo-style derivation**: mutation, crossover, and selection generate candidate agents or ToM-aware subteams from the current parent unit.
- **Cache reuse**: previously useful agents, subteams, bottleneck mappings, team-generation directions, and ToM inference traces can be reused when cheaper than recomputation.
- **Modular prompt management**: prompts are treated as versioned contracts with structured inputs and outputs rather than inline strings hidden inside agent logic.

## Current Implementation

The repository currently includes:

- action and planner primitives
- base, conversational, and action-oriented agents
- an executor layer with FSM and signal bus components
- LLM provider abstractions for Anthropic and OpenAI-compatible APIs
- short-term, long-term, and contextual memory interfaces
- prompt templates for conversational, action, FSM, and G1-style reasoning
- tool and skill registries
- configuration loading from environment variables and YAML
- structured logging/event transport modules
- an Express + Socket.IO server entry point
- Vitest coverage for core action and signal bus behavior

## Installation

```bash
npm install
```

## Configuration

Copy the example environment file and fill in at least one provider key:

```bash
cp .env.example .env
```

Supported environment variables:

```text
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=
DEFAULT_MODEL=claude-sonnet-4-20250514
PORT=3000
LOG_LEVEL=info
```

Roy can also load YAML configuration from `roy.config.yaml` or `roy.config.yml`, with optional secrets from `roy.secrets.yaml` or `roy.secrets.yml`.

## Development

Run the development server:

```bash
npm run dev
```

Build the TypeScript project:

```bash
npm run build
```

Run tests:

```bash
npm test
```

The server exposes:

- `GET /` for project metadata
- `GET /health` for agent and session status
- Socket.IO `user_message` events for streaming agent responses

## Roadmap

- implement structured ToM/MIA diagnosis outputs
- add market scoring for candidate thinking investments
- connect FSM transitions to diagnosis and budget decisions
- add Evo-style mutation, crossover, and selection operators
- persist reusable agent, subteam, and diagnosis cache entries
- version prompt contracts and record prompt versions per run
- expand verification and backtracking tests

## License

No license has been declared yet.
