import { mkdir, readFile, readdir, stat, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';
import type { RuntimeEvent } from '../runtime/Runtime.js';

export interface MemoryDocState {
  name: string;
  path: string;
  size: number;
}

export interface PatternCacheState {
  agents: number;
  teams: number;
  delegations: number;
}

export interface WorkspaceMemoryState {
  rootPath: string;
  initialized: boolean;
  memoryDocs: MemoryDocState[];
  publicMemoryDocs: MemoryDocState[];
  agentMemories: Array<{
    id: string;
    path: string;
    docs: MemoryDocState[];
  }>;
  patterns: PatternCacheState;
  traces: number;
  queuePath: string;
}

export interface RootMemoryContext {
  rootMemory: string;
  projectMemory: string;
  constraints: string;
  decisions: string;
  glossary: string;
  agentPatterns: unknown[];
  teamPatterns: unknown[];
  delegationPatterns: unknown[];
}

export interface ConversationEntry {
  id: string;
  sessionId: string;
  turnId?: string;
  correlationId?: string;
  role: 'user' | 'assistant' | 'system' | 'agent' | 'tool';
  speaker: string;
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface ConversationSessionState {
  sessionId: string;
  path: string;
  entries: number;
  updatedAt: number;
}

export type MemoryMode = 'off' | 'suggest' | 'auto-safe';

export interface MemoryUpdateProposal {
  id: string;
  target: {
    type: 'public' | 'agent' | 'team' | 'pattern';
    key?: string;
    path: string;
    section?: string;
  };
  operation: 'append' | 'replace_section' | 'merge_json';
  content: string | Record<string, unknown>;
  reason: string;
  confidence: number;
  risk: 'low' | 'medium' | 'high';
  status: 'pending' | 'accepted' | 'rejected' | 'committed';
  createdAt: number;
  updatedAt: number;
  source?: Record<string, unknown>;
}

export interface MemoryUpdateRecord {
  id: string;
  proposalId: string;
  targetPath: string;
  section?: string;
  operation: string;
  committedAt: number;
}

export interface MemoryProposalSummary {
  createdThisRun: number;
  skippedDuplicates: number;
  updatedPendingProposals: number;
  pendingProposals: number;
  alreadyCommitted: number;
}

export interface AgentMemoryBundle {
  key: string;
  path: string;
  identity: string;
  memory: string;
  context: string;
  prompt: string;
}

export interface AgentPatternInput {
  key: string;
  name: string;
  archetype: string;
  tomLevel: number;
  description?: string;
  tools?: string[];
  skills?: string[];
}

export interface DelegationPatternInput {
  archetype: string;
  task: string;
  parentId: string;
  agentPatternId: string;
}

export interface MemorySignals {
  source: {
    sessionId: string;
    sessionPath: string;
    traceName?: string;
  };
  counts: {
    userCommands: number;
    agentResults: number;
    rootFinalResponses: number;
    groundedAgentResults: number;
  };
  toolCalls: string[];
  agents: Array<{
    agentId: string;
    archetype: string;
    parentId?: string;
    grounded: boolean;
    toolGrounded: boolean;
    outputGrounded: boolean;
    toolCalls: string[];
    evidence: {
      observedPaths: string[];
      toolResultSummary?: string;
    };
  }>;
  candidateSignals: string[];
}

interface PatternFile {
  patterns: unknown[];
}

const PUBLIC_MEMORY_TEMPLATES: Record<string, string> = {
  'project.md': `# Project Context

## Project Name

## Purpose

## Architecture

<!-- ROY:BEGIN:project-structure -->
<!-- ROY:END:project-structure -->

## Important Files

## Development Commands

## Known Issues

## Open Questions
`,
  'context.md': `# Public Context

This document stores shared runtime context visible to Roy, subagents, teams, and the user.

## Shared Facts

<!-- ROY:BEGIN:shared-facts -->
<!-- ROY:END:shared-facts -->

## Shared Constraints

<!-- ROY:BEGIN:shared-constraints -->
<!-- ROY:END:shared-constraints -->
`,
  'decisions.md': `# Design Decisions

## Accepted Decisions

<!-- ROY:BEGIN:accepted-decisions -->
<!-- ROY:END:accepted-decisions -->

## Rejected Alternatives

## Pending Decisions
`,
  'constraints.md': `# Constraints

## Engineering Constraints

<!-- ROY:BEGIN:engineering-constraints -->
<!-- ROY:END:engineering-constraints -->

## Budget Constraints

## Tool Constraints

## Safety / Reliability Constraints
`,
  'glossary.md': `# Glossary

## Terms
`,
  'user.md': `# User Context

## Preferences

<!-- ROY:BEGIN:user-preferences -->
<!-- ROY:END:user-preferences -->

## Recurring Goals

<!-- ROY:BEGIN:recurring-goals -->
<!-- ROY:END:recurring-goals -->
`,
};

const AGENT_MEMORY_TEMPLATES: Record<string, string> = {
  'memory.md': `# Agent Memory

## Stable Lessons

<!-- ROY:BEGIN:stable-lessons -->
<!-- ROY:END:stable-lessons -->

## Delegation Lessons

<!-- ROY:BEGIN:delegation-lessons -->
<!-- ROY:END:delegation-lessons -->

## Failure Cases

<!-- ROY:BEGIN:failure-cases -->
<!-- ROY:END:failure-cases -->

## Tool Policy

<!-- ROY:BEGIN:tool-policy -->
<!-- ROY:END:tool-policy -->
`,
  'context.md': `# Agent Context

## Current Role

## Reusable Context

<!-- ROY:BEGIN:reusable-context -->
<!-- ROY:END:reusable-context -->
`,
  'user.md': `# User Context

## Preferences

## Recurring Goals
`,
  'decisions.md': `# Design Decisions

## Accepted Decisions

## Rejected Alternatives

## Pending Decisions
`,
  'constraints.md': `# Constraints

## Engineering Constraints

## Budget Constraints

## Tool Constraints

## Safety / Reliability Constraints
`,
  'glossary.md': `# Glossary

## Terms
`,
};

export class WorkspaceMemoryManager {
  private rootPath = '';
  private tracePath = '';
  private sessionId = '';
  private initialized = false;

  async initWorkspace(cwd: string, sessionId: string): Promise<WorkspaceMemoryState> {
    this.rootPath = path.join(cwd, '.roy');
    const publicPath = path.join(this.rootPath, 'public');
    const agentsRoyPath = path.join(this.rootPath, 'agents', 'roy');
    const teamsPath = path.join(this.rootPath, 'teams');
    const tracesPath = path.join(this.rootPath, 'traces');
    const cachePath = path.join(this.rootPath, 'cache');
    const sessionsPath = path.join(this.rootPath, 'sessions');
    const queuePath = path.join(this.rootPath, 'queue');
    this.sessionId = sessionId;

    await Promise.all([
      mkdir(publicPath, { recursive: true }),
      mkdir(agentsRoyPath, { recursive: true }),
      mkdir(teamsPath, { recursive: true }),
      mkdir(tracesPath, { recursive: true }),
      mkdir(cachePath, { recursive: true }),
      mkdir(sessionsPath, { recursive: true }),
      mkdir(queuePath, { recursive: true }),
    ]);

    for (const [fileName, content] of Object.entries(PUBLIC_MEMORY_TEMPLATES)) {
      await this.writeIfMissing(path.join(publicPath, fileName), content);
    }

    await this.ensureAgentMemory('roy', {
      name: 'Roy',
      role: 'root',
      description: 'Root agent of the local autonomous agent runtime.',
    });

    await this.writeIfMissing(path.join(cachePath, 'agent-patterns.json'), JSON.stringify({ patterns: [] }, null, 2) + '\n');
    await this.writeIfMissing(path.join(cachePath, 'team-patterns.json'), JSON.stringify({ patterns: [] }, null, 2) + '\n');
    await this.writeIfMissing(path.join(cachePath, 'delegation-patterns.json'), JSON.stringify({ patterns: [] }, null, 2) + '\n');
    await this.writeIfMissing(path.join(cachePath, 'tool-results.json'), JSON.stringify({ results: [] }, null, 2) + '\n');
    await this.writeIfMissing(path.join(cachePath, 'memory-proposals.json'), JSON.stringify({ proposals: [] }, null, 2) + '\n');
    await this.writeIfMissing(path.join(cachePath, 'memory-updates.json'), JSON.stringify({ updates: [] }, null, 2) + '\n');
    await this.writeIfMissing(
      path.join(this.rootPath, 'config.json'),
      JSON.stringify({ version: 1, traceEvents: true, memoryUpdates: 'suggest' }, null, 2) + '\n'
    );
    await this.writeIfMissing(
      path.join(this.rootPath, 'index.json'),
      JSON.stringify({
        version: 1,
        public: 'public/',
        agents: 'agents/',
        teams: 'teams/',
        traces: 'traces/',
        cache: 'cache/',
        queue: 'queue/',
      }, null, 2) + '\n'
    );

    const traceFileName = `${this.safeTimestamp(new Date())}.${sessionId}.jsonl`;
    this.tracePath = path.join(tracesPath, traceFileName);
    this.initialized = true;

    return this.getState();
  }

  async getState(): Promise<WorkspaceMemoryState> {
    if (!this.initialized) {
      return {
        rootPath: this.rootPath,
        initialized: false,
        memoryDocs: [],
        publicMemoryDocs: [],
        agentMemories: [],
        patterns: { agents: 0, teams: 0, delegations: 0 },
        traces: 0,
        queuePath: this.rootPath ? path.join(this.rootPath, 'queue') : '',
      };
    }

    const publicMemoryDocs = await this.listDocs(path.join(this.rootPath, 'public'));
    const agentMemories = await this.listAgentMemories();
    const patterns = await this.getPatternState();
    const traces = await this.countFiles(path.join(this.rootPath, 'traces'), '.jsonl');

    return {
      rootPath: this.rootPath,
      initialized: true,
      memoryDocs: publicMemoryDocs,
      publicMemoryDocs,
      agentMemories,
      patterns,
      traces,
      queuePath: path.join(this.rootPath, 'queue'),
    };
  }

  async loadRootContext(): Promise<RootMemoryContext> {
    const publicPath = path.join(this.rootPath, 'public');
    const royPath = path.join(this.rootPath, 'agents', 'roy');
    return {
      rootMemory: await this.readOptional(path.join(royPath, 'memory.md')),
      projectMemory: await this.readOptional(path.join(publicPath, 'project.md')),
      constraints: await this.readOptional(path.join(publicPath, 'constraints.md')),
      decisions: await this.readOptional(path.join(publicPath, 'decisions.md')),
      glossary: await this.readOptional(path.join(publicPath, 'glossary.md')),
      agentPatterns: await this.readPatterns('agent-patterns.json'),
      teamPatterns: await this.readPatterns('team-patterns.json'),
      delegationPatterns: await this.readPatterns('delegation-patterns.json'),
    };
  }

  async writeTrace(event: RuntimeEvent): Promise<void> {
    if (!this.initialized || !this.tracePath) return;
    await appendFile(this.tracePath, JSON.stringify(event) + '\n', 'utf8');
  }

  async ensureAgentMemory(agentKey: string, options: { name?: string; role?: string; description?: string } = {}): Promise<void> {
    const safeKey = this.safeKey(agentKey);
    const agentPath = path.join(this.rootPath, 'agents', safeKey);
    await mkdir(agentPath, { recursive: true });

    await this.writeIfMissing(
      path.join(agentPath, 'identity.md'),
      `# ${options.name ?? safeKey} Agent

Role: ${options.role ?? safeKey}

${options.description ?? 'Reusable agent archetype memory.'}
`
    );

    for (const [fileName, content] of Object.entries(AGENT_MEMORY_TEMPLATES)) {
      await this.writeIfMissing(path.join(agentPath, fileName), content);
    }
    if (safeKey === 'researcher') {
      await this.ensureResearcherPolicy(path.join(agentPath, 'memory.md'));
    }

    await this.writeIfMissing(
      path.join(agentPath, 'prompt.md'),
      `# ${options.name ?? safeKey} Prompt

## System Prompt Guidance

<!-- ROY:BEGIN:system-prompt-guidance -->
Keep this agent identity separate from the model provider identity.
<!-- ROY:END:system-prompt-guidance -->

## Runtime Template

\`\`\`txt
{{agent_identity}}

{{tom_profile}}

{{public_context}}

{{agent_private_memory}}

{{available_skills}}

{{available_tools}}

{{parent_context}}

{{task}}
\`\`\`
`
    );
    await this.ensurePromptSlots(path.join(agentPath, 'prompt.md'));
    await this.writeIfMissing(
      path.join(agentPath, 'state.json'),
      JSON.stringify({ id: safeKey, name: options.name ?? safeKey, role: options.role ?? safeKey, updatedAt: null }, null, 2) + '\n'
    );
    await this.writeIfMissing(path.join(agentPath, 'sessions.jsonl'), '');
  }

  async loadAgentMemory(agentKey: string): Promise<AgentMemoryBundle> {
    const safeKey = this.safeKey(agentKey);
    await this.ensureAgentMemory(safeKey);
    const agentPath = path.join(this.rootPath, 'agents', safeKey);
    return {
      key: safeKey,
      path: agentPath,
      identity: await this.readOptional(path.join(agentPath, 'identity.md')),
      memory: await this.readOptional(path.join(agentPath, 'memory.md')),
      context: await this.readOptional(path.join(agentPath, 'context.md')),
      prompt: await this.readOptional(path.join(agentPath, 'prompt.md')),
    };
  }

  async readPublicDoc(name: string): Promise<string> {
    const fileName = name.endsWith('.md') ? name : `${name}.md`;
    return this.readOptional(path.join(this.rootPath, 'public', fileName));
  }

  async readAgentDoc(agentKey: string, doc = 'memory'): Promise<string> {
    const safeKey = this.safeKey(agentKey);
    const fileName = doc.endsWith('.md') ? doc : `${doc}.md`;
    return this.readOptional(path.join(this.rootPath, 'agents', safeKey, fileName));
  }

  async getMemoryMode(): Promise<MemoryMode> {
    const config = await this.readJson<Record<string, unknown>>(path.join(this.rootPath, 'config.json'), {});
    const mode = config.memoryUpdates;
    return mode === 'off' || mode === 'auto-safe' || mode === 'suggest' ? mode : 'suggest';
  }

  async setMemoryMode(mode: MemoryMode): Promise<MemoryMode> {
    const configPath = path.join(this.rootPath, 'config.json');
    const config = await this.readJson<Record<string, unknown>>(configPath, {});
    config.memoryUpdates = mode;
    await writeFile(configPath, JSON.stringify(config, null, 2) + '\n', 'utf8');
    return mode;
  }

  async listMemoryProposals(): Promise<MemoryUpdateProposal[]> {
    const file = await this.readJson<{ proposals?: MemoryUpdateProposal[] }>(
      path.join(this.rootPath, 'cache', 'memory-proposals.json'),
      { proposals: [] }
    );
    return (file.proposals ?? []).filter(proposal => proposal.status === 'pending');
  }

  async listAllMemoryProposalRecords(): Promise<MemoryUpdateProposal[]> {
    return this.listAllMemoryProposals();
  }

  async getMemoryProposal(id: string): Promise<MemoryUpdateProposal | undefined> {
    const proposals = await this.listAllMemoryProposals();
    return proposals.find(proposal => proposal.id === id);
  }

  async listMemoryUpdates(): Promise<MemoryUpdateRecord[]> {
    const file = await this.readJson<{ updates?: MemoryUpdateRecord[] }>(
      path.join(this.rootPath, 'cache', 'memory-updates.json'),
      { updates: [] }
    );
    return file.updates ?? [];
  }

  async summarizeMemoryUpdates(): Promise<MemoryProposalSummary> {
    const before = (await this.listAllMemoryProposals()).map(proposal => ({ ...proposal, target: { ...proposal.target }, source: { ...(proposal.source ?? {}) } }));
    const signals = await this.collectMemorySignals();
    const created = await this.proposeMemoryUpdates();
    const after = await this.listAllMemoryProposals();
    const committed = after.filter(proposal => proposal.status === 'committed').length;
    const pending = after.filter(proposal => proposal.status === 'pending').length;
    const duplicateCandidates = signals.candidateSignals.filter(signal => {
      const [key, sectionName] = this.signalToProposalTarget(signal);
      return key && sectionName && before.some(proposal =>
        (proposal.status === 'pending' || proposal.status === 'committed')
        && proposal.target.key === key
        && proposal.target.section === sectionName
      );
    });

    return {
      createdThisRun: created.length,
      skippedDuplicates: duplicateCandidates.length,
      updatedPendingProposals: after.filter(proposal => {
        const previous = before.find(item => item.id === proposal.id);
        return previous && proposal.status === 'pending' && proposal.updatedAt > previous.updatedAt;
      }).length,
      pendingProposals: pending,
      alreadyCommitted: committed,
    };
  }

  async collectMemorySignals(): Promise<MemorySignals> {
    const conversations = await this.readConversation(this.sessionId, 500);
    const traces = await this.listTraces();
    const agentResults = conversations.filter(entry => entry.role === 'agent' && entry.metadata?.kind === 'agent.result');
    const rootFinalResponses = conversations.filter(entry => entry.role === 'assistant' && entry.metadata?.kind === 'root.final_response');
    const commandEntries = conversations.filter(entry => entry.metadata?.command === 'spawn' || entry.role === 'user');
    const agents = agentResults.map(entry => {
      const metadata = entry.metadata ?? {};
      const evidence = this.normalizeEvidence(metadata.evidence);
      const archetype = typeof metadata.archetype === 'string'
        ? metadata.archetype
        : this.inferArchetype(String(metadata.agentId ?? entry.speaker));
      const toolCalls = Array.isArray(metadata.toolCalls) ? metadata.toolCalls.map(String) : [];
      const toolGrounded = evidence.toolGrounded || toolCalls.includes('fs.list');
      return {
        agentId: String(metadata.agentId ?? entry.speaker),
        archetype,
        parentId: typeof metadata.parentId === 'string' ? metadata.parentId : undefined,
        grounded: metadata.grounded === true,
        toolGrounded,
        outputGrounded: evidence.outputGrounded,
        toolCalls,
        evidence: {
          observedPaths: evidence.observedPaths,
          toolResultSummary: evidence.toolResultSummary,
        },
      };
    });
    const candidateSignals = new Set<string>();
    for (const agent of agents) {
      if (agent.archetype === 'researcher' && agent.toolGrounded) {
        candidateSignals.add('researcher.tool_policy');
      }
      if (agent.archetype === 'researcher' && agent.toolGrounded && !agent.outputGrounded) {
        candidateSignals.add('researcher.failure_case');
      }
      if (agent.archetype === 'researcher' && agent.outputGrounded) {
        candidateSignals.add('public.project_structure');
      }
    }
    for (const rootResponse of rootFinalResponses) {
      const subagentId = rootResponse.metadata?.subagentId;
      if (typeof subagentId === 'string' && agents.some(agent => agent.agentId === subagentId)) {
        candidateSignals.add('roy.delegation_lesson');
      }
    }

    return {
      source: {
        sessionId: this.sessionId,
        sessionPath: this.getConversationPath(this.sessionId),
        traceName: traces[0]?.name,
      },
      counts: {
        userCommands: commandEntries.length,
        agentResults: agentResults.length,
        rootFinalResponses: rootFinalResponses.length,
        groundedAgentResults: agents.filter(agent => agent.grounded).length,
      },
      toolCalls: [...new Set(agents.flatMap(agent => agent.toolCalls))],
      agents,
      candidateSignals: [...candidateSignals],
    };
  }

  async proposeMemoryUpdates(): Promise<MemoryUpdateProposal[]> {
    const mode = await this.getMemoryMode();
    if (mode === 'off') return [];

    const conversations = await this.readConversation(this.sessionId, 200);
    const signals = await this.collectMemorySignals();
    const existing = await this.listAllMemoryProposals();
    const created: MemoryUpdateProposal[] = [];
    const agentResults = conversations.filter(entry => entry.role === 'agent' && entry.metadata?.kind === 'agent.result');
    const rootResponses = conversations.filter(entry => entry.role === 'assistant' && entry.metadata?.kind === 'root.final_response');

    for (const entry of agentResults) {
      const metadata = entry.metadata ?? {};
      const archetype = typeof metadata.archetype === 'string'
        ? metadata.archetype
        : this.inferArchetype(String(metadata.agentId ?? entry.speaker));
      const toolCalls = Array.isArray(metadata.toolCalls) ? metadata.toolCalls.map(String) : [];
      const grounded = metadata.grounded === true;
      const evidence = this.normalizeEvidence(metadata.evidence);
      const toolGrounded = evidence.toolGrounded || toolCalls.includes('fs.list');
      const outputGrounded = evidence.outputGrounded;

      if (archetype && toolGrounded && !this.hasOpenProposal(existing, 'agent', archetype, 'tool-policy')) {
        created.push(this.createProposal({
          type: 'agent',
          key: archetype,
          path: path.join(this.rootPath, 'agents', this.safeKey(archetype), 'memory.md'),
          section: 'tool-policy',
          content: `- For project inspection tasks, call \`fs.list\` and include the resulting file tree or directory summary in the report.`,
          reason: `${this.capitalize(archetype)} used filesystem listing during project inspection, so this is reusable tool policy.`,
          confidence: 0.87,
          risk: 'low',
          source: { sessionId: this.sessionId, conversationEntryId: entry.id, agentId: metadata.agentId, signalCounts: signals.counts },
        }));
      }

      if (archetype === 'researcher' && toolGrounded && !outputGrounded && !this.hasOpenProposal(existing, 'agent', archetype, 'failure-cases')) {
        created.push(this.createProposal({
          type: 'agent',
          key: archetype,
          path: path.join(this.rootPath, 'agents', this.safeKey(archetype), 'memory.md'),
          section: 'failure-cases',
          content: `- For project inspection tasks, do not only diagnose reasoning traces. Include concrete filesystem observations from \`fs.list\`, such as top-level directories and relevant source subdirectories.`,
          reason: `${this.capitalize(archetype)} used fs.list but did not include concrete filesystem observations in the final report.`,
          confidence: 0.84,
          risk: 'low',
          source: { sessionId: this.sessionId, conversationEntryId: entry.id, agentId: metadata.agentId, signalCounts: signals.counts },
        }));
      }

      if (grounded && outputGrounded) {
        const updated = this.updatePendingProposal(existing, {
          type: 'public',
          key: 'project',
          section: 'project-structure',
          content: this.projectStructureProposalContent(evidence),
          confidence: 0.78,
          reason: 'A newer grounded run produced stronger concrete project structure evidence.',
          source: { sessionId: this.sessionId, conversationEntryId: entry.id, agentId: metadata.agentId, signalCounts: signals.counts, updatedFromSignal: true },
        });
        if (updated) continue;
      }

      if (grounded && outputGrounded && !this.hasOpenProposal(existing, 'public', 'project', 'project-structure')) {
        created.push(this.createProposal({
          type: 'public',
          key: 'project',
          path: path.join(this.rootPath, 'public', 'project.md'),
          section: 'project-structure',
          content: this.projectStructureProposalContent(evidence),
          reason: 'Grounded filesystem inspection produced concrete project structure observations.',
          confidence: 0.74,
          risk: 'medium',
          source: { sessionId: this.sessionId, conversationEntryId: entry.id, agentId: metadata.agentId, signalCounts: signals.counts },
        }));
      }
    }

    for (const response of rootResponses) {
      const subagentId = response.metadata?.subagentId;
      const matchedAgent = agentResults.find(entry => entry.metadata?.agentId === subagentId);
      if (!matchedAgent || this.hasOpenProposal(existing, 'agent', 'roy', 'delegation-lessons')) continue;
      created.push(this.createProposal({
        type: 'agent',
        key: 'roy',
        path: path.join(this.rootPath, 'agents', 'roy', 'memory.md'),
        section: 'delegation-lessons',
        content: `- When synthesizing subagent reports, Roy should check whether the subagent output contains concrete tool results, not only whether a tool was called.`,
        reason: 'Roy consumed a researcher result and had enough context to preserve a reusable delegation lesson.',
        confidence: 0.82,
        risk: 'low',
        source: { sessionId: this.sessionId, conversationEntryId: response.id, subagentId, signalCounts: signals.counts },
      }));
    }

    if (created.length === 0) {
      await this.writeProposalFile(existing);
      return [];
    }

    const proposals = [...existing, ...created];
    await this.writeProposalFile(proposals);

    if (mode === 'auto-safe') {
      for (const proposal of created.filter(item => item.risk === 'low')) {
        await this.acceptMemoryProposal(proposal.id);
      }
    }

    return created;
  }

  async acceptMemoryProposal(id: string): Promise<MemoryUpdateRecord | undefined> {
    const proposals = await this.listAllMemoryProposals();
    const proposal = proposals.find(item => item.id === id && item.status === 'pending');
    if (!proposal) return undefined;

    if (proposal.operation === 'append' && typeof proposal.content === 'string') {
      await this.appendToSection(proposal.target.path, proposal.target.section, proposal.content);
    }

    proposal.status = 'committed';
    proposal.updatedAt = Date.now();
    await this.writeProposalFile(proposals);

    const updates = await this.listMemoryUpdates();
    const record: MemoryUpdateRecord = {
      id: `mem_update_${Date.now()}`,
      proposalId: proposal.id,
      targetPath: proposal.target.path,
      section: proposal.target.section,
      operation: proposal.operation,
      committedAt: Date.now(),
    };
    await writeFile(
      path.join(this.rootPath, 'cache', 'memory-updates.json'),
      JSON.stringify({ updates: [...updates, record] }, null, 2) + '\n',
      'utf8'
    );
    return record;
  }

  async rejectMemoryProposal(id: string): Promise<boolean> {
    const proposals = await this.listAllMemoryProposals();
    const proposal = proposals.find(item => item.id === id && item.status === 'pending');
    if (!proposal) return false;
    proposal.status = 'rejected';
    proposal.updatedAt = Date.now();
    await this.writeProposalFile(proposals);
    return true;
  }

  async upsertAgentPattern(input: AgentPatternInput): Promise<void> {
    const key = this.safeKey(input.key);
    const cachePath = path.join(this.rootPath, 'cache', 'agent-patterns.json');
    const file = await this.readJson<{ patterns?: Array<Record<string, unknown>> }>(cachePath, { patterns: [] });
    const patterns = file.patterns ?? [];
    const now = new Date().toISOString();
    const existing = patterns.find(pattern => pattern.key === key || pattern.id === `agent_pattern_${key}` || pattern.id === `agent_pattern_${key}_v1`);
    const pattern = {
      id: `agent_pattern_${key}_v1`,
      key,
      name: input.name,
      archetype: input.archetype,
      tomLevel: input.tomLevel,
      description: input.description ?? '',
      promptPath: `.roy/agents/${key}/prompt.md`,
      memoryPath: `.roy/agents/${key}/memory.md`,
      contextPath: `.roy/agents/${key}/context.md`,
      tools: input.tools ?? [],
      skills: input.skills ?? [],
      usage: {
        count: Number((existing?.usage as Record<string, unknown> | undefined)?.count ?? 0) + 1,
        lastUsedAt: now,
      },
      updatedAt: now,
    };

    if (existing) {
      Object.assign(existing, pattern);
    } else {
      patterns.push(pattern);
    }
    await writeFile(cachePath, JSON.stringify({ patterns }, null, 2) + '\n', 'utf8');
  }

  async findAgentPattern(archetype: string): Promise<Record<string, unknown> | undefined> {
    const key = this.safeKey(archetype);
    const patterns = await this.readPatterns('agent-patterns.json') as Array<Record<string, unknown>>;
    return patterns.find(pattern => pattern.key === key || pattern.archetype === key);
  }

  async findDelegationPattern(archetype: string, task: string): Promise<Record<string, unknown> | undefined> {
    const signature = this.delegationSignature(archetype, task);
    const patterns = await this.readPatterns('delegation-patterns.json') as Array<Record<string, unknown>>;
    return patterns.find(pattern => pattern.signature === signature || pattern.id === `delegation_${signature}_v1`);
  }

  async upsertDelegationPattern(input: DelegationPatternInput): Promise<Record<string, unknown>> {
    const signature = this.delegationSignature(input.archetype, input.task);
    const cachePath = path.join(this.rootPath, 'cache', 'delegation-patterns.json');
    const file = await this.readJson<{ patterns?: Array<Record<string, unknown>> }>(cachePath, { patterns: [] });
    const patterns = file.patterns ?? [];
    const now = new Date().toISOString();
    const existing = patterns.find(pattern => pattern.signature === signature || pattern.id === `delegation_${signature}_v1`);
    const pattern = {
      id: `delegation_${signature}_v1`,
      signature,
      archetype: input.archetype,
      parentId: input.parentId,
      agentPatternId: input.agentPatternId,
      taskSignature: signature.replace(`${this.safeKey(input.archetype)}_`, ''),
      usage: {
        count: Number((existing?.usage as Record<string, unknown> | undefined)?.count ?? 0) + 1,
        lastUsedAt: now,
      },
      updatedAt: now,
    };

    if (existing) {
      Object.assign(existing, pattern);
    } else {
      patterns.push(pattern);
    }
    await writeFile(cachePath, JSON.stringify({ patterns }, null, 2) + '\n', 'utf8');
    return pattern;
  }

  async updateCacheUsageMetrics(patternIds: string[], metrics: { definitionTokensSaved?: number; renderedPromptTokens?: number }): Promise<void> {
    await Promise.all([
      this.updatePatternUsageMetrics('agent-patterns.json', patternIds, metrics),
      this.updatePatternUsageMetrics('delegation-patterns.json', patternIds, metrics),
    ]);
  }

  async listTraces(): Promise<Array<{ name: string; path: string; size: number; updatedAt: number }>> {
    const tracesPath = path.join(this.rootPath, 'traces');
    const files = await readdir(tracesPath);
    const traces = [];
    for (const file of files.filter(item => item.endsWith('.jsonl')).sort()) {
      const fullPath = path.join(tracesPath, file);
      const fileStat = await stat(fullPath);
      traces.push({ name: file, path: fullPath, size: fileStat.size, updatedAt: fileStat.mtimeMs });
    }
    return traces.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async readTrace(name: string, limit = 50): Promise<RuntimeEvent[]> {
    const traces = await this.listTraces();
    const selected = name === 'latest'
      ? traces[0]
      : traces.find(trace => trace.name === name);
    if (!selected) return [];

    const raw = await this.readOptional(selected.path);
    if (!raw.trim()) return [];
    const events = raw.trim().split('\n')
      .map(line => {
        try {
          return JSON.parse(line) as RuntimeEvent;
        } catch {
          return undefined;
        }
      })
      .filter((event): event is RuntimeEvent => event !== undefined);

    return limit > 0 ? events.slice(-limit) : events;
  }

  async appendConversation(entry: Omit<ConversationEntry, 'id' | 'timestamp'>): Promise<ConversationEntry> {
    if (!this.initialized) {
      throw new Error('Workspace memory is not initialized');
    }

    const fullEntry: ConversationEntry = {
      ...entry,
      id: `conv_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`,
      timestamp: Date.now(),
    };

    const sessionPath = this.getConversationPath(entry.sessionId);
    await appendFile(sessionPath, JSON.stringify(fullEntry) + '\n', 'utf8');
    await this.appendAgentSession(fullEntry);
    await writeFile(
      path.join(this.rootPath, 'sessions', 'latest.json'),
      JSON.stringify({
        sessionId: entry.sessionId,
        path: sessionPath,
        updatedAt: fullEntry.timestamp,
      }, null, 2) + '\n',
      'utf8'
    );

    return fullEntry;
  }

  async readConversation(sessionId = this.sessionId, limit = 50): Promise<ConversationEntry[]> {
    const raw = await this.readOptional(this.getConversationPath(sessionId));
    if (!raw.trim()) return [];

    const entries = raw.trim().split('\n')
      .map(line => {
        try {
          return JSON.parse(line) as ConversationEntry;
        } catch {
          return undefined;
        }
      })
      .filter((entry): entry is ConversationEntry => entry !== undefined);

    return limit > 0 ? entries.slice(-limit) : entries;
  }

  async listConversationSessions(): Promise<ConversationSessionState[]> {
    const sessionsPath = path.join(this.rootPath, 'sessions');
    const files = await readdir(sessionsPath);
    const sessions: ConversationSessionState[] = [];

    for (const file of files.filter(item => item.endsWith('.jsonl')).sort()) {
      const fullPath = path.join(sessionsPath, file);
      const fileStat = await stat(fullPath);
      const raw = await this.readOptional(fullPath);
      sessions.push({
        sessionId: file.replace(/\.jsonl$/, ''),
        path: fullPath,
        entries: raw.trim() ? raw.trim().split('\n').length : 0,
        updatedAt: fileStat.mtimeMs,
      });
    }

    return sessions.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async importConversation(filePath: string, sessionId = this.sessionId): Promise<{ imported: number; path: string }> {
    const raw = await readFile(filePath, 'utf8');
    const parsed = this.parseConversationImport(raw);
    let imported = 0;

    for (const entry of parsed) {
      if (!entry.content.trim()) continue;
      await this.appendConversation({
        sessionId,
        role: entry.role,
        speaker: entry.speaker,
        content: entry.content,
        turnId: entry.turnId,
        correlationId: entry.correlationId,
        metadata: {
          ...(entry.metadata ?? {}),
          importedFrom: filePath,
          importedAt: new Date().toISOString(),
        },
      });
      imported += 1;
    }

    return {
      imported,
      path: this.getConversationPath(sessionId),
    };
  }

  private async listDocs(directory: string): Promise<MemoryDocState[]> {
    const files = await readdir(directory);
    const docs: MemoryDocState[] = [];
    for (const file of files.filter(item => item.endsWith('.md')).sort()) {
      const fullPath = path.join(directory, file);
      const fileStat = await stat(fullPath);
      docs.push({ name: file, path: fullPath, size: fileStat.size });
    }
    return docs;
  }

  private async listAgentMemories(): Promise<WorkspaceMemoryState['agentMemories']> {
    const agentsPath = path.join(this.rootPath, 'agents');
    const entries = await readdir(agentsPath, { withFileTypes: true });
    const memories = [];
    for (const entry of entries.filter(item => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
      const agentPath = path.join(agentsPath, entry.name);
      memories.push({
        id: entry.name,
        path: agentPath,
        docs: await this.listDocs(agentPath),
      });
    }
    return memories;
  }

  private async listAllMemoryProposals(): Promise<MemoryUpdateProposal[]> {
    const file = await this.readJson<{ proposals?: MemoryUpdateProposal[] }>(
      path.join(this.rootPath, 'cache', 'memory-proposals.json'),
      { proposals: [] }
    );
    return file.proposals ?? [];
  }

  private async writeProposalFile(proposals: MemoryUpdateProposal[]): Promise<void> {
    await writeFile(
      path.join(this.rootPath, 'cache', 'memory-proposals.json'),
      JSON.stringify({ proposals }, null, 2) + '\n',
      'utf8'
    );
  }

  private updatePendingProposal(proposals: MemoryUpdateProposal[], input: {
    type: MemoryUpdateProposal['target']['type'];
    key: string;
    section: string;
    content: string;
    confidence: number;
    reason: string;
    source?: Record<string, unknown>;
  }): boolean {
    const proposal = proposals.find(item =>
      item.status === 'pending'
      && item.target.type === input.type
      && item.target.key === input.key
      && item.target.section === input.section
    );
    if (!proposal || typeof proposal.content !== 'string') return false;
    if (proposal.content === input.content && proposal.confidence >= input.confidence) return false;
    proposal.content = input.content;
    proposal.confidence = Math.max(proposal.confidence, input.confidence);
    proposal.reason = input.reason;
    proposal.updatedAt = Date.now();
    proposal.source = { ...(proposal.source ?? {}), ...(input.source ?? {}) };
    return true;
  }

  private createProposal(input: {
    type: MemoryUpdateProposal['target']['type'];
    key: string;
    path: string;
    section: string;
    content: string;
    reason: string;
    confidence: number;
    risk: MemoryUpdateProposal['risk'];
    source?: Record<string, unknown>;
  }): MemoryUpdateProposal {
    const now = Date.now();
    return {
      id: this.createTimestampId(now),
      target: {
        type: input.type,
        key: input.key,
        path: input.path,
        section: input.section,
      },
      operation: 'append',
      content: input.content,
      reason: input.reason,
      confidence: input.confidence,
      risk: input.risk,
      status: 'pending',
      createdAt: now,
      updatedAt: now,
      source: input.source,
    };
  }

  private hasOpenProposal(proposals: MemoryUpdateProposal[], type: string, key: string, section: string): boolean {
    return proposals.some(proposal =>
      (proposal.status === 'pending' || proposal.status === 'committed')
      && proposal.target.type === type
      && proposal.target.key === key
      && proposal.target.section === section
    );
  }

  private async appendToSection(filePath: string, section: string | undefined, content: string): Promise<void> {
    const existing = await this.readOptional(filePath);
    if (!section) {
      await appendFile(filePath, `\n${content}\n`, 'utf8');
      return;
    }

    const begin = `<!-- ROY:BEGIN:${section} -->`;
    const end = `<!-- ROY:END:${section} -->`;
    if (!existing.includes(begin) || !existing.includes(end)) {
      await appendFile(filePath, `\n\n## ${section}\n\n${content}\n`, 'utf8');
      return;
    }

    const escapedBegin = this.escapeRegExp(begin);
    const escapedEnd = this.escapeRegExp(end);
    const pattern = new RegExp(`${escapedBegin}([\\s\\S]*?)${escapedEnd}`);
    const updated = existing.replace(pattern, (_match, body: string) => {
      const nextBody = `${body.trim() ? `${body.trim()}\n` : ''}${content}`;
      return `${begin}\n${nextBody}\n${end}`;
    });
    await writeFile(filePath, updated, 'utf8');
  }

  private async getPatternState(): Promise<PatternCacheState> {
    return {
      agents: (await this.readPatterns('agent-patterns.json')).length,
      teams: (await this.readPatterns('team-patterns.json')).length,
      delegations: (await this.readPatterns('delegation-patterns.json')).length,
    };
  }

  private async readPatterns(fileName: string): Promise<unknown[]> {
    const raw = await this.readOptional(path.join(this.rootPath, 'cache', fileName));
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw) as PatternFile;
      return Array.isArray(parsed.patterns) ? parsed.patterns : [];
    } catch {
      return [];
    }
  }

  private async updatePatternUsageMetrics(
    fileName: string,
    patternIds: string[],
    metrics: { definitionTokensSaved?: number; renderedPromptTokens?: number }
  ): Promise<void> {
    if (patternIds.length === 0) return;
    const filePath = path.join(this.rootPath, 'cache', fileName);
    const file = await this.readJson<{ patterns?: Array<Record<string, unknown>> }>(filePath, { patterns: [] });
    let changed = false;
    const patterns = (file.patterns ?? []).map(pattern => {
      if (!patternIds.includes(String(pattern.id))) return pattern;
      const usage = typeof pattern.usage === 'object' && pattern.usage !== null
        ? pattern.usage as Record<string, unknown>
        : {};
      pattern.usage = {
        ...usage,
        definitionTokensSaved: Number(usage.definitionTokensSaved ?? 0) + Number(metrics.definitionTokensSaved ?? 0),
        lastRenderedPromptTokens: metrics.renderedPromptTokens ?? usage.lastRenderedPromptTokens,
      };
      changed = true;
      return pattern;
    });
    if (changed) {
      await writeFile(filePath, JSON.stringify({ patterns }, null, 2) + '\n', 'utf8');
    }
  }

  private async ensurePromptSlots(filePath: string): Promise<void> {
    const requiredSlots = [
      '{{public_context}}',
      '{{agent_private_memory}}',
      '{{agent_identity}}',
      '{{tom_profile}}',
      '{{available_skills}}',
      '{{available_tools}}',
      '{{parent_context}}',
      '{{task}}',
    ];
    const existing = await this.readOptional(filePath);
    const missing = requiredSlots.filter(slot => !existing.includes(slot));
    if (missing.length === 0) return;
    await appendFile(
      filePath,
      `\n\n## Runtime Slots\n\n\`\`\`txt\n${missing.join('\n\n')}\n\`\`\`\n`,
      'utf8'
    );
  }

  private async ensureResearcherPolicy(filePath: string): Promise<void> {
    const existing = await this.readOptional(filePath);
    const required = [
      '- For project inspection tasks, call `fs.list` and include concrete file or directory names observed from the tool result.',
      '- If the user asks to list files/directories, do not produce a reasoning-trace diagnosis, bottleneck analysis, or meta-evaluation.',
      '- The final report must contain concrete observed paths.',
    ];
    const missing = required.filter(line => !existing.includes(line));
    if (missing.length === 0) return;
    const begin = '<!-- ROY:BEGIN:tool-policy -->';
    const end = '<!-- ROY:END:tool-policy -->';
    if (!existing.includes(begin) || !existing.includes(end)) {
      await appendFile(filePath, `\n\n## Tool Policy\n\n${missing.join('\n')}\n`, 'utf8');
      return;
    }
    const pattern = new RegExp(`${this.escapeRegExp(begin)}([\\s\\S]*?)${this.escapeRegExp(end)}`);
    const updated = existing.replace(pattern, (_match, body: string) => {
      const nextBody = `${body.trim() ? `${body.trim()}\n` : ''}${missing.join('\n')}`;
      return `${begin}\n${nextBody}\n${end}`;
    });
    await writeFile(filePath, updated, 'utf8');
  }

  private async readJson<T>(filePath: string, fallback: T): Promise<T> {
    const raw = await this.readOptional(filePath);
    if (!raw.trim()) return fallback;
    try {
      return JSON.parse(raw) as T;
    } catch {
      return fallback;
    }
  }

  private async countFiles(directory: string, extension: string): Promise<number> {
    const files = await readdir(directory);
    return files.filter(file => file.endsWith(extension)).length;
  }

  private async readOptional(filePath: string): Promise<string> {
    try {
      return await readFile(filePath, 'utf8');
    } catch {
      return '';
    }
  }

  private getConversationPath(sessionId: string): string {
    return path.join(this.rootPath, 'sessions', `${sessionId}.jsonl`);
  }

  private async appendAgentSession(entry: ConversationEntry): Promise<void> {
    const key = this.agentSessionKey(entry);
    if (!key) return;
    await this.ensureAgentMemory(key);
    await appendFile(path.join(this.rootPath, 'agents', key, 'sessions.jsonl'), JSON.stringify(entry) + '\n', 'utf8');
  }

  private agentSessionKey(entry: ConversationEntry): string | undefined {
    if (entry.role === 'assistant') return 'roy';
    if (entry.role !== 'agent') return undefined;
    const metadataKey = entry.metadata?.archetype;
    if (typeof metadataKey === 'string') return this.safeKey(metadataKey);
    return this.inferArchetype(entry.speaker);
  }

  private inferArchetype(value: string): string {
    return this.safeKey(value.replace(/-\d+$/, '').replace(/^agent_/, '').replace(/_\d+$/, ''));
  }

  private containsConcreteFileObservations(content: string): boolean {
    return /(^|\n)\s*(-\s*)?(src\/|docs\/|tests\/|package\.json|tsconfig\.json|README\.md|\.\/src|src:|files?:)/i.test(content);
  }

  private normalizeEvidence(value: unknown): {
    toolGrounded: boolean;
    outputGrounded: boolean;
    observedPaths: string[];
    toolResultSummary?: string;
  } {
    if (!value || typeof value !== 'object') {
      return { toolGrounded: false, outputGrounded: false, observedPaths: [] };
    }
    const record = value as Record<string, unknown>;
    return {
      toolGrounded: record.toolGrounded === true,
      outputGrounded: record.outputGrounded === true,
      observedPaths: Array.isArray(record.observedPaths) ? record.observedPaths.map(String) : [],
      toolResultSummary: typeof record.toolResultSummary === 'string' ? record.toolResultSummary : undefined,
    };
  }

  private projectStructureProposalContent(evidence: {
    observedPaths: string[];
    toolResultSummary?: string;
  }): string {
    const source = evidence.toolResultSummary?.trim()
      ? evidence.toolResultSummary
      : evidence.observedPaths.join('\n');
    const lines = source.split('\n')
      .map(line => line.trim())
      .filter(line => /src\/|docs\/|tests\/|package\.json|tsconfig\.json|README\.md/i.test(line))
      .slice(0, 12);
    if (lines.length === 0) {
      return '- Project structure was inspected with `fs.list`; review the related trace before making this permanent.';
    }
    return lines.map(line => line.startsWith('-') ? line : `- ${line}`).join('\n');
  }

  private signalToProposalTarget(signal: string): [string | undefined, string | undefined] {
    switch (signal) {
      case 'researcher.tool_policy':
        return ['researcher', 'tool-policy'];
      case 'researcher.failure_case':
        return ['researcher', 'failure-cases'];
      case 'roy.delegation_lesson':
        return ['roy', 'delegation-lessons'];
      case 'public.project_structure':
        return ['project', 'project-structure'];
      default:
        return [undefined, undefined];
    }
  }

  private createTimestampId(timestamp: number): string {
    const stamp = new Date(timestamp).toISOString().replace(/[-:.TZ]/g, '').slice(0, 17);
    const suffix = Math.random().toString(16).slice(2, 6);
    return `mem_prop_${stamp}_${suffix}`;
  }

  private delegationSignature(archetype: string, task: string): string {
    const taskKind = /\b(project|repo|repository|codebase|structure|files?|directories|inspect|list)\b/i.test(task)
      ? 'project_inspection'
      : this.safeKey(task).split('-').slice(0, 4).join('_') || 'task';
    return `${taskKind}_${this.safeKey(archetype)}`;
  }

  private parseConversationImport(raw: string): Array<Omit<ConversationEntry, 'id' | 'timestamp' | 'sessionId'>> {
    const trimmed = raw.trim();
    if (!trimmed) return [];

    const source = trimmed.startsWith('[')
      ? JSON.parse(trimmed) as unknown[]
      : trimmed.split('\n').map(line => JSON.parse(line) as unknown);

    return source
      .map(item => this.normalizeImportedConversationEntry(item))
      .filter((entry): entry is Omit<ConversationEntry, 'id' | 'timestamp' | 'sessionId'> => entry !== undefined);
  }

  private normalizeImportedConversationEntry(item: unknown): Omit<ConversationEntry, 'id' | 'timestamp' | 'sessionId'> | undefined {
    if (!item || typeof item !== 'object') return undefined;
    const record = item as Record<string, unknown>;
    const content = typeof record.content === 'string'
      ? record.content
      : typeof record.text === 'string'
        ? record.text
        : typeof record.message === 'string'
          ? record.message
          : '';
    if (!content.trim()) return undefined;

    const role = this.normalizeRole(record.role);
    const speaker = typeof record.speaker === 'string'
      ? record.speaker
      : role === 'assistant'
        ? 'assistant'
        : role;

    return {
      role,
      speaker,
      content,
      turnId: typeof record.turnId === 'string' ? record.turnId : undefined,
      correlationId: typeof record.correlationId === 'string' ? record.correlationId : undefined,
      metadata: typeof record.metadata === 'object' && record.metadata !== null
        ? record.metadata as Record<string, unknown>
        : undefined,
    };
  }

  private normalizeRole(role: unknown): ConversationEntry['role'] {
    if (role === 'assistant' || role === 'system' || role === 'agent' || role === 'tool') {
      return role;
    }
    return 'user';
  }

  private async writeIfMissing(filePath: string, content: string): Promise<void> {
    try {
      await stat(filePath);
    } catch {
      await writeFile(filePath, content, 'utf8');
    }
  }

  private safeTimestamp(date: Date): string {
    return date.toISOString().replace(/:/g, '-');
  }

  private safeKey(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'agent';
  }

  private capitalize(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  private escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

export default WorkspaceMemoryManager;
