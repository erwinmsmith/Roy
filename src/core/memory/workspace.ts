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

export type MemoryMode = 'off' | 'suggest' | 'auto';

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

    await this.writeIfMissing(
      path.join(agentPath, 'prompt.md'),
      `# ${options.name ?? safeKey} Prompt

## System Prompt Guidance

<!-- ROY:BEGIN:system-prompt-guidance -->
Keep this agent identity separate from the model provider identity.
<!-- ROY:END:system-prompt-guidance -->

## Runtime Template

\`\`\`txt
You are {agent_name}, a {agent_role} agent in the Roy runtime.
Parent: {parent_agent}
Task: {task}

<agent_prompt_file path=".roy/agents/${safeKey}/prompt.md">
{agent_prompt_notes}
</agent_prompt_file>

<agent_context_file path=".roy/agents/${safeKey}/context.md">
{agent_context}
</agent_context_file>
\`\`\`
`
    );
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
    return mode === 'off' || mode === 'auto' || mode === 'suggest' ? mode : 'suggest';
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

  async listMemoryUpdates(): Promise<MemoryUpdateRecord[]> {
    const file = await this.readJson<{ updates?: MemoryUpdateRecord[] }>(
      path.join(this.rootPath, 'cache', 'memory-updates.json'),
      { updates: [] }
    );
    return file.updates ?? [];
  }

  async proposeMemoryUpdates(): Promise<MemoryUpdateProposal[]> {
    const mode = await this.getMemoryMode();
    if (mode === 'off') return [];

    const conversations = await this.readConversation(this.sessionId, 200);
    const existing = await this.listAllMemoryProposals();
    const created: MemoryUpdateProposal[] = [];
    const agentResults = conversations.filter(entry => entry.role === 'agent' && entry.metadata?.kind === 'agent.result');

    for (const entry of agentResults) {
      const metadata = entry.metadata ?? {};
      const archetype = typeof metadata.archetype === 'string'
        ? metadata.archetype
        : this.inferArchetype(String(metadata.agentId ?? entry.speaker));
      const toolCalls = Array.isArray(metadata.toolCalls) ? metadata.toolCalls.map(String) : [];
      const grounded = metadata.grounded === true;

      if (archetype && toolCalls.includes('fs.list') && !this.hasPendingProposal(existing, 'agent', archetype, 'tool-policy')) {
        created.push(this.createProposal({
          type: 'agent',
          key: archetype,
          path: path.join(this.rootPath, 'agents', this.safeKey(archetype), 'memory.md'),
          section: 'tool-policy',
          content: `- For project inspection tasks, ${this.capitalize(archetype)} agents should call \`fs.list\` before summarizing repository structure.`,
          reason: `${this.capitalize(archetype)} learned reusable project inspection behavior from a grounded run.`,
          confidence: 0.87,
          risk: 'low',
          source: { conversationEntryId: entry.id, agentId: metadata.agentId },
        }));
      }

      if (grounded && toolCalls.includes('fs.list') && !this.hasPendingProposal(existing, 'public', 'project', 'project-structure')) {
        created.push(this.createProposal({
          type: 'public',
          key: 'project',
          path: path.join(this.rootPath, 'public', 'project.md'),
          section: 'project-structure',
          content: `- Project structure was inspected by ${entry.speaker} using \`fs.list\`. Review the latest trace before treating this as permanent architecture documentation.`,
          reason: 'Grounded filesystem inspection found project structure context that may be useful in future sessions.',
          confidence: 0.74,
          risk: 'medium',
          source: { conversationEntryId: entry.id, agentId: metadata.agentId },
        }));
      }
    }

    if (created.length === 0) return [];

    const proposals = [...existing, ...created];
    await this.writeProposalFile(proposals);

    if (mode === 'auto') {
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
    const existing = patterns.find(pattern => pattern.key === key || pattern.id === `agent_pattern_${key}`);
    const pattern = {
      id: `agent_pattern_${key}`,
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
      id: `mem_prop_${now}_${Math.random().toString(16).slice(2, 8)}`,
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

  private hasPendingProposal(proposals: MemoryUpdateProposal[], type: string, key: string, section: string): boolean {
    return proposals.some(proposal =>
      proposal.status === 'pending'
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
