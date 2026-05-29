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

interface PatternFile {
  patterns: unknown[];
}

const MEMORY_TEMPLATES: Record<string, string> = {
  'public.md': `# Public Workspace Memory

This document stores shared context visible to Roy, subagents, teams, and the user.

## Shared Facts

## Shared Constraints
`,
  'project.md': `# Project Context

## Project Name

## Purpose

## Architecture

## Important Files

## Development Commands

## Known Issues

## Open Questions
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
    const memoryPath = path.join(this.rootPath, 'memory');
    const agentsRoyPath = path.join(this.rootPath, 'agents', 'roy');
    const teamsPath = path.join(this.rootPath, 'teams');
    const tracesPath = path.join(this.rootPath, 'traces');
    const cachePath = path.join(this.rootPath, 'cache');
    const sessionsPath = path.join(this.rootPath, 'sessions');
    const queuePath = path.join(this.rootPath, 'queue');
    this.sessionId = sessionId;

    await Promise.all([
      mkdir(memoryPath, { recursive: true }),
      mkdir(agentsRoyPath, { recursive: true }),
      mkdir(teamsPath, { recursive: true }),
      mkdir(tracesPath, { recursive: true }),
      mkdir(cachePath, { recursive: true }),
      mkdir(sessionsPath, { recursive: true }),
      mkdir(queuePath, { recursive: true }),
    ]);

    for (const [fileName, content] of Object.entries(MEMORY_TEMPLATES)) {
      await this.writeIfMissing(path.join(memoryPath, fileName), content);
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
    await this.writeIfMissing(
      path.join(this.rootPath, 'config.json'),
      JSON.stringify({ version: 1, traceEvents: true, memoryUpdates: 'manual' }, null, 2) + '\n'
    );
    await this.writeIfMissing(
      path.join(this.rootPath, 'index.json'),
      JSON.stringify({
        version: 1,
        memory: 'memory/',
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

    const publicMemoryDocs = await this.listDocs(path.join(this.rootPath, 'memory'));
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
    const memoryPath = path.join(this.rootPath, 'memory');
    const royPath = path.join(this.rootPath, 'agents', 'roy');
    return {
      rootMemory: await this.readOptional(path.join(royPath, 'memory.md')),
      projectMemory: await this.readOptional(path.join(memoryPath, 'project.md')),
      constraints: await this.readOptional(path.join(royPath, 'constraints.md')),
      decisions: await this.readOptional(path.join(royPath, 'decisions.md')),
      glossary: await this.readOptional(path.join(royPath, 'glossary.md')),
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
      `# ${options.name ?? safeKey} Prompt Notes

## System Prompt Guidance
`
    );
    await this.writeIfMissing(
      path.join(agentPath, 'state.json'),
      JSON.stringify({ id: safeKey, name: options.name ?? safeKey, role: options.role ?? safeKey, updatedAt: null }, null, 2) + '\n'
    );
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
}

export default WorkspaceMemoryManager;
