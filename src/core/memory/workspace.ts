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

interface PatternFile {
  patterns: unknown[];
}

const MEMORY_TEMPLATES: Record<string, string> = {
  'root.md': `# Roy Root Memory

This document stores persistent context for Roy, the root agent.

## Current Project

<!-- Roy may summarize the project here. -->

## Long-term Goals

<!-- Persistent goals go here. -->

## Important Constraints

<!-- Constraints that should affect future reasoning. -->

## Recent Stable Decisions

<!-- Decisions that should be reused across sessions. -->
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
  private initialized = false;

  async initWorkspace(cwd: string, sessionId: string): Promise<WorkspaceMemoryState> {
    this.rootPath = path.join(cwd, '.roy');
    const memoryPath = path.join(this.rootPath, 'memory');
    const agentsRootPath = path.join(this.rootPath, 'agents', 'root');
    const teamsPath = path.join(this.rootPath, 'teams');
    const tracesPath = path.join(this.rootPath, 'traces');
    const cachePath = path.join(this.rootPath, 'cache');
    const sessionsPath = path.join(this.rootPath, 'sessions');
    const queuePath = path.join(this.rootPath, 'queue');

    await Promise.all([
      mkdir(memoryPath, { recursive: true }),
      mkdir(agentsRootPath, { recursive: true }),
      mkdir(teamsPath, { recursive: true }),
      mkdir(tracesPath, { recursive: true }),
      mkdir(cachePath, { recursive: true }),
      mkdir(sessionsPath, { recursive: true }),
      mkdir(queuePath, { recursive: true }),
    ]);

    for (const [fileName, content] of Object.entries(MEMORY_TEMPLATES)) {
      await this.writeIfMissing(path.join(memoryPath, fileName), content);
    }

    await this.writeIfMissing(
      path.join(agentsRootPath, 'identity.md'),
      `# Roy Root Agent

Roy is the root agent of the local autonomous agent runtime.
`
    );
    await this.writeIfMissing(
      path.join(agentsRootPath, 'memory.md'),
      `# Roy Agent Memory

<!-- ROY:BEGIN:auto-memory -->
<!-- ROY:END:auto-memory -->
`
    );
    await this.writeIfMissing(
      path.join(agentsRootPath, 'state.json'),
      JSON.stringify({ id: 'root', name: 'Roy', role: 'root', updatedAt: null }, null, 2) + '\n'
    );

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
        patterns: { agents: 0, teams: 0, delegations: 0 },
        traces: 0,
        queuePath: this.rootPath ? path.join(this.rootPath, 'queue') : '',
      };
    }

    const memoryDocs = await this.listMemoryDocs();
    const patterns = await this.getPatternState();
    const traces = await this.countFiles(path.join(this.rootPath, 'traces'), '.jsonl');

    return {
      rootPath: this.rootPath,
      initialized: true,
      memoryDocs,
      patterns,
      traces,
      queuePath: path.join(this.rootPath, 'queue'),
    };
  }

  async loadRootContext(): Promise<RootMemoryContext> {
    const memoryPath = path.join(this.rootPath, 'memory');
    return {
      rootMemory: await this.readOptional(path.join(memoryPath, 'root.md')),
      projectMemory: await this.readOptional(path.join(memoryPath, 'project.md')),
      constraints: await this.readOptional(path.join(memoryPath, 'constraints.md')),
      decisions: await this.readOptional(path.join(memoryPath, 'decisions.md')),
      glossary: await this.readOptional(path.join(memoryPath, 'glossary.md')),
      agentPatterns: await this.readPatterns('agent-patterns.json'),
      teamPatterns: await this.readPatterns('team-patterns.json'),
      delegationPatterns: await this.readPatterns('delegation-patterns.json'),
    };
  }

  async writeTrace(event: RuntimeEvent): Promise<void> {
    if (!this.initialized || !this.tracePath) return;
    await appendFile(this.tracePath, JSON.stringify(event) + '\n', 'utf8');
  }

  private async listMemoryDocs(): Promise<MemoryDocState[]> {
    const memoryPath = path.join(this.rootPath, 'memory');
    const files = await readdir(memoryPath);
    const docs: MemoryDocState[] = [];
    for (const file of files.filter(item => item.endsWith('.md')).sort()) {
      const fullPath = path.join(memoryPath, file);
      const fileStat = await stat(fullPath);
      docs.push({ name: file, path: fullPath, size: fileStat.size });
    }
    return docs;
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
}

export default WorkspaceMemoryManager;

