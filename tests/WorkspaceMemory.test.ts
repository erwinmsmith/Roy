import { describe, it, expect } from 'vitest';
import { mkdtemp, readFile, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';

describe('Workspace memory initialization', () => {
  it('creates .roy memory, cache, queue, and trace files', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-memory-'));
    const runtime = new Runtime();

    await runtime.initialize({
      sessionId: 'memory-test',
      fsmEnabled: false,
      workspaceCwd,
    });

    const state = await runtime.getMemoryState();
    expect(state.initialized).toBe(true);
    expect(state.rootPath).toBe(path.join(workspaceCwd, '.roy'));
    expect(state.memoryDocs.map(doc => doc.name)).toContain('root.md');
    expect(state.memoryDocs.map(doc => doc.name)).toContain('project.md');
    expect(state.patterns).toEqual({ agents: 0, teams: 0, delegations: 0 });
    expect(state.queuePath).toBe(path.join(workspaceCwd, '.roy', 'queue'));

    const context = await runtime.loadRootMemoryContext();
    expect(context.rootMemory).toContain('# Roy Root Memory');
    expect(context.projectMemory).toContain('# Project Context');

    runtime.emit({ type: 'turn.started', agentId: 'root', data: { turnId: 'turn_test' } });
    await new Promise(resolve => setTimeout(resolve, 10));

    const traceFiles = await readdir(path.join(workspaceCwd, '.roy', 'traces'));
    expect(traceFiles.some(file => file.endsWith('.memory-test.jsonl'))).toBe(true);

    const traceFile = traceFiles.find(file => file.endsWith('.memory-test.jsonl'))!;
    const traceContent = await readFile(path.join(workspaceCwd, '.roy', 'traces', traceFile), 'utf8');
    expect(traceContent).toContain('turn.started');

    await runtime.shutdown();
  });
});

