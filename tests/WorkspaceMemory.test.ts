import { describe, it, expect } from 'vitest';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
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
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('project.md');
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('context.md');
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('decisions.md');
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('constraints.md');
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('glossary.md');
    expect(state.publicMemoryDocs.map(doc => doc.name)).toContain('user.md');
    expect(state.agentMemories.map(memory => memory.id)).toContain('roy');
    expect(state.agentMemories.find(memory => memory.id === 'roy')?.docs.map(doc => doc.name)).toContain('memory.md');
    expect(state.agentMemories.find(memory => memory.id === 'roy')?.docs.map(doc => doc.name)).toContain('prompt.md');
    expect(state.patterns).toEqual({ agents: 0, teams: 0, delegations: 0 });
    expect(state.queuePath).toBe(path.join(workspaceCwd, '.roy', 'queue'));

    const context = await runtime.loadRootMemoryContext();
    expect(context.rootMemory).toContain('# Agent Memory');
    expect(context.projectMemory).toContain('# Project Context');
    expect(context.projectMemory).toContain(path.basename(workspaceCwd));
    expect(context.projectMemory).toContain(workspaceCwd);
    expect(await runtime.readPublicMemoryDoc('context')).toContain('# Public Context');
    expect(await runtime.readAgentMemoryDoc('roy', 'prompt')).toContain('# Roy Prompt');
    const rootState = JSON.parse(await readFile(path.join(workspaceCwd, '.roy', 'agents', 'roy', 'state.json'), 'utf8'));
    expect(rootState).toMatchObject({
      version: 1,
      id: 'roy',
      name: 'Roy',
      role: 'root',
      status: 'available',
      memoryPath: '.roy/agents/roy/memory.md',
      promptPath: '.roy/agents/roy/prompt.md',
    });
    expect(rootState.updatedAt).toEqual(expect.any(String));
    const workspaceConfig = JSON.parse(await readFile(path.join(workspaceCwd, '.roy', 'config.json'), 'utf8'));
    expect(workspaceConfig.delegation).toMatchObject({
      enabled: true,
      mode: 'auto',
      maxChildrenPerParent: 5,
      maxDepth: 3,
      maxTotalAgentsPerTurn: 10,
      allowCustomAgents: true,
      budgetAware: true,
    });
    expect(workspaceConfig.agents.defaultToolsByArchetype.researcher).toEqual(['fs.list', 'fs.read']);
    expect(workspaceConfig.agents.defaultSkillsByArchetype.researcher).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);

    runtime.emit({ type: 'turn.started', agentId: 'root', data: { turnId: 'turn_test' } });
    await new Promise(resolve => setTimeout(resolve, 10));

    const traceFiles = await readdir(path.join(workspaceCwd, '.roy', 'traces'));
    expect(traceFiles.some(file => file.endsWith('.memory-test.jsonl'))).toBe(true);

    const traceFile = traceFiles.find(file => file.endsWith('.memory-test.jsonl'))!;
    const traceContent = await readFile(path.join(workspaceCwd, '.roy', 'traces', traceFile), 'utf8');
    expect(traceContent).toContain('turn.started');
    expect((await runtime.listTraces()).length).toBe(1);
    expect((await runtime.readTrace('latest')).map(event => event.type)).toContain('turn.started');

    await runtime.shutdown();
  });

  it('persists and imports conversation entries', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-conversation-'));
    const runtime = new Runtime();

    await runtime.initialize({
      sessionId: 'conversation-test',
      fsmEnabled: false,
      workspaceCwd,
    });

    await runtime.recordConversation({
      role: 'user',
      speaker: 'user',
      content: 'hello Roy',
    });
    await runtime.recordConversation({
      role: 'assistant',
      speaker: 'Roy',
      content: 'hello user',
    });

    const entries = await runtime.getConversation(undefined, 10);
    expect(entries.map(entry => entry.content)).toEqual(['hello Roy', 'hello user']);

    const importPath = path.join(workspaceCwd, 'import.jsonl');
    await writeFile(importPath, [
      JSON.stringify({ role: 'user', speaker: 'imported-user', content: 'old question' }),
      JSON.stringify({ role: 'assistant', speaker: 'imported-assistant', content: 'old answer' }),
    ].join('\n'), 'utf8');

    const imported = await runtime.importConversation(importPath);
    expect(imported.imported).toBe(2);

    const sessions = await runtime.listConversationSessions();
    expect(sessions.map(session => session.sessionId)).toContain('conversation-test');
    expect(sessions.find(session => session.sessionId === 'conversation-test')?.entries).toBe(4);

    const afterImport = await runtime.getConversation(undefined, 10);
    expect(afterImport.map(entry => entry.content)).toContain('old question');
    expect(afterImport.map(entry => entry.content)).toContain('old answer');

    const latest = await readFile(path.join(workspaceCwd, '.roy', 'sessions', 'latest.json'), 'utf8');
    expect(latest).toContain('conversation-test');

    await runtime.shutdown();
  });

  it('loads delegation and archetype defaults from workspace config', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-config-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        version: 1,
        memoryUpdates: 'suggest',
        delegation: {
          maxChildrenPerParent: 2,
          maxDepth: 2,
          allowCustomAgents: false,
          budgetAware: false,
        },
        agents: {
          defaultToolsByArchetype: {
            researcher: ['fs.read'],
          },
          defaultSkillsByArchetype: {
            researcher: ['use_tool_when_needed'],
          },
        },
      }, null, 2) + '\n',
      'utf8'
    );
    const runtime = new Runtime();

    await runtime.initialize({
      sessionId: 'workspace-config-test',
      fsmEnabled: false,
      workspaceCwd,
    });

    const rootPolicy = runtime.getAgentPolicy('root');
    const researcher = runtime.getAgentArchetypeProfiles().find(profile => profile.archetype === 'researcher');
    expect(rootPolicy?.spawnPolicy.maxChildren).toBe(2);
    expect(rootPolicy?.spawnPolicy.maxDepth).toBe(2);
    expect(rootPolicy?.spawnPolicy.allowCustomAgents).toBe(false);
    expect(rootPolicy?.spawnPolicy.budgetAware).toBe(false);
    expect(researcher?.tools.map(tool => tool.name)).toEqual(['fs.read']);
    expect(researcher?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed']);

    await runtime.shutdown();
  });
});
