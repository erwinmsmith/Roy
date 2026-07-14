import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { WorkspaceMemoryManager } from '../src/core/memory/workspace.js';
import { ContextWindowManager } from '../src/core/context/index.js';

describe('ContextWindowManager', () => {
  it('loads public/private memory and compacts the latest configured turns', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-context-window-'));
    const memory = new WorkspaceMemoryManager();
    await memory.initWorkspace(cwd, 'context-session');
    for (let index = 1; index <= 4; index += 1) {
      await memory.appendConversation({
        sessionId: 'context-session',
        correlationId: `turn-${index}`,
        role: 'user',
        speaker: 'user',
        content: `User request ${index}`,
      });
      await memory.appendConversation({
        sessionId: 'context-session',
        correlationId: `turn-${index}`,
        role: 'assistant',
        speaker: 'Roy',
        content: `Roy response ${index}`,
      });
    }

    const manager = new ContextWindowManager(memory, {
      sessionWindowTurns: 2,
      maxContextTokens: 1200,
      includeToolResults: 'summary',
      includeSubagentReports: 'summary',
      includePrivateMemory: true,
      includePublicMemory: true,
    });
    const context = await manager.build({
      sessionId: 'context-session',
      agentId: 'agent_researcher_001',
      agentKey: 'researcher',
      role: 'subagent',
      task: 'Inspect the repository',
      parentContext: 'Roy delegated a bounded inspection task.',
      memoryScope: { public: true, private: true, parentContext: true, sessionWindowTurns: 2 },
    });

    expect(context.sessionContext).not.toContain('User request 1');
    expect(context.sessionContext).toContain('User request 3');
    expect(context.sessionContext).toContain('Roy response 4');
    expect(context.privateMemory).toContain('# Agent Memory');
    expect(context.parentContext).toContain('Roy delegated');
    expect(context.tokenUsage.total).toBeLessThanOrEqual(1200);
    expect(context.sources.session).toContain('last 2 turns');
  });
});
