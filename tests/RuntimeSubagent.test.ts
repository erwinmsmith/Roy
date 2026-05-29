import { describe, it, expect } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type { LLMProvider, LLMMessage, LLMCompletionOptions, LLMCompletionResult, LLMStreamChunk } from '../src/core/llm/types.js';

class EchoLLM implements LLMProvider {
  readonly name = 'echo-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'echo complete',
      usage: {
        promptTokens: 5,
        completionTokens: 3,
        totalTokens: 8,
      },
    };
  }

  async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'subagent ', done: false };
    yield {
      content: 'result',
      done: true,
      usage: {
        promptTokens: 7,
        completionTokens: 2,
        totalTokens: 9,
      },
    };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Runtime controlled subagent spawning', () => {
  it('spawns, registers, runs, and tracks a subagent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-subagent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'subagent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const spawned = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 2,
      description: 'Inspect runtime state',
      task: 'Inspect runtime state',
      budgetTokens: 8000,
    });

    expect(spawned.identity.id).toBe('agent_researcher_001');
    expect(spawned.identity.parentId).toBe('root');

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].agent.identity.id).toBe(spawned.identity.id);

    const result = await runtime.runAgent(spawned.identity.id, 'Check token accounting');
    expect(result.result).toBe('subagent result');
    expect(result.usage.totalTokens).toBe(9);

    const budget = runtime.getBudgetState();
    expect(budget.usedTokens).toBe(9);
    expect(budget.perAgent[spawned.identity.id].totalTokens).toBe(9);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.spawned');
    expect(eventTypes).toContain('budget.allocated');
    expect(eventTypes).toContain('agent.run.completed');

    await runtime.shutdown();
  });

  it('runs controlled spawn through root-mediated messages and synthesis', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-mediated-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'mediated-spawn-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });

    expect(result.correlationId).toMatch(/^del_/);
    expect(result.agent.identity.tomProfile.level).toBe(0);
    expect(result.subagentResult.grounded).toBe(true);
    expect(result.subagentResult.toolCalls.map(call => call.toolName)).toContain('fs.list');
    expect(result.finalResponse).toBe('subagent result');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toEqual([
      'user.command.spawn',
      'agent.task',
      'tool.call',
      'tool.result',
      'agent.result',
      'root.synthesis',
      'root.final_response',
    ]);

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBe(9);
    expect(budget.perAgent[result.agent.identity.id].totalTokens).toBe(9);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');
    expect(eventTypes).toContain('agent.result.sent');
    expect(eventTypes).toContain('memory.pattern.updated');
    expect((await runtime.getConversation(undefined, 20)).some(entry => entry.role === 'agent')).toBe(true);
    const memoryState = await runtime.getMemoryState();
    expect(memoryState.agentMemories.map(memory => memory.id)).toContain('researcher');
    expect(memoryState.patterns.agents).toBe(1);
    expect((await runtime.listMemoryProposals()).length).toBeGreaterThan(0);

    await runtime.shutdown();
  });
});
