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
});
