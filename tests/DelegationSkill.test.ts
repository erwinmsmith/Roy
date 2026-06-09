import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { skillRegistry } from '../src/core/skills/index.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class DelegationLLM implements LLMProvider {
  readonly name = 'delegation-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
    };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    const content = text.includes('Produce the final response to the user as Roy')
      ? 'Roy synthesized the researcher result.'
      : 'Observed project paths:\n- README.md\n- package.json\n- src/\n- tests/';
    yield {
      content,
      done: true,
      usage: { promptTokens: 12, completionTokens: 5, totalTokens: 17 },
    };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('delegate_to_subagent skill', () => {
  it('delegates through the runtime message-mediated subagent flow', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-skill-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-skill-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const output = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: {
          archetype: 'researcher',
          task: 'Inspect the project structure',
          parentId: 'root',
          requireRootSynthesis: true,
        },
      },
      {
        agentId: 'root',
        sessionId: 'delegation-skill-test',
        variables: {},
      }
    );

    expect(output.success).toBe(true);
    const result = output.result as Record<string, any>;
    expect(result.correlationId).toMatch(/^del_/);
    expect(result.agentId).toBe('agent_researcher_001');
    expect(result.agentName).toBe('Researcher-1');
    expect(result.agentResult.toolCalls.map((call: any) => call.toolName)).toContain('fs.list');
    expect(result.rootSynthesis).toBe('Roy synthesized the researcher result.');
    expect(result.tokenUsage.root.totalTokens).toBeGreaterThan(0);
    expect(result.tokenUsage.subagent.totalTokens).toBeGreaterThan(0);
    expect(result.events.map((event: any) => event.type)).toContain('agent.spawned');
    expect(result.events.map((event: any) => event.type)).toContain('agent.run.completed');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toEqual([
      'user.command.spawn',
      'agent.create.request',
      'agent.create.approved',
      'agent.task',
      'tool.call',
      'tool.result',
      'agent.result',
      'root.synthesis',
      'root.final_response',
    ]);

    await runtime.shutdown();
  });

  it('validates required delegation inputs', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-skill-invalid-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-skill-invalid',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const output = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: { archetype: 'unknown' },
      },
      {
        agentId: 'root',
        sessionId: 'delegation-skill-invalid',
        variables: {},
      }
    );

    expect(output.success).toBe(false);
    expect(output.error).toContain('Validation failed');

    await runtime.shutdown();
  });
});
