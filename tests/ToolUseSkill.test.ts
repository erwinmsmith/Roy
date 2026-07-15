import { describe, expect, it, beforeEach } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { UnifiedAgent } from '../src/core/agent/UnifiedAgent.js';
import Runtime from '../src/core/runtime/Runtime.js';
import { actionRegistry } from '../src/core/actions/index.js';
import { MessageQueue } from '../src/core/message/MessageQueue.js';
import { skillRegistry, UseToolWhenNeededSkill } from '../src/core/skills/index.js';
import { registerCoreTools, toolRegistry } from '../src/core/tools/index.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class ToolPlanningLLM implements LLMProvider {
  readonly name = 'tool-planning-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'unused' };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const result = messages.findLast(message => message.content.includes('Capability result:'))?.content ?? '';
    yield { content: `Synthesized tool evidence:\n${result}`, done: true };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return {
      action: 'use_tool_when_needed',
      params: {
        needed: true,
        toolName: 'shell.exec',
        params: { command: 'pwd' },
        reason: 'Need the current workspace path.',
      },
      reasoning: 'Use the controlled shell tool.',
    } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('use_tool_when_needed skill', () => {
  beforeEach(() => {
    actionRegistry.clear();
    toolRegistry.clear();
    skillRegistry.clear();
    registerCoreTools();
    skillRegistry.register(new UseToolWhenNeededSkill());
  });

  it('skips tool execution when no tool is needed', async () => {
    const result = await skillRegistry.execute(
      'use_tool_when_needed',
      {
        action: 'use_tool_when_needed',
        params: { needed: false, reason: 'Direct reasoning is enough.' },
      },
      { agentId: 'root', sessionId: 'tool-skill-test', variables: {} }
    );

    expect(result.success).toBe(true);
    expect(result.result).toMatchObject({ skipped: true });
    expect(result.metadata?.toolUsed).toBe(false);
  });

  it('executes an allowlisted tool when needed', async () => {
    const result = await skillRegistry.execute(
      'use_tool_when_needed',
      {
        action: 'use_tool_when_needed',
        params: {
          needed: true,
          toolName: 'shell.exec',
          params: { command: 'pwd' },
          reason: 'Need workspace path.',
        },
      },
      { agentId: 'root', sessionId: 'tool-skill-test', variables: {} }
    );

    expect(result.success).toBe(true);
    expect(result.metadata?.toolUsed).toBe(true);
    expect(result.metadata?.toolName).toBe('shell.exec');
    expect((result.result as { stdout: string }).stdout.trim()).toBe(process.cwd());
  });

  it('can be selected autonomously by UnifiedAgent planner', async () => {
    const agent = new UnifiedAgent({
      name: 'agent',
      goal: 'test',
      llm: new ToolPlanningLLM(),
      mode: 'hybrid',
    });
    const queue = new MessageQueue(['env', 'agent']);
    agent.setMessageQueue(queue);
    await agent.initialize('tool-planning-session');

    await agent.step('check the current workspace status');

    const output = await queue.receive('env');
    expect(String(output?.content)).toContain('stdout');
    expect(String(output?.content)).toContain(process.cwd());
  });

  it('is registered for Runtime agents', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-tool-skill-runtime-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'tool-skill-runtime-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: undefined,
    });

    expect(skillRegistry.has('use_tool_when_needed')).toBe(true);
    expect(runtime.getContext().agent.getCapabilities().tools).toContain('shell.exec');

    await runtime.shutdown();
  });
});
