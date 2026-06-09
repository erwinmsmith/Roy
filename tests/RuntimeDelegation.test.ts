import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class RootDelegationLLM implements LLMProvider {
  readonly name = 'root-delegation-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
    };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    let content = 'Roy direct response.';
    if (text.includes('Synthesize their results into one final user-facing response')) {
      content = 'Final synthesis from Researcher-1 and Critic-2.';
    } else if (text.includes('architectural risks') || text.includes('failure modes')) {
      content = 'Critic report: coupling and runtime observability risks.';
    } else if (text.includes('grounded project structure')) {
      content = 'Researcher report: observed README.md, package.json, src/, tests/.';
    }
    yield {
      content,
      done: true,
      usage: { promptTokens: 20, completionTokens: 7, totalTokens: 27 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller") && text.includes('architectural risks')) {
      return {
        action: 'spawn_subagents',
        reason: 'Architecture risk analysis needs grounded inspection and critique.',
        agents: [
          {
            archetype: 'researcher',
            name: 'Researcher-1',
            task: 'Inspect grounded project structure and collect concrete evidence for architectural risk analysis.',
            tomLevel: 0,
          },
          {
            archetype: 'critic',
            name: 'Critic-2',
            task: 'Identify architectural risks and hidden coupling from the project evidence.',
            tomLevel: 2,
          },
        ],
      } satisfies DelegationDecision as T;
    }

    if (!text.includes("Roy's root delegation controller")) {
      return { action: 'none', params: {} } as T;
    }

    return {
      action: 'solve_directly',
      reason: 'Simple conversational task.',
    } satisfies DelegationDecision as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Runtime root-controlled delegation', () => {
  it('assesses a complex task, spawns subagents, waits for results, and synthesizes', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-delegation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-delegation-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(2);
    expect(result.subagents.map(item => item.agent.identity.id)).toEqual([
      'agent_researcher_001',
      'agent_critic_002',
    ]);
    expect(result.finalResponse).toBe('Final synthesis from Researcher-1 and Critic-2.');

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(2);
    expect(tree.children.map(child => child.agent.identity.parentId)).toEqual(['root', 'root']);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('delegation.decision');
    expect(eventTypes).toContain('agent.spawned');
    expect(eventTypes).toContain('agent.run.started');
    expect(eventTypes).toContain('agent.run.completed');
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');
    expect(eventTypes).toContain('memory.update.propose.completed');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('user.input');
    expect(messages.filter(message => message.kind === 'agent.task')).toHaveLength(2);
    expect(messages.filter(message => message.kind === 'agent.result')).toHaveLength(2);
    expect(messages.map(message => message.kind)).toContain('root.synthesis');
    expect(messages.map(message => message.kind)).toContain('root.final_response');

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBeGreaterThan(0);
    expect(budget.perAgent.agent_researcher_001.totalTokens).toBeGreaterThan(0);
    expect(budget.perAgent.agent_critic_002.totalTokens).toBeGreaterThan(0);
    expect(result.usage.total.totalTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('solves simple turns directly without spawning subagents', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-solo-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-solo-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('who are you?');

    expect(result.decision.action).toBe('solve_directly');
    expect(result.subagents).toHaveLength(0);
    expect(result.finalResponse).toBe('Roy direct response.');
    expect(runtime.getAgentTree().children).toHaveLength(0);
    expect(runtime.getEvents().map(event => event.type)).toContain('root.solo.completed');

    await runtime.shutdown();
  });
});
