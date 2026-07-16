import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMJSONCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class LifecycleTestLLM implements LLMProvider {
  readonly name = 'lifecycle-test';
  readonly defaultModel = 'lifecycle-test-model';

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'completed', usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    yield {
      content: prompt.includes('formal subteam actor') ? 'Team synthesis completed.' : 'Actor task completed.',
      done: true,
      usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 },
    };
  }

  async completeJSON<T>(): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    return {
      value: await this.completeJSON<T>(messages, options),
      completion: { content: '{}', usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 } },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

async function createRuntime(prefix: string): Promise<Runtime> {
  const workspaceCwd = await mkdtemp(path.join(tmpdir(), prefix));
  const runtime = new Runtime();
  await runtime.initialize({
    sessionId: `${prefix}session`,
    workspaceCwd,
    fsmEnabled: true,
    llmProvider: new LifecycleTestLLM(),
  });
  return runtime;
}

describe('Runtime actor lifecycle', () => {
  it('releases completed automatic agents while preserving usage and lifecycle history', async () => {
    const runtime = await createRuntime('roy-lifecycle-auto-');
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'summarizer',
      tomLevel: 0,
      description: 'One-shot automatic summary.',
      lifecycleOrigin: 'automatic_delegation',
    });

    const result = await runtime.runAgent(agent.identity.id, 'Summarize this bounded input.');

    expect(result.usage.totalTokens).toBeGreaterThan(0);
    expect(runtime.getState().agents.some(item => item.identity.id === agent.identity.id)).toBe(false);
    expect(runtime.getBudgetState().perAgent[agent.identity.id].totalTokens).toBe(result.usage.totalTokens);
    expect(runtime.getActorLifecycle(agent.identity.id)).toMatchObject({
      status: 'released',
      lastDecision: { action: 'release', outcome: 'success' },
    });
    expect(runtime.getEvents().map(event => event.type)).toContain('actor.lifecycle.applied');

    await runtime.shutdown();
  });

  it('retains manually created agents for session reuse', async () => {
    const runtime = await createRuntime('roy-lifecycle-retain-');
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'summarizer',
      tomLevel: 0,
      description: 'Reusable session summary actor.',
    });

    await runtime.runAgent(agent.identity.id, 'Summarize this bounded input.');
    await runtime.runAgent(agent.identity.id, 'Summarize a second bounded input.');

    expect(runtime.getState().agents.some(item => item.identity.id === agent.identity.id)).toBe(true);
    expect(runtime.getActorLifecycle(agent.identity.id)).toMatchObject({
      status: 'retained',
      lastDecision: { action: 'retain_session' },
    });
    expect(runtime.getEvents().map(event => event.type)).toContain('actor.lifecycle.activated');

    await runtime.shutdown();
  });

  it('persists an agent as dormant and restores the same identity', async () => {
    const runtime = await createRuntime('roy-lifecycle-persist-agent-');
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      name: 'PersistentSummarizer-1',
      archetype: 'summarizer',
      tomLevel: 0,
      description: 'Persistent summary actor.',
      lifecycle: { mode: 'persist' },
    });

    await runtime.runAgent(agent.identity.id, 'Produce a reusable summary.');

    expect(runtime.getState().agents.some(item => item.identity.id === agent.identity.id)).toBe(false);
    expect(await runtime.getPersistedActors('agent')).toEqual([
      expect.objectContaining({ actorId: agent.identity.id, status: 'dormant' }),
    ]);

    const restored = await runtime.restoreActor(agent.identity.id);
    expect(restored.identity.id).toBe(agent.identity.id);
    expect(restored.identity.name).toBe('PersistentSummarizer-1');
    expect(runtime.getActorLifecycle(agent.identity.id)).toMatchObject({ status: 'active', origin: 'restored' });
    expect(await runtime.getPersistedActors('agent')).toHaveLength(0);

    await runtime.shutdown();
  });

  it('persists team composition and releases one-shot member instances', async () => {
    const runtime = await createRuntime('roy-lifecycle-persist-team-');
    const team = await runtime.spawnTeam({
      parentAgentId: 'root',
      name: 'PersistentSummaryTeam',
      description: 'Reusable summary team definition.',
      lifecycle: { mode: 'persist' },
      members: [
        { archetype: 'summarizer', name: 'Summarizer-1', task: 'Summarize the bounded input.', lead: true },
      ],
    });

    const result = await runtime.runTeam(team.identity.id, 'Summarize the bounded input.');
    const memberId = result.team.memberAgentIds[0];

    expect(runtime.getTeam(team.identity.id)).toBeUndefined();
    expect(runtime.getState().agents.some(item => item.identity.id === memberId)).toBe(false);
    expect(await runtime.getPersistedActors()).toEqual([
      expect.objectContaining({ actorId: team.identity.id, actorKind: 'team', status: 'dormant' }),
    ]);
    expect(runtime.getActorLifecycle(memberId)).toMatchObject({ status: 'released' });
    expect(runtime.getBudgetState().perTeam[team.identity.id]).toEqual(result.team.tokenUsage);

    const restored = await runtime.restoreActor(team.identity.id);
    expect(restored.identity.id).toBe(team.identity.id);
    expect(runtime.getTeam(team.identity.id)?.status).toBe('idle');

    await runtime.shutdown();
  });
});
