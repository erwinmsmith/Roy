import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
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

class EvolutionTestLLM implements LLMProvider {
  readonly name = 'evolution-test';
  readonly defaultModel = 'evolution-test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    const content = prompt.includes('subteam actor')
      ? 'The team synthesized concrete architecture evidence, risks, and limitations.'
      : prompt.includes('critic')
        ? 'The critic identified coupling, missing failure-path checks, and evidence limits.'
        : 'The specialist returned concrete project evidence and explicit limitations.';
    yield { content, done: true, usage: { promptTokens: 30, completionTokens: 12, totalTokens: 42 } };
  }

  async completeJSON<T>(): Promise<T> {
    return {} as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages, options);
    return { value, completion: { content: JSON.stringify(value), usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 } } };
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Phase 6 runtime evolution', () => {
  it('runs real team-first candidates, selects a genome, persists patterns, and reuses cache', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-evolution-runtime-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'evolution-runtime',
      workspaceCwd: cwd,
      fsmEnabled: true,
      llmProvider: new EvolutionTestLLM(),
    });

    const seedAgents = [
      {
        archetype: 'planner' as const,
        name: 'ArchitecturePlanner',
        role: 'architecture planner',
        task: 'Map the architecture and dependencies.',
        skills: ['delegate_to_subagent'],
        budgetTokens: 1800,
        tomLevel: 1,
      },
      {
        archetype: 'critic' as const,
        name: 'RiskCritic',
        role: 'architecture risk critic',
        task: 'Identify architectural risks and failure modes.',
        skills: ['delegate_to_subagent'],
        budgetTokens: 1800,
        tomLevel: 2,
      },
    ];
    const first = await runtime.runEvolution({
      task: 'Analyze this repository architecture and identify risks.',
      seedAgents,
      profile: 'evo_team',
      options: {
        populationSize: 2,
        generations: 1,
        maxExecutedCandidates: 2,
        integrationMinimumScore: 0.1,
        useLlmJudge: false,
      },
    });

    expect(first.state).toBe('S_evo_done');
    expect(first.candidates[0].genome.members).toHaveLength(2);
    expect(first.candidates.some(candidate => candidate.genome.members.length === 1)).toBe(true);
    expect(first.candidates.some(candidate => candidate.lineage.operators.length > 0)).toBe(true);
    expect(first.executions.length).toBeGreaterThanOrEqual(2);
    expect(first.selected).toBeDefined();
    expect(first.selectedEvaluation?.score).toBeGreaterThan(0.1);
    expect(first.integratedPatternId).toMatch(/^evo_pattern_/);
    expect(first.metrics.agentsSpawned).toBeGreaterThan(0);
    expect(first.metrics.teamsSpawned).toBeGreaterThan(0);
    expect(first.metrics.totalTokens).toBeGreaterThan(0);
    expect(runtime.getState().agents.map(agent => agent.identity.id)).toEqual(['root']);
    expect(Object.keys(runtime.getBudgetState().perAgent).length).toBeGreaterThan(1);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('evo.candidate.spawned');
    expect(eventTypes).toContain('evo.candidate.evaluated');
    expect(eventTypes).toContain('evo.candidate.selected');
    expect(eventTypes).toContain('evo.candidate.integrated');
    expect(eventTypes).toContain('evo.candidate.actor.archived');
    const messages = await runtime.getMessages({ correlationId: first.correlationId });
    expect(messages.map(message => message.kind)).toContain('evo.propose');
    expect(messages.map(message => message.kind)).toContain('evo.instantiate');
    expect(messages.map(message => message.kind)).toContain('evo.execute');
    expect(messages.map(message => message.kind)).toContain('evo.evaluate');
    expect(messages.map(message => message.kind)).toContain('evo.select');
    expect(messages.map(message => message.kind)).toContain('evo.integrate');

    const patterns = await runtime.getEvolutionPatterns();
    expect(patterns).toHaveLength(1);
    expect(patterns[0].genome.members.length).toBeGreaterThan(0);
    expect(patterns[0].linkedPatterns.agentPatternIds.length).toBeGreaterThan(0);
    const persisted = JSON.parse(await readFile(path.join(cwd, '.roy', 'cache', 'evolution-patterns.json'), 'utf8')) as { patterns: unknown[] };
    expect(persisted.patterns).toHaveLength(1);

    const second = await runtime.runEvolution({
      task: 'Analyze this repository architecture and identify risks.',
      seedAgents,
      profile: 'evo_team',
      options: {
        populationSize: 2,
        generations: 0,
        maxExecutedCandidates: 1,
        integrationMinimumScore: 0.1,
      },
    });
    expect(second.metrics.cacheHits).toBeGreaterThan(0);
    expect(second.candidates.some(candidate => candidate.source === 'cache_hit')).toBe(true);
    expect(runtime.getEvents().some(event => event.type === 'cache.hit' && event.data?.cacheType === 'evolution-pattern')).toBe(true);

    await runtime.shutdown();
  });

  it('applies benchmark profile ablations as real execution controls', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-evolution-profiles-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'evolution-profiles', workspaceCwd: cwd, llmProvider: new EvolutionTestLLM() });

    const solo = await runtime.runEvolution({ task: 'Explain the current architecture briefly.', profile: 'solo' });
    expect(solo.metrics.agentsSpawned).toBe(0);
    expect(solo.ablations.withoutSubagents).toBe(true);
    expect(solo.ablations.withoutBudgetMarket).toBe(true);

    const fixed = await runtime.runEvolution({
      task: 'Review architecture risks.',
      profile: 'fixed_subagents',
      seedAgents: [{ archetype: 'critic', task: 'Review risks.', tomLevel: 2 }],
      options: { populationSize: 1, maxExecutedCandidates: 1 },
    });
    expect(fixed.ablations.withoutToMProfile).toBe(true);
    expect(fixed.ablations.withoutEvoMutation).toBe(true);
    expect(fixed.ablations.withoutPatternMemory).toBe(true);
    expect(fixed.candidates[0].genome.members[0].tomProfile.level).toBe(0);
    expect(runtime.getEvents().some(event => event.type === 'budget.bypassed' && event.correlationId === fixed.correlationId)).toBe(true);

    await runtime.shutdown();
  });

  it('routes normal complex chat through evolution when auto mode is enabled', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-evolution-auto-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'evolution-auto', workspaceCwd: cwd, llmProvider: new EvolutionTestLLM() });
    await runtime.updateEvolutionConfig({
      mode: 'auto',
      profile: 'evo_team',
      populationSize: 1,
      generations: 0,
      maxExecutedCandidates: 1,
      integrationMinimumScore: 0.1,
    });

    const turn = await runtime.handleUserTurn('Analyze this repo architecture and identify concrete risks.');
    expect(turn.decision.action).toBe('spawn_subagents');
    expect(turn.evolution?.state).toBe('S_evo_done');
    expect(turn.evolution?.selectedExecution?.result).toBeTruthy();
    expect(turn.finalResponse).toBeTruthy();
    expect(turn.usage.total.totalTokens).toBeGreaterThan(0);
    expect(runtime.getEvents().some(event => event.type === 'evo.run.completed' && event.correlationId === turn.correlationId)).toBe(true);
    const conversation = await runtime.getConversation(undefined, 10);
    expect(conversation.at(-1)?.metadata?.evolutionRunId).toBe(turn.evolution?.id);

    const configFile = JSON.parse(await readFile(path.join(cwd, '.roy', 'config.json'), 'utf8')) as { evolution: { mode: string } };
    expect(configFile.evolution.mode).toBe('auto');
    await runtime.shutdown();
  });
});
