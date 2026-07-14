import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import { InvalidTeamTransitionError, TeamRegistry } from '../src/core/team/index.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class TeamTestLLM implements LLMProvider {
  readonly name = 'team-test';
  readonly defaultModel = 'team-test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    const content = prompt.includes('subteam actor')
      ? 'AnalysisTeam consolidated the grounded inspection, critique, and summary.'
      : prompt.includes('Researcher')
        ? 'Researcher observed README.md, package.json, src/, docs/, and tests/.'
        : prompt.includes('Critic')
          ? 'Critic identified coupling and missing failure-path validation.'
          : 'Summarizer produced a concise evidence-based report.';
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const prompt = messages.map(message => message.content).join('\n');
    if (prompt.includes('delegation controller')) {
      return { action: 'solve_directly', reason: 'The assigned team member task is bounded.' } satisfies DelegationDecision as T;
    }
    return {} as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Phase 3 subteam runtime', () => {
  it('enforces the formal team FSM', () => {
    const registry = new TeamRegistry();
    const team = registry.create({
      name: 'AnalysisTeam',
      parentAgentId: 'root',
      description: 'Analyze architecture.',
      generation: 1,
      tomLevel: 2,
    });

    expect(team.status).toBe('idle');
    expect(team.fsmState).toBe('S_team_created');
    expect(() => registry.transitionFsm(team.identity.id, 'S_team_done')).toThrow(InvalidTeamTransitionError);
    registry.transitionFsm(team.identity.id, 'S_team_plan');
    registry.transitionFsm(team.identity.id, 'S_member_spawn');
    registry.transitionFsm(team.identity.id, 'S_member_execute');
    registry.transitionFsm(team.identity.id, 'S_member_aggregate');
    registry.transitionFsm(team.identity.id, 'S_team_synthesize');
    const completed = registry.transitionFsm(team.identity.id, 'S_team_done');
    expect(completed.status).toBe('done');
    expect(completed.state).toBe('done');
  });

  it('runs a three-member team through team messages, synthesis, budget, and persistence', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase3-team-runtime',
      workspaceCwd: cwd,
      fsmEnabled: true,
      llmProvider: new TeamTestLLM(),
    });

    const team = await runtime.spawnTeam({
      parentAgentId: 'root',
      name: 'AnalysisTeam',
      description: 'Inspect architecture, challenge risks, and summarize findings.',
      tomLevel: 2,
      task: 'Analyze the Roy project architecture.',
      members: [
        { archetype: 'researcher', name: 'Researcher-1', task: 'Inspect the project structure.', tomLevel: 0, lead: true },
        { archetype: 'critic', name: 'Critic-1', task: 'Identify architecture and evidence risks.', tomLevel: 2 },
        { archetype: 'summarizer', name: 'Summarizer-1', task: 'Summarize the team findings.', tomLevel: 1 },
      ],
    });
    const result = await runtime.runTeam(team.identity.id, 'Analyze the Roy project architecture.');

    expect(result.team.status).toBe('done');
    expect(result.team.fsmState).toBe('S_team_done');
    expect(result.team.memberAgentIds).toHaveLength(3);
    expect(result.team.leadAgentId).toBe(result.team.memberAgentIds[0]);
    expect(result.result).toContain('AnalysisTeam consolidated');
    expect(result.usage.totalTokens).toBe(120);
    expect(result.team.synthesisUsage.totalTokens).toBe(30);
    expect(Object.values(result.team.memberUsage).map(usage => usage.totalTokens)).toEqual([30, 30, 30]);

    const tree = runtime.getTeamTree(team.identity.id);
    expect(tree?.team.identity.parentAgentId).toBe('root');
    expect(tree?.members.map(member => member.agent.identity.name)).toEqual(['Researcher-1', 'Critic-1', 'Summarizer-1']);

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'team.result' && message.to === 'root')).toHaveLength(1);

    const transitions = runtime.getEvents()
      .filter(event => event.type === 'team.fsm.transition' && event.agentId === team.identity.id)
      .map(event => event.data?.to);
    expect(transitions).toEqual([
      'S_team_plan',
      'S_member_spawn',
      'S_member_execute',
      'S_member_aggregate',
      'S_team_synthesize',
      'S_team_done',
    ]);
    expect(runtime.getEvents().map(event => event.type)).toContain('team.synthesis.completed');
    expect(runtime.getEvents().find(event => event.type === 'team.context.loaded')?.agentId).toBe(team.identity.id);

    const budget = runtime.getBudgetState();
    expect(budget.perTeam[team.identity.id].totalTokens).toBe(120);
    expect(budget.perAgent[result.team.memberAgentIds[0]].totalTokens).toBe(30);

    const topology = JSON.parse(await readFile(path.join(cwd, '.roy', 'teams', 'analysisteam', 'topology.json'), 'utf8')) as {
      status: string;
      members: unknown[];
    };
    expect(topology.status).toBe('done');
    expect(topology.members).toHaveLength(3);
    const sessions = await readFile(path.join(cwd, '.roy', 'teams', 'analysisteam', 'sessions.jsonl'), 'utf8');
    expect(sessions).toContain('AnalysisTeam consolidated');
    const patterns = JSON.parse(await readFile(path.join(cwd, '.roy', 'cache', 'team-patterns.json'), 'utf8')) as {
      patterns: Array<{
        id: string;
        memberArchetypes: string[];
        members: Array<{ tools: string[]; skills: string[] }>;
        usage: { completedCount: number; averageTokens: number };
      }>;
    };
    expect(patterns.patterns[0].id).toBe('team_pattern_analysisteam_v1');
    expect(patterns.patterns[0].memberArchetypes).toEqual(['researcher', 'critic', 'summarizer']);
    expect(patterns.patterns[0].members[0].tools).toEqual(['fs.list', 'fs.read']);
    expect(patterns.patterns[0].members[0].skills).toContain('delegate_to_subagent');
    expect(patterns.patterns[0].usage.completedCount).toBe(1);
    expect(patterns.patterns[0].usage.averageTokens).toBe(120);

    const rerun = await runtime.runTeam(team.identity.id, 'Re-run the architecture review with the same members.');
    expect(rerun.team.memberAgentIds).toEqual(result.team.memberAgentIds);
    expect(rerun.team.status).toBe('done');
    expect(rerun.usage.totalTokens).toBe(120);
    expect(rerun.team.tokenUsage.totalTokens).toBe(240);
    const updatedPatterns = await runtime.getCachePatterns('teams');
    const updatedUsage = updatedPatterns[0].usage as { completedCount: number; averageTokens: number };
    expect(updatedUsage.completedCount).toBe(2);
    expect(updatedUsage.averageTokens).toBe(120);
    expect(updatedPatterns[0].status).toBe('active');

    await runtime.shutdown();
  });

  it('marks a team failed when it has no members and enforces the member plan limit', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-policy-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-policy', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });

    const empty = await runtime.spawnTeam({ name: 'EmptyTeam', description: 'Validate failure handling.' });
    await expect(runtime.runTeam(empty.identity.id, 'Run without members.')).rejects.toThrow('has no members or member plans');
    expect(runtime.getTeamState(empty.identity.id)?.status).toBe('failed');
    expect(runtime.getEvents().map(event => event.type)).toContain('team.failed');

    await expect(runtime.spawnTeam({
      name: 'OversizedTeam',
      description: 'Should be rejected by policy.',
      members: Array.from({ length: 6 }, (_, index) => ({
        archetype: 'researcher' as const,
        task: `Inspect area ${index + 1}`,
      })),
    })).rejects.toThrow('Team member limit exceeded');
    expect((await runtime.getMessages()).map(message => message.kind)).toContain('team.create.rejected');
    expect(runtime.getEvents().map(event => event.type)).toContain('team.create.rejected');

    await runtime.shutdown();
  });

  it('reuses a stable team pattern while creating distinct runtime team instances', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-cache', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });

    const first = await runtime.spawnTeam({ name: 'ReviewTeam', description: 'Reusable architecture review team.' });
    const second = await runtime.spawnTeam({ name: 'ReviewTeam', description: 'Reusable architecture review team.' });
    expect(first.identity.id).not.toBe(second.identity.id);
    const hit = runtime.getEvents().find(event => event.type === 'cache.hit' && event.data?.cacheType === 'team-pattern');
    expect(hit?.data?.patternId).toBe('team_pattern_reviewteam_v1');

    const patterns = await runtime.getCachePatterns('teams');
    expect(patterns).toHaveLength(1);
    expect((patterns[0].usage as { count: number }).count).toBe(2);
    expect(patterns[0].status).toBe('active');
    await runtime.shutdown();
  });

  it('plans members without executing them before the team FSM run', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-plan-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-plan', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });

    const team = await runtime.spawnTeam({ name: 'ManualTeam', description: 'Manual member planning.' });
    const planned = await runtime.spawnAgentIntoTeam(team.identity.id, {
      archetype: 'researcher',
      task: 'Inspect the project structure.',
      lead: true,
    });
    expect(planned.memberAgentIds).toHaveLength(0);
    expect(runtime.getState().agents).toHaveLength(1);
    expect(runtime.getEvents().map(event => event.type)).toContain('team.member.planned');

    const result = await runtime.runTeam(team.identity.id, 'Inspect and summarize the project.');
    expect(result.team.memberAgentIds).toHaveLength(1);
    expect(result.team.status).toBe('done');
    await runtime.shutdown();
  });
});
