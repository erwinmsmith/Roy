import { describe, expect, it } from 'vitest';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import { InvalidTeamTransitionError, TeamRegistry } from '../src/core/team/index.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMJSONCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

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

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages);
    return { value, completion: { content: JSON.stringify(value), usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 } } };
  }

  isConfigured(): boolean {
    return true;
  }
}

class PartialFailureTeamLLM extends TeamTestLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const systemPrompt = messages.find(message => message.role === 'system')?.content ?? '';
    if (systemPrompt.includes('Name: Critic-1') || systemPrompt.includes('You are Critic-1')) {
      throw new Error('critic execution failed');
    }
    yield* super.stream(messages);
  }
}

class RootPartialFailureTeamLLM extends PartialFailureTeamLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const prompt = messages.map(message => message.content).join('\n');
    if (prompt.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Use independent inspection and critique.',
        agents: [
          { archetype: 'researcher', name: 'Researcher-1', task: 'Inspect bounded project evidence.', tomLevel: 0 },
          { archetype: 'critic', name: 'Critic-1', task: 'Critique the bounded project evidence.', tomLevel: 2 },
        ],
      } satisfies DelegationDecision as T;
    }
    if (prompt.includes("Roy's dynamic root-step controller")) {
      return { action: 'finalize', reason: 'Synthesize the successful result and explicit failure.' } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class EmptyVisibleTeamSynthesisLLM extends TeamTestLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const systemPrompt = messages.find(message => message.role === 'system')?.content ?? '';
    if (systemPrompt.includes('formal subteam actor')) {
      yield {
        content: '',
        done: true,
        usage: { promptTokens: 20, completionTokens: 1024, totalTokens: 1044, thinkingTokens: 1024 },
      };
      return;
    }
    yield* super.stream(messages);
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
    expect(rerun.correlationId).not.toBe(result.correlationId);
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

  it('falls back to structured member evidence when the model spends its output budget on reasoning', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-empty-synthesis-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase3-team-empty-synthesis',
      workspaceCwd: cwd,
      llmProvider: new EmptyVisibleTeamSynthesisLLM(),
    });
    const team = await runtime.spawnTeam({
      name: 'FallbackTeam',
      description: 'Preserve member results when visible synthesis is empty.',
      members: [{ archetype: 'summarizer', task: 'Summarize the bounded input.' }],
    });

    const result = await runtime.runTeam(team.identity.id, 'Summarize the bounded input.');

    expect(result.team.status).toBe('done');
    expect(result.result).toContain('[runtime_team_synthesis_fallback]');
    expect(result.result).toContain('# FallbackTeam Result');
    expect(result.result).toContain('Summarizer produced a concise evidence-based report.');
    expect(runtime.getEvents().some(event => event.type === 'team.synthesis.fallback')).toBe(true);
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

    await expect(runtime.spawnTeam({
      name: 'InvalidBindingTeam',
      description: 'Reject malformed runtime input.',
      members: [{ archetype: 'researcher', task: 'Inspect.', tools: 'fs.read' as unknown as string[] }],
    })).rejects.toThrow('Team member tools must be an array');
    await expect(runtime.spawnTeam({
      name: 'InvalidToMTeam',
      description: 'Reject malformed ToM input.',
      tomLevel: 4,
    })).rejects.toThrow('Team tomLevel must be an integer from 0 to 3');
    await expect(runtime.spawnTeam({
      name: 'InvalidExecutionTeam',
      description: 'Reject malformed execution policy.',
      executionPolicy: { mode: 'unsupported' as 'sequential' },
    })).rejects.toThrow('Unsupported team execution mode');
    await expect(runtime.spawnTeam({
      name: 'ImpossibleMinimumTeam',
      description: 'Reject an impossible success threshold.',
      executionPolicy: { minimumSuccessfulMembers: 2 },
      members: [{ archetype: 'researcher', task: 'Inspect one bounded area.' }],
    })).rejects.toThrow('exceeds planned members');

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

  it('hydrates members and execution policy from a cached team pattern', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-cache-hydration-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-cache-hydration', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });

    const first = await runtime.spawnTeam({
      name: 'HydratedReviewTeam',
      description: 'Reusable bounded review.',
      executionPolicy: { failureMode: 'fail_fast', minimumSuccessfulMembers: 2 },
      members: [
        { archetype: 'researcher', task: 'Inspect the bounded structure.', tools: ['fs.list'], lead: true },
        { archetype: 'critic', task: 'Critique the bounded structure.', tools: ['fs.read'] },
      ],
    });
    const second = await runtime.spawnTeam({
      name: 'HydratedReviewTeam',
      description: 'Reusable bounded review.',
    });

    expect(second.identity.id).not.toBe(first.identity.id);
    expect(second.executionPolicy.failureMode).toBe('fail_fast');
    expect(second.executionPolicy.minimumSuccessfulMembers).toBe(2);
    const result = await runtime.runTeam(second.identity.id, 'Run the cached team definition.');
    expect(result.members).toHaveLength(2);
    expect(result.team.leadAgentId).toBe(result.team.memberAgentIds[0]);
    expect(runtime.getEvents().some(event =>
      event.type === 'cache.hit'
      && event.correlationId === second.correlationId
      && event.data?.cacheType === 'team-pattern'
    )).toBe(true);
    await runtime.shutdown();
  });

  it('synthesizes successful member results under best-effort policy and records failures', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-partial-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-partial', workspaceCwd: cwd, llmProvider: new PartialFailureTeamLLM() });
    const team = await runtime.spawnTeam({
      name: 'PartialReviewTeam',
      description: 'Continue with explicit limitations when one member fails.',
      executionPolicy: { failureMode: 'best_effort', minimumSuccessfulMembers: 1 },
      members: [
        { archetype: 'researcher', name: 'Researcher-1', task: 'Inspect the bounded structure.', lead: true },
        { archetype: 'critic', name: 'Critic-1', task: 'Critique the bounded structure.' },
      ],
    });

    const result = await runtime.runTeam(team.identity.id, 'Review the bounded structure.');
    expect(result.team.status).toBe('done');
    expect(result.members).toHaveLength(1);
    expect(result.memberOutcomes.map(outcome => outcome.status)).toEqual(['completed', 'failed']);
    const failed = result.memberOutcomes[1];
    expect(failed.agentId).toBeDefined();
    expect(result.team.memberStatuses[failed.agentId!]).toBe('failed');
    expect(result.team.memberErrors[failed.agentId!]).toContain('critic execution failed');
    expect(runtime.getEvents().find(event => event.type === 'team.completed' && event.correlationId === result.correlationId)?.data)
      .toMatchObject({ partial: true, failedMembers: 1 });
    expect(runtime.getEvents().some(event => event.type === 'team.member.failed')).toBe(true);
    const pattern = (await runtime.getCachePatterns('teams'))[0];
    expect(pattern.usage).toMatchObject({ partialSuccessCount: 1, failedMemberRuns: 1, lastFailedMemberCount: 1 });
    const teamSession = await readFile(path.join(cwd, '.roy', 'teams', 'partialreviewteam', 'sessions.jsonl'), 'utf8');
    expect(teamSession).toContain('"partial":true');
    expect(teamSession).toContain('critic execution failed');
    await runtime.shutdown();
  });

  it('keeps failed automatic team members in the root step actor tree', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-root-partial-tree-'));
    await mkdir(path.join(cwd, '.roy'), { recursive: true });
    await writeFile(path.join(cwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: { rootSteps: { reassessAfterDelegation: true } },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase3-root-partial-tree',
      workspaceCwd: cwd,
      llmProvider: new RootPartialFailureTeamLLM(),
    });

    const result = await runtime.handleUserTurn('Inspect bounded project evidence and critique one risk.');
    const failed = result.teams[0].memberOutcomes.find(outcome => outcome.status === 'failed');

    expect(failed?.agentId).toBeDefined();
    expect(result.executionTree.steps[0].actorIds).toContain(failed?.agentId);
    expect(result.executionTree.nodes).toContainEqual(expect.objectContaining({
      id: failed?.agentId,
      status: 'failed',
    }));
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

  it('keeps a completed run successful when non-critical team persistence fails', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-persistence-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-persistence', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });
    const memory = (runtime as unknown as { ctx: { memory: { appendTeamSession: () => Promise<void> } } }).ctx.memory;
    memory.appendTeamSession = async () => {
      throw new Error('simulated session storage failure');
    };

    const team = await runtime.spawnTeam({
      name: 'ResilientTeam',
      description: 'Validate that persistence failures do not rewrite execution state.',
      members: [{ archetype: 'summarizer', task: 'Summarize the bounded input.' }],
    });
    const result = await runtime.runTeam(team.identity.id, 'Summarize the bounded input.');

    expect(result.team.status).toBe('done');
    expect(runtime.getEvents().find(event => event.type === 'team.persistence.failed')?.data?.operation)
      .toBe('append_team_session');
    expect(runtime.getEvents().map(event => event.type)).toContain('team.completed');
    expect(runtime.getEvents().map(event => event.type)).not.toContain('team.failed');
    await runtime.shutdown();
  });

  it('rejects team synthesis when the remaining token budget cannot cover its prompt', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-budget-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-budget', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });
    (runtime as unknown as { workspaceRuntimeConfig: { budgetMarket: { minimumGrantTokens: number } } })
      .workspaceRuntimeConfig.budgetMarket.minimumGrantTokens = 1;
    const team = await runtime.spawnTeam({
      name: 'BudgetedTeam',
      description: 'Validate synthesis budget enforcement.',
      members: [{ archetype: 'summarizer', task: 'Summarize the bounded input.', budgetTokens: 1000 }],
    });
    await runtime.runTeam(team.identity.id, 'Create the initial bounded summary.');
    // The member can run, but its low actual usage leaves less than the team synthesis prompt requires.
    runtime.setBudget(runtime.getBudgetState().usedTokens + 1000);
    await expect(runtime.runTeam(team.identity.id, 'Summarize the bounded input again.'))
      .rejects.toThrow('Team synthesis rejected: insufficient_remaining_budget');

    expect(runtime.getTeamState(team.identity.id)?.status).toBe('failed');
    const denial = runtime.getEvents().find(event => event.type === 'budget.denied' && event.agentId === team.identity.id);
    expect(denial?.data?.teamId).toBe(team.identity.id);
    expect(runtime.getBudgetMarketState().reservedTokens).toBe(0);
    await runtime.shutdown();
  });

  it('builds a mixed actor hierarchy with a subagent-owned team under its parent', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase3-team-tree-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'phase3-team-tree', workspaceCwd: cwd, llmProvider: new TeamTestLLM() });
    const delegated = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect a bounded architecture question.',
      requireRootSynthesis: true,
    });
    await (runtime as unknown as {
      prepareParentForDelegation: (parentId: string, correlationId: string, task: string) => Promise<void>;
    }).prepareParentForDelegation(delegated.agent.identity.id, 'nested-team-correlation', 'Create a review team.');
    const team = await runtime.spawnTeam({
      parentAgentId: delegated.agent.identity.id,
      name: 'NestedReviewTeam',
      description: 'Review the researcher result.',
    });

    const hierarchy = runtime.getTeamActorTree().hierarchy;
    const researcherNode = hierarchy.children.find(node => node.type === 'agent' && node.agent.identity.id === delegated.agent.identity.id);
    expect(researcherNode?.type).toBe('agent');
    const nestedTeam = researcherNode?.type === 'agent'
      ? researcherNode.children.find(node => node.type === 'team' && node.team.identity.id === team.identity.id)
      : undefined;
    expect(nestedTeam?.type).toBe('team');
    await runtime.shutdown();
  });
});
