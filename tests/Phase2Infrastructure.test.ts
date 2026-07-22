import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FSM, InvalidFSMTransitionError } from '../src/core/executor/FSM.js';
import { BudgetMarket } from '../src/core/budget/index.js';
import { ToolApprovalManager } from '../src/core/tools/approval.js';
import { InvalidTeamTransitionError, TeamRegistry } from '../src/core/team/index.js';
import { DefaultDelegationCandidatePlanner } from '../src/core/delegation/index.js';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMJSONCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class EngineeringLLM implements LLMProvider {
  readonly name = 'engineering-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('Synthesize their results')
      ? 'Roy engineering synthesis.'
      : text.includes('Researcher')
        ? 'Researcher observed README.md, package.json, src/, and tests/.'
        : 'Critic identified coupling and verification risks.';
    yield { content, done: true, usage: { promptTokens: 18, completionTokens: 8, totalTokens: 26 } };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('delegation candidate evaluator')) {
      const parsed = JSON.parse(messages[1].content) as { candidates: Array<{ id: string }> };
      return {
        scores: parsed.candidates.map(candidate => ({
          candidateId: candidate.id,
          score: candidate.id === 'candidate_mutated_cache' ? 1 : 0.2,
        })),
      } as T;
    }
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Engineering review benefits from grounded inspection and independent critique.',
        agents: [
          { archetype: 'researcher', name: 'Researcher-1', task: 'Inspect project structure and report concrete paths.', tomLevel: 0 },
          { archetype: 'critic', name: 'Critic-1', task: 'Review architectural risks and evidence gaps.', tomLevel: 2 },
        ],
      } satisfies DelegationDecision as T;
    }
    if (text.includes('delegation controller')) {
      return { action: 'solve_directly', reason: 'Complete the bounded child task directly.' } satisfies DelegationDecision as T;
    }
    return {} as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[]): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages);
    return {
      value,
      completion: {
        content: JSON.stringify(value),
        model: this.defaultModel,
        usage: {
          promptTokens: 7,
          completionTokens: 4,
          totalTokens: 13,
          inputTokens: 7,
          outputTokens: 4,
          thinkingTokens: 2,
          cachedInputTokens: 1,
          cacheCreationInputTokens: 0,
          provider: this.name,
          model: this.defaultModel,
          source: 'provider',
          availability: {
            input: 'reported',
            output: 'reported',
            thinking: 'reported',
            cachedInput: 'reported',
            cacheCreationInput: 'reported',
          },
        },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Phase 2 engineering infrastructure', () => {
  it('enforces strict actor FSM transitions', async () => {
    const invalid: Array<[string, string]> = [];
    const fsm = new FSM({
      initialState: 'S_created',
      strict: true,
      onInvalidTransition: (from, to) => invalid.push([from, to]),
    });
    await fsm.transition('S_ready');
    await expect(fsm.transition('S_synthesizing')).rejects.toBeInstanceOf(InvalidFSMTransitionError);
    expect(invalid).toEqual([['S_ready', 'S_synthesizing']]);
  });

  it('reserves, settles, and denies token allocations deterministically', () => {
    let used = 200;
    const market = new BudgetMarket(() => used);
    market.configure(2000);
    const first = market.request({ requesterId: 'a', parentId: 'root', requestedTokens: 1200, minimumTokens: 500, purpose: 'inspection' });
    expect(first.status).toBe('granted');
    expect(market.getState().reservedTokens).toBe(1200);
    const denied = market.request({ requesterId: 'b', parentId: 'root', requestedTokens: 900, minimumTokens: 700, purpose: 'critique' });
    expect(denied.status).toBe('denied');
    used = 700;
    market.settle(first.id, 500);
    expect(market.getState().reservedTokens).toBe(0);
  });

  it('supports explicit tool approval policy decisions', () => {
    const approvals = new ToolApprovalManager({ readOnly: 'auto', write: 'ask', execute: 'ask', overrides: {} });
    const read = approvals.authorize({ agentId: 'root', toolName: 'fs.read', permission: 'read_only', params: { path: 'README.md' } });
    expect(read.decision).toBe('approved');
    const execute = approvals.authorize({ agentId: 'agent_tester_001', toolName: 'shell.exec', permission: 'execute', params: { command: 'node --version' } });
    expect(execute.decision).toBe('pending');
    expect(approvals.resolve(execute.request.id, 'approved')?.status).toBe('approved');
  });

  it('enforces formal subteam lifecycle transitions', () => {
    const teams = new TeamRegistry();
    const team = teams.create({ name: 'Review Team', parentAgentId: 'root', description: 'review', generation: 1 });
    teams.transitionFsm(team.identity.id, 'S_team_plan');
    expect(() => teams.transitionFsm(team.identity.id, 'S_team_done')).toThrow(InvalidTeamTransitionError);
    teams.transitionFsm(team.identity.id, 'S_member_execute');
    teams.transitionFsm(team.identity.id, 'S_member_aggregate');
    teams.transitionFsm(team.identity.id, 'S_team_synthesize');
    expect(teams.transitionFsm(team.identity.id, 'S_team_done').status).toBe('done');
  });

  it('uses team policy defaults when a generated partial policy contains undefined fields', () => {
    const teams = new TeamRegistry();
    const team = teams.create({
      name: 'Partial Policy Team',
      parentAgentId: 'root',
      description: 'Validate generated partial execution policy normalization.',
      generation: 1,
      executionPolicy: { mode: 'parallel', failureMode: undefined },
    });

    expect(team.executionPolicy).toEqual({
      mode: 'parallel',
      failureMode: 'best_effort',
      maxConcurrency: 3,
      minimumSuccessfulMembers: 1,
    });
  });

  it('uses LLM scoring and cache mutation through the pluggable planner', async () => {
    const planner = new DefaultDelegationCandidatePlanner({ llm: new EngineeringLLM() });
    const selection = await planner.select({
      parentId: 'root',
      task: 'Inspect project structure and risks',
      decision: {
        action: 'spawn_subagents',
        reason: 'Need specialists.',
        agents: [
          { archetype: 'researcher', task: 'Inspect project structure' },
          { archetype: 'critic', task: 'Review risks', tomLevel: 2 },
        ],
      },
      allowedChildren: 5,
      remainingTotalAgentsForTurn: 10,
      budgetMode: 'unlimited',
      cacheUsed: true,
      cachedPatterns: [{ id: 'agent_pattern_researcher_v1', archetype: 'researcher', taskSignature: 'project structure inspection', tools: ['fs.list'] }],
    });
    expect(selection.selected?.id).toBe('candidate_mutated_cache');
    expect(selection.selected?.scoreBreakdown.llm).toBe(1);
    expect(selection.selected?.lineage?.parentPatternIds).toContain('agent_pattern_researcher_v1');
  });

  it('preserves an explicit team coordination contract during candidate selection', async () => {
    const planner = new DefaultDelegationCandidatePlanner({
      scorers: [{ name: 'uniform', score: candidates => new Map(candidates.map(candidate => [candidate.id, 1])) }],
    });
    const selection = await planner.select({
      parentId: 'root',
      task: 'Reconcile two independently grounded perspectives.',
      decision: {
        action: 'spawn_subagents',
        reason: 'The perspectives require an explicit synthesis boundary.',
        coordination: 'team',
        team: { name: 'GeneratedCell', description: 'Task-specific coordination boundary.' },
        agents: [
          { archetype: 'custom', name: 'Perspective-A', task: 'Establish the first bounded perspective.' },
          { archetype: 'custom', name: 'Perspective-B', task: 'Establish the second bounded perspective.' },
        ],
      },
      allowedChildren: 5,
      remainingTotalAgentsForTurn: 10,
      budgetMode: 'unlimited',
      cacheUsed: false,
    });

    expect(selection.candidates.every(candidate => candidate.agents.length >= 2)).toBe(true);
    expect(selection.decision.action).toBe('spawn_subagents');
    if (selection.decision.action !== 'spawn_subagents') throw new Error('Expected team delegation');
    expect(selection.decision.coordination).toBe('team');
    expect(selection.decision.team?.name).toBe('GeneratedCell');
  });

  it('does not let cached patterns expand automatic delegation capabilities', async () => {
    const planner = new DefaultDelegationCandidatePlanner({
      scorers: [{
        name: 'prefer-cache',
        score: candidates => new Map(candidates.map(candidate => [
          candidate.id,
          candidate.id === 'candidate_mutated_cache' ? 1 : 0.1,
        ])),
      }],
    });
    const selection = await planner.select({
      parentId: 'root',
      task: 'Review a runtime risk and verify one test gap.',
      decision: {
        action: 'spawn_subagents',
        reason: 'Need bounded review.',
        agents: [
          { archetype: 'critic', task: 'Review the runtime risk.' },
          { archetype: 'tester', task: 'Verify the test gap.', tools: ['shell.exec'] },
        ],
      },
      allowedChildren: 5,
      remainingTotalAgentsForTurn: 10,
      budgetMode: 'unlimited',
      cacheUsed: true,
      cachedPatterns: [
        { id: 'critic-risky', archetype: 'critic', tools: ['shell.exec'], skills: ['unknown_skill'] },
        { id: 'tester-risky', archetype: 'tester', tools: ['fs.read', 'shell.exec'] },
      ],
      allowedToolsByArchetype: { critic: ['fs.read'], tester: ['fs.read'] },
      allowedSkillsByArchetype: { critic: ['critique_report'], tester: ['run_test'] },
    });

    const mutated = selection.candidates.find(candidate => candidate.id === 'candidate_mutated_cache');
    expect(mutated?.agents[0].tools).toEqual(['fs.read']);
    expect(mutated?.agents[0].skills).toEqual(['critique_report']);
    expect(mutated?.agents[1].tools).toEqual(['fs.read']);
    expect(selection.candidates.flatMap(candidate => candidate.agents)
      .flatMap(agent => agent.tools ?? [])).not.toContain('shell.exec');
  });

  it('creates a formal subteam and records evo/budget/context lifecycle events', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-infrastructure-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-infrastructure',
      workspaceCwd: cwd,
      fsmEnabled: true,
      llmProvider: new EngineeringLLM(),
    });
    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');
    expect(result.subagents).toHaveLength(3);
    expect(result.teams).toHaveLength(1);
    const completedTeam = result.teams[0].team;
    const teams = runtime.getTeams();
    expect(teams).toHaveLength(0);
    expect(completedTeam.status).toBe('done');
    expect(completedTeam.fsmState).toBe('S_team_done');
    expect(completedTeam.identity.name).toBe('AnalysisTeam');
    expect(completedTeam.memberAgentIds).toHaveLength(3);
    expect(completedTeam.identity.tomProfile.level).toBe(2);
    expect(completedTeam.identity.tomProfile.cognitiveGaps.length).toBeGreaterThan(0);
    expect(runtime.getActorLifecycle(completedTeam.identity.id)).toMatchObject({
      status: 'released',
      lastDecision: { action: 'release' },
    });
    const events = runtime.getEvents().map(event => event.type);
    expect(events).toContain('team.created');
    expect(events).toContain('team.completed');
    expect(events).toContain('evo.proposed');
    expect(events).toContain('evo.evaluated');
    expect(events).toContain('evo.selected');
    expect(events).toContain('budget.granted');
    expect(events).toContain('budget.settled');
    expect(events).toContain('context.loaded');
    const scorerAllocation = runtime.getBudgetMarketState().allocations.find(
      allocation => allocation.request.purpose === 'delegation.candidate_scoring'
    );
    expect(scorerAllocation?.status).toBe('settled');
    expect(scorerAllocation?.request.requesterId).toBe('root');
    expect(scorerAllocation?.usage?.inputTokens).toBe(7);
    expect(scorerAllocation?.usage?.outputTokens).toBe(4);
    expect(scorerAllocation?.usage?.thinkingTokens).toBe(2);
    expect(runtime.getEvents().some(event =>
      event.type === 'agent.llm.called'
      && event.agentId === 'root'
      && event.data?.purpose === 'delegation.candidate_scoring'
    )).toBe(true);
    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('team.task');
    expect(messages.map(message => message.kind)).toContain('team.result');
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === completedTeam.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === completedTeam.identity.id)).toHaveLength(3);
    expect(messages.map(message => message.kind)).toContain('evo.select');
    expect(await runtime.getEvolutionHistory()).toHaveLength(1);
    expect(result.usage.teamSynthesis[completedTeam.identity.id].totalTokens).toBeGreaterThan(0);
    await runtime.shutdown();
  });

  it('requires approval before an execute-permission tool runs', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-tool-approval-runtime-'));
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'tool-approval-runtime', workspaceCwd: cwd, llmProvider: new EngineeringLLM() });
    const tester = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'tester',
      name: 'Tester-1',
      tomLevel: 0,
      description: 'Run an approved validation command.',
      task: 'Run node --version.',
    });
    const pending = await runtime.executeToolForAgent(tester.identity.id, 'shell.exec', { command: 'node --version' });
    expect(pending.metadata?.pendingApproval).toBe(true);
    const approvalId = String(pending.metadata?.approvalId);
    expect((await runtime.resolveToolApproval(approvalId, 'approved'))?.status).toBe('approved');
    const executed = await runtime.executeToolForAgent(tester.identity.id, 'shell.exec', { command: 'node --version' }, { approvalId });
    expect(executed.success).toBe(true);
    await runtime.shutdown();
  });
});
