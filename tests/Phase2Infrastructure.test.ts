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
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

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
    expect(result.subagents).toHaveLength(2);
    expect(result.teams).toHaveLength(1);
    const teams = runtime.getTeams();
    expect(teams).toHaveLength(1);
    expect(teams[0].status).toBe('done');
    expect(teams[0].fsmState).toBe('S_team_done');
    expect(teams[0].identity.name).toBe('ReviewTeam');
    expect(teams[0].memberAgentIds).toHaveLength(2);
    const events = runtime.getEvents().map(event => event.type);
    expect(events).toContain('team.created');
    expect(events).toContain('team.completed');
    expect(events).toContain('evo.proposed');
    expect(events).toContain('evo.evaluated');
    expect(events).toContain('evo.selected');
    expect(events).toContain('budget.granted');
    expect(events).toContain('budget.settled');
    expect(events).toContain('context.loaded');
    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('team.task');
    expect(messages.map(message => message.kind)).toContain('team.result');
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === teams[0].identity.id)).toHaveLength(2);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === teams[0].identity.id)).toHaveLength(2);
    expect(messages.map(message => message.kind)).toContain('evo.select');
    expect(await runtime.getEvolutionHistory()).toHaveLength(1);
    expect(result.usage.teamSynthesis[teams[0].identity.id].totalTokens).toBeGreaterThan(0);
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
