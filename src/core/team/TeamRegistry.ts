import type {
  CreateTeamSpec,
  TeamFSMState,
  TeamRuntimeState,
  TeamState,
  TeamStatus,
} from './types.js';
import type { TokenUsage } from '../runtime/Runtime.js';

const TEAM_TRANSITIONS: Record<TeamState, TeamState[]> = {
  created: ['ready', 'failed', 'cancelled'],
  ready: ['running', 'failed', 'cancelled'],
  running: ['waiting', 'synthesizing', 'failed', 'cancelled'],
  waiting: ['synthesizing', 'failed', 'cancelled'],
  synthesizing: ['done', 'failed', 'cancelled'],
  done: [],
  failed: [],
  cancelled: [],
};

const TEAM_FSM_TRANSITIONS: Record<TeamFSMState, TeamFSMState[]> = {
  S_team_created: ['S_team_plan', 'S_team_failed'],
  S_team_plan: ['S_member_spawn', 'S_member_execute', 'S_team_failed'],
  S_member_spawn: ['S_member_execute', 'S_team_failed'],
  S_member_execute: ['S_member_aggregate', 'S_team_failed'],
  S_member_aggregate: ['S_team_synthesize', 'S_team_failed'],
  S_team_synthesize: ['S_team_done', 'S_team_failed'],
  S_team_done: ['S_team_plan'],
  S_team_failed: ['S_team_plan'],
};

const ZERO_USAGE: TokenUsage = {
  llmCalls: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  thinkingTokens: null,
};

export class InvalidTeamTransitionError extends Error {
  constructor(readonly from: TeamState | TeamFSMState, readonly to: TeamState | TeamFSMState) {
    super(`Invalid team transition: ${from} -> ${to}`);
    this.name = 'InvalidTeamTransitionError';
  }
}

export class TeamRegistry {
  private teams = new Map<string, TeamRuntimeState>();
  private sequence = 0;

  create(spec: CreateTeamSpec): TeamRuntimeState {
    const now = Date.now();
    const id = `team_${String(++this.sequence).padStart(3, '0')}`;
    const parentAgentId = spec.parentAgentId ?? spec.parentId;
    if (!parentAgentId) throw new Error('Team parentAgentId is required');
    const description = spec.description ?? spec.purpose ?? '';
    const team: TeamRuntimeState = {
      identity: {
        id,
        name: spec.name,
        role: 'subteam',
        parentAgentId,
        parentId: parentAgentId,
        generation: spec.generation,
        tomLevel: spec.tomLevel ?? 2,
        description,
        purpose: description,
      },
      state: 'created',
      status: 'idle',
      fsmState: 'S_team_created',
      memberAgentIds: [],
      memberIds: [],
      leadAgentId: spec.leadAgentId,
      tokenUsage: { ...ZERO_USAGE },
      synthesisUsage: { ...ZERO_USAGE },
      memberUsage: {},
      memberTasks: {},
      memberResults: {},
      task: spec.task,
      correlationId: spec.correlationId,
      createdAt: now,
      updatedAt: now,
    };
    this.teams.set(id, team);
    return this.clone(team);
  }

  addMember(teamId: string, agentId: string, task?: string, lead = false): TeamRuntimeState {
    const team = this.require(teamId);
    if (!team.memberAgentIds.includes(agentId)) team.memberAgentIds.push(agentId);
    team.memberIds = [...team.memberAgentIds];
    if (task) team.memberTasks[agentId] = task;
    if (lead || !team.leadAgentId) team.leadAgentId = agentId;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  setTask(teamId: string, task: string, correlationId?: string): TeamRuntimeState {
    const team = this.require(teamId);
    team.task = task;
    if (correlationId) team.correlationId = correlationId;
    team.error = undefined;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordMemberResult(teamId: string, agentId: string, task: string, result: string, usage: TokenUsage): TeamRuntimeState {
    const team = this.require(teamId);
    team.memberTasks[agentId] = task;
    team.memberResults[agentId] = result;
    team.memberUsage[agentId] = this.addUsage(team.memberUsage[agentId] ?? ZERO_USAGE, usage);
    team.tokenUsage = this.addUsage(team.tokenUsage, usage);
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordSynthesis(teamId: string, result: string, usage: TokenUsage): TeamRuntimeState {
    const team = this.require(teamId);
    team.result = result;
    team.synthesisUsage = this.addUsage(team.synthesisUsage, usage);
    team.tokenUsage = this.addUsage(team.tokenUsage, usage);
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  transitionFsm(teamId: string, next: TeamFSMState, error?: string): TeamRuntimeState {
    const team = this.require(teamId);
    if (team.fsmState !== next && !TEAM_FSM_TRANSITIONS[team.fsmState].includes(next)) {
      throw new InvalidTeamTransitionError(team.fsmState, next);
    }
    team.fsmState = next;
    team.status = this.statusFor(next);
    team.state = this.legacyStateFor(next);
    team.error = error;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  transition(teamId: string, state: TeamState, error?: string): TeamRuntimeState {
    const team = this.require(teamId);
    if (team.state !== state && !TEAM_TRANSITIONS[team.state].includes(state)) {
      throw new InvalidTeamTransitionError(team.state, state);
    }
    team.state = state;
    team.status = state === 'done' ? 'done'
      : state === 'failed' || state === 'cancelled' ? 'failed'
        : state === 'waiting' ? 'waiting'
          : state === 'synthesizing' ? 'synthesizing'
            : state === 'running' ? 'running' : 'idle';
    team.error = error;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordUsage(teamId: string, totalTokens: number): TeamRuntimeState {
    const usage = { ...ZERO_USAGE, totalTokens: Math.max(0, totalTokens) };
    const team = this.require(teamId);
    team.tokenUsage = this.addUsage(team.tokenUsage, usage);
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  get(teamId: string): TeamRuntimeState | undefined {
    const team = this.teams.get(teamId);
    return team ? this.clone(team) : undefined;
  }

  list(): TeamRuntimeState[] {
    return [...this.teams.values()].map(team => this.clone(team));
  }

  clear(): void {
    this.teams.clear();
    this.sequence = 0;
  }

  private require(teamId: string): TeamRuntimeState {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    return team;
  }

  private statusFor(state: TeamFSMState): TeamStatus {
    if (state === 'S_team_done') return 'done';
    if (state === 'S_team_failed') return 'failed';
    if (state === 'S_team_synthesize') return 'synthesizing';
    if (state === 'S_member_aggregate') return 'waiting';
    if (state === 'S_team_created') return 'idle';
    return 'running';
  }

  private legacyStateFor(state: TeamFSMState): TeamState {
    if (state === 'S_team_done') return 'done';
    if (state === 'S_team_failed') return 'failed';
    if (state === 'S_team_synthesize') return 'synthesizing';
    if (state === 'S_member_aggregate') return 'waiting';
    if (state === 'S_team_created') return 'created';
    return 'running';
  }

  private addUsage(current: TokenUsage, delta: TokenUsage): TokenUsage {
    const thinkingTokens = current.thinkingTokens === null && delta.thinkingTokens === null
      ? null
      : (current.thinkingTokens ?? 0) + (delta.thinkingTokens ?? 0);
    return {
      llmCalls: current.llmCalls + delta.llmCalls,
      promptTokens: current.promptTokens + delta.promptTokens,
      completionTokens: current.completionTokens + delta.completionTokens,
      totalTokens: current.totalTokens + delta.totalTokens,
      thinkingTokens,
      estimatedCostUsd: (current.estimatedCostUsd ?? 0) + (delta.estimatedCostUsd ?? 0) || undefined,
    };
  }

  private clone(team: TeamRuntimeState): TeamRuntimeState {
    return {
      ...team,
      identity: { ...team.identity },
      memberAgentIds: [...team.memberAgentIds],
      memberIds: [...team.memberIds],
      tokenUsage: { ...team.tokenUsage },
      synthesisUsage: { ...team.synthesisUsage },
      memberUsage: Object.fromEntries(Object.entries(team.memberUsage).map(([id, usage]) => [id, { ...usage }])),
      memberTasks: { ...team.memberTasks },
      memberResults: { ...team.memberResults },
    };
  }
}
