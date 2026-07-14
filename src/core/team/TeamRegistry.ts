import type {
  CreateTeamSpec,
  TeamFSMState,
  TeamRuntimeState,
  TeamStatus,
} from './types.js';
import type { TokenUsage } from '../runtime/Runtime.js';
import { normalizeTeamExecutionPolicy } from './execution.js';

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
  constructor(readonly from: TeamFSMState, readonly to: TeamFSMState) {
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
    const parentAgentId = spec.parentAgentId;
    const description = spec.description;
    if (!parentAgentId?.trim()) throw new Error('Team parentAgentId is required');
    if (!spec.name?.trim()) throw new Error('Team name is required');
    if (!description?.trim()) throw new Error('Team description is required');
    const team: TeamRuntimeState = {
      identity: {
        id,
        name: spec.name,
        role: 'subteam',
        parentAgentId,
        generation: spec.generation,
        tomLevel: spec.tomLevel ?? 2,
        description,
      },
      status: 'idle',
      fsmState: 'S_team_created',
      memberAgentIds: [],
      leadAgentId: spec.leadAgentId,
      tokenUsage: { ...ZERO_USAGE },
      synthesisUsage: { ...ZERO_USAGE },
      memberUsage: {},
      memberTasks: {},
      memberResults: {},
      memberStatuses: {},
      memberErrors: {},
      executionPolicy: normalizeTeamExecutionPolicy(spec.executionPolicy),
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
    if (task) team.memberTasks[agentId] = task;
    team.memberStatuses[agentId] ??= 'pending';
    if (lead || !team.leadAgentId) team.leadAgentId = agentId;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  setTask(teamId: string, task: string, correlationId?: string): TeamRuntimeState {
    const team = this.require(teamId);
    team.task = task;
    if (correlationId) team.correlationId = correlationId;
    team.memberErrors = {};
    team.memberStatuses = Object.fromEntries(team.memberAgentIds.map(agentId => [agentId, 'pending']));
    team.error = undefined;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordMemberResult(teamId: string, agentId: string, task: string, result: string, usage: TokenUsage): TeamRuntimeState {
    const team = this.require(teamId);
    team.memberTasks[agentId] = task;
    team.memberResults[agentId] = result;
    team.memberUsage[agentId] = this.addUsage(team.memberUsage[agentId] ?? ZERO_USAGE, usage);
    team.memberStatuses[agentId] = 'completed';
    delete team.memberErrors[agentId];
    team.tokenUsage = this.addUsage(team.tokenUsage, usage);
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  markMemberRunning(teamId: string, memberKey: string): TeamRuntimeState {
    const team = this.require(teamId);
    team.memberStatuses[memberKey] = 'running';
    delete team.memberErrors[memberKey];
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordMemberFailure(teamId: string, memberKey: string, error: string): TeamRuntimeState {
    const team = this.require(teamId);
    team.memberStatuses[memberKey] = 'failed';
    team.memberErrors[memberKey] = error;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  markMemberSkipped(teamId: string, memberKey: string): TeamRuntimeState {
    const team = this.require(teamId);
    team.memberStatuses[memberKey] = 'skipped';
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  clearMemberTracking(teamId: string, memberKey: string): TeamRuntimeState {
    const team = this.require(teamId);
    delete team.memberStatuses[memberKey];
    delete team.memberErrors[memberKey];
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
    team.error = error;
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
      tokenUsage: { ...team.tokenUsage },
      synthesisUsage: { ...team.synthesisUsage },
      memberUsage: Object.fromEntries(Object.entries(team.memberUsage).map(([id, usage]) => [id, { ...usage }])),
      memberTasks: { ...team.memberTasks },
      memberResults: { ...team.memberResults },
      memberStatuses: { ...team.memberStatuses },
      memberErrors: { ...team.memberErrors },
      executionPolicy: { ...team.executionPolicy },
    };
  }
}
