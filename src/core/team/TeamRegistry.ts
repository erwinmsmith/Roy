import type { CreateTeamSpec, TeamRuntimeState, TeamState } from './types.js';

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

export class InvalidTeamTransitionError extends Error {
  constructor(readonly from: TeamState, readonly to: TeamState) {
    super(`Invalid team transition: ${from} -> ${to}`);
    this.name = 'InvalidTeamTransitionError';
  }
}

const ZERO_USAGE = {
  llmCalls: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  thinkingTokens: null,
} as const;

export class TeamRegistry {
  private teams = new Map<string, TeamRuntimeState>();
  private sequence = 0;

  create(spec: CreateTeamSpec): TeamRuntimeState {
    const now = Date.now();
    const id = `team_${String(++this.sequence).padStart(3, '0')}`;
    const team: TeamRuntimeState = {
      identity: { id, name: spec.name, parentId: spec.parentId, purpose: spec.purpose, generation: spec.generation },
      state: 'created',
      memberIds: [],
      correlationId: spec.correlationId,
      tokenUsage: { ...ZERO_USAGE },
      createdAt: now,
      updatedAt: now,
    };
    this.teams.set(id, team);
    return this.clone(team);
  }

  addMember(teamId: string, agentId: string): TeamRuntimeState {
    const team = this.require(teamId);
    if (!team.memberIds.includes(agentId)) team.memberIds.push(agentId);
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  transition(teamId: string, state: TeamState, error?: string): TeamRuntimeState {
    const team = this.require(teamId);
    if (team.state !== state && !TEAM_TRANSITIONS[team.state].includes(state)) {
      throw new InvalidTeamTransitionError(team.state, state);
    }
    team.state = state;
    team.error = error;
    team.updatedAt = Date.now();
    return this.clone(team);
  }

  recordUsage(teamId: string, totalTokens: number): TeamRuntimeState {
    const team = this.require(teamId);
    team.tokenUsage.totalTokens += Math.max(0, totalTokens);
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

  private clone(team: TeamRuntimeState): TeamRuntimeState {
    return { ...team, identity: { ...team.identity }, memberIds: [...team.memberIds], tokenUsage: { ...team.tokenUsage } };
  }
}
