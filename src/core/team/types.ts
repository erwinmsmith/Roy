import type { TokenUsage } from '../runtime/Runtime.js';

export type TeamState = 'created' | 'ready' | 'running' | 'waiting' | 'synthesizing' | 'done' | 'failed' | 'cancelled';
export type TeamStatus = 'idle' | 'running' | 'waiting' | 'synthesizing' | 'done' | 'failed';
export type TeamFSMState =
  | 'S_team_created'
  | 'S_team_plan'
  | 'S_member_spawn'
  | 'S_member_execute'
  | 'S_member_aggregate'
  | 'S_team_synthesize'
  | 'S_team_done'
  | 'S_team_failed';

export interface TeamIdentity {
  id: string;
  name: string;
  role: 'subteam';
  parentAgentId: string;
  generation: number;
  tomLevel: number;
  description: string;
  // Compatibility aliases for Phase 2 callers.
  parentId: string;
  purpose: string;
}

export interface TeamRuntimeState {
  identity: TeamIdentity;
  state: TeamState;
  status: TeamStatus;
  fsmState: TeamFSMState;
  memberAgentIds: string[];
  memberIds: string[];
  leadAgentId?: string;
  tokenUsage: TokenUsage;
  synthesisUsage: TokenUsage;
  memberUsage: Record<string, TokenUsage>;
  memberTasks: Record<string, string>;
  memberResults: Record<string, string>;
  task?: string;
  result?: string;
  correlationId?: string;
  createdAt: number;
  updatedAt: number;
  error?: string;
}

export interface CreateTeamSpec {
  name: string;
  parentAgentId?: string;
  parentId?: string;
  description?: string;
  purpose?: string;
  generation: number;
  tomLevel?: number;
  leadAgentId?: string;
  task?: string;
  correlationId?: string;
}
