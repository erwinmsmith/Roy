import type { TokenUsage } from '../runtime/Runtime.js';
import type { ToMProfile } from '../tom/index.js';

export type TeamStatus = 'idle' | 'running' | 'waiting' | 'synthesizing' | 'done' | 'failed';
export type TeamExecutionMode = 'sequential' | 'parallel';
export type TeamFailureMode = 'fail_fast' | 'best_effort';
export type TeamMemberExecutionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface TeamExecutionPolicy {
  mode: TeamExecutionMode;
  failureMode: TeamFailureMode;
  maxConcurrency: number;
  minimumSuccessfulMembers: number;
}
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
  tomProfile: ToMProfile;
  description: string;
}

export interface TeamRuntimeState {
  identity: TeamIdentity;
  status: TeamStatus;
  fsmState: TeamFSMState;
  memberAgentIds: string[];
  leadAgentId?: string;
  tokenUsage: TokenUsage;
  synthesisUsage: TokenUsage;
  memberUsage: Record<string, TokenUsage>;
  memberTasks: Record<string, string>;
  memberResults: Record<string, string>;
  memberStatuses: Record<string, TeamMemberExecutionStatus>;
  memberErrors: Record<string, string>;
  executionPolicy: TeamExecutionPolicy;
  synthesisPolicy?: string;
  task?: string;
  result?: string;
  correlationId?: string;
  createdAt: number;
  updatedAt: number;
  error?: string;
}

export interface CreateTeamSpec {
  id?: string;
  name: string;
  parentAgentId: string;
  description: string;
  generation: number;
  tomLevel?: number;
  tomProfile?: ToMProfile;
  leadAgentId?: string;
  task?: string;
  synthesisPolicy?: string;
  correlationId?: string;
  executionPolicy?: Partial<TeamExecutionPolicy>;
  createdAt?: number;
}
