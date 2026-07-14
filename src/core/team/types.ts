import type { TokenUsage } from '../runtime/Runtime.js';

export type TeamState = 'created' | 'ready' | 'running' | 'waiting' | 'synthesizing' | 'done' | 'failed' | 'cancelled';

export interface TeamIdentity {
  id: string;
  name: string;
  parentId: string;
  purpose: string;
  generation: number;
}

export interface TeamRuntimeState {
  identity: TeamIdentity;
  state: TeamState;
  memberIds: string[];
  correlationId?: string;
  tokenUsage: TokenUsage;
  createdAt: number;
  updatedAt: number;
  error?: string;
}

export interface CreateTeamSpec {
  name: string;
  parentId: string;
  purpose: string;
  generation: number;
  correlationId?: string;
}
