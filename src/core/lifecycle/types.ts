import type { AgentInfo } from '../agent/BaseAgent.js';
import type { TeamRuntimeState } from '../team/types.js';

export type ActorKind = 'agent' | 'team';
export type ActorLifecycleAction = 'release' | 'retain_session' | 'persist';
export type ActorLifecycleMode = 'adaptive' | ActorLifecycleAction;
export type ActorLifecycleOrigin =
  | 'manual'
  | 'automatic_delegation'
  | 'team_member'
  | 'evolution'
  | 'restored';
export type ActorLifecycleOutcome = 'success' | 'failure' | 'cancelled' | 'manual';
export type ActorLifecycleStatus = 'active' | 'retained' | 'persisted' | 'released';

export interface ActorLifecyclePolicy {
  mode: ActorLifecycleMode;
  retainOnFailure: boolean;
  cascade: boolean;
}

export interface ActorLifecycleRegistration {
  actorId: string;
  actorKind: ActorKind;
  origin: ActorLifecycleOrigin;
  parentId?: string;
  policy: ActorLifecyclePolicy;
  createdAt: number;
}

export interface ActorLifecycleDecision {
  id: string;
  actorId: string;
  actorKind: ActorKind;
  action: ActorLifecycleAction;
  outcome: ActorLifecycleOutcome;
  origin: ActorLifecycleOrigin;
  reason: string;
  cascade: boolean;
  correlationId?: string;
  decidedAt: number;
  appliedAt?: number;
  snapshotPath?: string;
}

export interface ActorLifecycleRecord extends ActorLifecycleRegistration {
  status: ActorLifecycleStatus;
  lastDecision?: ActorLifecycleDecision;
  updatedAt: number;
}

export interface PersistedActorSnapshot {
  version: 1;
  actorId: string;
  actorKind: ActorKind;
  status: 'dormant';
  origin: ActorLifecycleOrigin;
  parentId?: string;
  sessionId: string;
  persistedAt: string;
  policy: ActorLifecyclePolicy;
  agent?: AgentInfo;
  team?: TeamRuntimeState;
  restore: Record<string, unknown>;
}

export interface LifecyclePolicyDefaults {
  manual: ActorLifecycleMode;
  automaticDelegation: ActorLifecycleMode;
  teamMember: ActorLifecycleMode;
  evolutionCandidate: ActorLifecycleMode;
  retainFailures: boolean;
  cascade: boolean;
}
