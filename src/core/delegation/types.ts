import type { DelegationAgentPlan, DelegationDecision } from '../runtime/Runtime.js';

export type DelegationCandidateSource =
  | 'cache_hit'
  | 'generated'
  | 'custom_generated'
  | 'mixed';

export interface DelegationCandidate {
  id: string;
  source: DelegationCandidateSource;
  parentId: string;
  agents: DelegationAgentPlan[];
  expectedUtility: number;
  expectedCostTokens: number;
  score: number;
  rationale: string;
}

export interface DelegationCandidateInput {
  parentId: string;
  task: string;
  decision: DelegationDecision;
  allowedChildren: number;
  remainingTotalAgentsForTurn: number;
  budgetMode: 'unlimited' | 'limited';
  remainingBudgetTokens?: number;
  cacheUsed: boolean;
}

export interface DelegationCandidateSelection {
  candidates: DelegationCandidate[];
  selected?: DelegationCandidate;
  decision: DelegationDecision;
  rejectedReason?: string;
}
