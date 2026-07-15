import type { DelegationAgentPlan, DelegationDecision } from '../runtime/Runtime.js';
import type { ToMProfile } from '../agent/BaseAgent.js';
import type { ToMCoverageResult, ToMTaskAnalysis } from '../tom/index.js';

export type DelegationCandidateSource =
  | 'cache_hit'
  | 'generated'
  | 'custom_generated'
  | 'mutated_from_cache'
  | 'evolved'
  | 'mixed';

export interface DelegationCandidate {
  id: string;
  source: DelegationCandidateSource;
  parentId: string;
  agents: DelegationAgentPlan[];
  expectedUtility: number;
  expectedCostTokens: number;
  score: number;
  scoreBreakdown: Record<string, number>;
  rationale: string;
  tomCoverage?: ToMCoverageResult;
  lineage?: {
    parentPatternIds: string[];
    mutation?: string;
  };
}

export interface DelegationCandidateInput {
  parentId: string;
  correlationId?: string;
  task: string;
  decision: DelegationDecision;
  allowedChildren: number;
  remainingTotalAgentsForTurn: number;
  budgetMode: 'unlimited' | 'limited';
  remainingBudgetTokens?: number;
  cacheUsed: boolean;
  cachedPatterns?: Array<Record<string, unknown>>;
  parentToMProfile?: ToMProfile;
  tomAnalysis?: ToMTaskAnalysis;
}

export interface DelegationCandidateScore {
  scorer: string;
  values: Record<string, number>;
}

export interface DelegationCandidateScorer {
  readonly name: string;
  score(
    candidates: DelegationCandidate[],
    input: DelegationCandidateInput
  ): Promise<Map<string, number>> | Map<string, number>;
}

export interface DelegationCandidateSelection {
  candidates: DelegationCandidate[];
  selected?: DelegationCandidate;
  decision: DelegationDecision;
  rejectedReason?: string;
}
