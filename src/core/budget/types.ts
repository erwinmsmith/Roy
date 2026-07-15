import type { ModelTokenUsage } from '../llm/types.js';

export type BudgetPriority = 'low' | 'medium' | 'normal' | 'high' | 'critical';
export type BudgetPolicyMode = 'unlimited' | 'fixed' | 'market';
export type BudgetAllocationStatus = 'granted' | 'denied' | 'settled' | 'released' | 'exceeded';
export type BudgetActorType = 'agent' | 'team' | 'runtime';
export type BudgetAccountingDimension = 'total_tokens' | 'output_tokens' | 'thinking_tokens';

export interface BudgetRequest {
  requesterId: string;
  parentId: string;
  actorType?: BudgetActorType;
  correlationId?: string;
  requestedTokens: number;
  minimumTokens?: number;
  expectedUtility?: number;
  priority?: BudgetPriority;
  purpose: string;
  metadata?: Record<string, unknown>;
}

export interface BudgetAllocation {
  id: string;
  request: BudgetRequest;
  policy: BudgetPolicyMode;
  status: BudgetAllocationStatus;
  allocatedTokens: number;
  /** Backward-compatible alias for allocatedTokens. */
  grantedTokens: number;
  consumedTokens: number;
  /** Backward-compatible alias populated at settlement. */
  actualTokens?: number;
  usage?: ModelTokenUsage;
  utilization: number;
  efficiency: number | null;
  score?: number;
  rationale: string;
  /** Backward-compatible alias for rationale. */
  reason: string;
  createdAt: number;
  updatedAt: number;
  settledAt?: number;
}

export interface BudgetDecision {
  request: BudgetRequest;
  allocatedTokens: number;
  score?: number;
  rationale: string;
}

export interface BudgetPolicyContext {
  limitTokens: number | null;
  usedTokens: number;
  reservedTokens: number;
  availableTokens?: number;
  minimumGrantTokens: number;
  priorityWeights: Record<BudgetPriority, number>;
}

export interface BudgetAllocationPolicy {
  readonly mode: BudgetPolicyMode;
  allocate(requests: BudgetRequest[], context: BudgetPolicyContext): BudgetDecision[];
}

export interface BudgetLedgerEntry {
  id: string;
  type: 'requested' | 'allocated' | 'consumed' | 'exceeded' | 'settled' | 'released' | 'rebalanced';
  allocationId?: string;
  requesterId?: string;
  tokens?: number;
  timestamp: number;
  data?: Record<string, unknown>;
}

export interface BudgetMarketOptions {
  mode?: BudgetPolicyMode;
  minimumGrantTokens?: number;
  accountingDimension?: BudgetAccountingDimension;
  priorityWeights?: Partial<Record<BudgetPriority, number>>;
}

export interface BudgetMarketState {
  mode: BudgetPolicyMode;
  policy: BudgetPolicyMode;
  limitMode: 'unlimited' | 'limited';
  limitTokens?: number;
  sessionLimit?: number;
  usedTokens: number;
  reservedTokens: number;
  availableTokens?: number;
  accountingDimension: BudgetAccountingDimension;
  allocations: BudgetAllocation[];
  ledger: BudgetLedgerEntry[];
}

export interface BudgetRebalanceResult {
  changed: BudgetAllocation[];
  releasedTokens: number;
  reservedTokens: number;
}
