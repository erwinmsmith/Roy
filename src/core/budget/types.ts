import type { ModelTokenUsage } from '../llm/types.js';

export type BudgetPriority = 'low' | 'medium' | 'normal' | 'high' | 'critical';
export type BudgetPolicyMode = 'unlimited' | 'fixed' | 'market';
export type BudgetAllocationStatus = 'granted' | 'denied' | 'settled' | 'released' | 'exceeded';
export type BudgetActorType = 'agent' | 'team' | 'runtime';
export type BudgetAccountingDimension = 'total_tokens' | 'output_tokens' | 'thinking_tokens';

export interface ReasoningResourceEstimate {
  tokens: number;
  inputTokens?: number;
  outputTokens?: number;
  thinkingTokens?: number;
  contextTokens?: number;
  toolCalls?: number;
  latencyMs?: number;
}

export interface ReasoningUtilitySignals {
  rootUtility?: number;
  parentUtility?: number;
  historicalUtility?: number;
  evidenceGain?: number;
  uncertaintyReduction?: number;
  conflictResolution?: number;
  verificationGain?: number;
  cacheConfidence?: number;
  duplicationRisk?: number;
  executionRisk?: number;
  confidence?: number;
}

export interface ReasoningInvestmentInput {
  kind: string;
  requesterId: string;
  parentId: string;
  purpose: string;
  resources: ReasoningResourceEstimate;
  signals: ReasoningUtilitySignals;
  metadata?: Record<string, unknown>;
}

export interface ReasoningInvestmentEstimate {
  model: string;
  expectedUtility: number;
  riskAdjustedUtility: number;
  costScore: number;
  expectedReturn: number;
  confidence: number;
  components: Record<string, number>;
  rationale: string[];
}

export interface ReasoningInvestmentModel {
  readonly id: string;
  estimate(input: ReasoningInvestmentInput): ReasoningInvestmentEstimate;
}

export interface BudgetOutcome {
  success: boolean;
  realizedUtility?: number;
  quality?: number;
  evidenceGain?: number;
  uncertaintyReduction?: number;
  conflictResolution?: number;
  verificationGain?: number;
  error?: string;
  metadata?: Record<string, unknown>;
  recordedAt?: number;
}

export interface BudgetOutcomeSummary {
  key: string;
  purpose: string;
  count: number;
  successCount: number;
  successRate: number;
  averageRealizedUtility: number;
  averageEfficiency: number | null;
  efficiencySamples: number;
  lastRecordedAt: number;
}

export interface BudgetRequest {
  requesterId: string;
  parentId: string;
  actorType?: BudgetActorType;
  correlationId?: string;
  requestedTokens: number;
  minimumTokens?: number;
  expectedUtility?: number;
  resourceEstimate?: ReasoningResourceEstimate;
  investment?: ReasoningInvestmentEstimate;
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
  outcome?: BudgetOutcome;
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
  type: 'requested' | 'allocated' | 'consumed' | 'exceeded' | 'settled' | 'released' | 'rebalanced' | 'outcome';
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
  investmentModel?: ReasoningInvestmentModel;
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
  outcomeHistory: BudgetOutcomeSummary[];
  ledger: BudgetLedgerEntry[];
}

export interface BudgetRebalanceResult {
  changed: BudgetAllocation[];
  releasedTokens: number;
  reservedTokens: number;
}
