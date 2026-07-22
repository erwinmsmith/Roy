import type { ModelTokenUsage } from '../llm/types.js';
import { FixedBudgetPolicy, MarketBudgetPolicy, UnlimitedBudgetPolicy } from './policies.js';
import { WeightedReasoningInvestmentModel } from './utility.js';
import type {
  BudgetAllocation,
  BudgetAllocationPolicy,
  BudgetLedgerEntry,
  BudgetMarketOptions,
  BudgetMarketState,
  BudgetOutcome,
  BudgetOutcomeSummary,
  BudgetPolicyMode,
  BudgetPriority,
  BudgetRebalanceResult,
  BudgetRequest,
  ReasoningInvestmentModel,
} from './types.js';

const DEFAULT_WEIGHTS: Record<BudgetPriority, number> = {
  low: 0.6,
  medium: 1,
  normal: 1,
  high: 1.5,
  critical: 2.2,
};

export class BudgetMarket {
  private limitTokens: number | null = null;
  private allocations = new Map<string, BudgetAllocation>();
  private policies = new Map<BudgetPolicyMode, BudgetAllocationPolicy>();
  private ledger: BudgetLedgerEntry[] = [];
  private outcomeHistory = new Map<string, BudgetOutcomeSummary>();
  private sequence = 0;
  private ledgerSequence = 0;
  private options: Required<Omit<BudgetMarketOptions, 'priorityWeights' | 'investmentModel'>> & { priorityWeights: Record<BudgetPriority, number> };
  private investmentModel: ReasoningInvestmentModel;

  constructor(private readonly usedTokens: () => number, options: BudgetMarketOptions = {}) {
    this.options = {
      mode: options.mode ?? 'market',
      minimumGrantTokens: options.minimumGrantTokens ?? 256,
      accountingDimension: options.accountingDimension ?? 'total_tokens',
      priorityWeights: { ...DEFAULT_WEIGHTS, ...options.priorityWeights },
    };
    this.investmentModel = options.investmentModel ?? new WeightedReasoningInvestmentModel();
    this.registerPolicy(new UnlimitedBudgetPolicy());
    this.registerPolicy(new FixedBudgetPolicy());
    this.registerPolicy(new MarketBudgetPolicy());
  }

  registerPolicy(policy: BudgetAllocationPolicy): void {
    this.policies.set(policy.mode, policy);
  }

  configure(limitTokens: number | null, options: BudgetMarketOptions = {}): void {
    this.limitTokens = limitTokens === null ? null : Math.max(0, Math.floor(limitTokens));
    this.options = {
      ...this.options,
      ...options,
      priorityWeights: { ...this.options.priorityWeights, ...options.priorityWeights },
    };
    if (options.investmentModel) this.investmentModel = options.investmentModel;
  }

  request(input: BudgetRequest): BudgetAllocation {
    return this.requestMany([input])[0];
  }

  requestMany(inputs: BudgetRequest[]): BudgetAllocation[] {
    if (inputs.length === 0) return [];
    const enrichedInputs = inputs.map(input => this.enrichRequest(input));
    for (const input of enrichedInputs) this.appendLedger('requested', undefined, input.requesterId, input.requestedTokens, {
      purpose: input.purpose,
      investment: input.investment,
    });
    const policy = this.policies.get(this.options.mode);
    if (!policy) throw new Error(`Budget policy "${this.options.mode}" is not registered`);
    const decisions = policy.allocate(enrichedInputs, this.policyContext());
    return decisions.map(decision => {
      const now = Date.now();
      const granted = decision.allocatedTokens >= (decision.request.minimumTokens ?? this.options.minimumGrantTokens);
      const allocation: BudgetAllocation = {
        id: `budget_alloc_${now}_${String(++this.sequence).padStart(4, '0')}`,
        request: { ...decision.request },
        policy: this.options.mode,
        status: granted ? 'granted' : 'denied',
        allocatedTokens: granted ? decision.allocatedTokens : 0,
        grantedTokens: granted ? decision.allocatedTokens : 0,
        consumedTokens: 0,
        utilization: 0,
        efficiency: null,
        score: decision.score,
        rationale: decision.rationale,
        reason: granted ? decision.rationale : 'insufficient_remaining_budget',
        createdAt: now,
        updatedAt: now,
      };
      this.allocations.set(allocation.id, allocation);
      this.appendLedger('allocated', allocation.id, allocation.request.requesterId, allocation.allocatedTokens, {
        status: allocation.status,
        policy: allocation.policy,
        score: allocation.score,
        rationale: allocation.rationale,
        expectedReturn: allocation.request.investment?.expectedReturn,
        riskAdjustedUtility: allocation.request.investment?.riskAdjustedUtility,
      });
      return this.clone(allocation);
    });
  }

  assignRequester(allocationId: string, requesterId: string, actorType: 'agent' | 'team' | 'runtime' = 'agent'): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation) return undefined;
    allocation.request.requesterId = requesterId;
    allocation.request.actorType = actorType;
    allocation.updatedAt = Date.now();
    return this.clone(allocation);
  }

  augment(
    allocationId: string,
    requestedAdditionalTokens: number,
    minimumAdditionalTokens = 1
  ): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || allocation.status !== 'granted') return undefined;

    const requested = Math.max(0, Math.floor(requestedAdditionalTokens));
    const minimum = Math.max(1, Math.floor(minimumAdditionalTokens));
    if (requested === 0) return this.clone(allocation);

    this.appendLedger('requested', allocation.id, allocation.request.requesterId, requested, {
      purpose: `${allocation.request.purpose}:continuation`,
      continuation: true,
    });
    const available = this.limitTokens === null
      ? requested
      : Math.min(requested, this.availableTokens() ?? 0);
    if (available < minimum) {
      this.appendLedger('rebalanced', allocation.id, allocation.request.requesterId, 0, {
        previousTokens: allocation.allocatedTokens,
        requestedAdditionalTokens: requested,
        reason: 'insufficient_remaining_budget_for_continuation',
      });
      return this.clone(allocation);
    }

    const previousTokens = allocation.allocatedTokens;
    allocation.allocatedTokens += available;
    allocation.grantedTokens += available;
    allocation.rationale = 'continuation_budget_augmented';
    allocation.reason = allocation.rationale;
    allocation.updatedAt = Date.now();
    this.appendLedger('rebalanced', allocation.id, allocation.request.requesterId, available, {
      previousTokens,
      requestedAdditionalTokens: requested,
      continuation: true,
    });
    return this.clone(allocation);
  }

  consume(allocationId: string, usage: number | ModelTokenUsage): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || !['granted', 'exceeded'].includes(allocation.status)) return undefined;
    const tokens = typeof usage === 'number' ? Math.max(0, Math.floor(usage)) : this.tokensForDimension(usage);
    allocation.consumedTokens += tokens;
    if (typeof usage !== 'number') allocation.usage = {
      ...usage,
      availability: usage.availability ? { ...usage.availability } : undefined,
    };
    allocation.actualTokens = allocation.consumedTokens;
    allocation.utilization = allocation.allocatedTokens === 0 ? 0 : allocation.consumedTokens / allocation.allocatedTokens;
    allocation.efficiency = this.efficiency(allocation);
    if (allocation.consumedTokens > allocation.allocatedTokens) allocation.status = 'exceeded';
    allocation.updatedAt = Date.now();
    this.appendLedger('consumed', allocation.id, allocation.request.requesterId, tokens, {
      consumedTokens: allocation.consumedTokens,
      utilization: allocation.utilization,
    });
    if (allocation.status === 'exceeded') {
      this.appendLedger('exceeded', allocation.id, allocation.request.requesterId, allocation.consumedTokens - allocation.allocatedTokens);
    }
    return this.clone(allocation);
  }

  settle(allocationId: string, actual: number | ModelTokenUsage): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || !['granted', 'exceeded'].includes(allocation.status)) return undefined;
    const total = typeof actual === 'number' ? Math.max(0, Math.floor(actual)) : this.tokensForDimension(actual);
    if (total < allocation.consumedTokens) {
      throw new Error(`Settlement total ${total} is below already consumed ${allocation.consumedTokens}`);
    }
    const delta = Math.max(0, total - allocation.consumedTokens);
    if (delta > 0) this.consume(allocationId, delta);
    const current = this.allocations.get(allocationId)!;
    if (typeof actual !== 'number') current.usage = {
      ...actual,
      availability: actual.availability ? { ...actual.availability } : undefined,
    };
    current.consumedTokens = total;
    current.actualTokens = total;
    current.utilization = current.allocatedTokens === 0 ? 0 : total / current.allocatedTokens;
    current.efficiency = this.efficiency(current);
    current.status = total > current.allocatedTokens ? 'exceeded' : 'settled';
    current.settledAt = Date.now();
    current.updatedAt = current.settledAt;
    this.appendLedger('settled', current.id, current.request.requesterId, total, { status: current.status });
    return this.clone(current);
  }

  recordOutcome(allocationId: string, input: BudgetOutcome): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || allocation.status === 'denied') return undefined;
    if (allocation.outcome) throw new Error(`Budget allocation "${allocationId}" already has a recorded outcome`);
    allocation.outcome = {
      ...input,
      realizedUtility: input.realizedUtility === undefined ? undefined : clamp(input.realizedUtility),
      quality: input.quality === undefined ? undefined : clamp(input.quality),
      evidenceGain: input.evidenceGain === undefined ? undefined : clamp(input.evidenceGain),
      uncertaintyReduction: input.uncertaintyReduction === undefined ? undefined : clamp(input.uncertaintyReduction),
      conflictResolution: input.conflictResolution === undefined ? undefined : clamp(input.conflictResolution),
      verificationGain: input.verificationGain === undefined ? undefined : clamp(input.verificationGain),
      metadata: input.metadata ? { ...input.metadata } : undefined,
      recordedAt: input.recordedAt ?? Date.now(),
    };
    allocation.efficiency = this.efficiency(allocation);
    allocation.updatedAt = Date.now();
    this.updateOutcomeHistory(allocation);
    this.appendLedger('outcome', allocation.id, allocation.request.requesterId, allocation.consumedTokens, {
      success: allocation.outcome.success,
      realizedUtility: this.realizedUtility(allocation),
      efficiency: allocation.efficiency,
      error: allocation.outcome.error,
    });
    return this.clone(allocation);
  }

  release(allocationId: string, reason = 'released_without_execution'): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || allocation.status !== 'granted') return undefined;
    allocation.status = 'released';
    allocation.rationale = reason;
    allocation.reason = reason;
    allocation.updatedAt = Date.now();
    this.appendLedger('released', allocation.id, allocation.request.requesterId, allocation.allocatedTokens - allocation.consumedTokens, { reason });
    return this.clone(allocation);
  }

  rebalance(): BudgetRebalanceResult {
    const active = [...this.allocations.values()].filter(item => item.status === 'granted');
    if (active.length === 0) return { changed: [], releasedTokens: 0, reservedTokens: 0 };
    const availableSupply = this.limitTokens === null
      ? active.reduce((sum, item) => sum + item.allocatedTokens, 0)
      : Math.max(0, this.limitTokens - this.effectiveUsedTokens());
    const policy = this.policies.get(this.options.mode)!;
    const decisions = policy.allocate(active.map(item => ({ ...item.request })), {
      ...this.policyContext(),
      reservedTokens: 0,
      availableTokens: this.limitTokens === null ? undefined : availableSupply,
    });
    const changed: BudgetAllocation[] = [];
    let releasedTokens = 0;
    for (const [index, allocation] of active.entries()) {
      const decision = decisions[index];
      if (!decision) continue;
      const next = Math.max(allocation.consumedTokens, decision.allocatedTokens);
      if (next === allocation.allocatedTokens) continue;
      const previousTokens = allocation.allocatedTokens;
      releasedTokens += Math.max(0, previousTokens - next);
      allocation.allocatedTokens = next;
      allocation.grantedTokens = next;
      allocation.score = decision.score;
      allocation.rationale = 'market_policy_rebalanced';
      allocation.reason = allocation.rationale;
      allocation.updatedAt = Date.now();
      changed.push(this.clone(allocation));
      this.appendLedger('rebalanced', allocation.id, allocation.request.requesterId, next, { previousTokens });
    }
    return { changed, releasedTokens, reservedTokens: this.reservedTokens() };
  }

  getAllocation(allocationId: string): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    return allocation ? this.clone(allocation) : undefined;
  }

  getState(): BudgetMarketState {
    const usedTokens = this.effectiveUsedTokens();
    const reservedTokens = this.reservedTokens();
    return {
      mode: this.options.mode,
      policy: this.options.mode,
      limitMode: this.limitTokens === null ? 'unlimited' : 'limited',
      limitTokens: this.limitTokens ?? undefined,
      sessionLimit: this.limitTokens ?? undefined,
      usedTokens,
      reservedTokens,
      availableTokens: this.limitTokens === null ? undefined : Math.max(0, this.limitTokens - usedTokens - reservedTokens),
      accountingDimension: this.options.accountingDimension,
      allocations: [...this.allocations.values()].map(item => this.clone(item)),
      outcomeHistory: [...this.outcomeHistory.values()].map(item => ({ ...item })),
      ledger: this.ledger.map(item => ({ ...item, data: item.data ? { ...item.data } : undefined })),
    };
  }

  private policyContext() {
    return {
      limitTokens: this.limitTokens,
      usedTokens: this.effectiveUsedTokens(),
      reservedTokens: this.reservedTokens(),
      availableTokens: this.availableTokens(),
      minimumGrantTokens: this.options.minimumGrantTokens,
      priorityWeights: this.options.priorityWeights,
    };
  }

  private availableTokens(): number | undefined {
    if (this.limitTokens === null) return undefined;
    return Math.max(0, this.limitTokens - this.effectiveUsedTokens() - this.reservedTokens());
  }

  private effectiveUsedTokens(): number {
    const externalConsumed = [...this.allocations.values()]
      .filter(item => item.request.actorType === 'runtime')
      .reduce((sum, item) => sum + item.consumedTokens, 0);
    return this.usedTokens() + externalConsumed;
  }

  private reservedTokens(): number {
    return [...this.allocations.values()]
      .filter(item => item.status === 'granted')
      .reduce((sum, item) => sum + Math.max(0, item.allocatedTokens - item.consumedTokens), 0);
  }

  private tokensForDimension(usage: ModelTokenUsage): number {
    const value = this.options.accountingDimension === 'thinking_tokens'
      ? usage.thinkingTokens ?? usage.totalTokens
      : this.options.accountingDimension === 'output_tokens'
        ? usage.outputTokens ?? usage.completionTokens
        : usage.totalTokens;
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${this.options.accountingDimension} usage value`);
    return Math.floor(value);
  }

  private enrichRequest(input: BudgetRequest): BudgetRequest {
    this.validateRequest(input);
    const historyKey = this.historyKey(input);
    const history = this.outcomeHistory.get(historyKey);
    const resourceEstimate = {
      tokens: Math.max(0, Math.floor(input.resourceEstimate?.tokens ?? input.requestedTokens)),
      ...input.resourceEstimate,
    };
    const investment = input.investment ?? this.investmentModel.estimate({
      kind: String(input.metadata?.investmentKind ?? input.purpose),
      requesterId: input.requesterId,
      parentId: input.parentId,
      purpose: input.purpose,
      resources: resourceEstimate,
      signals: {
        rootUtility: input.expectedUtility ?? 0.5,
        parentUtility: input.expectedUtility ?? 0.5,
        historicalUtility: numberMetadata(input.metadata?.historicalUtility, history?.averageRealizedUtility ?? 0.5),
        evidenceGain: numberMetadata(input.metadata?.evidenceGain, 0),
        uncertaintyReduction: numberMetadata(input.metadata?.uncertaintyReduction, 0),
        conflictResolution: numberMetadata(input.metadata?.conflictResolution, 0),
        verificationGain: numberMetadata(input.metadata?.verificationGain, 0),
        cacheConfidence: numberMetadata(input.metadata?.cacheConfidence, 0),
        duplicationRisk: numberMetadata(input.metadata?.duplicationRisk, 0),
        executionRisk: numberMetadata(input.metadata?.executionRisk, 0),
        confidence: numberMetadata(input.metadata?.confidence, history ? Math.min(0.95, 0.55 + history.count * 0.08) : 0.65),
      },
      metadata: input.metadata,
    });
    this.validateInvestment(investment);
    return {
      ...input,
      expectedUtility: investment.riskAdjustedUtility,
      resourceEstimate,
      investment,
      metadata: input.metadata ? { ...input.metadata } : undefined,
    };
  }

  private validateRequest(input: BudgetRequest): void {
    if (!input.requesterId?.trim() || !input.parentId?.trim() || !input.purpose?.trim()) {
      throw new Error('Budget request requesterId, parentId, and purpose must not be empty');
    }
    if (!Number.isFinite(input.requestedTokens) || input.requestedTokens < 0) {
      throw new Error('Budget request requestedTokens must be a non-negative finite number');
    }
    if (input.minimumTokens !== undefined && (!Number.isFinite(input.minimumTokens) || input.minimumTokens < 0)) {
      throw new Error('Budget request minimumTokens must be a non-negative finite number');
    }
    if (input.expectedUtility !== undefined && (!Number.isFinite(input.expectedUtility) || input.expectedUtility < 0 || input.expectedUtility > 1)) {
      throw new Error('Budget request expectedUtility must be between 0 and 1');
    }
    for (const [key, value] of Object.entries(input.resourceEstimate ?? {})) {
      if (value !== undefined && (!Number.isFinite(value) || value < 0)) {
        throw new Error(`Budget resource estimate ${key} must be a non-negative finite number`);
      }
    }
  }

  private validateInvestment(investment: BudgetRequest['investment']): asserts investment is NonNullable<BudgetRequest['investment']> {
    if (!investment || !investment.model?.trim()) throw new Error('Reasoning investment model id must not be empty');
    for (const key of ['expectedUtility', 'riskAdjustedUtility', 'costScore', 'expectedReturn', 'confidence'] as const) {
      const value = investment[key];
      if (!Number.isFinite(value) || value < 0 || (key !== 'expectedReturn' && value > 1)) {
        throw new Error(`Reasoning investment ${key} is invalid`);
      }
    }
  }

  private efficiency(allocation: BudgetAllocation): number | null {
    if (allocation.consumedTokens === 0) return null;
    return this.realizedUtility(allocation) / allocation.consumedTokens * 1000;
  }

  private realizedUtility(allocation: BudgetAllocation): number {
    const outcome = allocation.outcome;
    if (!outcome) return allocation.request.investment?.riskAdjustedUtility ?? allocation.request.expectedUtility ?? 0.5;
    if (outcome.realizedUtility !== undefined) return outcome.realizedUtility;
    const components = [
      outcome.quality,
      outcome.evidenceGain,
      outcome.uncertaintyReduction,
      outcome.conflictResolution,
      outcome.verificationGain,
    ].filter((value): value is number => value !== undefined);
    const observed = components.length > 0 ? components.reduce((sum, value) => sum + value, 0) / components.length : undefined;
    return clamp((observed ?? (outcome.success ? 0.7 : 0.1)) * (outcome.success ? 1 : 0.35));
  }

  private updateOutcomeHistory(allocation: BudgetAllocation): void {
    const outcome = allocation.outcome;
    if (!outcome) return;
    const key = this.historyKey(allocation.request);
    const previous = this.outcomeHistory.get(key);
    const count = (previous?.count ?? 0) + 1;
    const realized = this.realizedUtility(allocation);
    const previousUtilityTotal = (previous?.averageRealizedUtility ?? 0) * (previous?.count ?? 0);
    const previousEfficiencySamples = previous?.efficiencySamples ?? 0;
    const efficiencySamples = previousEfficiencySamples + (allocation.efficiency === null ? 0 : 1);
    const previousEfficiencyTotal = (previous?.averageEfficiency ?? 0) * previousEfficiencySamples;
    this.outcomeHistory.set(key, {
      key,
      purpose: allocation.request.purpose,
      count,
      successCount: (previous?.successCount ?? 0) + (outcome.success ? 1 : 0),
      successRate: Number((((previous?.successCount ?? 0) + (outcome.success ? 1 : 0)) / count).toFixed(4)),
      averageRealizedUtility: Number(((previousUtilityTotal + realized) / count).toFixed(4)),
      averageEfficiency: allocation.efficiency === null
        ? previous?.averageEfficiency ?? null
        : Number(((previousEfficiencyTotal + allocation.efficiency) / efficiencySamples).toFixed(6)),
      efficiencySamples,
      lastRecordedAt: outcome.recordedAt ?? Date.now(),
    });
  }

  private historyKey(request: BudgetRequest): string {
    const explicit = request.metadata?.investmentHistoryKey;
    return typeof explicit === 'string' && explicit.trim() ? explicit.trim() : request.purpose;
  }

  private appendLedger(type: BudgetLedgerEntry['type'], allocationId?: string, requesterId?: string, tokens?: number, data?: Record<string, unknown>): void {
    this.ledger.push({
      id: `budget_event_${Date.now()}_${String(++this.ledgerSequence).padStart(5, '0')}`,
      type,
      allocationId,
      requesterId,
      tokens,
      timestamp: Date.now(),
      data,
    });
    if (this.ledger.length > 5000) this.ledger = this.ledger.slice(-5000);
  }

  private clone(allocation: BudgetAllocation): BudgetAllocation {
    return {
      ...allocation,
      request: {
        ...allocation.request,
        metadata: allocation.request.metadata ? { ...allocation.request.metadata } : undefined,
        resourceEstimate: allocation.request.resourceEstimate ? { ...allocation.request.resourceEstimate } : undefined,
        investment: allocation.request.investment ? {
          ...allocation.request.investment,
          components: { ...allocation.request.investment.components },
          rationale: [...allocation.request.investment.rationale],
        } : undefined,
      },
      outcome: allocation.outcome ? {
        ...allocation.outcome,
        metadata: allocation.outcome.metadata ? { ...allocation.outcome.metadata } : undefined,
      } : undefined,
      usage: allocation.usage ? {
        ...allocation.usage,
        availability: allocation.usage.availability ? { ...allocation.usage.availability } : undefined,
      } : undefined,
    };
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function numberMetadata(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}
