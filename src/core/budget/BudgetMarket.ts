import type { ModelTokenUsage } from '../llm/types.js';
import { FixedBudgetPolicy, MarketBudgetPolicy, UnlimitedBudgetPolicy } from './policies.js';
import type {
  BudgetAllocation,
  BudgetAllocationPolicy,
  BudgetLedgerEntry,
  BudgetMarketOptions,
  BudgetMarketState,
  BudgetPolicyMode,
  BudgetPriority,
  BudgetRebalanceResult,
  BudgetRequest,
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
  private sequence = 0;
  private ledgerSequence = 0;
  private options: Required<Omit<BudgetMarketOptions, 'priorityWeights'>> & { priorityWeights: Record<BudgetPriority, number> };

  constructor(private readonly usedTokens: () => number, options: BudgetMarketOptions = {}) {
    this.options = {
      mode: options.mode ?? 'market',
      minimumGrantTokens: options.minimumGrantTokens ?? 256,
      accountingDimension: options.accountingDimension ?? 'total_tokens',
      priorityWeights: { ...DEFAULT_WEIGHTS, ...options.priorityWeights },
    };
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
  }

  request(input: BudgetRequest): BudgetAllocation {
    return this.requestMany([input])[0];
  }

  requestMany(inputs: BudgetRequest[]): BudgetAllocation[] {
    if (inputs.length === 0) return [];
    for (const input of inputs) this.appendLedger('requested', undefined, input.requesterId, input.requestedTokens, { purpose: input.purpose });
    const policy = this.policies.get(this.options.mode);
    if (!policy) throw new Error(`Budget policy "${this.options.mode}" is not registered`);
    const decisions = policy.allocate(inputs, this.policyContext());
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
    allocation.efficiency = allocation.consumedTokens === 0
      ? null
      : (allocation.request.expectedUtility ?? 0.5) / allocation.consumedTokens * 1000;
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
    current.efficiency = total === 0 ? null : (current.request.expectedUtility ?? 0.5) / total * 1000;
    current.status = total > current.allocatedTokens ? 'exceeded' : 'settled';
    current.settledAt = Date.now();
    current.updatedAt = current.settledAt;
    this.appendLedger('settled', current.id, current.request.requesterId, total, { status: current.status });
    return this.clone(current);
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
    for (const allocation of active) {
      const decision = decisions.find(item => item.request.requesterId === allocation.request.requesterId && item.request.purpose === allocation.request.purpose);
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
      request: { ...allocation.request, metadata: allocation.request.metadata ? { ...allocation.request.metadata } : undefined },
      usage: allocation.usage ? {
        ...allocation.usage,
        availability: allocation.usage.availability ? { ...allocation.usage.availability } : undefined,
      } : undefined,
    };
  }
}
