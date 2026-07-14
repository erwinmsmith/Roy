export type BudgetPriority = 'low' | 'normal' | 'high' | 'critical';

export interface BudgetRequest {
  requesterId: string;
  parentId: string;
  correlationId?: string;
  requestedTokens: number;
  minimumTokens?: number;
  expectedUtility?: number;
  priority?: BudgetPriority;
  purpose: string;
}

export interface BudgetAllocation {
  id: string;
  request: BudgetRequest;
  status: 'granted' | 'denied' | 'settled' | 'released';
  grantedTokens: number;
  actualTokens?: number;
  reason: string;
  createdAt: number;
  updatedAt: number;
}

export interface BudgetMarketState {
  mode: 'unlimited' | 'limited';
  limitTokens?: number;
  usedTokens: number;
  reservedTokens: number;
  availableTokens?: number;
  allocations: BudgetAllocation[];
}

export class BudgetMarket {
  private limitTokens: number | null = null;
  private allocations = new Map<string, BudgetAllocation>();
  private sequence = 0;

  constructor(private readonly usedTokens: () => number) {}

  configure(limitTokens: number | null): void {
    this.limitTokens = limitTokens;
  }

  request(input: BudgetRequest): BudgetAllocation {
    const now = Date.now();
    const requested = Math.max(0, Math.floor(input.requestedTokens));
    const minimum = Math.max(0, Math.floor(input.minimumTokens ?? Math.min(512, requested)));
    const available = this.availableTokens();
    const grantedTokens = available === undefined ? requested : Math.min(requested, available);
    const granted = available === undefined || grantedTokens >= minimum;
    const allocation: BudgetAllocation = {
      id: `budget_alloc_${now}_${String(++this.sequence).padStart(4, '0')}`,
      request: { ...input, requestedTokens: requested, minimumTokens: minimum },
      status: granted ? 'granted' : 'denied',
      grantedTokens: granted ? grantedTokens : 0,
      reason: granted
        ? grantedTokens < requested ? 'partial_grant_due_to_remaining_budget' : 'grant_within_budget'
        : 'insufficient_remaining_budget',
      createdAt: now,
      updatedAt: now,
    };
    this.allocations.set(allocation.id, allocation);
    return { ...allocation, request: { ...allocation.request } };
  }

  settle(allocationId: string, actualTokens: number): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || allocation.status !== 'granted') return undefined;
    allocation.status = 'settled';
    allocation.actualTokens = Math.max(0, Math.floor(actualTokens));
    allocation.updatedAt = Date.now();
    return { ...allocation, request: { ...allocation.request } };
  }

  release(allocationId: string, reason = 'released_without_execution'): BudgetAllocation | undefined {
    const allocation = this.allocations.get(allocationId);
    if (!allocation || allocation.status !== 'granted') return undefined;
    allocation.status = 'released';
    allocation.reason = reason;
    allocation.updatedAt = Date.now();
    return { ...allocation, request: { ...allocation.request } };
  }

  getState(): BudgetMarketState {
    const usedTokens = this.usedTokens();
    const reservedTokens = this.reservedTokens();
    return {
      mode: this.limitTokens === null ? 'unlimited' : 'limited',
      limitTokens: this.limitTokens ?? undefined,
      usedTokens,
      reservedTokens,
      availableTokens: this.limitTokens === null ? undefined : Math.max(0, this.limitTokens - usedTokens - reservedTokens),
      allocations: [...this.allocations.values()].map(item => ({ ...item, request: { ...item.request } })),
    };
  }

  private availableTokens(): number | undefined {
    if (this.limitTokens === null) return undefined;
    return Math.max(0, this.limitTokens - this.usedTokens() - this.reservedTokens());
  }

  private reservedTokens(): number {
    return [...this.allocations.values()]
      .filter(item => item.status === 'granted')
      .reduce((sum, item) => sum + item.grantedTokens, 0);
  }
}
