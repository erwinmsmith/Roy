import type {
  BudgetAllocationPolicy,
  BudgetDecision,
  BudgetPolicyContext,
  BudgetPriority,
  BudgetRequest,
} from './types.js';

const normalize = (request: BudgetRequest): BudgetRequest => ({
  ...request,
  requestedTokens: Math.max(0, Math.floor(request.requestedTokens)),
  minimumTokens: Math.max(0, Math.floor(request.minimumTokens ?? Math.min(512, request.requestedTokens))),
  expectedUtility: Math.max(0, Math.min(1, request.expectedUtility ?? 0.5)),
  priority: request.priority ?? 'medium',
});

export class UnlimitedBudgetPolicy implements BudgetAllocationPolicy {
  readonly mode = 'unlimited' as const;

  allocate(requests: BudgetRequest[]): BudgetDecision[] {
    return requests.map(raw => {
      const request = normalize(raw);
      return { request, allocatedTokens: request.requestedTokens, rationale: 'unlimited_policy_full_allocation' };
    });
  }
}

export class FixedBudgetPolicy implements BudgetAllocationPolicy {
  readonly mode = 'fixed' as const;

  allocate(requests: BudgetRequest[], context: BudgetPolicyContext): BudgetDecision[] {
    let available = context.availableTokens ?? Number.POSITIVE_INFINITY;
    return requests.map(raw => {
      const request = normalize(raw);
      const allocatedTokens = Math.min(request.requestedTokens, Math.max(0, available));
      const granted = allocatedTokens >= (request.minimumTokens ?? context.minimumGrantTokens);
      if (granted) available -= allocatedTokens;
      return {
        request,
        allocatedTokens: granted ? allocatedTokens : 0,
        rationale: granted
          ? allocatedTokens < request.requestedTokens ? 'fixed_policy_partial_allocation' : 'fixed_policy_full_allocation'
          : 'fixed_policy_insufficient_budget',
      };
    });
  }
}

export class MarketBudgetPolicy implements BudgetAllocationPolicy {
  readonly mode = 'market' as const;

  allocate(requests: BudgetRequest[], context: BudgetPolicyContext): BudgetDecision[] {
    const normalized = requests.map(normalize);
    const scored = normalized.map(request => ({
      request,
      score: this.score(request, context.priorityWeights),
    })).sort((a, b) => b.score - a.score || a.request.requestedTokens - b.request.requestedTokens);

    if (context.availableTokens === undefined) {
      return scored.map(item => ({
        request: item.request,
        allocatedTokens: item.request.requestedTokens,
        score: item.score,
        rationale: 'market_policy_unlimited_supply',
      }));
    }

    let available = context.availableTokens;
    const decisions = new Map<BudgetRequest, BudgetDecision>();
    const totalScore = scored.reduce((sum, item) => sum + item.score, 0) || 1;
    for (const item of scored) {
      const minimum = item.request.minimumTokens ?? context.minimumGrantTokens;
      const proportionalShare = Math.floor(context.availableTokens * (item.score / totalScore));
      const desired = Math.min(item.request.requestedTokens, Math.max(minimum, proportionalShare));
      const allocated = Math.min(desired, available);
      const granted = allocated >= minimum;
      if (granted) available -= allocated;
      decisions.set(item.request, {
        request: item.request,
        allocatedTokens: granted ? allocated : 0,
        score: item.score,
        rationale: granted
          ? allocated < item.request.requestedTokens ? 'market_policy_competitive_partial_allocation' : 'market_policy_full_allocation'
          : 'market_policy_bid_below_available_supply',
      });
    }
    return normalized.map(request => decisions.get(request) ?? {
      request,
      allocatedTokens: 0,
      rationale: 'market_policy_no_allocation',
    });
  }

  private score(request: BudgetRequest, weights: Record<BudgetPriority, number>): number {
    const priority = request.priority ?? 'medium';
    const utility = request.expectedUtility ?? 0.5;
    const costPenalty = Math.min(0.35, request.requestedTokens / 100_000);
    return Math.max(0.01, (weights[priority] ?? 1) * (0.4 + utility * 0.6) - costPenalty);
  }
}
