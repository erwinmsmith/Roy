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
  minimumTokens: Math.min(
    Math.max(0, Math.floor(request.requestedTokens)),
    Math.max(0, Math.floor(request.minimumTokens ?? Math.min(512, request.requestedTokens)))
  ),
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

    // Admission pass: guarantee the minimum viable grant for the strongest affordable bids.
    for (const item of scored) {
      const minimum = item.request.minimumTokens ?? context.minimumGrantTokens;
      const granted = item.request.requestedTokens === 0 || available >= minimum;
      const allocated = granted ? Math.min(minimum, item.request.requestedTokens) : 0;
      available -= allocated;
      decisions.set(item.request, {
        request: item.request,
        allocatedTokens: allocated,
        score: item.score,
        rationale: granted ? 'market_policy_minimum_admitted' : 'market_policy_minimum_not_affordable',
      });
    }

    // Clearing pass: repeatedly distribute remaining supply by marginal score.
    while (available > 0) {
      const active = scored.filter(item => {
        const decision = decisions.get(item.request)!;
        return decision.allocatedTokens > 0 && decision.allocatedTokens < item.request.requestedTokens;
      });
      if (active.length === 0) break;
      const marginalWeights = new Map(active.map(item => {
        const decision = decisions.get(item.request)!;
        const unmet = item.request.requestedTokens - decision.allocatedTokens;
        return [item.request, item.score * Math.sqrt(Math.max(1, unmet))];
      }));
      const totalScore = [...marginalWeights.values()].reduce((sum, score) => sum + score, 0) || 1;
      const supplyAtStart = available;
      let distributed = 0;
      for (const item of active) {
        if (available <= 0) break;
        const decision = decisions.get(item.request)!;
        const unmet = item.request.requestedTokens - decision.allocatedTokens;
        const share = Math.max(1, Math.floor(supplyAtStart * ((marginalWeights.get(item.request) ?? item.score) / totalScore)));
        const increment = Math.min(unmet, share, available);
        decision.allocatedTokens += increment;
        available -= increment;
        distributed += increment;
      }
      if (distributed === 0) break;
    }

    for (const item of scored) {
      const decision = decisions.get(item.request)!;
      if (decision.allocatedTokens === 0) continue;
      decision.rationale = decision.allocatedTokens < item.request.requestedTokens
        ? 'market_policy_competitive_partial_allocation'
        : 'market_policy_full_allocation';
    }
    return normalized.map(request => decisions.get(request) ?? {
      request,
      allocatedTokens: 0,
      rationale: 'market_policy_no_allocation',
    });
  }

  private score(request: BudgetRequest, weights: Record<BudgetPriority, number>): number {
    const priority = request.priority ?? 'medium';
    const utility = request.investment?.riskAdjustedUtility ?? request.expectedUtility ?? 0.5;
    const expectedReturn = request.investment?.expectedReturn ?? utility;
    const normalizedReturn = expectedReturn / (1 + Math.max(0, expectedReturn));
    const confidence = request.investment?.confidence ?? 0.65;
    const costPenalty = Math.min(0.35, request.requestedTokens / 100_000);
    return Math.max(
      0.01,
      (weights[priority] ?? 1) * (0.2 + utility * 0.45 + normalizedReturn * 0.25 + confidence * 0.1) - costPenalty
    );
  }
}
