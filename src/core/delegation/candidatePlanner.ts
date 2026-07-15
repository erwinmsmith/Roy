import type { LLMProvider } from '../llm/types.js';
import { EvolutionEngine, type EvolutionEvaluation, type EvolutionStrategy } from '../evolution/index.js';
import type {
  DelegationCandidate,
  DelegationCandidateInput,
  DelegationCandidateScorer,
  DelegationCandidateSelection,
  DelegationCandidateSource,
} from './types.js';
import {
  CacheEvolutionDelegationScorer,
  CostDelegationScorer,
  HeuristicDelegationScorer,
  LLMDelegationScorer,
  type LLMDelegationScorerHooks,
  ToMDelegationScorer,
} from './scorers.js';
import { ToMDelegationPlanner } from '../tom/index.js';
import {
  WeightedReasoningInvestmentModel,
  type ReasoningInvestmentModel,
} from '../budget/index.js';

const ARCHETYPE_COST: Record<string, number> = {
  researcher: 2200,
  critic: 1600,
  planner: 1400,
  coder: 2600,
  summarizer: 1000,
  tester: 1800,
  custom: 1800,
};

export interface DelegationCandidatePlannerOptions {
  llm?: LLMProvider | null;
  scorers?: DelegationCandidateScorer[];
  minimumScore?: number;
  minimumToMCoverage?: number;
  enabledScorers?: string[];
  llmHooks?: LLMDelegationScorerHooks;
  investmentModel?: ReasoningInvestmentModel;
}

export class DefaultDelegationCandidatePlanner {
  private readonly scorers: DelegationCandidateScorer[];
  private readonly minimumScore: number;
  private readonly minimumToMCoverage: number;
  private readonly tomPlanner = new ToMDelegationPlanner();
  private readonly investmentModel: ReasoningInvestmentModel;

  constructor(options: DelegationCandidatePlannerOptions = {}) {
    const defaultScorers = [
      new HeuristicDelegationScorer(),
      new CostDelegationScorer(),
      new ToMDelegationScorer(),
      new CacheEvolutionDelegationScorer(),
      ...(options.llm ? [new LLMDelegationScorer(options.llm, options.llmHooks)] : []),
    ];
    this.scorers = options.scorers ?? defaultScorers.filter(scorer => !options.enabledScorers || options.enabledScorers.includes(scorer.name));
    this.minimumScore = options.minimumScore ?? 0.05;
    this.minimumToMCoverage = options.minimumToMCoverage ?? 0;
    this.investmentModel = options.investmentModel ?? new WeightedReasoningInvestmentModel();
  }

  async select(input: DelegationCandidateInput): Promise<DelegationCandidateSelection> {
    if (input.decision.action !== 'spawn_subagents') return { candidates: [], decision: input.decision };
    const limit = Math.max(0, Math.min(input.allowedChildren, input.remainingTotalAgentsForTurn, input.decision.agents.length));
    if (limit <= 0) {
      return {
        candidates: [],
        decision: { action: 'solve_directly', reason: `${input.decision.reason} Delegation skipped because no agent slots remain for this turn or parent.` },
        rejectedReason: 'no_agent_slots_remaining',
      };
    }

    const engine = new EvolutionEngine<DelegationCandidateInput, DelegationCandidate>(this.createStrategy(limit));
    const run = await engine.run(input);
    const candidates = run.evaluated.map(item => item.candidate).sort((a, b) => b.score - a.score);
    const selected = run.selected?.candidate;
    const coverageRejected = input.budgetMode === 'unlimited'
      && Math.min(input.allowedChildren, input.remainingTotalAgentsForTurn) > 1
      && selected?.tomCoverage
      && selected.tomCoverage.coverageScore < this.minimumToMCoverage;
    if (!selected || selected.score < this.minimumScore || selected.agents.length === 0 || coverageRejected) {
      return {
        candidates,
        decision: { action: 'solve_directly', reason: `${input.decision.reason} Delegation skipped because no candidate passed policy scoring.` },
        rejectedReason: coverageRejected ? 'tom_coverage_below_minimum' : 'no_candidate_selected',
      };
    }
    return {
      candidates,
      selected,
      decision: {
        action: 'spawn_subagents',
        reason: `${input.decision.reason} Candidate ${selected.id} selected: ${selected.rationale}`,
        agents: selected.agents,
      },
    };
  }

  private createStrategy(limit: number): EvolutionStrategy<DelegationCandidateInput, DelegationCandidate> {
    return {
      propose: input => this.generateCandidates(input, limit),
      evaluate: (candidates, input) => this.evaluateCandidates(candidates, input),
      select: evaluated => [...evaluated].sort((a, b) => b.score - a.score)[0],
    };
  }

  private generateCandidates(input: DelegationCandidateInput, limit: number): DelegationCandidate[] {
    if (input.decision.action !== 'spawn_subagents') return [];
    const bounded = input.decision.agents.slice(0, limit);
    const candidates: DelegationCandidate[] = [];
    if (bounded.length > 0) {
      candidates.push(this.createCandidate('candidate_full_plan', input.parentId, bounded, this.sourceFor(bounded, input.cacheUsed), 'uses the complete bounded delegation plan'));
    }
    if (bounded.length > 1) {
      candidates.push(this.createCandidate(`candidate_single_${bounded[0].archetype}`, input.parentId, [bounded[0]], input.cacheUsed ? 'cache_hit' : bounded[0].archetype === 'custom' ? 'custom_generated' : 'generated', 'uses the highest-priority specialist to reduce cost'));
    }
    const cached = input.cachedPatterns ?? [];
    if (cached.length > 0 && bounded.length > 0) {
      const patternIds = cached.map(pattern => String(pattern.id ?? '')).filter(Boolean);
      const mutated = bounded.map(agent => {
        const pattern = cached.find(item => item.archetype === agent.archetype || item.key === agent.archetype);
        return {
          ...agent,
          tools: this.mergeStringLists(agent.tools, pattern?.tools),
          skills: this.mergeStringLists(agent.skills, pattern?.skills),
        };
      });
      candidates.push({
        ...this.createCandidate('candidate_mutated_cache', input.parentId, mutated, 'mutated_from_cache', 'adapts cached bindings to the current task while preserving fresh runtime instances'),
        lineage: { parentPatternIds: patternIds, mutation: 'merge cached tool and skill bindings into current task plan' },
      });
    }
    return candidates;
  }

  private async evaluateCandidates(
    candidates: DelegationCandidate[],
    input: DelegationCandidateInput
  ): Promise<Array<EvolutionEvaluation<DelegationCandidate>>> {
    const outputs = await Promise.all(this.scorers.map(async scorer => ({
      name: scorer.name,
      values: await scorer.score(candidates, input),
    })));
    return candidates.map(candidate => {
      const breakdown = Object.fromEntries(outputs.map(output => [output.name, Number((output.values.get(candidate.id) ?? 0).toFixed(4))]));
      const tomCoverage = input.tomAnalysis
        ? this.tomPlanner.evaluateCoverage(input.tomAnalysis, candidate.agents)
        : undefined;
      const investment = this.investmentModel.estimate({
        kind: 'delegation_candidate',
        requesterId: candidate.id,
        parentId: candidate.parentId,
        purpose: candidate.rationale,
        resources: {
          tokens: candidate.expectedCostTokens,
          contextTokens: Math.min(4000, Math.round(candidate.expectedCostTokens * 0.35)),
          toolCalls: candidate.agents.reduce((sum, agent) => sum + (agent.tools?.length ?? 0), 0),
        },
        signals: {
          rootUtility: normalizePositive(breakdown.heuristic, Math.max(1, candidate.agents.length * 1.1)),
          parentUtility: breakdown.llm === undefined ? 0.5 : clamp(breakdown.llm),
          historicalUtility: normalizePositive(breakdown.cache_evolution, 0.75),
          evidenceGain: tomCoverage?.coverageScore ?? 0,
          uncertaintyReduction: tomCoverage?.coverageScore ?? 0,
          conflictResolution: candidate.agents.some(agent => agent.archetype === 'summarizer' || agent.archetype === 'critic')
            ? input.tomAnalysis?.signals.conflictLevel ?? 0.35
            : 0,
          verificationGain: candidate.agents.some(agent => agent.archetype === 'tester') ? 0.85 : 0,
          cacheConfidence: candidate.source === 'cache_hit' ? 0.8 : candidate.source === 'mutated_from_cache' ? 0.65 : 0,
          duplicationRisk: this.duplicationRisk(candidate),
          executionRisk: this.executionRisk(candidate, input, tomCoverage),
          confidence: input.tomAnalysis?.confidence ?? 0.62,
        },
        metadata: { source: candidate.source, lineage: candidate.lineage },
      });
      const normalizedReturn = investment.expectedReturn / (1 + investment.expectedReturn);
      const score = investment.riskAdjustedUtility * 1.5
        + normalizedReturn * 0.35
        + clamp(breakdown.llm ?? 0) * 0.35
        + normalizePositive(breakdown.cache_evolution, 0.75) * 0.1;
      breakdown.investment_utility = investment.riskAdjustedUtility;
      breakdown.investment_cost = -investment.costScore;
      breakdown.investment_return = Number(normalizedReturn.toFixed(4));
      breakdown.investment_confidence = investment.confidence;
      const evaluated = {
        ...candidate,
        tomCoverage,
        investment,
        expectedUtility: investment.riskAdjustedUtility,
        score: Number(score.toFixed(4)),
        scoreBreakdown: breakdown,
      };
      return { candidate: evaluated, score: evaluated.score, breakdown };
    });
  }

  private duplicationRisk(candidate: DelegationCandidate): number {
    if (candidate.agents.length <= 1) return 0;
    const uniqueArchetypes = new Set(candidate.agents.map(agent => agent.archetype)).size;
    return clamp(1 - uniqueArchetypes / candidate.agents.length);
  }

  private executionRisk(
    candidate: DelegationCandidate,
    input: DelegationCandidateInput,
    coverage?: DelegationCandidate['tomCoverage']
  ): number {
    const branchRisk = Math.max(0, candidate.agents.length - 1) * 0.12;
    const uncoveredRisk = input.tomAnalysis && coverage
      ? 1 - coverage.coverageScore
      : 0;
    const budgetRisk = input.budgetMode === 'limited' && input.remainingBudgetTokens !== undefined
      ? Math.max(0, candidate.expectedCostTokens - input.remainingBudgetTokens) / Math.max(1, input.remainingBudgetTokens)
      : 0;
    return clamp(branchRisk + uncoveredRisk * 0.35 + budgetRisk * 0.5);
  }

  private createCandidate(
    id: string,
    parentId: string,
    agents: DelegationCandidate['agents'],
    source: DelegationCandidateSource,
    rationale: string
  ): DelegationCandidate {
    return {
      id,
      source,
      parentId,
      agents,
      expectedUtility: 0,
      expectedCostTokens: agents.reduce((sum, agent) => sum + (agent.budgetTokens ?? ARCHETYPE_COST[agent.archetype] ?? 1800), 0),
      score: 0,
      scoreBreakdown: {},
      rationale,
    };
  }

  private sourceFor(agents: DelegationCandidate['agents'], cacheUsed: boolean): DelegationCandidateSource {
    if (cacheUsed && agents.some(agent => agent.archetype === 'custom')) return 'mixed';
    if (cacheUsed) return 'cache_hit';
    if (agents.some(agent => agent.archetype === 'custom')) return 'custom_generated';
    return 'generated';
  }

  private mergeStringLists(primary: string[] | undefined, cached: unknown): string[] | undefined {
    const cachedList = Array.isArray(cached) ? cached.filter((value): value is string => typeof value === 'string') : [];
    const merged = Array.from(new Set([...(primary ?? []), ...cachedList]));
    return merged.length > 0 ? merged : undefined;
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function normalizePositive(value: number | undefined, scale: number): number {
  return clamp(Math.max(0, value ?? 0) / Math.max(0.01, scale));
}
