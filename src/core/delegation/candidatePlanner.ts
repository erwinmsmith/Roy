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
}

export class DefaultDelegationCandidatePlanner {
  private readonly scorers: DelegationCandidateScorer[];
  private readonly minimumScore: number;
  private readonly minimumToMCoverage: number;
  private readonly tomPlanner = new ToMDelegationPlanner();

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
      const score = Object.values(breakdown).reduce((sum, value) => sum + value, 0);
      const expectedUtility = Math.max(0, (breakdown.heuristic ?? 0) + (breakdown.tom ?? 0) + (breakdown.cache_evolution ?? 0));
      const evaluated = {
        ...candidate,
        tomCoverage: input.tomAnalysis
          ? this.tomPlanner.evaluateCoverage(input.tomAnalysis, candidate.agents)
          : undefined,
        expectedUtility: Number(expectedUtility.toFixed(4)),
        score: Number(score.toFixed(4)),
        scoreBreakdown: breakdown,
      };
      return { candidate: evaluated, score: evaluated.score, breakdown };
    });
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
