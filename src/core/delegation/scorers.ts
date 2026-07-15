import type { LLMProvider } from '../llm/types.js';
import type {
  DelegationCandidate,
  DelegationCandidateInput,
  DelegationCandidateScorer,
} from './types.js';
import { HashTaskEmbeddingProvider, type TaskEmbeddingProvider } from './embedding.js';
import { ToMDelegationPlanner } from '../tom/index.js';

const ARCHETYPE_UTILITY: Record<string, number> = {
  researcher: 0.78,
  critic: 0.72,
  planner: 0.62,
  coder: 0.7,
  summarizer: 0.48,
  tester: 0.66,
  custom: 0.58,
};

export class HeuristicDelegationScorer implements DelegationCandidateScorer {
  readonly name = 'heuristic';

  score(candidates: DelegationCandidate[], input: DelegationCandidateInput): Map<string, number> {
    return new Map(candidates.map(candidate => {
      const utility = candidate.agents.reduce((sum, agent) => sum + (ARCHETYPE_UTILITY[agent.archetype] ?? 0.5), 0);
      const taskBonus = candidate.agents.reduce((sum, agent) => sum + this.taskMatch(input.task, agent.archetype), 0);
      const branchPenalty = Math.max(0, candidate.agents.length - 1) * 0.08;
      return [candidate.id, utility + taskBonus - branchPenalty];
    }));
  }

  private taskMatch(task: string, archetype: string): number {
    const lower = task.toLowerCase();
    if (archetype === 'researcher' && /\b(inspect|read|list|evidence|structure)\b/.test(lower)) return 0.16;
    if (archetype === 'critic' && /\b(risk|review|critique|failure|gap)\b/.test(lower)) return 0.16;
    if (archetype === 'tester' && /\b(test|verify|coverage|regression)\b/.test(lower)) return 0.14;
    if (archetype === 'planner' && /\b(plan|steps|roadmap|design)\b/.test(lower)) return 0.12;
    if (archetype === 'coder' && /\b(code|implement|fix|patch)\b/.test(lower)) return 0.14;
    if (archetype === 'custom' && /\b(prompt|slot|context|special)\b/.test(lower)) return 0.12;
    return 0;
  }
}

export class CostDelegationScorer implements DelegationCandidateScorer {
  readonly name = 'cost';

  score(candidates: DelegationCandidate[], input: DelegationCandidateInput): Map<string, number> {
    return new Map(candidates.map(candidate => {
      const budgetPenalty = input.budgetMode === 'limited' && input.remainingBudgetTokens !== undefined
        ? Math.max(0, candidate.expectedCostTokens - input.remainingBudgetTokens) / Math.max(1000, input.remainingBudgetTokens)
        : 0;
      return [candidate.id, -(candidate.expectedCostTokens / 10000) - budgetPenalty];
    }));
  }
}

export class ToMDelegationScorer implements DelegationCandidateScorer {
  readonly name = 'tom';
  private readonly planner = new ToMDelegationPlanner();

  score(candidates: DelegationCandidate[], input: DelegationCandidateInput): Map<string, number> {
    if (input.tomAnalysis) {
      return new Map(candidates.map(candidate => {
        const coverage = this.planner.evaluateCoverage(input.tomAnalysis!, candidate.agents);
        const score = coverage.coverageScore * 1.2
          + coverage.perspectiveDiversity * 0.25
          + coverage.higherOrderFit * 0.3
          - coverage.unjustifiedAgentCount * 0.2;
        return [candidate.id, Number(score.toFixed(4))];
      }));
    }
    const parentLevel = input.parentToMProfile?.level ?? 0;
    const modeledTargets = new Set(input.parentToMProfile?.models.map(model => model.targetId) ?? []);
    return new Map(candidates.map(candidate => {
      const levels = new Set(candidate.agents.map(agent => agent.tomLevel ?? 0));
      const criticBonus = candidate.agents.some(agent => agent.archetype === 'critic') ? 0.12 : 0;
      const diversityBonus = Math.max(0, levels.size - 1) * 0.07;
      const recursiveFit = parentLevel >= 1 && candidate.agents.some(agent => (agent.tomLevel ?? 0) >= 1) ? 0.08 : 0;
      const targetBonus = modeledTargets.has('user') ? 0.03 : 0;
      return [candidate.id, criticBonus + diversityBonus + recursiveFit + targetBonus];
    }));
  }
}

export class CacheEvolutionDelegationScorer implements DelegationCandidateScorer {
  readonly name = 'cache_evolution';

  constructor(private readonly embeddings: TaskEmbeddingProvider = new HashTaskEmbeddingProvider()) {}

  score(candidates: DelegationCandidate[], input: DelegationCandidateInput): Map<string, number> {
    const patterns = input.cachedPatterns ?? [];
    return new Map(candidates.map(candidate => {
      const similarities = patterns.map(pattern => {
        const signature = String(pattern.taskSignature ?? pattern.signature ?? pattern.description ?? '');
        return signature ? this.embeddings.similarity(input.task, signature.replaceAll('_', ' ')) : 0;
      });
      const similarity = similarities.length > 0 ? Math.max(...similarities) : 0;
      const reuseBonus = candidate.source === 'cache_hit' ? 0.14 : 0;
      const mutationBonus = candidate.source === 'mutated_from_cache' ? 0.24 : 0;
      return [candidate.id, similarity * 0.35 + reuseBonus + mutationBonus];
    }));
  }
}

interface LLMScoreResponse {
  scores?: Array<{ candidateId: string; score: number; reason?: string }>;
}

export class LLMDelegationScorer implements DelegationCandidateScorer {
  readonly name = 'llm';

  constructor(private readonly llm: LLMProvider) {}

  async score(candidates: DelegationCandidate[], input: DelegationCandidateInput): Promise<Map<string, number>> {
    if (candidates.length === 0) return new Map();
    try {
      const response = await this.llm.completeJSON<LLMScoreResponse>([
        {
          role: 'system',
          content: `You are Roy's delegation candidate evaluator. Score each candidate from 0 to 1 for task fit, role complementarity, grounding, recursive delegation safety, and expected value. Return strict JSON: {"scores":[{"candidateId":"...","score":0.0,"reason":"..."}]}`,
        },
        {
          role: 'user',
          content: JSON.stringify({
            task: input.task,
            budgetMode: input.budgetMode,
            remainingBudgetTokens: input.remainingBudgetTokens,
            parentToMProfile: input.parentToMProfile,
            candidates: candidates.map(candidate => ({
              id: candidate.id,
              source: candidate.source,
              agents: candidate.agents,
              expectedCostTokens: candidate.expectedCostTokens,
              lineage: candidate.lineage,
            })),
          }),
        },
      ], { temperature: 0, maxTokens: 700 });
      const scores = Array.isArray(response?.scores) ? response.scores : [];
      return new Map(scores
        .filter(item => candidates.some(candidate => candidate.id === item.candidateId) && Number.isFinite(item.score))
        .map(item => [item.candidateId, Math.max(0, Math.min(1, item.score))]));
    } catch {
      return new Map();
    }
  }
}
