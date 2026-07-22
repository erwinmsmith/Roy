import type {
  EvolutionCandidate,
  EvolutionCandidateEvaluator,
  EvolutionEvaluationDimensions,
  EvolutionEvaluationResult,
  EvolutionExecutionArtifact,
  EvolutionJudge,
  EvolutionSelectionPolicy,
} from './types.js';

export class CompositeEvolutionEvaluator implements EvolutionCandidateEvaluator {
  readonly name = 'composite';

  constructor(private readonly judge?: EvolutionJudge) {}

  async evaluate(
    task: string,
    candidate: EvolutionCandidate,
    execution: EvolutionExecutionArtifact
  ): Promise<EvolutionEvaluationResult> {
    const validExecution = execution.success && execution.unresolvedToolIntents === 0;
    const explicitNonCompletion = indicatesNonCompletion(execution.result);
    const taskCompleted = validExecution && !explicitNonCompletion;
    const toolSuccessRate = execution.toolCalls > 0
      ? execution.successfulToolCalls / execution.toolCalls
      : candidate.genome.members.some(member => member.outputContract.groundingRequired) ? 0 : 0.75;
    const groundedRate = execution.totalResults > 0 ? execution.groundedResults / execution.totalResults : 0;
    const successRate = execution.agentIds.length > 0
      ? Math.max(0, 1 - execution.failedActors / execution.agentIds.length)
      : execution.success ? 1 : 0;
    const expected = Math.max(1, candidate.expectedCostTokens);
    const costEfficiency = clamp(expected / Math.max(expected, execution.usage.totalTokens));
    const roleDiversity = new Set(candidate.genome.members.map(member => member.role)).size / candidate.genome.members.length;
    const tomCoverage = candidate.genome.members.length === 0
      ? 0
      : candidate.genome.members.reduce((sum, member) => sum + member.tomProfile.level / 3, 0) / candidate.genome.members.length;
    const base: EvolutionEvaluationDimensions = {
      taskSuccess: taskCompleted ? Math.max(0.5, successRate) : validExecution ? 0.25 : 0,
      answerQuality: validExecution && execution.result.trim().length > 0
        ? Math.min(this.judge ? 1 : 0.72, 0.45 + Math.log10(execution.result.length + 1) / 5)
        : 0,
      completeness: taskCompleted
        ? clamp((successRate + groundedRate + 1) / 3)
        : validExecution ? clamp((successRate + groundedRate) / 4) : 0,
      costEfficiency,
      novelty: candidate.source === 'mutated_from_cache' || candidate.lineage.operators.length > 0 ? 0.8 : candidate.source === 'cache_hit' ? 0.35 : 0.55,
      toolUse: clamp(toolSuccessRate * 0.55 + groundedRate * 0.45),
      consistency: validExecution
        ? clamp(successRate - execution.warnings.length * 0.08 + (roleDiversity >= 0.5 ? 0.15 : 0))
        : 0,
      tomCoverage,
    };
    let judged: Partial<EvolutionEvaluationDimensions> & { rationale?: string } = {};
    let judgeFailure: string | undefined;
    if (this.judge) {
      try {
        judged = await this.judge.evaluate(task, candidate, execution);
      } catch (error) {
        judgeFailure = error instanceof Error ? error.message : String(error);
      }
    }
    const dimensions = this.merge(base, judged);
    const compositeScore = clamp(
      dimensions.taskSuccess * 0.22
      + dimensions.answerQuality * 0.18
      + dimensions.completeness * 0.14
      + dimensions.costEfficiency * 0.14
      + dimensions.toolUse * 0.12
      + dimensions.consistency * 0.1
      + dimensions.tomCoverage * 0.07
      + dimensions.novelty * 0.03
    );
    const score = validExecution ? compositeScore : Math.min(0.15, compositeScore);
    return {
      candidateId: candidate.id,
      score: Number(score.toFixed(4)),
      dimensions,
      tokenUsed: execution.usage.totalTokens,
      success: validExecution && dimensions.taskSuccess >= 0.5,
      rationale: explicitNonCompletion
        ? 'The execution explicitly reported that the requested result could not be verified or completed.'
        : judgeFailure
        ? `LLM judge failed (${judgeFailure}); composite observable metrics were used.`
        : typeof judged.rationale === 'string'
        ? judged.rationale
        : 'Observable execution score from completion, evidence, cost, consistency, and ToM coverage. Semantic correctness was not independently judged.',
      evaluator: this.judge ? `${this.name}+${this.judge.name}` : `${this.name}_observable`,
    };
  }

  private merge(
    base: EvolutionEvaluationDimensions,
    judged: Partial<EvolutionEvaluationDimensions>
  ): EvolutionEvaluationDimensions {
    return Object.fromEntries(Object.entries(base).map(([key, value]) => {
      const judgeValue = judged[key as keyof EvolutionEvaluationDimensions];
      return [key, judgeValue === undefined ? value : clamp(value * 0.55 + judgeValue * 0.45)];
    })) as unknown as EvolutionEvaluationDimensions;
  }
}

export class WeightedTopKSelectionPolicy implements EvolutionSelectionPolicy {
  readonly name = 'weighted_top_k';

  select(
    candidates: EvolutionCandidate[],
    evaluations: EvolutionEvaluationResult[],
    topK: number
  ): EvolutionCandidate[] {
    const scoreById = new Map(evaluations.map(evaluation => [evaluation.candidateId, evaluation.score]));
    return [...candidates]
      .filter(candidate => scoreById.has(candidate.id))
      .sort((left, right) => (scoreById.get(right.id) ?? 0) - (scoreById.get(left.id) ?? 0))
      .slice(0, Math.max(1, topK));
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function indicatesNonCompletion(result: string): boolean {
  return result.includes('[runtime_team_synthesis_fallback]')
    || /\b(?:no\s+(?:grounded|concrete|verified)\b[^.\n]{0,100}\b(?:identified|found|available|verified)|(?:cannot|could\s+not|unable\s+to)\s+(?:identify|provide|complete)\b[^.\n]{0,100}\b(?:requested|concrete|specific|task|answer|result)|insufficient\s+(?:grounded\s+)?evidence\s+to\s+(?:identify|provide|complete|answer)|no\s+concrete\s+(?:architecture\s+)?risk\s+can\s+be\s+verified)\b/i.test(result);
}
