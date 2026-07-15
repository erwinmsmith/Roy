import { EvolutionStateMachine } from './stateMachine.js';
import type {
  EvolutionCandidate,
  EvolutionCandidateEvaluator,
  EvolutionEvaluationResult,
  EvolutionExecutionArtifact,
  EvolutionLifecycleHooks,
  EvolutionProposalInput,
  EvolutionSelectionPolicy,
  GenomeMutationOperator,
} from './types.js';
import { validateTeamGenome } from './genome.js';
import type { TeamFirstGenomePlanner } from './genome.js';

export interface EvolutionLifecycleResult {
  state: ReturnType<EvolutionStateMachine['getState']>;
  candidates: EvolutionCandidate[];
  executions: EvolutionExecutionArtifact[];
  evaluations: EvolutionEvaluationResult[];
  selected?: EvolutionCandidate;
  selectedExecution?: EvolutionExecutionArtifact;
  selectedEvaluation?: EvolutionEvaluationResult;
  integratedPatternId?: string;
}

export class EvolutionLifecycleEngine {
  constructor(
    private readonly planner: TeamFirstGenomePlanner,
    private readonly evaluator: EvolutionCandidateEvaluator,
    private readonly selector: EvolutionSelectionPolicy,
    private readonly operators: GenomeMutationOperator[],
    private readonly hooks: EvolutionLifecycleHooks
  ) {}

  async run(input: EvolutionProposalInput): Promise<EvolutionLifecycleResult> {
    const fsm = new EvolutionStateMachine();
    const candidates: EvolutionCandidate[] = [];
    const executions: EvolutionExecutionArtifact[] = [];
    const evaluations: EvolutionEvaluationResult[] = [];
    try {
      await this.transition(fsm, 'S_evo_propose', { runId: input.runId });
      const proposed = this.planner.propose(input);
      candidates.push(...proposed);
      let generationCandidates = proposed;
      let selected: EvolutionCandidate | undefined;
      let consumedAgentSlots = 0;

      for (let generation = 0; generation <= input.options.generations; generation += 1) {
        if (generationCandidates.length === 0) break;
        const executable: EvolutionCandidate[] = [];
        const preflightFailures: EvolutionExecutionArtifact[] = [];
        for (const candidate of generationCandidates) {
          if (executable.length >= input.options.maxExecutedCandidates) break;
          try {
            validateTeamGenome(candidate.genome);
          } catch (error) {
            const reason = error instanceof Error ? error.message : String(error);
            preflightFailures.push(this.rejectedExecution(candidate, reason));
            await this.hooks.onCandidateRejected?.(candidate, reason);
            continue;
          }
          const requiredSlots = candidate.genome.members.length;
          if (input.availableAgentSlots !== undefined
            && consumedAgentSlots + requiredSlots > input.availableAgentSlots) continue;
          executable.push(candidate);
          consumedAgentSlots += requiredSlots;
        }
        if (executable.length === 0 && preflightFailures.length === 0) break;
        await this.transition(fsm, 'S_evo_instantiate', {
          generation,
          count: executable.length,
          rejected: preflightFailures.length,
        });
        for (const candidate of executable) {
          await this.hooks.instantiate(candidate);
        }

        await this.transition(fsm, 'S_evo_execute', { generation, count: executable.length });
        const generationExecutions: EvolutionExecutionArtifact[] = [...preflightFailures];
        executions.push(...preflightFailures);
        for (const candidate of executable) {
          const execution = await this.hooks.execute(candidate);
          executions.push(execution);
          generationExecutions.push(execution);
        }

        await this.transition(fsm, 'S_evo_evaluate', { generation, count: generationExecutions.length });
        const generationEvaluations = await Promise.all(generationExecutions.map(execution => {
          const candidate = generationCandidates.find(item => item.id === execution.candidateId);
          if (!candidate) throw new Error(`Missing candidate for execution ${execution.candidateId}`);
          return this.evaluator.evaluate(input.task, candidate, execution);
        }));
        evaluations.push(...generationEvaluations);
        for (const evaluation of generationEvaluations) {
          const candidate = generationCandidates.find(item => item.id === evaluation.candidateId);
          if (candidate) candidate.expectedUtility = evaluation.score;
        }

        await this.transition(fsm, 'S_evo_select', { generation, count: generationEvaluations.length });
        const successfulGenerationIds = new Set(
          generationEvaluations.filter(evaluation => evaluation.success).map(evaluation => evaluation.candidateId)
        );
        const selectedGeneration = this.selector.select(
          generationCandidates.filter(candidate => successfulGenerationIds.has(candidate.id)),
          generationEvaluations.filter(evaluation => evaluation.success),
          input.options.topK
        );
        selected = selectedGeneration[0] ?? selected;

        if (generation >= input.options.generations || input.options.ablations.withoutEvoMutation) break;
        await this.transition(fsm, 'S_evo_mutate', { generation: generation + 1, parents: selectedGeneration.length });
        generationCandidates = this.planner.mutate(selectedGeneration, input, this.operators, generation + 1);
        candidates.push(...generationCandidates);
      }

      const executedIds = new Set(executions.map(execution => execution.candidateId));
      const successfulIds = new Set(
        evaluations.filter(evaluation => evaluation.success).map(evaluation => evaluation.candidateId)
      );
      const globalSelection = this.selector.select(
        candidates.filter(candidate => executedIds.has(candidate.id) && successfulIds.has(candidate.id)),
        evaluations.filter(evaluation => evaluation.success),
        input.options.topK
      );
      selected = globalSelection[0];
      if (!selected) throw new Error('Evolution produced no successful candidate');
      const selectedEvaluation = [...evaluations]
        .filter(item => item.candidateId === selected!.id)
        .sort((left, right) => right.score - left.score)[0];
      const selectedExecution = [...executions].reverse().find(item => item.candidateId === selected!.id);
      if (!selectedEvaluation || !selectedExecution) throw new Error('Selected candidate has no completed evaluation');

      await this.transition(fsm, 'S_evo_integrate', { candidateId: selected.id, score: selectedEvaluation.score });
      const integratedPatternId = selectedEvaluation.score >= input.options.integrationMinimumScore
        ? await this.hooks.integrate(selected, selectedEvaluation, selectedExecution)
        : undefined;
      await this.transition(fsm, 'S_evo_done', { candidateId: selected.id, integratedPatternId });
      return {
        state: fsm.getState(), candidates, executions, evaluations, selected,
        selectedExecution, selectedEvaluation, integratedPatternId,
      };
    } catch (error) {
      if (fsm.getState() !== 'S_evo_failed') {
        await this.transition(fsm, 'S_evo_failed', { error: error instanceof Error ? error.message : String(error) });
      }
      throw error;
    }
  }

  private async transition(
    fsm: EvolutionStateMachine,
    next: Parameters<EvolutionStateMachine['transition']>[0],
    data?: Record<string, unknown>
  ): Promise<void> {
    const { from, to } = fsm.transition(next);
    await this.hooks.onTransition?.(from, to, data);
  }

  private rejectedExecution(candidate: EvolutionCandidate, reason: string): EvolutionExecutionArtifact {
    return {
      candidateId: candidate.id,
      actorKind: candidate.genome.members.length === 1 ? 'agent' : 'team',
      actorId: 'preflight-rejected',
      success: false,
      result: '',
      usage: { inputTokens: 0, outputTokens: 0, thinkingTokens: null, totalTokens: 0 },
      wallClockMs: 0,
      agentIds: [],
      teamIds: [],
      toolCalls: 0,
      successfulToolCalls: 0,
      unresolvedToolIntents: 0,
      groundedResults: 0,
      totalResults: candidate.genome.members.length,
      failedActors: candidate.genome.members.length,
      recoveredFailures: 0,
      warnings: [`Genome preflight rejected: ${reason}`],
    };
  }
}
