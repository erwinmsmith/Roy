import { describe, expect, it } from 'vitest';
import {
  CompositeEvolutionEvaluator,
  EvolutionLifecycleEngine,
  EvolutionStateMachine,
  InvalidEvolutionTransitionError,
  TeamFirstGenomePlanner,
  WeightedTopKSelectionPolicy,
  defaultMutationOperators,
  type EvolutionExecutionArtifact,
  type EvolutionProposalInput,
} from '../src/core/evolution/index.js';

function proposalInput(): EvolutionProposalInput {
  return {
    runId: 'evo_test',
    task: 'Inspect the repository and identify architecture risks.',
    parentId: 'root',
    agents: [
      {
        archetype: 'researcher',
        task: 'Inspect repository structure.',
        tools: ['fs.list', 'fs.read'],
        skills: ['use_tool_when_needed'],
        budgetTokens: 1200,
        tomLevel: 0,
        groundingRequired: true,
      },
      {
        archetype: 'critic',
        task: 'Identify architecture risks.',
        tools: ['fs.read'],
        skills: ['use_tool_when_needed'],
        budgetTokens: 900,
        tomLevel: 2,
      },
    ],
    patterns: [],
    availableTokens: 6000,
    availableAgentSlots: 6,
    options: {
      profile: 'evo_team',
      populationSize: 3,
      generations: 1,
      topK: 1,
      maxExecutedCandidates: 2,
      integrationMinimumScore: 0.4,
      patternSimilarityThreshold: 0.3,
      useLlmJudge: false,
      ablations: {
        withoutSubagents: false,
        withoutToMProfile: false,
        withoutBudgetMarket: false,
        withoutEvoMutation: false,
        withoutPatternMemory: false,
      },
    },
  };
}

describe('Phase 6 evolution core', () => {
  it('enforces the evolution FSM', () => {
    const fsm = new EvolutionStateMachine();
    expect(() => fsm.transition('S_evo_execute')).toThrow(InvalidEvolutionTransitionError);
    fsm.transition('S_evo_propose');
    fsm.transition('S_evo_instantiate');
    fsm.transition('S_evo_execute');
    fsm.transition('S_evo_evaluate');
    fsm.transition('S_evo_select');
    fsm.transition('S_evo_integrate');
    expect(fsm.transition('S_evo_done').to).toBe('S_evo_done');
  });

  it('proposes team-first genomes and represents the solo specialist as a one-member team', () => {
    const candidates = new TeamFirstGenomePlanner().propose(proposalInput());
    expect(candidates[0].genome.members).toHaveLength(2);
    expect(candidates[0].genome.coordinationPolicy).toBe('critic_refine');
    expect(candidates[1].genome.members).toHaveLength(1);
    expect(candidates[1].rationale).toContain('degenerates');
  });

  it('executes, evaluates, mutates, selects, and integrates through the full lifecycle', async () => {
    const transitions: string[] = [];
    const integrated: string[] = [];
    const engine = new EvolutionLifecycleEngine(
      new TeamFirstGenomePlanner(),
      new CompositeEvolutionEvaluator(),
      new WeightedTopKSelectionPolicy(),
      defaultMutationOperators(),
      {
        onTransition: (_from, to) => transitions.push(to),
        instantiate: async () => undefined,
        execute: async candidate => ({
          candidateId: candidate.id,
          actorKind: candidate.genome.members.length === 1 ? 'agent' : 'team',
          actorId: `actor_${candidate.id}`,
          success: true,
          result: 'Grounded architecture findings with evidence and limitations.',
          usage: {
            inputTokens: 200,
            outputTokens: 100,
            thinkingTokens: 30,
            totalTokens: 330,
          },
          wallClockMs: 20,
          agentIds: candidate.genome.members.map(member => member.id),
          teamIds: candidate.genome.members.length > 1 ? [candidate.genome.id] : [],
          toolCalls: 2,
          successfulToolCalls: 2,
          unresolvedToolIntents: 0,
          groundedResults: candidate.genome.members.length,
          totalResults: candidate.genome.members.length,
          failedActors: 0,
          recoveredFailures: 0,
          warnings: [],
        } satisfies EvolutionExecutionArtifact),
        integrate: async selected => {
          integrated.push(selected.id);
          return `pattern_${selected.genome.id}`;
        },
      }
    );
    const result = await engine.run(proposalInput());
    expect(result.state).toBe('S_evo_done');
    expect(result.executions.length).toBeGreaterThanOrEqual(2);
    expect(result.evaluations.every(item => item.score > 0.5)).toBe(true);
    expect(result.candidates.some(candidate => candidate.lineage.operators.length > 0)).toBe(true);
    expect(result.integratedPatternId).toContain('pattern_');
    expect(integrated).toHaveLength(1);
    expect(transitions).toContain('S_evo_mutate');
    expect(transitions.at(-1)).toBe('S_evo_done');
  });

  it('honors the total agent-slot limit across all generations', async () => {
    const input = proposalInput();
    input.availableAgentSlots = 2;
    const executed: number[] = [];
    const engine = new EvolutionLifecycleEngine(
      new TeamFirstGenomePlanner(),
      new CompositeEvolutionEvaluator(),
      new WeightedTopKSelectionPolicy(),
      defaultMutationOperators(),
      {
        instantiate: async candidate => executed.push(candidate.genome.members.length),
        execute: async candidate => ({
          candidateId: candidate.id, actorKind: 'team', actorId: candidate.id,
          success: true, result: 'ok',
          usage: { inputTokens: 1, outputTokens: 1, thinkingTokens: null, totalTokens: 2 },
          wallClockMs: 1, agentIds: candidate.genome.members.map(member => member.id), teamIds: [candidate.genome.id],
          toolCalls: 0, successfulToolCalls: 0, unresolvedToolIntents: 0, groundedResults: 0,
          totalResults: candidate.genome.members.length, failedActors: 0, recoveredFailures: 0, warnings: [],
        }),
        integrate: async () => undefined,
      }
    );
    await engine.run(input);
    expect(executed.reduce((sum, count) => sum + count, 0)).toBeLessThanOrEqual(2);
  });

  it('rejects long outputs that only contain an unexecuted tool request', async () => {
    const candidate = new TeamFirstGenomePlanner().propose(proposalInput())[0];
    const evaluation = await new CompositeEvolutionEvaluator().evaluate(
      proposalInput().task,
      candidate,
      {
        candidateId: candidate.id,
        actorKind: 'team',
        actorId: 'invalid_tool_team',
        success: true,
        result: '<tool_call><tool_name>shell.exec</tool_name><arguments>{"command":"ls"}</arguments></tool_call>'.repeat(20),
        usage: { inputTokens: 100, outputTokens: 100, thinkingTokens: 20, totalTokens: 220 },
        wallClockMs: 5,
        agentIds: ['agent_invalid'],
        teamIds: ['team_invalid'],
        toolCalls: 0,
        successfulToolCalls: 0,
        unresolvedToolIntents: 1,
        groundedResults: 0,
        totalResults: 1,
        failedActors: 0,
        recoveredFailures: 0,
        warnings: ['Tool request was not executed.'],
      }
    );
    expect(evaluation.success).toBe(false);
    expect(evaluation.score).toBeLessThanOrEqual(0.15);
    expect(evaluation.dimensions.answerQuality).toBe(0);
  });
});
