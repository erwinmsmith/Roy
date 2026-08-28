import { describe, expect, it } from 'vitest';
import { searchOrganizationMCTS } from '../src/core/structural/mctsOrganizationSearch.js';

interface Candidate { id: string; kind: string; delta: number; prior: number }

describe('organization MCTS', () => {
  const candidates: Candidate[] = [
    { id: 'useful-derive', kind: 'DERIVE', delta: 0.35, prior: 0.2 },
    { id: 'neutral-execute', kind: 'EXECUTE', delta: 0, prior: 0.7 },
    { id: 'harmful-connect', kind: 'CONNECT', delta: -0.2, prior: 0.1 },
  ];

  it('uses frozen target-value deltas for PUCT selection and records exact visits', async () => {
    const result = await searchOrganizationMCTS({ rootState: 0.4, candidates,
      simulations: 64, maximumDepth: 2, cpuCT: 1.5, temperature: 1, seed: 7,
      expand: async (state, remaining) => ({ targetValue: state, targetRevision: 3,
        actorPriors: Object.fromEntries(remaining.map(value => [value.id, value.prior])),
        policyState: { state_fingerprint: `state-${state}`, context_node_id: 'root',
          candidates: remaining },
        children: remaining.map(candidate => ({ candidate, state: state + candidate.delta,
          prior: candidate.prior, targetValue: state + candidate.delta, terminal: true })) }) });
    expect(result.visitCounts['useful-derive']).toBeGreaterThan(
      result.visitCounts['neutral-execute']
    );
    expect(result.visitCounts['neutral-execute']).toBeGreaterThan(
      result.visitCounts['harmful-connect']
    );
    expect(Object.values(result.behaviorProbabilities)
      .reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    expect(result.trace.some(value => value.phase === 'selection')).toBe(true);
    expect(result.trace.some(value => value.phase === 'backup')).toBe(true);
    expect(result.rootTargetValue).toBeCloseTo(0.4);
    expect(result.searchSamples).toHaveLength(3);
    expect(result.searchSamples.map(value => Math.sign(Number(value.immediateProcessReward))))
      .toEqual(expect.arrayContaining([-1, 0, 1]));
    expect(result.searchSamples.every(value =>
      value.rewardSource === 'frozen_value_bootstrap')).toBe(true);
  });

  it('does not invent a dense score when the target value is constant', async () => {
    const result = await searchOrganizationMCTS({ rootState: 0.5, candidates,
      simulations: 64, maximumDepth: 2, cpuCT: 1.5, temperature: 1, seed: 9,
      expand: async (state, remaining) => ({ targetValue: state, targetRevision: 0,
        actorPriors: Object.fromEntries(remaining.map(value => [value.id, value.prior])),
        policyState: { state_fingerprint: `state-${state}`, context_node_id: 'root',
          candidates: remaining },
        children: remaining.map(candidate => ({ candidate, state,
          prior: candidate.prior, targetValue: state, terminal: true })) }) });
    expect(result.visitCounts['neutral-execute']).toBeGreaterThan(
      result.visitCounts['useful-derive']
    );
    expect(result.selectedProcessReward).toBe(0);
  });

  it('asks the agent expansion callback for fresh node-local directions at child states',
    async () => {
      type DynamicState = { id: string; value: number; context: string };
      const rootCandidates: Candidate[] = [
        { id: 'derive-worker', kind: 'DERIVE', delta: 0, prior: 1 },
      ];
      const expanded: Array<{ depth: number; context: string; offered: string[] }> = [];
      const result = await searchOrganizationMCTS<DynamicState, Candidate>({
        rootState: { id: 'root-state', value: 0.4, context: 'root' },
        candidates: rootCandidates, simulations: 8, maximumDepth: 2,
        cpuCT: 1.5, temperature: 1, seed: 4,
        expand: async (state, offered, depth) => {
          const directions = depth === 0 ? offered : [
            { id: 'child-acquire', kind: 'ACQUIRE', delta: 0.3, prior: 0.7 },
            { id: 'child-return', kind: 'RETURN', delta: -0.1, prior: 0.3 },
          ];
          expanded.push({ depth, context: state.context,
            offered: directions.map(value => value.id) });
          return {
            targetValue: state.value, targetRevision: 1,
            actorPriors: Object.fromEntries(directions.map(value => [value.id, value.prior])),
            policyState: { state_fingerprint: state.id,
              context_node_id: state.context, candidates: directions },
            expansionMetadata: { proposalSource: depth === 0
              ? 'real_step_agent_proposal' : 'dynamic_agent_search_expansion' },
            children: directions.map(candidate => ({ candidate,
              state: { id: `${state.id}:${candidate.id}`, value: state.value + candidate.delta,
                context: depth === 0 ? 'worker' : state.context },
              prior: candidate.prior, targetValue: state.value + candidate.delta,
              terminal: depth > 0 })),
          };
        },
      });
      expect(expanded).toContainEqual({ depth: 1, context: 'worker',
        offered: ['child-acquire', 'child-return'] });
      expect(result.searchSamples.map(value => value.candidateId))
        .toEqual(expect.arrayContaining(['derive-worker', 'child-acquire', 'child-return']));
      expect(result.searchSamples.filter(value => value.contextNodeId === 'worker')).toHaveLength(2);
      expect(result.trace.some(value =>
        value.proposalSource === 'dynamic_agent_search_expansion')).toBe(true);
    });
});
