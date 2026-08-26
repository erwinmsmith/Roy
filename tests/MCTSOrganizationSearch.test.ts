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
  });

  it('does not invent a dense score when the target value is constant', async () => {
    const result = await searchOrganizationMCTS({ rootState: 0.5, candidates,
      simulations: 64, maximumDepth: 2, cpuCT: 1.5, temperature: 1, seed: 9,
      expand: async (state, remaining) => ({ targetValue: state, targetRevision: 0,
        actorPriors: Object.fromEntries(remaining.map(value => [value.id, value.prior])),
        children: remaining.map(candidate => ({ candidate, state,
          prior: candidate.prior, targetValue: state, terminal: true })) }) });
    expect(result.visitCounts['neutral-execute']).toBeGreaterThan(
      result.visitCounts['useful-derive']
    );
    expect(result.selectedProcessReward).toBe(0);
  });
});
