import { describe, expect, it } from 'vitest';
import { ToMDelegationPlanner, normalizeToMProfile } from '../src/core/tom/index.js';
import { ToMDelegationScorer } from '../src/core/delegation/scorers.js';
import type { DelegationCandidate, DelegationCandidateInput } from '../src/core/delegation/types.js';
import { UnifiedAgent } from '../src/core/agent/UnifiedAgent.js';

const rootProfile = normalizeToMProfile({
  level: 1,
  subjectAgentId: 'root',
  beliefScope: ['user intent', 'runtime constraints'],
  goalModel: ['Produce a grounded architecture assessment.'],
  uncertainty: ['Whether the current runtime has hidden coupling.'],
  perspective: 'root coordinator',
  observesAgents: [],
  modelsAgents: ['user'],
  capabilityScope: ['delegation', 'synthesis'],
  cognitiveGaps: [],
  models: [{ targetId: 'user', targetType: 'user' }],
  purpose: 'Coordinate reliable task completion.',
}, {
  level: 1,
  subjectAgentId: 'root',
  purpose: 'Coordinate reliable task completion.',
});

describe('ToM-aware delegation', () => {
  it('uses the normalized profile level as the single identity source of truth', () => {
    const agent = new UnifiedAgent({
      name: 'ProfileAgent',
      role: 'subagent',
      tomLevel: 0,
      tomProfile: { ...rootProfile, level: 2, subjectAgentId: 'profile-agent' },
    });
    expect(agent.getIdentity().tomLevel).toBe(2);
    expect(agent.getIdentity().tomProfile.level).toBe(2);
  });

  it('turns a complex task into explicit evidence, risk, and synthesis gaps', () => {
    const planner = new ToMDelegationPlanner();
    const analysis = planner.analyzeTask({
      parentId: 'root',
      parentProfile: rootProfile,
      task: 'Inspect this repository architecture and identify design risks.',
    });

    expect(analysis.gaps.map(gap => gap.kind)).toEqual(expect.arrayContaining(['evidence', 'risk', 'synthesis']));
    expect(analysis.requiresHigherOrderToM).toBe(true);
    expect(analysis.parentUncertainties).toContain('Whether the current runtime has hidden coupling.');
  });

  it('completes a partial role plan with gap-backed profiles and existence reasons', () => {
    const planner = new ToMDelegationPlanner();
    const analysis = planner.analyzeTask({
      parentId: 'root',
      parentProfile: rootProfile,
      task: 'Inspect this repository architecture and identify design risks.',
    });
    const plans = planner.completePlans(analysis, [{
      archetype: 'researcher',
      name: 'Researcher-1',
      task: 'Inspect the project with filesystem evidence.',
    }], 3);

    expect(plans.map(plan => plan.archetype)).toEqual(['researcher', 'critic', 'summarizer']);
    for (const plan of plans) {
      expect(plan.existenceReason).toContain('Created to cover');
      expect(plan.cognitiveGapIds?.length).toBeGreaterThan(0);
      expect(plan.tomProfile?.beliefScope.length).toBeGreaterThan(0);
      expect(plan.tomProfile?.goalModel.length).toBeGreaterThan(0);
      expect(plan.tomProfile?.uncertainty.length).toBeGreaterThan(0);
      expect(plan.tomProfile?.perspective).toBeTruthy();
    }
    const synthesizer = plans.find(plan => plan.archetype === 'summarizer')!;
    expect(synthesizer.tomProfile?.level).toBe(2);
    expect(synthesizer.tomProfile?.modelsAgents).toEqual(expect.arrayContaining(['root', 'Researcher-1']));

    const coverage = planner.evaluateCoverage(analysis, plans);
    expect(coverage.coverageScore).toBe(1);
    expect(coverage.uncoveredGapIds).toEqual([]);
    expect(coverage.higherOrderFit).toBe(1);
  });

  it('scores cognitive coverage above a cheaper but incomplete candidate', () => {
    const planner = new ToMDelegationPlanner();
    const analysis = planner.analyzeTask({
      parentId: 'root',
      parentProfile: rootProfile,
      task: 'Inspect this repository architecture and identify design risks.',
    });
    const fullPlans = planner.completePlans(analysis, [{
      archetype: 'researcher',
      task: 'Inspect project evidence.',
    }], 3);
    const candidates: DelegationCandidate[] = [
      {
        id: 'full', source: 'generated', parentId: 'root', agents: fullPlans,
        expectedUtility: 0, expectedCostTokens: 5000, score: 0, scoreBreakdown: {}, rationale: 'full coverage',
      },
      {
        id: 'single', source: 'generated', parentId: 'root', agents: [fullPlans[0]],
        expectedUtility: 0, expectedCostTokens: 2000, score: 0, scoreBreakdown: {}, rationale: 'partial coverage',
      },
    ];
    const input: DelegationCandidateInput = {
      parentId: 'root',
      task: analysis.task,
      decision: { action: 'spawn_subagents', reason: 'test', agents: fullPlans },
      allowedChildren: 3,
      remainingTotalAgentsForTurn: 3,
      budgetMode: 'unlimited',
      cacheUsed: false,
      parentToMProfile: rootProfile,
      tomAnalysis: analysis,
    };

    const scores = new ToMDelegationScorer().score(candidates, input);
    expect(scores.get('full')).toBeGreaterThan(scores.get('single')!);
  });

  it('builds a team-level higher-order profile over direct member beliefs', () => {
    const planner = new ToMDelegationPlanner();
    const analysis = planner.analyzeTask({
      parentId: 'root',
      parentProfile: rootProfile,
      task: 'Inspect this repository architecture and identify design risks.',
    });
    const members = planner.completePlans(analysis, [{ archetype: 'researcher', task: 'Inspect evidence.' }], 3);
    const profile = planner.createTeamProfile({
      teamId: 'team_001',
      parentId: 'root',
      task: analysis.task,
      members,
    });

    expect(profile.level).toBe(2);
    expect(profile.modelsAgents).toContain('root');
    expect(profile.observesAgents.length).toBe(3);
    expect(profile.cognitiveGaps).toEqual(expect.arrayContaining(analysis.gaps.map(gap => gap.id)));
    expect(profile.recursiveModels?.length).toBe(3);
  });
});
