import type { SubAgentArchetype } from '../runtime/Runtime.js';
import type {
  DelegationCandidate,
  DelegationCandidateInput,
  DelegationCandidateSelection,
  DelegationCandidateSource,
} from './types.js';

const ARCHETYPE_UTILITY: Record<SubAgentArchetype, number> = {
  researcher: 0.78,
  critic: 0.72,
  planner: 0.62,
  coder: 0.7,
  summarizer: 0.48,
  tester: 0.66,
  custom: 0.58,
};

const ARCHETYPE_COST: Record<SubAgentArchetype, number> = {
  researcher: 2200,
  critic: 1600,
  planner: 1400,
  coder: 2600,
  summarizer: 1000,
  tester: 1800,
  custom: 1800,
};

export class DefaultDelegationCandidatePlanner {
  select(input: DelegationCandidateInput): DelegationCandidateSelection {
    if (input.decision.action !== 'spawn_subagents') {
      return {
        candidates: [],
        decision: input.decision,
      };
    }

    const limit = Math.max(0, Math.min(
      input.allowedChildren,
      input.remainingTotalAgentsForTurn,
      input.decision.agents.length
    ));

    if (limit <= 0) {
      return {
        candidates: [],
        decision: {
          action: 'solve_directly',
          reason: `${input.decision.reason} Delegation skipped because no agent slots remain for this turn or parent.`,
        },
        rejectedReason: 'no_agent_slots_remaining',
      };
    }

    const candidates = this.generateCandidates(input, limit);
    const selected = candidates[0];
    if (!selected || selected.agents.length === 0) {
      return {
        candidates,
        decision: {
          action: 'solve_directly',
          reason: `${input.decision.reason} Delegation skipped because no candidate passed policy scoring.`,
        },
        rejectedReason: 'no_candidate_selected',
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

  private generateCandidates(input: DelegationCandidateInput, limit: number): DelegationCandidate[] {
    const boundedAgents = input.decision.action === 'spawn_subagents'
      ? input.decision.agents.slice(0, limit)
      : [];
    const candidates: DelegationCandidate[] = [];

    if (boundedAgents.length > 0) {
      candidates.push(this.scoreCandidate({
        id: 'candidate_full_plan',
        parentId: input.parentId,
        agents: boundedAgents,
        source: this.sourceFor(boundedAgents, input.cacheUsed),
        rationale: 'uses the highest-ranked bounded delegation plan',
        task: input.task,
        budgetMode: input.budgetMode,
        remainingBudgetTokens: input.remainingBudgetTokens,
      }));
    }

    const topSingle = boundedAgents[0];
    if (topSingle && boundedAgents.length > 1) {
      candidates.push(this.scoreCandidate({
        id: `candidate_single_${topSingle.archetype}`,
        parentId: input.parentId,
        agents: [topSingle],
        source: input.cacheUsed ? 'cache_hit' : topSingle.archetype === 'custom' ? 'custom_generated' : 'generated',
        rationale: 'uses the most valuable single specialist to reduce cost and branching',
        task: input.task,
        budgetMode: input.budgetMode,
        remainingBudgetTokens: input.remainingBudgetTokens,
      }));
    }

    return candidates.sort((a, b) => b.score - a.score);
  }

  private scoreCandidate(input: {
    id: string;
    parentId: string;
    agents: DelegationCandidate['agents'];
    source: DelegationCandidateSource;
    rationale: string;
    task: string;
    budgetMode: 'unlimited' | 'limited';
    remainingBudgetTokens?: number;
  }): DelegationCandidate {
    const expectedUtility = input.agents.reduce((total, agent) => {
      const base = ARCHETYPE_UTILITY[agent.archetype] ?? 0.5;
      return total + base + this.taskMatchBonus(input.task, agent.archetype);
    }, 0);
    const expectedCostTokens = input.agents.reduce((total, agent) => total + (agent.budgetTokens ?? ARCHETYPE_COST[agent.archetype] ?? 1800), 0);
    const budgetPenalty = input.budgetMode === 'limited' && input.remainingBudgetTokens !== undefined
      ? Math.max(0, expectedCostTokens - input.remainingBudgetTokens) / 2000
      : 0;
    const branchingPenalty = Math.max(0, input.agents.length - 1) * 0.1;
    const cacheBonus = input.source === 'cache_hit' || input.source === 'mixed' ? 0.15 : 0;
    const score = expectedUtility + cacheBonus - expectedCostTokens / 10000 - budgetPenalty - branchingPenalty;

    return {
      id: input.id,
      source: input.source,
      parentId: input.parentId,
      agents: input.agents,
      expectedUtility: Number(expectedUtility.toFixed(3)),
      expectedCostTokens,
      score: Number(score.toFixed(3)),
      rationale: input.rationale,
    };
  }

  private taskMatchBonus(task: string, archetype: SubAgentArchetype): number {
    const lower = task.toLowerCase();
    if (archetype === 'researcher' && /\b(inspect|read|list|evidence|structure)\b/.test(lower)) return 0.16;
    if (archetype === 'critic' && /\b(risk|review|critique|failure|gap)\b/.test(lower)) return 0.16;
    if (archetype === 'tester' && /\b(test|verify|coverage|regression)\b/.test(lower)) return 0.14;
    if (archetype === 'planner' && /\b(plan|steps|roadmap|design)\b/.test(lower)) return 0.12;
    if (archetype === 'coder' && /\b(code|implement|fix|patch)\b/.test(lower)) return 0.14;
    if (archetype === 'custom' && /\b(prompt|slot|context|special)\b/.test(lower)) return 0.12;
    return 0;
  }

  private sourceFor(agents: DelegationCandidate['agents'], cacheUsed: boolean): DelegationCandidateSource {
    if (cacheUsed && agents.some(agent => agent.archetype === 'custom')) return 'mixed';
    if (cacheUsed) return 'cache_hit';
    if (agents.some(agent => agent.archetype === 'custom')) return 'custom_generated';
    return 'generated';
  }
}
