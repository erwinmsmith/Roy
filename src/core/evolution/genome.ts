import type {
  AgentGenome,
  EvolutionCandidate,
  EvolutionProposalInput,
  EvolutionSeedAgent,
  GenomeMutationContext,
  GenomeMutationOperator,
  GenomeToolPolicy,
  TeamGenome,
} from './types.js';

const DEFAULT_COST: Record<AgentGenome['archetype'], number> = {
  researcher: 2200,
  critic: 1600,
  planner: 1400,
  coder: 2600,
  summarizer: 1000,
  tester: 1800,
  custom: 1800,
};

const TOOL_PERMISSIONS: Record<string, GenomeToolPolicy['permission']> = {
  'fs.list': 'read_only',
  'fs.read': 'read_only',
  'fs.write': 'write',
  'shell.exec': 'execute',
};

export class TeamFirstGenomePlanner {
  propose(input: EvolutionProposalInput): EvolutionCandidate[] {
    if (input.options.ablations.withoutSubagents || input.options.profile === 'solo') return [];
    const candidates: EvolutionCandidate[] = [];
    const boundedAgents = this.boundAgents(input);
    if (boundedAgents.length === 0) return [];

    if (!input.options.ablations.withoutPatternMemory) {
      for (const pattern of input.patterns.slice(0, input.options.populationSize)) {
        candidates.push(this.candidate(
          input,
          'cache_hit',
          this.retargetGenome(pattern.genome, input.task, `${input.runId}_cache_${pattern.id}`),
          `Reusable genome loaded from ${pattern.id}.`,
          [pattern.id],
          0
        ));
      }
    }

    candidates.push(this.candidate(
      input,
      'default',
      this.teamGenome(input, boundedAgents, `${input.runId}_default`),
      'Team-first candidate compiled from the parent delegation plan.',
      [],
      0
    ));

    if (input.options.profile !== 'evo_team') return candidates.slice(0, 1);

    if (boundedAgents.length > 1) {
      candidates.push(this.candidate(
        input,
        'default',
        this.teamGenome(input, [boundedAgents[0]], `${input.runId}_solo_specialist`),
        'One-member team candidate that degenerates to a runtime agent.',
        [],
        0
      ));
    }

    return this.unique(candidates).slice(0, input.options.populationSize);
  }

  mutate(
    selected: EvolutionCandidate[],
    input: EvolutionProposalInput,
    operators: GenomeMutationOperator[],
    generation: number
  ): EvolutionCandidate[] {
    if (input.options.ablations.withoutEvoMutation || generation <= 0) return [];
    const mutations: EvolutionCandidate[] = [];
    for (const parent of selected) {
      const context: GenomeMutationContext = {
        runId: input.runId,
        generation,
        task: input.task,
        availableTokens: input.availableTokens,
      };
      for (const operator of operators) {
        if (!operator.supports(parent.genome, context)) continue;
        const genome = operator.mutate(parent.genome, context);
        mutations.push(this.candidate(
          input,
          parent.source === 'cache_hit' ? 'mutated_from_cache' : 'generated',
          genome,
          `${operator.name} applied to ${parent.id}.`,
          parent.lineage.parentPatternIds,
          generation,
          parent.id,
          [...parent.lineage.operators, operator.name]
        ));
      }
    }
    return this.unique(mutations).slice(0, input.options.populationSize);
  }

  private boundAgents(input: EvolutionProposalInput): EvolutionSeedAgent[] {
    const requested = input.agents.slice(0, input.options.populationSize);
    if (input.availableTokens === undefined || input.options.ablations.withoutBudgetMarket) return requested;
    let remaining = input.availableTokens;
    const accepted: EvolutionSeedAgent[] = [];
    for (const agent of requested) {
      const cost = Math.max(256, agent.budgetTokens ?? DEFAULT_COST[agent.archetype]);
      if (remaining < Math.min(cost, 1000)) break;
      accepted.push({ ...agent, budgetTokens: Math.min(cost, remaining) });
      remaining -= Math.min(cost, remaining);
    }
    return accepted;
  }

  private teamGenome(input: EvolutionProposalInput, agents: EvolutionSeedAgent[], id: string): TeamGenome {
    const members = agents.map((agent, index) => this.createAgentGenome(input, agent, `${id}_member_${index + 1}`));
    return {
      id,
      name: this.teamName(members),
      taskSignature: input.task,
      purpose: input.task,
      tomLevel: input.options.ablations.withoutToMProfile ? 0 : members.length > 1 ? 2 : members[0].tomProfile.level,
      members,
      coordinationPolicy: members.length <= 1 ? 'sequential' : this.coordinationPolicy(members),
      synthesisPolicy: members.length <= 1
        ? 'Return the specialist result through the parent synthesis boundary.'
        : 'Aggregate direct member evidence, preserve disagreements, and synthesize one grounded result.',
      budgetPolicy: {
        requestedTokens: members.reduce((sum, member) => sum + member.budgetPolicy.requestedTokens, 0),
        maxTokens: input.availableTokens,
      },
      evaluationCriteria: ['task_success', 'answer_quality', 'cost_efficiency', 'tool_correctness', 'consistency'],
    };
  }

  createAgentGenome(input: EvolutionProposalInput, seed: EvolutionSeedAgent, id: string): AgentGenome {
    const level = input.options.ablations.withoutToMProfile
      ? 0
      : Math.max(0, Math.min(3, Math.floor(seed.tomLevel ?? this.defaultTomLevel(seed.archetype)))) as 0 | 1 | 2 | 3;
    return {
      id,
      archetype: seed.archetype,
      name: seed.name,
      role: seed.role ?? seed.archetype,
      task: seed.task,
      rolePrompt: this.rolePrompt(seed),
      taskDecompositionStyle: this.decompositionStyle(seed.archetype),
      toolPolicy: (seed.tools ?? []).map(name => ({
        name,
        permission: TOOL_PERMISSIONS[name] ?? 'read_only',
        required: seed.groundingRequired && (name === 'fs.list' || name === 'fs.read'),
      })),
      skills: [...new Set(seed.skills ?? [])],
      memoryPolicy: 'local',
      tomProfile: {
        level,
        beliefScope: level === 0 ? [] : ['task evidence', 'runtime constraints'],
        goalModel: [seed.task],
        uncertainty: level === 0 ? [] : ['whether available evidence is sufficient'],
        perspective: seed.perspective ?? seed.role ?? seed.archetype,
        observesAgents: [],
        modelsAgents: level >= 1 ? ['parent'] : [],
      },
      budgetPolicy: {
        requestedTokens: Math.max(256, seed.budgetTokens ?? DEFAULT_COST[seed.archetype]),
        maxTokens: seed.budgetTokens,
      },
      evaluationCriteria: ['task_success', 'grounding', 'cost_efficiency'],
      outputContract: {
        format: 'structured_report',
        groundingRequired: seed.groundingRequired ?? (seed.archetype === 'researcher' || seed.archetype === 'tester'),
        requiredFields: ['findings', 'evidence', 'limitations'],
      },
    };
  }

  private candidate(
    input: EvolutionProposalInput,
    source: EvolutionCandidate['source'],
    genome: TeamGenome,
    rationale: string,
    parentPatternIds: string[],
    generation: number,
    parentCandidateId?: string,
    operators: string[] = []
  ): EvolutionCandidate {
    const suffix = `${source}_${genome.id}`.replace(/[^a-zA-Z0-9_-]/g, '_');
    return {
      id: `evo_candidate_${suffix}`,
      source,
      genome,
      rationale,
      expectedCostTokens: genome.budgetPolicy.requestedTokens,
      expectedUtility: 0,
      lineage: { parentCandidateId, parentPatternIds, operators, generation },
    };
  }

  private retargetGenome(genome: TeamGenome, task: string, id: string): TeamGenome {
    const members = genome.members.map((member, index) => ({
      ...member,
      id: `${id}_member_${index + 1}`,
      task,
      tomProfile: { ...member.tomProfile, goalModel: [task] },
    }));
    return { ...genome, id, taskSignature: task, purpose: task, members };
  }

  private unique(candidates: EvolutionCandidate[]): EvolutionCandidate[] {
    const seen = new Set<string>();
    return candidates.filter(candidate => {
      const signature = candidate.genome.members
        .map(member => `${member.archetype}:${member.role}:${member.toolPolicy.map(tool => tool.name).sort().join(',')}`)
        .join('|');
      if (seen.has(signature)) return false;
      seen.add(signature);
      return true;
    });
  }

  private teamName(members: AgentGenome[]): string {
    if (members.length === 1) return `${this.capitalize(members[0].archetype)}Cell`;
    const labels = members.slice(0, 2).map(member => this.capitalize(member.archetype)).join('');
    return `${labels}Team`;
  }

  private coordinationPolicy(members: AgentGenome[]): TeamGenome['coordinationPolicy'] {
    if (members.some(member => member.archetype === 'critic')) return 'critic_refine';
    if (members.some(member => member.archetype === 'summarizer')) return 'leader_worker';
    return 'parallel';
  }

  private defaultTomLevel(archetype: AgentGenome['archetype']): number {
    if (archetype === 'critic') return 2;
    if (archetype === 'planner' || archetype === 'summarizer') return 1;
    return 0;
  }

  private rolePrompt(seed: EvolutionSeedAgent): string {
    return `Act as ${seed.name ?? seed.role ?? seed.archetype}. Fulfill only the assigned responsibility and return evidence, findings, and limitations.`;
  }

  private decompositionStyle(archetype: AgentGenome['archetype']): string {
    if (archetype === 'planner') return 'decompose dependencies before execution';
    if (archetype === 'critic') return 'identify claims, assumptions, and failure modes';
    if (archetype === 'researcher') return 'collect evidence before inference';
    if (archetype === 'tester') return 'derive checks and verify observable behavior';
    return 'execute the bounded task and report structured results';
  }

  private capitalize(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }
}

abstract class BaseMutationOperator implements GenomeMutationOperator {
  abstract readonly name: string;
  supports(_genome: TeamGenome, _context: GenomeMutationContext): boolean {
    return true;
  }
  abstract mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome;

  protected clone(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    return structuredClone({
      ...genome,
      id: `${context.runId}_g${context.generation}_${this.name}_${genome.id}`,
      members: genome.members.map(member => ({ ...member })),
    });
  }
}

export class MutateRolePromptOperator extends BaseMutationOperator {
  readonly name = 'mutate_role_prompt';
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    next.members = next.members.map(member => ({
      ...member,
      rolePrompt: `${member.rolePrompt} Explicitly distinguish observed facts, inferences, and unresolved uncertainty.`,
    }));
    return next;
  }
}

export class MutateToolPolicyOperator extends BaseMutationOperator {
  readonly name = 'mutate_tool_policy';
  supports(genome: TeamGenome): boolean {
    return genome.members.some(member => ['researcher', 'critic', 'tester', 'coder'].includes(member.archetype));
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    next.members = next.members.map(member => {
      if (!['researcher', 'critic', 'tester', 'coder'].includes(member.archetype)) return member;
      const names = new Set(member.toolPolicy.map(tool => tool.name));
      const toolPolicy = [...member.toolPolicy];
      if (!names.has('fs.read')) toolPolicy.push({ name: 'fs.read', permission: 'read_only', required: false });
      return { ...member, toolPolicy };
    });
    return next;
  }
}

export class MutateToMLevelOperator extends BaseMutationOperator {
  readonly name = 'mutate_tom_level';
  supports(genome: TeamGenome): boolean {
    return genome.members.some(member => member.tomProfile.level < 2);
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    next.members = next.members.map(member => ({
      ...member,
      tomProfile: {
        ...member.tomProfile,
        level: Math.min(3, member.tomProfile.level + 1) as 0 | 1 | 2 | 3,
        modelsAgents: [...new Set([...member.tomProfile.modelsAgents ?? [], 'parent', 'peer_outputs'])],
      },
    }));
    next.tomLevel = Math.min(3, next.tomLevel + 1) as 0 | 1 | 2 | 3;
    return next;
  }
}

export class MutateBudgetOperator extends BaseMutationOperator {
  readonly name = 'mutate_budget';
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    const cap = context.availableTokens ?? Number.MAX_SAFE_INTEGER;
    next.members = next.members.map(member => ({
      ...member,
      budgetPolicy: {
        ...member.budgetPolicy,
        requestedTokens: Math.max(256, Math.floor(member.budgetPolicy.requestedTokens * 0.85)),
      },
    }));
    next.budgetPolicy.requestedTokens = Math.min(
      cap,
      next.members.reduce((sum, member) => sum + member.budgetPolicy.requestedTokens, 0)
    );
    return next;
  }
}

export class AddCriticOperator extends BaseMutationOperator {
  readonly name = 'add_critic';
  supports(genome: TeamGenome, context: GenomeMutationContext): boolean {
    return genome.members.length < 5
      && !genome.members.some(member => member.archetype === 'critic')
      && (context.availableTokens === undefined || context.availableTokens >= genome.budgetPolicy.requestedTokens + DEFAULT_COST.critic);
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    const critic: EvolutionSeedAgent = {
      archetype: 'critic',
      role: 'evidence and failure-mode critic',
      task: `Critique candidate findings for: ${context.task}`,
      tools: ['fs.read'],
      skills: ['use_tool_when_needed', 'delegate_to_subagent'],
      budgetTokens: DEFAULT_COST.critic,
      tomLevel: 2,
    };
    const planner = new TeamFirstGenomePlanner();
    const input: EvolutionProposalInput = {
      runId: context.runId,
      task: context.task,
      parentId: 'runtime',
      agents: [critic],
      patterns: [],
      availableTokens: context.availableTokens,
      options: {
        profile: 'evo_team', populationSize: 1, generations: 1, topK: 1,
        maxExecutedCandidates: 1, integrationMinimumScore: 0, patternSimilarityThreshold: 0,
        useLlmJudge: false,
        ablations: {
          withoutSubagents: false, withoutToMProfile: false, withoutBudgetMarket: false,
          withoutEvoMutation: false, withoutPatternMemory: false,
        },
      },
    };
    next.members.push(planner.createAgentGenome(input, critic, `${next.id}_critic`));
    next.coordinationPolicy = 'critic_refine';
    next.budgetPolicy.requestedTokens += DEFAULT_COST.critic;
    return next;
  }
}

export class AddSynthesizerOperator extends BaseMutationOperator {
  readonly name = 'add_synthesizer';
  supports(genome: TeamGenome, context: GenomeMutationContext): boolean {
    return genome.members.length > 1
      && genome.members.length < 5
      && !genome.members.some(member => member.archetype === 'summarizer')
      && (context.availableTokens === undefined || context.availableTokens >= genome.budgetPolicy.requestedTokens + DEFAULT_COST.summarizer);
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    next.members.push({
      id: `${next.id}_synthesizer`,
      archetype: 'summarizer',
      role: 'cross-agent synthesizer',
      task: `Reconcile member evidence for: ${context.task}`,
      rolePrompt: 'Reconcile direct member outputs, preserve disagreement, and produce one traceable conclusion.',
      taskDecompositionStyle: 'compare claims, evidence, conflicts, and confidence',
      toolPolicy: [],
      skills: [],
      memoryPolicy: 'shared',
      tomProfile: {
        level: 2,
        beliefScope: ['member outputs'],
        goalModel: ['resolve disagreement and preserve evidence'],
        uncertainty: ['conflicting member claims'],
        perspective: 'cross-agent synthesizer',
        observesAgents: next.members.map(member => member.id),
        modelsAgents: next.members.map(member => member.id),
      },
      budgetPolicy: { requestedTokens: DEFAULT_COST.summarizer },
      evaluationCriteria: ['consistency', 'completeness'],
      outputContract: {
        format: 'structured_report', groundingRequired: false,
        requiredFields: ['consensus', 'disagreements', 'limitations'],
      },
    });
    next.coordinationPolicy = 'leader_worker';
    next.budgetPolicy.requestedTokens += DEFAULT_COST.summarizer;
    return next;
  }
}

export class SplitAgentOperator extends BaseMutationOperator {
  readonly name = 'split_agent';
  supports(genome: TeamGenome, context: GenomeMutationContext): boolean {
    return genome.members.length === 1
      && (context.availableTokens === undefined || context.availableTokens >= genome.budgetPolicy.requestedTokens * 1.5);
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    const source = next.members[0];
    const firstBudget = Math.max(256, Math.floor(source.budgetPolicy.requestedTokens * 0.6));
    const secondBudget = Math.max(256, source.budgetPolicy.requestedTokens - firstBudget);
    next.members = [
      { ...source, id: `${next.id}_evidence`, role: `${source.role} evidence collector`, budgetPolicy: { requestedTokens: firstBudget } },
      {
        ...source,
        id: `${next.id}_verifier`,
        archetype: source.archetype === 'researcher' ? 'critic' : source.archetype,
        role: `${source.role} verifier`,
        task: `Independently verify the primary result for: ${context.task}`,
        budgetPolicy: { requestedTokens: secondBudget },
        tomProfile: { ...source.tomProfile, level: Math.max(1, source.tomProfile.level) as 0 | 1 | 2 | 3 },
      },
    ];
    next.coordinationPolicy = 'critic_refine';
    return next;
  }
}

export class MergeAgentsOperator extends BaseMutationOperator {
  readonly name = 'merge_agents';
  supports(genome: TeamGenome): boolean {
    return genome.members.length > 1;
  }
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome {
    const next = this.clone(genome, context);
    const first = next.members[0];
    const tools = new Map<string, GenomeToolPolicy>();
    for (const member of next.members) for (const tool of member.toolPolicy) tools.set(tool.name, tool);
    const requestedTokens = Math.max(256, Math.floor(next.budgetPolicy.requestedTokens * 0.72));
    next.members = [{
      ...first,
      id: `${next.id}_merged`,
      role: `merged ${next.members.map(member => member.role).join(' and ')}`,
      task: context.task,
      rolePrompt: `Combine these responsibilities without duplicating work: ${next.members.map(member => member.role).join('; ')}.`,
      toolPolicy: [...tools.values()],
      skills: [...new Set(next.members.flatMap(member => member.skills))],
      budgetPolicy: { requestedTokens },
    }];
    next.coordinationPolicy = 'sequential';
    next.budgetPolicy.requestedTokens = requestedTokens;
    return next;
  }
}

export function defaultMutationOperators(): GenomeMutationOperator[] {
  return [
    new AddCriticOperator(),
    new AddSynthesizerOperator(),
    new SplitAgentOperator(),
    new MergeAgentsOperator(),
    new MutateRolePromptOperator(),
    new MutateToolPolicyOperator(),
    new MutateToMLevelOperator(),
    new MutateBudgetOperator(),
  ];
}

export function validateTeamGenome(genome: TeamGenome): void {
  if (!genome.id.trim()) throw new Error('Team genome id is required');
  if (!genome.name.trim()) throw new Error('Team genome name is required');
  if (genome.members.length === 0) throw new Error('Team genome must contain at least one member');
  if (genome.members.length > 5) throw new Error('Team genome may contain at most five members');
  const ids = new Set<string>();
  for (const member of genome.members) {
    if (!member.id.trim() || ids.has(member.id)) throw new Error(`Invalid or duplicate member genome id: ${member.id}`);
    ids.add(member.id);
    if (!member.task.trim()) throw new Error(`Genome member ${member.id} requires a task`);
    if (member.budgetPolicy.requestedTokens <= 0) throw new Error(`Genome member ${member.id} requires a positive token request`);
  }
}
