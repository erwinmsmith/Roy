import { normalizeToMProfile } from './profile.js';
import type {
  CognitiveGap,
  CognitiveGapKind,
  ToMCoverageResult,
  ToMPlanAgent,
  ToMProfile,
  ToMTaskAnalysis,
} from './types.js';

const ARCHETYPE_BY_GAP: Record<CognitiveGapKind, ToMPlanAgent['archetype']> = {
  knowledge: 'researcher',
  evidence: 'researcher',
  perspective: 'critic',
  risk: 'critic',
  planning: 'planner',
  implementation: 'coder',
  verification: 'tester',
  synthesis: 'summarizer',
  capability: 'custom',
};

const ARCHETYPE_PERSPECTIVE: Record<ToMPlanAgent['archetype'], string> = {
  researcher: 'grounded evidence collector',
  critic: 'skeptical failure-mode reviewer',
  planner: 'constraint-aware implementation planner',
  coder: 'scoped implementation specialist',
  summarizer: 'belief reconciliation and synthesis specialist',
  tester: 'independent verification specialist',
  custom: 'task-specific specialist',
};

export class ToMDelegationPlanner {
  private analysisSequence = 0;

  analyzeTask(input: { task: string; parentId: string; parentProfile?: ToMProfile }): ToMTaskAnalysis {
    const lower = input.task.toLowerCase();
    const gaps: CognitiveGap[] = [];
    const add = (kind: CognitiveGapKind, values: Omit<CognitiveGap, 'id' | 'kind'>): void => {
      if (gaps.some(gap => gap.kind === kind)) return;
      gaps.push({ id: `tom_gap_${kind}_${String(gaps.length + 1).padStart(2, '0')}`, kind, ...values });
    };

    if (/\b(inspect|analy[sz]e|read|list|structure|architecture|repo|repository|codebase|evidence)\b|检查|分析|读取|结构|架构|仓库/.test(lower)) {
      add('evidence', {
        description: 'The parent lacks grounded observations about the target system.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.researcher,
        beliefScope: ['observable project facts', 'runtime and file structure'],
        goal: 'Collect concrete evidence before conclusions are formed.',
        uncertainty: ['Which facts are directly supported by tools or source files?'],
        requiredCapabilities: ['inspect_project', 'fs.list', 'fs.read'],
        modelsTargets: ['environment'],
        priority: 1,
      });
    }
    if (/\b(risk|risks|review|critique|failure|coupling|weakness|threat|audit)\b|风险|审查|批判|故障|耦合/.test(lower)) {
      add('risk', {
        description: 'The parent needs an adversarial perspective on failure modes and hidden assumptions.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.critic,
        beliefScope: ['failure modes', 'hidden coupling', 'unsupported assumptions'],
        goal: 'Stress-test evidence and proposed conclusions.',
        uncertainty: ['Which important failure modes would the primary investigator miss?'],
        requiredCapabilities: ['critique_report', 'check_grounding'],
        modelsTargets: [input.parentId, 'peer:researcher'],
        priority: 0.95,
      });
    }
    if (/\b(test|tests|verify|verification|validate|coverage|regression)\b|测试|验证|覆盖率|回归/.test(lower)) {
      add('verification', {
        description: 'The result needs independent behavioral verification.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.tester,
        beliefScope: ['observable behavior', 'test evidence', 'regression risk'],
        goal: 'Verify claims against executable checks.',
        uncertainty: ['Do the claimed behaviors hold under tests and failure cases?'],
        requiredCapabilities: ['run_test', 'inspect_failure'],
        modelsTargets: ['environment', input.parentId],
        priority: 0.9,
      });
    }
    if (/\b(plan|roadmap|steps|refactor|migration|strategy)\b|\b(design)\b(?=.*\b(plan|proposal|solution|implementation)\b)|计划|路线|重构|迁移|设计方案/.test(lower)) {
      add('planning', {
        description: 'The parent needs a coherent action sequence that respects constraints.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.planner,
        beliefScope: ['dependencies', 'constraints', 'implementation order'],
        goal: 'Turn findings into an executable plan.',
        uncertainty: ['Which dependency or sequencing constraint can invalidate the plan?'],
        requiredCapabilities: ['decompose_task'],
        modelsTargets: [input.parentId, 'user'],
        priority: 0.8,
      });
    }
    if (/\b(implement|code|fix|patch|modify|build|develop)\b|实现|编码|修复|修改|开发/.test(lower)) {
      add('implementation', {
        description: 'The task requires implementation-specific capability and source-level reasoning.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.coder,
        beliefScope: ['implementation boundaries', 'source behavior', 'change impact'],
        goal: 'Produce a scoped implementation grounded in the current code.',
        uncertainty: ['Which source changes are necessary and sufficient?'],
        requiredCapabilities: ['edit_code', 'fs.read'],
        modelsTargets: ['environment'],
        priority: 0.85,
      });
    }
    if (/\b(prompt|slot|specialized|domain-specific|custom)\b|提示词|插槽|专用|自定义/.test(lower)) {
      add('capability', {
        description: 'The task may require a capability not fully represented by a built-in archetype.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.custom,
        beliefScope: ['task-specific contract', 'specialized correctness criteria'],
        goal: 'Cover the specialized capability gap explicitly.',
        uncertainty: ['Can a built-in archetype satisfy the specialized contract?'],
        requiredCapabilities: ['task_specific_analysis'],
        modelsTargets: [input.parentId],
        priority: 0.65,
      });
    }

    for (const uncertainty of input.parentProfile?.uncertainty ?? []) {
      if (gaps.some(gap => gap.uncertainty.includes(uncertainty))) continue;
      add('knowledge', {
        description: `The parent explicitly reports unresolved uncertainty: ${uncertainty}`,
        requiredPerspective: ARCHETYPE_PERSPECTIVE.researcher,
        beliefScope: input.parentProfile?.beliefScope ?? [],
        goal: 'Reduce a parent-declared uncertainty with bounded evidence.',
        uncertainty: [uncertainty],
        requiredCapabilities: ['inspect_project'],
        modelsTargets: ['environment'],
        priority: 0.75,
      });
    }

    if (gaps.length === 0) {
      add('knowledge', {
        description: 'The parent needs a bounded specialist interpretation of the task.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.researcher,
        beliefScope: ['task facts and constraints'],
        goal: 'Resolve the most important factual uncertainty.',
        uncertainty: ['Which facts are required for a reliable answer?'],
        requiredCapabilities: ['inspect_project'],
        modelsTargets: ['environment'],
        priority: 0.55,
      });
    }

    const substantive = gaps.filter(gap => gap.kind !== 'synthesis');
    const requiresHigherOrderToM = substantive.length >= 2
      && new Set(substantive.map(gap => gap.requiredPerspective)).size >= 2;
    if (requiresHigherOrderToM) {
      add('synthesis', {
        description: 'Multiple specialist belief sets must be compared and reconciled.',
        requiredPerspective: ARCHETYPE_PERSPECTIVE.summarizer,
        beliefScope: substantive.flatMap(gap => gap.beliefScope),
        goal: 'Resolve disagreement and expose remaining uncertainty across specialist outputs.',
        uncertainty: ['Where do specialist conclusions conflict or rely on different assumptions?'],
        requiredCapabilities: ['synthesize_results'],
        modelsTargets: substantive.map(gap => `gap:${gap.id}`),
        priority: 0.7,
      });
    }

    return {
      id: `tom_analysis_${Date.now()}_${String(++this.analysisSequence).padStart(4, '0')}`,
      parentId: input.parentId,
      task: input.task,
      parentBeliefs: [...(input.parentProfile?.beliefScope ?? [])],
      parentGoals: [...(input.parentProfile?.goalModel ?? [])],
      parentUncertainties: [...(input.parentProfile?.uncertainty ?? [])],
      gaps,
      requiresHigherOrderToM,
      rationale: requiresHigherOrderToM
        ? 'The task contains complementary cognitive gaps that require specialist beliefs and higher-order reconciliation.'
        : 'The task contains a bounded cognitive gap that can be assigned to one specialist perspective.',
      createdAt: Date.now(),
    };
  }

  completePlans(
    analysis: ToMTaskAnalysis,
    inputPlans: ToMPlanAgent[],
    maxAgents: number
  ): ToMPlanAgent[] {
    const boundedMax = Math.max(0, Math.min(3, maxAgents));
    if (boundedMax === 0) return [];
    const plans = inputPlans.slice(0, boundedMax).map(plan => ({ ...plan }));
    const assigned = new Set<string>();

    for (const plan of plans) {
      const matched = this.matchGaps(plan.archetype, analysis.gaps, assigned);
      for (const gap of matched) assigned.add(gap.id);
      Object.assign(plan, this.enrichPlan(plan, matched, analysis, plans));
    }

    for (const gap of [...analysis.gaps].sort((a, b) => b.priority - a.priority)) {
      if (plans.length >= boundedMax || assigned.has(gap.id)) continue;
      const archetype = ARCHETYPE_BY_GAP[gap.kind];
      if (plans.some(plan => plan.archetype === archetype)) continue;
      const plan: ToMPlanAgent = {
        archetype,
        name: `${capitalize(archetype)}-${plans.length + 1}`,
        task: this.taskForGap(gap, analysis.task),
      };
      const matched = this.matchGaps(archetype, analysis.gaps, assigned);
      for (const item of matched) assigned.add(item.id);
      Object.assign(plan, this.enrichPlan(plan, matched, analysis, [...plans, plan]));
      plans.push(plan);
    }

    return plans.map(plan => {
      const gaps = analysis.gaps.filter(gap => plan.cognitiveGapIds?.includes(gap.id));
      return this.enrichPlan(plan, gaps, analysis, plans);
    });
  }

  evaluateCoverage(analysis: ToMTaskAnalysis, plans: ToMPlanAgent[]): ToMCoverageResult {
    const covered = new Set(plans.flatMap(plan => plan.cognitiveGapIds ?? []));
    const totalWeight = analysis.gaps.reduce((sum, gap) => sum + gap.priority, 0) || 1;
    const coveredWeight = analysis.gaps
      .filter(gap => covered.has(gap.id))
      .reduce((sum, gap) => sum + gap.priority, 0);
    const perspectives = new Set(plans
      .map(plan => plan.tomProfile?.perspective)
      .filter((value): value is string => Boolean(value)));
    const hasHigherOrderAgent = plans.some(plan => (plan.tomProfile?.level ?? plan.tomLevel ?? 0) >= 2
      && (plan.tomProfile?.modelsAgents.length ?? 0) >= 2);
    return {
      coveredGapIds: analysis.gaps.filter(gap => covered.has(gap.id)).map(gap => gap.id),
      uncoveredGapIds: analysis.gaps.filter(gap => !covered.has(gap.id)).map(gap => gap.id),
      coverageScore: Number((coveredWeight / totalWeight).toFixed(4)),
      perspectiveDiversity: Number((Math.min(1, perspectives.size / Math.max(1, Math.min(3, analysis.gaps.length)))).toFixed(4)),
      higherOrderFit: analysis.requiresHigherOrderToM ? (hasHigherOrderAgent ? 1 : 0) : 1,
      unjustifiedAgentCount: plans.filter(plan => !plan.existenceReason?.trim() || (plan.cognitiveGapIds?.length ?? 0) === 0).length,
    };
  }

  createTeamProfile(input: {
    teamId: string;
    parentId: string;
    task: string;
    members: ToMPlanAgent[];
  }): ToMProfile {
    const memberTargets = input.members.map(member => member.name ?? member.archetype);
    const beliefScope = unique(input.members.flatMap(member => member.tomProfile?.beliefScope ?? []));
    const uncertainty = unique(input.members.flatMap(member => member.tomProfile?.uncertainty ?? []));
    return normalizeToMProfile({
      level: memberTargets.length >= 2 ? 2 : 1,
      subjectAgentId: input.teamId,
      beliefScope,
      goalModel: ['Aggregate member evidence and reconcile incompatible beliefs before reporting upward.'],
      uncertainty,
      perspective: 'team-level belief reconciler',
      observesAgents: memberTargets,
      modelsAgents: [input.parentId, ...memberTargets],
      capabilityScope: unique(input.members.flatMap(member => member.tomProfile?.capabilityScope ?? [])),
      cognitiveGaps: unique(input.members.flatMap(member => member.cognitiveGapIds ?? [])),
      models: [
        {
          targetId: input.parentId,
          targetType: 'agent',
          goalModel: ['Receive a synthesized result that covers the assigned cognitive gaps.'],
        },
        ...memberTargets.map(targetId => ({
          targetId,
          targetType: 'agent' as const,
          beliefModel: ['Provides one bounded specialist belief set.'],
        })),
      ],
      recursiveModels: memberTargets.map(targetId => ({
        observerId: input.teamId,
        targetId,
        relation: 'team evaluates member belief contribution',
        description: 'Compare this member output against sibling evidence, parent goals, and unresolved uncertainty.',
      })),
      purpose: `Coordinate specialist perspectives for: ${input.task}`,
    }, {
      level: 2,
      subjectAgentId: input.teamId,
      purpose: `Coordinate specialist perspectives for: ${input.task}`,
    });
  }

  private enrichPlan(
    plan: ToMPlanAgent,
    gaps: CognitiveGap[],
    analysis: ToMTaskAnalysis,
    peers: ToMPlanAgent[]
  ): ToMPlanAgent {
    const gapIds = gaps.map(gap => gap.id);
    const peerTargets = peers
      .filter(peer => peer !== plan && peer.archetype !== plan.archetype)
      .map(peer => peer.name ?? peer.archetype);
    const level = this.levelFor(plan.archetype, analysis.requiresHigherOrderToM, peerTargets.length);
    const modelsAgents = level >= 1
      ? unique([analysis.parentId, ...(level >= 2 ? peerTargets : [])])
      : [];
    const profile = normalizeToMProfile(plan.tomProfile, {
      level,
      subjectAgentId: plan.name ?? plan.archetype,
      purpose: gaps.length > 0
        ? `Fill cognitive gap(s): ${gaps.map(gap => gap.description).join(' ')}`
        : `Complete the assigned ${plan.archetype} task.`,
    });
    profile.level = plan.tomProfile?.level ?? level;
    profile.beliefScope = unique([...profile.beliefScope, ...gaps.flatMap(gap => gap.beliefScope)]);
    profile.goalModel = unique([...profile.goalModel, ...gaps.map(gap => gap.goal)]);
    profile.uncertainty = unique([...profile.uncertainty, ...gaps.flatMap(gap => gap.uncertainty)]);
    profile.perspective ??= gaps[0]?.requiredPerspective ?? ARCHETYPE_PERSPECTIVE[plan.archetype];
    profile.observesAgents = unique([...profile.observesAgents, analysis.parentId, ...peerTargets]);
    profile.modelsAgents = unique([...profile.modelsAgents, ...modelsAgents]);
    profile.capabilityScope = unique([...profile.capabilityScope, ...gaps.flatMap(gap => gap.requiredCapabilities)]);
    profile.cognitiveGaps = unique([...profile.cognitiveGaps, ...gapIds]);
    profile.models = this.modelsFor(plan.archetype, analysis.parentId, peerTargets, gaps, profile.models);
    if (profile.level >= 2 && peerTargets.length > 0) {
      profile.recursiveModels = uniqueRecursive([
        ...(profile.recursiveModels ?? []),
        ...peerTargets.map(targetId => ({
          observerId: analysis.parentId,
          targetId,
          relation: `${analysis.parentId} expects ${targetId} to cover a complementary cognitive gap`,
          description: `${plan.name ?? plan.archetype} evaluates whether ${targetId}'s belief supports the parent goal.`,
        })),
      ]);
    }
    return {
      ...plan,
      tomLevel: profile.level,
      tomProfile: profile,
      cognitiveGapIds: gapIds,
      existenceReason: gaps.length > 0
        ? `Created to cover ${gaps.map(gap => `${gap.kind}: ${gap.description}`).join('; ')}`
        : plan.existenceReason ?? 'Created to provide a bounded specialist perspective.',
    };
  }

  private matchGaps(archetype: ToMPlanAgent['archetype'], gaps: CognitiveGap[], assigned: Set<string>): CognitiveGap[] {
    const direct = gaps.filter(gap => !assigned.has(gap.id) && ARCHETYPE_BY_GAP[gap.kind] === archetype);
    if (direct.length > 0) return direct;
    const compatible = gaps.find(gap => !assigned.has(gap.id) && this.compatible(archetype, gap.kind));
    return compatible ? [compatible] : [];
  }

  private compatible(archetype: ToMPlanAgent['archetype'], kind: CognitiveGapKind): boolean {
    if (archetype === 'researcher') return kind === 'knowledge' || kind === 'evidence';
    if (archetype === 'critic') return kind === 'risk' || kind === 'perspective' || kind === 'verification';
    if (archetype === 'planner') return kind === 'planning' || kind === 'synthesis';
    if (archetype === 'summarizer') return kind === 'synthesis';
    if (archetype === 'tester') return kind === 'verification';
    if (archetype === 'coder') return kind === 'implementation';
    return kind === 'capability';
  }

  private levelFor(archetype: ToMPlanAgent['archetype'], higherOrder: boolean, peerCount: number): ToMProfile['level'] {
    if (archetype === 'critic' || archetype === 'summarizer') return higherOrder && peerCount > 0 ? 2 : 1;
    if (archetype === 'planner' || archetype === 'custom') return 1;
    return 0;
  }

  private modelsFor(
    archetype: ToMPlanAgent['archetype'],
    parentId: string,
    peers: string[],
    gaps: CognitiveGap[],
    existing: ToMProfile['models']
  ): ToMProfile['models'] {
    const models = [...existing];
    if (archetype === 'researcher' || archetype === 'coder' || archetype === 'tester') {
      models.push({
        targetId: 'environment',
        targetType: 'environment',
        beliefModel: unique(gaps.flatMap(gap => gap.beliefScope)),
        uncertaintyModel: unique(gaps.flatMap(gap => gap.uncertainty)),
      });
    }
    if (archetype === 'critic' || archetype === 'planner' || archetype === 'summarizer' || archetype === 'custom') {
      models.push({
        targetId: parentId,
        targetType: 'agent',
        goalModel: ['Receive a result that closes the assigned cognitive gap.'],
        uncertaintyModel: unique(gaps.flatMap(gap => gap.uncertainty)),
      });
    }
    if (archetype === 'critic' || archetype === 'summarizer') {
      models.push(...peers.map(targetId => ({
        targetId,
        targetType: 'agent' as const,
        beliefModel: ['Provides a complementary specialist belief that must be checked or reconciled.'],
      })));
    }
    return dedupeModels(models);
  }

  private taskForGap(gap: CognitiveGap, parentTask: string): string {
    return `${gap.goal} Address this uncertainty: ${gap.uncertainty.join(' ')} Parent task: ${parentTask}`;
  }
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.map(value => value.trim()).filter(Boolean)));
}

function dedupeModels(models: ToMProfile['models']): ToMProfile['models'] {
  const byTarget = new Map<string, ToMProfile['models'][number]>();
  for (const model of models) {
    const key = `${model.targetType}:${model.targetId}`;
    const current = byTarget.get(key);
    byTarget.set(key, current ? {
      ...current,
      beliefModel: unique([...(current.beliefModel ?? []), ...(model.beliefModel ?? [])]),
      goalModel: unique([...(current.goalModel ?? []), ...(model.goalModel ?? [])]),
      intentModel: unique([...(current.intentModel ?? []), ...(model.intentModel ?? [])]),
      uncertaintyModel: unique([...(current.uncertaintyModel ?? []), ...(model.uncertaintyModel ?? [])]),
    } : { ...model });
  }
  return [...byTarget.values()];
}

function uniqueRecursive(models: NonNullable<ToMProfile['recursiveModels']>): NonNullable<ToMProfile['recursiveModels']> {
  return [...new Map(models.map(model => [`${model.observerId}:${model.targetId}:${model.relation}`, model])).values()];
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
