import type { ToMLevel, ToMProfile, ToMTargetModel } from './types.js';

export type ToMProfileInput = Partial<Omit<ToMProfile, 'level' | 'subjectAgentId' | 'models' | 'purpose'>> & {
  level?: ToMLevel;
  subjectAgentId?: string;
  models?: ToMTargetModel[];
  purpose?: string;
};

export function normalizeToMProfile(
  input: ToMProfileInput | undefined,
  fallback: Pick<ToMProfile, 'level' | 'subjectAgentId' | 'purpose'>
): ToMProfile {
  const models = Array.isArray(input?.models)
    ? input.models.map(model => ({
      ...model,
      beliefModel: unique(model.beliefModel),
      goalModel: unique(model.goalModel),
      intentModel: unique(model.intentModel),
      uncertaintyModel: unique(model.uncertaintyModel),
    }))
    : [];
  return {
    level: input?.level ?? fallback.level,
    subjectAgentId: input?.subjectAgentId?.trim() || fallback.subjectAgentId,
    beliefScope: unique(input?.beliefScope),
    goalModel: unique(input?.goalModel),
    uncertainty: unique(input?.uncertainty),
    perspective: input?.perspective?.trim() || undefined,
    observesAgents: unique(input?.observesAgents),
    modelsAgents: unique(input?.modelsAgents ?? models
      .filter(model => model.targetType === 'agent' || model.targetType === 'team')
      .map(model => model.targetId)),
    capabilityScope: unique(input?.capabilityScope),
    cognitiveGaps: unique(input?.cognitiveGaps),
    models,
    recursiveModels: input?.recursiveModels?.map(model => ({ ...model })),
    purpose: input?.purpose?.trim() || fallback.purpose,
  };
}

export function cloneToMProfile(profile: ToMProfile): ToMProfile {
  return normalizeToMProfile(profile, profile);
}

function unique(values: string[] | undefined): string[] {
  return Array.from(new Set((values ?? []).map(value => value.trim()).filter(Boolean)));
}

