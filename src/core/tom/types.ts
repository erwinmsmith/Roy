export type ToMLevel = 0 | 1 | 2 | 3;

export type ToMTargetType = 'user' | 'agent' | 'team' | 'environment';

export interface ToMTargetModel {
  targetId: string;
  targetType: ToMTargetType;
  beliefModel?: string[];
  goalModel?: string[];
  intentModel?: string[];
  uncertaintyModel?: string[];
}

export interface ToMRecursiveModel {
  observerId: string;
  targetId: string;
  relation: string;
  description: string;
}

export interface ToMProfile {
  level: ToMLevel;
  subjectAgentId: string;
  beliefScope: string[];
  goalModel: string[];
  uncertainty: string[];
  perspective?: string;
  observesAgents: string[];
  modelsAgents: string[];
  capabilityScope: string[];
  cognitiveGaps: string[];
  models: ToMTargetModel[];
  recursiveModels?: ToMRecursiveModel[];
  purpose: string;
}

export type CognitiveGapKind =
  | 'knowledge'
  | 'evidence'
  | 'perspective'
  | 'risk'
  | 'planning'
  | 'implementation'
  | 'verification'
  | 'synthesis'
  | 'capability';

export interface CognitiveGap {
  id: string;
  kind: CognitiveGapKind;
  description: string;
  requiredPerspective: string;
  beliefScope: string[];
  goal: string;
  uncertainty: string[];
  requiredCapabilities: string[];
  modelsTargets: string[];
  priority: number;
}

export interface ToMTaskAnalysis {
  id: string;
  parentId: string;
  task: string;
  parentBeliefs: string[];
  parentGoals: string[];
  parentUncertainties: string[];
  gaps: CognitiveGap[];
  requiresHigherOrderToM: boolean;
  rationale: string;
  createdAt: number;
}

export interface ToMPlanAgent {
  archetype: 'researcher' | 'critic' | 'planner' | 'coder' | 'summarizer' | 'tester' | 'custom';
  name?: string;
  task: string;
  tools?: string[];
  skills?: string[];
  tomLevel?: number;
  budgetTokens?: number;
  tomProfile?: ToMProfile;
  cognitiveGapIds?: string[];
  existenceReason?: string;
}

export interface ToMCoverageResult {
  coveredGapIds: string[];
  uncoveredGapIds: string[];
  coverageScore: number;
  perspectiveDiversity: number;
  higherOrderFit: number;
  unjustifiedAgentCount: number;
}
