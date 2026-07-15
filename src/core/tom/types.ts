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

export interface ToMAnalysisSignals {
  traceCount: number;
  participantCount: number;
  failedTraceCount: number;
  cancelledTraceCount: number;
  toolResultCount: number;
  evidenceTraceCount: number;
  conflictingTraceCount: number;
  evidenceCoverage: number;
  conflictLevel: number;
  uncertaintyLevel: number;
  observedKinds: string[];
  reliabilityConcerns: string[];
}

export interface ToMTaskAnalysisInput {
  task: string;
  parentId: string;
  parentProfile?: ToMProfile;
  signals?: Partial<ToMAnalysisSignals>;
}

export interface ToMTaskAnalysis {
  id: string;
  parentId: string;
  task: string;
  parentBeliefs: string[];
  parentGoals: string[];
  parentUncertainties: string[];
  gaps: CognitiveGap[];
  signals: ToMAnalysisSignals;
  source: 'task_only' | 'trace_augmented' | 'extension';
  confidence: number;
  requiresHigherOrderToM: boolean;
  rationale: string;
  createdAt: number;
}

export interface ToMDelegationEngine {
  analyzeTask(input: ToMTaskAnalysisInput): ToMTaskAnalysis;
  completePlans(analysis: ToMTaskAnalysis, inputPlans: ToMPlanAgent[], maxAgents: number): ToMPlanAgent[];
  evaluateCoverage(analysis: ToMTaskAnalysis, plans: ToMPlanAgent[]): ToMCoverageResult;
  createTeamProfile(input: {
    teamId: string;
    parentId: string;
    task: string;
    members: ToMPlanAgent[];
  }): ToMProfile;
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
