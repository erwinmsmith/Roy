import type { ResourceEnvelope } from './types.js';

export const INFORMATION_REALIZATION_SCHEMA_VERSION = 2 as const;

export type OrganizationActionKind =
  | 'DERIVE'
  | 'ACQUIRE'
  | 'CONNECT'
  | 'EXECUTE'
  | 'RETURN'
  | 'PRUNE'
  | 'STOP';

export type RealizationMechanism = 'acquisition' | 'representation' | 'conversion' | 'mixed';

export interface ResidualRequirement {
  id: string;
  description: string;
  whyItMatters: string;
  likelyMechanism: RealizationMechanism;
  requiredInformation: string;
  suggestedCapability?: string;
  possibleExternalAccess?: string[];
  status: 'open' | 'assigned' | 'resolved' | 'rejected';
  parentNodeId: string;
}

export interface RefinementCheck {
  parentScope: string;
  childScope: string;
  triggeringRequirementId: string;
  narrowerThanParent: boolean;
  newInformationNeeded: string;
  executableEndCondition: string;
  duplicatedByExistingNode: boolean;
}

export interface OpenAgentSpecification {
  id: string;
  nodeId: string;
  parentId: string;
  depth: number;
  parentGoal: string;
  triggeringGapId: string;
  localObjective: string;
  refinement: RefinementCheck;
  requiredClaims: string[];
  requiredEvidence: string[];
  relevantReportIds: string[];
  dependencies?: Array<{
    producerNodeId: string;
    artifactId: string;
  }>;
  externalAccess: {
    allowed: boolean;
    tools: string[];
    purpose?: string;
  };
  expectedOutput: {
    requiredInformation: string;
    outputType: 'epistemic_report';
  };
  terminationCondition: string;
  expectedResourceCost?: number;
}

export interface EpistemicClaim {
  id: string;
  statement: string;
  status: 'supported' | 'tentative' | 'rejected';
  originNodeId: string;
}

export interface EpistemicEvidence {
  id: string;
  supports: string[];
  contradicts?: string[];
  content: string;
  provenance: string;
}

export interface ExternalObservation {
  id: string;
  sourceType: 'web' | 'database' | 'memory' | 'code' | 'api' | 'environment' | 'tool' | 'kb';
  queryOrAction: string;
  observation: string;
  provenance: string;
  supports: string[];
}

export interface EpistemicReport {
  id: string;
  nodeId: string;
  parentId?: string;
  depth: number;
  localObjective: string;
  triggeringGapId?: string;
  conclusion: string;
  reasoningSummary: string;
  claims: EpistemicClaim[];
  evidence: EpistemicEvidence[];
  externalObservations: ExternalObservation[];
  assumptions: Array<{
    id: string;
    statement: string;
    status: 'verified' | 'unverified' | 'contradicted';
    supportingEvidence: string[];
  }>;
  uncertainty: {
    confidence: number;
    uncertainAbout: string[];
    confidenceBasis: string;
  };
  conflicts: Array<{
    target: string;
    issue: string;
    status: 'unresolved' | 'partial' | 'resolved';
    proposedResolution?: string;
  }>;
  coverage: {
    resolved: string[];
    unresolved: string[];
    notExamined: string[];
  };
  blindSpots: string[];
  residualRequirements: ResidualRequirement[];
  proposedChildren: OpenAgentSpecification[];
  resolvedParentGap: boolean;
  informationToPropagate: string[];
}

export interface InformationRealizationNode {
  id: string;
  parentId?: string;
  depth: number;
  localObjective: string;
  triggeringGapId?: string;
  status: 'ready' | 'running' | 'waiting' | 'completed' | 'returned' | 'pruned' | 'failed';
  specification?: OpenAgentSpecification;
  reportId?: string;
  createdAt: number;
  updatedAt: number;
}

export interface OrganizationAction {
  kind: OrganizationActionKind;
  actorNodeId: string;
  childSpecification?: OpenAgentSpecification;
  requirementId?: string;
  observation?: ExternalObservation;
  connection?: { from: string; to: string; required: boolean };
  report?: EpistemicReport;
  targetNodeId?: string;
  finalOutput?: unknown;
}

export interface OrganizationCandidate {
  id: string;
  kind: OrganizationActionKind;
  actorNodeId: string;
  description: string;
  schedulerComplexity: number;
  action: OrganizationAction;
}

export interface ExplorationEnvelope {
  id: string;
  minimumNodes: number;
  maximumNodes: number;
  minimumDepth: number;
  maximumDepth: number;
  mode: 'shallow' | 'medium' | 'deep' | 'expansive';
}

export interface OrganizationPolicyRecord {
  stateFingerprint: string;
  activeNodeId: string;
  candidateId: string;
  maskedOldLogProbability: number;
  envelopeId: string;
  policyState?: unknown;
}

export interface OrganizationRuntimeBudget {
  maximumLlmCalls: number;
  maximumToolCalls: number;
  maximumNodes: number;
  maximumDepth: number;
  maximumDecisions: number;
}

export interface InformationRealizationTrajectory {
  schemaVersion: typeof INFORMATION_REALIZATION_SCHEMA_VERSION;
  id: string;
  groupId: string;
  benchmark: 'tau3';
  domain: string;
  taskId: string;
  split: 'train' | 'validation' | 'test' | 'heldout';
  epoch: number;
  rolloutIndex: number;
  environmentSeed: number;
  organizationSeed: number;
  initialSnapshotFingerprint: string;
  envelope: ExplorationEnvelope;
  runtimeBudget: OrganizationRuntimeBudget;
  actions: OrganizationAction[];
  policyRecords: OrganizationPolicyRecord[];
  terminalUtility: number;
  realizedResources: ResourceEnvelope;
  benchmarkEpisode: unknown;
  terminal: boolean;
  terminated: boolean;
  truncated: boolean;
  terminationType: 'policy_stop' | 'truncated_resource' | 'truncated_environment';
  terminationReason?: string;
  finalOutput?: unknown;
}

export interface LHTBInformationRealizationTrajectory {
  schemaVersion: 1;
  id: string;
  groupId: string;
  benchmark: 'lhtb';
  taskId: string;
  category: string;
  split: 'train' | 'dev' | 'test';
  epoch: number;
  rolloutIndex: number;
  policyRevision: number;
  organizationSeed: number;
  taskChecksum: string;
  dockerDigest: string;
  runtimeConfig: Record<string, unknown>;
  initialSnapshotFingerprint: string;
  actions: OrganizationAction[];
  policyRecords: OrganizationPolicyRecord[];
  processStates: unknown[];
  terminalReward: number;
  complete: boolean;
  environmentFailure: boolean;
  acceptedForTraining: boolean;
  semanticAuditPath: string;
  runtimeEventsPath: string;
  harborResultPath: string;
}
