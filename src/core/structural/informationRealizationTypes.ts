import type { ResourceEnvelope } from './types.js';

export const INFORMATION_REALIZATION_SCHEMA_VERSION = 1 as const;

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
  expectedResourceCost: number;
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
  behaviorLogProbability: number;
  policyLogProbability: number;
  explorationAlpha: number;
  envelopeId: string;
  expectedResourceBudget: number;
  projectedExpectedResourceCost: number;
  policyState?: unknown;
}

export interface InformationRealizationTrajectory {
  schemaVersion: typeof INFORMATION_REALIZATION_SCHEMA_VERSION;
  id: string;
  groupId: string;
  benchmark: 'tau3';
  domain: string;
  taskId: string;
  split: 'train' | 'validation' | 'test' | 'heldout';
  envelope: ExplorationEnvelope;
  actions: OrganizationAction[];
  policyRecords: OrganizationPolicyRecord[];
  terminalUtility: number;
  realizedResources: ResourceEnvelope;
  terminal: boolean;
  finalOutput?: unknown;
}
