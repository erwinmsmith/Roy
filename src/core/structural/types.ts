export const STRUCTURAL_SCHEMA_VERSION = 1 as const;

export type StructuralActionKind = 'CONTINUE' | 'BRANCH' | 'RETURN';

export interface ResourceEnvelope {
  computeTokens?: number;
  wallClockMs?: number;
  parallelSlots?: number;
  communicationEdges?: number;
  toolCalls?: number;
}

export type StructuralEventNodeKind =
  | 'agent'
  | 'message'
  | 'subtask'
  | 'tool_call'
  | 'tool_result'
  | 'child_result'
  | 'dependency'
  | 'artifact'
  | 'resource';

export type StructuralEventEdgeKind =
  | 'temporal'
  | 'derivation'
  | 'dependency'
  | 'communication'
  | 'tool_use'
  | 'return'
  | 'produces'
  | 'consumes';

export interface StructuralEventNode {
  id: string;
  kind: StructuralEventNodeKind;
  timestamp: number;
  actorId?: string;
  text?: string;
  status?: string;
  attributes?: Record<string, string | number | boolean | null>;
}

export interface StructuralEventEdge {
  id: string;
  kind: StructuralEventEdgeKind;
  from: string;
  to: string;
  required?: boolean;
  active?: boolean;
  attributes?: Record<string, string | number | boolean | null>;
}

export interface LocalObservableEventGraph {
  parentId: string;
  nodes: StructuralEventNode[];
  edges: StructuralEventEdge[];
  observedAt: number;
}

export interface DerivationNode {
  id: string;
  parentId?: string;
  status: 'ready' | 'running' | 'waiting' | 'completed' | 'closed' | 'failed';
  createdAt: number;
  closedAt?: number;
}

export interface DerivationTree {
  rootId: string;
  nodes: DerivationNode[];
  edges: Array<{ parentId: string; childId: string; createdAt: number }>;
}

export interface DependencyNode {
  id: string;
  kind: 'subgoal' | 'claim' | 'artifact' | 'tool_result' | 'child_result' | 'predicate';
  producerId?: string;
  resolved: boolean;
  valueRef?: string;
}

export interface DependencyEdge {
  producerId: string;
  consumerId: string;
  required: boolean;
}

export interface DependencyGraph {
  nodes: DependencyNode[];
  edges: DependencyEdge[];
}

export interface CommunicationEdge {
  from: string;
  to: string;
  required: boolean;
  active: boolean;
  createdAt: number;
  closedAt?: number;
}

export interface CommunicationGraph {
  nodes: string[];
  edges: CommunicationEdge[];
}

export interface ChildSpecification {
  id: string;
  task: string;
  context: string[];
  tools: string[];
  resources: ResourceEnvelope;
  outputContract: {
    format: 'text' | 'json' | 'artifact';
    requiredFields: string[];
    groundingRequired: boolean;
  };
  dependencies: string[];
}

export interface StructuralRandomness {
  environmentSeed: number;
  repeat: number;
  providerSeed?: number;
  deterministicProvider: boolean;
}

export interface StructuralCheckpoint {
  schemaVersion: typeof STRUCTURAL_SCHEMA_VERSION;
  id: string;
  taskId: string;
  correlationId: string;
  parentId: string;
  task: string;
  eventGraph: LocalObservableEventGraph;
  derivationTree: DerivationTree;
  dependencyGraph: DependencyGraph;
  communicationGraph: CommunicationGraph;
  resources: ResourceEnvelope;
  legalActions: StructuralActionKind[];
  environmentRevision: string;
  snapshotMode: 'clone' | 'fixture' | 'replay' | 'isolated';
  environmentState?: unknown;
  model: {
    provider: string;
    name: string;
    temperature?: number;
    maxTokens?: number;
  };
  randomness: StructuralRandomness;
  createdAt: number;
  fingerprint: string;
}

export interface StructuralDecision {
  action: StructuralActionKind;
  childSpecification?: ChildSpecification;
  rationale?: string;
  policyVersion?: string;
  confidence?: number;
}

export interface StructuralPolicy {
  readonly name: string;
  readonly version: string;
  decide(checkpoint: StructuralCheckpoint): Promise<StructuralDecision> | StructuralDecision;
  close?(): Promise<void> | void;
}

export interface StructuralRolloutResult {
  action: StructuralActionKind;
  childSpecification?: ChildSpecification;
  utility: number;
  durationMs: number;
  resourcesBefore: ResourceEnvelope;
  resourcesAfter: ResourceEnvelope;
  terminal: boolean;
  output?: unknown;
  error?: string;
  repeat: number;
}

export interface CounterfactualGroupResult {
  schemaVersion: typeof STRUCTURAL_SCHEMA_VERSION;
  checkpointId: string;
  checkpointFingerprint: string;
  rolloutPolicy: string;
  branchAggregate: 'mean';
  results: StructuralRolloutResult[];
  actionValues: Partial<Record<StructuralActionKind, number>>;
  outerAdvantages: Partial<Record<StructuralActionKind, number>>;
  branchAdvantages: Record<string, number>;
  createdAt: number;
}

export interface StructuralTraceRecord {
  schemaVersion: typeof STRUCTURAL_SCHEMA_VERSION;
  taskId: string;
  checkpointId: string;
  checkpointFingerprint: string;
  parentId: string;
  eventGraph: LocalObservableEventGraph;
  legalActions: StructuralActionKind[];
  action: StructuralActionKind;
  childSpecification?: ChildSpecification;
  resourcesBefore: ResourceEnvelope;
  resourcesAfter: ResourceEnvelope;
  utility: number;
  provider: string;
  model: string;
  tokenUsage: number;
  latencyMs: number;
  repeat: number;
  environmentRevision: string;
  error?: string;
}

export interface CloneableStructuralEnvironment<TSnapshot = unknown> {
  readonly revision: string;
  snapshot(): Promise<TSnapshot> | TSnapshot;
  restore(snapshot: TSnapshot): Promise<void> | void;
  execute(
    action: StructuralActionKind,
    childSpecification: ChildSpecification | undefined,
    repeat: number
  ): Promise<Omit<StructuralRolloutResult, 'action' | 'childSpecification' | 'repeat'>>
    | Omit<StructuralRolloutResult, 'action' | 'childSpecification' | 'repeat'>;
}
