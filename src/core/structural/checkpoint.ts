import { stableStructuralFingerprint } from './graphs.js';
import {
  STRUCTURAL_SCHEMA_VERSION,
  type CommunicationGraph,
  type DependencyGraph,
  type DerivationTree,
  type LocalObservableEventGraph,
  type ResourceEnvelope,
  type StructuralActionKind,
  type StructuralCheckpoint,
  type StructuralRandomness,
} from './types.js';

export interface CreateStructuralCheckpointInput {
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
  environmentRevision: string;
  snapshotMode?: StructuralCheckpoint['snapshotMode'];
  environmentState?: unknown;
  model: StructuralCheckpoint['model'];
  randomness: StructuralRandomness;
  outputContractSatisfied?: boolean;
  createdAt?: number;
}

export class StructuralCheckpointStore {
  private readonly checkpoints = new Map<string, StructuralCheckpoint>();

  create(input: CreateStructuralCheckpointInput): StructuralCheckpoint {
    const unresolved = input.dependencyGraph.nodes.some(node => !node.resolved)
      || input.dependencyGraph.edges.some(edge => edge.required
        && !input.dependencyGraph.nodes.find(node => node.id === edge.producerId)?.resolved);
    const legalActions: StructuralActionKind[] = [];
    if (hasContinueCapacity(input.resources)) legalActions.push('CONTINUE');
    if (hasBranchCapacity(input.resources)) legalActions.push('BRANCH');
    if (!unresolved && input.outputContractSatisfied === true) legalActions.push('RETURN');
    const createdAt = input.createdAt ?? Date.now();
    const checkpointWithoutFingerprint: Omit<StructuralCheckpoint, 'fingerprint'> = {
      schemaVersion: STRUCTURAL_SCHEMA_VERSION,
      id: input.id,
      taskId: input.taskId,
      correlationId: input.correlationId,
      parentId: input.parentId,
      task: input.task,
      eventGraph: structuredClone(input.eventGraph),
      derivationTree: structuredClone(input.derivationTree),
      dependencyGraph: structuredClone(input.dependencyGraph),
      communicationGraph: structuredClone(input.communicationGraph),
      resources: normalizeResources(input.resources),
      legalActions,
      environmentRevision: input.environmentRevision,
      snapshotMode: input.snapshotMode ?? 'replay',
      environmentState: structuredClone(input.environmentState),
      model: structuredClone(input.model),
      randomness: structuredClone(input.randomness),
      createdAt,
    };
    const fingerprint = stableStructuralFingerprint({
      ...checkpointWithoutFingerprint,
      createdAt: undefined,
      eventGraph: { ...checkpointWithoutFingerprint.eventGraph, observedAt: undefined },
    });
    const checkpoint = { ...checkpointWithoutFingerprint, fingerprint };
    this.checkpoints.set(checkpoint.id, structuredClone(checkpoint));
    return structuredClone(checkpoint);
  }

  get(id: string): StructuralCheckpoint | undefined {
    const checkpoint = this.checkpoints.get(id);
    return checkpoint ? structuredClone(checkpoint) : undefined;
  }

  restore(checkpoint: StructuralCheckpoint): StructuralCheckpoint {
    if (checkpoint.schemaVersion !== STRUCTURAL_SCHEMA_VERSION) {
      throw new Error(`Unsupported structural checkpoint schema ${checkpoint.schemaVersion}`);
    }
    const expected = stableStructuralFingerprint({
      ...checkpoint,
      fingerprint: undefined,
      createdAt: undefined,
      eventGraph: { ...checkpoint.eventGraph, observedAt: undefined },
    });
    if (expected !== checkpoint.fingerprint) throw new Error('Structural checkpoint fingerprint mismatch');
    this.checkpoints.set(checkpoint.id, structuredClone(checkpoint));
    return structuredClone(checkpoint);
  }

  list(): StructuralCheckpoint[] {
    return [...this.checkpoints.values()].map(item => structuredClone(item));
  }
}

export function consumeResources(
  envelope: ResourceEnvelope,
  cost: ResourceEnvelope
): ResourceEnvelope {
  const result: ResourceEnvelope = {};
  for (const key of resourceKeys) {
    const available = envelope[key];
    const consumed = cost[key] ?? 0;
    if (available !== undefined) {
      if (consumed > available) throw new Error(`Structural resource ${key} exhausted`);
      result[key] = available - consumed;
    }
  }
  return result;
}

const resourceKeys = [
  'computeTokens',
  'wallClockMs',
  'parallelSlots',
  'communicationEdges',
  'toolCalls',
] as const;

function normalizeResources(envelope: ResourceEnvelope): ResourceEnvelope {
  return Object.fromEntries(resourceKeys
    .filter(key => envelope[key] !== undefined)
    .map(key => [key, Math.max(0, Math.floor(envelope[key]!))])) as ResourceEnvelope;
}

function hasBranchCapacity(envelope: ResourceEnvelope): boolean {
  return (envelope.computeTokens === undefined || envelope.computeTokens > 0)
    && (envelope.parallelSlots === undefined || envelope.parallelSlots > 0);
}

function hasContinueCapacity(envelope: ResourceEnvelope): boolean {
  return (envelope.computeTokens === undefined || envelope.computeTokens > 0)
    && (envelope.wallClockMs === undefined || envelope.wallClockMs > 0);
}
