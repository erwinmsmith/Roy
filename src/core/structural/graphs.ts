import { createHash } from 'node:crypto';
import type {
  CommunicationEdge,
  CommunicationGraph,
  DependencyEdge,
  DependencyGraph,
  DependencyNode,
  DerivationNode,
  DerivationTree,
  LocalObservableEventGraph,
  StructuralEventEdge,
  StructuralEventNode,
} from './types.js';

export class AppendOnlyDerivationTree {
  private readonly nodes = new Map<string, DerivationNode>();
  private readonly edges: DerivationTree['edges'] = [];

  constructor(readonly rootId: string, createdAt = Date.now()) {
    this.nodes.set(rootId, { id: rootId, status: 'ready', createdAt });
  }

  addChild(parentId: string, child: DerivationNode): void {
    if (!this.nodes.has(parentId)) throw new Error(`Unknown derivation parent ${parentId}`);
    if (this.nodes.has(child.id)) throw new Error(`Derivation node ${child.id} already exists`);
    this.nodes.set(child.id, { ...child, parentId });
    this.edges.push({ parentId, childId: child.id, createdAt: child.createdAt });
  }

  setStatus(id: string, status: DerivationNode['status'], at = Date.now()): void {
    const node = this.nodes.get(id);
    if (!node) throw new Error(`Unknown derivation node ${id}`);
    if (node.status === 'closed' && status !== 'closed') {
      throw new Error(`Closed derivation node ${id} is immutable`);
    }
    node.status = status;
    if (status === 'closed') node.closedAt = at;
  }

  snapshot(): DerivationTree {
    return {
      rootId: this.rootId,
      nodes: structuredClone([...this.nodes.values()]),
      edges: structuredClone(this.edges),
    };
  }
}

export class StructuralDependencyGraph {
  private readonly nodes = new Map<string, DependencyNode>();
  private readonly edges: DependencyEdge[] = [];

  addNode(node: DependencyNode): void {
    if (this.nodes.has(node.id)) throw new Error(`Dependency node ${node.id} already exists`);
    this.nodes.set(node.id, structuredClone(node));
  }

  addEdge(edge: DependencyEdge): void {
    if (!this.nodes.has(edge.producerId) || !this.nodes.has(edge.consumerId)) {
      throw new Error('Dependency edge endpoints must exist');
    }
    if (this.reachable(edge.consumerId, edge.producerId)) {
      throw new Error(`Dependency edge ${edge.producerId} -> ${edge.consumerId} creates a cycle`);
    }
    if (!this.edges.some(item => item.producerId === edge.producerId && item.consumerId === edge.consumerId)) {
      this.edges.push(structuredClone(edge));
    }
  }

  resolve(id: string, valueRef?: string): string[] {
    const node = this.nodes.get(id);
    if (!node) throw new Error(`Unknown dependency node ${id}`);
    node.resolved = true;
    node.valueRef = valueRef;
    return this.edges
      .filter(edge => edge.producerId === id)
      .map(edge => edge.consumerId)
      .filter(consumerId => this.unresolvedFor(consumerId).length === 0);
  }

  unresolvedFor(consumerId: string): string[] {
    return this.edges
      .filter(edge => edge.consumerId === consumerId && edge.required)
      .map(edge => edge.producerId)
      .filter(id => !this.nodes.get(id)?.resolved);
  }

  snapshot(): DependencyGraph {
    return { nodes: structuredClone([...this.nodes.values()]), edges: structuredClone(this.edges) };
  }

  private reachable(from: string, target: string): boolean {
    const pending = [from];
    const visited = new Set<string>();
    while (pending.length > 0) {
      const id = pending.pop()!;
      if (id === target) return true;
      if (visited.has(id)) continue;
      visited.add(id);
      for (const edge of this.edges) if (edge.producerId === id) pending.push(edge.consumerId);
    }
    return false;
  }
}

export class DependencyScheduler {
  constructor(
    private readonly dependencies: StructuralDependencyGraph,
    private readonly tree: AppendOnlyDerivationTree
  ) {}

  waitOrRun(consumerId: string, at = Date.now()): 'waiting' | 'running' {
    const status = this.dependencies.unresolvedFor(consumerId).length > 0 ? 'waiting' : 'running';
    this.tree.setStatus(consumerId, status, at);
    return status;
  }

  resolveAndWake(producerId: string, valueRef?: string, at = Date.now()): string[] {
    const ready = this.dependencies.resolve(producerId, valueRef);
    for (const consumerId of ready) this.tree.setStatus(consumerId, 'running', at);
    return ready;
  }
}

export class StructuralCommunicationGraph {
  private readonly nodes = new Set<string>();
  private readonly edges: CommunicationEdge[] = [];

  addNode(id: string): void {
    this.nodes.add(id);
  }

  addEdge(edge: CommunicationEdge): void {
    if (!this.nodes.has(edge.from) || !this.nodes.has(edge.to)) {
      throw new Error('Communication edge endpoints must exist');
    }
    const existing = this.edges.find(item => item.from === edge.from && item.to === edge.to && item.active);
    if (existing) return;
    this.edges.push(structuredClone(edge));
  }

  closeOptionalEdge(from: string, to: string, at = Date.now()): void {
    const edge = this.edges.find(item => item.from === from && item.to === to && item.active);
    if (!edge) return;
    if (edge.required) throw new Error('Required communication edges cannot be removed');
    edge.active = false;
    edge.closedAt = at;
  }

  canReach(from: string, to: string): boolean {
    const pending = [from];
    const visited = new Set<string>();
    while (pending.length > 0) {
      const id = pending.pop()!;
      if (id === to) return true;
      if (visited.has(id)) continue;
      visited.add(id);
      for (const edge of this.edges) if (edge.active && edge.from === id) pending.push(edge.to);
    }
    return false;
  }

  assertDependenciesReachable(dependencies: DependencyGraph): void {
    const producerByNode = new Map(dependencies.nodes.map(node => [node.id, node.producerId]));
    for (const dependency of dependencies.edges.filter(edge => edge.required)) {
      const producer = producerByNode.get(dependency.producerId);
      const consumer = producerByNode.get(dependency.consumerId);
      if (producer && consumer && !this.canReach(producer, consumer)) {
        throw new Error(`Required dependency ${dependency.producerId} cannot reach ${dependency.consumerId}`);
      }
    }
  }

  snapshot(): CommunicationGraph {
    return { nodes: [...this.nodes], edges: structuredClone(this.edges) };
  }
}

export function projectParentLocalEventGraph(input: {
  parentId: string;
  nodes: StructuralEventNode[];
  edges: StructuralEventEdge[];
  visibleActorIds: string[];
  observedAt?: number;
}): LocalObservableEventGraph {
  const visibleActors = new Set([input.parentId, ...input.visibleActorIds]);
  const nodes = input.nodes.filter(node => !node.actorId || visibleActors.has(node.actorId));
  const nodeIds = new Set(nodes.map(node => node.id));
  const edges = input.edges.filter(edge => nodeIds.has(edge.from) && nodeIds.has(edge.to));
  return {
    parentId: input.parentId,
    nodes: structuredClone(nodes).sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id)),
    edges: structuredClone(edges).sort((a, b) => a.id.localeCompare(b.id)),
    observedAt: input.observedAt ?? Date.now(),
  };
}

export function stableStructuralFingerprint(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(sortValue(value))).digest('hex');
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => [key, sortValue(item)]));
}
