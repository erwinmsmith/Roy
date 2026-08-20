import { stableStructuralFingerprint } from './graphs.js';
import type {
  EpistemicReport,
  ExternalObservation,
  InformationRealizationNode,
  OpenAgentSpecification,
  OrganizationAction,
  ResidualRequirement,
} from './informationRealizationTypes.js';

export interface RecursiveRuntimeSnapshot {
  rootId: string;
  nodes: InformationRealizationNode[];
  requirements: ResidualRequirement[];
  reports: EpistemicReport[];
  observations: ExternalObservation[];
  derivationEdges: Array<{ parentId: string; childId: string }>;
  dependencyEdges: Array<{ producerId: string; consumerId: string; artifactId: string; resolved: boolean }>;
  communicationEdges: Array<{ from: string; to: string; required: boolean; active: boolean }>;
  finalOutput?: unknown;
  stopped: boolean;
  fingerprint: string;
}

export class RecursiveInformationRealizationRuntime {
  private readonly nodes = new Map<string, InformationRealizationNode>();
  private readonly requirements = new Map<string, ResidualRequirement>();
  private readonly reports = new Map<string, EpistemicReport>();
  private readonly observations = new Map<string, ExternalObservation>();
  private readonly derivationEdges: Array<{ parentId: string; childId: string }> = [];
  private readonly dependencyEdges: Array<{
    producerId: string;
    consumerId: string;
    artifactId: string;
    resolved: boolean;
  }> = [];
  private readonly communicationEdges: Array<{ from: string; to: string; required: boolean; active: boolean }> = [];
  private finalOutput: unknown;
  private stopped = false;

  constructor(readonly rootId: string, rootObjective: string, createdAt = Date.now()) {
    this.nodes.set(rootId, {
      id: rootId,
      depth: 0,
      localObjective: rootObjective,
      status: 'ready',
      createdAt,
      updatedAt: createdAt,
    });
  }

  static restore(snapshot: RecursiveRuntimeSnapshot): RecursiveInformationRealizationRuntime {
    const { fingerprint, ...material } = snapshot;
    if (stableStructuralFingerprint(material) !== fingerprint) {
      throw new Error('Recursive runtime snapshot fingerprint does not match its contents');
    }
    const root = snapshot.nodes.find(node => node.id === snapshot.rootId);
    if (!root) throw new Error(`Recursive runtime snapshot is missing root ${snapshot.rootId}`);
    const runtime = new RecursiveInformationRealizationRuntime(
      snapshot.rootId,
      root.localObjective,
      root.createdAt,
    );
    runtime.nodes.clear();
    for (const node of snapshot.nodes) runtime.nodes.set(node.id, structuredClone(node));
    for (const requirement of snapshot.requirements) {
      runtime.requirements.set(requirement.id, structuredClone(requirement));
    }
    for (const report of snapshot.reports) runtime.reports.set(report.id, structuredClone(report));
    for (const observation of snapshot.observations) {
      runtime.observations.set(observation.id, structuredClone(observation));
    }
    runtime.derivationEdges.push(...structuredClone(snapshot.derivationEdges));
    runtime.dependencyEdges.push(...structuredClone(snapshot.dependencyEdges));
    runtime.communicationEdges.push(...structuredClone(snapshot.communicationEdges));
    runtime.finalOutput = structuredClone(snapshot.finalOutput);
    runtime.stopped = snapshot.stopped;
    return runtime;
  }

  apply(action: OrganizationAction, at = Date.now()): void {
    if (this.stopped) throw new Error('Organization has already stopped');
    const actor = this.requireNode(action.actorNodeId);
    if (actor.status === 'pruned' || actor.status === 'returned' || actor.status === 'failed') {
      throw new Error(`Node ${actor.id} cannot act from status ${actor.status}`);
    }
    switch (action.kind) {
      case 'DERIVE':
        this.derive(actor, this.requireSpecification(action), at);
        break;
      case 'ACQUIRE':
        this.acquire(actor, action.observation, at);
        break;
      case 'CONNECT':
        this.connect(action.connection);
        break;
      case 'EXECUTE':
        actor.status = 'running';
        actor.updatedAt = at;
        if (action.report) this.recordReport(actor, action.report, false, at);
        break;
      case 'RETURN':
        this.returnReport(actor, action.report, at);
        break;
      case 'PRUNE':
        this.prune(action.targetNodeId, at);
        break;
      case 'STOP':
        if (actor.id !== this.rootId) throw new Error('Only the root node may STOP the organization');
        if (this.hasUnresolvedRequiredDependency()) {
          throw new Error('Cannot STOP while required dependencies remain unresolved');
        }
        this.finalOutput = action.finalOutput;
        this.stopped = true;
        actor.status = 'completed';
        actor.updatedAt = at;
        break;
    }
  }

  addDependency(producerId: string, consumerId: string, artifactId: string): void {
    this.requireNode(producerId);
    this.requireNode(consumerId);
    if (producerId === consumerId) throw new Error('Self-dependencies are not allowed');
    if (this.dependencyReachable(consumerId, producerId)) {
      throw new Error(`Dependency ${producerId} -> ${consumerId} creates a cycle`);
    }
    if (!this.dependencyEdges.some(edge => edge.producerId === producerId
      && edge.consumerId === consumerId && edge.artifactId === artifactId)) {
      this.dependencyEdges.push({ producerId, consumerId, artifactId, resolved: false });
    }
  }

  ingestRequirement(requirement: ResidualRequirement): void {
    this.requireNode(requirement.parentNodeId);
    if (this.requirements.has(requirement.id)) return;
    this.requirements.set(requirement.id, structuredClone(requirement));
  }

  waitForDependencies(nodeId: string, at = Date.now()): boolean {
    const node = this.requireNode(nodeId);
    const waiting = this.dependencyEdges.some(edge => edge.consumerId === nodeId && !edge.resolved);
    node.status = waiting ? 'waiting' : 'ready';
    node.updatedAt = at;
    return waiting;
  }

  resolveArtifact(producerId: string, artifactId: string, at = Date.now()): string[] {
    const awakened = new Set<string>();
    for (const edge of this.dependencyEdges) {
      if (edge.producerId !== producerId || edge.artifactId !== artifactId) continue;
      edge.resolved = true;
      if (!this.dependencyEdges.some(candidate => candidate.consumerId === edge.consumerId && !candidate.resolved)) {
        const consumer = this.requireNode(edge.consumerId);
        if (consumer.status === 'waiting') {
          consumer.status = 'ready';
          consumer.updatedAt = at;
          awakened.add(consumer.id);
        }
      }
    }
    return [...awakened];
  }

  snapshot(): RecursiveRuntimeSnapshot {
    const material = {
      rootId: this.rootId,
      nodes: [...this.nodes.values()],
      requirements: [...this.requirements.values()],
      reports: [...this.reports.values()],
      observations: [...this.observations.values()],
      derivationEdges: this.derivationEdges,
      dependencyEdges: this.dependencyEdges,
      communicationEdges: this.communicationEdges,
      finalOutput: this.finalOutput,
      stopped: this.stopped,
    };
    return structuredClone({ ...material, fingerprint: stableStructuralFingerprint(material) });
  }

  private derive(parent: InformationRealizationNode, specification: OpenAgentSpecification, at: number): void {
    if (specification.parentId !== parent.id) throw new Error('Child specification parent does not match actor');
    if (specification.depth !== parent.depth + 1) throw new Error('Child depth must increment parent depth by one');
    if (specification.nodeId === parent.id || this.nodes.has(specification.nodeId)) {
      throw new Error(`Duplicate child node ${specification.nodeId}`);
    }
    const refinement = specification.refinement;
    if (!refinement.narrowerThanParent || refinement.duplicatedByExistingNode) {
      throw new Error('Child specification fails strict refinement validation');
    }
    if (refinement.triggeringRequirementId !== specification.triggeringGapId) {
      throw new Error('Refinement must reference the triggering residual requirement');
    }
    if (!refinement.executableEndCondition.trim() || !specification.terminationCondition.trim()) {
      throw new Error('Child specification requires an executable termination condition');
    }
    const requirement = this.requirements.get(specification.triggeringGapId);
    if (!requirement || requirement.parentNodeId !== parent.id || requirement.status !== 'open') {
      throw new Error(`Triggering gap ${specification.triggeringGapId} is not an open parent requirement`);
    }
    requirement.status = 'assigned';
    this.nodes.set(specification.nodeId, {
      id: specification.nodeId,
      parentId: parent.id,
      depth: specification.depth,
      localObjective: specification.localObjective,
      triggeringGapId: specification.triggeringGapId,
      status: 'ready',
      specification: structuredClone(specification),
      createdAt: at,
      updatedAt: at,
    });
    this.derivationEdges.push({ parentId: parent.id, childId: specification.nodeId });
    for (const dependency of specification.dependencies ?? []) {
      this.addDependency(dependency.producerNodeId, specification.nodeId, dependency.artifactId);
    }
    if (specification.dependencies?.length) this.waitForDependencies(specification.nodeId, at);
  }

  private acquire(actor: InformationRealizationNode, observation: ExternalObservation | undefined, at: number): void {
    if (!observation) throw new Error('ACQUIRE requires an external observation');
    if (this.observations.has(observation.id)) throw new Error(`Duplicate observation ${observation.id}`);
    this.observations.set(observation.id, structuredClone(observation));
    actor.updatedAt = at;
  }

  private connect(connection: OrganizationAction['connection']): void {
    if (!connection) throw new Error('CONNECT requires endpoints');
    this.requireNode(connection.from);
    this.requireNode(connection.to);
    if (connection.from === connection.to) throw new Error('Communication self-edges are not allowed');
    if (!this.communicationEdges.some(edge => edge.from === connection.from
      && edge.to === connection.to && edge.active)) {
      this.communicationEdges.push({ ...connection, active: true });
    }
  }

  private returnReport(actor: InformationRealizationNode, report: EpistemicReport | undefined, at: number): void {
    this.recordReport(actor, report, true, at);
  }

  private recordReport(
    actor: InformationRealizationNode,
    report: EpistemicReport | undefined,
    returning: boolean,
    at: number
  ): void {
    if (!report || report.nodeId !== actor.id) {
      throw new Error(`${returning ? 'RETURN' : 'EXECUTE'} requires the actor epistemic report`);
    }
    if (this.reports.has(report.id)) throw new Error(`Duplicate report ${report.id}`);
    for (const claim of report.claims) {
      if (claim.originNodeId !== actor.id) throw new Error(`Claim ${claim.id} has an invalid origin`);
    }
    for (const requirement of report.residualRequirements) {
      if (requirement.parentNodeId !== actor.id || this.requirements.has(requirement.id)) {
        throw new Error(`Invalid residual requirement ${requirement.id}`);
      }
      this.requirements.set(requirement.id, structuredClone(requirement));
    }
    for (const observation of report.externalObservations) {
      if (!this.observations.has(observation.id)) this.observations.set(observation.id, structuredClone(observation));
    }
    this.reports.set(report.id, structuredClone(report));
    actor.reportId = report.id;
    actor.status = returning ? (actor.id === this.rootId ? 'completed' : 'returned') : 'ready';
    actor.updatedAt = at;
    if (actor.triggeringGapId && report.resolvedParentGap) {
      const requirement = this.requirements.get(actor.triggeringGapId);
      if (requirement) requirement.status = 'resolved';
    }
  }

  private prune(nodeId: string | undefined, at: number): void {
    if (!nodeId || nodeId === this.rootId) throw new Error('PRUNE requires a non-root target');
    const node = this.requireNode(nodeId);
    if (this.dependencyEdges.some(edge => edge.producerId === nodeId && !edge.resolved)) {
      throw new Error(`Cannot prune ${nodeId}; unresolved consumers depend on it`);
    }
    for (const child of this.derivationEdges.filter(edge => edge.parentId === nodeId)) {
      const childNode = this.requireNode(child.childId);
      if (childNode.status !== 'pruned' && childNode.status !== 'returned' && childNode.status !== 'completed') {
        throw new Error(`Cannot prune ${nodeId}; active child ${child.childId} remains`);
      }
    }
    node.status = 'pruned';
    node.updatedAt = at;
    for (const edge of this.communicationEdges) {
      if ((edge.from === nodeId || edge.to === nodeId) && !edge.required) edge.active = false;
    }
  }

  private requireSpecification(action: OrganizationAction): OpenAgentSpecification {
    if (!action.childSpecification) throw new Error('DERIVE requires a child specification');
    return action.childSpecification;
  }

  private requireNode(id: string): InformationRealizationNode {
    const node = this.nodes.get(id);
    if (!node) throw new Error(`Unknown information-realization node ${id}`);
    return node;
  }

  private hasUnresolvedRequiredDependency(): boolean {
    return this.dependencyEdges.some(edge => !edge.resolved);
  }

  private dependencyReachable(from: string, target: string): boolean {
    const pending = [from];
    const visited = new Set<string>();
    while (pending.length > 0) {
      const node = pending.pop()!;
      if (node === target) return true;
      if (visited.has(node)) continue;
      visited.add(node);
      for (const edge of this.dependencyEdges) if (edge.producerId === node) pending.push(edge.consumerId);
    }
    return false;
  }
}
