import { describe, expect, it } from 'vitest';
import Runtime from '../src/core/runtime/Runtime.js';
import {
  AppendOnlyDerivationTree,
  ControlledDerivationEnvironment,
  DependencyScheduler,
  LegacyStructuralPolicyAdapter,
  RecursiveInformationRealizationRuntime,
  PythonStructuralPolicyClient,
  StructuralCheckpointStore,
  StructuralCommunicationGraph,
  StructuralDependencyGraph,
  StructuralTokenLedger,
  consumeResources,
  projectParentLocalEventGraph,
  runCounterfactualGroup,
  standardizedRecord,
  type ChildSpecification,
  type EpistemicReport,
  type OpenAgentSpecification,
  type StructuralCheckpoint,
} from '../src/core/structural/index.js';

const child = (id: string): ChildSpecification => ({
  id,
  task: `solve ${id}`,
  context: [],
  tools: [],
  resources: { computeTokens: 1, parallelSlots: 1 },
  outputContract: { format: 'json', requiredFields: ['answer'], groundingRequired: false },
  dependencies: [],
});

function checkpoint(unresolved = false): StructuralCheckpoint {
  const store = new StructuralCheckpointStore();
  return store.create({
    id: 'checkpoint-1',
    taskId: 'task-1',
    correlationId: 'correlation-1',
    parentId: 'root',
    task: 'controlled task',
    eventGraph: { parentId: 'root', nodes: [], edges: [], observedAt: 123 },
    derivationTree: {
      rootId: 'root',
      nodes: [{ id: 'root', status: 'running', createdAt: 1 }],
      edges: [],
    },
    dependencyGraph: unresolved ? {
      nodes: [
        { id: 'evidence', kind: 'artifact', resolved: false },
        { id: 'answer', kind: 'subgoal', resolved: false },
      ],
      edges: [{ producerId: 'evidence', consumerId: 'answer', required: true }],
    } : { nodes: [], edges: [] },
    communicationGraph: { nodes: ['root'], edges: [] },
    resources: { computeTokens: 8, parallelSlots: 3, toolCalls: 3 },
    environmentRevision: 'controlled-derivation-v1',
    snapshotMode: 'clone',
    environmentState: { cursor: 0 },
    model: { provider: 'mock', name: 'fixture' },
    randomness: { environmentSeed: 7, repeat: 0, deterministicProvider: true },
    outputContractSatisfied: true,
    createdAt: 42,
  });
}

describe('structural runtime primitives', () => {
  it('keeps structural learning disabled by default for compatibility', () => {
    expect(new Runtime().isStructuralLearningEnabled()).toBe(false);
  });
  it('projects only Parent-visible event nodes and induced edges', () => {
    const graph = projectParentLocalEventGraph({
      parentId: 'parent',
      visibleActorIds: ['child-a'],
      nodes: [
        { id: 'p', kind: 'agent', actorId: 'parent', timestamp: 2 },
        { id: 'a', kind: 'message', actorId: 'child-a', timestamp: 1 },
        { id: 'b', kind: 'message', actorId: 'child-b', timestamp: 3 },
        { id: 'public', kind: 'artifact', timestamp: 4 },
      ],
      edges: [
        { id: 'visible', kind: 'communication', from: 'a', to: 'p' },
        { id: 'private', kind: 'communication', from: 'b', to: 'p' },
      ],
      observedAt: 10,
    });
    expect(graph.nodes.map(node => node.id)).toEqual(['a', 'p', 'public']);
    expect(graph.edges.map(edge => edge.id)).toEqual(['visible']);
  });

  it('round-trips immutable checkpoints and detects tampering', () => {
    const original = checkpoint();
    const store = new StructuralCheckpointStore();
    expect(store.restore(original)).toEqual(original);
    expect(() => store.restore({ ...original, task: 'tampered' })).toThrow(/fingerprint/);
  });

  it('masks RETURN while required dependencies are unresolved', () => {
    expect(checkpoint(true).legalActions).toEqual(['CONTINUE', 'BRANCH']);
    expect(checkpoint(false).legalActions).toEqual(['CONTINUE', 'BRANCH', 'RETURN']);
  });

  it('requires explicit output completion for RETURN and masks exhausted work actions', () => {
    const store = new StructuralCheckpointStore();
    const base = checkpoint();
    const exhausted = store.create({
      id: 'exhausted', taskId: base.taskId, correlationId: base.correlationId,
      parentId: base.parentId, task: base.task, eventGraph: base.eventGraph,
      derivationTree: base.derivationTree, dependencyGraph: base.dependencyGraph,
      communicationGraph: base.communicationGraph,
      resources: { computeTokens: 0, wallClockMs: 0, parallelSlots: 0 },
      environmentRevision: base.environmentRevision, snapshotMode: 'clone',
      model: base.model, randomness: base.randomness,
      outputContractSatisfied: false,
    });
    expect(exhausted.legalActions).toEqual([]);
  });

  it('conserves finite resources and rejects overspending', () => {
    expect(consumeResources({ computeTokens: 3, toolCalls: 1 }, { computeTokens: 2 }))
      .toEqual({ computeTokens: 1, toolCalls: 1 });
    expect(() => consumeResources({ computeTokens: 1 }, { computeTokens: 2 })).toThrow(/exhausted/);
  });

  it('keeps closed tree lineage immutable', () => {
    const tree = new AppendOnlyDerivationTree('root', 1);
    tree.addChild('root', { id: 'child', status: 'running', createdAt: 2 });
    tree.setStatus('child', 'closed', 3);
    expect(() => tree.setStatus('child', 'running')).toThrow(/immutable/);
    expect(tree.snapshot().edges).toEqual([{ parentId: 'root', childId: 'child', createdAt: 2 }]);
  });

  it('rejects dependency cycles and wakes consumers when requirements resolve', () => {
    const graph = new StructuralDependencyGraph();
    graph.addNode({ id: 'evidence', kind: 'artifact', resolved: false });
    graph.addNode({ id: 'answer', kind: 'subgoal', resolved: false });
    graph.addEdge({ producerId: 'evidence', consumerId: 'answer', required: true });
    expect(graph.unresolvedFor('answer')).toEqual(['evidence']);
    expect(graph.resolve('evidence', 'artifact://evidence')).toEqual(['answer']);
    expect(() => graph.addEdge({ producerId: 'answer', consumerId: 'evidence', required: true }))
      .toThrow(/cycle/);
  });

  it('automatically waits and wakes dependency consumers', () => {
    const tree = new AppendOnlyDerivationTree('root', 1);
    tree.addChild('root', { id: 'answer', status: 'ready', createdAt: 2 });
    const graph = new StructuralDependencyGraph();
    graph.addNode({ id: 'evidence', kind: 'artifact', resolved: false });
    graph.addNode({ id: 'answer', kind: 'subgoal', resolved: false });
    graph.addEdge({ producerId: 'evidence', consumerId: 'answer', required: true });
    const scheduler = new DependencyScheduler(graph, tree);
    expect(scheduler.waitOrRun('answer', 3)).toBe('waiting');
    expect(scheduler.resolveAndWake('evidence', 'artifact://evidence', 4)).toEqual(['answer']);
    expect(tree.snapshot().nodes.find(node => node.id === 'answer')?.status).toBe('running');
  });

  it('preserves required communication routes while optional shortcuts can close', () => {
    const graph = new StructuralCommunicationGraph();
    for (const id of ['a', 'b', 'c']) graph.addNode(id);
    graph.addEdge({ from: 'a', to: 'b', required: true, active: true, createdAt: 1 });
    graph.addEdge({ from: 'b', to: 'c', required: false, active: true, createdAt: 2 });
    expect(graph.canReach('a', 'c')).toBe(true);
    graph.closeOptionalEdge('b', 'c', 3);
    expect(graph.canReach('a', 'c')).toBe(false);
    expect(() => graph.closeOptionalEdge('a', 'b')).toThrow(/Required/);
  });

  it('enforces a hard token cap and restores ledger state', () => {
    const ledger = new StructuralTokenLedger(10);
    ledger.reserve(8);
    expect(() => ledger.reserve(3)).toThrow(/exceeded/);
    ledger.restore({ limit: 10, used: 4 });
    expect(ledger.state()).toMatchObject({ used: 4, remaining: 6, exhausted: false });
  });

  it('restores the exact base snapshot before every counterfactual', async () => {
    const environment = new ControlledDerivationEnvironment({
      id: 'task-1', family: 'activation', evidenceAvailable: true,
      requiredComputation: 2, hiddenEvidenceValue: 0.2, childActivationValue: 1,
      directComputationValue: 0.5,
      resources: { computeTokens: 8, parallelSlots: 3, toolCalls: 3 },
    });
    const base = environment.snapshot();
    const group = await runCounterfactualGroup({
      checkpoint: checkpoint(), environment, branchSpecifications: [child('a'), child('b')],
      repeats: 2, rolloutPolicy: 'fixture-mu',
    });
    expect(group.results).toHaveLength(8);
    expect(group.actionValues.BRANCH).toBeCloseTo(0.6);
    expect(environment.snapshot()).toEqual(base);
    expect(group.checkpointFingerprint).toBe(checkpoint().fingerprint);
  });

  it('gives zero advantages to zero-variance groups', () => {
    expect(standardizedRecord({ a: 0.5, b: 0.5 })).toEqual({ a: 0, b: 0 });
  });

  it('falls back when the Python sidecar times out', async () => {
    const policy = new PythonStructuralPolicyClient({
      command: process.execPath,
      args: ['-e', 'process.stdin.resume()'],
      timeoutMs: 50,
      fallback: new LegacyStructuralPolicyAdapter(),
    });
    await expect(policy.decide(checkpoint())).resolves.toMatchObject({ action: 'CONTINUE' });
    await policy.close();
  });
});

describe('recursive information realization runtime', () => {
  const report = (nodeId: string, gapId?: string): EpistemicReport => ({
    id: `report-${nodeId}`, nodeId, depth: nodeId === 'root' ? 0 : 1,
    localObjective: `objective-${nodeId}`, conclusion: 'partial', reasoningSummary: 'summary',
    claims: [{ id: `claim-${nodeId}`, statement: 'claim', status: 'tentative', originNodeId: nodeId }],
    evidence: [], externalObservations: [], assumptions: [],
    uncertainty: { confidence: 0.5, uncertainAbout: [], confidenceBasis: 'insufficient evidence' },
    conflicts: [], coverage: { resolved: [], unresolved: gapId ? [gapId] : [], notExamined: [] },
    blindSpots: [],
    residualRequirements: gapId ? [{
      id: gapId, description: 'obtain missing evidence', whyItMatters: 'claim depends on it',
      likelyMechanism: 'acquisition', requiredInformation: 'primary evidence', status: 'open',
      parentNodeId: nodeId,
    }] : [],
    proposedChildren: [], resolvedParentGap: false, informationToPropagate: [],
  });

  const specification = (): OpenAgentSpecification => ({
    id: 'spec-child', nodeId: 'child', parentId: 'root', depth: 1,
    parentGoal: 'answer task', triggeringGapId: 'gap-1',
    localObjective: 'obtain the primary evidence for the claim',
    refinement: {
      parentScope: 'answer task', childScope: 'obtain one primary source',
      triggeringRequirementId: 'gap-1', narrowerThanParent: true,
      newInformationNeeded: 'primary evidence', executableEndCondition: 'source is found or absence established',
      duplicatedByExistingNode: false,
    },
    requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
    externalAccess: { allowed: true, tools: ['kb.search'], purpose: 'obtain evidence' },
    expectedOutput: { requiredInformation: 'source and finding', outputType: 'epistemic_report' },
    terminationCondition: 'return a sourced finding',
  });

  it('derives only from explicit residual requirements and preserves observations', () => {
    const runtime = new RecursiveInformationRealizationRuntime('root', 'answer task', 1);
    runtime.apply({ kind: 'EXECUTE', actorNodeId: 'root', report: report('root', 'gap-1') }, 2);
    runtime.apply({ kind: 'DERIVE', actorNodeId: 'root', childSpecification: specification() }, 3);
    runtime.apply({
      kind: 'ACQUIRE', actorNodeId: 'child',
      observation: {
        id: 'observation-1', sourceType: 'kb', queryOrAction: 'search evidence',
        observation: 'external fact', provenance: 'fixture://kb', supports: ['claim-child'],
      },
    }, 4);
    const snapshot = runtime.snapshot();
    expect(snapshot.nodes.map(node => node.id)).toEqual(['root', 'child']);
    expect(snapshot.requirements[0]?.status).toBe('assigned');
    expect(snapshot.observations[0]?.observation).toBe('external fact');
    expect(snapshot.derivationEdges).toEqual([{ parentId: 'root', childId: 'child' }]);
    expect(RecursiveInformationRealizationRuntime.restore(snapshot).snapshot()).toEqual(snapshot);
    expect(() => RecursiveInformationRealizationRuntime.restore({
      ...snapshot,
      stopped: true,
    })).toThrow(/fingerprint/);
  });

  it('keeps dependency and communication graphs distinct and wakes consumers', () => {
    const runtime = new RecursiveInformationRealizationRuntime('root', 'answer task', 1);
    runtime.apply({ kind: 'EXECUTE', actorNodeId: 'root', report: report('root', 'gap-1') }, 2);
    runtime.apply({ kind: 'DERIVE', actorNodeId: 'root', childSpecification: specification() }, 3);
    runtime.addDependency('child', 'root', 'report:child');
    expect(runtime.waitForDependencies('root', 4)).toBe(true);
    runtime.apply({
      kind: 'CONNECT', actorNodeId: 'root',
      connection: { from: 'child', to: 'root', required: true },
    }, 5);
    expect(() => runtime.apply({ kind: 'STOP', actorNodeId: 'root', finalOutput: 'answer' }, 6))
      .toThrow(/dependencies/);
    runtime.apply({ kind: 'RETURN', actorNodeId: 'child', report: report('child') }, 7);
    expect(runtime.snapshot().dependencyEdges[0]?.resolved).toBe(true);
    expect(runtime.snapshot().nodes.find(node => node.id === 'root')?.status).toBe('ready');
    runtime.apply({ kind: 'STOP', actorNodeId: 'root', finalOutput: 'answer' }, 8);
    const snapshot = runtime.snapshot();
    expect(snapshot.stopped).toBe(true);
    expect(snapshot.dependencyEdges).toHaveLength(1);
    expect(snapshot.communicationEdges).toHaveLength(1);
  });
});
