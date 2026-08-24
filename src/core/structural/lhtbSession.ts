import type { OrganizationAction, OrganizationPolicyRecord } from './informationRealizationTypes.js';
import { GlobalEpistemicStateRecorder, type GlobalEpistemicState,
  type RuntimeProcessEvent } from './globalEpistemicState.js';
import { RecursiveInformationRealizationRuntime,
  type RecursiveRuntimeSnapshot } from './recursiveRuntime.js';
import { stableStructuralFingerprint } from './graphs.js';
import type { SemanticUpdate } from './pythonSemanticState.js';
import type { EpistemicClaim, EpistemicEvidence, ExternalObservation } from './informationRealizationTypes.js';

export interface LHTBSessionSnapshot {
  schemaVersion: 1;
  trajectoryId: string;
  taskId: string;
  instruction: string;
  environmentRevision: string;
  organizationMode: 'single_agent_direct' | 'roy_runtime_heuristic' | 'learned_information_realization';
  initialSnapshotFingerprint: string;
  organizationSeed: number;
  runtime: RecursiveRuntimeSnapshot;
  processStates: GlobalEpistemicState[];
  policyRecords: OrganizationPolicyRecord[];
  processedSemanticEventIds: string[];
  semanticOverlay: {
    requirements: GlobalEpistemicState['requirements'];
    claims: EpistemicClaim[];
    assumptions: Array<Record<string, unknown>>;
    evidence: EpistemicEvidence[];
    observations: ExternalObservation[];
    relations: GlobalEpistemicState['semanticRelations'];
    blindSpots: string[];
  };
  pendingTerminalRequest?: TerminalRequest;
}

export interface TerminalRequest {
  id: string;
  command: string;
  cwd?: string;
  timeoutMs: number;
  nodeId: string;
  organizationActionKind?: 'ACQUIRE' | 'EXECUTE';
}

export interface TerminalResult {
  requestId: string;
  exitCode: number;
  stdout: string;
  stderr: string;
  durationMs: number;
  fileChanges?: string[];
}

export class RoyLHTBSession {
  private runtime: RecursiveInformationRealizationRuntime;
  private recorder = new GlobalEpistemicStateRecorder();
  private events: RuntimeProcessEvent[] = [];
  private policyRecords: OrganizationPolicyRecord[] = [];
  private usage = { inputTokens: 0, outputTokens: 0 };
  private processedSemanticEventIds = new Set<string>();
  private semanticRequirements: GlobalEpistemicState['requirements'] = [];
  private semanticClaims: EpistemicClaim[] = [];
  private semanticAssumptions: Array<Record<string, unknown>> = [];
  private semanticEvidence: EpistemicEvidence[] = [];
  private semanticObservations: ExternalObservation[] = [];
  private semanticRelations: GlobalEpistemicState['semanticRelations'] = [];
  private semanticBlindSpots: string[] = [];
  private pendingTerminalRequest?: TerminalRequest;

  constructor(readonly trajectoryId: string, readonly taskId: string,
    readonly instruction: string, readonly environmentRevision: string,
    readonly organizationMode: LHTBSessionSnapshot['organizationMode'] = 'learned_information_realization',
    readonly initialSnapshotFingerprint = '', readonly organizationSeed = 20260820) {
    this.runtime = new RecursiveInformationRealizationRuntime('root', instruction);
    this.runtime.ingestRequirement({
      id: 'root-task-requirement', description: instruction,
      whyItMatters: 'This is the environment task the organization must complete.',
      likelyMechanism: 'mixed', requiredInformation: 'A verified task completion result.',
      status: 'open', parentNodeId: 'root',
    });
    this.recordState();
  }

  static restore(snapshot: LHTBSessionSnapshot): RoyLHTBSession {
    const session = new RoyLHTBSession(snapshot.trajectoryId, snapshot.taskId,
      snapshot.instruction, snapshot.environmentRevision, snapshot.organizationMode,
      snapshot.initialSnapshotFingerprint, snapshot.organizationSeed);
    session.runtime = RecursiveInformationRealizationRuntime.restore(snapshot.runtime);
    session.recorder.restore(snapshot.processStates);
    session.events = structuredClone(snapshot.processStates.at(-1)?.runtimeEvents ?? []);
    session.policyRecords = structuredClone(snapshot.policyRecords ?? []);
    session.processedSemanticEventIds = new Set(snapshot.processedSemanticEventIds ?? []);
    const overlay = snapshot.semanticOverlay;
    session.semanticRequirements = structuredClone(overlay?.requirements ?? []);
    session.semanticClaims = structuredClone(overlay?.claims ?? []);
    session.semanticAssumptions = structuredClone(overlay?.assumptions ?? []);
    session.semanticEvidence = structuredClone(overlay?.evidence ?? []);
    session.semanticObservations = structuredClone(overlay?.observations ?? []);
    session.semanticRelations = structuredClone(overlay?.relations ?? []);
    session.semanticBlindSpots = structuredClone(overlay?.blindSpots ?? []);
    const usage = snapshot.processStates.at(-1)?.usage;
    session.usage = { inputTokens: usage?.inputTokens ?? 0, outputTokens: usage?.outputTokens ?? 0 };
    session.pendingTerminalRequest = structuredClone(snapshot.pendingTerminalRequest);
    return session;
  }

  applyOrganizationAction(action: OrganizationAction): GlobalEpistemicState {
    if (this.organizationMode === 'single_agent_direct'
      && (action.actorNodeId !== 'root' || action.kind === 'DERIVE' || action.kind === 'CONNECT')) {
      throw new Error('single_agent_direct enforces one root node without derivation or communication');
    }
    this.runtime.apply(action);
    this.events.push({
      id: `action-${this.events.length}`, kind: 'organization_action', at: Date.now(),
      nodeId: action.actorNodeId, attributes: { action: structuredClone(action) },
    });
    return this.recordState();
  }

  recordPolicyDecision(record: OrganizationPolicyRecord): void {
    this.policyRecords.push(structuredClone(record));
  }

  recordModelUsage(inputTokens: number, outputTokens: number, model?: string): void {
    this.usage.inputTokens += inputTokens;
    this.usage.outputTokens += outputTokens;
    this.events.push({ id: `usage-${this.events.length}`, kind: 'usage', at: Date.now(),
      attributes: { inputTokens, outputTokens, model } });
  }

  unprocessedSemanticEvents(): RuntimeProcessEvent[] {
    const semanticEvents: RuntimeProcessEvent[] = [];
    for (const event of this.events) {
      if (this.processedSemanticEventIds.has(event.id)) continue;
      if (event.kind === 'terminal_result') semanticEvents.push(event);
      else this.processedSemanticEventIds.add(event.id);
    }
    return structuredClone(semanticEvents);
  }

  applySemanticUpdate(update: SemanticUpdate): GlobalEpistemicState {
    if (this.processedSemanticEventIds.has(update.event_id)) {
      throw new Error(`Semantic event ${update.event_id} was already processed`);
    }
    this.processedSemanticEventIds.add(update.event_id);
    const sourceNodeId = this.events.find(event => event.id === update.event_id)?.nodeId
      ?? this.runtime.snapshot().rootId;
    for (const value of update.requirements) this.runtime.ingestRequirement({
      id: String(value.id), description: String(value.description ?? ''),
      whyItMatters: String(value.whyItMatters ?? value.why_it_matters ?? ''),
      likelyMechanism: ['acquisition', 'representation', 'conversion'].includes(
        String(value.likelyMechanism ?? value.likely_mechanism))
        ? String(value.likelyMechanism ?? value.likely_mechanism) as
          'acquisition' | 'representation' | 'conversion' : 'mixed',
      requiredInformation: String(value.requiredInformation ?? value.required_information ?? ''),
      status: 'open', parentNodeId: String(
        value.parentNodeId ?? value.parent_node_id ?? sourceNodeId
      ),
    });
    for (const value of update.claims) this.semanticClaims.push({
      id: String(value.id), statement: String(value.statement),
      status: ['supported', 'rejected'].includes(String(value.status))
        ? String(value.status) as 'supported' | 'rejected' : 'tentative',
      originNodeId: String(value.originNodeId ?? 'root'),
    });
    this.semanticAssumptions.push(...structuredClone(update.assumptions));
    for (const value of update.evidence) this.semanticEvidence.push({
      id: String(value.id), supports: Array.isArray(value.supports) ? value.supports.map(String) : [],
      contradicts: Array.isArray(value.contradicts) ? value.contradicts.map(String) : [],
      content: String(value.content ?? ''), provenance: String(value.provenance ?? 'deepseek-extractor'),
    });
    for (const value of update.external_observations) this.semanticObservations.push({
      id: String(value.id), sourceType: 'environment',
      queryOrAction: String(value.queryOrAction ?? ''), observation: String(value.observation ?? ''),
      provenance: String(value.provenance ?? 'deepseek-extractor'),
      supports: Array.isArray(value.supports) ? value.supports.map(String) : [],
    });
    for (const value of update.relations) {
      const provenance = value.provenance as Record<string, unknown> | undefined;
      this.semanticRelations.push({ leftId: String(value.left_id), rightId: String(value.right_id),
        label: String(value.label) as 'entail' | 'contradict' | 'unknown',
        probabilities: value.probabilities as { entail: number; contradict: number; unknown: number },
        model: String(provenance?.model ?? 'deepseek-v4-flash'),
        modelRevision: String(provenance?.model_revision ?? 'frozen'),
        requestId: String(provenance?.cache_key ?? ''), candidateSource: 'minilm_top_k',
        candidateRank: Number(value.candidate_rank ?? 0) });
    }
    this.semanticBlindSpots.push(...update.blind_spots);
    return this.recordState();
  }

  requestTerminal(request: TerminalRequest): GlobalEpistemicState {
    if (this.pendingTerminalRequest) throw new Error('A terminal request is already pending');
    this.pendingTerminalRequest = structuredClone(request);
    this.events.push({ id: `terminal-${request.id}`, kind: 'terminal_command', at: Date.now(),
      nodeId: request.nodeId, command: request.command, cwd: request.cwd,
      timeoutMs: request.timeoutMs });
    return this.recordState();
  }

  acceptTerminalResult(result: TerminalResult): GlobalEpistemicState {
    if (!this.pendingTerminalRequest || this.pendingTerminalRequest.id !== result.requestId) {
      throw new Error('Terminal result does not match the pending request');
    }
    this.events.push({ id: `result-${result.requestId}`, kind: 'terminal_result', at: Date.now(),
      nodeId: this.pendingTerminalRequest.nodeId, exitCode: result.exitCode,
      output: `${result.stdout}${result.stderr}`,
      attributes: { durationMs: result.durationMs, fileChanges: result.fileChanges ?? [] } });
    this.pendingTerminalRequest = undefined;
    return this.recordState();
  }

  resumeAfterVerifierRejection(feedback?: string): GlobalEpistemicState {
    const snapshot = this.runtime.snapshot();
    const material = {
      ...snapshot,
      nodes: snapshot.nodes.map(node => node.id === snapshot.rootId && node.status === 'completed'
        ? { ...node, status: 'ready' as const, updatedAt: Date.now() } : node),
      finalOutput: undefined,
      stopped: false,
    };
    const withoutFingerprint = { ...material } as Partial<RecursiveRuntimeSnapshot>;
    delete withoutFingerprint.fingerprint;
    this.runtime = RecursiveInformationRealizationRuntime.restore({
      ...withoutFingerprint,
      fingerprint: stableStructuralFingerprint(withoutFingerprint),
    } as RecursiveRuntimeSnapshot);
    this.events.push({ id: `verifier-${this.events.length}`, kind: 'verifier', at: Date.now(),
      attributes: { result: 'rejected', continuation: 'same_session', feedback } });
    return this.recordState();
  }

  snapshot(): LHTBSessionSnapshot {
    return { schemaVersion: 1, trajectoryId: this.trajectoryId, taskId: this.taskId,
      instruction: this.instruction, environmentRevision: this.environmentRevision,
      organizationMode: this.organizationMode,
      initialSnapshotFingerprint: this.initialSnapshotFingerprint,
      organizationSeed: this.organizationSeed,
      runtime: this.runtime.snapshot(), processStates: this.recorder.snapshot(),
      policyRecords: structuredClone(this.policyRecords),
      processedSemanticEventIds: [...this.processedSemanticEventIds],
      semanticOverlay: { requirements: structuredClone(this.semanticRequirements),
        claims: structuredClone(this.semanticClaims),
        assumptions: structuredClone(this.semanticAssumptions),
        evidence: structuredClone(this.semanticEvidence),
        observations: structuredClone(this.semanticObservations),
        relations: structuredClone(this.semanticRelations),
        blindSpots: structuredClone(this.semanticBlindSpots) },
      pendingTerminalRequest: structuredClone(this.pendingTerminalRequest) };
  }

  private recordState(): GlobalEpistemicState {
    const snapshot = this.runtime.snapshot();
    const reports = snapshot.reports;
    const previous = this.recorder.latest();
    return this.recorder.append({
      trajectoryId: this.trajectoryId, taskId: this.taskId,
      sequence: previous ? previous.sequence + 1 : 0,
      requirements: snapshot.requirements,
      claims: [...reports.flatMap(report => report.claims), ...this.semanticClaims],
      assumptions: [...reports.flatMap(report => report.assumptions), ...this.semanticAssumptions],
      evidence: [...reports.flatMap(report => report.evidence), ...this.semanticEvidence],
      externalObservations: [...snapshot.observations, ...this.semanticObservations],
      semanticRelations: this.semanticRelations,
      blindSpots: [...reports.flatMap(report => report.blindSpots), ...this.semanticBlindSpots],
      dependencies: snapshot.dependencyEdges,
      nodes: snapshot.nodes,
      dagEdges: [
        ...snapshot.derivationEdges.map(edge => ({ kind: 'derivation', from: edge.parentId,
          to: edge.childId })),
        ...snapshot.dependencyEdges.map(edge => ({ kind: 'dependency', from: edge.producerId,
          to: edge.consumerId, artifactId: edge.artifactId, resolved: edge.resolved })),
        ...snapshot.communicationEdges.map(edge => ({ kind: 'communication', from: edge.from,
          to: edge.to, required: edge.required, active: edge.active })),
      ],
      activeSubtree: snapshot.nodes.filter(node => ['ready', 'running', 'waiting'].includes(node.status))
        .map(node => node.id),
      runtimeEvents: this.events,
      usage: { inputTokens: this.usage.inputTokens, outputTokens: this.usage.outputTokens,
        wallTimeMs: this.events.reduce((total, event) => total
          + Number(event.attributes?.durationMs ?? 0), 0) },
      environmentRevision: this.environmentRevision,
      previousFingerprint: previous?.fingerprint,
    });
  }
}
