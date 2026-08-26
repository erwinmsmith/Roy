import { DeepSeekProvider, LLMJSONParseError } from '../llm/providers/openai.js';
import type { OrganizationCandidate,
  OrganizationPolicyRecord } from './informationRealizationTypes.js';
import { PythonOrganizationPolicyClient } from './pythonOrganizationPolicy.js';
import { RoyLHTBSession } from './lhtbSession.js';
import { searchOrganizationMCTS } from './mctsOrganizationSearch.js';
import { PythonSemanticStateClient } from './pythonSemanticState.js';
import type { LLMJSONCompletionResult, LLMMessage, LLMCompletionOptions } from '../llm/types.js';
import { appendFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { RecursiveInformationRealizationRuntime } from './recursiveRuntime.js';

interface ProposedCandidate extends OrganizationCandidate {
  command?: string;
  cwd?: string;
  timeoutMs?: number;
}

interface RawProposedCandidate {
  id?: unknown;
  kind?: unknown;
  actorNodeId?: unknown;
  description?: unknown;
  schedulerComplexity?: unknown;
  command?: unknown;
  cwd?: unknown;
  timeoutMs?: unknown;
  action?: unknown;
}

interface ProposalResponse {
  preferred_candidate_id: string;
  candidates: RawProposedCandidate[];
}

interface CandidateValidation {
  candidates: ProposedCandidate[];
  dispositions: Array<{ index: number; id?: string; accepted: boolean; reasons: string[] }>;
}

export interface TopologySamplingProfile {
  id: 'compact' | 'branching' | 'recursive' | 'connected';
  preferredNodeRange: [number, number];
  preferredMinimumDepth: number;
  focus: string;
}

export function topologySamplingProfile(organizationSeed: number): TopologySamplingProfile {
  const profiles: TopologySamplingProfile[] = [
    { id: 'compact', preferredNodeRange: [2, 3], preferredMinimumDepth: 1,
      focus: 'small rooted structure; execute existing nodes before adding optional branches' },
    { id: 'branching', preferredNodeRange: [3, 5], preferredMinimumDepth: 1,
      focus: 'independent root-local branches with selective information handoff' },
    { id: 'recursive', preferredNodeRange: [5, 7], preferredMinimumDepth: 2,
      focus: 'execute a child, then refine a real child-local residual gap recursively' },
    { id: 'connected', preferredNodeRange: [6, 8], preferredMinimumDepth: 2,
      focus: 'richer derivation with novel communication and genuine report dependencies' },
  ];
  return profiles[Math.abs(Math.trunc(organizationSeed)) % profiles.length] ?? profiles[0];
}

export function resolveTopologySamplingProfile(
  organizationSeed: number,
  override = process.env.ROY_LHTB_TOPOLOGY_PROFILE
): TopologySamplingProfile {
  if (!override) return topologySamplingProfile(organizationSeed);
  const ids: TopologySamplingProfile['id'][] = ['compact', 'branching', 'recursive', 'connected'];
  const index = ids.indexOf(override as TopologySamplingProfile['id']);
  if (index < 0) throw new Error(`Invalid ROY_LHTB_TOPOLOGY_PROFILE ${override}`);
  return topologySamplingProfile(index);
}

interface SamplingPhase {
  id: 'expand_width' | 'seed_child_local_residual' | 'derive_child_local_residual' | 'stabilize';
  deepestActiveNodeIds: Set<string>;
  childLocalOpenGapIds: Set<string>;
}

export type TopologySamplingCandidate = Pick<OrganizationCandidate,
  'kind' | 'actorNodeId' | 'action'>;

function topologySamplingPhase(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>
): SamplingPhase {
  const profile = resolveTopologySamplingProfile(snapshot.organizationSeed);
  const activeNodes = snapshot.runtime.nodes.filter(node =>
    ['ready', 'running', 'waiting', 'completed'].includes(node.status));
  const maximumDepth = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
  const deepestActiveNodeIds = new Set(activeNodes.filter(node => node.depth === maximumDepth)
    .map(node => node.id));
  const childLocalOpenGapIds = new Set(snapshot.runtime.requirements.filter(requirement =>
    requirement.status === 'open' && deepestActiveNodeIds.has(requirement.parentNodeId)
    && requirement.parentNodeId !== snapshot.runtime.rootId).map(requirement => requirement.id));
  if (maximumDepth < profile.preferredMinimumDepth && snapshot.runtime.nodes.length > 1) {
    return { id: childLocalOpenGapIds.size > 0
      ? 'derive_child_local_residual' : 'seed_child_local_residual',
    deepestActiveNodeIds, childLocalOpenGapIds };
  }
  if (snapshot.runtime.nodes.length >= profile.preferredNodeRange[1]) {
    return { id: 'stabilize', deepestActiveNodeIds, childLocalOpenGapIds };
  }
  return { id: 'expand_width', deepestActiveNodeIds, childLocalOpenGapIds };
}

function topologySamplingCandidateMatchesPhase(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>,
  candidate: TopologySamplingCandidate
): boolean {
  const phase = topologySamplingPhase(snapshot);
  if (phase.id === 'stabilize') return candidate.kind !== 'DERIVE';
  if (phase.id === 'seed_child_local_residual') {
    return phase.deepestActiveNodeIds.has(candidate.actorNodeId)
      && (candidate.kind === 'ACQUIRE' || candidate.kind === 'EXECUTE');
  }
  if (phase.id === 'derive_child_local_residual') {
    const gapId = candidate.action.childSpecification?.triggeringGapId;
    const reuseGapId = candidate.action.requirementId;
    return phase.deepestActiveNodeIds.has(candidate.actorNodeId)
      && ((candidate.kind === 'DERIVE'
        && typeof gapId === 'string' && phase.childLocalOpenGapIds.has(gapId))
        || (candidate.kind === 'CONNECT'
          && typeof reuseGapId === 'string' && phase.childLocalOpenGapIds.has(reuseGapId)));
  }
  return true;
}

export function topologySamplingCandidateLogitBias(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>,
  candidate: TopologySamplingCandidate
): number {
  const profile = resolveTopologySamplingProfile(snapshot.organizationSeed);
  const phase = topologySamplingPhase(snapshot);
  const matchesPhase = topologySamplingCandidateMatchesPhase(snapshot, candidate);
  if (phase.id === 'stabilize') return candidate.kind === 'DERIVE' ? -8 : 0;
  if (phase.id === 'seed_child_local_residual') {
    if (matchesPhase) return 4;
    return candidate.kind === 'DERIVE' ? -8 : -2;
  }
  if (phase.id === 'derive_child_local_residual') return matchesPhase ? 4 : -4;
  if (candidate.kind === 'DERIVE'
    && snapshot.runtime.nodes.length < profile.preferredNodeRange[0]) return 2;
  return 0;
}

function topologySamplingActiveNodeLogitBias(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>, nodeId: string
): number {
  const phase = topologySamplingPhase(snapshot);
  if (phase.id !== 'seed_child_local_residual'
    && phase.id !== 'derive_child_local_residual') return 0;
  return phase.deepestActiveNodeIds.has(nodeId) ? 4 : -4;
}

export type ControllerResult =
  | { status: 'terminal_request'; request: { id: string; command: string; cwd?: string;
    timeoutMs: number; nodeId: string }; snapshot: ReturnType<RoyLHTBSession['snapshot']> }
  | { status: 'continue'; snapshot: ReturnType<RoyLHTBSession['snapshot']> }
  | { status: 'completed'; snapshot: ReturnType<RoyLHTBSession['snapshot']> };

interface ControllerProvider {
  isConfigured(): boolean;
  completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions):
    Promise<LLMJSONCompletionResult<T>>;
}

interface ControllerSemanticClient {
  processEvent(event: Parameters<PythonSemanticStateClient['processEvent']>[0],
    existingState: Record<string, unknown>):
    ReturnType<PythonSemanticStateClient['processEvent']>;
  close(): void;
}

export interface LHTBControllerOptions {
  provider?: ControllerProvider;
  semantic?: ControllerSemanticClient;
  learnedPolicy?: PythonOrganizationPolicyClient;
  auditRoot?: string | false;
}

const PROPOSER_PROMPT = `You generate open organization candidates for Roy on LHTB.
Return JSON only: {"preferred_candidate_id": string, "candidates": [...]}.
Every candidate must have this exact outer shape:
{"id": string, "kind": string, "actorNodeId": string, "description": string,
 "schedulerComplexity": number, "action": object}.
Allowed kinds are DERIVE, ACQUIRE, CONNECT, EXECUTE, RETURN, PRUNE, STOP.
For ACQUIRE and EXECUTE, command, optional cwd and timeoutMs are outer candidate fields, while
action contains only kind and actorNodeId. ACQUIRE inspects external state. EXECUTE changes the
workspace or runs verification. Keep every command under 8000 characters. Prefer short incremental
edits or an existing script over embedding an entire source file in one JSON response. Examples:
{"id":"inspect","kind":"ACQUIRE","actorNodeId":"root","description":"Inspect files",
 "schedulerComplexity":1,"command":"find . -maxdepth 2 -type f","timeoutMs":120000,
 "action":{"kind":"ACQUIRE","actorNodeId":"root"}}.
{"id":"implement","kind":"EXECUTE","actorNodeId":"root","description":"Implement and test",
 "schedulerComplexity":3,"command":"python /app/implement.py && pytest -q","timeoutMs":120000,
 "action":{"kind":"EXECUTE","actorNodeId":"root"}}.
Do not put command inside action and do not fabricate an observation or execution report.
DERIVE must use an exact open requirement ID from organization.requirements and this action schema:
{"kind":"DERIVE","actorNodeId":"<parent>","childSpecification":{"id":"<spec-id>",
"nodeId":"<new-agent-id>","parentId":"<parent>","depth":<parent-depth+1>,
"parentGoal":"<parent objective>","triggeringGapId":"<open requirement id>",
"localObjective":"<strictly narrower executable objective>","refinement":{"parentScope":"...",
"childScope":"...","triggeringRequirementId":"<same requirement id>",
"narrowerThanParent":true,"newInformationNeeded":"...","executableEndCondition":"...",
"duplicatedByExistingNode":false},"requiredClaims":[],"requiredEvidence":[],
"relevantReportIds":[],"externalAccess":{"allowed":true,"tools":["terminal"],
"purpose":"..."},"expectedOutput":{"requiredInformation":"...",
"outputType":"epistemic_report"},"terminationCondition":"...",
"reuseReview":{"searchedNodeIds":["<every spawned agent id>"],
"decision":"spawn_distinct","reason":"why no existing spawned agent can perform this gap"}}}.
Before every DERIVE, semantically inspect every entry in spawnedAgentLibrary. Do not use lexical
overlap, keywords, regexes, or an embedding threshold as the final reuse decision. If an active
existing agent can satisfy the requirement, do not DERIVE. Reuse it with:
{"kind":"CONNECT","actorNodeId":"<requirement owner>","requirementId":"<open requirement>",
"connection":{"from":"<requirement owner>","to":"<existing agent>","required":true},
"reuseReview":{"searchedNodeIds":["<every spawned agent id>"],"decision":"reuse_existing",
"reusableNodeId":"<existing agent>","reason":"semantic capability match"}}.
Creating a semantically equivalent agent is legal only for genuine concurrent load that one agent
cannot handle. In that case DERIVE must set reuseReview.decision="spawn_for_load", identify the
existing reusableNodeId, and include loadJustification with numeric parallelWorkUnits greater than
availableCapacity, every concrete parallelRequirementId, and a concrete reason. The list must
include the proposed triggering gap and an unresolved requirement already occupying the existing
agent. Topology coverage alone is never load justification.
When structuralExploration requests expansion and independent unresolved requirements exist,
include up to three distinct legal DERIVE candidates, each using a different exact open gap ID.
This is candidate coverage, not permission to invent work: never create a fake gap, duplicate an
existing node ID or objective, reuse an assigned gap, or derive non-refining work. A requirement extracted from a child event belongs
to that child unless its provenance explicitly identifies another active parent.
CONNECT action is {"kind":"CONNECT","actorNodeId":"<actor>","connection":{"from":"<existing>",
"to":"<different existing>","required":false}}. Once at least three nodes exist, include a novel
CONNECT candidate when a real information handoff would help; never repeat an active edge.
When a new child genuinely needs a report from an existing producer, childSpecification.dependencies
may use {"producerNodeId":"<existing>","artifactId":"report:<existing>"}; never fabricate a
dependency merely to make a DAG. PRUNE uses targetNodeId for a non-root node.
The sampling profile is an intervention for coverage, never a quality label. Progress one legal
organization action at a time. Once currentNodeCount reaches preferredTopologyRange[1], omit
DERIVE and stabilize the current organization with EXECUTE, ACQUIRE, RETURN, PRUNE, or a useful
novel CONNECT. This is a profile-conditioned sampling intervention, not a resource cost or reward.
For recursive and connected profiles, once the first child exists and currentMaximumDepth is below
minimumDepthTarget, do not keep assigning root gaps. Follow requiredNextStructuralPhase exactly:
run ACQUIRE/EXECUTE on one of deepestActiveNodeIds until it exposes a real child-local residual,
then DERIVE from that child using one of childLocalOpenGapIds. This produces root -> sub -> subsub
one action at a time. Never relabel a root requirement as child-local merely to satisfy depth.
RETURN action uses the property report (never epistemicReport), with this exact report shape:
{"id":"...","nodeId":"<actor>","parentId":"...","depth":<actor-depth>,"localObjective":"...",
"triggeringGapId":"...","conclusion":"...","reasoningSummary":"...","claims":[],
"evidence":[],"externalObservations":[],"assumptions":[],"uncertainty":{"confidence":0.5,
"uncertainAbout":[],"confidenceBasis":"..."},"conflicts":[],"coverage":{"resolved":[],
"unresolved":[],"notExamined":[]},"blindSpots":[],"residualRequirements":[],
"proposedChildren":[],"resolvedParentGap":false,"informationToPropagate":[]}.
Claims, evidence, observations, assumptions, conflicts, residual requirements and proposed children
must use their typed runtime schemas when non-empty. STOP is root-only and uses finalOutput.
Cover every active node that has a semantically useful next action; candidate actorNodeId must be
that node's exact ID. A node with no proposed candidate cannot be selected by the policy.
Never repeat an unchanged terminal command immediately after it failed without file changes;
propose a command that diagnoses or repairs the observed failure.
When the requested artifacts exist and the task's own end-to-end command succeeds, include a
legal RETURN candidate for a child or STOP candidate for root alongside any optional extra check;
record remaining uncertainty instead of creating an endless verification loop.
Propose only semantically useful actions. Do not expose hidden reasoning, benchmark grader data,
keyword fields or reward. The preferred candidate is your best next organization decision.`;

const PROPOSER_RECENT_EVENT_COUNT = 6;
const PROPOSER_EVENT_TEXT_LIMIT = 2_000;
const PROPOSER_ENTITY_TEXT_LIMIT = 600;
const PROPOSER_ENTITY_COUNT = 16;
const PROPOSER_GOAL_TEXT_LIMIT = 6_000;
const SEMANTIC_RECALL_ENTITY_COUNT = 32;
const PROPOSER_COMMAND_LIMIT = 8_000;

function projectText(value: unknown): unknown {
  if (typeof value !== 'string' || value.length <= PROPOSER_EVENT_TEXT_LIMIT) return value;
  const half = PROPOSER_EVENT_TEXT_LIMIT / 2;
  const omitted = value.length - (2 * half);
  return `${value.slice(0, half)}\n...[proposer projection omitted ${omitted} characters]...\n${value.slice(-half)}`;
}

function compactText(value: unknown, limit = PROPOSER_ENTITY_TEXT_LIMIT): unknown {
  if (typeof value !== 'string' || value.length <= limit) return value;
  const retained = Math.floor(limit / 2);
  return `${value.slice(0, retained)}\n...[compact state ref omitted ${value.length - 2 * retained} characters]...\n${value.slice(-retained)}`;
}

function compactEntities(values: unknown[], textFields: string[]): unknown[] {
  return values.slice(-PROPOSER_ENTITY_COUNT).map(value => {
    if (!value || typeof value !== 'object') return value;
    const source = value as Record<string, unknown>;
    const compact: Record<string, unknown> = {};
    for (const key of ['id', 'status', 'originNodeId', 'parentNodeId', 'sourceType',
      'provenance', 'supports', 'contradicts', 'likelyMechanism']) {
      if (source[key] !== undefined) compact[key] = source[key];
    }
    for (const key of textFields) {
      if (source[key] !== undefined) compact[key] = compactText(source[key]);
    }
    return compact;
  });
}

function compactEvent(event: Record<string, unknown>): Record<string, unknown> {
  const attributes = event.attributes && typeof event.attributes === 'object'
    ? event.attributes as Record<string, unknown> : undefined;
  const action = attributes?.action && typeof attributes.action === 'object'
    ? attributes.action as Record<string, unknown> : undefined;
  return {
    id: event.id, kind: event.kind, nodeId: event.nodeId, exitCode: event.exitCode,
    command: compactText(event.command, 1_000), cwd: event.cwd,
    outputPreview: projectText(event.output),
    traceRef: `raw-event:${String(event.id ?? 'unknown')}`,
    fileChanges: Array.isArray(attributes?.fileChanges)
      ? attributes.fileChanges.slice(0, 32) : undefined,
    action: action ? { kind: action.kind, actorNodeId: action.actorNodeId,
      targetNodeId: action.targetNodeId,
      childNodeId: (action.childSpecification as Record<string, unknown> | undefined)?.nodeId }
      : undefined,
  };
}

export function compactEpistemicWorkingState(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>
): Record<string, unknown> {
  const latest = snapshot.processStates.at(-1);
  const recentRuntimeEvents = (latest?.runtimeEvents ?? [])
    .slice(-PROPOSER_RECENT_EVENT_COUNT)
    .map(event => compactEvent(event as unknown as Record<string, unknown>));
  const activeNodes = snapshot.runtime.nodes.filter(node =>
    ['ready', 'running', 'waiting', 'completed'].includes(node.status));
  const compactNodes = activeNodes.map(node => ({ id: node.id, parentId: node.parentId,
    depth: node.depth, status: node.status, triggeringGapId: node.triggeringGapId,
    assignedRequirementIds: node.assignedRequirementIds ?? [],
    objective: compactText(node.localObjective) }));
  const requirements = snapshot.runtime.requirements.map(value => ({ id: value.id,
    status: value.status, parentNodeId: value.parentNodeId,
    assignedNodeId: value.assignedNodeId,
    description: compactText(value.description),
    requiredInformation: compactText(value.requiredInformation) }));
  const openRequirements = snapshot.runtime.requirements.filter(value => value.status === 'open'
    && activeNodes.some(node => node.id === value.parentNodeId));
  const profile = resolveTopologySamplingProfile(snapshot.organizationSeed);
  const maximumDepthReached = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
  return {
    rootGoal: compactText(snapshot.instruction, PROPOSER_GOAL_TEXT_LIMIT),
    organizationMode: snapshot.organizationMode,
    organization: {
      rootId: snapshot.runtime.rootId, stopped: snapshot.runtime.stopped,
      activeNodes: compactNodes,
      archivedNodeRefs: snapshot.runtime.nodes.filter(node => !activeNodes.some(
        active => active.id === node.id)).map(node => ({ id: node.id, status: node.status,
        reportId: node.reportId })),
      requirements,
      spawnedAgentLibrary: snapshot.runtime.nodes.filter(node => node.id !== snapshot.runtime.rootId)
        .map(node => ({ id: node.id, objective: compactText(node.localObjective),
          status: node.status, depth: node.depth,
          reusable: !['returned', 'pruned', 'failed'].includes(node.status),
          assignedRequirementIds: node.assignedRequirementIds ?? [],
          triggeringGapId: node.triggeringGapId,
          tools: node.specification?.externalAccess.tools ?? [],
          expectedOutput: node.specification?.expectedOutput.requiredInformation })),
      deriveEdges: snapshot.runtime.derivationEdges,
      dependencies: snapshot.runtime.dependencyEdges,
      communications: snapshot.runtime.communicationEdges,
      reportRefs: snapshot.runtime.reports.map(report => ({ id: report.id,
        nodeId: report.nodeId, residualRequirementIds: report.residualRequirements.map(
          value => value.id) })),
      observationRefs: snapshot.runtime.observations.slice(-PROPOSER_ENTITY_COUNT).map(
        value => ({ id: value.id, sourceType: value.sourceType, provenance: value.provenance,
          supports: value.supports })),
    },
    compactEpistemicState: latest ? {
      sequence: latest.sequence, fingerprint: latest.fingerprint,
      unresolvedRequirements: compactEntities(latest.requirements.filter(
        value => value.status === 'open'), ['description', 'requiredInformation', 'whyItMatters']),
      resolvedRequirementRefs: latest.requirements.filter(value => value.status !== 'open')
        .slice(-PROPOSER_ENTITY_COUNT).map(value => ({ id: value.id, status: value.status })),
      highRelevanceClaims: compactEntities(latest.claims, ['statement']),
      unresolvedAssumptions: compactEntities(latest.assumptions.filter(value =>
        String(value.status ?? 'unverified') !== 'verified'), ['statement']),
      evidenceRefs: compactEntities(latest.evidence, ['content']),
      observationRefs: compactEntities(latest.externalObservations,
        ['queryOrAction', 'observation']),
      unresolvedConflicts: latest.semanticRelations.filter(value => value.label === 'contradict')
        .slice(-PROPOSER_ENTITY_COUNT),
      blindSpots: latest.blindSpots.slice(-PROPOSER_ENTITY_COUNT).map(value => compactText(value)),
      recentStateDelta: recentRuntimeEvents,
      activeSubtree: latest.activeSubtree,
      resourceState: latest.usage,
    } : undefined,
    structuralExploration: {
      currentNodeCount: snapshot.runtime.nodes.length,
      currentMaximumDepth: maximumDepthReached,
      samplingProfile: profile.id,
      minimumNodeTarget: profile.preferredNodeRange[0],
      preferredTopologyRange: profile.preferredNodeRange,
      minimumDepthTarget: profile.preferredMinimumDepth,
      focus: profile.focus,
      realOpenGapCount: openRequirements.length,
      realOpenGapIds: openRequirements.map(value => value.id),
      desiredAdditionalChildren: Math.min(3, Math.max(0,
        profile.preferredNodeRange[0] - snapshot.runtime.nodes.length), openRequirements.length),
      preferredMaximumNodes: profile.preferredNodeRange[1],
      requiredNextStructuralPhase: topologySamplingPhase(snapshot).id,
      deepestActiveNodeIds: [...topologySamplingPhase(snapshot).deepestActiveNodeIds],
      childLocalOpenGapIds: [...topologySamplingPhase(snapshot).childLocalOpenGapIds],
      semantics: 'sampling capability target only; never synthesize gaps or add reward',
    },
    projection: { recentRuntimeEventCount: PROPOSER_RECENT_EVENT_COUNT,
      eventTextLimit: PROPOSER_EVENT_TEXT_LIMIT, entityTextLimit: PROPOSER_ENTITY_TEXT_LIMIT,
      entityCount: PROPOSER_ENTITY_COUNT, immutableRawLedgerPreserved: true,
      retrieval: 'Use raw-event/report/observation references only when more detail is needed' },
  };
}

export class LHTBAutonomousController {
  private readonly provider: ControllerProvider;
  private readonly learnedPolicy?: PythonOrganizationPolicyClient;
  private readonly semantic: ControllerSemanticClient;
  private readonly auditRoot?: string;

  constructor(options: LHTBControllerOptions = {}) {
    this.auditRoot = options.auditRoot === false ? undefined
      : options.auditRoot ?? process.env.ROY_LHTB_AUDIT_ROOT;
    this.provider = options.provider ?? new DeepSeekProvider({
      apiKey: process.env.DEEPSEEK_API_KEY,
      baseUrl: process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com',
      model: process.env.DEEPSEEK_MODEL ?? 'deepseek-v4-flash',
      temperature: 0,
      maxTokens: 32_768,
      timeoutMs: Number(process.env.ROY_LHTB_LLM_TIMEOUT_MS ?? 600_000),
      maxRetries: 0,
    });
    if (!this.provider.isConfigured()) throw new Error('DEEPSEEK_API_KEY is required');
    const semanticCommand = process.env.ROY_LHTB_SEMANTIC_COMMAND;
    if (!options.semantic && !semanticCommand) throw new Error('ROY_LHTB_SEMANTIC_COMMAND is required');
    const [semanticExecutable, ...semanticArgs] = (semanticCommand ?? '').split(' ').filter(Boolean);
    this.semantic = options.semantic
      ?? new PythonSemanticStateClient(semanticExecutable, semanticArgs);
    this.learnedPolicy = options.learnedPolicy;
    if (!this.learnedPolicy && process.env.ROY_LHTB_POLICY_COMMAND) {
      const [command, ...args] = process.env.ROY_LHTB_POLICY_COMMAND.split(' ').filter(Boolean);
      this.learnedPolicy = new PythonOrganizationPolicyClient(command, args);
    }
  }

  async advance(session: RoyLHTBSession, seed: number): Promise<ControllerResult> {
    for (const event of session.unprocessedSemanticEvents()) {
      const latest = session.snapshot().processStates.at(-1);
      const existingRequirements = (latest?.requirements ?? [])
        .filter(value => event.kind !== 'task_instruction'
          || value.id !== 'root-task-requirement')
        .slice(-SEMANTIC_RECALL_ENTITY_COUNT);
      const update = await this.semantic.processEvent(event, {
        requirements: existingRequirements,
        claims: (latest?.claims ?? []).slice(-SEMANTIC_RECALL_ENTITY_COUNT),
        assumptions: (latest?.assumptions ?? []).slice(-SEMANTIC_RECALL_ENTITY_COUNT),
        evidence: (latest?.evidence ?? []).slice(-SEMANTIC_RECALL_ENTITY_COUNT),
        external_observations: (latest?.externalObservations ?? [])
          .slice(-SEMANTIC_RECALL_ENTITY_COUNT),
      });
      session.applySemanticUpdate(update);
    }
    const snapshot = session.snapshot();
    if (snapshot.runtime.stopped) return { status: 'completed', snapshot };
    const requestState = compactEpistemicWorkingState(snapshot);
    const messages: LLMMessage[] = [
      { role: 'system', content: PROPOSER_PROMPT },
      { role: 'user', content: JSON.stringify(requestState) },
    ];
    const proposalAttempts = Math.max(1, Number(process.env.ROY_LHTB_PROPOSAL_ATTEMPTS ?? 3));
    let completion: LLMJSONCompletionResult<ProposalResponse> | undefined;
    let validation: CandidateValidation | undefined;
    for (let attempt = 1; attempt <= proposalAttempts; attempt += 1) {
      let current: LLMJSONCompletionResult<ProposalResponse>;
      try {
        current = await this.provider.completeJSONWithUsage<ProposalResponse>(messages,
          { temperature: 0, maxTokens: 32_768, thinking: { type: 'disabled' } });
      } catch (error) {
        if (!(error instanceof LLMJSONParseError)) throw error;
        session.recordModelUsage(error.completion.usage?.inputTokens
          ?? error.completion.usage?.promptTokens ?? 0,
        error.completion.usage?.outputTokens
          ?? error.completion.usage?.completionTokens ?? 0,
        error.completion.model);
        await this.auditProposalFailure(requestState, error, attempt);
        if (attempt === proposalAttempts) throw error;
        messages.splice(0, messages.length,
          { role: 'system', content: PROPOSER_PROMPT },
          { role: 'user', content: `${JSON.stringify(requestState)}\nA prior independent attempt was malformed. Return one concise, complete JSON object only.` }
        );
        continue;
      }
      await this.auditProposal(requestState, current);
      session.recordModelUsage(current.completion.usage?.inputTokens
        ?? current.completion.usage?.promptTokens ?? 0,
      current.completion.usage?.outputTokens
        ?? current.completion.usage?.completionTokens ?? 0,
      current.completion.model);
      const currentValidation = this.validateCandidates(current.value, session);
      const structuralDeficits = this.structuralCandidateDeficits(snapshot,
        currentValidation.candidates);
      if (structuralDeficits.length > 0) {
        currentValidation.dispositions.push({ index: -2, accepted: false,
          reasons: structuralDeficits });
      }
      await this.auditCandidateValidation(currentValidation);
      if (currentValidation.candidates.length > 0 && structuralDeficits.length === 0) {
        completion = current;
        validation = currentValidation;
        break;
      }
      const reasons = [...new Set(currentValidation.dispositions.flatMap(
        value => value.reasons
      ))];
      if (attempt === proposalAttempts) {
        if (currentValidation.candidates.length > 0 && structuralDeficits.length > 0) {
          // Topology profiles are sampling interventions, not hard resource or
          // validity constraints.  Give the proposer its configured repair
          // attempts, then preserve any genuinely legal continuation instead
          // of discarding the whole trajectory merely because the final
          // candidate interface did not cover the preferred topology.
          completion = current;
          validation = currentValidation;
          break;
        }
        if (reasons.length > 0 && reasons.every(reason => reason === 'inactive_actor')) {
          throw new Error('environment_invalid:inactive_actor');
        }
        const root = snapshot.runtime.nodes.find(node => node.id === snapshot.runtime.rootId
          && !['returned', 'pruned', 'failed'].includes(node.status));
        if (root) {
          try {
            session.applyOrganizationAction({ kind: 'STOP', actorNodeId: root.id,
              finalOutput: { status: 'policy_dead_end', reasons,
                message: 'The current policy produced no legal continuation.' } });
            return { status: 'completed', snapshot: session.snapshot() };
          } catch {
            // A required unresolved dependency can make STOP illegal. Preserve the
            // trajectory for audit rather than misclassifying it as environment-invalid.
          }
        }
        throw new Error(`policy_dead_end:no_legal_candidates:${reasons.join('; ')}`);
      }
      const rejectedCandidates = currentValidation.dispositions
        .filter(value => !value.accepted && value.index >= 0)
        .map(value => ({
          candidate: current.value.candidates[value.index],
          reasons: value.reasons,
        }));
      const rejectedContext = JSON.stringify(rejectedCandidates).slice(0, 12_000);
      messages.splice(0, messages.length,
        { role: 'system', content: PROPOSER_PROMPT },
        { role: 'user', content: `${JSON.stringify(requestState)}\nA prior candidate set was rejected by the runtime or did not cover the requested structural sampling interface: ${reasons.join('; ')}. Rejected candidates and their exact reasons: ${rejectedContext}. Use the exact realOpenGapIds in structuralExploration when DERIVE coverage is requested. Do not reproduce an unchanged rejected candidate, invent a gap, or repeat an active CONNECT edge. Return a concise, genuinely legal candidate set.` }
      );
    }
    if (!completion || !validation) throw new Error('DeepSeek proposer did not return a completion');
    const candidates = validation.candidates;
    let selected: ProposedCandidate;
    let policyRecord: OrganizationPolicyRecord | undefined;
    if (snapshot.organizationMode === 'learned_information_realization'
      && (this.mctsEnabled() || this.shouldInvokeOrganizationPolicy(snapshot))) {
      if (!this.learnedPolicy) {
        throw new Error('Learned mode requires ROY_LHTB_POLICY_COMMAND and has no heuristic fallback');
      }
      const decision = this.mctsEnabled()
        ? await this.selectWithMCTS(snapshot, candidates, seed)
        : await this.learnedPolicy.select(this.policyState(snapshot, candidates), candidates, seed);
      selected = decision.candidate as ProposedCandidate;
      policyRecord = decision.record;
    } else if (snapshot.organizationMode === 'learned_information_realization') {
      const local = this.localContinuationCandidate(snapshot, candidates,
        completion.value.preferred_candidate_id);
      if (local) selected = local;
      else {
        if (!this.learnedPolicy) {
          throw new Error('Learned mode requires ROY_LHTB_POLICY_COMMAND and has no heuristic fallback');
        }
        const decision = await this.learnedPolicy.select(
          this.policyState(snapshot, candidates), candidates, seed
        );
        selected = decision.candidate as ProposedCandidate;
        policyRecord = decision.record;
      }
    } else {
      selected = candidates.find(value => value.id === completion.value.preferred_candidate_id)
        ?? candidates[0];
    }
    if (policyRecord) session.recordPolicyDecision(policyRecord);
    if (selected.kind === 'ACQUIRE' || selected.kind === 'EXECUTE') {
      if (!selected.command?.trim()) {
        throw new Error(`${selected.kind} candidate requires a terminal command`);
      }
      const request = { id: `terminal-${snapshot.processStates.length}-${selected.id}`,
        command: selected.command, cwd: selected.cwd, timeoutMs: selected.timeoutMs ?? 120_000,
        nodeId: selected.actorNodeId, organizationActionKind: selected.kind };
      session.requestTerminal(request);
      return { status: 'terminal_request', request, snapshot: session.snapshot() };
    }
    session.applyOrganizationAction(selected.action);
    const result = session.snapshot();
    return result.runtime.stopped ? { status: 'completed', snapshot: result }
      : { status: 'continue', snapshot: result };
  }

  close(): void {
    this.learnedPolicy?.close();
    this.semantic.close();
  }

  private mctsEnabled(): boolean {
    return process.env.ROY_LHTB_MCTS_ENABLED === 'true';
  }

  private async selectWithMCTS(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, candidates: ProposedCandidate[], seed: number
  ): Promise<{ candidate: ProposedCandidate; record: OrganizationPolicyRecord }> {
    if (!this.learnedPolicy) throw new Error('MCTS requires the learned policy sidecar');
    const simulations = Math.max(1, Number(process.env.ROY_LHTB_MCTS_SIMULATIONS ?? 24));
    const maximumDepth = Math.max(1, Number(process.env.ROY_LHTB_MCTS_MAX_DEPTH ?? 3));
    const cpuCT = Math.max(0, Number(process.env.ROY_LHTB_MCTS_CPUCT ?? 1.5));
    const temperature = Math.max(1e-6,
      Number(process.env.ROY_LHTB_MCTS_TEMPERATURE ?? 1));
    const result = await searchOrganizationMCTS({ rootState: snapshot, candidates,
      simulations, maximumDepth, cpuCT, temperature, seed,
      expand: async (state, remaining) => this.expandMCTSNode(state, remaining) });
    const selected = result.candidate;
    const rootPolicyState = this.policyState(snapshot, candidates);
    const behaviorProbability = result.behaviorProbabilities[selected.id];
    if (!(behaviorProbability > 0)) throw new Error('MCTS selected zero-probability action');
    const actionProbability = Object.entries(result.behaviorProbabilities)
      .filter(([candidateId]) => candidates.find(value => value.id === candidateId)?.kind
        === selected.kind)
      .reduce((sum, [, value]) => sum + value, 0);
    const actorActionProbabilities = this.actionProbabilitySummary(candidates, result.actorPriors);
    const behaviorActionProbabilities = this.actionProbabilitySummary(
      candidates, result.behaviorProbabilities
    );
    return { candidate: selected, record: {
      stateFingerprint: String(rootPolicyState.state_fingerprint ?? ''),
      activeNodeId: selected.actorNodeId, candidateId: selected.id,
      maskedOldLogProbability: Math.log(behaviorProbability),
      maskedOldActionLogProbability: Math.log(actionProbability),
      maskedOldCandidateConditionalLogProbability:
        Math.log(behaviorProbability / actionProbability),
      envelopeId: String((rootPolicyState.envelope as Record<string, unknown>).id),
      policyState: rootPolicyState,
      availableActions: [...new Set(candidates.map(value => value.kind))],
      rawProbabilities: actorActionProbabilities,
      maskedProbabilities: behaviorActionProbabilities,
      selectedAction: selected.kind,
      numRealResidualGaps: Number(rootPolicyState.num_real_residual_gaps ?? 0),
      numChildProposals: Number(rootPolicyState.num_child_proposals ?? 0),
      stopLegalReason: String(rootPolicyState.stop_legal_reason ?? ''),
      explorationStopMasked: Boolean(rootPolicyState.exploration_stop_masked),
      behaviorPolicy: 'mcts_puct', actorCandidatePriors: result.actorPriors,
      mctsVisitCounts: result.visitCounts,
      mctsBehaviorProbabilities: result.behaviorProbabilities,
      mctsSimulations: simulations, mctsMaximumDepth: maximumDepth, mctsCpuCT: cpuCT,
      rootTargetValue: result.rootTargetValue,
      selectedChildTargetValue: result.selectedChildTargetValue,
      selectedProcessReward: result.selectedProcessReward,
      targetValueRevision: result.targetRevision,
      mctsSearchTrace: result.trace,
    } };
  }

  private async expandMCTSNode(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, candidates: ProposedCandidate[]
  ) {
    if (!this.learnedPolicy) throw new Error('MCTS requires the learned policy sidecar');
    const valid: Array<{ candidate: ProposedCandidate;
      state: ReturnType<RoyLHTBSession['snapshot']>; terminal: boolean }> = [];
    for (const candidate of candidates) {
      try {
        const probe = RoyLHTBSession.restore(snapshot);
        let terminal = false;
        if (candidate.kind === 'ACQUIRE' || candidate.kind === 'EXECUTE') {
          if (!candidate.command?.trim()) continue;
          const activeActor = snapshot.runtime.nodes.some(node => node.id === candidate.actorNodeId
            && ['ready', 'running', 'waiting', 'completed'].includes(node.status));
          if (!activeActor) continue;
          probe.requestTerminal({ id: `mcts-${snapshot.processStates.length}-${candidate.id}`,
            command: candidate.command, cwd: candidate.cwd,
            timeoutMs: candidate.timeoutMs ?? 120_000, nodeId: candidate.actorNodeId,
            organizationActionKind: candidate.kind });
          terminal = true;
        } else {
          probe.applyOrganizationAction(candidate.action);
          terminal = probe.snapshot().runtime.stopped;
        }
        valid.push({ candidate, state: probe.snapshot(), terminal });
      } catch {
        // A candidate may be legal at the root but become invalid after an
        // earlier hypothetical intervention. It simply leaves this search node.
      }
    }
    const currentPolicyState = this.policyState(snapshot,
      valid.length ? valid.map(value => value.candidate) : candidates);
    if (!valid.length) {
      const graph = currentPolicyState.event_graph as Record<string, unknown>;
      const current = await this.learnedPolicy.targetValue(graph);
      return { targetValue: current.targetValue, targetRevision: current.targetRevision,
        actorPriors: {}, children: [] };
    }
    const analysis = await this.learnedPolicy.analyze(currentPolicyState);
    const children = await Promise.all(valid.map(async value => {
      const childCandidates = candidates.filter(candidate => candidate.id !== value.candidate.id);
      const childPolicyState = this.policyState(value.state,
        childCandidates.length ? childCandidates : [value.candidate]);
      const childValue = await this.learnedPolicy!.targetValue(
        childPolicyState.event_graph as Record<string, unknown>
      );
      return { ...value, prior: analysis.candidatePriors[value.candidate.id] ?? 0,
        targetValue: childValue.targetValue };
    }));
    return { targetValue: analysis.targetValue, targetRevision: analysis.targetRevision,
      actorPriors: analysis.candidatePriors, children };
  }

  private actionProbabilitySummary(candidates: ProposedCandidate[],
    probabilities: Record<string, number>): OrganizationPolicyRecord['rawProbabilities'] {
    const result: Record<string, number> = { DERIVE: 0, ACQUIRE: 0, CONNECT: 0, EXECUTE: 0,
      RETURN: 0, PRUNE: 0, STOP: 0 };
    for (const candidate of candidates) {
      result[candidate.kind] = (result[candidate.kind] ?? 0) + (probabilities[candidate.id] ?? 0);
    }
    return result as OrganizationPolicyRecord['rawProbabilities'];
  }

  private shouldInvokeOrganizationPolicy(snapshot: ReturnType<RoyLHTBSession['snapshot']>): boolean {
    const previous = snapshot.policyRecords.at(-1)?.policyState;
    if (!previous || typeof previous !== 'object') return true;
    const policyState = previous as Record<string, unknown>;
    const events = snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    const priorTerminalCount = Number(policyState.terminal_result_count ?? 0);
    const terminalResults = events.filter(event => event.kind === 'terminal_result');
    const newTerminalResults = terminalResults.slice(priorTerminalCount);
    const interval = Math.max(1, Number(process.env.ROY_LHTB_ORGANIZATION_INTERVAL ?? 5));
    if (newTerminalResults.length >= interval) return true;
    if (newTerminalResults.some(event => (event.exitCode ?? 0) !== 0
      || (Array.isArray(event.attributes?.fileChanges) && event.attributes.fileChanges.length > 0))) {
      return true;
    }
    const previousRequirements = new Set(Array.isArray(policyState.unresolved_requirement_ids)
      ? policyState.unresolved_requirement_ids.map(String) : []);
    const currentRequirements = snapshot.runtime.requirements
      .filter(value => value.status === 'open').map(value => value.id);
    if (currentRequirements.some(value => !previousRequirements.has(value))) return true;
    const priorOrganizationActionCount = Number(policyState.organization_action_count ?? 0);
    const newOrganizationActions = events.filter(event => event.kind === 'organization_action')
      .slice(priorOrganizationActionCount);
    if (newOrganizationActions.some(event => {
      const action = event.attributes?.action as Record<string, unknown> | undefined;
      return action && ['DERIVE', 'CONNECT', 'RETURN', 'PRUNE'].includes(String(action.kind));
    })) return true;
    const contradictionCount = snapshot.processStates.at(-1)?.semanticRelations.filter(value =>
      value.label === 'contradict').length ?? 0;
    if (contradictionCount > Number(policyState.contradiction_count ?? 0)) return true;
    const completedNodeCount = snapshot.runtime.nodes.filter(node =>
      ['completed', 'returned'].includes(node.status)).length;
    return completedNodeCount > Number(policyState.completed_node_count ?? 0);
  }

  private localContinuationCandidate(snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[], preferredId: string): ProposedCandidate | undefined {
    const events = snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    const actorId = [...events].reverse().find(event =>
      event.kind === 'terminal_result' || event.kind === 'terminal_command')?.nodeId;
    const local = candidates.filter(candidate =>
      ['ACQUIRE', 'EXECUTE'].includes(candidate.kind)
      && (!actorId || candidate.actorNodeId === actorId));
    return local.find(candidate => candidate.id === preferredId) ?? local[0];
  }

  private validateCandidates(response: ProposalResponse, session: RoyLHTBSession): CandidateValidation {
    if (!Array.isArray(response.candidates) || response.candidates.length === 0) {
      return { candidates: [], dispositions: [{ index: -1, accepted: false,
        reasons: ['response_has_no_candidates'] }] };
    }
    const snapshot = session.snapshot();
    const active = new Set(snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status))
      .map(node => node.id));
    const direct = snapshot.organizationMode === 'single_agent_direct';
    const runtimeEvents = snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    let lastFailedCommand: { command?: string; cwd?: string } | undefined;
    for (let index = runtimeEvents.length - 1; index >= 0; index -= 1) {
      const event = runtimeEvents[index];
      if (event.kind !== 'terminal_result') continue;
      const fileChanges = event.attributes?.fileChanges;
      if ((event.exitCode ?? 0) !== 0 && (!Array.isArray(fileChanges) || fileChanges.length === 0)) {
        for (let commandIndex = index - 1; commandIndex >= 0; commandIndex -= 1) {
          const commandEvent = runtimeEvents[commandIndex];
          if (commandEvent.kind === 'terminal_command') {
            lastFailedCommand = { command: commandEvent.command, cwd: commandEvent.cwd };
            break;
          }
        }
      }
      break;
    }
    const allowedKinds = new Set(['DERIVE', 'ACQUIRE', 'CONNECT', 'EXECUTE', 'RETURN',
      'PRUNE', 'STOP']);
    const seenIds = new Set<string>();
    const seenConnections = new Set(snapshot.runtime.communicationEdges
      .filter(edge => edge.active).map(edge => `${edge.from}\u0000${edge.to}`));
    const spawnedAgentIds = snapshot.runtime.nodes
      .filter(node => node.id !== snapshot.runtime.rootId).map(node => node.id);
    const reusableAgentIds = new Set(snapshot.runtime.nodes.filter(node =>
      node.id !== snapshot.runtime.rootId
      && !['returned', 'pruned', 'failed'].includes(node.status)).map(node => node.id));
    const validateReuseReview = (
      value: unknown,
      decisions: Array<'reuse_existing' | 'spawn_distinct' | 'spawn_for_load'>,
      expectedNodeId?: string,
    ): string[] => {
      if (!value || typeof value !== 'object') return ['missing_agent_reuse_review'];
      const review = value as Record<string, unknown>;
      const searched = Array.isArray(review.searchedNodeIds)
        ? new Set(review.searchedNodeIds.map(String)) : new Set<string>();
      const reasons = spawnedAgentIds.filter(id => !searched.has(id))
        .map(id => `agent_reuse_search_omitted:${id}`);
      const decision = String(review.decision ?? '');
      if (!decisions.includes(decision as typeof decisions[number])) {
        reasons.push('invalid_agent_reuse_decision');
      }
      if (typeof review.reason !== 'string' || !review.reason.trim()) {
        reasons.push('missing_agent_reuse_reason');
      }
      const reusableNodeId = typeof review.reusableNodeId === 'string'
        ? review.reusableNodeId : undefined;
      if (decision === 'reuse_existing' || decision === 'spawn_for_load') {
        if (!reusableNodeId || !reusableAgentIds.has(reusableNodeId)) {
          reasons.push('agent_reuse_target_is_not_active');
        }
        if (expectedNodeId && reusableNodeId !== expectedNodeId) {
          reasons.push('agent_reuse_target_mismatch');
        }
      }
      if (decision === 'spawn_distinct' && reusableNodeId) {
        reasons.push('spawn_distinct_cannot_name_reusable_node');
      }
      if (decision === 'spawn_for_load') {
        const load = review.loadJustification;
        const record = load && typeof load === 'object' ? load as Record<string, unknown> : {};
        const units = Number(record.parallelWorkUnits);
        const capacity = Number(record.availableCapacity);
        if (!Number.isFinite(units) || !Number.isFinite(capacity)
          || capacity < 1 || units <= capacity) {
          reasons.push('invalid_agent_load_justification');
        }
        const requirementIds = Array.isArray(record.parallelRequirementIds)
          ? [...new Set(record.parallelRequirementIds.map(String))] : [];
        if (requirementIds.length !== units) {
          reasons.push('agent_load_requirement_count_mismatch');
        }
        const target = snapshot.runtime.nodes.find(node => node.id === reusableNodeId);
        const occupiedIds = new Set([
          target?.triggeringGapId,
          ...(target?.assignedRequirementIds ?? []),
        ].filter((id): id is string => typeof id === 'string').filter(id =>
          snapshot.runtime.requirements.some(requirement => requirement.id === id
            && ['open', 'assigned'].includes(requirement.status))));
        if (!requirementIds.some(id => occupiedIds.has(id))) {
          reasons.push('agent_load_target_has_no_parallel_occupied_requirement');
        }
        if (typeof record.reason !== 'string' || !record.reason.trim()) {
          reasons.push('missing_agent_load_reason');
        }
      }
      return reasons;
    };
    const candidates: ProposedCandidate[] = [];
    const dispositions: CandidateValidation['dispositions'] = [];
    response.candidates.forEach((raw, index) => {
      const reasons: string[] = [];
      const id = typeof raw?.id === 'string' ? raw.id.trim() : '';
      const kind = typeof raw?.kind === 'string' ? raw.kind.trim() : '';
      const actorNodeId = typeof raw?.actorNodeId === 'string' ? raw.actorNodeId.trim() : '';
      const description = typeof raw?.description === 'string' ? raw.description.trim() : '';
      const schedulerComplexity = Number(raw?.schedulerComplexity);
      const actionValue = raw?.action && typeof raw.action === 'object'
        ? raw.action as Record<string, unknown> : {};
      const explicitActionKind = typeof actionValue.kind === 'string' ? actionValue.kind
        : typeof actionValue.type === 'string' ? actionValue.type : undefined;
      const explicitActionActor = typeof actionValue.actorNodeId === 'string'
        ? actionValue.actorNodeId : undefined;
      if (!id) reasons.push('missing_id');
      else if (seenIds.has(id)) reasons.push('duplicate_id');
      if (!allowedKinds.has(kind)) reasons.push('invalid_kind');
      if (!actorNodeId || !active.has(actorNodeId)) reasons.push('inactive_actor');
      if (!description) reasons.push('missing_description');
      if (!Number.isFinite(schedulerComplexity)) reasons.push('invalid_scheduler_complexity');
      if (explicitActionKind && explicitActionKind !== kind) reasons.push('action_kind_mismatch');
      if (explicitActionActor && explicitActionActor !== actorNodeId) {
        reasons.push('action_actor_mismatch');
      }
      const directLegal = !direct || (actorNodeId === 'root'
        && ['ACQUIRE', 'EXECUTE', 'RETURN', 'STOP'].includes(kind));
      if (!directLegal) reasons.push('direct_mode_forbids_action');
      const commandValue = typeof raw?.command === 'string' ? raw.command
        : typeof actionValue.command === 'string' ? actionValue.command : undefined;
      const cwdValue = typeof raw?.cwd === 'string' ? raw.cwd
        : typeof actionValue.cwd === 'string' ? actionValue.cwd : undefined;
      const timeoutValue = raw?.timeoutMs ?? actionValue.timeoutMs;
      const timeoutMs = timeoutValue === undefined ? undefined : Number(timeoutValue);
      if (kind === 'ACQUIRE' && !commandValue?.trim()) reasons.push('acquire_missing_command');
      if (kind === 'EXECUTE' && !commandValue?.trim()) reasons.push('execute_missing_command');
      if ((kind === 'ACQUIRE' || kind === 'EXECUTE') && commandValue?.trim()
        && commandValue.trim() === lastFailedCommand?.command?.trim()
        && (cwdValue?.trim() ?? '') === (lastFailedCommand.cwd?.trim() ?? '')) {
        reasons.push('repeats_unchanged_failed_command');
      }
      if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) {
        reasons.push('invalid_timeout');
      }
      if (typeof commandValue === 'string' && commandValue.length > PROPOSER_COMMAND_LIMIT) {
        reasons.push(`command_too_large:${commandValue.length}/${PROPOSER_COMMAND_LIMIT}`);
      }
      const action = { ...actionValue, kind, actorNodeId } as OrganizationCandidate['action'];
      if (kind === 'DERIVE' && spawnedAgentIds.length > 0) {
        const specification = actionValue.childSpecification
          && typeof actionValue.childSpecification === 'object'
          ? actionValue.childSpecification as Record<string, unknown> : {};
        reasons.push(...validateReuseReview(specification.reuseReview,
          ['spawn_distinct', 'spawn_for_load']));
        const review = specification.reuseReview as Record<string, unknown> | undefined;
        const load = review?.loadJustification as Record<string, unknown> | undefined;
        if (review?.decision === 'spawn_for_load') {
          const requirementIds = Array.isArray(load?.parallelRequirementIds)
            ? load.parallelRequirementIds.map(String) : [];
          const triggeringGapId = typeof specification.triggeringGapId === 'string'
            ? specification.triggeringGapId : '';
          if (!triggeringGapId || !requirementIds.includes(triggeringGapId)) {
            reasons.push('agent_load_omits_triggering_requirement');
          }
        }
      }
      let connectionKeyToAccept: string | undefined;
      if (kind === 'CONNECT') {
        const connection = actionValue.connection && typeof actionValue.connection === 'object'
          ? actionValue.connection as Record<string, unknown> : undefined;
        const from = typeof connection?.from === 'string' ? connection.from : '';
        const to = typeof connection?.to === 'string' ? connection.to : '';
        const connectionKey = `${from}\u0000${to}`;
        const reuseRequirementId = typeof actionValue.requirementId === 'string'
          ? actionValue.requirementId : undefined;
        if (from && to && seenConnections.has(connectionKey) && !reuseRequirementId) {
          reasons.push('duplicate_active_connection');
        }
        if (reuseRequirementId) {
          reasons.push(...validateReuseReview(actionValue.reuseReview,
            ['reuse_existing'], to));
          const requirement = snapshot.runtime.requirements.find(value =>
            value.id === reuseRequirementId);
          if (!requirement || requirement.status !== 'open') {
            reasons.push('agent_reuse_requirement_is_not_open');
          } else if (requirement.parentNodeId !== actorNodeId || from !== actorNodeId) {
            reasons.push('agent_reuse_requirement_owner_mismatch');
          }
          if (connection?.required !== true) reasons.push('agent_reuse_connection_not_required');
        }
        if (from && to) connectionKeyToAccept = connectionKey;
      }
      const candidate = { id, kind, actorNodeId, description, schedulerComplexity,
        action, command: commandValue?.trim(), cwd: cwdValue?.trim(), timeoutMs } as ProposedCandidate;
      if (reasons.length === 0 && kind !== 'ACQUIRE' && kind !== 'EXECUTE') {
        try {
          const probe = RecursiveInformationRealizationRuntime.restore(snapshot.runtime);
          probe.apply(action, Date.now());
        } catch (error) {
          reasons.push(`runtime_rejected:${error instanceof Error ? error.message : String(error)}`);
        }
      }
      if (id) seenIds.add(id);
      const accepted = reasons.length === 0;
      dispositions.push({ index, id: id || undefined, accepted, reasons });
      if (accepted) {
        candidates.push(candidate);
        if (connectionKeyToAccept) seenConnections.add(connectionKeyToAccept);
      }
    });
    return { candidates, dispositions };
  }

  private structuralCandidateDeficits(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[]
  ): string[] {
    if (snapshot.organizationMode !== 'learned_information_realization') return [];
    const profile = resolveTopologySamplingProfile(snapshot.organizationSeed);
    const minimumNodes = profile.preferredNodeRange[0];
    if (minimumNodes <= 0) return [];
    const active = new Set(snapshot.runtime.nodes.filter(node =>
      ['ready', 'running', 'waiting', 'completed'].includes(node.status)).map(node => node.id));
    const openGapIds = new Set(snapshot.runtime.requirements.filter(requirement =>
      requirement.status === 'open' && active.has(requirement.parentNodeId)
    ).map(requirement => requirement.id));
    const phase = topologySamplingPhase(snapshot);
    if (phase.id === 'seed_child_local_residual') {
      return candidates.some(candidate => topologySamplingCandidateMatchesPhase(snapshot, candidate))
        ? [] : ['missing_deepest_child_progress_candidate'];
    }
    if (phase.id === 'derive_child_local_residual') {
      return candidates.some(candidate => topologySamplingCandidateMatchesPhase(snapshot, candidate))
        ? [] : ['missing_child_local_recursive_derive_candidate'];
    }
    const requiredAssignments = snapshot.runtime.nodes.length < minimumNodes
      ? Math.min(3, minimumNodes - snapshot.runtime.nodes.length, openGapIds.size) : 0;
    const offeredGapIds = new Set(candidates.filter(candidate => candidate.kind === 'DERIVE'
      || (candidate.kind === 'CONNECT' && candidate.action.requirementId))
      .map(candidate => candidate.kind === 'DERIVE'
        ? candidate.action.childSpecification?.triggeringGapId
        : candidate.action.requirementId)
      .filter((gapId): gapId is string => typeof gapId === 'string'
        && openGapIds.has(gapId)));
    const deficits: string[] = [];
    if (offeredGapIds.size < requiredAssignments) {
      deficits.push(`missing_real_gap_assignment_candidates:${offeredGapIds.size}/${requiredAssignments}`);
    }
    if (profile.id === 'connected' && snapshot.runtime.nodes.length >= minimumNodes
      && snapshot.runtime.communicationEdges.every(edge => !edge.active)) {
      const activeIds = [...active];
      const existing = new Set(snapshot.runtime.communicationEdges.filter(edge => edge.active)
        .map(edge => `${edge.from}\u0000${edge.to}`));
      const hasAvailableConnection = activeIds.some(from => activeIds.some(to => from !== to
        && !existing.has(`${from}\u0000${to}`)));
      const offersNovelConnection = candidates.some(candidate => {
        if (candidate.kind !== 'CONNECT') return false;
        const connection = candidate.action.connection;
        return Boolean(connection && connection.from !== connection.to
          && active.has(connection.from) && active.has(connection.to)
          && !existing.has(`${connection.from}\u0000${connection.to}`));
      });
      if (hasAvailableConnection && !offersNovelConnection) {
        deficits.push('missing_novel_connect_candidate');
      }
    }
    return deficits;
  }

  private policyState(snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[]): Record<string, unknown> {
    const activeNodes = snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status));
    const latest = snapshot.processStates.at(-1);
    const events = latest?.runtimeEvents ?? [];
    const agentNodes = snapshot.runtime.nodes.map(node => ({ id: node.id, kind: 'agent',
      timestamp: node.createdAt, text: node.localObjective, status: node.status,
      attributes: { signal: node.status === 'returned' || node.status === 'completed' ? 1 : 0 } }));
    const epistemicNodes = [
      ...(latest?.requirements ?? []).map(value => ({ id: value.id, kind: 'subtask',
        timestamp: 0, text: value.description, status: value.status,
        attributes: { signal: value.status === 'resolved' ? 1 : 0 } })),
      ...(latest?.claims ?? []).map(value => ({ id: value.id, kind: 'child_result',
        timestamp: 0, text: value.statement, status: value.status,
        attributes: { signal: value.status === 'supported' ? 1 : 0 } })),
      ...(latest?.assumptions ?? []).map((value, index) => ({
        id: String(value.id ?? `assumption-${index}`), kind: 'message', timestamp: 0,
        text: String(value.statement ?? ''), status: String(value.status ?? 'unverified'),
        attributes: { signal: value.status === 'verified' ? 1 : 0 } })),
      ...(latest?.evidence ?? []).map(value => ({ id: value.id, kind: 'artifact',
        timestamp: 0, text: value.content, status: 'observed', attributes: { signal: 1 } })),
      ...(latest?.externalObservations ?? []).map(value => ({ id: value.id,
        kind: 'tool_result', timestamp: 0, text: value.observation, status: 'observed',
        attributes: { signal: 1 } })),
      ...(latest?.blindSpots ?? []).map((value, index) => ({ id: `blind-spot:${index}`,
        kind: 'message', timestamp: 0, text: value, status: 'open',
        attributes: { signal: -1 } })),
    ];
    const runtimeNodes = events.map((event, index) => {
      const action = event.attributes?.action as Record<string, unknown> | undefined;
      const text = [event.kind, action?.kind, event.command, projectText(event.output)]
        .filter(value => value !== undefined && value !== '').join(' ');
      const failed = event.kind === 'failure'
        || (event.kind === 'terminal_result' && (event.exitCode ?? 0) !== 0);
      const kind = event.kind === 'terminal_command' ? 'tool_call'
        : ['terminal_result', 'failure', 'verifier'].includes(event.kind) ? 'tool_result'
          : event.kind === 'file_change' ? 'artifact'
            : event.kind === 'usage' ? 'resource'
              : event.kind === 'task_instruction' ? 'subtask' : 'message';
      return { id: `runtime:${event.id ?? index}`, kind, timestamp: event.at,
        text, status: failed ? 'failed' : 'observed',
        attributes: { signal: failed ? -1
          : ['terminal_result', 'file_change', 'verifier'].includes(event.kind) ? 1 : 0 } };
    });
    const usage = latest?.usage ?? { inputTokens: 0, outputTokens: 0, wallTimeMs: 0 };
    const metricValues: Record<string, number> = {
      node_count: snapshot.runtime.nodes.length,
      edge_count: snapshot.runtime.derivationEdges.length + snapshot.runtime.dependencyEdges.length
        + snapshot.runtime.communicationEdges.length,
      maximum_depth: Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth)),
      active_node_count: activeNodes.length,
      blind_spot_count: latest?.blindSpots.length ?? 0,
      input_tokens: usage.inputTokens,
      output_tokens: usage.outputTokens,
      wall_time_ms: usage.wallTimeMs,
    };
    const stateNodes = [
      { id: 'state:active-subtree', kind: 'resource', timestamp: 0,
        text: `active subtree ${(latest?.activeSubtree ?? []).join(' ')}`, status: 'observed',
        attributes: { signal: Math.log1p(latest?.activeSubtree.length ?? 0) } },
      ...Object.entries(metricValues).map(([name, value]) => ({ id: `metric:${name}`,
        kind: 'resource', timestamp: 0, text: `${name} ${value}`, status: 'observed',
        attributes: { signal: Math.log1p(Math.max(0, value)) } })),
    ];
    const seenNodeIds = new Set<string>();
    const nodes = [...agentNodes, ...epistemicNodes, ...runtimeNodes, ...stateNodes].filter(node => {
      if (seenNodeIds.has(node.id)) return false;
      seenNodeIds.add(node.id);
      return true;
    });
    const edges = [
      ...snapshot.runtime.derivationEdges.map((edge, index) => ({ id: `d-${index}`,
        kind: 'derivation', from: edge.parentId, to: edge.childId })),
      ...snapshot.runtime.dependencyEdges.map((edge, index) => ({ id: `p-${index}`,
        kind: 'dependency', from: edge.producerId, to: edge.consumerId })),
      ...snapshot.runtime.communicationEdges.map((edge, index) => ({ id: `c-${index}`,
        kind: 'communication', from: edge.from, to: edge.to })),
      ...(latest?.requirements ?? []).map((value, index) => ({ id: `r-${index}`,
        kind: 'dependency', from: value.parentNodeId, to: value.id })),
      ...(latest?.claims ?? []).map((value, index) => ({ id: `cl-${index}`,
        kind: 'produces', from: value.originNodeId, to: value.id })),
      ...(latest?.evidence ?? []).flatMap((value, evidenceIndex) => value.supports.map(
        (claimId, supportIndex) => ({ id: `e-${evidenceIndex}-${supportIndex}`,
          kind: 'produces', from: value.id, to: claimId }))),
      ...(latest?.externalObservations ?? []).flatMap((value, observationIndex) =>
        value.supports.map((claimId, supportIndex) => ({
          id: `o-${observationIndex}-${supportIndex}`, kind: 'produces',
          from: value.id, to: claimId }))),
      ...runtimeNodes.flatMap((value, index) => {
        const event = events[index];
        const values: Array<{ id: string; kind: string; from: string; to: string }> = [];
        if (event?.nodeId) values.push({ id: `runtime-owner-${index}`,
          kind: ['terminal_command', 'terminal_result'].includes(event.kind)
            ? 'tool_use' : 'produces', from: event.nodeId, to: value.id });
        if (index > 0) values.push({ id: `runtime-temporal-${index}`,
          kind: 'temporal', from: runtimeNodes[index - 1].id, to: value.id });
        return values;
      }),
      ...(latest?.activeSubtree ?? []).map((nodeId, index) => ({ id: `active-${index}`,
        kind: 'consumes', from: nodeId, to: 'state:active-subtree' })),
    ];
    const openRequirements = snapshot.runtime.requirements.filter(value => value.status === 'open');
    const profile = resolveTopologySamplingProfile(snapshot.organizationSeed);
    const minimumNodes = profile.preferredNodeRange[0];
    const minimumDepth = profile.preferredMinimumDepth;
    const maximumDepthReached = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
    const explorationStopMasked = openRequirements.length > 0
      && (snapshot.runtime.nodes.length < minimumNodes || maximumDepthReached < minimumDepth);
    const policyCandidates = candidates.map(candidate => ({ id: candidate.id, kind: candidate.kind,
      actor_node_id: candidate.actorNodeId, description: candidate.description,
      scheduler_complexity: candidate.schedulerComplexity,
      external_access: candidate.kind === 'ACQUIRE',
      resolves_gap: candidate.kind === 'ACQUIRE' || candidate.kind === 'RETURN',
      depth_delta: candidate.kind === 'DERIVE' ? 1 : 0,
      sampling_logit_bias: topologySamplingCandidateLogitBias(snapshot, candidate),
      legal: !(candidate.kind === 'STOP' && explorationStopMasked) }));
    return {
      interface_revision: 'compact-epistemic-event-driven-v4',
      sampling_profile: { id: profile.id, preferred_node_range: profile.preferredNodeRange,
        preferred_minimum_depth: profile.preferredMinimumDepth, focus: profile.focus,
        reward_semantics: 'coverage intervention only; topology has no intrinsic reward' },
      state_fingerprint: snapshot.processStates.at(-1)?.fingerprint,
      event_graph: { nodes, edges },
      active_node_ids: activeNodes.map(node => node.id),
      active_node_legal: activeNodes.map(node => candidates.some(
        candidate => candidate.actorNodeId === node.id
      )),
      active_node_sampling_logit_bias: activeNodes.map(node =>
        topologySamplingActiveNodeLogitBias(snapshot, node.id)),
      candidates: policyCandidates,
      envelope: { id: 'lhtb-open', minimum_nodes: minimumNodes, maximum_nodes: 1_000_000,
        minimum_depth: minimumDepth, maximum_depth: 1_000_000, mode: 'expansive' },
      node_count: snapshot.runtime.nodes.length,
      maximum_depth_reached: maximumDepthReached,
      unresolved_gap_exists: openRequirements.length > 0,
      unresolved_requirement_ids: openRequirements.map(value => value.id),
      num_real_residual_gaps: openRequirements.length,
      num_child_proposals: candidates.filter(candidate => candidate.kind === 'DERIVE').length,
      available_actions: [...new Set(candidates.map(candidate => candidate.kind))],
      topology_sampling_phase: topologySamplingPhase(snapshot).id,
      exploration_stop_masked: explorationStopMasked,
      stop_legal_reason: explorationStopMasked
        ? 'masked_during_early_exploration_with_real_open_residual_gap'
        : openRequirements.length === 0 ? 'no_real_open_residual_gap'
          : 'exploration_minimum_satisfied',
      terminal_result_count: events.filter(event => event.kind === 'terminal_result').length,
      organization_action_count: events.filter(event => event.kind === 'organization_action').length,
      contradiction_count: snapshot.processStates.at(-1)?.semanticRelations.filter(value =>
        value.label === 'contradict').length ?? 0,
      completed_node_count: snapshot.runtime.nodes.filter(node =>
        ['completed', 'returned'].includes(node.status)).length,
      resources: { llm_calls_remaining_fraction: 1, tool_calls_remaining_fraction: 1,
        nodes_remaining_fraction: 1, depth_remaining_fraction: 1,
        decisions_remaining_fraction: 1 },
      organization_temperature: Number(process.env.ROY_LHTB_ORGANIZATION_TEMPERATURE ?? 1),
      unbounded_structure: true,
    };
  }

  private async auditProposal(request: Record<string, unknown>,
    completion: LLMJSONCompletionResult<ProposalResponse>): Promise<void> {
    const root = this.auditRoot;
    if (!root && this.provider.constructor.name !== 'DeepSeekProvider') return;
    if (!root) throw new Error('ROY_LHTB_AUDIT_ROOT is required');
    await mkdir(root, { recursive: true });
    await appendFile(path.join(root, 'organization-proposals.jsonl'), `${JSON.stringify({
      schemaVersion: 1, provider: 'deepseek', model: completion.completion.model,
      configuredRevision: process.env.DEEPSEEK_MODEL_REVISION ?? 'api-alias-unversioned',
      temperature: 0, maxTokens: 32_768, request,
      response: completion.value, usage: completion.completion.usage,
    })}\n`, 'utf8');
  }

  private async auditCandidateValidation(validation: CandidateValidation): Promise<void> {
    const root = this.auditRoot;
    if (!root && this.provider.constructor.name !== 'DeepSeekProvider') return;
    if (!root) throw new Error('ROY_LHTB_AUDIT_ROOT is required');
    await mkdir(root, { recursive: true });
    await appendFile(path.join(root, 'candidate-validation.jsonl'), `${JSON.stringify({
      schemaVersion: 1,
      acceptedCandidateIds: validation.candidates.map(value => value.id),
      dispositions: validation.dispositions,
    })}\n`, 'utf8');
  }

  private async auditProposalFailure(request: Record<string, unknown>, error: LLMJSONParseError,
    attempt: number): Promise<void> {
    const root = this.auditRoot;
    if (!root) throw new Error('ROY_LHTB_AUDIT_ROOT is required');
    await mkdir(root, { recursive: true });
    await appendFile(path.join(root, 'proposal-failures.jsonl'), `${JSON.stringify({
      schemaVersion: 1, provider: 'deepseek', model: error.completion.model,
      configuredRevision: process.env.DEEPSEEK_MODEL_REVISION ?? 'api-alias-unversioned',
      temperature: 0, maxTokens: 32_768, attempt, request,
      responseContent: error.completion.content, usage: error.completion.usage,
      error: error.message,
    })}\n`, 'utf8');
  }
}
