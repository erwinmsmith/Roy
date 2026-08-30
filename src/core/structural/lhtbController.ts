import { DeepSeekProvider, LLMJSONParseError } from '../llm/providers/openai.js';
import { LHTB_POLICY_INTERFACE_REVISION } from './informationRealizationTypes.js';
import type { OrganizationCandidate,
  OrganizationPolicyRecord, StructuralControllerActionKind,
  StructuralControllerCandidate } from './informationRealizationTypes.js';
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
  report?: unknown;
}

interface ProposalResponse {
  preferred_candidate_id: string;
  candidates: RawProposedCandidate[];
}

interface CandidateValidation {
  candidates: ProposedCandidate[];
  dispositions: Array<{ index: number; id?: string; accepted: boolean; reasons: string[] }>;
}

interface MCTSAgentProposalResult {
  candidates: ProposedCandidate[];
  calls: number;
  inputTokens: number;
  outputTokens: number;
  models: string[];
  attempts: number;
}

const CONTROLLER_ACTION_DESCRIPTIONS: Record<StructuralControllerActionKind, string> = {
  CONTINUE: 'Continue this node semantic work without spawning a new agent',
  DERIVE_INFO: 'Derive one child to acquire missing task-relevant information',
  DERIVE_ORG: 'Derive one child to organize, verify, or integrate available information',
  PRUNE: 'Prune one low-value branch selected by the frozen Worker',
  RETURN: 'Return this node result to its parent',
  FINISH: 'Finish the root task and submit to the official verifier',
};

function controllerActionForCandidate(
  candidate: ProposedCandidate
): StructuralControllerActionKind {
  if (candidate.kind === 'DERIVE') {
    return candidate.action.childSpecification?.realizationMode === 'acquire_external'
      ? 'DERIVE_INFO' : 'DERIVE_ORG';
  }
  if (candidate.kind === 'PRUNE' || candidate.kind === 'RETURN') return candidate.kind;
  if (candidate.kind === 'STOP') return 'FINISH';
  // ACQUIRE, EXECUTE, and CONNECT are frozen-Worker/Runtime payloads.  They are
  // all semantic continuation from the learned Controller's perspective.
  return 'CONTINUE';
}

export interface TopologySamplingProfile {
  id: 'single' | 'compact' | 'branching' | 'recursive' | 'connected';
  preferredNodeRange: [number, number];
  preferredMinimumDepth: number;
  focus: string;
}

export function topologySamplingProfile(organizationSeed: number): TopologySamplingProfile {
  const profiles: TopologySamplingProfile[] = [
    { id: 'single', preferredNodeRange: [1, 1], preferredMinimumDepth: 0,
      focus: 'single root agent only; solve or stop without derivation or communication' },
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
  const ids: TopologySamplingProfile['id'][] = [
    'single', 'compact', 'branching', 'recursive', 'connected',
  ];
  const index = ids.indexOf(override as TopologySamplingProfile['id']);
  if (index < 0) throw new Error(`Invalid ROY_LHTB_TOPOLOGY_PROFILE ${override}`);
  return topologySamplingProfile(index);
}

/**
 * Topology profiles are retained only for explicit controlled diagnostics.
 * Formal learned-policy sampling leaves this unset so node count, depth and
 * graph connectivity emerge from repeated node-local actor decisions rather
 * than seed-conditioned inputs.
 */
export function activeTopologySamplingProfile(
  organizationSeed: number,
  override = process.env.ROY_LHTB_TOPOLOGY_PROFILE
): TopologySamplingProfile | undefined {
  return override ? resolveTopologySamplingProfile(organizationSeed, override) : undefined;
}

interface SamplingPhase {
  id: 'unconstrained' | 'expand_width' | 'seed_child_local_residual'
    | 'derive_child_local_residual' | 'stabilize';
  deepestActiveNodeIds: Set<string>;
  childLocalOpenGapIds: Set<string>;
}

const SCHEDULER_RUNNABLE_STATUSES = new Set(['ready', 'running', 'completed']);

type LHTBSessionSnapshot = ReturnType<RoyLHTBSession['snapshot']>;
type RuntimeRequirement = LHTBSessionSnapshot['runtime']['requirements'][number];

/**
 * Resolve the node that currently owns an unresolved requirement. DERIVE
 * transfers execution ownership without changing the requirement's original
 * provenance parent. Older checkpoints did not persist assignedNodeId, so the
 * active triggering-gap lineage is used only as a backwards-compatible repair.
 */
function effectiveRequirementOwner(
  snapshot: LHTBSessionSnapshot,
  requirement: RuntimeRequirement,
): string {
  if (requirement.status === 'assigned' && requirement.assignedNodeId) {
    return requirement.assignedNodeId;
  }
  if (requirement.status === 'assigned') {
    const inferred = snapshot.runtime.nodes.filter(node =>
      node.triggeringGapId === requirement.id
      && SCHEDULER_RUNNABLE_STATUSES.has(node.status))
      .sort((left, right) => right.depth - left.depth || right.updatedAt - left.updatedAt)[0];
    if (inferred) return inferred.id;
  }
  return requirement.parentNodeId;
}

function isDelegableRequirement(requirement: RuntimeRequirement): boolean {
  return requirement.status === 'open' || requirement.status === 'assigned';
}

/** Deterministic execution ownership; never part of the learned action. */
export function scheduledOrganizationContextNode(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>
): string {
  const nodes = new Map(snapshot.runtime.nodes.map(node => [node.id, node]));
  const runnable = (id: unknown): id is string => typeof id === 'string'
    && SCHEDULER_RUNNABLE_STATUSES.has(nodes.get(id)?.status ?? '');
  const events = snapshot.runtimeEvents ?? snapshot.processStates.at(-1)?.runtimeEvents ?? [];
  for (const event of [...events].reverse()) {
    if (event.kind === 'terminal_result') {
      const actor = typeof event.nodeId === 'string' ? nodes.get(event.nodeId) : undefined;
      const parentId = actor?.parentId;
      const parentHasUnassignedLocalWork = runnable(parentId)
        && snapshot.runtime.requirements.some(requirement =>
          requirement.parentNodeId === parentId && requirement.status === 'open');
      // A child gets at least one local action after DERIVE.  Once that action
      // produces an observation, a parent with other unassigned work gets one
      // turn to reuse/connect the live child or derive a distinct worker.
      if (parentHasUnassignedLocalWork && parentId) return parentId;
      if (runnable(event.nodeId)) return event.nodeId;
      continue;
    }
    if (event.kind === 'terminal_command') {
      if (runnable(event.nodeId)) return event.nodeId;
      continue;
    }
    if (event.kind !== 'organization_action') continue;
    const action = event.attributes?.action as Record<string, unknown> | undefined;
    if (!action) continue;
    if (action.kind === 'DERIVE') {
      const child = action.childSpecification as Record<string, unknown> | undefined;
      if (runnable(child?.nodeId)) return child.nodeId;
      // A derivation transfers execution ownership to the newly created child.
      // The parent may still own other open requirements, but it must wait for
      // this child to act or return before another parent-local decision.  This
      // makes every root -> child -> grandchild transition an observed,
      // node-local policy decision rather than a batch of inert spawn records.
      if (runnable(action.actorNodeId)) return action.actorNodeId;
    }
    if (action.kind === 'RETURN') {
      const actor = typeof action.actorNodeId === 'string' ? nodes.get(action.actorNodeId) : undefined;
      if (runnable(actor?.parentId)) return actor.parentId;
    }
    if (action.kind === 'CONNECT' && action.requirementId) {
      const connection = action.connection as Record<string, unknown> | undefined;
      if (runnable(connection?.to)) return connection.to;
    }
    if (runnable(action.actorNodeId)) return action.actorNodeId;
  }
  const ready = snapshot.runtime.nodes.filter(node => SCHEDULER_RUNNABLE_STATUSES.has(node.status));
  if (!ready.length) throw new Error('environment_invalid:no_scheduler_runnable_node');
  return [...ready].sort((left, right) => {
    const running = Number(right.status === 'running') - Number(left.status === 'running');
    return running || right.updatedAt - left.updatedAt || right.depth - left.depth
      || left.id.localeCompare(right.id);
  })[0].id;
}

export type TopologySamplingCandidate = Pick<OrganizationCandidate,
  'kind' | 'actorNodeId' | 'action'>;

function topologySamplingPhase(
  snapshot: ReturnType<RoyLHTBSession['snapshot']>
): SamplingPhase {
  const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
  const activeNodes = snapshot.runtime.nodes.filter(node =>
    ['ready', 'running', 'waiting', 'completed'].includes(node.status));
  const maximumDepth = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
  const deepestActiveNodeIds = new Set(activeNodes.filter(node => node.depth === maximumDepth)
    .map(node => node.id));
  const childLocalOpenGapIds = new Set(snapshot.runtime.requirements.filter(requirement =>
    requirement.status === 'open' && deepestActiveNodeIds.has(requirement.parentNodeId)
    && requirement.parentNodeId !== snapshot.runtime.rootId).map(requirement => requirement.id));
  if (!profile) return { id: 'unconstrained', deepestActiveNodeIds, childLocalOpenGapIds };
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
  if (phase.id === 'unconstrained') return true;
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
  const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
  if (!profile) return 0;
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

export type ControllerResult = (
  | { status: 'terminal_request'; request: { id: string; command: string; cwd?: string;
    timeoutMs: number; nodeId: string }; snapshot: ReturnType<RoyLHTBSession['snapshot']> }
  | { status: 'continue'; snapshot: ReturnType<RoyLHTBSession['snapshot']> }
  | { status: 'completed'; snapshot: ReturnType<RoyLHTBSession['snapshot']> }
) & { controllerActionKind?: StructuralControllerActionKind };

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

const PROPOSER_PROMPT = `You are Roy's frozen semantic Worker for LHTB.
You materialize legal semantic payloads; you are not the trainable structural Controller.
Return JSON only: {"preferred_candidate_id": string, "candidates": [...]}.
Every candidate must have this exact outer shape:
{"id": string, "kind": string, "actorNodeId": string, "description": string,
 "schedulerComplexity": number, "action": object}.
Allowed kinds are DERIVE, ACQUIRE, CONNECT, EXECUTE, RETURN, PRUNE, STOP.
These are Runtime payload kinds, not the learned action vocabulary. The shared Controller sees only
CONTINUE, DERIVE_INFO, DERIVE_ORG, PRUNE, RETURN, FINISH. Runtime ACQUIRE/EXECUTE/CONNECT payloads
all realize CONTINUE; DERIVE realizationMode maps to DERIVE_INFO or DERIVE_ORG; STOP maps to FINISH.
The Controller never sees or chooses your command, child description, report, connection, or target.
The Runtime deterministically supplies organization.schedulerContextNode. Generate candidates only
for that exact node. actorNodeId is observed execution context, never a routing choice. ACQUIRE and
EXECUTE are local actions by that node; DERIVE is the only action that spawns a new child.
For ACQUIRE and EXECUTE, command, optional cwd and timeoutMs are outer candidate fields, while
action contains only kind and actorNodeId. ACQUIRE inspects external state. EXECUTE changes the
workspace or runs verification. Keep every command under 50000 characters. Prefer short incremental
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
"realizationMode":"acquire_external|organize_knowledge",
"localObjective":"<strictly narrower executable objective>","refinement":{"parentScope":"...",
"childScope":"...","triggeringRequirementId":"<same requirement id>",
"narrowerThanParent":true,"newInformationNeeded":"...","executableEndCondition":"...",
"duplicatedByExistingNode":false},"requiredClaims":[],"requiredEvidence":[],
"relevantReportIds":[],"externalAccess":{"allowed":true,"tools":["terminal"],
"purpose":"..."},"expectedOutput":{"requiredInformation":"...",
"outputType":"epistemic_report"},"terminationCondition":"...",
"reuseReview":{"searchedNodeIds":["<every spawned agent id>"],
"decision":"spawn_distinct","reason":"why no existing spawned agent can perform this gap"}}}.
For DERIVE, the selected requirement's parentNodeId is its owner: candidate actorNodeId,
action.actorNodeId, childSpecification.parentId, and refinement parent scope must all remain under
that owner. Never assign a root-owned gap to a child merely because the child's objective is
semantically related. structuralExploration.realOpenGaps is the authoritative gap-to-owner map.
Use realizationMode=acquire_external when the child is spawned to obtain missing external evidence,
and realizationMode=organize_knowledge when it is spawned to reason over, verify, or integrate
information already available to Roy. This conditional child type is selected by the policy as part
of the open DERIVE specification; do not encode it as an Agent routing decision.
Keep child specifications concise. Never copy rootGoal or the full task instruction into parentGoal,
parentScope, childScope, or reason. Keep each descriptive child field below 1000 characters.
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
When independent unresolved requirements exist, include useful legal DERIVE payloads alongside
non-structural alternatives so the Controller has a valid categorical structural action mask.
Never target a node count or depth. This is candidate support, not permission to invent work:
never create a fake gap, duplicate an existing node ID or objective, reuse an assigned gap, or
derive non-refining work. A requirement extracted from a child event belongs
to that child unless its provenance explicitly identifies another active parent.
CONNECT action is {"kind":"CONNECT","actorNodeId":"<actor>","connection":{"from":"<existing>",
"to":"<different existing>","required":false}}. Once at least three nodes exist, include a novel
CONNECT candidate when a real information handoff would help; never repeat an active edge.
When a new child genuinely needs a report from an existing producer, childSpecification.dependencies
may use {"producerNodeId":"<existing>","artifactId":"report:<existing>"}; never fabricate a
dependency merely to make a DAG. PRUNE uses targetNodeId for a non-root node.
Formal sampling uses no MCTS or look-ahead search: node count, derivation depth and graph
connectivity must be outcomes of repeated real node-local decisions. Do not use a requested
topology size, minimum depth, profile identity or
complexity target. Progress one legal organization action at a time. A diagnostic profile may be
present only in explicitly controlled tests; it is never terminal utility or a semantic reason to
derive. Never relabel a root requirement as child-local to manufacture recursive depth.
RETURN is child-only and transfers a report to its parent; root must use STOP. RETURN action uses
the property report (never epistemicReport), with this exact report shape:
{"id":"...","nodeId":"<actor>","parentId":"...","depth":<actor-depth>,"localObjective":"...",
"triggeringGapId":"...","conclusion":"...","reasoningSummary":"...","claims":[],
"evidence":[],"externalObservations":[],"assumptions":[],"uncertainty":{"confidence":0.5,
"uncertainAbout":[],"confidenceBasis":"..."},"conflicts":[],"coverage":{"resolved":[],
"unresolved":[],"notExamined":[]},"blindSpots":[],"residualRequirements":[],
"proposedChildren":[],"resolvedParentGap":false,"informationToPropagate":[]}.
Claims, evidence, observations, assumptions, conflicts, residual requirements and proposed children
must use their typed runtime schemas when non-empty. Use an empty array instead of a placeholder,
partial object, null, or undefined item. In particular, residualRequirements should normally be [];
every non-empty entry requires id, description, whyItMatters, likelyMechanism,
requiredInformation, status, and parentNodeId equal to the returning actor. STOP is root-only and
uses finalOutput.
Every candidate actorNodeId must equal organization.schedulerContextNode. Other active nodes are
visible only for dependency and CONNECT/reuse decisions; never propose a turn for them.
Never repeat an unchanged terminal command immediately after it failed without file changes;
propose a command that diagnoses or repairs the observed failure.
Always leave root STOP in the candidate support when required report dependencies permit it; the
runtime also supplies a deterministic STOP candidate if it is omitted. Stopping early is legal and
may receive a poor official verifier score. When requested artifacts exist and the task's own
end-to-end command succeeds, include a legal RETURN for a child or STOP for root alongside at most
one materially useful repair or verification action; record remaining uncertainty instead of
creating an endless verification loop. More generally, after any successful local terminal result,
a non-root actor must include one legal RETURN candidate with its evidence-grounded report. This
only supplies a legal semantic payload: the shared Controller decides RETURN versus CONTINUE.
Propose only semantically useful payloads. Do not expose hidden reasoning, benchmark grader data,
keyword fields or reward. preferred_candidate_id selects your deterministic payload within a
Controller category; it is not a learned structural decision.`;

const PROPOSER_RECENT_EVENT_COUNT = 6;
const PROPOSER_EVENT_TEXT_LIMIT = 2_000;
const PROPOSER_ENTITY_TEXT_LIMIT = 600;
const PROPOSER_ENTITY_COUNT = 16;
const PROPOSER_GOAL_TEXT_LIMIT = 6_000;
const SEMANTIC_RECALL_ENTITY_COUNT = 32;
const PROPOSER_COMMAND_LIMIT = 50_000;
const PROPOSER_RESPONSE_MAX_TOKENS = 32_768;

/**
 * Repair only JSON delimiter structure. This never invents fields, values, actions, or semantics;
 * the repaired object must still pass the normal candidate and Runtime legality checks.
 */
export function repairProposalJSONStructure(content: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(content) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown> : undefined;
  } catch {
    // Continue with a delimiter-only repair.
  }
  const stack: Array<'{' | '['> = [];
  let repaired = '';
  let inString = false;
  let escaped = false;
  const nextNonWhitespace = (start: number): string | undefined => {
    for (let index = start; index < content.length; index += 1) {
      if (!/\s/.test(content[index])) return content[index];
    }
    return undefined;
  };
  for (let index = 0; index < content.length; index += 1) {
    const character = content[index];
    if (inString) {
      repaired += character;
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') {
      inString = true;
      repaired += character;
      continue;
    }
    if (character === '{' || character === '[') {
      stack.push(character);
      repaired += character;
      continue;
    }
    if (character === '}' || character === ']') {
      const expected = character === '}' ? '{' : '[';
      if (stack.at(-1) === expected) {
        stack.pop();
        repaired += character;
      } else if (character === ']' && stack.includes('[')) {
        while (stack.at(-1) === '{') {
          stack.pop();
          repaired += '}';
        }
        if (stack.at(-1) === '[') {
          stack.pop();
          repaired += ']';
        }
      }
      continue;
    }
    if (character === ',' && nextNonWhitespace(index + 1) === '{'
      && stack.at(-1) === '{' && stack.includes('[')) {
      while (stack.at(-1) === '{') {
        stack.pop();
        repaired += '}';
      }
    }
    repaired += character;
  }
  if (inString) return undefined;
  while (stack.length) repaired += stack.pop() === '{' ? '}' : ']';
  try {
    const parsed = JSON.parse(repaired) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
    const value = parsed as Record<string, unknown>;
    return Array.isArray(value.candidates) ? value : undefined;
  } catch {
    return undefined;
  }
}

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

function compactChildSpecification(value: unknown): unknown {
  if (typeof value === 'string') return compactText(value, 1_000);
  if (Array.isArray(value)) return value.map(compactChildSpecification);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .map(([key, child]) => [key, compactChildSpecification(child)]));
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
  const recentRuntimeEvents = (snapshot.runtimeEvents ?? latest?.runtimeEvents ?? [])
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
  const openRequirements = snapshot.runtime.requirements
    .filter(value => isDelegableRequirement(value)
      && activeNodes.some(node => node.id === effectiveRequirementOwner(snapshot, value)))
    .map(value => ({ ...value, parentNodeId: effectiveRequirementOwner(snapshot, value) }));
  const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
  const maximumDepthReached = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
  const schedulerContextNode = scheduledOrganizationContextNode(snapshot);
  return {
    rootGoal: compactText(snapshot.instruction, PROPOSER_GOAL_TEXT_LIMIT),
    organizationMode: snapshot.organizationMode,
    organization: {
      rootId: snapshot.runtime.rootId, stopped: snapshot.runtime.stopped,
      schedulerContextNode,
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
      searchMode: profile ? 'profile_conditioned_diagnostic' : 'direct_node_actor_on_policy',
      currentNodeCount: snapshot.runtime.nodes.length,
      currentMaximumDepth: maximumDepthReached,
      samplingProfile: profile?.id,
      minimumNodeTarget: profile?.preferredNodeRange[0] ?? 0,
      preferredTopologyRange: profile?.preferredNodeRange,
      minimumDepthTarget: profile?.preferredMinimumDepth ?? 0,
      focus: profile?.focus ?? 'topology emerges from repeated real node-local Controller actions',
      realOpenGapCount: openRequirements.length,
      realOpenGapIds: openRequirements.map(value => value.id),
      realOpenGaps: openRequirements.map(value => ({ id: value.id,
        parentNodeId: value.parentNodeId, description: compactText(value.description),
        requiredInformation: compactText(value.requiredInformation) })),
      desiredAdditionalChildren: profile ? Math.min(3, Math.max(0,
        profile.preferredNodeRange[0] - snapshot.runtime.nodes.length), openRequirements.length) : 0,
      preferredMaximumNodes: profile?.preferredNodeRange[1],
      requiredNextStructuralPhase: topologySamplingPhase(snapshot).id,
      deepestActiveNodeIds: [...topologySamplingPhase(snapshot).deepestActiveNodeIds],
      childLocalOpenGapIds: [...topologySamplingPhase(snapshot).childLocalOpenGapIds],
      semantics: profile
        ? 'explicit diagnostic intervention only; never synthesize gaps or add reward'
        : 'no search; topology is an on-policy trajectory outcome',
    },
    projection: { recentRuntimeEventCount: PROPOSER_RECENT_EVENT_COUNT,
      eventTextLimit: PROPOSER_EVENT_TEXT_LIMIT, entityTextLimit: PROPOSER_ENTITY_TEXT_LIMIT,
      entityCount: PROPOSER_ENTITY_COUNT, immutableRawLedgerPreserved: true,
      retrieval: 'Use raw-event/report/observation references only when more detail is needed' },
  };
}

export function compactProposalRepairRequest(
  requestState: Record<string, unknown>,
  reasons: string[],
  rejectedCandidates: Array<{ candidate: RawProposedCandidate; reasons: string[] }>
): string {
  const organization = requestState.organization && typeof requestState.organization === 'object'
    ? requestState.organization as Record<string, unknown> : {};
  const exploration = requestState.structuralExploration
    && typeof requestState.structuralExploration === 'object'
    ? requestState.structuralExploration as Record<string, unknown> : {};
  const epistemic = requestState.compactEpistemicState
    && typeof requestState.compactEpistemicState === 'object'
    ? requestState.compactEpistemicState as Record<string, unknown> : {};
  const requiredExternalChildProgressNodeIds = reasons.filter(reason =>
    reason.startsWith('missing_external_child_progress_candidate:'))
    .map(reason => reason.slice('missing_external_child_progress_candidate:'.length));
  const requiredChildReturnNodeIds = reasons.filter(reason =>
    reason.startsWith('missing_child_return_candidate:'))
    .map(reason => reason.slice('missing_child_return_candidate:'.length));
  const requiredPostVerifierProgressNodeIds = reasons.filter(reason =>
    reason.startsWith('missing_post_verifier_progress_candidate:'))
    .map(reason => reason.slice('missing_post_verifier_progress_candidate:'.length));
  return JSON.stringify({
    repairProtocol: 'legal-candidate-interface-v3',
    instruction: [
      'Return a fresh concise candidate set that passes this exact legal interface.',
      'For DERIVE, use one realOpenGaps entry and set candidate actorNodeId, action.actorNodeId, and childSpecification.parentId to that entry parentNodeId.',
      'Do not reproduce an unchanged rejected candidate, move a gap to a semantically related child, invent a gap, or repeat an active CONNECT edge.',
      'Topology emerges from repeated direct Controller actions; do not target a node count or depth or fabricate work.',
      'For every requiredExternalChildProgressNodeId, include at least one ACQUIRE or EXECUTE candidate whose actorNodeId is that exact node. A RETURN is not a substitute.',
      'For every requiredChildReturnNodeId, include one legal RETURN candidate with an evidence-grounded report for that exact node. Other useful actions may remain as alternatives.',
      'After an official verifier rejection, every requiredPostVerifierProgressNodeId must receive a new ACQUIRE or EXECUTE candidate before STOP can be legal again.',
    ],
    rejectionReasons: reasons,
    rejectedCandidates,
    taskGoal: requestState.rootGoal,
    organizationMode: requestState.organizationMode,
    legalInterface: {
      rootId: organization.rootId,
      activeNodes: organization.activeNodes,
      spawnedAgentLibrary: organization.spawnedAgentLibrary,
      realOpenGaps: exploration.realOpenGaps,
      searchMode: exploration.searchMode,
      samplingProfile: exploration.samplingProfile,
      requiredNextStructuralPhase: exploration.requiredNextStructuralPhase,
      deepestActiveNodeIds: exploration.deepestActiveNodeIds,
      childLocalOpenGapIds: exploration.childLocalOpenGapIds,
      desiredAdditionalChildren: exploration.desiredAdditionalChildren,
      preferredTopologyRange: exploration.preferredTopologyRange,
      requiredExternalChildProgressNodeIds,
      requiredChildReturnNodeIds,
      requiredPostVerifierProgressNodeIds,
      returnCandidateSchema: {
        id: '<unique-return-candidate-id>', kind: 'RETURN',
        actorNodeId: '<requiredChildReturnNodeId>',
        description: '<concise evidence-grounded handoff>', schedulerComplexity: 1,
        action: { kind: 'RETURN', actorNodeId: '<same requiredChildReturnNodeId>',
          report: {
            id: '<unique-report-id>', nodeId: '<same requiredChildReturnNodeId>',
            parentId: '<activeNodes parentId>', depth: '<activeNodes depth>',
            localObjective: '<activeNodes objective>',
            triggeringGapId: '<activeNodes triggeringGapId>',
            conclusion: '<conclusion grounded in observed terminal results>',
            reasoningSummary: '<concise summary>', claims: [], evidence: [],
            externalObservations: [], assumptions: [],
            uncertainty: { confidence: 0.5, uncertainAbout: [],
              confidenceBasis: '<observed evidence basis>' },
            conflicts: [], coverage: { resolved: [], unresolved: [], notExamined: [] },
            blindSpots: [], residualRequirements: [], proposedChildren: [],
            resolvedParentGap: false, informationToPropagate: [],
          } },
      },
      dependencies: organization.dependencies,
      communications: organization.communications,
    },
    recentContext: {
      unresolvedRequirements: epistemic.unresolvedRequirements,
      unresolvedAssumptions: epistemic.unresolvedAssumptions,
      unresolvedConflicts: epistemic.unresolvedConflicts,
      blindSpots: epistemic.blindSpots,
      recentStateDelta: epistemic.recentStateDelta,
      activeSubtree: epistemic.activeSubtree,
    },
  });
}

function normalizeReturnReportCollections(
  action: Record<string, unknown>, actorNodeId: string
): void {
  const value = action.report;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  const report = value as Record<string, unknown>;
  for (const field of ['claims', 'evidence', 'externalObservations', 'assumptions', 'conflicts',
    'blindSpots', 'residualRequirements', 'proposedChildren', 'informationToPropagate']) {
    if (!Array.isArray(report[field])) report[field] = [];
  }
  const claimStatuses = new Set(['supported', 'tentative', 'rejected']);
  report.claims = (report.claims as unknown[]).filter(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false;
    const claim = item as Record<string, unknown>;
    if (!['id', 'statement'].every(field =>
      typeof claim[field] === 'string' && String(claim[field]).trim().length > 0)) return false;
    // Providers commonly emit type/confidence/provenance fields even when the
    // JSON prompt requests the runtime schema. Fill only missing structural
    // provenance; an explicit conflicting origin remains invalid downstream.
    if (typeof claim.originNodeId !== 'string' || !claim.originNodeId.trim()) {
      claim.originNodeId = actorNodeId;
    }
    if (!claimStatuses.has(String(claim.status))) claim.status = 'tentative';
    return true;
  });
  report.evidence = (report.evidence as unknown[]).filter(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false;
    const evidence = item as Record<string, unknown>;
    if (!['id', 'content', 'provenance'].every(field =>
      typeof evidence[field] === 'string' && String(evidence[field]).trim().length > 0)) {
      return false;
    }
    if (!Array.isArray(evidence.supports)) evidence.supports = [];
    if (evidence.contradicts !== undefined && !Array.isArray(evidence.contradicts)) {
      evidence.contradicts = [];
    }
    return true;
  });
  const assumptionStatuses = new Set(['verified', 'unverified', 'contradicted']);
  report.assumptions = (report.assumptions as unknown[]).filter(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false;
    const assumption = item as Record<string, unknown>;
    if (!['id', 'statement'].every(field =>
      typeof assumption[field] === 'string' && String(assumption[field]).trim().length > 0)) {
      return false;
    }
    if (!assumptionStatuses.has(String(assumption.status))) assumption.status = 'unverified';
    if (!Array.isArray(assumption.supportingEvidence)) assumption.supportingEvidence = [];
    return true;
  });
  const mechanisms = new Set(['acquisition', 'representation', 'conversion', 'mixed']);
  const statuses = new Set(['open', 'assigned', 'resolved', 'rejected']);
  report.residualRequirements = (report.residualRequirements as unknown[]).filter(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false;
    const residual = item as Record<string, unknown>;
    return ['id', 'description', 'whyItMatters', 'requiredInformation'].every(field =>
      typeof residual[field] === 'string' && String(residual[field]).trim().length > 0)
      && residual.parentNodeId === actorNodeId
      && mechanisms.has(String(residual.likelyMechanism))
      && statuses.has(String(residual.status));
  });
  const coverage = report.coverage && typeof report.coverage === 'object'
    && !Array.isArray(report.coverage) ? report.coverage as Record<string, unknown> : {};
  for (const field of ['resolved', 'unresolved', 'notExamined']) {
    if (!Array.isArray(coverage[field])) coverage[field] = [];
  }
  report.coverage = coverage;
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
    await this.prepareDecisionBoundary(session);
    const snapshot = session.snapshot();
    if (snapshot.runtime.stopped) return { status: 'completed', snapshot };
    const requestState = compactEpistemicWorkingState(snapshot);
    if (snapshot.organizationMode === 'learned_information_realization') {
      return this.advanceLearnedController(session, snapshot, requestState, seed);
    }
    const messages: LLMMessage[] = [
      { role: 'system', content: PROPOSER_PROMPT },
      { role: 'user', content: JSON.stringify(requestState) },
    ];
    const proposalAttempts = Math.max(1, Number(process.env.ROY_LHTB_PROPOSAL_ATTEMPTS ?? 5));
    let completion: LLMJSONCompletionResult<ProposalResponse> | undefined;
    let validation: CandidateValidation | undefined;
    for (let attempt = 1; attempt <= proposalAttempts; attempt += 1) {
      let current: LLMJSONCompletionResult<ProposalResponse>;
      try {
        current = await this.provider.completeJSONWithUsage<ProposalResponse>(messages,
          { temperature: 0, maxTokens: PROPOSER_RESPONSE_MAX_TOKENS,
            thinking: { type: 'disabled' } });
      } catch (error) {
        if (!(error instanceof LLMJSONParseError)) throw error;
        session.recordModelUsage(error.completion.usage?.inputTokens
          ?? error.completion.usage?.promptTokens ?? 0,
        error.completion.usage?.outputTokens
          ?? error.completion.usage?.completionTokens ?? 0,
        error.completion.model);
        await this.auditProposalFailure(requestState, error, attempt);
        const repaired = repairProposalJSONStructure(error.completion.content);
        if (repaired) {
          current = { value: repaired as unknown as ProposalResponse,
            completion: error.completion };
        } else {
          if (attempt === proposalAttempts) {
            throw new Error('sampling_invalid:proposal_json_malformed', { cause: error });
          }
          messages.splice(0, messages.length,
            { role: 'system', content: PROPOSER_PROMPT },
            { role: 'user', content: `${JSON.stringify(requestState)}\nA prior independent attempt was malformed or truncated. Return one concise, complete JSON object only. Keep commands below ${PROPOSER_COMMAND_LIMIT} characters, do not inline whole source files or use a large here-document, and prefer a short incremental command or an existing script.` }
          );
          continue;
        }
      }
      await this.auditProposal(requestState, current);
      session.recordModelUsage(current.completion.usage?.inputTokens
        ?? current.completion.usage?.promptTokens ?? 0,
      current.completion.usage?.outputTokens
        ?? current.completion.usage?.completionTokens ?? 0,
      current.completion.model);
      const currentValidation = this.validateCandidates(current.value, session);
      this.ensureLearnedTerminationCandidate(snapshot, currentValidation);
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
        const hasHardProgressDeficit = structuralDeficits.some(reason =>
          reason.startsWith('missing_external_child_progress_candidate:')
          || reason.startsWith('missing_child_return_candidate:')
          || reason.startsWith('missing_post_verifier_progress_candidate:'));
        if (currentValidation.candidates.length > 0 && structuralDeficits.length > 0
          && !hasHardProgressDeficit) {
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
        throw new Error(`sampling_invalid:no_legal_candidates:${reasons.join('; ')}`);
      }
      const rejectedCandidates = currentValidation.dispositions
        .filter(value => !value.accepted && value.index >= 0)
        .map(value => ({
          candidate: current.value.candidates[value.index],
          reasons: value.reasons,
        }));
      const repairRequest = compactProposalRepairRequest(requestState, reasons,
        rejectedCandidates);
      messages.splice(0, messages.length,
        { role: 'system', content: PROPOSER_PROMPT },
        { role: 'user', content: repairRequest }
      );
    }
    if (!completion || !validation) throw new Error('DeepSeek proposer did not return a completion');
    const candidates = validation.candidates;
    const policyRecord: OrganizationPolicyRecord | undefined = undefined;
    const selected = candidates.find(value => value.id === completion.value.preferred_candidate_id)
      ?? candidates[0];
    if (selected.kind === 'ACQUIRE' || selected.kind === 'EXECUTE') {
      if (!selected.command?.trim()) {
        throw new Error(`${selected.kind} candidate requires a terminal command`);
      }
      const request = { id: `terminal-${snapshot.processStates.length}-${selected.id}`,
        command: selected.command, cwd: selected.cwd, timeoutMs: selected.timeoutMs ?? 120_000,
        nodeId: selected.actorNodeId, organizationActionKind: selected.kind };
      session.requestTerminal(request);
      if (policyRecord) session.recordPolicyDecision(policyRecord);
      return { status: 'terminal_request', request, snapshot: session.snapshot() };
    }
    session.applyOrganizationAction(selected.action);
    if (policyRecord) session.recordPolicyDecision(policyRecord);
    const result = session.snapshot();
    return result.runtime.stopped ? { status: 'completed', snapshot: result }
      : { status: 'continue', snapshot: result };
  }

  /** Execute one sampled Controller action through its meaningful SMDP boundary. */
  async advanceMacro(session: RoyLHTBSession, seed: number): Promise<ControllerResult> {
    const controllerResult = await this.advance(session, seed);
    const kind = controllerResult.controllerActionKind;
    if (kind !== 'DERIVE_INFO' && kind !== 'DERIVE_ORG') return controllerResult;
    if (controllerResult.status !== 'continue') {
      throw new Error(`${kind} did not leave a live child Worker boundary`);
    }
    await this.prepareDecisionBoundary(session);
    const snapshot = session.snapshot();
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const contextNode = snapshot.runtime.nodes.find(node => node.id === contextNodeId);
    const expectedMode = kind === 'DERIVE_INFO' ? 'acquire_external' : 'organize_knowledge';
    if (!contextNode || contextNode.id === snapshot.runtime.rootId
      || contextNode.specification?.realizationMode !== expectedMode) {
      throw new Error(`${kind} did not schedule its newly derived child Worker`);
    }
    const selectedKind: StructuralControllerActionKind = kind === 'DERIVE_INFO'
      ? 'CONTINUE' : 'RETURN';
    const worker = await this.materializeWorkerPayload(
      session, snapshot, compactEpistemicWorkingState(snapshot), selectedKind,
      kind === 'DERIVE_INFO' ? ['ACQUIRE'] : ['RETURN'],
    );
    if (worker.actorNodeId !== contextNodeId) {
      throw new Error(`${kind} Worker payload was materialized for another node`);
    }
    const result = this.applyMaterializedWorkerPayload(
      session, snapshot, worker, undefined, kind,
    );
    if (result.status === 'terminal_request') return result;
    await this.prepareDecisionBoundary(session);
    return { ...result, snapshot: session.snapshot() };
  }

  /** Frozen A0 readout: integrate children, then take one root-local readout step. */
  async advanceFinalizeNow(session: RoyLHTBSession): Promise<ControllerResult> {
    await this.prepareDecisionBoundary(session);
    const snapshot = session.snapshot();
    if (snapshot.runtime.stopped) return { status: 'completed', snapshot };
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const isRoot = contextNodeId === snapshot.runtime.rootId;
    const selectedKind: StructuralControllerActionKind = isRoot ? 'CONTINUE' : 'RETURN';
    const worker = await this.materializeWorkerPayload(
      session, snapshot, compactEpistemicWorkingState(snapshot), selectedKind,
      isRoot ? ['ACQUIRE', 'EXECUTE'] : ['RETURN'],
    );
    if (worker.actorNodeId !== contextNodeId) {
      throw new Error('Frozen finalize-now Worker payload was materialized for another node');
    }
    const result = this.applyMaterializedWorkerPayload(session, snapshot, worker);
    if (result.status === 'terminal_request') return result;
    await this.prepareDecisionBoundary(session);
    return { ...result, snapshot: session.snapshot() };
  }

  /** Complete deterministic/frozen semantic projection before actor sampling. */
  async prepareDecisionBoundary(session: RoyLHTBSession): Promise<void> {
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
  }

  private async advanceLearnedController(
    session: RoyLHTBSession,
    snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    requestState: Record<string, unknown>,
    seed: number,
  ): Promise<ControllerResult> {
    if (!this.learnedPolicy) {
      throw new Error('Learned mode requires ROY_LHTB_POLICY_COMMAND and has no heuristic fallback');
    }
    const controllerCandidates = this.legalStructuralControllerCandidates(snapshot);
    const policyState = this.policyState(snapshot, controllerCandidates);
    const decision = await this.learnedPolicy.select(policyState, controllerCandidates, seed);
    const selected = await this.materializeWorkerPayload(
      session, snapshot, requestState, decision.candidate.kind
    );
    decision.record.selectedSpawnMode = selected.action.childSpecification?.realizationMode;
    return this.applyMaterializedWorkerPayload(
      session, snapshot, selected, decision.record, decision.candidate.kind,
    );
  }

  private applyMaterializedWorkerPayload(
    session: RoyLHTBSession,
    snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    selected: ProposedCandidate,
    policyRecord?: OrganizationPolicyRecord,
    controllerActionKind?: StructuralControllerActionKind,
  ): ControllerResult {
    if (selected.kind === 'ACQUIRE' || selected.kind === 'EXECUTE') {
      if (!selected.command?.trim()) {
        throw new Error(`${selected.kind} Worker payload requires a terminal command`);
      }
      const request = { id: `terminal-${snapshot.processStates.length}-${selected.id}`,
        command: selected.command, cwd: selected.cwd, timeoutMs: selected.timeoutMs ?? 120_000,
        nodeId: selected.actorNodeId, organizationActionKind: selected.kind };
      session.requestTerminal(request);
      if (policyRecord) session.recordPolicyDecision(policyRecord);
      return { status: 'terminal_request', request, snapshot: session.snapshot(),
        controllerActionKind };
    }
    session.applyOrganizationAction(selected.action);
    if (policyRecord) session.recordPolicyDecision(policyRecord);
    const result = session.snapshot();
    return result.runtime.stopped
      ? { status: 'completed', snapshot: result, controllerActionKind }
      : { status: 'continue', snapshot: result, controllerActionKind };
  }

  /** Runtime legality only; frozen Worker preference must never define actor support. */
  private legalStructuralControllerCandidates(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>
  ): StructuralControllerCandidate[] {
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const contextNode = snapshot.runtime.nodes.find(node => node.id === contextNodeId);
    if (!contextNode) throw new Error(`Scheduler context node does not exist: ${contextNodeId}`);
    const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
    const kinds: StructuralControllerActionKind[] = ['CONTINUE'];
    const hasOpenLocalGap = snapshot.runtime.requirements.some(requirement =>
      isDelegableRequirement(requirement)
      && effectiveRequirementOwner(snapshot, requirement) === contextNodeId);
    if (hasOpenLocalGap && profile?.id !== 'single') {
      kinds.push('DERIVE_INFO', 'DERIVE_ORG');
    }
    const prunable = snapshot.runtime.nodes.some(node => node.id !== snapshot.runtime.rootId
      && !['returned', 'pruned', 'failed'].includes(node.status)
      && !snapshot.runtime.dependencyEdges.some(edge => edge.producerId === node.id && !edge.resolved)
      && !snapshot.runtime.derivationEdges.some(edge => edge.parentId === node.id
        && snapshot.runtime.nodes.some(child => child.id === edge.childId
          && !['returned', 'pruned', 'completed'].includes(child.status))));
    if (prunable && profile?.id !== 'single') kinds.push('PRUNE');
    const events = snapshot.runtimeEvents ?? snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    const externalChildNeedsResult = contextNodeId !== snapshot.runtime.rootId
      && contextNode.specification?.realizationMode === 'acquire_external'
      && !events.some(event => event.kind === 'terminal_result' && event.nodeId === contextNodeId);
    if (contextNodeId !== snapshot.runtime.rootId && !externalChildNeedsResult) kinds.push('RETURN');
    if (contextNodeId === snapshot.runtime.rootId
      && !snapshot.runtime.dependencyEdges.some(edge => !edge.resolved)
      && !this.verifierRetryNeedsProgress(snapshot)) kinds.push('FINISH');
    return kinds.map(kind => ({ id: `controller:${kind}`, kind, actorNodeId: contextNodeId,
      description: CONTROLLER_ACTION_DESCRIPTIONS[kind], schedulerComplexity: 0 }));
  }

  private async materializeWorkerPayload(
    session: RoyLHTBSession,
    snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    requestState: Record<string, unknown>,
    selectedKind: StructuralControllerActionKind,
    requiredRuntimeKindsOverride?: OrganizationCandidate['kind'][],
  ): Promise<ProposedCandidate> {
    if (selectedKind === 'FINISH') {
      return { id: 'worker:finish-official-verifier', kind: 'STOP',
        actorNodeId: snapshot.runtime.rootId,
        description: 'Submit the current task environment to the official verifier',
        schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: snapshot.runtime.rootId,
          finalOutput: { status: 'submitted_to_official_verifier',
            stateFingerprint: snapshot.processStates.at(-1)?.fingerprint } } };
    }
    const requiredRuntimeKinds: Record<Exclude<StructuralControllerActionKind, 'FINISH'>,
      OrganizationCandidate['kind'][]> = {
      CONTINUE: ['ACQUIRE', 'CONNECT', 'EXECUTE'],
      DERIVE_INFO: ['DERIVE'], DERIVE_ORG: ['DERIVE'], PRUNE: ['PRUNE'], RETURN: ['RETURN'],
    };
    const selectedRuntimeKinds = requiredRuntimeKindsOverride ?? requiredRuntimeKinds[selectedKind];
    const runtimeKindConstraint = requiredRuntimeKindsOverride
      ? `This frozen macro boundary requires exactly one of these Runtime payload kinds: `
        + `${selectedRuntimeKinds.join(', ')}. Payloads of every other Runtime kind are invalid. `
      : '';
    const semanticBoundary = selectedKind === 'DERIVE_INFO'
      ? 'The child may only acquire a specific missing external observation, fact, measurement, or piece of evidence through an allowed tool/environment. It must not implement the task, edit the deliverable, organize all existing knowledge, or take ownership of the parent objective.'
      : selectedKind === 'DERIVE_ORG'
        ? 'The child may only compare, synthesize, verify, disambiguate, or plan from information already represented in Roy. It must not retrieve new external evidence, implement the whole task, edit the final deliverable, or take ownership of the parent objective.'
        : selectedKind === 'CONTINUE'
          ? 'The current Worker performs the actual local reasoning, tool use, environment inspection, implementation, or verification. Do not spawn a child in this payload.'
          : selectedKind === 'PRUNE'
            ? 'Choose only the concrete low-value non-root branch to remove; do not perform other work.'
            : 'Return only this node evidence-grounded report to its parent; do not perform other work.';
    const selectedInstruction = `The shared Controller has already selected ${selectedKind}. `
      + `Return concise Runtime payloads only for this selected category. Do not choose another `
      + `Controller action. ${runtimeKindConstraint}${semanticBoundary} `
      + `For DERIVE, realizationMode must be `
      + `${selectedKind === 'DERIVE_INFO' ? 'acquire_external'
        : selectedKind === 'DERIVE_ORG' ? 'organize_knowledge' : 'unchanged'}.`;
    const messages: LLMMessage[] = [{ role: 'system', content: PROPOSER_PROMPT },
      { role: 'user', content: JSON.stringify({ ...requestState,
        selectedControllerAction: selectedKind, selectedControllerInstruction: selectedInstruction }) }];
    const attempts = Math.max(1, Number(process.env.ROY_LHTB_PROPOSAL_ATTEMPTS ?? 5));
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      let completion: LLMJSONCompletionResult<ProposalResponse>;
      try {
        completion = await this.provider.completeJSONWithUsage<ProposalResponse>(messages,
          { temperature: 0, maxTokens: PROPOSER_RESPONSE_MAX_TOKENS,
            thinking: { type: 'disabled' } });
      } catch (error) {
        if (!(error instanceof LLMJSONParseError)) throw error;
        session.recordModelUsage(error.completion.usage?.inputTokens
          ?? error.completion.usage?.promptTokens ?? 0,
        error.completion.usage?.outputTokens
          ?? error.completion.usage?.completionTokens ?? 0,
        error.completion.model);
        await this.auditProposalFailure(requestState, error, attempt);
        if (attempt === attempts) {
          throw new Error('sampling_invalid:worker_payload_json_malformed', { cause: error });
        }
        continue;
      }
      await this.auditProposal(requestState, completion);
      session.recordModelUsage(completion.completion.usage?.inputTokens
        ?? completion.completion.usage?.promptTokens ?? 0,
      completion.completion.usage?.outputTokens
        ?? completion.completion.usage?.completionTokens ?? 0,
      completion.completion.model);
      const validation = this.validateCandidates(completion.value, session);
      await this.auditCandidateValidation(validation);
      const matching = validation.candidates.filter(candidate =>
        selectedRuntimeKinds.includes(candidate.kind)
        && controllerActionForCandidate(candidate) === selectedKind);
      const selected = matching.find(candidate =>
        candidate.id === completion.value.preferred_candidate_id) ?? matching[0];
      if (selected) return selected;
      messages.splice(1, 1, { role: 'user', content: JSON.stringify({ ...requestState,
        selectedControllerAction: selectedKind,
        selectedControllerInstruction: `${selectedInstruction} The prior payload did not match `
          + `${selectedKind} or failed Runtime legality. Return a fresh matching payload only.` }) });
    }
    throw new Error(`sampling_invalid:no_worker_payload_for:${selectedKind}`);
  }

  /** Collapse frozen-Worker payloads into one categorical option per Controller action. */
  private structuralControllerCandidates(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[]
  ): StructuralControllerCandidate[] {
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const kinds = new Set<StructuralControllerActionKind>();
    const result: StructuralControllerCandidate[] = [];
    for (const candidate of candidates) {
      if (candidate.actorNodeId !== contextNodeId) continue;
      const kind = controllerActionForCandidate(candidate);
      if (kinds.has(kind)) continue;
      kinds.add(kind);
      result.push({
        id: `controller:${kind}`,
        kind,
        actorNodeId: contextNodeId,
        description: CONTROLLER_ACTION_DESCRIPTIONS[kind],
        schedulerComplexity: 0,
      });
    }
    if (result.length === 0) {
      throw new Error(`sampling_invalid:no_legal_controller_action:${contextNodeId}`);
    }
    return result;
  }

  close(): void {
    this.learnedPolicy?.close();
    this.semantic.close();
  }

  private mctsEnabled(): boolean {
    return process.env.ROY_LHTB_MCTS_ENABLED === 'true';
  }

  private async selectWithMCTS(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, candidates: ProposedCandidate[], seed: number,
    session: RoyLHTBSession,
  ): Promise<{ candidate: ProposedCandidate; record: OrganizationPolicyRecord }> {
    if (!this.learnedPolicy) throw new Error('MCTS requires the learned policy sidecar');
    const simulations = Math.max(1, Number(process.env.ROY_LHTB_MCTS_SIMULATIONS ?? 24));
    const maximumDepth = Math.max(1, Number(process.env.ROY_LHTB_MCTS_MAX_DEPTH ?? 3));
    const cpuCT = Math.max(0, Number(process.env.ROY_LHTB_MCTS_CPUCT ?? 1.5));
    const temperature = Math.max(1e-6,
      Number(process.env.ROY_LHTB_MCTS_TEMPERATURE ?? 1));
    const agentExpansionLimit = Math.max(0, Math.trunc(Number(
      process.env.ROY_LHTB_MCTS_AGENT_EXPANSIONS ?? Math.min(simulations, 4)
    )));
    let agentExpansionAttemptCount = 0;
    let agentExpansionCount = 0;
    let proposalCalls = 0;
    let proposalInputTokens = 0;
    let proposalOutputTokens = 0;
    const proposalModels = new Set<string>();
    const result = await searchOrganizationMCTS({
      rootState: RoyLHTBSession.compactForSearch(snapshot), candidates,
      simulations, maximumDepth, cpuCT, temperature, seed,
      expand: async (state, remaining, depth) => {
        if (depth === 0) {
          return this.expandMCTSNode(state, remaining, {
            proposalSource: 'real_step_agent_proposal', agentGenerated: true, depth,
          });
        }
        if (agentExpansionAttemptCount >= agentExpansionLimit) {
          return this.expandMCTSNode(state, [], {
            proposalSource: 'agent_expansion_budget_leaf', agentGenerated: false, depth,
            agentExpansionLimit,
          });
        }
        const expansionIndex = agentExpansionAttemptCount;
        agentExpansionAttemptCount += 1;
        try {
          const proposal = await this.proposeMCTSCandidates(state, depth, expansionIndex);
          proposalCalls += proposal.calls;
          proposalInputTokens += proposal.inputTokens;
          proposalOutputTokens += proposal.outputTokens;
          proposal.models.forEach(model => proposalModels.add(model));
          agentExpansionCount += 1;
          return this.expandMCTSNode(state, proposal.candidates, {
            proposalSource: 'dynamic_agent_search_expansion', agentGenerated: true, depth,
            expansionIndex, proposalAttempts: proposal.attempts,
            proposedCandidateCount: proposal.candidates.length,
          });
        } catch (error) {
          const failedUsage = error as Partial<MCTSAgentProposalResult>;
          proposalCalls += Number(failedUsage.calls ?? 0);
          proposalInputTokens += Number(failedUsage.inputTokens ?? 0);
          proposalOutputTokens += Number(failedUsage.outputTokens ?? 0);
          failedUsage.models?.forEach(model => proposalModels.add(model));
          return this.expandMCTSNode(state, [], {
            proposalSource: 'dynamic_agent_search_expansion_failed', agentGenerated: true,
            depth, expansionIndex,
            proposalError: error instanceof Error ? error.message : String(error),
          });
        }
      } });
    if (proposalCalls > 0) {
      session.recordModelUsage(proposalInputTokens, proposalOutputTokens,
        [...proposalModels].join(',') || 'deepseek-mcts-proposer');
    }
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
      contextNodeId: String(rootPolicyState.context_node_id), candidateId: selected.id,
      maskedOldLogProbability: Math.log(behaviorProbability),
      maskedOldActionLogProbability: Math.log(actionProbability),
      maskedOldCandidateConditionalLogProbability:
        Math.log(behaviorProbability / actionProbability),
      envelopeId: String((rootPolicyState.envelope as Record<string, unknown>).id),
      policyState: rootPolicyState,
      availableActions: [...new Set(candidates.map(controllerActionForCandidate))],
      rawProbabilities: actorActionProbabilities,
      maskedProbabilities: behaviorActionProbabilities,
      selectedAction: controllerActionForCandidate(selected),
      selectedSpawnMode: selected.action.childSpecification?.realizationMode,
      spawnModeProbabilities: Object.fromEntries(['acquire_external', 'organize_knowledge'].map(
        mode => [mode, candidates.filter(candidate =>
          candidate.action.childSpecification?.realizationMode === mode)
          .reduce((sum, candidate) => sum + (result.behaviorProbabilities[candidate.id] ?? 0), 0)]
      )) as OrganizationPolicyRecord['spawnModeProbabilities'],
      numRealResidualGaps: Number(rootPolicyState.num_real_residual_gaps ?? 0),
      numChildProposals: Number(rootPolicyState.num_child_proposals ?? 0),
      stopLegalReason: String(rootPolicyState.stop_legal_reason ?? ''),
      explorationStopMasked: Boolean(rootPolicyState.exploration_stop_masked),
      behaviorPolicy: 'mcts_puct', actorCandidatePriors: result.actorPriors,
      mctsVisitCounts: result.visitCounts,
      mctsBehaviorProbabilities: result.behaviorProbabilities,
      mctsSimulations: simulations, mctsMaximumDepth: maximumDepth, mctsCpuCT: cpuCT,
      mctsAgentExpansionLimit: agentExpansionLimit,
      mctsAgentExpansionAttemptCount: agentExpansionAttemptCount,
      mctsAgentExpansionCount: agentExpansionCount,
      mctsAgentFailedExpansionCount: agentExpansionAttemptCount - agentExpansionCount,
      mctsAgentProposalCalls: proposalCalls,
      mctsAgentProposalInputTokens: proposalInputTokens,
      mctsAgentProposalOutputTokens: proposalOutputTokens,
      rootTargetValue: result.rootTargetValue,
      selectedChildTargetValue: result.selectedChildTargetValue,
      selectedProcessReward: result.selectedProcessReward,
      targetValueRevision: result.targetRevision,
      mctsSearchTrace: result.trace,
      mctsSearchSamples: result.searchSamples as OrganizationPolicyRecord['mctsSearchSamples'],
      mctsSearchStates: result.searchStates,
    } };
  }

  private async proposeMCTSCandidates(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, depth: number, expansionIndex: number
  ): Promise<MCTSAgentProposalResult> {
    const requestState = {
      ...compactEpistemicWorkingState(snapshot),
      mctsSearchExpansion: {
        mode: 'dynamic_agent_candidate_generation', depth, expansionIndex,
        stateFingerprint: snapshot.processStates.at(-1)?.fingerprint,
        contextNodeId: scheduledOrganizationContextNode(snapshot),
        instruction: 'Generate fresh legal directions for this hypothetical search state.',
      },
    };
    const messages: LLMMessage[] = [
      { role: 'system', content: PROPOSER_PROMPT },
      { role: 'user', content: JSON.stringify(requestState) },
    ];
    const maximumAttempts = Math.max(1, Math.trunc(Number(
      process.env.ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS ?? 2
    )));
    let calls = 0;
    let inputTokens = 0;
    let outputTokens = 0;
    const models = new Set<string>();
    const fail = (message: string, cause?: unknown): Error & MCTSAgentProposalResult =>
      Object.assign(new Error(message, cause === undefined ? undefined : { cause }), {
        candidates: [], calls, inputTokens, outputTokens, models: [...models],
        attempts: maximumAttempts,
      });
    const account = (completion: LLMJSONCompletionResult<ProposalResponse>['completion']): void => {
      calls += 1;
      inputTokens += completion.usage?.inputTokens ?? completion.usage?.promptTokens ?? 0;
      outputTokens += completion.usage?.outputTokens ?? completion.usage?.completionTokens ?? 0;
      models.add(completion.model ?? 'deepseek-mcts-proposer');
    };
    for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
      let current: LLMJSONCompletionResult<ProposalResponse>;
      let completionAccounted = false;
      try {
        current = await this.provider.completeJSONWithUsage<ProposalResponse>(messages,
          { temperature: 0, maxTokens: PROPOSER_RESPONSE_MAX_TOKENS,
            thinking: { type: 'disabled' } });
      } catch (error) {
        if (!(error instanceof LLMJSONParseError)) {
          throw fail(error instanceof Error ? error.message : String(error), error);
        }
        account(error.completion as LLMJSONCompletionResult<ProposalResponse>['completion']);
        completionAccounted = true;
        await this.auditProposalFailure(requestState, error, attempt);
        const repaired = repairProposalJSONStructure(error.completion.content);
        if (repaired) {
          current = { value: repaired as unknown as ProposalResponse,
            completion: error.completion };
        } else {
          if (attempt === maximumAttempts) {
            throw fail('mcts_agent_expansion_json_malformed', error);
          }
          messages.splice(0, messages.length,
            { role: 'system', content: PROPOSER_PROMPT },
            { role: 'user', content: `${JSON.stringify(requestState)}\nThe previous search-expansion response was malformed. Return one concise complete JSON object with fresh legal directions for this exact context node.` }
          );
          continue;
        }
      }
      if (!completionAccounted) account(current.completion);
      await this.auditProposal(requestState, current);
      const probe = RoyLHTBSession.restore(snapshot);
      const validation = this.validateCandidates(current.value, probe);
      this.ensureLearnedTerminationCandidate(snapshot, validation);
      const structuralDeficits = this.structuralCandidateDeficits(snapshot,
        validation.candidates);
      if (structuralDeficits.length > 0) {
        validation.dispositions.push({ index: -2, accepted: false,
          reasons: structuralDeficits });
      }
      await this.auditCandidateValidation(validation, {
        mode: 'dynamic_agent_candidate_generation', depth, expansionIndex, attempt,
        stateFingerprint: snapshot.processStates.at(-1)?.fingerprint,
        contextNodeId: scheduledOrganizationContextNode(snapshot),
      });
      const hardDeficit = structuralDeficits.some(reason =>
        reason.startsWith('missing_external_child_progress_candidate:')
        || reason.startsWith('missing_child_return_candidate:')
        || reason.startsWith('missing_post_verifier_progress_candidate:'));
      if (validation.candidates.length > 0 && !hardDeficit) {
        return { candidates: validation.candidates, calls, inputTokens, outputTokens,
          models: [...models], attempts: attempt };
      }
      if (attempt < maximumAttempts) {
        const reasons = [...new Set(validation.dispositions.flatMap(value => value.reasons))];
        const rejected = validation.dispositions.filter(value => !value.accepted && value.index >= 0)
          .map(value => ({ candidate: current.value.candidates[value.index],
            reasons: value.reasons }));
        messages.splice(0, messages.length,
          { role: 'system', content: PROPOSER_PROMPT },
          { role: 'user', content: compactProposalRepairRequest(requestState, reasons, rejected) }
        );
      }
    }
    throw fail('mcts_agent_expansion_has_no_legal_candidates');
  }

  private async expandMCTSNode(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, candidates: ProposedCandidate[],
    expansionMetadata: Record<string, unknown> = {},
  ) {
    if (!this.learnedPolicy) throw new Error('MCTS requires the learned policy sidecar');
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const localCandidates = candidates.filter(candidate => candidate.actorNodeId === contextNodeId);
    // The mask is part of the behavior policy. A zero actor prior alone does not
    // make a child impossible under PUCT when visited scores are close.
    const rootPolicyState = this.policyState(snapshot, localCandidates);
    const legalCandidateIds = new Set((rootPolicyState.candidates as Array<{
      id: string; legal?: boolean }>)
      .filter(candidate => candidate.legal !== false).map(candidate => candidate.id));
    const valid: Array<{ candidate: ProposedCandidate;
      state: ReturnType<RoyLHTBSession['snapshot']>; terminal: boolean }> = [];
    for (const candidate of localCandidates) {
      if (!legalCandidateIds.has(candidate.id)) continue;
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
      valid.length ? valid.map(value => value.candidate) : localCandidates);
    currentPolicyState.search_expansion = expansionMetadata;
    if (!valid.length) {
      const graph = currentPolicyState.event_graph as Record<string, unknown>;
      const current = await this.learnedPolicy.targetValue(graph);
      return { targetValue: current.targetValue, targetRevision: current.targetRevision,
        actorPriors: {}, policyState: currentPolicyState, children: [], expansionMetadata };
    }
    const analysis = await this.learnedPolicy.analyze(currentPolicyState);
    const childPolicyStates = valid.map(value => {
      const childContextNodeId = scheduledOrganizationContextNode(value.state);
      const childCandidates = candidates.filter(candidate => candidate.id !== value.candidate.id
        && candidate.actorNodeId === childContextNodeId);
      return this.policyState(value.state,
        childCandidates);
    });
    const childValues = await this.learnedPolicy.targetValues(childPolicyStates.map(value =>
      value.event_graph as Record<string, unknown>));
    if (childValues.targetRevision !== analysis.targetRevision) {
      throw new Error('MCTS expansion mixed target-value revisions');
    }
    const children = valid.map((value, index) => {
      return { ...value, prior: analysis.candidatePriors[value.candidate.id] ?? 0,
        targetValue: childValues.targetValues[index] };
    });
    return { targetValue: analysis.targetValue, targetRevision: analysis.targetRevision,
      actorPriors: analysis.candidatePriors, policyState: currentPolicyState, children,
      expansionMetadata };
  }

  private actionProbabilitySummary(candidates: ProposedCandidate[],
    probabilities: Record<string, number>): OrganizationPolicyRecord['rawProbabilities'] {
    const result: Partial<Record<StructuralControllerActionKind, number>> = {};
    for (const candidate of candidates) {
      const kind = controllerActionForCandidate(candidate);
      result[kind] = (result[kind] ?? 0) + (probabilities[candidate.id] ?? 0);
    }
    return result;
  }

  private ensureLearnedTerminationCandidate(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>, validation: CandidateValidation
  ): void {
    if (snapshot.organizationMode !== 'learned_information_realization'
      || validation.candidates.some(candidate => candidate.kind === 'STOP')) return;
    if (scheduledOrganizationContextNode(snapshot) !== snapshot.runtime.rootId) return;
    const root = snapshot.runtime.nodes.find(node => node.id === snapshot.runtime.rootId);
    if (!root || !['ready', 'running', 'waiting', 'completed'].includes(root.status)) return;
    const candidate: ProposedCandidate = {
      id: 'stop-official-verifier', kind: 'STOP', actorNodeId: snapshot.runtime.rootId,
      description: 'Stop the organization and submit the current environment to the official verifier',
      schedulerComplexity: 1,
      action: { kind: 'STOP', actorNodeId: snapshot.runtime.rootId,
        finalOutput: { status: 'submitted_to_official_verifier',
          stateFingerprint: snapshot.processStates.at(-1)?.fingerprint } },
    };
    try {
      const probe = RecursiveInformationRealizationRuntime.restore(snapshot.runtime);
      probe.apply(candidate.action, Date.now());
    } catch {
      return;
    }
    validation.candidates.push(candidate);
    validation.dispositions.push({ index: -3, id: candidate.id, accepted: true,
      reasons: ['system_direct_actor_termination_support'] });
  }

  private validateCandidates(response: ProposalResponse, session: RoyLHTBSession): CandidateValidation {
    if (!Array.isArray(response.candidates) || response.candidates.length === 0) {
      return { candidates: [], dispositions: [{ index: -1, accepted: false,
        reasons: ['response_has_no_candidates'] }] };
    }
    const snapshot = session.snapshot();
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const active = new Set(snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status))
      .map(node => node.id));
    const direct = snapshot.organizationMode === 'single_agent_direct';
    const singleProfile = snapshot.organizationMode === 'learned_information_realization'
      && activeTopologySamplingProfile(snapshot.organizationSeed)?.id === 'single';
    const runtimeEvents = snapshot.runtimeEvents
      ?? snapshot.processStates.at(-1)?.runtimeEvents ?? [];
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
      if (kind === 'DERIVE' && actionValue.childSpecification) {
        actionValue.childSpecification = compactChildSpecification(actionValue.childSpecification);
      }
      if (kind === 'RETURN' && !actionValue.report && raw?.report
        && typeof raw.report === 'object' && !Array.isArray(raw.report)) {
        // Some JSON providers preserve the documented report shape but place
        // it beside action. Normalize that unambiguous representation before
        // Runtime legality checks; the learned Controller never sees payloads.
        actionValue.report = raw.report;
      }
      if (kind === 'RETURN') normalizeReturnReportCollections(actionValue, actorNodeId);
      if (kind === 'RETURN') {
        const actor = snapshot.runtime.nodes.find(node => node.id === actorNodeId);
        const report = actionValue.report && typeof actionValue.report === 'object'
          ? actionValue.report as Record<string, unknown> : undefined;
        if (!report || typeof report.conclusion !== 'string' || !report.conclusion.trim()) {
          reasons.push('return_report_missing_conclusion');
        }
        if (actor?.specification?.realizationMode === 'organize_knowledge') {
          const represented = ['claims', 'evidence', 'assumptions', 'blindSpots',
            'residualRequirements', 'informationToPropagate'].some(field =>
            Array.isArray(report?.[field]) && (report?.[field] as unknown[]).length > 0);
          if (!represented) reasons.push('organization_report_has_no_represented_information');
        }
      }
      const explicitActionKind = typeof actionValue.kind === 'string' ? actionValue.kind
        : typeof actionValue.type === 'string' ? actionValue.type : undefined;
      const explicitActionActor = typeof actionValue.actorNodeId === 'string'
        ? actionValue.actorNodeId : undefined;
      if (!id) reasons.push('missing_id');
      else if (seenIds.has(id)) reasons.push('duplicate_id');
      if (!allowedKinds.has(kind)) reasons.push('invalid_kind');
      if (!actorNodeId || !active.has(actorNodeId)) reasons.push('inactive_actor');
      else if (actorNodeId !== contextNodeId) reasons.push(
        `actor_is_not_scheduler_context:${actorNodeId}:expected=${contextNodeId}`
      );
      if (!description) reasons.push('missing_description');
      if (!Number.isFinite(schedulerComplexity)) reasons.push('invalid_scheduler_complexity');
      if (explicitActionKind && explicitActionKind !== kind) reasons.push('action_kind_mismatch');
      if (explicitActionActor && explicitActionActor !== actorNodeId) {
        reasons.push('action_actor_mismatch');
      }
      const directLegal = !direct || (actorNodeId === 'root'
        && ['ACQUIRE', 'EXECUTE', 'RETURN', 'STOP'].includes(kind));
      if (!directLegal) reasons.push('direct_mode_forbids_action');
      if (kind === 'RETURN' && actorNodeId === snapshot.runtime.rootId) {
        reasons.push('root_must_stop_not_return');
      }
      const singleProfileLegal = !singleProfile || (actorNodeId === snapshot.runtime.rootId
        && !['DERIVE', 'CONNECT', 'PRUNE'].includes(kind));
      if (!singleProfileLegal) reasons.push('single_agent_profile_forbids_topology_change');
      const commandValue = typeof raw?.command === 'string' ? raw.command
        : typeof actionValue.command === 'string' ? actionValue.command : undefined;
      const cwdValue = typeof raw?.cwd === 'string' ? raw.cwd
        : typeof actionValue.cwd === 'string' ? actionValue.cwd : undefined;
      const timeoutValue = raw?.timeoutMs ?? actionValue.timeoutMs;
      const timeoutMs = timeoutValue === undefined ? undefined : Number(timeoutValue);
      if (kind === 'ACQUIRE' && !commandValue?.trim()) reasons.push('acquire_missing_command');
      if (kind === 'EXECUTE' && !commandValue?.trim()) reasons.push('execute_missing_command');
      if (kind === 'RETURN' && actorNodeId !== snapshot.runtime.rootId) {
        const actor = snapshot.runtime.nodes.find(node => node.id === actorNodeId);
        const requiresExternalWork = actor?.specification?.externalAccess.allowed === true;
        const hasLocalTerminalResult = runtimeEvents.some(event =>
          event.kind === 'terminal_result' && event.nodeId === actorNodeId);
        if (requiresExternalWork && !hasLocalTerminalResult) {
          reasons.push('external_child_return_without_local_terminal_result');
        }
      }
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
      if (kind === 'DERIVE') {
        const specification = actionValue.childSpecification
          && typeof actionValue.childSpecification === 'object'
          ? actionValue.childSpecification as Record<string, unknown> : {};
        const triggeringGapId = typeof specification.triggeringGapId === 'string'
          ? specification.triggeringGapId : '';
        const requirement = snapshot.runtime.requirements.find(value =>
          value.id === triggeringGapId);
        const realizationMode = String(specification.realizationMode ?? '');
        if (!['acquire_external', 'organize_knowledge'].includes(realizationMode)) {
          reasons.push('derive_missing_or_invalid_realization_mode');
        }
        const externalAccess = specification.externalAccess as Record<string, unknown> | undefined;
        if (realizationMode === 'acquire_external' && externalAccess?.allowed !== true) {
          reasons.push('external_acquisition_child_requires_external_access');
        }
        if (requirement && (!isDelegableRequirement(requirement)
          || effectiveRequirementOwner(snapshot, requirement) !== actorNodeId)) {
          reasons.push(`derive_requirement_owner_mismatch:${triggeringGapId}:expected=${effectiveRequirementOwner(snapshot, requirement)}`);
        }
        if (typeof specification.parentId === 'string'
          && specification.parentId !== actorNodeId) {
          reasons.push('derive_parent_actor_mismatch');
        }
      }
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
    const events = snapshot.runtimeEvents ?? snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const contextNode = snapshot.runtime.nodes.find(node => node.id === contextNodeId);
    const needsExternalProgress = contextNode?.id !== snapshot.runtime.rootId
      && contextNode?.specification?.realizationMode === 'acquire_external'
      && !events.some(event => event.kind === 'terminal_result' && event.nodeId === contextNodeId);
    const deficits = needsExternalProgress && !candidates.some(candidate =>
      candidate.actorNodeId === contextNodeId && ['ACQUIRE', 'EXECUTE'].includes(candidate.kind))
      ? [`missing_external_child_progress_candidate:${contextNodeId}`] : [];
    if (this.verifierRetryNeedsProgress(snapshot) && !candidates.some(candidate =>
      candidate.actorNodeId === contextNodeId && ['ACQUIRE', 'EXECUTE'].includes(candidate.kind))) {
      deficits.push(`missing_post_verifier_progress_candidate:${contextNodeId}`);
    }
    const hasSuccessfulLocalResult = contextNode?.id !== snapshot.runtime.rootId
      && events.some(event => event.kind === 'terminal_result' && event.nodeId === contextNodeId
        && (event.exitCode ?? 0) === 0);
    if (hasSuccessfulLocalResult && !candidates.some(candidate =>
      candidate.actorNodeId === contextNodeId && candidate.kind === 'RETURN')) {
      deficits.push(`missing_child_return_candidate:${contextNodeId}`);
    }
    const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
    if (!profile) return deficits;
    const minimumNodes = profile.preferredNodeRange[0];
    if (minimumNodes <= 0) return deficits;
    const active = new Set(snapshot.runtime.nodes.filter(node =>
      ['ready', 'running', 'waiting', 'completed'].includes(node.status)).map(node => node.id));
    const openGapIds = new Set(snapshot.runtime.requirements.filter(requirement =>
      requirement.status === 'open' && active.has(requirement.parentNodeId)
    ).map(requirement => requirement.id));
    const phase = topologySamplingPhase(snapshot);
    if (phase.id === 'seed_child_local_residual') {
      return candidates.some(candidate => topologySamplingCandidateMatchesPhase(snapshot, candidate))
        ? deficits : [...deficits, 'missing_deepest_child_progress_candidate'];
    }
    if (phase.id === 'derive_child_local_residual') {
      return candidates.some(candidate => topologySamplingCandidateMatchesPhase(snapshot, candidate))
        ? deficits : [...deficits, 'missing_child_local_recursive_derive_candidate'];
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
    candidates: StructuralControllerCandidate[] | ProposedCandidate[]): Record<string, unknown> {
    const contextNodeId = scheduledOrganizationContextNode(snapshot);
    const contextNode = snapshot.runtime.nodes.find(node => node.id === contextNodeId);
    if (!contextNode) throw new Error(`Scheduler context node does not exist: ${contextNodeId}`);
    const controllerCandidates: StructuralControllerCandidate[] = candidates.some(candidate =>
      ['ACQUIRE', 'CONNECT', 'EXECUTE', 'DERIVE', 'STOP'].includes(candidate.kind))
      ? this.structuralControllerCandidates(snapshot, candidates as ProposedCandidate[])
      : candidates as StructuralControllerCandidate[];
    const contextCandidates = controllerCandidates.filter(candidate =>
      candidate.actorNodeId === contextNodeId);
    const activeNodes = snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status));
    const latest = snapshot.processStates.at(-1);
    const events = snapshot.runtimeEvents ?? latest?.runtimeEvents ?? [];
    const projectedEvents = (latest?.runtimeEvents ?? events).slice(-24);
    const requirementsByNode = new Map(snapshot.runtime.nodes.map(node => [node.id,
      snapshot.runtime.requirements.filter(requirement => requirement.parentNodeId === node.id
        || requirement.assignedNodeId === node.id)]));
    const nodesById = new Map(snapshot.runtime.nodes.map(node => [node.id, node]));
    const ancestryForNode = (nodeId: string) => {
      const result: Array<Record<string, unknown>> = [];
      const visited = new Set<string>([nodeId]);
      let parentId = nodesById.get(nodeId)?.parentId;
      while (parentId && !visited.has(parentId)) {
        visited.add(parentId);
        const ancestor = nodesById.get(parentId);
        if (!ancestor) break;
        result.push({ id: ancestor.id, parent_id: ancestor.parentId, depth: ancestor.depth,
          status: ancestor.status, local_objective: ancestor.localObjective,
          realization_mode: ancestor.specification?.realizationMode,
          unresolved_requirement_count: (requirementsByNode.get(ancestor.id) ?? [])
            .filter(requirement => requirement.status === 'open'
              || requirement.status === 'assigned').length });
        parentId = ancestor.parentId;
      }
      return result.reverse();
    };
    const nodeRuntimeEvents = (nodeId: string) => events.filter(event => event.nodeId === nodeId)
      .slice(-12);
    const agentNodes = snapshot.runtime.nodes.map(node => {
      const localRequirements = requirementsByNode.get(node.id) ?? [];
      const recentLocalEvents = nodeRuntimeEvents(node.id);
      const specification = node.specification;
      const localContext = {
        objective: node.localObjective,
        parent_id: node.parentId,
        depth: node.depth,
        status: node.status,
        triggering_gap_id: node.triggeringGapId,
        assigned_requirement_ids: node.assignedRequirementIds ?? [],
        ancestry: ancestryForNode(node.id),
        requirements: localRequirements.map(requirement => ({ id: requirement.id,
          description: requirement.description, required_information: requirement.requiredInformation,
          status: requirement.status, parent_node_id: requirement.parentNodeId,
          assigned_node_id: requirement.assignedNodeId })),
        realization_mode: specification?.realizationMode,
        required_claims: specification?.requiredClaims ?? [],
        required_evidence: specification?.requiredEvidence ?? [],
        allowed_tools: specification?.externalAccess.tools ?? [],
        expected_output: specification?.expectedOutput.requiredInformation,
        termination_condition: specification?.terminationCondition,
        recent_events: recentLocalEvents.map(event => ({ kind: event.kind,
          command: event.command, exit_code: event.exitCode,
          output: projectText(event.output), action: event.attributes?.action })),
      };
      return { id: node.id, kind: 'agent', timestamp: node.createdAt,
        text: JSON.stringify(localContext), status: node.status,
        attributes: { signal: node.status === 'returned' || node.status === 'completed' ? 1 : 0,
          depth: node.depth, is_context_node: node.id === contextNodeId ? 1 : 0 } };
    });
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
    const runtimeNodes = projectedEvents.map((event, index) => {
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
        const event = projectedEvents[index];
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
    const profile = activeTopologySamplingProfile(snapshot.organizationSeed);
    const minimumNodes = profile?.preferredNodeRange[0] ?? 0;
    const minimumDepth = profile?.preferredMinimumDepth ?? 0;
    const maximumDepthReached = Math.max(0, ...snapshot.runtime.nodes.map(node => node.depth));
    const explorationStopMasked = openRequirements.length > 0
      && (snapshot.runtime.nodes.length < minimumNodes || maximumDepthReached < minimumDepth);
    const verifierRetryStopMasked = this.verifierRetryNeedsProgress(snapshot);
    const hasExplorationAlternative = contextCandidates.some(candidate => candidate.kind !== 'FINISH');
    const policyCandidates = contextCandidates.map(candidate => ({ id: candidate.id, kind: candidate.kind,
      actor_node_id: candidate.actorNodeId, description: candidate.description,
      scheduler_complexity: candidate.schedulerComplexity,
      external_access: candidate.kind === 'DERIVE_INFO',
      resolves_gap: candidate.kind === 'RETURN',
      depth_delta: ['DERIVE_INFO', 'DERIVE_ORG'].includes(candidate.kind) ? 1 : 0,
      legal: !(candidate.kind === 'FINISH' && (explorationStopMasked || verifierRetryStopMasked)
        && hasExplorationAlternative) }));
    return {
      interface_revision: LHTB_POLICY_INTERFACE_REVISION,
      topology_search: { mode: profile ? 'profile_conditioned_diagnostic'
        : 'actor_direct_on_policy',
        profile_id: profile?.id,
        reward_semantics: 'official terminal task utility only; topology has no intrinsic reward' },
      sampling_profile: profile ? { id: profile.id,
        preferred_node_range: profile.preferredNodeRange,
        preferred_minimum_depth: profile.preferredMinimumDepth, focus: profile.focus,
        reward_semantics: 'explicit diagnostic intervention only' } : undefined,
      state_fingerprint: snapshot.processStates.at(-1)?.fingerprint,
      event_graph: { nodes, edges },
      context_node_id: contextNodeId,
      context_node: {
        id: contextNode.id,
        parent_id: contextNode.parentId,
        depth: contextNode.depth,
        status: contextNode.status,
        local_objective: contextNode.localObjective,
        triggering_gap_id: contextNode.triggeringGapId,
        assigned_requirement_ids: contextNode.assignedRequirementIds ?? [],
        ancestry: ancestryForNode(contextNodeId),
        requirements: (requirementsByNode.get(contextNodeId) ?? []).map(requirement => ({
          id: requirement.id, description: requirement.description,
          required_information: requirement.requiredInformation, status: requirement.status,
          parent_node_id: requirement.parentNodeId, assigned_node_id: requirement.assignedNodeId,
        })),
        specification: contextNode.specification,
        recent_runtime_events: nodeRuntimeEvents(contextNodeId),
      },
      scheduler: { kind: 'deterministic_dependency_event_locality', learned: false },
      candidates: policyCandidates,
      envelope: { id: 'lhtb-open', minimum_nodes: minimumNodes, maximum_nodes: 1_000_000,
        minimum_depth: minimumDepth, maximum_depth: 1_000_000, mode: 'expansive' },
      node_count: snapshot.runtime.nodes.length,
      maximum_depth_reached: maximumDepthReached,
      unresolved_gap_exists: openRequirements.length > 0,
      unresolved_requirement_ids: openRequirements.map(value => value.id),
      num_real_residual_gaps: openRequirements.length,
      num_child_proposals: contextCandidates.filter(candidate =>
        candidate.kind === 'DERIVE_INFO' || candidate.kind === 'DERIVE_ORG').length,
      available_actions: [...new Set(contextCandidates.map(candidate => candidate.kind))],
      topology_sampling_phase: topologySamplingPhase(snapshot).id,
      exploration_stop_masked: explorationStopMasked,
      verifier_retry_stop_masked: verifierRetryStopMasked,
      stop_legal_reason: verifierRetryStopMasked
        ? 'masked_until_external_progress_after_verifier_rejection'
        : explorationStopMasked
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

  private verifierRetryNeedsProgress(
    snapshot: ReturnType<RoyLHTBSession['snapshot']>
  ): boolean {
    const events = snapshot.runtimeEvents ?? snapshot.processStates.at(-1)?.runtimeEvents ?? [];
    let rejectionIndex = -1;
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const event = events[index];
      if (event.kind === 'verifier' && event.attributes?.result === 'rejected') {
        rejectionIndex = index;
        break;
      }
    }
    if (rejectionIndex < 0) return false;
    return !events.slice(rejectionIndex + 1).some(event => event.kind === 'terminal_result');
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
      temperature: 0, maxTokens: PROPOSER_RESPONSE_MAX_TOKENS, request,
      response: completion.value, usage: completion.completion.usage,
    })}\n`, 'utf8');
  }

  private async auditCandidateValidation(validation: CandidateValidation,
    searchExpansion?: Record<string, unknown>): Promise<void> {
    const root = this.auditRoot;
    if (!root && this.provider.constructor.name !== 'DeepSeekProvider') return;
    if (!root) throw new Error('ROY_LHTB_AUDIT_ROOT is required');
    await mkdir(root, { recursive: true });
    await appendFile(path.join(root, 'candidate-validation.jsonl'), `${JSON.stringify({
      schemaVersion: 1,
      searchExpansion,
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
      temperature: 0, maxTokens: PROPOSER_RESPONSE_MAX_TOKENS, attempt, request,
      responseContent: error.completion.content, usage: error.completion.usage,
      error: error.message,
    })}\n`, 'utf8');
  }
}
