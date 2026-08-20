import { DeepSeekProvider, LLMJSONParseError } from '../llm/providers/openai.js';
import type { OrganizationCandidate,
  OrganizationPolicyRecord } from './informationRealizationTypes.js';
import { PythonOrganizationPolicyClient } from './pythonOrganizationPolicy.js';
import type { RoyLHTBSession } from './lhtbSession.js';
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
workspace or runs verification. Examples:
{"id":"inspect","kind":"ACQUIRE","actorNodeId":"root","description":"Inspect files",
 "schedulerComplexity":1,"command":"find . -maxdepth 2 -type f","timeoutMs":120000,
 "action":{"kind":"ACQUIRE","actorNodeId":"root"}}.
{"id":"implement","kind":"EXECUTE","actorNodeId":"root","description":"Implement and test",
 "schedulerComplexity":3,"command":"python /app/implement.py && pytest -q","timeoutMs":120000,
 "action":{"kind":"EXECUTE","actorNodeId":"root"}}.
Do not put command inside action and do not fabricate an observation or execution report.
DERIVE must use an exact open requirement ID from runtime.requirements and this action schema:
{"kind":"DERIVE","actorNodeId":"<parent>","childSpecification":{"id":"<spec-id>",
"nodeId":"<new-agent-id>","parentId":"<parent>","depth":<parent-depth+1>,
"parentGoal":"<parent objective>","triggeringGapId":"<open requirement id>",
"localObjective":"<strictly narrower executable objective>","refinement":{"parentScope":"...",
"childScope":"...","triggeringRequirementId":"<same requirement id>",
"narrowerThanParent":true,"newInformationNeeded":"...","executableEndCondition":"...",
"duplicatedByExistingNode":false},"requiredClaims":[],"requiredEvidence":[],
"relevantReportIds":[],"externalAccess":{"allowed":true,"tools":["terminal"],
"purpose":"..."},"expectedOutput":{"requiredInformation":"...",
"outputType":"epistemic_report"},"terminationCondition":"..."}}.
When independent unresolved requirements can be handled concurrently, include distinct legal
DERIVE candidates. Do not force a node count and do not derive duplicate or non-refining work.
CONNECT action is {"kind":"CONNECT","actorNodeId":"<actor>","connection":{"from":"<existing>",
"to":"<different existing>","required":false}}. PRUNE uses targetNodeId for a non-root node.
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
Propose only semantically useful actions. Do not expose hidden reasoning, benchmark grader data,
keyword fields or reward. The preferred candidate is your best next organization decision.`;

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
      const update = await this.semantic.processEvent(event, {
        requirements: latest?.requirements ?? [], claims: latest?.claims ?? [],
        assumptions: latest?.assumptions ?? [], evidence: latest?.evidence ?? [],
        external_observations: latest?.externalObservations ?? [],
      });
      session.applySemanticUpdate(update);
    }
    const snapshot = session.snapshot();
    if (snapshot.runtime.stopped) return { status: 'completed', snapshot };
    const requestState = {
      task: snapshot.instruction,
      organizationMode: snapshot.organizationMode,
      runtime: snapshot.runtime,
      recentProcessStates: snapshot.processStates.slice(-3),
    };
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
        messages.push(
          { role: 'assistant', content: error.completion.content },
          { role: 'user', content: 'The preceding response was malformed JSON. Return only one corrected JSON object in the exact candidate schema.' },
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
      await this.auditCandidateValidation(currentValidation);
      if (currentValidation.candidates.length > 0) {
        completion = current;
        validation = currentValidation;
        break;
      }
      const reasons = [...new Set(currentValidation.dispositions.flatMap(
        value => value.reasons
      ))];
      if (attempt === proposalAttempts) {
        throw new Error(`DeepSeek proposer returned no legal candidates: ${reasons.join('; ')}`);
      }
      messages.push(
        { role: 'assistant', content: current.completion.content },
        { role: 'user', content: `Every preceding candidate was rejected by the runtime: ${reasons.join('; ')}. Return a corrected set of genuinely legal candidates.` },
      );
    }
    if (!completion || !validation) throw new Error('DeepSeek proposer did not return a completion');
    const candidates = validation.candidates;
    let selected: ProposedCandidate;
    let policyRecord: OrganizationPolicyRecord | undefined;
    if (snapshot.organizationMode === 'learned_information_realization') {
      if (!this.learnedPolicy) {
        throw new Error('Learned mode requires ROY_LHTB_POLICY_COMMAND and has no heuristic fallback');
      }
      const policyState = this.policyState(snapshot, candidates);
      const decision = await this.learnedPolicy.select(policyState, candidates, seed);
      selected = decision.candidate as ProposedCandidate;
      policyRecord = decision.record;
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
      const action = { ...actionValue, kind, actorNodeId } as OrganizationCandidate['action'];
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
      if (accepted) candidates.push(candidate);
    });
    return { candidates, dispositions };
  }

  private policyState(snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[]): Record<string, unknown> {
    const activeNodes = snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status));
    const nodes = snapshot.runtime.nodes.map(node => ({ id: node.id, kind: 'agent',
      timestamp: node.createdAt, text: node.localObjective, status: node.status,
      attributes: { signal: node.status === 'returned' || node.status === 'completed' ? 1 : 0 } }));
    const edges = [
      ...snapshot.runtime.derivationEdges.map((edge, index) => ({ id: `d-${index}`,
        kind: 'derivation', from: edge.parentId, to: edge.childId })),
      ...snapshot.runtime.dependencyEdges.map((edge, index) => ({ id: `p-${index}`,
        kind: 'dependency', from: edge.producerId, to: edge.consumerId })),
      ...snapshot.runtime.communicationEdges.map((edge, index) => ({ id: `c-${index}`,
        kind: 'communication', from: edge.from, to: edge.to })),
    ];
    return {
      state_fingerprint: snapshot.processStates.at(-1)?.fingerprint,
      event_graph: { nodes, edges },
      active_node_ids: activeNodes.map(node => node.id),
      active_node_legal: activeNodes.map(node => candidates.some(
        candidate => candidate.actorNodeId === node.id
      )),
      candidates: candidates.map(candidate => ({ id: candidate.id, kind: candidate.kind,
        actor_node_id: candidate.actorNodeId, description: candidate.description,
        scheduler_complexity: candidate.schedulerComplexity, external_access: candidate.kind === 'ACQUIRE',
        resolves_gap: candidate.kind === 'ACQUIRE' || candidate.kind === 'RETURN',
        depth_delta: candidate.kind === 'DERIVE' ? 1 : 0, legal: true })),
      envelope: { id: 'lhtb-open', minimum_nodes: 0, maximum_nodes: 1_000_000,
        minimum_depth: 0, maximum_depth: 1_000_000, mode: 'expansive' },
      node_count: snapshot.runtime.nodes.length,
      maximum_depth_reached: Math.max(...snapshot.runtime.nodes.map(node => node.depth)),
      unresolved_gap_exists: snapshot.runtime.requirements.some(value => value.status === 'open'),
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
