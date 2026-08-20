import { DeepSeekProvider } from '../llm/providers/openai.js';
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

interface ProposalResponse {
  preferred_candidate_id: string;
  candidates: ProposedCandidate[];
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
Each candidate has id, kind, actorNodeId, description, schedulerComplexity and action.
Allowed kinds are DERIVE, ACQUIRE, CONNECT, EXECUTE, RETURN, PRUNE, STOP.
ACQUIRE also has command, optional cwd, timeoutMs; do not fabricate its observation.
DERIVE action must contain a strict childSpecification tied to an existing open requirement.
EXECUTE/RETURN action must contain a complete epistemic report using the repository schema.
CONNECT uses existing distinct nodes. PRUNE targets a non-root node. STOP is root-only.
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
    const completion = await this.provider.completeJSONWithUsage<ProposalResponse>([
      { role: 'system', content: PROPOSER_PROMPT },
      { role: 'user', content: JSON.stringify(requestState) },
    ], { temperature: 0, maxTokens: 32_768, thinking: { type: 'disabled' } });
    await this.auditProposal(requestState, completion);
    session.recordModelUsage(completion.completion.usage?.inputTokens
      ?? completion.completion.usage?.promptTokens ?? 0,
    completion.completion.usage?.outputTokens
      ?? completion.completion.usage?.completionTokens ?? 0,
    completion.completion.model);
    const candidates = this.validateCandidates(completion.value, session);
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
    if (selected.kind === 'ACQUIRE') {
      if (!selected.command?.trim()) throw new Error('ACQUIRE candidate requires a terminal command');
      const request = { id: `terminal-${snapshot.processStates.length}-${selected.id}`,
        command: selected.command, cwd: selected.cwd, timeoutMs: selected.timeoutMs ?? 120_000,
        nodeId: selected.actorNodeId };
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

  private validateCandidates(response: ProposalResponse, session: RoyLHTBSession): ProposedCandidate[] {
    if (!Array.isArray(response.candidates) || response.candidates.length === 0) {
      throw new Error('DeepSeek proposer returned no organization candidates');
    }
    const snapshot = session.snapshot();
    const active = new Set(snapshot.runtime.nodes
      .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status))
      .map(node => node.id));
    const direct = snapshot.organizationMode === 'single_agent_direct';
    const candidates = response.candidates.filter(candidate => {
      if (!candidate?.id || candidate.action?.kind !== candidate.kind
        || candidate.action?.actorNodeId !== candidate.actorNodeId || !active.has(candidate.actorNodeId)) {
        return false;
      }
      const directLegal = !direct || (candidate.actorNodeId === 'root'
        && ['ACQUIRE', 'EXECUTE', 'RETURN', 'STOP'].includes(candidate.kind));
      if (!directLegal) return false;
      if (candidate.kind === 'ACQUIRE') return Boolean(candidate.command?.trim());
      try {
        const probe = RecursiveInformationRealizationRuntime.restore(snapshot.runtime);
        probe.apply(candidate.action, Date.now());
        return true;
      } catch {
        return false;
      }
    });
    if (candidates.length === 0) throw new Error('DeepSeek proposer returned no legal candidates');
    return candidates;
  }

  private policyState(snapshot: ReturnType<RoyLHTBSession['snapshot']>,
    candidates: ProposedCandidate[]): Record<string, unknown> {
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
      active_node_ids: snapshot.runtime.nodes
        .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status))
        .map(node => node.id),
      active_node_legal: snapshot.runtime.nodes
        .filter(node => ['ready', 'running', 'waiting', 'completed'].includes(node.status))
        .map(() => true),
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
}
