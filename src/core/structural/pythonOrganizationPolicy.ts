import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import readline from 'node:readline';
import type { OrganizationCandidate, OrganizationPolicyRecord } from './informationRealizationTypes.js';

interface OrganizationResponse {
  id: string;
  candidate_id?: string;
  policy_record?: Record<string, unknown>;
  target_value?: number;
  target_revision?: number;
  candidate_priors?: Record<string, number>;
  action_priors?: Record<string, number>;
  actor_paths?: Array<Record<string, unknown>>;
  error?: string;
}

export class PythonOrganizationPolicyClient {
  private process?: ChildProcessWithoutNullStreams;
  private stderr = '';
  private pending = new Map<string, { resolve: (value: OrganizationResponse) => void;
    reject: (error: Error) => void; timer: NodeJS.Timeout }>();

  constructor(private readonly command = 'python3',
    private readonly args = ['-m', 'roy_research.lhtb_policy_server'],
    private readonly timeoutMs = 120_000) {}

  async select(policyState: Record<string, unknown>, candidates: OrganizationCandidate[],
    seed: number): Promise<{ candidate: OrganizationCandidate; record: OrganizationPolicyRecord }> {
    const child = this.ensureProcess();
    const id = randomUUID();
    const response = await new Promise<OrganizationResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Python organization policy timed out after ${this.timeoutMs}ms`));
        this.close();
      }, this.timeoutMs);
      timer.unref?.();
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ id, policy_state: policyState, seed })}\n`);
    });
    if (response.error || !response.candidate_id || !response.policy_record) {
      throw new Error(response.error ?? 'Learned organization policy returned no candidate');
    }
    const candidate = candidates.find(value => value.id === response.candidate_id);
    if (!candidate) throw new Error('Learned policy selected an unavailable open candidate');
    const raw = response.policy_record;
    return { candidate, record: {
      stateFingerprint: String(raw.state_fingerprint), activeNodeId: String(raw.active_node_id),
      candidateId: String(raw.candidate_id),
      maskedOldLogProbability: Number(raw.masked_old_log_probability),
      maskedOldActionLogProbability: raw.masked_old_action_log_probability === undefined
        ? undefined : Number(raw.masked_old_action_log_probability),
      maskedOldCandidateConditionalLogProbability:
        raw.masked_old_candidate_conditional_log_probability === undefined
          ? undefined : Number(raw.masked_old_candidate_conditional_log_probability),
      envelopeId: String(raw.envelope_id), policyState: raw.policy_state,
      availableActions: Array.isArray(raw.available_actions)
        ? raw.available_actions.map(String) as OrganizationPolicyRecord['availableActions']
        : undefined,
      rawProbabilities: raw.raw_probabilities as OrganizationPolicyRecord['rawProbabilities'],
      maskedProbabilities: raw.masked_probabilities as
        OrganizationPolicyRecord['maskedProbabilities'],
      selectedAction: raw.selected_action as OrganizationPolicyRecord['selectedAction'],
      numRealResidualGaps: Number(raw.num_real_residual_gaps ?? 0),
      numChildProposals: Number(raw.num_child_proposals ?? 0),
      stopLegalReason: String(raw.stop_legal_reason ?? ''),
      explorationStopMasked: Boolean(raw.exploration_stop_masked),
    } };
  }

  async analyze(policyState: Record<string, unknown>): Promise<{
    targetValue: number; targetRevision: number; candidatePriors: Record<string, number>;
    actionPriors: Record<string, number>; actorPaths: Array<Record<string, unknown>>;
  }> {
    const response = await this.request({ operation: 'analyze', policy_state: policyState });
    if (response.error || response.target_value === undefined || !response.candidate_priors) {
      throw new Error(response.error ?? 'Learned organization policy returned no analysis');
    }
    return { targetValue: Number(response.target_value),
      targetRevision: Number(response.target_revision ?? 0),
      candidatePriors: response.candidate_priors,
      actionPriors: response.action_priors ?? {}, actorPaths: response.actor_paths ?? [] };
  }

  async targetValue(eventGraph: Record<string, unknown>): Promise<{
    targetValue: number; targetRevision: number;
  }> {
    const response = await this.request({ operation: 'value', event_graph: eventGraph });
    if (response.error || response.target_value === undefined) {
      throw new Error(response.error ?? 'Learned organization policy returned no target value');
    }
    return { targetValue: Number(response.target_value),
      targetRevision: Number(response.target_revision ?? 0) };
  }

  close(): void {
    const child = this.process;
    this.process = undefined;
    if (child && !child.killed) child.kill('SIGTERM');
  }

  private async request(payload: Record<string, unknown>): Promise<OrganizationResponse> {
    const child = this.ensureProcess();
    const id = randomUUID();
    return new Promise<OrganizationResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Python organization policy timed out after ${this.timeoutMs}ms`));
        this.close();
      }, this.timeoutMs);
      timer.unref?.();
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
    });
  }

  private ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.process && this.process.exitCode === null) return this.process;
    const child = spawn(this.command, this.args, { env: process.env, stdio: ['pipe', 'pipe', 'pipe'] });
    this.process = child;
    this.stderr = '';
    readline.createInterface({ input: child.stdout }).on('line', line => this.handleLine(line));
    child.stderr.on('data', chunk => { this.stderr = `${this.stderr}${String(chunk)}`.slice(-4000); });
    child.on('exit', code => {
      this.rejectPending(new Error(`Python organization policy exited (${code}): ${this.stderr}`));
      if (this.process === child) this.process = undefined;
    });
    child.on('error', error => this.rejectPending(error));
    return child;
  }

  private handleLine(line: string): void {
    let response: OrganizationResponse;
    try { response = JSON.parse(line) as OrganizationResponse; } catch { return; }
    const pending = this.pending.get(response.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(response.id);
    pending.resolve(response);
  }

  private rejectPending(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }
}
