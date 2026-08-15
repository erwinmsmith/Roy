import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import readline from 'node:readline';
import type { StructuralCheckpoint, StructuralDecision, StructuralPolicy } from './types.js';
import { validateStructuralDecision } from './policy.js';

export interface PythonStructuralPolicyOptions {
  command?: string;
  args?: string[];
  cwd?: string;
  timeoutMs?: number;
  environment?: NodeJS.ProcessEnv;
  fallback?: StructuralPolicy;
}

interface PolicyResponse {
  id: string;
  decision?: StructuralDecision;
  error?: string;
}

export class PythonStructuralPolicyClient implements StructuralPolicy {
  readonly name = 'python-structural-policy';
  readonly version = '1.0.0';
  private process?: ChildProcessWithoutNullStreams;
  private stderr = '';
  private readonly pending = new Map<string, {
    resolve: (value: StructuralDecision) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
    checkpoint: StructuralCheckpoint;
  }>();

  constructor(private readonly options: PythonStructuralPolicyOptions = {}) {}

  async decide(checkpoint: StructuralCheckpoint): Promise<StructuralDecision> {
    try {
      const process = this.ensureProcess();
      const id = randomUUID();
      const decision = await new Promise<StructuralDecision>((resolve, reject) => {
        const timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`Python structural policy timed out after ${this.options.timeoutMs ?? 10_000}ms`));
          this.stopProcess();
        }, this.options.timeoutMs ?? 10_000);
        timer.unref?.();
        this.pending.set(id, { resolve, reject, timer, checkpoint });
        process.stdin.write(`${JSON.stringify({ id, checkpoint })}\n`, error => {
          if (!error) return;
          clearTimeout(timer);
          this.pending.delete(id);
          reject(error);
        });
      });
      return validateStructuralDecision(checkpoint, decision);
    } catch (error) {
      if (!this.options.fallback) throw error;
      return validateStructuralDecision(checkpoint, await this.options.fallback.decide(checkpoint));
    }
  }

  async close(): Promise<void> {
    this.stopProcess();
  }

  private ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.process && !this.process.killed && this.process.exitCode === null) return this.process;
    const child = spawn(
      this.options.command ?? 'python3',
      this.options.args ?? ['-m', 'roy_research.policy_server'],
      {
        cwd: this.options.cwd,
        env: { ...process.env, ...this.options.environment },
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    );
    this.process = child;
    this.stderr = '';
    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', line => this.handleLine(line));
    child.stderr.on('data', chunk => {
      this.stderr = `${this.stderr}${String(chunk)}`.slice(-4_000);
    });
    child.on('error', error => this.rejectPending(error));
    child.on('exit', (code, signal) => {
      this.rejectPending(new Error(
        `Python structural policy exited (${code ?? signal ?? 'unknown'}): ${this.stderr.trim()}`
      ));
      if (this.process === child) this.process = undefined;
    });
    return child;
  }

  private handleLine(line: string): void {
    let response: PolicyResponse;
    try {
      response = JSON.parse(line) as PolicyResponse;
    } catch {
      return;
    }
    const pending = this.pending.get(response.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(response.id);
    if (response.error || !response.decision) {
      pending.reject(new Error(response.error ?? 'Python policy returned no decision'));
      return;
    }
    pending.resolve(response.decision);
  }

  private rejectPending(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  private stopProcess(): void {
    const child = this.process;
    this.process = undefined;
    if (child && !child.killed) child.kill('SIGTERM');
  }
}
