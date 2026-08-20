import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import readline from 'node:readline';
import type { RuntimeProcessEvent } from './globalEpistemicState.js';

export interface SemanticUpdate {
  event_id: string;
  requirements: Array<Record<string, unknown>>;
  claims: Array<Record<string, unknown>>;
  assumptions: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown>>;
  external_observations: Array<Record<string, unknown>>;
  blind_spots: string[];
  relations: Array<Record<string, unknown>>;
  provenance?: Record<string, unknown>;
}

interface Response { id: string; result?: SemanticUpdate; error?: string }

export class PythonSemanticStateClient {
  private process?: ChildProcessWithoutNullStreams;
  private pending = new Map<string, { resolve: (value: SemanticUpdate) => void;
    reject: (error: Error) => void; timer: NodeJS.Timeout }>();
  private stderr = '';

  constructor(private readonly command: string, private readonly args: string[],
    private readonly timeoutMs = 600_000) {}

  processEvent(event: RuntimeProcessEvent,
    existingState: Record<string, unknown>): Promise<SemanticUpdate> {
    const child = this.ensureProcess();
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Semantic state sidecar timed out after ${this.timeoutMs}ms`));
        this.close();
      }, this.timeoutMs);
      timer.unref?.();
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ id, event, existing_state: existingState })}\n`);
    });
  }

  close(): void {
    const child = this.process;
    this.process = undefined;
    if (child && !child.killed) child.kill('SIGTERM');
  }

  private ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.process && this.process.exitCode === null) return this.process;
    const child = spawn(this.command, this.args, { env: process.env, stdio: ['pipe', 'pipe', 'pipe'] });
    this.process = child;
    readline.createInterface({ input: child.stdout }).on('line', line => this.handle(line));
    child.stderr.on('data', chunk => { this.stderr = `${this.stderr}${String(chunk)}`.slice(-4000); });
    child.on('exit', code => {
      for (const pending of this.pending.values()) pending.reject(
        new Error(`Semantic sidecar exited (${code}): ${this.stderr}`));
      this.pending.clear();
    });
    return child;
  }

  private handle(line: string): void {
    let response: Response;
    try { response = JSON.parse(line) as Response; } catch { return; }
    const pending = this.pending.get(response.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(response.id);
    if (response.error || !response.result) pending.reject(new Error(response.error ?? 'No semantic update'));
    else pending.resolve(response.result);
  }
}
