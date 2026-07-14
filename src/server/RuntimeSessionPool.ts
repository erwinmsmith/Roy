import Runtime, { type RuntimeConfig, type RuntimeContext } from '../core/runtime/Runtime.js';

interface RuntimeSessionEntry {
  runtime: Runtime;
}

export interface RuntimeSessionPoolOptions {
  defaultSessionId: string;
  defaultRuntime: Runtime;
  defaultContext: RuntimeContext;
  workspaceCwd: string;
  maxSessions?: number;
  runtimeFactory?: () => Runtime;
}

export class RuntimeSessionPool {
  private readonly sessions = new Map<string, Promise<RuntimeSessionEntry>>();
  private readonly maxSessions: number;
  private readonly runtimeFactory: () => Runtime;

  constructor(private readonly options: RuntimeSessionPoolOptions) {
    this.maxSessions = Math.max(1, Math.floor(options.maxSessions ?? 100));
    this.runtimeFactory = options.runtimeFactory ?? (() => new Runtime());
  }

  normalizeSessionId(value: unknown): string {
    if (value === undefined || value === null || value === '') return this.options.defaultSessionId;
    if (typeof value !== 'string') throw new Error('Session ID must be a string');
    const sessionId = value.trim();
    if (!/^[A-Za-z0-9._-]{1,100}$/.test(sessionId)) {
      throw new Error('Session ID must be 1-100 characters using letters, numbers, dot, underscore, or hyphen');
    }
    return sessionId;
  }

  async get(sessionIdInput: unknown): Promise<Runtime> {
    const sessionId = this.normalizeSessionId(sessionIdInput);
    if (sessionId === this.options.defaultSessionId) return this.options.defaultRuntime;
    let entryPromise = this.sessions.get(sessionId);
    if (!entryPromise) {
      if (this.sessions.size >= this.maxSessions) {
        throw new Error(`Runtime session limit exceeded: maximum ${this.maxSessions}`);
      }
      entryPromise = this.create(sessionId);
      this.sessions.set(sessionId, entryPromise);
      entryPromise.catch(() => this.sessions.delete(sessionId));
    }
    return (await entryPromise).runtime;
  }

  async close(sessionIdInput: unknown): Promise<boolean> {
    const sessionId = this.normalizeSessionId(sessionIdInput);
    if (sessionId === this.options.defaultSessionId) return false;
    const entryPromise = this.sessions.get(sessionId);
    if (!entryPromise) return false;
    this.sessions.delete(sessionId);
    const entry = await entryPromise;
    await entry.runtime.shutdown();
    return true;
  }

  async shutdown(): Promise<void> {
    const entries = [...this.sessions.values()];
    this.sessions.clear();
    await Promise.allSettled(entries.map(async entryPromise => {
      const entry = await entryPromise;
      await entry.runtime.shutdown();
    }));
  }

  private async create(sessionId: string): Promise<RuntimeSessionEntry> {
    const runtime = this.runtimeFactory();
    const defaultFsm = this.options.defaultContext.fsm.getContext();
    const runtimeConfig: RuntimeConfig = {
      agentName: 'Roy',
      agentGoal: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      sessionId,
      fsmEnabled: true,
      budget: defaultFsm.budget ?? undefined,
      llmProvider: this.options.defaultContext.llm ?? undefined,
      workspaceCwd: this.options.workspaceCwd,
    };
    await runtime.initialize(runtimeConfig);
    return { runtime };
  }
}

export default RuntimeSessionPool;
