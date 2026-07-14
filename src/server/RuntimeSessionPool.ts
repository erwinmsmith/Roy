import Runtime, { type RuntimeConfig, type RuntimeContext } from '../core/runtime/Runtime.js';

interface RuntimeSessionEntry {
  runtime: Runtime;
}

interface RuntimeSessionSlot {
  entry: Promise<RuntimeSessionEntry>;
  createdAt: number;
  lastAccessedAt: number;
}

export interface RuntimeSessionInfo {
  sessionId: string;
  isDefault: boolean;
  createdAt: number;
  lastAccessedAt: number;
}

export interface RuntimeSessionPoolOptions {
  defaultSessionId: string;
  defaultRuntime: Runtime;
  defaultContext: RuntimeContext;
  workspaceCwd: string;
  maxSessions?: number;
  idleTimeoutMs?: number;
  runtimeFactory?: () => Runtime;
}

export class RuntimeSessionPool {
  private readonly sessions = new Map<string, RuntimeSessionSlot>();
  private readonly maxSessions: number;
  private readonly idleTimeoutMs: number;
  private readonly runtimeFactory: () => Runtime;
  private readonly defaultCreatedAt = Date.now();
  private defaultLastAccessedAt = this.defaultCreatedAt;

  constructor(private readonly options: RuntimeSessionPoolOptions) {
    this.maxSessions = Math.max(1, Math.floor(options.maxSessions ?? 100));
    this.idleTimeoutMs = Math.max(1_000, Math.floor(options.idleTimeoutMs ?? 30 * 60 * 1_000));
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
    const now = Date.now();
    if (sessionId === this.options.defaultSessionId) {
      this.defaultLastAccessedAt = now;
      return this.options.defaultRuntime;
    }
    let slot = this.sessions.get(sessionId);
    if (!slot) {
      if (this.sessions.size >= this.maxSessions) {
        throw new Error(`Runtime session limit exceeded: maximum ${this.maxSessions}`);
      }
      const entry = this.create(sessionId);
      slot = { entry, createdAt: now, lastAccessedAt: now };
      this.sessions.set(sessionId, slot);
      entry.catch(() => {
        if (this.sessions.get(sessionId)?.entry === entry) this.sessions.delete(sessionId);
      });
    } else {
      slot.lastAccessedAt = now;
    }
    return (await slot.entry).runtime;
  }

  async close(sessionIdInput: unknown): Promise<boolean> {
    const sessionId = this.normalizeSessionId(sessionIdInput);
    if (sessionId === this.options.defaultSessionId) return false;
    const slot = this.sessions.get(sessionId);
    if (!slot) return false;
    this.sessions.delete(sessionId);
    const entry = await slot.entry;
    await entry.runtime.shutdown();
    return true;
  }

  list(): RuntimeSessionInfo[] {
    return [
      {
        sessionId: this.options.defaultSessionId,
        isDefault: true,
        createdAt: this.defaultCreatedAt,
        lastAccessedAt: this.defaultLastAccessedAt,
      },
      ...[...this.sessions.entries()].map(([sessionId, slot]) => ({
        sessionId,
        isDefault: false,
        createdAt: slot.createdAt,
        lastAccessedAt: slot.lastAccessedAt,
      })),
    ];
  }

  async sweepIdle(now = Date.now()): Promise<string[]> {
    const expired = [...this.sessions.entries()]
      .filter(([, slot]) => now - slot.lastAccessedAt >= this.idleTimeoutMs);
    for (const [sessionId] of expired) this.sessions.delete(sessionId);
    await Promise.allSettled(expired.map(async ([, slot]) => {
      const entry = await slot.entry;
      await entry.runtime.shutdown();
    }));
    return expired.map(([sessionId]) => sessionId);
  }

  async shutdown(): Promise<void> {
    const entries = [...this.sessions.values()];
    this.sessions.clear();
    await Promise.allSettled(entries.map(async slot => {
      const entry = await slot.entry;
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
