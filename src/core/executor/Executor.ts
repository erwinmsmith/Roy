// Executor - Execution engine interface

import { signalBus as defaultSignalBus, type SignalBus } from './SignalBus.js';

export interface ExecutorConfig {
  maxConcurrentActivities?: number;
  timeoutMs?: number;
  retryPolicy?: {
    maxAttempts: number;
    initialDelayMs: number;
    backoffMultiplier: number;
  };
}

export interface ActivityResult<T = unknown> {
  success: boolean;
  value?: T;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface Executor {
  readonly engine: string;

  /**
   * Execute a single activity
   */
  execute<T>(
    activity: () => T | Promise<T>,
    options?: { timeout?: number }
  ): Promise<ActivityResult<T>>;

  /**
   * Execute multiple activities concurrently
   */
  executeAll<T>(
    activities: Array<() => T | Promise<T>>,
    options?: { maxConcurrent?: number }
  ): Promise<ActivityResult<T>[]>;

  /**
   * Execute activities and stream results
   */
  executeStream<T>(
    activities: Array<() => T | Promise<T>>
  ): AsyncGenerator<ActivityResult<T>, void, unknown>;

  /**
   * Map a function over inputs with concurrency control
   */
  map<T, R>(
    fn: (item: T) => R | Promise<R>,
    inputs: T[],
    options?: { maxConcurrent?: number }
  ): Promise<ActivityResult<R>[]>;

  /**
   * Wait for a signal
   */
  waitForSignal<T>(signalName: string, timeoutMs?: number): Promise<T>;

  /**
   * Emit a signal
   */
  signal(signalName: string, payload?: unknown): Promise<void>;

  /**
   * Cleanup executor resources
   */
  cleanup(): Promise<void>;
}

class AsyncSemaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  constructor(private readonly limit: number) {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new Error('Concurrency limit must be a positive integer');
    }
  }

  async acquire(): Promise<() => void> {
    if (this.active >= this.limit) {
      await new Promise<void>(resolve => this.waiters.push(resolve));
    }
    this.active += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active -= 1;
      this.waiters.shift()?.();
    };
  }
}

/**
 * Asyncio-based executor implementation
 */
export class AsyncioExecutor implements Executor {
  readonly engine = 'asyncio';
  private config: ExecutorConfig;
  private signalBus: SignalBus;
  private semaphore?: AsyncSemaphore;

  constructor(config: ExecutorConfig = {}, signalBus?: SignalBus) {
    this.config = config;
    this.signalBus = signalBus ?? defaultSignalBus;

    if (config.maxConcurrentActivities) {
      this.semaphore = new AsyncSemaphore(config.maxConcurrentActivities);
    }
  }

  async execute<T>(
    activity: () => T | Promise<T>,
    options?: { timeout?: number }
  ): Promise<ActivityResult<T>> {
    const timeoutMs = options?.timeout || this.config.timeoutMs;
    const release = await this.semaphore?.acquire();
    const retry = this.config.retryPolicy;
    const maxAttempts = Math.max(1, retry?.maxAttempts ?? 1);
    let delayMs = retry?.initialDelayMs ?? 0;

    try {
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
          const result = await this.runWithTimeout(activity, timeoutMs);
          return { success: true, value: result, metadata: { attempts: attempt } };
        } catch (error) {
          if (attempt === maxAttempts) {
            return {
              success: false,
              error: error instanceof Error ? error.message : String(error),
              metadata: { attempts: attempt },
            };
          }
          if (delayMs > 0) {
            await new Promise(resolve => setTimeout(resolve, delayMs));
            delayMs *= retry?.backoffMultiplier ?? 1;
          }
        }
      }
      return { success: false, error: 'Activity did not execute', metadata: { attempts: 0 } };
    } finally {
      release?.();
    }
  }

  async executeAll<T>(
    activities: Array<() => T | Promise<T>>,
    options?: { maxConcurrent?: number }
  ): Promise<ActivityResult<T>[]> {
    const maxConcurrent = options?.maxConcurrent || this.config.maxConcurrentActivities;

    if (maxConcurrent && maxConcurrent > 0) {
      const results = new Array<ActivityResult<T>>(activities.length);
      let nextIndex = 0;
      const workers = Array.from(
        { length: Math.min(maxConcurrent, activities.length) },
        async () => {
          while (nextIndex < activities.length) {
            const index = nextIndex++;
            results[index] = await this.execute(activities[index]);
          }
        }
      );
      await Promise.all(workers);
      return results;
    }

    return Promise.all(activities.map(activity => this.execute(activity)));
  }

  async *executeStream<T>(
    activities: Array<() => T | Promise<T>>
  ): AsyncGenerator<ActivityResult<T>, void, unknown> {
    for (const activity of activities) {
      yield await this.execute(activity);
    }
  }

  async map<T, R>(
    fn: (item: T) => R | Promise<R>,
    inputs: T[],
    options?: { maxConcurrent?: number }
  ): Promise<ActivityResult<R>[]> {
    return this.executeAll(inputs.map(input => () => fn(input)), options);
  }

  async waitForSignal<T>(signalName: string, timeoutMs?: number): Promise<T> {
    return this.signalBus.waitForSignal<T>(signalName, timeoutMs);
  }

  async signal(signalName: string, payload?: unknown): Promise<void> {
    await this.signalBus.signal({
      name: signalName,
      payload,
      timestamp: Date.now(),
    });
  }

  async cleanup(): Promise<void> {
    // Cleanup resources
  }

  private async runWithTimeout<T>(
    activity: () => T | Promise<T>,
    timeoutMs?: number
  ): Promise<T> {
    if (!timeoutMs) return Promise.resolve().then(activity);

    let timeoutId: NodeJS.Timeout | undefined;
    try {
      return await Promise.race([
        Promise.resolve().then(activity),
        new Promise<T>((_, reject) => {
          timeoutId = setTimeout(() => reject(new Error('Activity timeout')), timeoutMs);
        }),
      ]);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }
}

export default AsyncioExecutor;
