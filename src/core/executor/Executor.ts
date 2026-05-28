// Executor - Execution engine interface

import type { SignalBus } from './SignalBus.js';

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

/**
 * Asyncio-based executor implementation
 */
export class AsyncioExecutor implements Executor {
  readonly engine = 'asyncio';
  private config: ExecutorConfig;
  private signalBus: SignalBus;
  private semaphore?: { permit: () => void };

  constructor(config: ExecutorConfig = {}, signalBus?: SignalBus) {
    this.config = config;
    this.signalBus = signalBus!;

    if (config.maxConcurrentActivities) {
      this.semaphore = {
        permit: () => { /* placeholder */ }
      };
    }
  }

  async execute<T>(
    activity: () => T | Promise<T>,
    options?: { timeout?: number }
  ): Promise<ActivityResult<T>> {
    const timeoutMs = options?.timeout || this.config.timeoutMs;

    try {
      let result: T;

      if (timeoutMs) {
        result = await Promise.race([
          Promise.resolve(activity()),
          new Promise<T>((_, reject) =>
            setTimeout(() => reject(new Error('Activity timeout')), timeoutMs)
          ),
        ]);
      } else {
        result = await Promise.resolve(activity());
      }

      return { success: true, value: result };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async executeAll<T>(
    activities: Array<() => T | Promise<T>>,
    options?: { maxConcurrent?: number }
  ): Promise<ActivityResult<T>[]> {
    const maxConcurrent = options?.maxConcurrent || this.config.maxConcurrentActivities;

    if (maxConcurrent) {
      // Execute with concurrency limit
      const results: ActivityResult<T>[] = [];
      const chunks: Array<() => T | Promise<T>>[] = [];

      for (let i = 0; i < activities.length; i += maxConcurrent) {
        chunks.push(activities.slice(i, i + maxConcurrent));
      }

      for (const chunk of chunks) {
        const chunkResults = await Promise.all(
          chunk.map(activity => this.execute(activity))
        );
        results.push(...chunkResults);
      }

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
}

export default AsyncioExecutor;