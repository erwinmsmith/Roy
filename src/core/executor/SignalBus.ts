// Enhanced Signal Bus - supports waitForSignal, subscriptions, and rich handlers

import { v4 as uuidv4 } from 'uuid';

export interface Signal<T = unknown> {
  name: string;
  payload?: T;
  description?: string;
  metadata?: Record<string, unknown>;
  workflowId?: string;
  timestamp: number;
}

export interface SignalSubscription {
  id: string;
  signalName: string;
  handler: (signal: Signal) => void | Promise<void>;
  filter?: (signal: Signal) => boolean;
  once?: boolean;
}

export interface SignalHandler {
  onSignal(signalName: string): (handler: (signal: Signal) => Promise<void>) => void;
  signal(signal: Signal): Promise<void>;
  waitForSignal<T>(signalName: string, timeoutMs?: number): Promise<T>;
  subscribe(
    signalName: string,
    handler: (signal: Signal) => void | Promise<void>,
    options?: { filter?: (signal: Signal) => boolean; once?: boolean }
  ): string;
  unsubscribe(subscriptionId: string): void;
}

interface PendingSignalHandler {
  resolve: (value?: unknown) => void;
  reject: (error: Error) => void;
  timeoutId?: NodeJS.Timeout;
}

export class SignalBus implements SignalHandler {
  private handlers: Map<string, Array<(signal: Signal) => Promise<void>>> = new Map();
  private subscriptions: Map<string, SignalSubscription> = new Map();
  private pendingSignals: Map<string, PendingSignalHandler[]> = new Map();
  private locks: Map<string, Promise<void>> = new Map();

  /**
   * Register a handler for a signal (decorator style)
   */
  onSignal(signalName: string): (handler: (signal: Signal) => Promise<void>) => void {
    return (handler: (signal: Signal) => Promise<void>) => {
      if (!this.handlers.has(signalName)) {
        this.handlers.set(signalName, []);
      }
      this.handlers.get(signalName)!.push(handler);
    };
  }

  /**
   * Subscribe to a signal with optional filter
   * Returns a subscription ID that can be used to unsubscribe
   */
  subscribe(
    signalName: string,
    handler: (signal: Signal) => void | Promise<void>,
    options?: { filter?: (signal: Signal) => boolean; once?: boolean }
  ): string {
    const id = uuidv4();
    const subscription: SignalSubscription = {
      id,
      signalName,
      handler,
      filter: options?.filter,
      once: options?.once ?? false,
    };

    this.subscriptions.set(id, subscription);
    return id;
  }

  /**
   * Unsubscribe from a signal
   */
  unsubscribe(subscriptionId: string): void {
    this.subscriptions.delete(subscriptionId);
  }

  /**
   * Unsubscribe all subscriptions for a signal name
   */
  unsubscribeAll(signalName: string): void {
    for (const [id, sub] of this.subscriptions) {
      if (sub.signalName === signalName) {
        this.subscriptions.delete(id);
      }
    }
  }

  /**
   * Emit a signal
   */
  async signal(signal: Signal): Promise<void> {
    const signalWithTimestamp: Signal = {
      ...signal,
      timestamp: signal.timestamp || Date.now(),
    };

    // Execute registered handlers
    const handlers = this.handlers.get(signal.name) || [];
    await this.executeHandlers(handlers, signalWithTimestamp);

    // Execute subscriptions
    const subs = Array.from(this.subscriptions.values()).filter(
      sub => sub.signalName === signal.name
    );

    for (const sub of subs) {
      // Apply filter if present
      if (sub.filter && !sub.filter(signalWithTimestamp)) {
        continue;
      }

      try {
        const result = sub.handler(signalWithTimestamp);
        if (result instanceof Promise) {
          await result;
        }
      } catch (error) {
        console.error(`Error in subscription handler for ${signal.name}:`, error);
      }

      // Remove if once subscription
      if (sub.once) {
        this.subscriptions.delete(sub.id);
      }
    }

    // Resolve pending waiters
    await this.resolvePendingSignals(signal.name, signalWithTimestamp.payload);
  }

  /**
   * Wait for a specific signal with optional timeout
   */
  async waitForSignal<T>(signalName: string, timeoutMs?: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let timeoutId: NodeJS.Timeout | undefined;

      const pending: PendingSignalHandler = {
        resolve: (value?: unknown) => {
          if (timeoutId) clearTimeout(timeoutId);
          resolve(value as T);
        },
        reject: (error: Error) => {
          if (timeoutId) clearTimeout(timeoutId);
          reject(error);
        },
      };

      if (!this.pendingSignals.has(signalName)) {
        this.pendingSignals.set(signalName, []);
      }
      this.pendingSignals.get(signalName)!.push(pending);

      if (timeoutMs) {
        timeoutId = setTimeout(() => {
          const pendingList = this.pendingSignals.get(signalName);
          if (pendingList) {
            const index = pendingList.indexOf(pending);
            if (index > -1) {
              pendingList.splice(index, 1);
            }
          }
          reject(new Error(`Timeout waiting for signal: ${signalName} (${timeoutMs}ms)`));
        }, timeoutMs);
        pending.timeoutId = timeoutId;
      }
    });
  }

  /**
   * Wait for a signal matching a predicate
   */
  async waitForSignalWithPredicate<T>(
    signalName: string,
    predicate: (payload: T) => boolean,
    timeoutMs?: number
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let timeoutId: NodeJS.Timeout | undefined;
      let resolved = false;

      const subscriptionId = this.subscribe(signalName, (signal) => {
        if (resolved) return;

        const payload = signal.payload as T;
        if (predicate(payload)) {
          resolved = true;
          if (timeoutId) clearTimeout(timeoutId);
          this.unsubscribe(subscriptionId);
          resolve(payload);
        }
      });

      if (timeoutMs) {
        timeoutId = setTimeout(() => {
          if (!resolved) {
            resolved = true;
            this.unsubscribe(subscriptionId);
            reject(new Error(`Timeout waiting for signal: ${signalName} matching predicate`));
          }
        }, timeoutMs);
      }
    });
  }

  /**
   * Create a signal and emit it
   */
  async emit<T>(
    name: string,
    payload?: T,
    description?: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    await this.signal({
      name,
      payload,
      description,
      metadata,
      timestamp: Date.now(),
    });
  }

  /**
   * Create and emit a signal with workflow ID
   */
  async emitForWorkflow<T>(
    name: string,
    workflowId: string,
    payload?: T,
    description?: string
  ): Promise<void> {
    await this.signal({
      name,
      payload,
      description,
      workflowId,
      timestamp: Date.now(),
    });
  }

  /**
   * Clear all handlers for a signal
   */
  clearHandlers(signalName?: string): void {
    if (signalName) {
      this.handlers.delete(signalName);
    } else {
      this.handlers.clear();
    }
  }

  /**
   * Clear all pending signals
   */
  clearPending(): void {
    for (const pending of this.pendingSignals.values()) {
      for (const p of pending) {
        if (p.timeoutId) clearTimeout(p.timeoutId);
        p.reject(new Error('SignalBus cleared'));
      }
    }
    this.pendingSignals.clear();
  }

  /**
   * Cleanup all subscriptions and handlers
   */
  cleanup(): void {
    this.subscriptions.clear();
    this.clearHandlers();
    this.clearPending();
  }

  /**
   * Get registered signal names
   */
  getSignalNames(): string[] {
    return Array.from(this.handlers.keys());
  }

  /**
   * Get active subscription count
   */
  getSubscriptionCount(): number {
    return this.subscriptions.size;
  }

  private async executeHandlers(
    handlers: Array<(signal: Signal) => Promise<void>>,
    signal: Signal
  ): Promise<void> {
    const tasks = handlers.map(handler => {
      try {
        return handler(signal);
      } catch (error) {
        console.error(`Error in signal handler for ${signal.name}:`, error);
        return Promise.resolve();
      }
    });

    await Promise.allSettled(tasks);
  }

  private async resolvePendingSignals(signalName: string, payload?: unknown): Promise<void> {
    const pending = this.pendingSignals.get(signalName) || [];
    for (const p of pending) {
      p.resolve(payload);
    }
    this.pendingSignals.delete(signalName);
  }

  private async getLock(key: string): Promise<void> {
    while (this.locks.has(key)) {
      await this.locks.get(key);
    }
    let release: () => void;
    const lock = new Promise<void>(res => { release = res; });
    this.locks.set(key, lock);

    setImmediate(() => {
      this.locks.delete(key);
      release!();
    });
  }
}

// Singleton instance
export const signalBus = new SignalBus();

export default SignalBus;
