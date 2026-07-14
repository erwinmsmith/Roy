// Event Listeners - process and format events

import type { LogEvent, EventFilter, EventType } from './events.js';

/**
 * Base listener interface
 */
export interface EventListener {
  /**
   * Handle a batch of events
   */
  handle(events: LogEvent[]): Promise<void>;

  /**
   * Flush any buffered events
   */
  flush(): Promise<void>;

  /**
   * Close the listener
   */
  close(): Promise<void>;
}

/**
 * Logging listener - outputs events to console
 */
export class LoggingListener implements EventListener {
  private filter: EventFilter | null;
  private closed = false;

  constructor(filter: EventFilter | null = null) {
    this.filter = filter;
  }

  async handle(events: LogEvent[]): Promise<void> {
    if (this.closed) throw new Error('LoggingListener is closed');
    for (const event of events) {
      if (this.filter && !this.matchesFilter(event)) {
        continue;
      }
      this.logEvent(event);
    }
  }

  private matchesFilter(event: LogEvent): boolean {
    if (!this.filter) return true;

    const levels: EventType[] = ['debug', 'info', 'warning', 'error', 'progress'];
    const eventIdx = levels.indexOf(event.type);
    const minIdx = levels.indexOf(this.filter.minLevel);

    return eventIdx >= minIdx;
  }

  private logEvent(event: LogEvent): void {
    const timestamp = new Date(event.timestamp).toISOString();
    const prefix = `[${event.type.toUpperCase()}] ${timestamp} [${event.namespace}]`;

    switch (event.type) {
      case 'debug':
        console.debug(prefix, event.message, event.data || {});
        break;
      case 'info':
        console.info(prefix, event.message, event.data || {});
        break;
      case 'warning':
        console.warn(prefix, event.message, event.data || {});
        break;
      case 'error':
        console.error(prefix, event.message, event.data || {});
        break;
      case 'progress':
        // Progress events could show a progress bar
        console.log(prefix, event.message, event.data?.percentage);
        break;
    }
  }

  async flush(): Promise<void> {
    return;
  }

  async close(): Promise<void> {
    this.closed = true;
  }
}

/**
 * Batching listener - collects events and sends in batches
 */
export class BatchingListener implements EventListener {
  private filter: EventFilter | null;
  private batchSize: number;
  private flushInterval: number;
  private buffer: LogEvent[] = [];
  private lastFlush: number = Date.now();
  private timer: NodeJS.Timeout | null = null;
  private transport: (events: LogEvent[]) => Promise<void>;
  private onError?: (error: Error) => void;

  constructor(
    transport: (events: LogEvent[]) => Promise<void>,
    options?: {
      filter?: EventFilter | null;
      batchSize?: number;
      flushInterval?: number;
      onError?: (error: Error) => void;
    }
  ) {
    this.filter = options?.filter || null;
    this.batchSize = options?.batchSize || 100;
    this.flushInterval = (options?.flushInterval || 2) * 1000;
    this.transport = transport;
    this.onError = options?.onError;

    this.startTimer();
  }

  private startTimer(): void {
    this.timer = setInterval(() => {
      this.maybeFlush();
    }, this.flushInterval);
  }

  private stopTimer(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  async handle(events: LogEvent[]): Promise<void> {
    for (const event of events) {
      if (this.filter && !this.matchesFilter(event)) {
        continue;
      }
      this.buffer.push(event);
    }

    if (this.buffer.length >= this.batchSize) {
      await this.flushBuffer();
    }
  }

  private matchesFilter(event: LogEvent): boolean {
    if (!this.filter) return true;

    const levels: EventType[] = ['debug', 'info', 'warning', 'error', 'progress'];
    const eventIdx = levels.indexOf(event.type);
    const minIdx = levels.indexOf(this.filter.minLevel);

    return eventIdx >= minIdx;
  }

  private maybeFlush(): void {
    const elapsed = Date.now() - this.lastFlush;
    if (elapsed >= this.flushInterval && this.buffer.length > 0) {
      this.flush().catch(console.error);
    }
  }

  async flush(): Promise<void> {
    await this.flushBuffer();
    this.lastFlush = Date.now();
  }

  private async flushBuffer(): Promise<void> {
    if (this.buffer.length === 0) return;

    const events = [...this.buffer];
    this.buffer = [];

    try {
      await this.transport(events);
    } catch (error) {
      if (this.onError) {
        this.onError(error instanceof Error ? error : new Error(String(error)));
      } else {
        console.error('BatchingListener error:', error);
      }
      // Put events back in buffer
      this.buffer = [...events, ...this.buffer];
    }
  }

  async close(): Promise<void> {
    this.stopTimer();
    await this.flush();
  }
}

/**
 * Progress listener - displays progress events
 */
export class ProgressListener implements EventListener {
  private activeProgress: Map<string, { message: string; percentage: number }> = new Map();
  private closed = false;

  async handle(events: LogEvent[]): Promise<void> {
    if (this.closed) throw new Error('ProgressListener is closed');
    for (const event of events) {
      if (event.type === 'progress') {
        this.updateProgress(event);
      }
    }
  }

  private updateProgress(event: LogEvent): void {
    const id = event.name || event.id;
    const percentage = event.data?.percentage as number | undefined;
    const message = event.message;

    if (percentage !== undefined) {
      const bounded = Math.max(0, Math.min(100, percentage));
      this.activeProgress.set(id, { message, percentage: bounded });
      this.renderProgress(id, message, bounded);
      if (bounded >= 100) this.activeProgress.delete(id);
    } else {
      this.activeProgress.delete(id);
    }
  }

  private renderProgress(id: string, message: string, percentage: number): void {
    const width = 30;
    const filled = Math.round((percentage / 100) * width);
    const bar = '█'.repeat(filled) + '░'.repeat(width - filled);

    process.stdout.write(`\r${bar} ${Math.round(percentage)}% - ${message}`);
    if (percentage >= 100) {
      process.stdout.write('\n');
    }
  }

  async flush(): Promise<void> {
    return;
  }

  async close(): Promise<void> {
    this.activeProgress.clear();
    this.closed = true;
  }
}

/**
 * Filter listener - filters events before passing to next listener
 */
export class FilterListener implements EventListener {
  private filter: EventFilter;
  private next: EventListener;

  constructor(filter: EventFilter, next: EventListener) {
    this.filter = filter;
    this.next = next;
  }

  async handle(events: LogEvent[]): Promise<void> {
    const filtered = events.filter(event => this.matchesFilter(event));
    if (filtered.length > 0) {
      await this.next.handle(filtered);
    }
  }

  private matchesFilter(event: LogEvent): boolean {
    const levels: EventType[] = ['debug', 'info', 'warning', 'error', 'progress'];
    const eventIdx = levels.indexOf(event.type);
    const minIdx = levels.indexOf(this.filter.minLevel);

    if (eventIdx < minIdx) return false;

    if (this.filter.namespaces && this.filter.namespaces.length > 0) {
      if (!this.filter.namespaces.includes(event.namespace)) return false;
    }

    if (this.filter.names && this.filter.names.length > 0) {
      if (!event.name || !this.filter.names.includes(event.name)) return false;
    }

    return true;
  }

  async flush(): Promise<void> {
    await this.next.flush();
  }

  async close(): Promise<void> {
    await this.next.close();
  }
}

/**
 * Composite listener - sends events to multiple listeners
 */
export class CompositeListener implements EventListener {
  private listeners: EventListener[] = [];

  addListener(listener: EventListener): void {
    this.listeners.push(listener);
  }

  removeListener(listener: EventListener): void {
    this.listeners = this.listeners.filter(l => l !== listener);
  }

  async handle(events: LogEvent[]): Promise<void> {
    const promises = this.listeners.map(listener =>
      listener.handle(events).catch(err => {
        console.error('Listener error:', err);
      })
    );
    await Promise.all(promises);
  }

  async flush(): Promise<void> {
    await Promise.all(this.listeners.map(l => l.flush()));
  }

  async close(): Promise<void> {
    await Promise.all(this.listeners.map(l => l.close()));
    this.listeners = [];
  }
}

export default EventListener;
