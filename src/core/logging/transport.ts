import { appendFile } from 'node:fs/promises';
import { eventMatchesFilter, type LogEvent, type EventFilter, type EventType } from './events.js';

export interface EventTransport {
  send(events: LogEvent[]): Promise<void>;
  flush(): Promise<void>;
  close(): Promise<void>;
}

abstract class StatefulTransport implements EventTransport {
  protected closed = false;

  abstract send(events: LogEvent[]): Promise<void>;

  async flush(): Promise<void> {
    return;
  }

  async close(): Promise<void> {
    this.closed = true;
  }

  protected assertOpen(): void {
    if (this.closed) throw new Error(`${this.constructor.name} is closed`);
  }
}

export class ConsoleTransport extends StatefulTransport {
  constructor(private readonly pretty = true) {
    super();
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    for (const event of events) this.printEvent(event);
  }

  private printEvent(event: LogEvent): void {
    const timestamp = new Date(event.timestamp).toISOString();
    const prefix = `[${event.type.toUpperCase()}]`;
    if (!this.pretty) {
      console.log(JSON.stringify(event));
      return;
    }

    const parts = [prefix, timestamp, `[${event.namespace}]`];
    if (event.name) parts.push(`[${event.name}]`);
    parts.push(event.message);
    console.log(`%c${parts.join(' ')}`, this.getColor(event.type));
    if (event.data && Object.keys(event.data).length > 0) {
      console.log('  Data:', JSON.stringify(event.data, null, 2));
    }
  }

  private getColor(type: EventType): string {
    switch (type) {
      case 'debug': return 'color: #888';
      case 'info': return 'color: #2196F3';
      case 'warning': return 'color: #FF9800';
      case 'error': return 'color: #F44336';
      case 'progress': return 'color: #4CAF50';
    }
  }
}

export class FileTransport extends StatefulTransport {
  private pendingWrite: Promise<void> = Promise.resolve();

  constructor(private readonly path: string) {
    super();
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    if (events.length === 0) return;
    const payload = `${events.map(event => JSON.stringify(event)).join('\n')}\n`;
    this.pendingWrite = this.pendingWrite
      .catch(() => undefined)
      .then(() => appendFile(this.path, payload, 'utf8'));
    await this.pendingWrite;
  }

  override async flush(): Promise<void> {
    await this.pendingWrite;
  }

  override async close(): Promise<void> {
    if (this.closed) return;
    await this.flush();
    await super.close();
  }
}

export class HttpTransport extends StatefulTransport {
  private readonly headers: Record<string, string>;
  private readonly controllers = new Set<AbortController>();

  constructor(
    private readonly endpoint: string,
    headers: Record<string, string> = {},
    private readonly timeout = 5000
  ) {
    super();
    this.headers = { 'Content-Type': 'application/json', ...headers };
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    if (events.length === 0) return;
    const controller = new AbortController();
    this.controllers.add(controller);
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ events }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    } finally {
      clearTimeout(timeoutId);
      this.controllers.delete(controller);
    }
  }

  override async close(): Promise<void> {
    if (this.closed) return;
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
    await super.close();
  }
}

export class CompositeTransport extends StatefulTransport {
  constructor(private readonly transports: EventTransport[]) {
    super();
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    const results = await Promise.allSettled(this.transports.map(transport => transport.send(events)));
    const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
    if (failures.length > 0) {
      throw new AggregateError(failures.map(result => result.reason), `${failures.length} logging transport(s) failed`);
    }
  }

  override async flush(): Promise<void> {
    await Promise.all(this.transports.map(transport => transport.flush()));
  }

  override async close(): Promise<void> {
    if (this.closed) return;
    await this.flush();
    await Promise.all(this.transports.map(transport => transport.close()));
    await super.close();
  }
}

export class FilteredTransport extends StatefulTransport {
  constructor(
    private readonly transport: EventTransport,
    private readonly filter: EventFilter
  ) {
    super();
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    const filtered = events.filter(event => eventMatchesFilter(event, this.filter));
    if (filtered.length > 0) await this.transport.send(filtered);
  }

  override async flush(): Promise<void> {
    await this.transport.flush();
  }

  override async close(): Promise<void> {
    if (this.closed) return;
    await this.transport.close();
    await super.close();
  }
}

export class BatchingTransport extends StatefulTransport {
  private buffer: LogEvent[] = [];
  private timer?: NodeJS.Timeout;
  private flushing: Promise<void> = Promise.resolve();

  constructor(
    private readonly transport: EventTransport,
    private readonly batchSize = 100,
    private readonly flushIntervalMs = 2000
  ) {
    super();
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error('batchSize must be a positive integer');
    if (!Number.isFinite(flushIntervalMs) || flushIntervalMs < 0) throw new Error('flushIntervalMs must be non-negative');
  }

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    if (events.length === 0) return;
    this.buffer.push(...events);
    if (this.buffer.length >= this.batchSize) {
      await this.flush();
    } else {
      this.scheduleFlush();
    }
  }

  override async flush(): Promise<void> {
    this.clearTimer();
    this.flushing = this.flushing.catch(() => undefined).then(async () => {
      if (this.buffer.length === 0) return;
      const events = this.buffer.splice(0, this.buffer.length);
      try {
        await this.transport.send(events);
        await this.transport.flush();
      } catch (error) {
        this.buffer.unshift(...events);
        throw error;
      }
    });
    await this.flushing;
  }

  override async close(): Promise<void> {
    if (this.closed) return;
    await this.flush();
    await this.transport.close();
    await super.close();
  }

  private scheduleFlush(): void {
    if (this.timer || this.flushIntervalMs === 0) return;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.flush().catch(error => console.error('Batching transport flush failed:', error));
    }, this.flushIntervalMs);
    this.timer.unref?.();
  }

  private clearTimer(): void {
    if (!this.timer) return;
    clearTimeout(this.timer);
    this.timer = undefined;
  }
}

export class NullTransport extends StatefulTransport {
  private acceptedEvents = 0;

  async send(events: LogEvent[]): Promise<void> {
    this.assertOpen();
    this.acceptedEvents += events.length;
  }

  getAcceptedEventCount(): number {
    return this.acceptedEvents;
  }
}

export default EventTransport;
