// Event Transport - abstract interface for sending events to external systems

import type { LogEvent, EventFilter } from './events.js';
import { EventType } from './events.js';

/**
 * Transport interface for sending events
 */
export interface EventTransport {
  /**
   * Send a batch of events
   */
  send(events: LogEvent[]): Promise<void>;

  /**
   * Close the transport
   */
  close(): Promise<void>;
}

/**
 * Console transport - logs to console
 */
export class ConsoleTransport implements EventTransport {
  private pretty: boolean;

  constructor(pretty = true) {
    this.pretty = pretty;
  }

  async send(events: LogEvent[]): Promise<void> {
    for (const event of events) {
      this.printEvent(event);
    }
  }

  private printEvent(event: LogEvent): void {
    const timestamp = new Date(event.timestamp).toISOString();
    const prefix = `[${event.type.toUpperCase()}]`;

    if (this.pretty) {
      const parts = [prefix, timestamp, `[${event.namespace}]`];
      if (event.name) parts.push(`[${event.name}]`);
      parts.push(event.message);

      const color = this.getColor(event.type);
      console.log(`%c${parts.join(' ')}`, color);

      if (event.data && Object.keys(event.data).length > 0) {
        console.log('  Data:', JSON.stringify(event.data, null, 2));
      }
    } else {
      console.log(JSON.stringify(event));
    }
  }

  private getColor(type: EventType): string {
    switch (type) {
      case 'debug':
        return 'color: #888';
      case 'info':
        return 'color: #2196F3';
      case 'warning':
        return 'color: #FF9800';
      case 'error':
        return 'color: #F44336';
      case 'progress':
        return 'color: #4CAF50';
      default:
        return '';
    }
  }

  async close(): Promise<void> {
    // Nothing to close for console
  }
}

/**
 * File transport - writes events to a file
 */
export class FileTransport implements EventTransport {
  private path: string;
  private stream: { write: (data: string) => void } | null = null;

  constructor(path: string) {
    this.path = path;
  }

  async send(events: LogEvent[]): Promise<void> {
    if (events.length === 0) return;

    // Lazy import to avoid circular dependencies
    const { appendFileSync } = await import('fs');

    const lines = events.map(e => JSON.stringify(e)).join('\n');
    appendFileSync(this.path, lines + '\n');
  }

  async close(): Promise<void> {
    // File handle closed automatically
  }
}

/**
 * HTTP transport - sends events to a remote endpoint
 */
export class HttpTransport implements EventTransport {
  private endpoint: string;
  private headers: Record<string, string>;
  private timeout: number;

  constructor(
    endpoint: string,
    headers: Record<string, string> = {},
    timeout = 5000
  ) {
    this.endpoint = endpoint;
    this.headers = { 'Content-Type': 'application/json', ...headers };
    this.timeout = timeout;
  }

  async send(events: LogEvent[]): Promise<void> {
    if (events.length === 0) return;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ events }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async close(): Promise<void> {
    // Nothing to close for HTTP
  }
}

/**
 * Null transport - discards all events
 */
export class NullTransport implements EventTransport {
  async send(_events: LogEvent[]): Promise<void> {
    // Discard all events
  }

  async close(): Promise<void> {
    // Nothing to close
  }
}

export default EventTransport;