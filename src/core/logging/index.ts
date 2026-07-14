// Logger - event-based logging system with async event bus

import { v4 as uuidv4 } from 'uuid';
import type { LogEvent, EventType, EventContext } from './events.js';
import { createEventFilter } from './events.js';
import type { EventTransport } from './transport.js';
import {
  BatchingTransport,
  CompositeTransport,
  ConsoleTransport,
  FileTransport,
  FilteredTransport,
  HttpTransport,
  NullTransport,
} from './transport.js';
import {
  ProgressListener,
  type EventListener,
} from './listeners.js';
import { getLoggerConfig } from '../../config/index.js';

/**
 * Event bus for distributing log events
 */
class AsyncEventBus {
  private static instance: AsyncEventBus;
  private listeners: Map<string, EventListener> = new Map();
  private transport: EventTransport | null = null;
  private started = false;
  private eventQueue: LogEvent[] = [];
  private processingPromise: Promise<void> | null = null;

  private constructor() {}

  static get(): AsyncEventBus {
    if (!AsyncEventBus.instance) {
      AsyncEventBus.instance = new AsyncEventBus();
    }
    return AsyncEventBus.instance;
  }

  /**
   * Add a listener
   */
  addListener(name: string, listener: EventListener): void {
    this.listeners.set(name, listener);
  }

  /**
   * Remove a listener
   */
  async removeListener(name: string): Promise<void> {
    const listener = this.listeners.get(name);
    if (listener) {
      await listener.close();
      this.listeners.delete(name);
    }
  }

  /**
   * Set transport
   */
  setTransport(transport: EventTransport): void {
    this.transport = transport;
  }

  /**
   * Start the event bus
   */
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;
    await this.ensureProcessing();
  }

  /**
   * Stop the event bus
   */
  async stop(): Promise<void> {
    this.started = false;
    await this.processingPromise;
    await this.flush();
    await this.closeListeners();
  }

  /**
   * Emit an event
   */
  async emit(event: LogEvent): Promise<void> {
    this.eventQueue.push(event);
    if (this.started) await this.ensureProcessing();
  }

  /**
   * Flush queued events
   */
  async flush(): Promise<void> {
    await this.processingPromise;
    while (this.eventQueue.length > 0) {
      await this.processBatch(this.eventQueue.splice(0, 100));
    }
    await Promise.all(Array.from(this.listeners.values()).map(listener => listener.flush()));
    await this.transport?.flush();
  }

  private async drainQueue(): Promise<void> {
    while (this.eventQueue.length > 0 && this.started) {
      await this.processBatch(this.eventQueue.splice(0, 100));
    }
  }

  private ensureProcessing(): Promise<void> {
    if (!this.processingPromise) {
      this.processingPromise = this.drainQueue().finally(() => {
        this.processingPromise = null;
        if (this.started && this.eventQueue.length > 0) void this.ensureProcessing();
      });
    }
    return this.processingPromise;
  }

  private async processBatch(events: LogEvent[]): Promise<void> {
    await Promise.all(Array.from(this.listeners.values()).map(listener =>
      listener.handle(events).catch(error => console.error('Listener error:', error))
    ));
    if (!this.transport) return;
    try {
      await this.transport.send(events);
    } catch (error) {
      console.error('Transport error:', error);
    }
  }

  private async closeListeners(): Promise<void> {
    const promises = Array.from(this.listeners.values()).map(listener =>
      listener.close().catch(err => console.error('Close listener error:', err))
    );
    await Promise.all(promises);
    this.listeners.clear();

    if (this.transport) {
      await this.transport.close();
      this.transport = null;
    }
  }

  /**
   * Get listener names
   */
  getListenerNames(): string[] {
    return Array.from(this.listeners.keys());
  }
}

/**
 * Logger - developer-friendly logger with event emission
 */
export class Logger {
  private namespace: string;
  private sessionId: string | null;
  private eventBus: AsyncEventBus;

  constructor(namespace: string, sessionId?: string) {
    this.namespace = namespace;
    this.sessionId = sessionId || null;
    this.eventBus = AsyncEventBus.get();
  }

  /**
   * Set session ID for all future logs
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Create a child logger with additional context
   */
  child(namespace: string): Logger {
    return new Logger(`${this.namespace}.${namespace}`, this.sessionId ?? undefined);
  }

  /**
   * Create and emit an event
   */
  private emit(
    type: EventType,
    name: string | null,
    message: string,
    context: EventContext | null,
    data: Record<string, unknown>
  ): void {
    // Add session ID to context
    let finalContext = context;
    if (this.sessionId) {
      finalContext = finalContext ? { ...finalContext, sessionId: this.sessionId } : { sessionId: this.sessionId };
    }

    const event: LogEvent = {
      id: uuidv4(),
      type,
      name: name || undefined,
      namespace: this.namespace,
      message,
      context: finalContext || undefined,
      data: Object.keys(data).length > 0 ? data : undefined,
      timestamp: Date.now(),
    };

    this.eventBus.emit(event).catch(err => {
      console.error('Failed to emit event:', err);
    });
  }

  /**
   * Log debug message
   */
  debug(message: string, data?: Record<string, unknown>): void {
    this.emit('debug', null, message, null, data || {});
  }

  /**
   * Log info message
   */
  info(message: string, data?: Record<string, unknown>): void {
    this.emit('info', null, message, null, data || {});
  }

  /**
   * Log warning message
   */
  warn(message: string, data?: Record<string, unknown>): void {
    this.emit('warning', null, message, null, data || {});
  }

  /**
   * Log error message
   */
  error(message: string, data?: Record<string, unknown>): void {
    this.emit('error', null, message, null, data || {});
  }

  /**
   * Log progress message
   */
  progress(message: string, percentage?: number, data?: Record<string, unknown>): void {
    this.emit('progress', null, message, null, { percentage, ...data });
  }

  /**
   * Log with custom name
   */
  log(type: EventType, name: string, message: string, data?: Record<string, unknown>): void {
    this.emit(type, name, message, null, data || {});
  }
}

/**
 * Timing context manager for sync operations
 */
export function eventContext(
  logger: Logger,
  message: string,
  eventType: EventType = 'info',
  data?: Record<string, unknown>
): { end: () => void } {
  const startTime = Date.now();

  return {
    end: () => {
      const duration = Date.now() - startTime;
      logger.log(eventType, 'timing', `${message} finished in ${duration}ms`, {
        duration,
        ...data,
      });
    },
  };
}

/**
 * Async timing context manager
 */
export async function asyncEventContext(
  logger: Logger,
  message: string,
  eventType: EventType = 'info',
  data?: Record<string, unknown>
): Promise<{ end: () => void }> {
  const startTime = Date.now();

  return {
    end: () => {
      const duration = Date.now() - startTime;
      logger.log(eventType, 'timing', `${message} finished in ${duration}ms`, {
        duration,
        ...data,
      });
    },
  };
}

/**
 * Configure logging system
 */
export async function configureLogging(
  options?: {
    level?: 'debug' | 'info' | 'warn' | 'warning' | 'error';
    transports?: Array<'console' | 'file' | 'http'>;
    batchSize?: number;
    flushInterval?: number;
    progressDisplay?: boolean;
    httpEndpoint?: string;
    httpHeaders?: Record<string, string>;
    filePath?: string;
  }
): Promise<void> {
  const config = getLoggerConfig();
  const configuredLevel = options?.level || config?.level || 'info';
  const level = configuredLevel === 'warn' ? 'warning' : configuredLevel;
  const batchSize = options?.batchSize || config?.batchSize || 100;
  const flushInterval = options?.flushInterval || config?.flushInterval || 2.0;
  const progressDisplay = options?.progressDisplay ?? config?.progressDisplay ?? false;

  const bus = AsyncEventBus.get();
  const filter = createEventFilter(level);

  await bus.stop();

  // Add progress listener if enabled
  if (progressDisplay) {
    bus.addListener('progress', new ProgressListener());
  }

  const transports = options?.transports || config?.transports || ['console'];
  const configuredTransports: EventTransport[] = [];
  for (const transportType of transports) {
    if (transportType === 'console') {
      configuredTransports.push(new ConsoleTransport());
    } else if (transportType === 'file') {
      const filePath = options?.filePath || config?.path || 'roy.jsonl';
      configuredTransports.push(new FileTransport(filePath));
    } else if (transportType === 'http') {
      if (options?.httpEndpoint || config?.httpEndpoint) {
        configuredTransports.push(
          new HttpTransport(
            options!.httpEndpoint || config!.httpEndpoint!,
            options?.httpHeaders || config?.httpHeaders,
            config?.httpTimeout || 5.0
          )
        );
      }
    }
  }

  const baseTransport = configuredTransports.length === 0
    ? new NullTransport()
    : configuredTransports.length === 1
      ? configuredTransports[0]
      : new CompositeTransport(configuredTransports);
  bus.setTransport(new BatchingTransport(
    new FilteredTransport(baseTransport, filter),
    batchSize,
    flushInterval * 1000
  ));

  await bus.start();
}

/**
 * Shutdown logging system
 */
export async function shutdownLogging(): Promise<void> {
  const bus = AsyncEventBus.get();
  await bus.stop();
}

/**
 * Get or create a logger
 */
const _loggers: Map<string, Logger> = new Map();

export function getLogger(namespace: string, sessionId?: string): Logger {
  const key = sessionId ? `${namespace}:${sessionId}` : namespace;
  if (!_loggers.has(key)) {
    _loggers.set(key, new Logger(namespace, sessionId));
  }
  return _loggers.get(key)!;
}

/**
 * Clear all loggers (for testing)
 */
export function clearLoggers(): void {
  _loggers.clear();
}

export default Logger;
