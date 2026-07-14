import { v4 as uuidv4 } from 'uuid';
import type {
  DequeueOptions,
  EnqueueMessageInput,
  MessageFilter,
  MessagePriority,
  MessageStatus,
  MessageQueue,
  QueueStats,
  QueueTransition,
  RuntimeMessage,
} from './types.js';

const PRIORITY_RANK: Record<MessagePriority, number> = {
  critical: 4,
  high: 3,
  normal: 2,
  low: 1,
};

export class InMemoryMessageQueue implements MessageQueue {
  private messages = new Map<string, RuntimeMessage>();
  private order: string[] = [];

  constructor(private readonly onTransition?: (transition: QueueTransition) => void) {}

  async enqueue<TPayload>(input: EnqueueMessageInput<TPayload>): Promise<RuntimeMessage<TPayload>> {
    const now = Date.now();
    const message: RuntimeMessage<TPayload> = {
      id: uuidv4(),
      kind: input.kind,
      sessionId: input.sessionId,
      turnId: input.turnId,
      traceId: input.traceId,
      from: input.from,
      to: input.to,
      parentMessageId: input.parentMessageId,
      correlationId: input.correlationId,
      priority: input.priority ?? 'normal',
      status: 'pending',
      createdAt: now,
      updatedAt: now,
      availableAt: input.availableAt,
      expiresAt: input.expiresAt,
      payload: input.payload,
      metadata: input.metadata,
    };

    this.messages.set(message.id, message);
    this.order.push(message.id);
    this.emit({ type: 'message.enqueued', message });
    return message;
  }

  async dequeue(options: DequeueOptions = {}): Promise<RuntimeMessage | undefined> {
    const now = Date.now();
    this.expireMessages(now);

    const candidates = this.orderedMessages()
      .filter(message => message.status === 'pending')
      .filter(message => !options.to || message.to === options.to)
      .filter(message => !options.kind || options.kind.includes(message.kind))
      .filter(message => options.readyOnly === false || !message.availableAt || message.availableAt <= now)
      .sort((a, b) => {
        const priority = PRIORITY_RANK[b.priority] - PRIORITY_RANK[a.priority];
        return priority !== 0 ? priority : a.createdAt - b.createdAt;
      });

    const message = candidates[0];
    if (!message) return undefined;

    message.status = 'processing';
    message.updatedAt = now;
    this.emit({ type: 'message.processing', message });
    return { ...message };
  }

  async ack(messageId: string): Promise<void> {
    const message = this.requireMessage(messageId);
    this.assertStatus(message, ['processing'], 'acknowledge');
    message.status = 'completed';
    message.updatedAt = Date.now();
    this.emit({ type: 'message.completed', message });
  }

  async fail(messageId: string, error: Error): Promise<void> {
    const message = this.requireMessage(messageId);
    this.assertStatus(message, ['pending', 'processing'], 'fail');
    message.status = 'failed';
    message.error = error.message;
    message.updatedAt = Date.now();
    this.emit({ type: 'message.failed', message, error: error.message });
  }

  async retry(messageId: string, availableAt = Date.now()): Promise<void> {
    const message = this.requireMessage(messageId);
    this.assertStatus(message, ['processing', 'failed'], 'retry');
    const retryCount = (message.metadata?.retryCount ?? 0) + 1;
    message.status = 'pending';
    message.error = undefined;
    message.availableAt = availableAt;
    message.metadata = { ...message.metadata, retryCount };
    message.updatedAt = Date.now();
    this.emit({ type: 'message.retried', message });
  }

  async cancel(messageId: string, reason?: string): Promise<void> {
    const message = this.requireMessage(messageId);
    this.assertStatus(message, ['pending', 'processing'], 'cancel');
    message.status = 'cancelled';
    message.error = reason;
    message.updatedAt = Date.now();
    this.emit({ type: 'message.cancelled', message, reason });
  }

  async getMessage(messageId: string): Promise<RuntimeMessage | undefined> {
    const message = this.messages.get(messageId);
    return message ? { ...message } : undefined;
  }

  async listMessages(filter: MessageFilter = {}): Promise<RuntimeMessage[]> {
    let messages = this.orderedMessages()
      .filter(message => !filter.status || message.status === filter.status)
      .filter(message => !filter.kind || message.kind === filter.kind)
      .filter(message => !filter.to || message.to === filter.to)
      .filter(message => !filter.from || message.from === filter.from);

    if (filter.limit !== undefined) {
      messages = messages.slice(-filter.limit);
    }

    return messages.map(message => ({ ...message }));
  }

  async getStats(): Promise<QueueStats> {
    const stats: QueueStats = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      total: this.messages.size,
    };

    for (const message of this.messages.values()) {
      stats[message.status] += 1;
    }

    return stats;
  }

  private orderedMessages(): RuntimeMessage[] {
    return this.order
      .map(id => this.messages.get(id))
      .filter((message): message is RuntimeMessage => message !== undefined);
  }

  private requireMessage(messageId: string): RuntimeMessage {
    const message = this.messages.get(messageId);
    if (!message) {
      throw new Error(`Message "${messageId}" not found`);
    }
    return message;
  }

  private assertStatus(message: RuntimeMessage, allowed: MessageStatus[], operation: string): void {
    if (!allowed.includes(message.status)) {
      throw new Error(`Cannot ${operation} message "${message.id}" while status is ${message.status}`);
    }
  }

  private expireMessages(now: number): void {
    for (const message of this.messages.values()) {
      if (message.status === 'pending' && message.expiresAt && message.expiresAt <= now) {
        message.status = 'failed';
        message.error = 'Message expired';
        message.updatedAt = now;
        this.emit({ type: 'message.expired', message });
      }
    }
  }

  private emit(transition: QueueTransition): void {
    this.onTransition?.({ ...transition, message: { ...transition.message } });
  }
}

export default InMemoryMessageQueue;
