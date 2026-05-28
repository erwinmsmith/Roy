// Message Queue for Agent communication

import { v4 as uuidv4 } from 'uuid';

export interface QueueMessage {
  id: string;
  sender: string;
  recipient: string;
  content: unknown;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export class MessageQueue {
  private queues: Map<string, AsyncQueue<QueueMessage>> = new Map();
  private receivers: string[];

  constructor(receivers: string[] = ['env']) {
    // Ensure 'env' is always in receivers
    if (!receivers.includes('env')) {
      receivers = [...receivers, 'env'];
    }
    this.receivers = receivers;
    this.initializeQueues();
  }

  private initializeQueues(): void {
    for (const receiver of this.receivers) {
      this.queues.set(receiver, new AsyncQueue<QueueMessage>());
    }
  }

  /**
   * Add a new receiver
   */
  addReceiver(receiver: string): void {
    if (!this.receivers.includes(receiver)) {
      this.receivers.push(receiver);
      this.queues.set(receiver, new AsyncQueue<QueueMessage>());
    }
  }

  /**
   * Remove a receiver
   */
  removeReceiver(receiver: string): boolean {
    if (receiver === 'env') return false; // Cannot remove env
    const queue = this.queues.get(receiver);
    if (queue) {
      queue.clear();
      this.queues.delete(receiver);
      this.receivers = this.receivers.filter(r => r !== receiver);
      return true;
    }
    return false;
  }

  /**
   * Send a message to a recipient
   */
  async send(sender: string, recipient: string, content: unknown, metadata?: Record<string, unknown>): Promise<string> {
    if (!this.queues.has(recipient)) {
      this.addReceiver(recipient);
    }

    const message: QueueMessage = {
      id: uuidv4(),
      sender,
      recipient,
      content,
      timestamp: Date.now(),
      metadata,
    };

    await this.queues.get(recipient)!.push(message);
    return message.id;
  }

  /**
   * Send a message to multiple recipients
   */
  async broadcast(sender: string, recipients: string[], content: unknown, metadata?: Record<string, unknown>): Promise<string[]> {
    return Promise.all(
      recipients.map(recipient => this.send(sender, recipient, content, metadata))
    );
  }

  /**
   * Receive a message for a specific recipient
   */
  async receive(recipient: string): Promise<QueueMessage | undefined> {
    const queue = this.queues.get(recipient);
    if (!queue) {
      return undefined;
    }
    return queue.shift();
  }

  /**
   * Stream messages for a specific recipient
   */
  async *stream(recipient: string): AsyncGenerator<QueueMessage, void, unknown> {
    const queue = this.queues.get(recipient);
    if (!queue) return;

    while (true) {
      const message = await queue.shift();
      if (message) {
        yield message;
      }
    }
  }

  /**
   * Peek at the next message without removing it
   */
  peek(recipient: string): QueueMessage | undefined {
    const queue = this.queues.get(recipient);
    return queue?.peek();
  }

  /**
   * Get queue size for a recipient
   */
  size(recipient: string): number {
    const queue = this.queues.get(recipient);
    return queue?.size() ?? 0;
  }

  /**
   * Check if a recipient's queue is empty
   */
  isEmpty(recipient: string): boolean {
    return this.size(recipient) === 0;
  }

  /**
   * Clear a recipient's queue
   */
  clear(recipient: string): void {
    const queue = this.queues.get(recipient);
    if (queue) {
      queue.clear();
    }
  }

  /**
   * Clear all queues
   */
  clearAll(): void {
    for (const queue of this.queues.values()) {
      queue.clear();
    }
  }

  /**
   * Get all receiver names
   */
  getReceivers(): string[] {
    return [...this.receivers];
  }

  /**
   * Cleanup and destroy the message queue
   */
  async cleanup(): Promise<void> {
    this.clearAll();
    this.queues.clear();
    this.receivers = [];
  }
}

/**
 * Simple async queue implementation
 */
class AsyncQueue<T> {
  private queue: T[] = [];
  private resolvers: Array<(value: T | undefined) => void> = [];

  async push(item: T): Promise<void> {
    const resolver = this.resolvers.shift();
    if (resolver) {
      resolver(item);
    } else {
      this.queue.push(item);
    }
  }

  async shift(): Promise<T | undefined> {
    const item = this.queue.shift();
    if (item !== undefined) {
      return item;
    }

    // No items available, wait for one
    return new Promise<T | undefined>(resolve => {
      this.resolvers.push(resolve);
    });
  }

  peek(): T | undefined {
    return this.queue[0];
  }

  size(): number {
    return this.queue.length + this.resolvers.length;
  }

  clear(): void {
    this.queue = [];
    // Reject pending resolvers
    for (const resolver of this.resolvers) {
      resolver(undefined);
    }
    this.resolvers = [];
  }
}

export default MessageQueue;