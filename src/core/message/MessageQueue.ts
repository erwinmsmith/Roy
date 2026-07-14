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
  private closed = false;

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
    this.assertOpen();
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
      queue.close();
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
    this.assertOpen();
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
    this.assertOpen();
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

    while (!queue.isClosed()) {
      const message = await queue.shift();
      if (message) {
        yield message;
      } else if (queue.isClosed()) {
        return;
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
    if (this.closed) return;
    this.closed = true;
    for (const queue of this.queues.values()) queue.close();
    this.queues.clear();
    this.receivers = [];
  }

  isClosed(): boolean {
    return this.closed;
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('Message queue is closed');
  }
}

/**
 * Simple async queue implementation
 */
class AsyncQueue<T> {
  private queue: T[] = [];
  private resolvers: Array<(value: T | undefined) => void> = [];
  private closed = false;

  async push(item: T): Promise<void> {
    if (this.closed) throw new Error('Queue is closed');
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
    if (this.closed) return undefined;

    // No items available, wait for one
    return new Promise<T | undefined>(resolve => {
      this.resolvers.push(resolve);
    });
  }

  peek(): T | undefined {
    return this.queue[0];
  }

  size(): number {
    return this.queue.length;
  }

  clear(): void {
    this.queue = [];
    for (const resolver of this.resolvers) {
      resolver(undefined);
    }
    this.resolvers = [];
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.clear();
  }

  isClosed(): boolean {
    return this.closed;
  }
}

export default MessageQueue;
