import type { EnqueueMessageInput, MessageQueue, RuntimeMessage } from './types.js';
import type { MessageWorker } from './worker.js';

export interface MessageSchedulerOptions {
  pollIntervalMs?: number;
  concurrency?: number;
}

export class MessageScheduler {
  private workers: MessageWorker[] = [];
  private running = false;
  private loop?: Promise<void>;
  private readonly pollIntervalMs: number;

  constructor(
    private readonly queue: MessageQueue,
    options: MessageSchedulerOptions = {}
  ) {
    this.pollIntervalMs = options.pollIntervalMs ?? 20;
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.loop = this.runLoop();
  }

  async stop(): Promise<void> {
    this.running = false;
    await this.loop;
    this.loop = undefined;
  }

  registerWorker(worker: MessageWorker): void {
    this.workers.push(worker);
  }

  async enqueue<TPayload>(message: EnqueueMessageInput<TPayload>): Promise<RuntimeMessage<TPayload>> {
    return this.queue.enqueue(message);
  }

  async processNext(): Promise<boolean> {
    const message = await this.queue.dequeue({ readyOnly: true });
    if (!message) return false;

    const worker = this.workers.find(item => item.accepts(message.kind, message));
    if (!worker) {
      await this.queue.fail(message.id, new Error(`No worker accepts ${message.kind} for ${message.to}`));
      return true;
    }

    try {
      await worker.handle(message);
      await this.queue.ack(message.id);
    } catch (error) {
      await this.queue.fail(message.id, error instanceof Error ? error : new Error(String(error)));
    }

    return true;
  }

  private async runLoop(): Promise<void> {
    while (this.running) {
      const processed = await this.processNext();
      if (!processed) {
        await new Promise(resolve => setTimeout(resolve, this.pollIntervalMs));
      }
    }
  }
}

export default MessageScheduler;

