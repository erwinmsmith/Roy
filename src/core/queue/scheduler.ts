import type { EnqueueMessageInput, MessageQueue, RuntimeMessage } from './types.js';
import type { MessageWorker } from './worker.js';

export interface MessageSchedulerOptions {
  pollIntervalMs?: number;
  concurrency?: number;
  retryDelayMs?: number;
}

export class MessageScheduler {
  private workers: MessageWorker[] = [];
  private running = false;
  private loops: Promise<void>[] = [];
  private readonly pollIntervalMs: number;
  private readonly concurrency: number;
  private readonly retryDelayMs: number;

  constructor(
    private readonly queue: MessageQueue,
    options: MessageSchedulerOptions = {}
  ) {
    this.pollIntervalMs = options.pollIntervalMs ?? 20;
    this.concurrency = options.concurrency ?? 1;
    this.retryDelayMs = options.retryDelayMs ?? 100;
    if (!Number.isInteger(this.concurrency) || this.concurrency < 1) {
      throw new Error('Scheduler concurrency must be a positive integer');
    }
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.loops = Array.from({ length: this.concurrency }, (_, index) => this.runLoop(index));
  }

  async stop(): Promise<void> {
    this.running = false;
    await Promise.all(this.loops);
    this.loops = [];
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
      const retryCount = message.metadata?.retryCount ?? 0;
      const maxRetries = message.metadata?.maxRetries ?? 0;
      if (retryCount < maxRetries) {
        const availableAt = Date.now() + this.retryDelayMs * (2 ** retryCount);
        await this.queue.retry(message.id, availableAt);
      } else {
        await this.queue.fail(message.id, error instanceof Error ? error : new Error(String(error)));
      }
    }

    return true;
  }

  private async runLoop(_workerIndex: number): Promise<void> {
    while (this.running) {
      const processed = await this.processNext();
      if (!processed) {
        await new Promise(resolve => setTimeout(resolve, this.pollIntervalMs));
      }
    }
  }
}

export default MessageScheduler;
