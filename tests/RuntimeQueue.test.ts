import { describe, it, expect } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { InMemoryMessageQueue, MessageScheduler } from '../src/core/queue/index.js';
import type { MessageWorker, RuntimeMessage } from '../src/core/queue/index.js';

describe('InMemoryMessageQueue', () => {
  it('tracks message lifecycle and stats', async () => {
    const transitions: string[] = [];
    const queue = new InMemoryMessageQueue(transition => transitions.push(transition.type));

    const message = await queue.enqueue({
      kind: 'agent.task',
      sessionId: 'queue-test',
      from: 'root',
      to: 'agent_researcher_001',
      payload: { task: 'inspect runtime' },
      priority: 'high',
    });

    expect((await queue.getStats()).pending).toBe(1);

    const next = await queue.dequeue({ to: 'agent_researcher_001' });
    expect(next?.id).toBe(message.id);
    expect(next?.status).toBe('processing');

    await queue.ack(message.id);
    const stats = await queue.getStats();
    expect(stats.completed).toBe(1);
    expect(stats.pending).toBe(0);
    expect(transitions).toEqual(['message.enqueued', 'message.processing', 'message.completed']);
  });

  it('enforces lifecycle transitions and records retries', async () => {
    const transitions: string[] = [];
    const queue = new InMemoryMessageQueue(transition => transitions.push(transition.type));
    const message = await queue.enqueue({
      kind: 'agent.task',
      sessionId: 'retry-test',
      from: 'root',
      to: 'worker',
      payload: {},
      metadata: { maxRetries: 2 },
    });

    await expect(queue.ack(message.id)).rejects.toThrow('while status is pending');
    await queue.dequeue();
    await queue.retry(message.id, Date.now() - 1);
    expect((await queue.getMessage(message.id))?.metadata?.retryCount).toBe(1);
    await queue.dequeue();
    await queue.ack(message.id);
    await expect(queue.cancel(message.id)).rejects.toThrow('while status is completed');
    expect(transitions).toContain('message.retried');
  });

  it('uses scheduler concurrency and retries transient worker failures', async () => {
    const queue = new InMemoryMessageQueue();
    let active = 0;
    let peak = 0;
    const attempts = new Map<string, number>();
    const completed: string[] = [];
    const worker: MessageWorker = {
      id: 'test-worker',
      accepts: () => true,
      async handle(message: RuntimeMessage) {
        active += 1;
        peak = Math.max(peak, active);
        const attempt = (attempts.get(message.id) ?? 0) + 1;
        attempts.set(message.id, attempt);
        await new Promise(resolve => setTimeout(resolve, 5));
        active -= 1;
        if (attempt === 1) throw new Error('transient');
        completed.push(message.id);
      },
    };
    const scheduler = new MessageScheduler(queue, {
      concurrency: 2,
      pollIntervalMs: 1,
      retryDelayMs: 0,
    });
    scheduler.registerWorker(worker);

    for (let index = 0; index < 4; index += 1) {
      await queue.enqueue({
        kind: 'agent.task',
        sessionId: 'scheduler-test',
        from: 'root',
        to: 'worker',
        payload: { index },
        metadata: { maxRetries: 1 },
      });
    }

    await scheduler.start();
    const deadline = Date.now() + 2000;
    while ((await queue.getStats()).completed < 4 && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    await scheduler.stop();

    expect((await queue.getStats()).completed).toBe(4);
    expect(completed).toHaveLength(4);
    expect(peak).toBe(2);
    expect([...attempts.values()]).toEqual([2, 2, 2, 2]);
  });
});

describe('Runtime queue state', () => {
  it('emits queue transition events and exposes queue state', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-queue-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'runtime-queue-test',
      fsmEnabled: false,
      workspaceCwd,
    });

    const message = await runtime.enqueueMessage({
      kind: 'user.input',
      sessionId: 'runtime-queue-test',
      from: 'cli',
      to: 'root',
      payload: { text: 'hello' },
    });

    const state = await runtime.getQueueState();
    expect(state.stats.pending).toBe(1);
    expect(state.recent[0].id).toBe(message.id);

    const events = runtime.getEvents().map(event => event.type);
    expect(events).toContain('message.enqueued');
    expect((await runtime.getMemoryState()).queuePath).toContain('.roy/queue');

    await runtime.shutdown();
  });
});
