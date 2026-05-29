import { describe, it, expect } from 'vitest';
import Runtime from '../src/core/runtime/Runtime.js';
import { InMemoryMessageQueue } from '../src/core/queue/index.js';

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
});

describe('Runtime queue state', () => {
  it('emits queue transition events and exposes queue state', async () => {
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'runtime-queue-test',
      fsmEnabled: false,
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

    await runtime.shutdown();
  });
});

