import { describe, expect, it } from 'vitest';
import { MessageQueue } from '../src/core/message/MessageQueue.js';

describe('agent MessageQueue lifecycle', () => {
  it('does not count pending receivers as queued messages', async () => {
    const queue = new MessageQueue(['agent']);
    const pending = queue.receive('agent');

    expect(queue.size('agent')).toBe(0);
    expect(queue.isEmpty('agent')).toBe(true);
    await queue.send('root', 'agent', 'work');
    expect((await pending)?.content).toBe('work');
    expect(queue.size('agent')).toBe(0);
  });

  it('unblocks receivers on cleanup and cannot be reopened', async () => {
    const queue = new MessageQueue(['agent']);
    const pending = queue.receive('agent');

    await queue.cleanup();

    await expect(pending).resolves.toBeUndefined();
    expect(queue.isClosed()).toBe(true);
    expect(() => queue.addReceiver('late')).toThrow('Message queue is closed');
    await expect(queue.send('root', 'late', 'work')).rejects.toThrow('Message queue is closed');
  });
});
