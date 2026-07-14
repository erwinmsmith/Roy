import { describe, expect, it } from 'vitest';
import { AsyncioExecutor } from '../src/core/executor/Executor.js';

describe('AsyncioExecutor', () => {
  it('enforces the configured concurrency limit', async () => {
    const executor = new AsyncioExecutor({ maxConcurrentActivities: 2 });
    let active = 0;
    let peak = 0;

    const results = await executor.executeAll(
      Array.from({ length: 6 }, (_, value) => async () => {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise(resolve => setTimeout(resolve, 5));
        active -= 1;
        return value;
      })
    );

    expect(peak).toBe(2);
    expect(results.map(result => result.value)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('retries failed activities with attempt metadata', async () => {
    const executor = new AsyncioExecutor({
      retryPolicy: { maxAttempts: 3, initialDelayMs: 0, backoffMultiplier: 2 },
    });
    let attempts = 0;

    const result = await executor.execute(() => {
      attempts += 1;
      if (attempts < 3) throw new Error('temporary failure');
      return 'done';
    });

    expect(result).toMatchObject({ success: true, value: 'done', metadata: { attempts: 3 } });
  });

  it('returns a timeout failure without leaking the timer', async () => {
    const executor = new AsyncioExecutor({ timeoutMs: 5 });
    const result = await executor.execute(
      () => new Promise(resolve => setTimeout(() => resolve('late'), 30))
    );

    expect(result).toMatchObject({ success: false, error: 'Activity timeout' });
  });

  it('rejects queued and future activities after cleanup', async () => {
    const executor = new AsyncioExecutor({ maxConcurrentActivities: 1 });
    let releaseFirst!: () => void;
    const first = executor.execute(() => new Promise<string>(resolve => {
      releaseFirst = () => resolve('first');
    }));
    await new Promise(resolve => setTimeout(resolve, 0));
    const queued = executor.execute(() => 'queued');

    await executor.cleanup();
    releaseFirst();

    expect(await first).toMatchObject({ success: true, value: 'first' });
    expect(await queued).toMatchObject({ success: false, error: 'Executor is closed' });
    expect(await executor.execute(() => 'late')).toMatchObject({
      success: false,
      error: 'Executor is closed',
      metadata: { attempts: 0 },
    });
  });
});
