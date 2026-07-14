import { describe, expect, it } from 'vitest';
import type { LogEvent } from '../src/core/logging/events.js';
import type { EventTransport } from '../src/core/logging/transport.js';
import {
  BatchingTransport,
  CompositeTransport,
  FilteredTransport,
} from '../src/core/logging/transport.js';

const event = (id: string, type: LogEvent['type'] = 'info'): LogEvent => ({
  id,
  type,
  namespace: 'test',
  message: id,
  timestamp: Date.now(),
});

class RecordingTransport implements EventTransport {
  batches: LogEvent[][] = [];
  flushes = 0;
  closed = false;
  failures = 0;

  async send(events: LogEvent[]): Promise<void> {
    if (this.failures > 0) {
      this.failures -= 1;
      throw new Error('transport unavailable');
    }
    this.batches.push([...events]);
  }

  async flush(): Promise<void> {
    this.flushes += 1;
  }

  async close(): Promise<void> {
    this.closed = true;
  }
}

describe('logging transports', () => {
  it('fans out events and applies filters before delivery', async () => {
    const first = new RecordingTransport();
    const second = new RecordingTransport();
    const transport = new FilteredTransport(new CompositeTransport([first, second]), {
      minLevel: 'warning',
    });

    await transport.send([event('debug', 'debug'), event('warning', 'warning')]);
    await transport.close();

    expect(first.batches.flat().map((item) => item.id)).toEqual(['warning']);
    expect(second.batches.flat().map((item) => item.id)).toEqual(['warning']);
    expect(first.closed).toBe(true);
    expect(second.closed).toBe(true);
  });

  it('restores a failed batch and retries it on the next flush', async () => {
    const target = new RecordingTransport();
    target.failures = 1;
    const transport = new BatchingTransport(target, 10, 0);

    await transport.send([event('one')]);
    await expect(transport.flush()).rejects.toThrow('transport unavailable');
    await transport.flush();
    await transport.close();

    expect(target.batches.flat().map((item) => item.id)).toEqual(['one']);
    expect(target.closed).toBe(true);
  });

  it('flushes automatically at the configured batch size', async () => {
    const target = new RecordingTransport();
    const transport = new BatchingTransport(target, 2, 0);

    await transport.send([event('one')]);
    expect(target.batches).toHaveLength(0);
    await transport.send([event('two')]);

    expect(target.batches).toHaveLength(1);
    expect(target.batches[0].map((item) => item.id)).toEqual(['one', 'two']);
    await transport.close();
  });
});
