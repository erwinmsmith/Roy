import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SignalBus, signalBus, type Signal } from '../src/core/executor/SignalBus.js';

describe('SignalBus', () => {
  let signalBus: SignalBus;

  beforeEach(() => {
    signalBus = new SignalBus();
  });

  afterEach(() => {
    signalBus.cleanup();
  });

  describe('signal()', () => {
    it('should emit signal and notify handlers', async () => {
      const received: Signal[] = [];
      const handler = vi.fn((signal: Signal) => {
        received.push(signal);
      });

      signalBus.onSignal('test')(handler);
      await signalBus.signal({
        name: 'test',
        payload: 'hello',
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledTimes(1);
      expect(received[0].payload).toBe('hello');
    });

    it('should emit signal with metadata', async () => {
      const handler = vi.fn(() => {});
      signalBus.onSignal('test')(handler);

      await signalBus.signal({
        name: 'test',
        payload: 'data',
        metadata: { key: 'value' },
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'test',
          payload: 'data',
          metadata: { key: 'value' },
        })
      );
    });
  });

  describe('emit()', () => {
    it('should create and emit signal', async () => {
      const handler = vi.fn(() => {});
      signalBus.onSignal('custom')(handler);

      await signalBus.emit('custom', 'payload', 'Test description');

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'custom',
          payload: 'payload',
          description: 'Test description',
        })
      );
    });
  });

  describe('waitForSignal()', () => {
    it('should resolve when signal is emitted', async () => {
      const promise = signalBus.waitForSignal<string>('test-signal', 5000);

      // Emit after a short delay
      setTimeout(() => {
        signalBus.signal({
          name: 'test-signal',
          payload: 'resolved',
          timestamp: Date.now(),
        });
      }, 50);

      const result = await promise;
      expect(result).toBe('resolved');
    });

    it('should timeout if signal not received', async () => {
      const promise = signalBus.waitForSignal('timeout-test', 100);

      await expect(promise).rejects.toThrow('Timeout');
    });
  });

  describe('subscribe()', () => {
    it('should return subscription id', () => {
      const id = signalBus.subscribe('test', vi.fn());
      expect(typeof id).toBe('string');
      expect(id.length).toBe(36); // UUID format
    });

    it('should call handler on signal', async () => {
      const handler = vi.fn();
      signalBus.subscribe('sub-test', handler);

      await signalBus.signal({
        name: 'sub-test',
        payload: 'data',
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('should support filter option', async () => {
      const handler = vi.fn();
      signalBus.subscribe('filtered', handler, {
        filter: (signal) => signal.payload === 'allowed',
      });

      await signalBus.signal({
        name: 'filtered',
        payload: 'blocked',
        timestamp: Date.now(),
      });

      await signalBus.signal({
        name: 'filtered',
        payload: 'allowed',
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('should support once option', async () => {
      const handler = vi.fn();
      signalBus.subscribe('once-test', handler, { once: true });

      await signalBus.signal({
        name: 'once-test',
        payload: 'first',
        timestamp: Date.now(),
      });

      await signalBus.signal({
        name: 'once-test',
        payload: 'second',
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe('unsubscribe()', () => {
    it('should stop calling handler after unsubscribe', async () => {
      const handler = vi.fn();
      const id = signalBus.subscribe('remove-test', handler);

      await signalBus.signal({
        name: 'remove-test',
        payload: 'first',
        timestamp: Date.now(),
      });

      signalBus.unsubscribe(id);

      await signalBus.signal({
        name: 'remove-test',
        payload: 'second',
        timestamp: Date.now(),
      });

      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe('clearHandlers()', () => {
    it('should clear handlers for specific signal', async () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();
      signalBus.onSignal('clear-me')(handler1);
      signalBus.onSignal('clear-me')(handler2);

      await signalBus.signal({
        name: 'clear-me',
        payload: 'first',
        timestamp: Date.now(),
      });

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);

      signalBus.clearHandlers('clear-me');

      await signalBus.signal({
        name: 'clear-me',
        payload: 'second',
        timestamp: Date.now(),
      });

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it('should clear all handlers when no name provided', async () => {
      const handler = vi.fn();
      signalBus.onSignal('clear1')(handler);
      signalBus.onSignal('clear2')(handler);

      signalBus.clearHandlers();

      await signalBus.signal({ name: 'clear1', timestamp: Date.now() });
      await signalBus.signal({ name: 'clear2', timestamp: Date.now() });

      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('getSignalNames()', () => {
    it('should return registered signal names', () => {
      signalBus.onSignal('signal-a')(vi.fn());
      signalBus.onSignal('signal-b')(vi.fn());

      const names = signalBus.getSignalNames();
      expect(names).toContain('signal-a');
      expect(names).toContain('signal-b');
    });
  });

  describe('getSubscriptionCount()', () => {
    it('should return active subscription count', () => {
      expect(signalBus.getSubscriptionCount()).toBe(0);

      signalBus.subscribe('count-test', vi.fn());
      signalBus.subscribe('count-test', vi.fn());

      expect(signalBus.getSubscriptionCount()).toBe(2);
    });
  });
});

describe('SignalBus Singleton', () => {
  it('should export a singleton instance', () => {
    expect(signalBus).toBeInstanceOf(SignalBus);
  });
});