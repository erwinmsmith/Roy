import { afterEach, describe, expect, it, vi } from 'vitest';
import { AgentToolExecutionLoop } from '../src/core/tools/executionLoop.js';

describe('AgentToolExecutionLoop', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('executes sequential observe/plan rounds and rejects duplicate calls', async () => {
    const loop = new AgentToolExecutionLoop({
      maxRounds: 4,
      maxCalls: 6,
      maxConsecutiveFailures: 2,
      maxWallClockMs: 10_000,
    });
    const summary = await loop.run({
      task: 'Search then fetch a source',
      initialPlans: [{ toolName: 'web.search', params: { query: 'fetch api' }, reason: 'discover', groundingRequired: true }],
      execute: async plan => ({ success: true, result: { tool: plan.toolName } }),
      planNext: async context => context.round === 1
        ? [{ toolName: 'web.fetch', params: { url: 'https://example.com' }, reason: 'read', groundingRequired: true }]
        : [{ toolName: 'web.fetch', params: { url: 'https://example.com' }, reason: 'repeat', groundingRequired: true }],
    });

    expect(summary.rounds).toHaveLength(2);
    expect(summary.totalCalls).toBe(2);
    expect(summary.stopReason).toBe('duplicate_plan');
  });

  it('stops after the configured consecutive failure limit', async () => {
    const loop = new AgentToolExecutionLoop({
      maxRounds: 4,
      maxCalls: 8,
      maxConsecutiveFailures: 2,
      maxWallClockMs: 10_000,
    });
    const summary = await loop.run({
      task: 'failure case',
      initialPlans: [
        { toolName: 'web.fetch', params: { url: 'https://example.com/1' }, reason: 'one', groundingRequired: true },
        { toolName: 'web.fetch', params: { url: 'https://example.com/2' }, reason: 'two', groundingRequired: true },
        { toolName: 'web.fetch', params: { url: 'https://example.com/3' }, reason: 'three', groundingRequired: true },
      ],
      execute: async () => ({ success: false, error: 'network failed' }),
      planNext: async () => [],
    });

    expect(summary.totalCalls).toBe(2);
    expect(summary.failedCalls).toBe(2);
    expect(summary.stopReason).toBe('consecutive_failures');
  });

  it('checks the global wall-clock deadline between calls in the same round', async () => {
    let now = 1_000;
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    const loop = new AgentToolExecutionLoop({
      maxRounds: 2,
      maxCalls: 4,
      maxConsecutiveFailures: 2,
      maxWallClockMs: 100,
    });
    const execute = vi.fn(async () => {
      now += 101;
      return { success: true, result: 'completed slowly' };
    });
    const summary = await loop.run({
      task: 'bounded task',
      initialPlans: [
        { toolName: 'web.fetch', params: { url: 'https://example.com/one' }, reason: 'one', groundingRequired: true },
        { toolName: 'web.fetch', params: { url: 'https://example.com/two' }, reason: 'two', groundingRequired: true },
      ],
      execute,
      planNext: async () => [],
    });

    expect(execute).toHaveBeenCalledTimes(1);
    expect(summary.totalCalls).toBe(1);
    expect(summary.stopReason).toBe('max_wall_clock');
  });

  it('supports domain-aware fingerprints for equivalent tool calls', async () => {
    const loop = new AgentToolExecutionLoop({
      maxRounds: 3,
      maxCalls: 4,
      maxConsecutiveFailures: 2,
      maxWallClockMs: 10_000,
    });
    const execute = vi.fn(async () => ({ success: true, result: 'page' }));
    const summary = await loop.run({
      task: 'fetch one document',
      initialPlans: [{
        toolName: 'web.fetch', params: { url: 'https://example.com/docs#first' }, reason: 'first', groundingRequired: true,
      }],
      execute,
      planNext: async () => [{
        toolName: 'web.fetch', params: { url: 'https://example.com/docs#second' }, reason: 'same document', groundingRequired: true,
      }],
      fingerprint: plan => `${plan.toolName}:${String(plan.params.url).split('#')[0]}`,
    });

    expect(execute).toHaveBeenCalledTimes(1);
    expect(summary.stopReason).toBe('duplicate_plan');
  });
});
