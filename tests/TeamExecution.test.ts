import { describe, expect, it } from 'vitest';
import { executeTeamItems, normalizeTeamExecutionPolicy } from '../src/core/team/index.js';

describe('team execution policy', () => {
  it('runs with bounded concurrency and preserves declaration order', async () => {
    let active = 0;
    let maximumActive = 0;
    const outcomes = await executeTeamItems(
      [1, 2, 3, 4].map(value => ({
        key: `member-${value}`,
        execute: async () => {
          active += 1;
          maximumActive = Math.max(maximumActive, active);
          await new Promise(resolve => setTimeout(resolve, 5));
          active -= 1;
          return value;
        },
      })),
      { mode: 'parallel', maxConcurrency: 2 }
    );

    expect(maximumActive).toBe(2);
    expect(outcomes.map(outcome => outcome.key)).toEqual(['member-1', 'member-2', 'member-3', 'member-4']);
    expect(outcomes.map(outcome => outcome.value)).toEqual([1, 2, 3, 4]);
  });

  it('stops unscheduled sequential work in fail-fast mode', async () => {
    const executed: string[] = [];
    const outcomes = await executeTeamItems([
      { key: 'first', execute: async () => { executed.push('first'); return 1; } },
      { key: 'broken', execute: async () => { executed.push('broken'); throw new Error('broken member'); } },
      { key: 'never', execute: async () => { executed.push('never'); return 3; } },
    ], { mode: 'sequential', failureMode: 'fail_fast' });

    expect(executed).toEqual(['first', 'broken']);
    expect(outcomes.map(outcome => outcome.status)).toEqual(['completed', 'failed', 'skipped']);
    expect(outcomes[1].error).toBe('broken member');
  });

  it('rejects malformed policies before work starts', () => {
    expect(() => normalizeTeamExecutionPolicy({ maxConcurrency: 0 })).toThrow('positive integer');
    expect(() => normalizeTeamExecutionPolicy({ minimumSuccessfulMembers: 0 })).toThrow('positive integer');
  });
});
