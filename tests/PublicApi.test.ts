import { describe, expect, it } from 'vitest';
import {
  BudgetMarket,
  Runtime,
  TeamRegistry,
  ToMDelegationPlanner,
  WorkspaceMemoryManager,
  executeTeamItems,
} from '../src/index.js';

describe('public package API', () => {
  it('exports the runtime, team policy, budget, and memory entry points', () => {
    expect(Runtime).toBeTypeOf('function');
    expect(TeamRegistry).toBeTypeOf('function');
    expect(ToMDelegationPlanner).toBeTypeOf('function');
    expect(WorkspaceMemoryManager).toBeTypeOf('function');
    expect(BudgetMarket).toBeTypeOf('function');
    expect(executeTeamItems).toBeTypeOf('function');
  });
});
