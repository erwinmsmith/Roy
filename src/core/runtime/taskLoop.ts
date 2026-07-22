import type { RootExecutionLoopState, RootExecutionTreeState } from './executionTree.js';

export interface RootTaskLoopConfig {
  maxIterations: number;
  maxWallClockMs: number;
  maxStalledIterations: number;
  reserveFinalSteps?: number;
}

export interface RootTaskLoopGuard {
  continue: boolean;
  reason: 'continue' | 'max_iterations' | 'timeout' | 'stalled';
  remainingSteps: number;
  elapsedMs: number;
}

export class RootTaskLoopController {
  constructor(private readonly config: RootTaskLoopConfig) {}

  getState(tree: RootExecutionTreeState): RootExecutionLoopState {
    return {
      ...tree.loop,
      maxIterations: this.config.maxIterations,
      maxWallClockMs: this.config.maxWallClockMs,
      maxStalledIterations: this.config.maxStalledIterations,
      elapsedMs: Math.max(0, Date.now() - tree.createdAt),
    };
  }

  evaluate(tree: RootExecutionTreeState): RootTaskLoopGuard {
    const reserveFinalSteps = Math.max(1, this.config.reserveFinalSteps ?? 1);
    const elapsedMs = Math.max(0, Date.now() - tree.createdAt);
    const remainingSteps = Math.max(0, this.config.maxIterations - tree.steps.length);
    if (elapsedMs >= this.config.maxWallClockMs) {
      return { continue: false, reason: 'timeout', remainingSteps, elapsedMs };
    }
    if (tree.loop.stalledIterations >= this.config.maxStalledIterations) {
      return { continue: false, reason: 'stalled', remainingSteps, elapsedMs };
    }
    if (remainingSteps <= reserveFinalSteps) {
      return { continue: false, reason: 'max_iterations', remainingSteps, elapsedMs };
    }
    return { continue: true, reason: 'continue', remainingSteps, elapsedMs };
  }
}
