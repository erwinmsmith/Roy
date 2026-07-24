import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type { ToolCallRecord } from '../src/core/runtime/Runtime.js';
import type { PlannedToolCall } from '../src/core/tools/planner.js';
import type { WorkspaceRuntimeConfig } from '../src/core/memory/workspace.js';

describe('Runtime external wall-clock deadline', () => {
  it('clamps exploration, execution, finalization, and tool-loop budgets', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-deadline-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'runtime-deadline',
      workspaceCwd: cwd,
      wallClockLimitMs: 120_000,
    });

    const workspaceConfig = (
      runtime as unknown as { workspaceRuntimeConfig: WorkspaceRuntimeConfig }
    ).workspaceRuntimeConfig;
    const rootSteps = workspaceConfig.delegation.rootSteps;
    expect(rootSteps.maxWallClockMs).toBe(120_000);
    expect(rootSteps.executionReserveMs).toBeLessThanOrEqual(48_000);
    expect(rootSteps.finalizationReserveMs).toBeLessThanOrEqual(18_000);
    expect(workspaceConfig.tools.executionLoop.maxWallClockMs).toBeLessThanOrEqual(
      120_000 - rootSteps.finalizationReserveMs
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'runtime.wall_clock_limit.applied',
      data: expect.objectContaining({
        requestedWallClockMs: 120_000,
        appliedWallClockMs: 120_000,
      }),
    }));
    await runtime.shutdown();
  });

  it('rejects invalid external deadlines', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-invalid-deadline-'));
    const runtime = new Runtime();
    await expect(runtime.initialize({
      sessionId: 'runtime-invalid-deadline',
      workspaceCwd: cwd,
      wallClockLimitMs: 999,
    })).rejects.toThrow('at least 1000ms');
  });

  it('skips an unchanged failed verification but retries after a real mutation', () => {
    const runtime = new Runtime() as unknown as {
      shouldSkipUnchangedFailedVerification(
        plan: PlannedToolCall,
        priorCalls: ToolCallRecord[]
      ): boolean;
    };
    const verification: PlannedToolCall = {
      toolName: 'shell.exec',
      params: { command: 'pytest -q' },
      reason: 'Run verification.',
      groundingRequired: true,
    };
    const failed: ToolCallRecord = {
      toolName: 'shell.exec',
      params: { command: 'pytest -q' },
      success: false,
      error: 'tests failed',
    };

    expect(runtime.shouldSkipUnchangedFailedVerification(verification, [failed])).toBe(true);
    expect(runtime.shouldSkipUnchangedFailedVerification(verification, [
      failed,
      {
        toolName: 'fs.replace',
        params: { path: 'src/app.py', oldText: 'broken', newText: 'fixed' },
        success: true,
      },
    ])).toBe(false);
  });
});
