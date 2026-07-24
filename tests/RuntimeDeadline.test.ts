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
      cachedToolPlanDecision(
        plan: PlannedToolCall,
        priorCalls: ToolCallRecord[]
      ): { skip: boolean; reason?: string };
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

    expect(runtime.cachedToolPlanDecision(verification, [failed])).toMatchObject({
      skip: true,
      reason: 'cached_failed_verification_without_later_mutation',
    });
    expect(runtime.cachedToolPlanDecision({
      ...verification,
      params: { command: 'pytest -q', timeoutMs: 60_000 },
    }, [{
      ...failed,
      params: { command: 'pytest -q', timeoutMs: 30_000 },
    }])).toMatchObject({
      skip: true,
      reason: 'cached_failed_verification_without_later_mutation',
    });
    expect(runtime.cachedToolPlanDecision(verification, [
      failed,
      {
        toolName: 'fs.replace',
        params: { path: 'src/app.py', oldText: 'broken', newText: 'fixed' },
        success: true,
      },
    ])).toEqual({ skip: false });
  });

  it('invalidates cached inspection only when a relevant workspace path changes', () => {
    const runtime = new Runtime() as unknown as {
      cachedToolPlanDecision(
        plan: PlannedToolCall,
        priorCalls: ToolCallRecord[]
      ): { skip: boolean; reason?: string };
    };
    const configRead: PlannedToolCall = {
      toolName: 'fs.read',
      params: { path: 'configs/public_audit.yml' },
      reason: 'Read configuration.',
      groundingRequired: true,
    };
    const sourceRead: PlannedToolCall = {
      toolName: 'fs.read',
      params: { path: 'src/dq_audit/audit.py', startLine: 580, endLine: 630 },
      reason: 'Read failing source.',
      groundingRequired: true,
    };
    const priorReads: ToolCallRecord[] = [
      { ...configRead, success: true, result: { content: 'config' } },
      { ...sourceRead, success: true, result: { content: 'broken source' } },
      {
        toolName: 'fs.replace',
        params: {
          path: 'src/dq_audit/audit.py',
          oldText: 'broken',
          newText: 'fixed',
        },
        success: true,
      },
    ];

    expect(runtime.cachedToolPlanDecision(configRead, priorReads)).toMatchObject({
      skip: true,
      reason: 'cached_file_read_still_current',
    });
    expect(runtime.cachedToolPlanDecision(sourceRead, priorReads)).toEqual({
      skip: false,
    });
  });

  it('skips a differently paginated read when a full unchanged file is already cached', () => {
    const runtime = new Runtime() as unknown as {
      cachedToolPlanDecision(
        plan: PlannedToolCall,
        priorCalls: ToolCallRecord[]
      ): { skip: boolean; reason?: string };
    };
    const completed: ToolCallRecord = {
      toolName: 'fs.read',
      params: { path: 'rules/public_expectations.yml', maxBytes: 4_000 },
      success: true,
      result: {
        path: 'rules/public_expectations.yml',
        truncated: false,
        startLine: 1,
        endLine: 44,
        totalLines: 44,
      },
    };

    expect(runtime.cachedToolPlanDecision({
      toolName: 'fs.read',
      params: { path: './rules/public_expectations.yml', startLine: 1, endLine: 100 },
      reason: 'Read the same unchanged rules again.',
      groundingRequired: true,
    }, [completed])).toMatchObject({
      skip: true,
      reason: 'cached_file_read_still_current',
    });
  });

  it('does not repeat an unchanged failed acceptance audit before a new repair', () => {
    const runtime = new Runtime() as unknown as {
      shouldRunRootAcceptanceAudit(
        closure: {
          mutationApplied: boolean;
          verificationPassed: boolean;
        },
        audit: { performed: boolean; passed: boolean } | undefined,
        required: boolean,
        invalidated: boolean
      ): boolean;
      shouldRequireFreshAcceptanceMutation(
        closure: {
          acceptanceAuditPerformed: boolean;
          acceptanceAuditPassed: boolean;
        },
        invalidated: boolean
      ): boolean;
    };
    const closure = { mutationApplied: true, verificationPassed: true };
    const failedAudit = { performed: true, passed: false };

    expect(runtime.shouldRunRootAcceptanceAudit(
      closure,
      failedAudit,
      true,
      false
    )).toBe(false);
    expect(runtime.shouldRunRootAcceptanceAudit(
      closure,
      failedAudit,
      true,
      true
    )).toBe(true);
    expect(runtime.shouldRunRootAcceptanceAudit(
      closure,
      undefined,
      true,
      false
    )).toBe(true);
    expect(runtime.shouldRequireFreshAcceptanceMutation({
      acceptanceAuditPerformed: true,
      acceptanceAuditPassed: false,
    }, false)).toBe(true);
    expect(runtime.shouldRequireFreshAcceptanceMutation({
      acceptanceAuditPerformed: true,
      acceptanceAuditPassed: false,
    }, true)).toBe(false);
  });

  it('keeps fenced output contracts attached to acceptance requirements', () => {
    const runtime = new Runtime() as unknown as {
      extractTaskAcceptanceItems(task: string): string[];
    };
    const items = runtime.extractTaskAcceptanceItems([
      'The report must contain exactly these section headings:',
      '```markdown',
      '# Data Quality Audit Summary',
      '## Overall',
      '## Dataset Counts',
      '```',
    ].join('\n'));

    expect(items).toContain(
      'The report must contain exactly these section headings: # Data Quality Audit Summary | ## Overall | ## Dataset Counts'
    );
  });

  it('carries failed acceptance evidence into the next repair task', () => {
    const runtime = new Runtime() as unknown as {
      buildRootExecutionRepairTask(
        task: string,
        prior: Record<string, unknown>,
        attempt: number
      ): string;
    };
    const repairTask = runtime.buildRootExecutionRepairTask('Repair the project.', {
      toolCalls: [],
      acceptanceAudit: {
        required: true,
        performed: true,
        passed: false,
        items: [{
          id: 'acceptance_04',
          status: 'failed',
          evidence: 'The generated manifest still references the removed package.',
        }],
      },
      evidence: { toolResultSummary: 'pytest reported one failing assertion.' },
      warnings: [],
    }, 2);

    expect(repairTask).toContain(
      'acceptance_04 [failed]: The generated manifest still references the removed package.'
    );
  });
});
