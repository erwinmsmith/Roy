import type { PlannedToolCall } from './planner.js';
import { isSuccessfulWorkspaceMutationCall } from './executionIntent.js';

export type ToolLoopStopReason = 'completed' | 'max_rounds' | 'max_calls' | 'max_wall_clock' | 'consecutive_failures' | 'duplicate_plan';

export interface ToolLoopCallRecord {
  toolName: string;
  params: Record<string, unknown>;
  reason: string;
  groundingRequired: boolean;
  result?: unknown;
  success: boolean;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

export interface ToolLoopRound {
  round: number;
  plans: PlannedToolCall[];
  calls: ToolLoopCallRecord[];
  startedAt: number;
  completedAt: number;
}

export interface ToolLoopSummary {
  rounds: ToolLoopRound[];
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  stopReason: ToolLoopStopReason;
  startedAt: number;
  completedAt: number;
}

export interface ToolLoopContinuationContext {
  task: string;
  round: number;
  calls: ToolLoopCallRecord[];
  rounds: ToolLoopRound[];
  remainingCalls: number;
}

export interface ToolExecutionLoopOptions {
  maxRounds: number;
  maxCalls: number;
  maxConsecutiveFailures: number;
  maxWallClockMs: number;
}

export class AgentToolExecutionLoop {
  constructor(private readonly options: ToolExecutionLoopOptions) {}

  async run(input: {
    task: string;
    initialPlans: PlannedToolCall[];
    execute: (plan: PlannedToolCall, round: number) => Promise<Omit<ToolLoopCallRecord, keyof PlannedToolCall>>;
    planNext: (context: ToolLoopContinuationContext) => Promise<PlannedToolCall[]>;
    fingerprint?: (plan: PlannedToolCall) => string;
    onRoundStarted?: (round: number, plans: PlannedToolCall[]) => void | Promise<void>;
    onRoundCompleted?: (round: ToolLoopRound) => void | Promise<void>;
  }): Promise<ToolLoopSummary> {
    const startedAt = Date.now();
    const rounds: ToolLoopRound[] = [];
    const fingerprints = new Set<string>();
    let pending = this.uniquePlans(input.initialPlans, fingerprints, input.fingerprint);
    let consecutiveFailures = 0;
    let stopReason: ToolLoopStopReason = 'completed';

    while (pending.length > 0) {
      if (rounds.length >= this.options.maxRounds) {
        stopReason = 'max_rounds';
        break;
      }
      if (Date.now() - startedAt >= this.options.maxWallClockMs) {
        stopReason = 'max_wall_clock';
        break;
      }
      const priorCalls = rounds.reduce((sum, round) => sum + round.calls.length, 0);
      const remainingCalls = this.options.maxCalls - priorCalls;
      if (remainingCalls <= 0) {
        stopReason = 'max_calls';
        break;
      }

      const plans = pending.slice(0, remainingCalls);
      const roundNumber = rounds.length + 1;
      await input.onRoundStarted?.(roundNumber, plans);
      const roundStartedAt = Date.now();
      const calls: ToolLoopCallRecord[] = [];
      let wallClockExceeded = false;
      for (const plan of plans) {
        if (Date.now() - startedAt >= this.options.maxWallClockMs) {
          wallClockExceeded = true;
          break;
        }
        const callStartedAt = Date.now();
        const outcome = await input.execute(plan, roundNumber);
        calls.push({
          ...plan,
          ...outcome,
          startedAt: callStartedAt,
          completedAt: Date.now(),
        });
        consecutiveFailures = outcome.success ? 0 : consecutiveFailures + 1;
        if (Date.now() - startedAt >= this.options.maxWallClockMs) {
          wallClockExceeded = true;
          break;
        }
        if (consecutiveFailures >= this.options.maxConsecutiveFailures) break;
      }
      const round: ToolLoopRound = {
        round: roundNumber,
        plans,
        calls,
        startedAt: roundStartedAt,
        completedAt: Date.now(),
      };
      rounds.push(round);
      await input.onRoundCompleted?.(round);

      if (wallClockExceeded) {
        stopReason = 'max_wall_clock';
        break;
      }
      if (consecutiveFailures >= this.options.maxConsecutiveFailures) {
        stopReason = 'consecutive_failures';
        break;
      }
      const callCount = rounds.reduce((sum, item) => sum + item.calls.length, 0);
      if (callCount >= this.options.maxCalls) {
        stopReason = 'max_calls';
        break;
      }
      const next = await input.planNext({
        task: input.task,
        round: roundNumber,
        calls: rounds.flatMap(item => item.calls),
        rounds,
        remainingCalls: this.options.maxCalls - callCount,
      });
      pending = this.uniquePlans(
        next,
        fingerprints,
        input.fingerprint,
        rounds.flatMap(item => item.calls)
      );
      if (next.length > 0 && pending.length === 0) {
        stopReason = 'duplicate_plan';
        break;
      }
    }

    const allCalls = rounds.flatMap(round => round.calls);
    return {
      rounds,
      totalCalls: allCalls.length,
      successfulCalls: allCalls.filter(call => call.success).length,
      failedCalls: allCalls.filter(call => !call.success).length,
      stopReason,
      startedAt,
      completedAt: Date.now(),
    };
  }

  private uniquePlans(
    plans: PlannedToolCall[],
    fingerprints: Set<string>,
    fingerprint?: (plan: PlannedToolCall) => string,
    calls: ToolLoopCallRecord[] = []
  ): PlannedToolCall[] {
    const unique: PlannedToolCall[] = [];
    for (const plan of plans) {
      const value = fingerprint?.(plan) ?? `${plan.toolName}:${stableStringify(plan.params)}`;
      if (fingerprints.has(value)
        && !this.shellPlanWasInvalidatedByWorkspaceStateChange(
          plan,
          value,
          calls,
          fingerprint
        )) {
        continue;
      }
      fingerprints.add(value);
      unique.push(plan);
    }
    return unique;
  }

  private shellPlanWasInvalidatedByWorkspaceStateChange(
    plan: PlannedToolCall,
    planFingerprint: string,
    calls: ToolLoopCallRecord[],
    fingerprint?: (plan: PlannedToolCall) => string
  ): boolean {
    if (plan.toolName !== 'shell.exec') return false;
    let previousIndex = -1;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const value = fingerprint?.(call)
        ?? `${call.toolName}:${stableStringify(call.params)}`;
      if (value === planFingerprint) {
        previousIndex = index;
        break;
      }
    }
    if (previousIndex < 0) return false;
    return calls.slice(previousIndex + 1).some(call => {
      if (!call.success) return false;
      if ((
        call.toolName === 'fs.write'
        || call.toolName === 'fs.replace'
        || call.toolName === 'fs.synthesize'
      ) && isSuccessfulWorkspaceMutationCall(call)) {
        return true;
      }
      if (call.toolName !== 'shell.exec') return false;
      return this.isEnvironmentRecoveryCommand(
        String(call.params.command ?? '')
      );
    });
  }

  private isEnvironmentRecoveryCommand(command: string): boolean {
    const normalized = command.trim().replace(
      /^(?:(?:cd|pushd)\s+\S+\s*&&\s*)+/i,
      ''
    );
    return /^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|uv)\s+install\b/i.test(
      normalized
    )
      || /^(?:uv\s+sync|poetry\s+install|pdm\s+sync|pipenv\s+sync)\b/i.test(
        normalized
      )
      || /^(?:npm|pnpm|yarn|bun)\s+(?:install|ci)\b/i.test(normalized)
      || /^(?:bundle\s+install|composer\s+install|cargo\s+fetch|go\s+mod\s+download)\b/i.test(
        normalized
      );
  }
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}
