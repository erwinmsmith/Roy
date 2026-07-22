import type { TeamExecutionPolicy, TeamMemberExecutionStatus } from './types.js';

export const DEFAULT_TEAM_EXECUTION_POLICY: TeamExecutionPolicy = {
  mode: 'sequential',
  failureMode: 'best_effort',
  maxConcurrency: 3,
  minimumSuccessfulMembers: 1,
};

export interface TeamExecutionItem<T> {
  key: string;
  execute: () => Promise<T>;
}

export interface TeamExecutionOutcome<T> {
  key: string;
  status: Extract<TeamMemberExecutionStatus, 'completed' | 'failed' | 'skipped'>;
  value?: T;
  error?: string;
  cause?: unknown;
}

export function normalizeTeamExecutionPolicy(
  input: Partial<TeamExecutionPolicy> = {}
): TeamExecutionPolicy {
  const policy: TeamExecutionPolicy = { ...DEFAULT_TEAM_EXECUTION_POLICY };
  if (input.mode !== undefined) policy.mode = input.mode;
  if (input.failureMode !== undefined) policy.failureMode = input.failureMode;
  if (input.maxConcurrency !== undefined) policy.maxConcurrency = input.maxConcurrency;
  if (input.minimumSuccessfulMembers !== undefined) {
    policy.minimumSuccessfulMembers = input.minimumSuccessfulMembers;
  }
  if (policy.mode !== 'sequential' && policy.mode !== 'parallel') {
    throw new Error(`Unsupported team execution mode "${String(policy.mode)}"`);
  }
  if (policy.failureMode !== 'fail_fast' && policy.failureMode !== 'best_effort') {
    throw new Error(`Unsupported team failure mode "${String(policy.failureMode)}"`);
  }
  if (!Number.isInteger(policy.maxConcurrency) || policy.maxConcurrency < 1) {
    throw new Error('Team maxConcurrency must be a positive integer');
  }
  if (!Number.isInteger(policy.minimumSuccessfulMembers) || policy.minimumSuccessfulMembers < 1) {
    throw new Error('Team minimumSuccessfulMembers must be a positive integer');
  }
  return policy;
}

export async function executeTeamItems<T>(
  items: TeamExecutionItem<T>[],
  policyInput: Partial<TeamExecutionPolicy> = {}
): Promise<TeamExecutionOutcome<T>[]> {
  const policy = normalizeTeamExecutionPolicy(policyInput);
  if (items.length === 0) return [];
  if (policy.mode === 'sequential') return executeSequentially(items, policy);

  const outcomes: Array<TeamExecutionOutcome<T> | undefined> = new Array(items.length);
  let cursor = 0;
  let stopped = false;
  const worker = async (): Promise<void> => {
    while (!stopped) {
      const index = cursor++;
      if (index >= items.length) return;
      const item = items[index];
      try {
        outcomes[index] = { key: item.key, status: 'completed', value: await item.execute() };
      } catch (error) {
        outcomes[index] = { key: item.key, status: 'failed', error: errorMessage(error), cause: error };
        if (policy.failureMode === 'fail_fast') stopped = true;
      }
    }
  };
  const concurrency = Math.min(policy.maxConcurrency, items.length);
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return items.map((item, index) => outcomes[index] ?? { key: item.key, status: 'skipped' });
}

async function executeSequentially<T>(
  items: TeamExecutionItem<T>[],
  policy: TeamExecutionPolicy
): Promise<TeamExecutionOutcome<T>[]> {
  const outcomes: TeamExecutionOutcome<T>[] = [];
  let stopped = false;
  for (const item of items) {
    if (stopped) {
      outcomes.push({ key: item.key, status: 'skipped' });
      continue;
    }
    try {
      outcomes.push({ key: item.key, status: 'completed', value: await item.execute() });
    } catch (error) {
      outcomes.push({ key: item.key, status: 'failed', error: errorMessage(error), cause: error });
      stopped = policy.failureMode === 'fail_fast';
    }
  }
  return outcomes;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
