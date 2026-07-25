export interface ExecutionIntentCall {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  success: boolean;
}

export interface ParallelSourceMutation {
  requestedPath: string;
  authoritativeRoot: string;
  packageName: string;
}

export interface WorkspaceCandidateRollback {
  restored: boolean;
  path?: string;
  reason?: string;
  baselineReward?: number;
  candidateReward?: number;
  regressedReward?: number;
  candidateFingerprint?: string;
  baselineGroups?: Record<string, number>;
  candidateGroups?: Record<string, number>;
  regressedGroups?: Array<{ group: string; before?: number; after?: number }>;
  improvedGroups?: Array<{ group: string; before?: number; after?: number }>;
}

export interface WorkspaceCandidateRetention {
  retained: true;
  path?: string;
  reason?: string;
  baselineReward?: number;
  candidateReward?: number;
  improvedGroups?: Array<{ group: string; before?: number; after?: number }>;
}

export function workspaceToolIntentFingerprint(
  call: Pick<ExecutionIntentCall, 'toolName' | 'params'>
): string {
  const params = { ...call.params };
  if (call.toolName === 'shell.exec') {
    delete params.timeoutMs;
    if (typeof params.command === 'string') params.command = params.command.trim();
  }
  if (call.toolName === 'fs.read' || call.toolName === 'fs.list' || call.toolName === 'fs.search') {
    params.path = normalizeWorkspaceRelativePath(String(params.path ?? '.')) || '.';
  }
  const sortedParams = Object.fromEntries(
    Object.entries(params).sort(([left], [right]) => left.localeCompare(right))
  );
  return `${call.toolName}:${JSON.stringify(sortedParams)}`;
}

export function completedWorkspaceReadCoversPlan(
  completed: ExecutionIntentCall,
  planned: Pick<ExecutionIntentCall, 'toolName' | 'params'>
): boolean {
  if (!completed.success || completed.toolName !== 'fs.read' || planned.toolName !== 'fs.read') {
    return false;
  }
  const completedPath = normalizeWorkspaceRelativePath(String(completed.params.path ?? ''));
  const plannedPath = normalizeWorkspaceRelativePath(String(planned.params.path ?? ''));
  if (!completedPath || completedPath !== plannedPath) return false;

  const result = completed.result as {
    truncated?: unknown;
    startLine?: unknown;
    endLine?: unknown;
    totalLines?: unknown;
    persistedContentCompacted?: unknown;
  } | undefined;
  if (result?.truncated === true) return false;
  const completedStart = positiveInteger(result?.startLine)
    ?? positiveInteger(completed.params.startLine)
    ?? 1;
  const completedEnd = positiveInteger(result?.endLine);
  const totalLines = positiveInteger(result?.totalLines);
  const completedFullFile = completedStart === 1
    && completedEnd !== undefined
    && totalLines !== undefined
    && completedEnd >= totalLines;
  if (completedFullFile) {
    return result?.persistedContentCompacted === true
      ? planned.params.startLine === undefined && planned.params.endLine === undefined
      : true;
  }

  const plannedStart = positiveInteger(planned.params.startLine) ?? 1;
  const plannedEnd = positiveInteger(planned.params.endLine);
  return completedEnd !== undefined
    && plannedEnd !== undefined
    && completedStart <= plannedStart
    && completedEnd >= plannedEnd;
}

const NULL_OUTPUT_TARGETS = new Set([
  '/dev/null',
  '/dev/stdout',
  '/dev/stderr',
  '/proc/self/fd/1',
  '/proc/self/fd/2',
  '&1',
  '&2',
]);

const NON_WORKSPACE_OUTPUT_PREFIXES = [
  '/tmp/',
  '/var/tmp/',
  '/private/tmp/',
];

export function taskRequestsWorkspaceMutation(task: string): boolean {
  const normalized = task.toLowerCase().replace(/\s+/g, ' ');
  const explicitlyGlobalReadOnly =
    /^\s*(?:read[- ]only|analysis only|review only|plan only)\b/.test(normalized)
    || /\b(?:read[- ]only|analysis only|review only|plan only)\s+(?:mode|task|request|work)\b/.test(
      normalized
    )
    || /\b(?:work|operate|proceed|run|review|analy[sz]e)\b[^.;\n]{0,100}\bread[- ]only(?:\s+mode)?\b/.test(
      normalized
    );
  if (explicitlyGlobalReadOnly) {
    return false;
  }
  const mutationScopedTask = task
    .replace(
      /\b(?:do not|don't|without)\s+(?:modify|modifying|edit|editing|write|writing|change|changing|patch|patching|mutate|mutating)\b[^.;\n]*/gi,
      ' '
    )
    .replace(/(?:不要|无需|仅|只)\s*(?:修改|写入|改动|执行)[^。；;\n]*/g, ' ');
  return /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor|migrate|upgrade|downgrade|install|remove|replace|apply|build)\b[\s\S]{0,240}\b(?:file|code|project|repository|repo|workspace|artifact|solution|dependency|dependencies|implementation|migration|application|package|tests?)\b/i.test(mutationScopedTask)
    || /\b(?:fix|repair|migrate|upgrade|refactor|implement)\b[\s\S]{0,160}\b(?:bug|issue|failure|task|feature|api|cli|runtime|system)\b/i.test(mutationScopedTask)
    || /(?:实现|修改|编辑|创建|写入|修复|重构|迁移|升级|安装|替换|落盘|改动)[\s\S]{0,120}(?:文件|代码|项目|仓库|工作区|依赖|实现|测试|系统)/.test(mutationScopedTask);
}

export function isSuccessfulWorkspaceMutationCall(call: ExecutionIntentCall): boolean {
  if (!call.success) return false;
  if (call.toolName === 'fs.write'
    || call.toolName === 'fs.replace'
    || call.toolName === 'fs.synthesize') return true;
  if (call.toolName !== 'shell.exec') return false;
  const command = String(call.params.command ?? '');
  if (/(?:^|[;&|]\s*)(?:apply_patch|touch|mkdir|cp|mv|rm|install|chmod|truncate|git\s+apply|npm\s+(?:install|uninstall)|pnpm\s+(?:add|remove|install)|yarn\s+(?:add|remove|install)|uv\s+(?:add|remove)|sed\s+-i|perl\s+-pi)\b/i.test(command)) {
    return true;
  }
  if (/(?:^|\s)(?:python|python3|node)\b[\s\S]*(?:writeFile|write_text|write_bytes|open\s*\([^)]*['"][wa]['"])/i.test(command)) {
    return true;
  }
  if (/\b(?:dd)\b[\s\S]*\bof=(?!\/dev\/(?:null|stdout|stderr)\b)\S+/i.test(command)) {
    return true;
  }
  if (/\btee(?:\s+-a)?\s+(?!\/dev\/(?:null|stdout|stderr)\b)\S+/i.test(command)) {
    return true;
  }
  return extractRedirectionTargets(command).some(target => !isNonWorkspaceOutputTarget(target));
}

/**
 * Return the mutation calls whose workspace effects are still present.
 *
 * A verifier can transactionally restore the source snapshot that preceded a
 * candidate. The original mutation remains useful trace evidence, but it must
 * not invalidate cached reads, extend a convergence horizon, or satisfy the
 * mutation side of execution closure after its effects have been rolled back.
 */
export function effectiveWorkspaceMutationCallIndices(
  calls: ExecutionIntentCall[]
): number[] {
  const effective: number[] = [];
  for (let index = 0; index < calls.length; index += 1) {
    const call = calls[index]!;
    if (isSuccessfulWorkspaceMutationCall(call)) {
      effective.push(index);
      continue;
    }
    const rollback = workspaceCandidateRollbackFromCall(call);
    if (!rollback?.restored || effective.length === 0) continue;
    const rollbackPath = normalizeWorkspaceRelativePath(String(rollback.path ?? ''));
    let matchedPosition = -1;
    for (let position = effective.length - 1; position >= 0; position -= 1) {
      const mutation = calls[effective[position]!]!;
      if (!rollbackPath || mutation.toolName === 'shell.exec') {
        matchedPosition = position;
        break;
      }
      const mutationPath = normalizeWorkspaceRelativePath(String(mutation.params.path ?? ''));
      if (mutationPath === rollbackPath) {
        matchedPosition = position;
        break;
      }
    }
    if (matchedPosition >= 0) effective.splice(matchedPosition, 1);
  }
  return effective;
}

export function hasEffectiveWorkspaceMutationCall(
  calls: ExecutionIntentCall[]
): boolean {
  return effectiveWorkspaceMutationCallIndices(calls).length > 0;
}

export function lastEffectiveWorkspaceMutationCallIndex(
  calls: ExecutionIntentCall[]
): number {
  return effectiveWorkspaceMutationCallIndices(calls).at(-1) ?? -1;
}

export function workspaceCandidateRollbackFromCall(
  call: ExecutionIntentCall
): WorkspaceCandidateRollback | undefined {
  if (!call.result || typeof call.result !== 'object') return undefined;
  const result = call.result as {
    candidateRollback?: unknown;
    regressionRollback?: unknown;
  };
  const value = result.candidateRollback ?? result.regressionRollback;
  if (!value || typeof value !== 'object') return undefined;
  const rollback = value as Record<string, unknown>;
  if (rollback.restored !== true) return undefined;
  const parseGroups = (
    groups: unknown
  ): Array<{ group: string; before?: number; after?: number }> | undefined => {
    if (!Array.isArray(groups)) return undefined;
    const parsed = groups.flatMap(item => {
      if (!item || typeof item !== 'object') return [];
      const group = item as Record<string, unknown>;
      if (typeof group.group !== 'string') return [];
      return [{
        group: group.group,
        before: typeof group.before === 'number' ? group.before : undefined,
        after: typeof group.after === 'number' ? group.after : undefined,
      }];
    });
    return parsed.length > 0 ? parsed : undefined;
  };
  const parseGroupScores = (groups: unknown): Record<string, number> | undefined => {
    if (!groups || typeof groups !== 'object' || Array.isArray(groups)) return undefined;
    const parsed = Object.fromEntries(
      Object.entries(groups)
        .filter((entry): entry is [string, number] =>
          typeof entry[1] === 'number' && Number.isFinite(entry[1])
        )
    );
    return Object.keys(parsed).length > 0 ? parsed : undefined;
  };
  return {
    restored: true,
    path: typeof rollback.path === 'string' ? rollback.path : undefined,
    reason: typeof rollback.reason === 'string' ? rollback.reason : undefined,
    baselineReward: typeof rollback.baselineReward === 'number'
      ? rollback.baselineReward
      : undefined,
    candidateReward: typeof rollback.candidateReward === 'number'
      ? rollback.candidateReward
      : undefined,
    regressedReward: typeof rollback.regressedReward === 'number'
      ? rollback.regressedReward
      : undefined,
    candidateFingerprint: typeof rollback.candidateFingerprint === 'string'
      ? rollback.candidateFingerprint
      : undefined,
    baselineGroups: parseGroupScores(rollback.baselineGroups),
    candidateGroups: parseGroupScores(rollback.candidateGroups),
    regressedGroups: parseGroups(rollback.regressedGroups),
    improvedGroups: parseGroups(rollback.improvedGroups),
  };
}

export function workspaceCandidateRetentionFromCall(
  call: ExecutionIntentCall
): WorkspaceCandidateRetention | undefined {
  if (!call.result || typeof call.result !== 'object') return undefined;
  const value = (call.result as { candidateRetention?: unknown }).candidateRetention;
  if (!value || typeof value !== 'object') return undefined;
  const retention = value as Record<string, unknown>;
  if (retention.retained !== true) return undefined;
  const improvedGroups = Array.isArray(retention.improvedGroups)
    ? retention.improvedGroups.flatMap(item => {
      if (!item || typeof item !== 'object') return [];
      const group = item as Record<string, unknown>;
      if (typeof group.group !== 'string') return [];
      return [{
        group: group.group,
        before: typeof group.before === 'number' ? group.before : undefined,
        after: typeof group.after === 'number' ? group.after : undefined,
      }];
    })
    : undefined;
  return {
    retained: true,
    path: typeof retention.path === 'string' ? retention.path : undefined,
    reason: typeof retention.reason === 'string' ? retention.reason : undefined,
    baselineReward: typeof retention.baselineReward === 'number'
      ? retention.baselineReward
      : undefined,
    candidateReward: typeof retention.candidateReward === 'number'
      ? retention.candidateReward
      : undefined,
    improvedGroups,
  };
}

/**
 * Defer a path after repeated retained candidates fail to move any verifier
 * objective. The path becomes eligible again only after both its current
 * snapshot and immutable verifier evidence are freshly grounded.
 */
export function workspaceTargetNeedsFreshNoGainEvidence(
  calls: ExecutionIntentCall[],
  targetPath: string,
  attemptLimit = 2
): boolean {
  const normalizedTarget = normalizeWorkspaceRelativePath(targetPath);
  if (!normalizedTarget || attemptLimit < 1) return false;
  const retainedIndices: number[] = [];
  for (let index = 0; index < calls.length; index += 1) {
    const retention = workspaceCandidateRetentionFromCall(calls[index]!);
    if (!retention) continue;
    const retainedPath = normalizeWorkspaceRelativePath(String(retention.path ?? ''));
    const noRewardGain = typeof retention.baselineReward !== 'number'
      || typeof retention.candidateReward !== 'number'
      || retention.candidateReward <= retention.baselineReward + 1e-12;
    if (retainedPath === normalizedTarget
      && noRewardGain
      && (retention.improvedGroups?.length ?? 0) === 0) {
      retainedIndices.push(index);
    }
  }
  if (retainedIndices.length < attemptLimit) return false;
  const latestRetentionIndex = retainedIndices.at(-1)!;
  const newerCalls = calls.slice(latestRetentionIndex + 1);
  const targetRegrounded = newerCalls.some(call =>
    call.toolName === 'fs.read'
    && call.success
    && normalizeWorkspaceRelativePath(String(
      (call.result as { path?: unknown } | undefined)?.path
        ?? call.params.path
        ?? ''
    )) === normalizedTarget
  );
  const verifierRegrounded = newerCalls.some(call =>
    call.toolName === 'fs.read'
    && call.success
    && normalizeWorkspaceRelativePath(String(
      (call.result as { path?: unknown } | undefined)?.path
        ?? call.params.path
        ?? ''
    )).startsWith('.roy/official-verifier/')
  );
  return !(targetRegrounded && verifierRegrounded);
}

export function plannedWorkspaceMutationPath(
  call: Pick<ExecutionIntentCall, 'toolName' | 'params'>
): string | undefined {
  if (call.toolName !== 'fs.write'
    && call.toolName !== 'fs.replace'
    && call.toolName !== 'fs.synthesize') {
    return undefined;
  }
  const normalized = normalizeWorkspaceRelativePath(String(call.params.path ?? ''));
  return normalized || undefined;
}

export function isSuccessfulWorkspaceVerificationCall(call: ExecutionIntentCall): boolean {
  if (!call.success || !isWorkspaceVerificationCall(call)) return false;
  const shell = call.result as { exitCode?: unknown; stdout?: unknown; stderr?: unknown } | undefined;
  if (typeof shell?.exitCode === 'number' && shell.exitCode !== 0) return false;
  const output = `${String(shell?.stdout ?? '')}\n${String(shell?.stderr ?? '')}`;
  const command = String(call.params.command ?? '');
  if (/\.roy\/official-verifier\//i.test(command)) {
    const trimmed = output.trim();
    const numericScore = /^(?:0(?:\.\d+)?|1(?:\.0+)?)$/.test(trimmed)
      ? Number(trimmed)
      : undefined;
    const labeledScore = [...output.matchAll(
      /(?:^|\n)\s*(?:reward|score)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*(?:\n|$)/gi
    )].map(match => Number(match[1])).at(-1);
    const verifierScore = labeledScore ?? numericScore;
    if (verifierScore !== undefined && verifierScore < 1) return false;
  }
  const reportedStatuses = [...output.matchAll(/(?:^|\n)\s*(?:exit(?:_code)?|status)\s*[:=]\s*(-?\d+)\s*(?:\n|$)/gi)]
    .map(match => Number(match[1]));
  return reportedStatuses.length === 0 || reportedStatuses.every(status => status === 0);
}

export function isWorkspaceVerificationCall(call: ExecutionIntentCall): boolean {
  if (call.toolName !== 'shell.exec') return false;
  const command = String(call.params.command ?? '');
  if (masksShellFailure(command)) return false;
  return /\b(?:test|pytest|vitest|jest|mocha|cargo\s+test|go\s+test|npm\s+(?:test|run\s+(?:test|check|build|lint|typecheck))|pnpm\s+(?:test|run)|yarn\s+(?:test|run)|ruff|eslint|tsc|mypy|pyright|compileall)\b/i.test(command)
    || /\bpython(?:3)?\s+(?:-m\s+[A-Za-z_][\w.]*|(?:\.\/)?[\w./-]+\.py)(?:\s|$)/i.test(command);
}

export function findParallelSourceMutation(
  call: Pick<ExecutionIntentCall, 'toolName' | 'params'>,
  observations: ExecutionIntentCall[]
): ParallelSourceMutation | undefined {
  if (call.toolName !== 'fs.write'
    && call.toolName !== 'fs.replace'
    && call.toolName !== 'fs.synthesize') return undefined;
  const requestedPath = normalizeWorkspaceRelativePath(String(call.params.path ?? ''));
  if (!requestedPath || requestedPath.startsWith('/')) return undefined;

  const authoritativeRoots = new Map<string, string>();
  for (const observation of observations) {
    if (!observation.success) continue;
    const observedPaths = [
      String(observation.params.path ?? ''),
      ...extractResultPaths(observation.result),
    ];
    for (const observedPath of observedPaths) {
      const normalized = normalizeWorkspaceRelativePath(observedPath);
      const sourceMatch = normalized.match(/^(src|lib)\/([^/]+)\/.*\.(?:py|ts|tsx|js|mjs|cjs)$/i);
      const packageMatch = normalized.match(/^packages\/([^/]+)\/(?:src\/)?.*\.(?:py|ts|tsx|js|mjs|cjs)$/i);
      if (sourceMatch) {
        authoritativeRoots.set(sourceMatch[2]!, `${sourceMatch[1]}/${sourceMatch[2]}`);
      } else if (packageMatch) {
        authoritativeRoots.set(packageMatch[1]!, `packages/${packageMatch[1]}`);
      }
    }
  }

  const requestedPackage = requestedPath.split('/')[0] ?? '';
  const authoritativeRoot = authoritativeRoots.get(requestedPackage);
  if (!authoritativeRoot || requestedPath.startsWith(`${authoritativeRoot}/`)) return undefined;
  return {
    requestedPath,
    authoritativeRoot,
    packageName: requestedPackage,
  };
}

function masksShellFailure(command: string): boolean {
  return /\|\|\s*(?:true|:)(?:\s*(?:[;&|]|$))|;\s*(?:true|:)\s*;?\s*$|\bset\s+\+e\b/i.test(command)
    || /;\s*(?:printf|echo)\b[^;\n]*(?:\$\?|exit(?:_code)?|status)/i.test(command);
}

function extractResultPaths(result: unknown): string[] {
  if (!result || typeof result !== 'object') return [];
  const value = result as { path?: unknown; entries?: unknown };
  const entries = Array.isArray(value.entries)
    ? value.entries.filter((entry): entry is string => typeof entry === 'string')
    : [];
  return [
    ...(typeof value.path === 'string' ? [value.path] : []),
    ...entries,
  ];
}

function normalizeWorkspaceRelativePath(value: string): string {
  return value.trim().replace(/\\/g, '/').replace(/^(?:\.\/)+/, '').replace(/\/+/g, '/');
}

function positiveInteger(value: unknown): number | undefined {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) && number >= 1 ? Math.floor(number) : undefined;
}

function isNonWorkspaceOutputTarget(target: string): boolean {
  return NULL_OUTPUT_TARGETS.has(target)
    || NON_WORKSPACE_OUTPUT_PREFIXES.some(prefix => target.startsWith(prefix));
}

function extractRedirectionTargets(command: string): string[] {
  return [...command.matchAll(/(?:^|[\s;&|])\d*>>?\s*(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))/g)]
    .map(match => String(match[1] ?? match[2] ?? match[3] ?? '').trim())
    .filter(Boolean);
}
