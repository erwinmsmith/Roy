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
  if (/\b(?:do not|don't|without)\s+(?:modify|edit|write|change|patch|mutate)\b/.test(normalized)
    || /\b(?:read[- ]only|analysis only|review only|plan only)\b/.test(normalized)
    || /(?:不要|无需|仅|只)\s*(?:修改|写入|改动|执行)/.test(task)) {
    return false;
  }
  return /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor|migrate|upgrade|downgrade|install|remove|replace|apply|build)\b[\s\S]{0,240}\b(?:file|code|project|repository|repo|workspace|artifact|solution|dependency|dependencies|implementation|migration|application|package|tests?)\b/i.test(task)
    || /\b(?:fix|repair|migrate|upgrade|refactor|implement)\b[\s\S]{0,160}\b(?:bug|issue|failure|task|feature|api|cli|runtime|system)\b/i.test(task)
    || /(?:实现|修改|编辑|创建|写入|修复|重构|迁移|升级|安装|替换|落盘|改动)[\s\S]{0,120}(?:文件|代码|项目|仓库|工作区|依赖|实现|测试|系统)/.test(task);
}

export function isSuccessfulWorkspaceMutationCall(call: ExecutionIntentCall): boolean {
  if (!call.success) return false;
  if (call.toolName === 'fs.write' || call.toolName === 'fs.replace') return true;
  if (call.toolName !== 'shell.exec') return false;
  const command = String(call.params.command ?? '');
  if (/(?:^|[;&|]\s*|\s)(?:apply_patch|touch|mkdir|cp|mv|rm|install|chmod|truncate|git\s+apply|npm\s+(?:install|uninstall)|pnpm\s+(?:add|remove|install)|yarn\s+(?:add|remove|install)|pip\s+install|uv\s+(?:add|remove|pip\s+install)|sed\s+-i|perl\s+-pi)\b/i.test(command)) {
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

export function isSuccessfulWorkspaceVerificationCall(call: ExecutionIntentCall): boolean {
  if (!call.success || !isWorkspaceVerificationCall(call)) return false;
  const shell = call.result as { exitCode?: unknown; stdout?: unknown; stderr?: unknown } | undefined;
  if (typeof shell?.exitCode === 'number' && shell.exitCode !== 0) return false;
  const output = `${String(shell?.stdout ?? '')}\n${String(shell?.stderr ?? '')}`;
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
  if (call.toolName !== 'fs.write' && call.toolName !== 'fs.replace') return undefined;
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

function isNonWorkspaceOutputTarget(target: string): boolean {
  return NULL_OUTPUT_TARGETS.has(target)
    || NON_WORKSPACE_OUTPUT_PREFIXES.some(prefix => target.startsWith(prefix));
}

function extractRedirectionTargets(command: string): string[] {
  return [...command.matchAll(/(?:^|[\s;&|])\d*>>?\s*(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))/g)]
    .map(match => String(match[1] ?? match[2] ?? match[3] ?? '').trim())
    .filter(Boolean);
}
