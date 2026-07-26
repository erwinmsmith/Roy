import {
  effectiveWorkspaceMutationCallIndices,
  isSuccessfulWorkspaceMutationCall,
  isSuccessfulWorkspaceVerificationCall,
  isUnavailableWorkspaceVerificationCall,
  isWorkspaceVerificationCall,
  workspaceCandidateRollbackFromCall,
} from './executionIntent.js';
import yaml from 'js-yaml';

export interface ToolPlanBinding {
  name: string;
  enabled: boolean;
}

export interface PlannedToolCall {
  toolName: string;
  params: Record<string, unknown>;
  reason: string;
  groundingRequired: boolean;
}

export interface ToolPlanningInput {
  task: string;
  workspacePath: string;
  bindings: ToolPlanBinding[];
  archetype?: string;
}

export interface ObservedToolCall {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  success: boolean;
  error?: unknown;
}

export interface RecoveryFeedbackFocus {
  summary: string;
  path?: string;
  authoritativeVerifierCommand?: string;
}

export function recoveryFeedbackFocus(task: string): RecoveryFeedbackFocus | undefined {
  const capsuleMatch = /<recovery_capsule>\s*([\s\S]*?)\s*<\/recovery_capsule>/i.exec(
    task
  );
  if (!capsuleMatch) return undefined;
  try {
    const capsule = JSON.parse(capsuleMatch[1]!) as {
      recoveryTrigger?: unknown;
      externalFeedback?: {
        summary?: unknown;
        path?: unknown;
      };
      authoritativeVerifierCommand?: unknown;
    };
    if (capsule.recoveryTrigger !== 'new_external_verifier_feedback') {
      return undefined;
    }
    const summary = typeof capsule.externalFeedback?.summary === 'string'
      ? capsule.externalFeedback.summary.trim().slice(0, 1_200)
      : '';
    if (!summary) return undefined;
    const path = typeof capsule.externalFeedback?.path === 'string'
      ? capsule.externalFeedback.path.trim()
      : '';
    const authoritativeVerifierCommand =
      typeof capsule.authoritativeVerifierCommand === 'string'
        ? capsule.authoritativeVerifierCommand.trim()
        : '';
    return {
      summary,
      ...(path ? { path } : {}),
      ...(authoritativeVerifierCommand ? { authoritativeVerifierCommand } : {}),
    };
  } catch {
    return undefined;
  }
}

export function effectiveRecoveryFeedbackFocus(
  task: string,
  calls: ObservedToolCall[]
): RecoveryFeedbackFocus | undefined {
  const focus = recoveryFeedbackFocus(task);
  for (const latestFailure of [...calls].reverse()) {
    if (!isWorkspaceVerificationCall(latestFailure)
      || isSuccessfulWorkspaceVerificationCall(latestFailure)) {
      continue;
    }
    const result = latestFailure.result as {
      stdout?: unknown;
      stderr?: unknown;
      verifierDiagnostics?: unknown;
    } | undefined;
    const diagnostics = Array.isArray(result?.verifierDiagnostics)
      ? result.verifierDiagnostics
        .map(item =>
          item && typeof item === 'object'
            ? String((item as { content?: unknown }).content ?? '')
            : ''
        )
        .filter(Boolean)
      : [];
    const output = [
      String(result?.stdout ?? ''),
      String(result?.stderr ?? ''),
      String(latestFailure.error ?? ''),
      ...diagnostics,
    ].filter(Boolean).join('\n');
    const failures = [...output.matchAll(
      /\b(?:AssertionError|AttributeError|ImportError|ModuleNotFoundError|RuntimeError|TypeError|ValueError):\s*([^\n]+)/g
    )];
    let summary = failures.at(-1)?.[1]?.trim();
    if (!summary) {
      for (const diagnostic of [...diagnostics].reverse()) {
        try {
          const parsed = JSON.parse(diagnostic) as { failure?: unknown };
          if (typeof parsed.failure === 'string' && parsed.failure.trim()) {
            summary = parsed.failure.trim();
            break;
          }
        } catch {
          // Non-JSON verifier artifacts are covered by the traceback matcher.
        }
      }
    }
    if (!summary) continue;
    return {
      summary: summary.slice(0, 1_200),
      ...(focus?.authoritativeVerifierCommand
        ? { authoritativeVerifierCommand: focus.authoritativeVerifierCommand }
        : {}),
    };
  }
  return focus;
}

function semanticFeedbackIdentifiers(feedback: string): string[] {
  const identifiers = new Set<string>();
  for (const match of feedback.matchAll(
    /\b(?:[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Z][A-Za-z0-9_]{3,})\b/g
  )) {
    const identifier = match[0]!;
    if (/^(?:AssertionError|ImportError|ModuleNotFoundError|RuntimeError|TypeError)$/i.test(
      identifier
    )) {
      continue;
    }
    identifiers.add(identifier);
  }
  return [...identifiers];
}

export class AgentToolPlanner {
  plan(input: ToolPlanningInput): PlannedToolCall[] {
    const enabled = new Set(input.bindings.filter(binding => binding.enabled).map(binding => binding.name));
    const recoveryFocus = recoveryFeedbackFocus(input.task);
    const planningTask = recoveryFocus
      ? `Repair the newest verifier failure: ${recoveryFocus.summary}${
        recoveryFocus.path ? `\nAuthoritative source path: ${recoveryFocus.path}` : ''
      }`
      : input.task;
    const lower = planningTask.toLowerCase();
    const plans: PlannedToolCall[] = [];
    const referencedPaths = this.extractReferencedPaths(
      planningTask.replace(/https?:\/\/[^\s`'"<>),]+/gi, ' ')
    );
    const referencedDirectories = referencedPaths.filter(item => item.endsWith('/'));
    const referencedFiles = referencedPaths.filter(item => !item.endsWith('/'));
    const referencedUrls = this.extractReferencedUrls(planningTask);
    const explicitShellCommands = recoveryFocus
      ? []
      : this.extractExplicitShellCommands(input.task);
    const runtimeApiInspection = /\bruntime\s+apis?\b[\s\S]{0,80}\b(?:exports?|surface|inspection|declarations?|signatures?|source|symbols?)\b|\bexported runtime apis?\b/.test(lower);
    const mutationTask = /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor|migrate|upgrade|install|replace|apply)\b/.test(lower);
    const explicitUrlReading = /\b(?:open|read|fetch|visit|consult)\b[\s\S]{0,120}https?:\/\//i.test(input.task);
    const webEvidenceRequired = this.requiresWebEvidence(lower);
    const broadWorkspaceInspection = /\b(?:actual|current|entire|full|all)\b[\s\S]{0,100}\b(?:workspace|project|repository|repo|codebase|files?|metadata|manifests?)\b/.test(lower)
      || /\b(?:workspace|project|repository|repo|codebase)\b[\s\S]{0,100}\b(?:structure|layout|inventory|metadata|manifests?|all files?)\b/.test(lower);

    if (enabled.has('web.fetch')
      && referencedUrls.length > 0
      && (webEvidenceRequired || explicitUrlReading)) {
      plans.push(...referencedUrls.map(url => ({
        toolName: 'web.fetch',
        params: { url },
        reason: `The task explicitly references ${url}.`,
        groundingRequired: true,
      })));
    } else if (enabled.has('web.search') && webEvidenceRequired) {
      const independentQuestions = this.extractIndependentWebQuestions(input.task);
      if (independentQuestions.length > 1) {
        plans.push(...independentQuestions.map(query => ({
          toolName: 'web.search',
          params: { query, maxResults: 5 },
          reason: 'This independent question requires externally verifiable web evidence.',
          groundingRequired: true,
        })));
      } else {
        plans.push({
          toolName: 'web.search',
          params: { query: this.buildSearchQuery(input.task), maxResults: 5 },
          reason: 'The task requires current or externally verifiable web evidence.',
          groundingRequired: true,
        });
      }
    }

    if (recoveryFocus) {
      const focusedPath = recoveryFocus.path
        ? this.normalizeWorkspacePath(recoveryFocus.path)
        : '';
      const dependencyManifestPaths = this.recoveryDependencyManifestPaths(
        recoveryFocus.summary,
        input.task
      );
      if (focusedPath && enabled.has('fs.read')) {
        plans.push({
          toolName: 'fs.read',
          params: { path: focusedPath },
          reason: `The newest verifier feedback identifies ${focusedPath} as its authoritative repair target.`,
          groundingRequired: true,
        });
      } else if (dependencyManifestPaths.length > 0 && enabled.has('fs.read')) {
        plans.push(...dependencyManifestPaths.map(path => ({
          toolName: 'fs.read',
          params: { path },
          reason: `The newest dependency or runtime-version feedback requires grounding the authoritative ${path} declaration before considering source changes.`,
          groundingRequired: true,
        })));
      } else if (enabled.has('fs.search')) {
        const identifier = semanticFeedbackIdentifiers(recoveryFocus.summary)[0];
        if (identifier) {
          plans.push({
            toolName: 'fs.search',
            params: {
              path: '.',
              query: identifier,
              maxResults: 20,
            },
            reason: `Locate the current source definition for ${identifier}, the semantic symbol named by the newest verifier feedback.`,
            groundingRequired: true,
          });
        }
      }
      if (enabled.has('fs.read')) {
        const verifierPath = this.extractReferencedPaths(input.task)
          .map(path => this.normalizeWorkspacePath(path))
          .find(path => path.startsWith('.roy/official-verifier/'));
        if (verifierPath) {
          plans.push({
            toolName: 'fs.read',
            params: { path: verifierPath },
            reason: 'Read the immutable verifier assertion that produced the newest focused failure.',
            groundingRequired: true,
          });
        }
      }
    } else if (enabled.has('fs.list') && !this.isWebOnlyTask(lower) && (mutationTask || broadWorkspaceInspection)) {
      plans.push({
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
        reason: 'Establish the authoritative workspace layout before choosing paths or applying changes.',
        groundingRequired: true,
      });
    } else if (enabled.has('fs.list') && referencedDirectories.length > 0) {
      plans.push(...referencedDirectories.map(directory => ({
        toolName: 'fs.list',
        params: { path: directory.replace(/\/$/, ''), maxDepth: 3 },
        reason: `The task explicitly requests directory evidence from ${directory}.`,
        groundingRequired: true,
      })));
    } else if (enabled.has('fs.list')
      && /\b(inspect|analy[sz]e|review|list|structure|project|codebase|repo|repository|files?|evidence|coverage|verify)\b/.test(lower)
      && referencedPaths.length === 0
      && !runtimeApiInspection
      && !this.isWebOnlyTask(lower)) {
      plans.push({
        toolName: 'fs.list',
        params: { path: input.workspacePath, maxDepth: 2 },
        reason: 'The task requires concrete workspace structure evidence.',
        groundingRequired: true,
      });
    }

    if (enabled.has('shell.exec') && explicitShellCommands.length > 0) {
      plans.push(...explicitShellCommands.map(command => ({
        toolName: 'shell.exec',
        params: { command },
        reason: 'The task marks this as an explicit command to execute and preserve the real exit status.',
        groundingRequired: true,
      })));
    }

    const inferredFilePaths: string[] = [];
    if (/\b(?:package exports?|export map|package manifest|package entr(?:y|ies))\b/.test(lower)
      || input.archetype === 'critic'
        && /\b(?:architecture|architectural|repository|codebase|dependency|coupling)\b/.test(lower)) {
      inferredFilePaths.push('package.json');
    }
    if (runtimeApiInspection) {
      inferredFilePaths.push('src/index.ts', 'src/core/runtime/index.ts');
    }
    const filePaths = Array.from(new Set([...referencedFiles, ...inferredFilePaths]));
    if (enabled.has('fs.read')
      && filePaths.length > 0
      && /\b(read|inspect|review|check|open|identify|analy[sz]e)\b/.test(lower)
      && !this.isWebOnlyTask(lower)) {
      plans.push(...filePaths.map(filePath => ({
        toolName: 'fs.read',
        params: { path: filePath },
        reason: referencedFiles.length > 0
          ? `The task explicitly references ${filePath}.`
          : `The package export request requires evidence from ${filePath}.`,
        groundingRequired: true,
      })));
    }

    if (enabled.has('shell.exec')) {
      const explicitNpmTest = /\bnpm test\b/.test(lower);
      if (explicitNpmTest) {
        plans.push({ toolName: 'shell.exec', params: { command: 'npm test' }, reason: 'The task explicitly requests the test suite.', groundingRequired: true });
      } else if (input.archetype === 'critic'
        && !enabled.has('fs.read')
        && /\b(?:architecture|architectural|repository|codebase|dependency|coupling)\b/.test(lower)) {
        plans.push({
          toolName: 'shell.exec',
          params: { command: 'cat package.json' },
          reason: 'The architecture critique requires manifest evidence and the cached actor exposes shell execution instead of fs.read.',
          groundingRequired: true,
        });
      } else if (/\b(?:run|execute)\s+(?:the\s+)?build\b|\bnpm run build\b/.test(lower)) {
        plans.push({ toolName: 'shell.exec', params: { command: 'npm run build' }, reason: 'The task explicitly requests the build.', groundingRequired: true });
      }
    }

    const seen = new Set<string>();
    const uniquePlans = plans.filter(plan => {
      const fingerprint = `${plan.toolName}:${JSON.stringify(plan.params)}`;
      if (seen.has(fingerprint)) return false;
      seen.add(fingerprint);
      return true;
    });
    const initialPlanLimit = uniquePlans.length > 0
      && uniquePlans.every(plan => plan.toolName === 'web.search')
      ? 10
      : 3;
    return uniquePlans.slice(0, initialPlanLimit);
  }

  planWebFollowUps(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    maxFetches: number;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding => binding.enabled && binding.name === 'web.fetch')) return [];
    const requiredSources = this.requiredWebSourceCount(input.task);
    const relevanceTasks = this.webEvidenceTasks(input.task);
    const relevantDocuments = this.relevantWebDocuments(input.task, input.calls);
    const focusedSections = input.calls.filter(call => {
      if (call.toolName !== 'web.fetch'
        || !call.success
        || !relevanceTasks.some(task => this.webEvidenceScore(task, call) >= 6)) return false;
      const value = String((call.result as { finalUrl?: unknown } | undefined)?.finalUrl ?? call.params.url ?? '');
      try {
        return Boolean(new URL(value).hash);
      } catch {
        return false;
      }
    }).length;
    // A focused section is enough to hand control back to the LLM planner. It can
    // finish or search for another distinct source instead of crawling navigation.
    if (relevantDocuments.size >= requiredSources || focusedSections > 0) return [];
    const fetched = new Set(input.calls
      .filter(call => call.toolName === 'web.fetch')
      .flatMap(call => [
        String(call.params.url ?? ''),
        String((call.result as { finalUrl?: unknown } | undefined)?.finalUrl ?? ''),
      ])
      .filter(Boolean)
      .map(url => this.canonicalWebDocumentUrl(url)));
    const latestSearch = [...input.calls].reverse().find(call => call.toolName === 'web.search' && call.success);
    const results = (latestSearch?.result as {
      results?: Array<{ url?: unknown; title?: unknown; snippet?: unknown }>;
    } | undefined)?.results;
    if (Array.isArray(results)) {
      const latestQuery = String(latestSearch?.params.query ?? input.task);
      const relevanceTask = relevanceTasks.length > 1 ? latestQuery : input.task;
      const relevanceThreshold = relevanceTasks.length > 1 ? 6 : 4;
      const resultUrls = new Set(results
        .filter(item => typeof item.url === 'string')
        .map(item => this.canonicalWebDocumentUrl(String(item.url))));
      const alreadyFetched = [...fetched].filter(url => resultUrls.has(url)).length;
      const remaining = Math.max(0, input.maxFetches - alreadyFetched);
      const searchPlans = results
        .filter(item => typeof item.url === 'string' && !fetched.has(this.canonicalWebDocumentUrl(item.url)))
        .map(item => ({ item, score: this.webRelevanceScore(relevanceTask, `${String(item.title ?? '')} ${String(item.snippet ?? '')} ${String(item.url)}`) }))
        .filter(candidate => candidate.score >= relevanceThreshold)
        .sort((left, right) => right.score - left.score)
        .slice(0, remaining)
        .map(({ item }) => ({
          toolName: 'web.fetch',
          params: { url: String(item.url) },
          reason: `Fetch the relevant discovered source "${String(item.title ?? item.url)}" before making source-backed claims.`,
          groundingRequired: true,
        } satisfies PlannedToolCall));
      if (searchPlans.length > 0) return searchPlans;
    }

    const pageLinks = input.calls
      .filter(call => call.toolName === 'web.fetch' && call.success)
      .flatMap(call => {
        const links = (call.result as { links?: Array<{ text?: unknown; url?: unknown }> } | undefined)?.links;
        return Array.isArray(links) ? links : [];
      });
    return pageLinks
      .filter(link => typeof link.url === 'string' && !fetched.has(this.canonicalWebDocumentUrl(link.url)))
      .map(link => ({
        link,
        score: Math.max(...relevanceTasks.map(task =>
          this.webRelevanceScore(task, `${String(link.text ?? '')} ${String(link.url)}`)
        )),
      }))
      .filter(candidate => candidate.score >= 4)
      .sort((left, right) => right.score - left.score)
      .slice(0, Math.max(0, input.maxFetches))
      .map(({ link }) => ({
        toolName: 'web.fetch',
        params: { url: String(link.url) },
        reason: `Follow a task-relevant link discovered in fetched page content: ${String(link.text ?? link.url)}.`,
        groundingRequired: true,
      }));
  }

  planWorkspaceEvidenceFollowUps(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'fs.read'
    )) {
      return [];
    }
    const observed = new Set(input.calls
      .filter(call => call.toolName === 'fs.read')
      .map(call => this.normalizeWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      )))
      .filter(Boolean));
    const listedPaths = input.calls
      .filter(call => call.toolName === 'fs.list' && call.success)
      .flatMap(call => {
        const entries = (call.result as { entries?: unknown } | undefined)?.entries;
        return Array.isArray(entries)
          ? entries.filter((entry): entry is string => typeof entry === 'string')
          : [];
      })
      .map(path => this.normalizeWorkspacePath(path));
    const resolvePath = (rawPath: string, relativeTo?: string): string | undefined => {
      let candidate = this.normalizeWorkspacePath(rawPath);
      const workspaceRoot = this.normalizeWorkspacePath(String(input.workspaceRoot ?? ''));
      if (workspaceRoot && candidate.startsWith(`${workspaceRoot}/`)) {
        candidate = candidate.slice(workspaceRoot.length + 1);
      }
      if (candidate.startsWith('/') || candidate.startsWith('../')) return undefined;
      if (relativeTo
        && !candidate.includes('/')
        && relativeTo.includes('/')) {
        candidate = `${relativeTo.slice(0, relativeTo.lastIndexOf('/') + 1)}${candidate}`;
      }
      const listedMatch = listedPaths.find(path =>
        path === candidate || path.endsWith(`/${candidate}`)
      );
      return listedMatch ?? candidate;
    };
    const readablePath = (path: string): boolean =>
      /\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh|json|jsonl|csv|ya?ml|toml|ini|cfg|md|txt|xml)$/i.test(
        path
      );
    const candidates: Array<{ path: string; reason: string; priority: number }> = [];
    for (const explicitPath of this.extractReferencedPaths(input.task)) {
      const taskDeclaredPath = this.taskDeclaredDirectoryPath(input.task, explicitPath);
      const path = resolvePath(taskDeclaredPath ?? explicitPath);
      if (!path || !readablePath(path)) continue;
      const lower = path.toLowerCase();
      const priority = /(?:^|\/)(?:src|lib|app|packages)\/.+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh)$/i.test(
        path
      )
        ? 340
        : lower.startsWith('.roy/official-verifier/')
          ? 320
          : /(?:^|\/)[^/]*manifest[^/]*\.(?:json|ya?ml|toml)$/i.test(path)
            ? 260
            : /(?:^|\/)outputs?\//i.test(path)
              ? 40
              : 200;
      candidates.push({
        path,
        reason: `Read the still-unobserved file explicitly named by the task: ${path}.`,
        priority,
      });
    }

    for (const call of input.calls) {
      if (call.toolName !== 'fs.read' || !call.success) continue;
      const result = call.result as { path?: unknown; content?: unknown } | undefined;
      const sourcePath = this.normalizeWorkspacePath(String(
        result?.path ?? call.params.path ?? ''
      ));
      if (!/(?:^|\/)(?:[^/]*manifest[^/]*\.json|[^/]*(?:config|configuration|settings|rules|expectations|audit)[^/]*\.(?:json|ya?ml))$/i.test(sourcePath)
        || typeof result?.content !== 'string') {
        continue;
      }
      let parsed: unknown;
      try {
        parsed = /\.ya?ml$/i.test(sourcePath)
          ? yaml.load(result.content)
          : JSON.parse(result.content);
      } catch {
        continue;
      }
      const referenced = new Set<string>();
      const visit = (value: unknown, depth: number): void => {
        if (depth > 8 || value === null || value === undefined) return;
        if (typeof value === 'string') {
          if (/[./\\][A-Za-z0-9_.@/-]+\.[A-Za-z0-9]{1,8}$/.test(value)
            || /^[A-Za-z0-9_.@-]+\.[A-Za-z0-9]{1,8}$/.test(value)) {
            referenced.add(value);
          }
          return;
        }
        if (Array.isArray(value)) {
          value.slice(0, 200).forEach(item => visit(item, depth + 1));
          return;
        }
        if (typeof value === 'object') {
          Object.values(value as Record<string, unknown>)
            .slice(0, 200)
            .forEach(item => visit(item, depth + 1));
        }
      };
      visit(parsed, 0);
      for (const rawPath of referenced) {
        const path = resolvePath(rawPath, sourcePath);
        if (!path || !readablePath(path)) continue;
        const lower = path.toLowerCase();
        candidates.push({
          path,
          reason: `Follow the task input dependency referenced by ${sourcePath}: ${path}.`,
          priority: /(?:ocr|tokens?)/.test(lower)
            ? 150
            : /(?:metadata|schema|expected|rules?)/.test(lower)
              ? 120
              : 80,
        });
      }
    }

    const seen = new Set<string>();
    return candidates
      .filter(candidate => {
        if (!candidate.path || observed.has(candidate.path) || seen.has(candidate.path)) {
          return false;
        }
        seen.add(candidate.path);
        return true;
      })
      .sort((left, right) =>
        right.priority - left.priority || left.path.localeCompare(right.path)
      )
      .slice(0, 3)
      .map(candidate => ({
        toolName: 'fs.read',
        params: { path: candidate.path },
        reason: candidate.reason,
        groundingRequired: true,
      }));
  }

  planGroundedImplementationTransition(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'fs.synthesize'
    ) || !this.taskRequestsWorkspaceMutation(input.task)) {
      return [];
    }
    const taskLower = input.task.toLowerCase().replaceAll('\\', '/');
    const workspaceRoot = this.normalizeWorkspacePath(String(input.workspaceRoot ?? ''))
      .replace(/\/+$/, '');
    const normalizeObservedPath = (value: unknown): string => {
      let candidate = this.normalizeWorkspacePath(String(value ?? ''));
      if (workspaceRoot && candidate.startsWith(`${workspaceRoot}/`)) {
        candidate = candidate.slice(workspaceRoot.length + 1);
      }
      return candidate.replace(/^\/+/, '');
    };
    const effectiveMutations = new Set(
      effectiveWorkspaceMutationCallIndices(input.calls)
    );
    const acceptanceRepairFeedback =
      /<runtime_acceptance_repair_targets>([\s\S]*?)<\/runtime_acceptance_repair_targets>/i.exec(
        input.task
      )?.[1]?.toLowerCase() ?? '';
    const attemptedMutationIndexByPath = new Map<string, number>();
    input.calls.forEach((call, index) => {
      if (!isSuccessfulWorkspaceMutationCall(call)) return;
      const path = normalizeObservedPath(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
      );
      if (path) attemptedMutationIndexByPath.set(path, index);
    });
    const mutatedPaths = new Set(input.calls
      .map((call, index) => ({ call, index }))
      .filter(item => effectiveMutations.has(item.index))
      .map(({ call }) => normalizeObservedPath(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
      ))
      .filter(Boolean));
    const contractTerms = this.extractImplementationContractTerms(input.task);
    const latestFailureIndex = input.calls.reduce((latest, call, index) =>
      isWorkspaceVerificationCall(call)
      && !isSuccessfulWorkspaceVerificationCall(call)
        ? index
        : latest
    , -1);
    const latestFailure = latestFailureIndex >= 0
      ? input.calls[latestFailureIndex]
      : undefined;
    const latestFailureResult = latestFailure?.result
      && typeof latestFailure.result === 'object'
      ? latestFailure.result as {
        stdout?: unknown;
        stderr?: unknown;
        verifierDiagnostics?: unknown;
      }
      : undefined;
    const latestFailureText = latestFailure
      ? [
        String(latestFailure.error ?? ''),
        ...this.extractVerifierDiagnosticText(
          latestFailureResult?.verifierDiagnostics
        ),
        String(latestFailureResult?.stdout ?? ''),
        String(latestFailureResult?.stderr ?? ''),
      ].join('\n').toLowerCase()
      : '';
    const candidates = input.calls
      .map((call, index) => {
        if (call.toolName !== 'fs.read' || !call.success) return undefined;
        const result = call.result as {
          path?: unknown;
          content?: unknown;
          truncated?: unknown;
        } | undefined;
        const path = normalizeObservedPath(result?.path ?? call.params.path);
        const content = typeof result?.content === 'string' ? result.content : '';
        const basename = path.slice(path.lastIndexOf('/') + 1);
        const basenameStem = basename.replace(/\.[^.]+$/, '');
        const acceptanceFailureMentioned = Boolean(
          acceptanceRepairFeedback
          && (
            acceptanceRepairFeedback.includes(path.toLowerCase())
            || acceptanceRepairFeedback.includes(basename.toLowerCase())
            || new RegExp(
              `\\b${basenameStem.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`,
              'i'
            ).test(acceptanceRepairFeedback)
          )
        );
        if (!path
          || !content
          || result?.truncated === true
          || mutatedPaths.has(path) && !acceptanceFailureMentioned
          || !this.isMutableImplementationPath(path)) {
          return undefined;
        }
        const taskPathIndex = taskLower.indexOf(path.toLowerCase());
        const taskBasenameIndex = taskLower.indexOf(basename.toLowerCase());
        const taskExplicit = taskPathIndex >= 0 || taskBasenameIndex >= 0;
        const matchedTerms = contractTerms.filter(term =>
          content.toLowerCase().includes(term.toLowerCase())
        );
        const failureMentioned = latestFailureText.includes(path.toLowerCase())
          || latestFailureText.includes(basename.toLowerCase());
        const previousMutationIndex = attemptedMutationIndexByPath.get(path);
        if (previousMutationIndex !== undefined
          && !acceptanceFailureMentioned
          && (
            latestFailureIndex <= previousMutationIndex
            || !failureMentioned
          )) {
          return undefined;
        }
        if (effectiveMutations.size > 0
          && !taskExplicit
          && !failureMentioned
          && !acceptanceFailureMentioned) {
          return undefined;
        }
        if (!taskExplicit
          && matchedTerms.length === 0
          && !failureMentioned
          && !acceptanceFailureMentioned) {
          return undefined;
        }
        const firstMention = taskPathIndex >= 0
          ? taskPathIndex
          : taskBasenameIndex >= 0
            ? taskBasenameIndex
            : Number.MAX_SAFE_INTEGER;
        return {
          path,
          content,
          index,
          firstMention,
          matchedTerms,
          acceptanceFailureMentioned,
          score: (acceptanceFailureMentioned ? 1_200 : 0)
            + (failureMentioned ? 900 : 0)
            + (taskPathIndex >= 0 ? 500 : taskBasenameIndex >= 0 ? 360 : 0)
            + Math.min(10, matchedTerms.length) * 45,
        };
      })
      .filter((candidate): candidate is NonNullable<typeof candidate> =>
        candidate !== undefined
      )
      .sort((left, right) =>
        right.score - left.score
        || left.firstMention - right.firstMention
        || right.index - left.index
      );
    const candidate = candidates[0];
    if (!candidate) return [];
    const relevantContract = candidate.matchedTerms.slice(0, 12);
    const instructions = [
      'Apply the next uncompleted task-declared implementation slice to this already observed source file.',
      relevantContract.length > 0
        ? `The current source still contains these identifiers or contract terms named by the assignment: ${relevantContract.join(', ')}.`
        : 'The assignment explicitly names this file as an implementation target.',
      candidate.acceptanceFailureMentioned
        ? `The latest global acceptance audit identifies this component as failed:\n${acceptanceRepairFeedback.slice(0, 2_400)}`
        : '',
      'Use the current file as the patch base, preserve public interfaces and unrelated working behavior, and replace only the legacy or incomplete behavior required by the immutable assignment.',
      'Do not modify tests, benchmark assets, verifier files, or unrelated source paths.',
    ].join(' ');
    const duplicate = input.calls.some(call =>
      call.toolName === 'fs.synthesize'
      && normalizeObservedPath(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
      ) === candidate.path
      && call.params.strategy === 'patch'
      && String(call.params.instructions ?? '') === instructions
    );
    if (duplicate) return [];
    return [{
      toolName: 'fs.synthesize',
      params: {
        path: candidate.path,
        instructions,
        strategy: 'patch',
      },
      reason: `All required evidence for ${candidate.path} is grounded, while that explicit implementation slice has no retained successful mutation; transition from cached inspection to execution.`,
      groundingRequired: true,
    }];
  }

  planPostMutationVerification(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'shell.exec'
    )) {
      return [];
    }
    const mutationIndices = effectiveWorkspaceMutationCallIndices(input.calls);
    const latestMutationIndex = mutationIndices.at(-1);
    if (latestMutationIndex === undefined) return [];
    const commands = this.extractExplicitShellCommands(input.task);
    const latestFailedVerificationIndex = input.calls.reduce(
      (latest, call, index) =>
        index > latestMutationIndex
        && isWorkspaceVerificationCall(call)
        && !isUnavailableWorkspaceVerificationCall(call)
        && !isSuccessfulWorkspaceVerificationCall(call)
          ? index
          : latest,
      -1
    );
    if (latestFailedVerificationIndex >= 0) {
      const failedVerification = input.calls[latestFailedVerificationIndex]!;
      const failedCommand = String(failedVerification.params.command ?? '').trim();
      const recoveryIndex = input.calls.reduce((latest, call, index) =>
        index > latestFailedVerificationIndex
        && call.toolName === 'shell.exec'
        && call.success
        && this.isDependencyInstallCommand(String(call.params.command ?? ''))
          ? index
          : latest
      , -1);
      const retriedAfterRecovery = recoveryIndex >= 0
        && input.calls.some((call, index) =>
          index > recoveryIndex
          && call.toolName === 'shell.exec'
          && String(call.params.command ?? '').trim() === failedCommand
        );
      if (recoveryIndex >= 0
        && !retriedAfterRecovery
        && commands.some(command => command.trim() === failedCommand)) {
        return [{
          toolName: 'shell.exec',
          params: { command: failedCommand },
          reason: 'Re-run the task-declared acceptance command whose missing runtime tool was just installed successfully before considering any source repair.',
          groundingRequired: true,
        }];
      }
      return [];
    }
    const latestDependencyMutationIndex = mutationIndices
      .filter(index => this.isDependencyManifestPath(
        this.normalizeWorkspacePath(String(
          (input.calls[index]!.result as { path?: unknown } | undefined)?.path
            ?? input.calls[index]!.params.path
            ?? ''
        ))
      ))
      .at(-1);
    const pending = commands
      .filter(command => {
        if (/^(?:cd|pushd|popd)\s+\S+\s*$/i.test(command.trim())) {
          return false;
        }
        const matchingCalls = input.calls
          .map((call, index) => ({ call, index }))
          .filter(item =>
            item.call.toolName === 'shell.exec'
            && String(item.call.params.command ?? '').trim() === command.trim()
          );
        if (this.isDependencyInstallCommand(command)) {
          const latestSuccessfulInstall = matchingCalls
            .filter(item => item.call.success)
            .map(item => item.index)
            .at(-1) ?? -1;
          if (latestDependencyMutationIndex === undefined
            ? latestSuccessfulInstall >= 0
            : latestSuccessfulInstall > latestDependencyMutationIndex) {
            return false;
          }
        } else if (this.isEnvironmentSetupCommand(command)
          && matchingCalls.some(item => item.call.success)) {
          return false;
        }
        return !input.calls.some((call, index) =>
          index > latestMutationIndex
          && call.toolName === 'shell.exec'
          && String(call.params.command ?? '').trim() === command.trim()
        );
      });
    const invalidatedInstall = pending.find(command =>
      this.isDependencyInstallCommand(command)
    );
    const selected = invalidatedInstall
      ? [invalidatedInstall]
      : pending
        .map(command => ({
          command,
          latestFailureIndex: input.calls.reduce((latest, call, index) =>
            index < latestMutationIndex
            && call.toolName === 'shell.exec'
            && String(call.params.command ?? '').trim() === command.trim()
            && !call.success
              ? index
              : latest
          , -1),
        }))
        .filter(item => item.latestFailureIndex >= 0)
        .sort((left, right) => right.latestFailureIndex - left.latestFailureIndex)
        .slice(0, 1)
        .map(item => item.command);
    const verificationFrontier = selected.length > 0
      ? selected
      : pending.slice(0, 1);
    return verificationFrontier
      .map(command => ({
        toolName: 'shell.exec',
        params: { command },
        reason: invalidatedInstall === command
          ? 'Refresh the active environment from the newly mutated dependency manifest before running behavior checks.'
          : selected.includes(command)
            ? 'Re-run the most recent failing task-declared acceptance command first so the mutation receives focused causal feedback.'
            : 'Run the task-declared acceptance command against the newly mutated workspace before closing execution.',
        groundingRequired: true,
      }));
  }

  planExternalFeedbackRepair(input: {
    task: string;
    calls: ObservedToolCall[];
    currentCalls?: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'fs.synthesize'
    )) {
      return [];
    }
    const recoveryFocus = effectiveRecoveryFeedbackFocus(input.task, input.calls);
    const feedbackMatch = /<official_verifier_feedback>([\s\S]*?)<\/official_verifier_feedback>/i.exec(
      input.task
    );
    const continuationMatch = /##\s+VERIFICATION FAILED\b([\s\S]*)/i.exec(input.task);
    const feedback = String(
      recoveryFocus?.summary
        ?? feedbackMatch?.[1]
        ?? continuationMatch?.[1]
        ?? ''
    ).trim();
    if (!feedback) return [];
    const dependencyFeedback = feedback
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line =>
        line.length > 0
        && line.length <= 600
        && !/^(?:requirement already satisfied|collecting|downloading|installing collected packages|successfully (?:built|installed)|building (?:wheel|editable)|created wheel|stored in directory|warning: running pip)\b/i.test(
          line
        ))
      .some(line =>
        /\b(?:dependenc(?:y|ies)|requirements?|manifest|runtime version|version constraint|pin(?:ned|ning)?|package install)\b/i.test(
          line
        )
        && /\b(?:fail(?:ed|ure)?|error|invalid|incorrect|wrong|legacy|outdated|incompatible|unsupported|mismatch|must|should|expect(?:ed)?|require[sd]?|target(?:s|ed)?|pin(?:ned|ning)?|upgrade|downgrade|missing|not found|cannot|can't)\b/i.test(
          line
        ));
    const feedbackLocations = this.extractFailureLocations(
      feedback,
      input.workspaceRoot ?? '',
      input.workspaceRoot
    );
    const explicitFeedbackPaths = new Set([
      ...(recoveryFocus?.path ? [recoveryFocus.path] : []),
      ...this.extractReferencedPaths(feedback),
      ...feedbackLocations.map(location => location.path),
    ].map(path => this.normalizeWorkspacePath(path)));
    const semanticIdentifiers = semanticFeedbackIdentifiers(feedback);
    const manifestPriority = (candidatePath: string): number => {
      const lowerPath = candidatePath.toLowerCase();
      if (/(?:^|\/)pyproject\.toml$/.test(lowerPath)) return 420;
      if (/(?:^|\/)requirements[^/]*\.txt$/.test(lowerPath)) return 400;
      if (/(?:^|\/)package\.json$/.test(lowerPath)) return 380;
      if (/(?:^|\/)(?:cargo\.toml|go\.mod|pom\.xml|build\.gradle|gemfile)$/.test(lowerPath)) return 360;
      if (/(?:^|\/)[^/]*(?:lock|manifest|dependencies)[^/]*\.(?:json|toml|ya?ml|txt)$/.test(lowerPath)) return 320;
      return 0;
    };
    const observedReads = new Set(input.calls
      .filter(call => call.toolName === 'fs.read' && call.success)
      .map(call => this.normalizeWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      )))
      .filter(Boolean));
    if (dependencyFeedback
      && input.bindings.some(binding => binding.enabled && binding.name === 'fs.read')) {
      const unreadManifest = input.calls
        .filter(call => call.toolName === 'fs.list' && call.success)
        .flatMap(call => {
          const entries = (call.result as { entries?: unknown } | undefined)?.entries;
          return Array.isArray(entries)
            ? entries.filter((entry): entry is string => typeof entry === 'string')
            : [];
        })
        .map(candidatePath => this.normalizeWorkspacePath(candidatePath))
        .filter(candidatePath =>
          manifestPriority(candidatePath) > 0 && !observedReads.has(candidatePath)
        )
        .sort((left, right) =>
          manifestPriority(right) - manifestPriority(left)
          || left.localeCompare(right)
        )[0];
      if (unreadManifest) {
        return [{
          toolName: 'fs.read',
          params: { path: unreadManifest },
          reason: `External dependency feedback requires checking the parallel manifest ${unreadManifest} before changing an already observed declaration.`,
          groundingRequired: true,
        }];
      }
    }
    const mutatedPaths = new Set((input.currentCalls ?? input.calls)
      .filter(call => isSuccessfulWorkspaceMutationCall(call))
      .map(call => this.normalizeWorkspacePath(String(call.params.path ?? '')))
      .filter(Boolean));
    const candidates = input.calls
      .filter(call => call.toolName === 'fs.read' && call.success)
      .map(call => {
        const result = call.result as { path?: unknown; content?: unknown } | undefined;
        const candidatePath = this.normalizeWorkspacePath(String(
          result?.path ?? call.params.path ?? ''
        ));
        return {
          path: candidatePath,
          content: typeof result?.content === 'string' ? result.content : '',
        };
      })
      .filter(candidate =>
        candidate.path
        && candidate.content
        && (
          this.isMutableImplementationPath(candidate.path)
          || dependencyFeedback && manifestPriority(candidate.path) > 0
        )
        && !mutatedPaths.has(candidate.path)
      )
      .map(candidate => {
        const lowerPath = candidate.path.toLowerCase();
        let score = explicitFeedbackPaths.has(candidate.path) ? 500 : 0;
        if (this.isMutableImplementationPath(candidate.path)) score += 40;
        const matchingIdentifiers = semanticIdentifiers.filter(identifier =>
          new RegExp(
            `\\b${identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`,
            'i'
          ).test(candidate.content)
        );
        if (matchingIdentifiers.length > 0) {
          score += 600 + Math.min(200, matchingIdentifiers.length * 40);
        }
        if (dependencyFeedback) {
          const dependencyManifestPriority = manifestPriority(lowerPath);
          score += dependencyManifestPriority;
          if (dependencyManifestPriority > 0
            && /\b(?:legacy|outdated|incompatible|unsupported|upgrade|migration)\b/i.test(
              feedback
            )) {
            const declarationLines = candidate.content
              .split(/\r?\n/)
              .filter(line =>
                !/^\s*(?:#|description\s*=)/i.test(line)
              )
              .join('\n');
            if (/(?:={2,3}|~=)\s*0\.\d|pydantic\s*<\s*2/im.test(
              declarationLines
            )) {
              // A concrete obsolete constraint is stronger causal evidence
              // than a manifest merely named by a verifier traceback. This
              // matters when parallel manifests disagree and the modern
              // primary manifest still describes the project as "legacy".
              score += 1_200;
            }
          }
        }
        const basename = candidate.path.slice(candidate.path.lastIndexOf('/') + 1);
        if (feedback.toLowerCase().includes(basename.toLowerCase())) score += 160;
        const basenameStem = basename.replace(/\.[^.]+$/, '');
        if (basenameStem.length >= 5
          && new RegExp(
            `\\b${basenameStem.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`,
            'i'
          ).test(feedback)) {
          score += 220;
        }
        return { ...candidate, score };
      })
      .filter(candidate => candidate.score > 0)
      .sort((left, right) =>
        right.score - left.score || left.path.localeCompare(right.path)
      );
    const candidate = candidates[0];
    if (!candidate) return [];
    const compactFeedback = feedback.length <= 2_800
      ? feedback
      : `${feedback.slice(0, 700)}\n[older external feedback compacted]\n${feedback.slice(-2_100)}`;
    const instructions = [
      'Apply the smallest coherent, interface-preserving change that resolves the newest external verifier feedback.',
      `Authoritative external feedback:\n${compactFeedback}`,
      'Use the already observed current file as the patch base. Preserve unrelated working declarations and do not alter benchmark or verifier files.',
    ].join('\n\n');
    const duplicateAttempt = (input.currentCalls ?? input.calls).some(call =>
      call.toolName === 'fs.synthesize'
      && this.normalizeWorkspacePath(String(call.params.path ?? '')) === candidate.path
      && String(call.params.instructions ?? '') === instructions
      && call.params.strategy === 'patch'
    );
    if (duplicateAttempt) return [];
    return [{
      toolName: 'fs.synthesize',
      params: {
        path: candidate.path,
        instructions,
        strategy: 'patch',
      },
      reason: `The external verifier identifies a concrete failure category and ${candidate.path} is the highest-priority observed file that controls it.`,
      groundingRequired: true,
    }];
  }

  planWorkspaceFailureFollowUps(input: {
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding => binding.enabled && binding.name === 'fs.read')) return [];
    const latestFailureIndex = this.latestShellFailureIndex(input.calls);
    if (latestFailureIndex < 0) return [];
    const latestMutationIndex =
      effectiveWorkspaceMutationCallIndices(input.calls).at(-1) ?? -1;
    const describeFailure = (index: number): {
      index: number;
      output: string;
      locations: Array<{ path: string; line?: number }>;
    } => {
      const failure = input.calls[index]!;
      const shell = failure.result as {
        cwd?: unknown;
        stdout?: unknown;
        stderr?: unknown;
        verifierDiagnostics?: unknown;
      } | undefined;
      const output = [
        String(shell?.stdout ?? ''),
        String(shell?.stderr ?? ''),
        String(failure.error ?? ''),
        ...this.extractVerifierDiagnosticText(shell?.verifierDiagnostics),
      ].filter(Boolean).join('\n');
      return {
        index,
        output,
        locations: this.extractFailureLocations(
          output,
          String(shell?.cwd ?? input.workspaceRoot ?? ''),
          input.workspaceRoot
        ),
      };
    };
    let causalFailure = describeFailure(latestFailureIndex);
    if (!causalFailure.locations.some(location =>
      this.isMutableImplementationPath(location.path)
    )) {
      for (
        let index = latestFailureIndex - 1;
        index > latestMutationIndex;
        index -= 1
      ) {
        const call = input.calls[index]!;
        if (call.toolName !== 'shell.exec'
          || (
            call.success
            && (!isWorkspaceVerificationCall(call)
              || isSuccessfulWorkspaceVerificationCall(call))
          )) {
          continue;
        }
        const candidate = describeFailure(index);
        if (!candidate.locations.some(location =>
          this.isMutableImplementationPath(location.path)
        )) {
          continue;
        }
        causalFailure = candidate;
        break;
      }
    }
    const failureIndex = causalFailure.index;
    const locations = causalFailure.locations;
    const abstractClass = /can't instantiate abstract class\s+([A-Za-z_][A-Za-z0-9_]*)\s+with abstract methods?\s+([^\n]+)/i.exec(
      causalFailure.output
    );
    if (abstractClass) {
      const className = abstractClass[1]!;
      const query = `class\\s+${className}\\b`;
      const searchAfterFailure = input.calls
        .slice(failureIndex + 1)
        .find(call =>
          call.toolName === 'fs.search'
          && call.success
          && String(call.params.query ?? '') === query
        );
      if (!searchAfterFailure
        && input.bindings.some(binding =>
          binding.enabled && binding.name === 'fs.search'
        )) {
        return [{
          toolName: 'fs.search',
          params: {
            path: '.',
            filePattern: '*.py',
            query,
            regex: true,
            maxResults: 20,
          },
          reason: `The traceback fails while instantiating abstract class ${className}; locate its definition instead of patching the caller frame.`,
          groundingRequired: true,
        }];
      }
      const matches = (searchAfterFailure?.result as {
        matches?: Array<{ path?: unknown; line?: unknown }>;
      } | undefined)?.matches;
      const definition = Array.isArray(matches)
        ? matches.find(match =>
          typeof match.path === 'string'
          && this.isMutableImplementationPath(String(match.path))
        )
        : undefined;
      if (definition) {
        const path = this.normalizeWorkspacePath(String(definition.path));
        const alreadyRead = input.calls
          .slice(failureIndex + 1)
          .some(call =>
            call.toolName === 'fs.read'
            && call.success
            && this.normalizeWorkspacePath(String(
              (call.result as { path?: unknown } | undefined)?.path
                ?? call.params.path
                ?? ''
            )) === path
          );
        if (!alreadyRead) {
          const line = Number(definition.line);
          return [{
            toolName: 'fs.read',
            params: Number.isInteger(line) && line > 0
              ? {
                path,
                startLine: Math.max(1, line - 15),
                endLine: line + 80,
              }
              : { path },
            reason: `Read the definition of abstract class ${className}; its missing methods are the semantic cause, while the traceback caller is only the instantiation site.`,
            groundingRequired: true,
          }];
        }
      }
    }
    const unexpectedKeyword = /(?:typeerror:\s*)?([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+got an unexpected keyword argument\s+['"]([A-Za-z_][A-Za-z0-9_]*)['"]/i.exec(
      causalFailure.output
    );
    if (unexpectedKeyword) {
      const callableName = unexpectedKeyword[1]!.split('.').at(-1)!;
      const caller = locations.find(location =>
        this.isMutableImplementationPath(location.path)
      );
      if (caller) {
        const callerPath = this.normalizeWorkspacePath(caller.path);
        const slash = callerPath.lastIndexOf('/');
        const searchParams = {
          path: slash >= 0 ? callerPath.slice(0, slash) : '.',
          filePattern: slash >= 0 ? callerPath.slice(slash + 1) : callerPath,
          query: callableName,
          maxResults: 20,
        };
        const importSearch = input.calls
          .slice(failureIndex + 1)
          .find(call =>
            call.toolName === 'fs.search'
            && call.success
            && JSON.stringify(call.params) === JSON.stringify(searchParams)
          );
        if (!importSearch
          && input.bindings.some(binding =>
            binding.enabled && binding.name === 'fs.search'
          )) {
          return [{
            toolName: 'fs.search',
            params: searchParams,
            reason: `The installed callable ${callableName} rejected a keyword; locate its import in the caller before changing arguments.`,
            groundingRequired: true,
          }];
        }
        const importEvidence = this.importedCallableEvidence(
          importSearch,
          callableName
        );
        if (importEvidence) {
          const signatureCommand = this.pythonSignatureInspectionCommand(
            importEvidence.moduleName,
            callableName
          );
          const signatureObserved = input.calls
            .slice(failureIndex + 1)
            .some(call =>
              call.toolName === 'shell.exec'
              && call.success
              && String(call.params.command ?? '') === signatureCommand
            );
          if (!signatureObserved) {
            return [{
              toolName: 'shell.exec',
              params: {
                command: signatureCommand,
                maxOutputBytes: 8_000,
              },
              reason: `Inspect the installed runtime signature for ${importEvidence.moduleName}.${callableName}; do not guess replacement keyword names.`,
              groundingRequired: true,
            }];
          }
        }
      }
    }
    const prioritizedLocations = [...locations].sort((left, right) =>
      Number(this.isMutableImplementationPath(right.path))
      - Number(this.isMutableImplementationPath(left.path))
    );
    return prioritizedLocations
      .map(location => ({
        toolName: 'fs.read',
        params: location.line === undefined
          ? { path: location.path }
          : {
              path: location.path,
              startLine: Math.max(1, location.line - 25),
              endLine: location.line + 25,
            },
        reason: location.line === undefined
          ? `Read the source module named by the verifier failure at ${location.path}.`
          : `Read bounded source context around the verifier-reported failure at ${location.path}:${location.line}.`,
        groundingRequired: true,
      } satisfies PlannedToolCall))
      .filter(plan => {
        let latestReadIndex = -1;
        for (let index = input.calls.length - 1; index > failureIndex; index -= 1) {
          const call = input.calls[index]!;
          if (call.toolName === plan.toolName
            && JSON.stringify(call.params) === JSON.stringify(plan.params)) {
            latestReadIndex = index;
            break;
          }
        }
        if (latestReadIndex < 0) return true;
        return input.calls.slice(latestReadIndex + 1).some(call =>
          call.success && (call.toolName === 'fs.write'
            || call.toolName === 'fs.replace'
            || call.toolName === 'fs.synthesize'
            || call.toolName === 'shell.exec' && this.looksLikeShellMutation(String(call.params.command ?? '')))
        );
      })
      .slice(0, 1);
  }

  planEnvironmentRecovery(input: {
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'shell.exec'
    )) {
      return [];
    }
    const latestFailureIndex = this.latestShellFailureIndex(input.calls);
    if (latestFailureIndex < 0) return [];
    const latestMutationIndex =
      effectiveWorkspaceMutationCallIndices(input.calls).at(-1) ?? -1;
    const failureOutput = (call: ObservedToolCall): string => {
      const result = call.result as {
        stdout?: unknown;
        stderr?: unknown;
        verifierDiagnostics?: unknown;
      } | undefined;
      return [
        String(result?.stdout ?? ''),
        String(result?.stderr ?? ''),
        String(call.error ?? ''),
        ...this.extractVerifierDiagnosticText(result?.verifierDiagnostics),
      ].filter(Boolean).join('\n');
    };
    const actionableSourceFailure = input.calls
      .slice(latestMutationIndex + 1, latestFailureIndex)
      .some(call => {
        if (call.toolName !== 'shell.exec'
          || (
            call.success
            && (!isWorkspaceVerificationCall(call)
              || isSuccessfulWorkspaceVerificationCall(call))
          )) {
          return false;
        }
        const result = call.result as { cwd?: unknown } | undefined;
        return this.extractFailureLocations(
          failureOutput(call),
          String(result?.cwd ?? input.workspaceRoot ?? ''),
          input.workspaceRoot
        ).some(location => this.isMutableImplementationPath(location.path));
      });
    if (actionableSourceFailure) return [];

    const failure = input.calls[latestFailureIndex]!;
    const result = failure.result as { command?: unknown } | undefined;
    const command = String(result?.command ?? failure.params.command ?? '').trim();
    const invokedModule = /^((?:python|python3))\s+-m\s+([A-Za-z0-9_.-]+)\b/.exec(
      command
    );
    if (!invokedModule) return [];
    const python = invokedModule[1]!;
    const moduleName = invokedModule[2]!;
    const installableDevelopmentTools = new Map<string, string>([
      ['pytest', 'pytest'],
      ['ruff', 'ruff'],
      ['mypy', 'mypy'],
      ['coverage', 'coverage'],
      ['tox', 'tox'],
      ['nox', 'nox'],
      ['hypothesis', 'hypothesis'],
    ]);
    const packageName = installableDevelopmentTools.get(moduleName);
    if (!packageName) return [];
    const output = failureOutput(failure);
    const escapedModule = moduleName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (!new RegExp(
      `(?:No module named ['"]?${escapedModule}['"]?|ModuleNotFoundError:[^\\n]*${escapedModule})`,
      'i'
    ).test(output)) {
      return [];
    }
    const installCommand = `${python} -m pip install ${packageName}`;
    if (input.calls.some((call, index) =>
      index > latestFailureIndex
      && call.toolName === 'shell.exec'
      && String(call.params.command ?? '').trim() === installCommand
      && call.success
    )) {
      return [];
    }
    return [{
      toolName: 'shell.exec',
      params: { command: installCommand },
      reason: `The task-declared ${moduleName} command cannot start because its own development-tool module is absent, and no newer source-localized failure remains; install the runner before retrying verification.`,
      groundingRequired: true,
    }];
  }

  planWorkspaceRepairTransition(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding =>
      binding.enabled && binding.name === 'fs.synthesize'
    )) {
      return [];
    }
    const latestFailureIndex = this.latestShellFailureIndex(input.calls);
    if (latestFailureIndex < 0) return [];
    const latestFailure = input.calls[latestFailureIndex]!;
    const latestFailureResult = latestFailure.result as {
      stdout?: unknown;
      stderr?: unknown;
    } | undefined;
    const latestFailureOutput = [
      String(latestFailureResult?.stdout ?? ''),
      String(latestFailureResult?.stderr ?? ''),
      String(latestFailure.error ?? ''),
    ].filter(Boolean).join('\n');
    const unavailableDependency = /(?:could not find a version that satisfies the requirement|no matching distribution found for)\s+([A-Za-z0-9_.-]+)/i.exec(
      latestFailureOutput
    )?.[1];
    if (latestFailure.toolName === 'shell.exec'
      && this.isDependencyInstallCommand(String(latestFailure.params.command ?? ''))
      && unavailableDependency) {
      const dependencyToken = unavailableDependency
        .toLowerCase()
        .replace(/[-_.]+/g, '[-_.]');
      const dependencyPattern = new RegExp(`\\b${dependencyToken}\\b`, 'i');
      const manifestRead = input.calls
        .map((call, index) => ({ call, index }))
        .reverse()
        .find(({ call }) => {
          if (call.toolName !== 'fs.read' || !call.success) return false;
          const result = call.result as {
            path?: unknown;
            content?: unknown;
          } | undefined;
          const path = this.normalizeWorkspacePath(String(
            result?.path ?? call.params.path ?? ''
          ));
          return this.isDependencyManifestPath(path)
            && typeof result?.content === 'string'
            && dependencyPattern.test(result.content);
        });
      if (manifestRead) {
        const targetPath = this.normalizeWorkspacePath(String(
          (manifestRead.call.result as { path?: unknown } | undefined)?.path
            ?? manifestRead.call.params.path
            ?? ''
        ));
        const staleRead = input.calls
          .slice(manifestRead.index + 1)
          .some(call =>
            isSuccessfulWorkspaceMutationCall(call)
            && this.normalizeWorkspacePath(String(call.params.path ?? ''))
              === targetPath
          );
        if (staleRead && input.bindings.some(binding =>
          binding.enabled && binding.name === 'fs.read'
        )) {
          return [{
            toolName: 'fs.read',
            params: { path: targetPath },
            reason: `The installer rejected ${unavailableDependency} after ${targetPath} changed; refresh the manifest snapshot before repairing the unresolved constraint.`,
            groundingRequired: true,
          }];
        }
        if (!staleRead) {
          const compactFailure = latestFailureOutput.length <= 2_400
            ? latestFailureOutput
            : `${latestFailureOutput.slice(0, 700)}\n[installer output compacted]\n${latestFailureOutput.slice(-1_700)}`;
          const instructions = [
            'Repair the dependency manifest so the project can install in the authoritative current environment.',
            `The installer cannot resolve ${unavailableDependency}. Reconcile only the declaration that causes this failure; do not add guessed replacement packages or weaken unrelated required runtime constraints.`,
            `Authoritative installer failure:\n${compactFailure}`,
            'Use the freshly read manifest as the patch base and preserve unrelated metadata.',
          ].join('\n\n');
          const duplicate = input.calls
            .slice(manifestRead.index + 1)
            .some(call =>
              call.toolName === 'fs.synthesize'
              && this.normalizeWorkspacePath(String(call.params.path ?? ''))
                === targetPath
              && String(call.params.instructions ?? '') === instructions
              && call.params.strategy === 'patch'
            );
          if (!duplicate) {
            return [{
              toolName: 'fs.synthesize',
              params: {
                path: targetPath,
                instructions,
                strategy: 'patch',
              },
              reason: `The current environment rejected ${unavailableDependency}; the freshly read dependency manifest is the causal repair target.`,
              groundingRequired: true,
            }];
          }
        }
      }
    }
    if (!isWorkspaceVerificationCall(latestFailure)
      || isUnavailableWorkspaceVerificationCall(latestFailure)
      || isSuccessfulWorkspaceVerificationCall(latestFailure)) {
      return [];
    }
    let latestScorecard: {
      groups: Record<string, number>;
      weights: Record<string, number>;
    } | undefined;
    let latestScorecardIndex = -1;
    for (let index = input.calls.length - 1; index >= 0; index -= 1) {
      const state = this.verifierScorecardState(input.calls[index]!);
      if (!state) continue;
      latestScorecard = state;
      latestScorecardIndex = index;
      break;
    }
    const latestMutationIndex = effectiveWorkspaceMutationCallIndices(input.calls).at(-1) ?? -1;
    const freshUnresolvedScorecard = latestScorecardIndex > latestMutationIndex
      && Object.values(latestScorecard?.groups ?? {}).some(score => score < 1);
    if (latestFailureIndex < latestMutationIndex && !freshUnresolvedScorecard) {
      return [];
    }
    type ShellFailureResult = {
      cwd?: unknown;
      stdout?: unknown;
      stderr?: unknown;
      verifierDiagnostics?: unknown;
    };
    const describeFailure = (index: number): {
      index: number;
      call: ObservedToolCall;
      output: string;
      locations: Array<{ path: string; line?: number }>;
    } => {
      const call = input.calls[index]!;
      const shell = call.result as ShellFailureResult | undefined;
      const output = [
        String(shell?.stdout ?? ''),
        String(shell?.stderr ?? ''),
        String(call.error ?? ''),
        ...this.extractVerifierDiagnosticText(shell?.verifierDiagnostics),
      ].filter(Boolean).join('\n');
      return {
        index,
        call,
        output,
        locations: this.extractFailureLocations(
          output,
          String(shell?.cwd ?? input.workspaceRoot ?? ''),
          input.workspaceRoot
        ),
      };
    };
    let causalFailure = describeFailure(latestFailureIndex);
    if (!causalFailure.locations.some(location =>
      this.isMutableImplementationPath(location.path)
    )) {
      for (
        let index = latestFailureIndex - 1;
        index > latestMutationIndex;
        index -= 1
      ) {
        const candidate = input.calls[index]!;
        if (!isWorkspaceVerificationCall(candidate)
          || isUnavailableWorkspaceVerificationCall(candidate)
          || isSuccessfulWorkspaceVerificationCall(candidate)) {
          continue;
        }
        const described = describeFailure(index);
        if (!described.locations.some(location =>
          this.isMutableImplementationPath(location.path)
        )) {
          continue;
        }
        causalFailure = described;
        break;
      }
    }
    const rejectedCandidate = this.latestRollbackForCurrentWorkspace(input.calls);
    const abstractClass = /can't instantiate abstract class\s+([A-Za-z_][A-Za-z0-9_]*)\s+with abstract methods?\s+([^\n]+)/i.exec(
      causalFailure.output
    );
    const semanticDefinitionRead = abstractClass
      ? [...input.calls]
        .map((call, index) => ({ call, index }))
        .reverse()
        .find(({ call, index }) => {
          if (index <= causalFailure.index
            || call.toolName !== 'fs.read'
            || !call.success) {
            return false;
          }
          const result = call.result as {
            path?: unknown;
            content?: unknown;
          } | undefined;
          const candidatePath = this.normalizeWorkspacePath(String(
            result?.path ?? call.params.path ?? ''
          ));
          return this.isMutableImplementationPath(candidatePath)
            && typeof result?.content === 'string'
            && new RegExp(
              `\\bclass\\s+${abstractClass[1]!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`
            ).test(result.content);
        })
      : undefined;
    const semanticDefinitionPath = semanticDefinitionRead
      ? this.normalizeWorkspacePath(String(
        (semanticDefinitionRead.call.result as { path?: unknown } | undefined)?.path
          ?? semanticDefinitionRead.call.params.path
          ?? ''
      ))
      : undefined;
    const failureLocation = semanticDefinitionPath
      ? { path: semanticDefinitionPath }
      : causalFailure.locations.find(location =>
      this.isMutableImplementationPath(location.path)
      );
    if (failureLocation && !rejectedCandidate) {
      const targetPath = failureLocation.path;
      let sourceReadIndex = -1;
      for (
        let index = input.calls.length - 1;
        index > causalFailure.index;
        index -= 1
      ) {
        const call = input.calls[index]!;
        const readPath = this.normalizeWorkspacePath(String(
          (call.result as { path?: unknown } | undefined)?.path
            ?? call.params.path
            ?? ''
        ));
        if (call.toolName === 'fs.read'
          && call.success
          && readPath === targetPath
          && typeof (call.result as { content?: unknown } | undefined)?.content === 'string') {
          sourceReadIndex = index;
          break;
        }
      }
      if (sourceReadIndex < 0
        || effectiveWorkspaceMutationCallIndices(input.calls)
          .some(index => index > sourceReadIndex)) {
        return [];
      }
      const unexpectedKeyword = /(?:typeerror:\s*)?([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+got an unexpected keyword argument\s+['"]([A-Za-z_][A-Za-z0-9_]*)['"]/i.exec(
        causalFailure.output
      );
      let runtimeSignature = '';
      if (unexpectedKeyword) {
        const callableName = unexpectedKeyword[1]!.split('.').at(-1)!;
        const importSearch = input.calls
          .slice(causalFailure.index + 1)
          .find(call =>
            call.toolName === 'fs.search'
            && call.success
            && String(call.params.query ?? '') === callableName
          );
        const importEvidence = this.importedCallableEvidence(
          importSearch,
          callableName
        );
        if (importEvidence) {
          const signatureCommand = this.pythonSignatureInspectionCommand(
            importEvidence.moduleName,
            callableName
          );
          const signatureCall = input.calls
            .slice(causalFailure.index + 1)
            .find(call =>
              call.toolName === 'shell.exec'
              && call.success
              && String(call.params.command ?? '') === signatureCommand
            );
          if (!signatureCall) return [];
          const signatureResult = signatureCall.result as {
            stdout?: unknown;
            stderr?: unknown;
          } | undefined;
          runtimeSignature = [
            String(signatureResult?.stdout ?? ''),
            String(signatureResult?.stderr ?? ''),
          ].filter(Boolean).join('\n').trim().slice(0, 4_000);
        }
      }
      const localizedFailure = causalFailure.output.length <= 2_400
        ? causalFailure.output
        : `[earlier failure output compacted]\n${causalFailure.output.slice(-2_400)}`;
      const instructions = [
        'Apply the smallest interface-preserving patch that fixes this exact localized execution failure.',
        `Authoritative failure:\n${localizedFailure}`,
        runtimeSignature
          ? `Authoritative installed runtime signature:\n${runtimeSignature}`
          : '',
        'Use the freshly read current source as the only patch base. Initialize values on every control-flow path, preserve already working behavior, and do not broaden the change beyond the causal failure.',
      ].filter(Boolean).join('\n\n');
      const duplicate = input.calls.slice(sourceReadIndex + 1).some(call =>
        call.toolName === 'fs.synthesize'
        && this.normalizeWorkspacePath(String(call.params.path ?? '')) === targetPath
        && call.params.strategy === 'patch'
        && String(call.params.instructions ?? '') === instructions
      );
      if (duplicate) return [];
      return [{
        toolName: 'fs.synthesize',
        params: {
          path: targetPath,
          instructions,
          strategy: 'patch',
        },
        reason: causalFailure.index === latestFailureIndex
          ? 'A fresh source read now grounds the exact traceback location; transition directly from diagnosis to a minimal local repair.'
          : 'The newest failure is environment-only, while an earlier post-mutation traceback and fresh source read still identify the actionable code frontier; transition directly to that minimal local repair.',
        groundingRequired: true,
      }];
    }

    const authoritativeReads = input.calls
      .slice(rejectedCandidate ? 0 : latestFailureIndex + 1)
      .filter(call => {
        if (!call.success || call.toolName !== 'fs.read') return false;
        const result = call.result as { content?: unknown } | undefined;
        return typeof result?.content === 'string';
      });
    const verifierObserved = authoritativeReads.some(call =>
      this.normalizeWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      )).startsWith('.roy/official-verifier/')
    );
    if (!verifierObserved) return [];
    const taskPath = input.task.toLowerCase().replaceAll('\\', '/');
    const implementationReads = [...authoritativeReads].reverse().filter(call => {
      const candidate = this.normalizeWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      ));
      return this.isMutableImplementationPath(candidate);
    });
    const rejectedImportConsumerPath = rejectedCandidate
      ? this.rejectedImportConsumerPath(
        input.calls,
        String(rejectedCandidate.path ?? ''),
        input.workspaceRoot
      )
      : undefined;
    const implementationRead = (
      rejectedImportConsumerPath
        ? implementationReads.find(call => {
          const candidate = this.normalizeWorkspacePath(String(
            (call.result as { path?: unknown } | undefined)?.path
              ?? call.params.path
              ?? ''
          ));
          return candidate === rejectedImportConsumerPath;
        })
        : undefined
    ) ?? implementationReads.find(call => {
      const candidate = this.normalizeWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      ));
      return taskPath.includes(candidate.toLowerCase());
    }) ?? implementationReads[0];
    if (!implementationRead) return [];
    const implementationReadIndex = input.calls.lastIndexOf(implementationRead);
    if (effectiveWorkspaceMutationCallIndices(input.calls)
      .some(index => index > implementationReadIndex)) {
      return [];
    }
    const targetPath = this.normalizeWorkspacePath(String(
      (implementationRead.result as { path?: unknown } | undefined)?.path
        ?? implementationRead.params.path
        ?? ''
    ));
    const targetRejectedCandidate = this.latestRollbackForCurrentWorkspace(
      input.calls,
      targetPath
    );
    const effectiveRejectedCandidate = targetRejectedCandidate
      ?? (rejectedImportConsumerPath ? rejectedCandidate : undefined);
    const verifierGroups = {
      ...(effectiveRejectedCandidate?.baselineGroups
        ?? latestScorecard?.groups
        ?? {}),
    };
    for (const group of [
      ...(effectiveRejectedCandidate?.regressedGroups ?? []),
      ...(effectiveRejectedCandidate?.improvedGroups ?? []),
    ]) {
      if (typeof group.before === 'number') verifierGroups[group.group] = group.before;
    }
    const failedGroups = Object.entries(verifierGroups ?? {})
      .filter(([, score]) => score < 1)
      .map(([group, score]) => ({
        group,
        score,
        weight: latestScorecard?.weights[group] ?? 0,
      }))
      .sort((left, right) =>
        right.weight - left.weight || left.group.localeCompare(right.group)
      );
    const passingGroups = Object.entries(verifierGroups ?? {})
      .filter(([, score]) => score >= 1)
      .map(([group]) => group);
    const diagnosticSummary = this.verifierDiagnosticSummary(input.calls);
    const instructions = effectiveRejectedCandidate
      ? [
        'Apply a focused semantics-preserving repair after the prior candidate was transactionally rejected.',
        failedGroups.length > 0
          ? `Repair only the unresolved verifier capabilities in descending objective weight: ${failedGroups.map(item =>
            `${item.group} (score ${item.score.toFixed(3)}, weight ${item.weight.toFixed(3)})`
          ).join(', ')}. Prioritize the highest-weight causal mismatch before low-weight cosmetic groups.`
          : 'Repair only the newest unresolved verifier behavior.',
        passingGroups.length > 0
          ? `Preserve the currently passing capabilities exactly: ${passingGroups.join(', ')}.`
          : 'Preserve all behavior not contradicted by the verifier evidence.',
        effectiveRejectedCandidate.regressedGroups?.length
          ? `Do not repeat the rejected regression in: ${effectiveRejectedCandidate.regressedGroups.map(item => item.group).join(', ')}.`
          : 'Do not repeat the rejected whole-file strategy; make the smallest coherent semantic change supported by the official verifier source.',
        rejectedImportConsumerPath
          ? 'The rejected provider-side compatibility patch regressed accepted behavior. Repair the stale importing consumer instead of restoring the removed provider symbol.'
          : '',
        diagnosticSummary
          ? `Use this newest focused expected-versus-actual verifier reproduction as the causal repair evidence:\n${diagnosticSummary}`
          : '',
        'Treat the grounded official verifier implementation as the executable specification and keep public interfaces stable.',
      ].filter(Boolean).join(' ')
      : 'Structurally repair the implementation to satisfy the grounded aggregate official-verifier failures and immutable assignment, preserving behavior already proven by passing verifier groups.';
    if (effectiveRejectedCandidate && input.calls.some(call =>
      call.toolName === 'fs.synthesize'
      && this.normalizeWorkspacePath(String(call.params.path ?? '')) === targetPath
      && String(call.params.instructions ?? '') === instructions
      && call.params.strategy === 'patch'
      && (
        call.success
        || (call.result as { synthesisRejected?: unknown } | undefined)
          ?.synthesisRejected === true
      )
    )) {
      return [];
    }
    return [{
      toolName: 'fs.synthesize',
      params: {
        path: targetPath,
        instructions,
        strategy: 'patch',
      },
      reason: effectiveRejectedCandidate
        ? rejectedImportConsumerPath
          ? 'The provider-side import compatibility candidate regressed accepted verifier groups; repair the grounded stale consumer while preserving the accepted baseline.'
          : 'The prior candidate was rolled back; change the repair hypothesis to the unresolved verifier capabilities while preserving the accepted baseline.'
        : 'The causal frontier contains a non-perfect aggregate verifier result plus complete verifier and implementation source evidence; transition from inspection to a preserving repair.',
      groundingRequired: true,
    }];
  }

  private importedCallableEvidence(
    searchCall: ObservedToolCall | undefined,
    callableName: string
  ): { moduleName: string } | undefined {
    if (!searchCall || searchCall.toolName !== 'fs.search' || !searchCall.success) {
      return undefined;
    }
    const matches = (searchCall.result as {
      matches?: Array<{ preview?: unknown }>;
    } | undefined)?.matches;
    if (!Array.isArray(matches)) return undefined;
    const escapedCallable = callableName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    for (const match of matches) {
      const preview = String(match.preview ?? '');
      const imported = new RegExp(
        `\\bfrom\\s+([A-Za-z_][A-Za-z0-9_.]*)\\s+import\\s+[^\\n#]*\\b${escapedCallable}\\b`
      ).exec(preview);
      if (imported?.[1]) return { moduleName: imported[1] };
    }
    return undefined;
  }

  private pythonSignatureInspectionCommand(
    moduleName: string,
    callableName: string
  ): string {
    return `python -c "import inspect; from ${moduleName} import ${callableName}; print(inspect.signature(${callableName}))"`;
  }

  private verifierScorecardState(call: ObservedToolCall): {
    groups: Record<string, number>;
    weights: Record<string, number>;
  } | undefined {
    const diagnostics = (call.result as { verifierDiagnostics?: unknown } | undefined)
      ?.verifierDiagnostics;
    if (!Array.isArray(diagnostics)) return undefined;
    for (const item of diagnostics) {
      if (!item || typeof item !== 'object') continue;
      const content = (item as { content?: unknown }).content;
      if (typeof content !== 'string') continue;
      try {
        const parsed = JSON.parse(content) as { groups?: unknown; weights?: unknown };
        if (!parsed.groups
          || typeof parsed.groups !== 'object'
          || Array.isArray(parsed.groups)) {
          continue;
        }
        const groups = Object.fromEntries(
          Object.entries(parsed.groups)
            .filter((entry): entry is [string, number] =>
              typeof entry[1] === 'number' && Number.isFinite(entry[1])
            )
        );
        const weights = parsed.weights
          && typeof parsed.weights === 'object'
          && !Array.isArray(parsed.weights)
          ? Object.fromEntries(
            Object.entries(parsed.weights)
              .filter((entry): entry is [string, number] =>
                typeof entry[1] === 'number' && Number.isFinite(entry[1])
              )
          )
          : {};
        if (Object.keys(groups).length > 0) return { groups, weights };
      } catch {
        // Other verifier diagnostics can be free-form logs.
      }
    }
    return undefined;
  }

  private isMutableImplementationPath(candidatePath: string): boolean {
    const candidate = this.normalizeWorkspacePath(candidatePath);
    if (!candidate
      || /(?:^|\/)\.roy(?:\/|$)/i.test(candidate)
      || /(?:^|\/)(?:tests?|fixtures?|examples?|benchmarks?|configs?|data|datasets?|inputs?|outputs?|artifacts?|logs?|verifier)(?:\/|$)/i.test(
        candidate
      )) {
      return false;
    }
    return /(?:^|\/)(?:src|lib|app|packages)\/.+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php)$/i.test(
      candidate
    );
  }

  private latestRollbackForCurrentWorkspace(
    calls: ObservedToolCall[],
    targetPath?: string
  ): ReturnType<typeof workspaceCandidateRollbackFromCall> {
    const normalizedTarget = targetPath
      ? this.normalizeWorkspacePath(targetPath)
      : undefined;
    let unrolledMutationObserved = false;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const rollback = workspaceCandidateRollbackFromCall(call);
      const rollbackPath = this.normalizeWorkspacePath(String(rollback?.path ?? ''));
      if (rollback?.restored === true
        && (!normalizedTarget || rollbackPath === normalizedTarget)) {
        if (!unrolledMutationObserved) return rollback;
        return undefined;
      }
      const mutationPath = this.normalizeWorkspacePath(String(call.params.path ?? ''));
      if (isSuccessfulWorkspaceMutationCall(call)
        && (!normalizedTarget || mutationPath === normalizedTarget)) {
        unrolledMutationObserved = true;
      }
    }
    return undefined;
  }

  private verifierDiagnosticSummary(calls: ObservedToolCall[]): string | undefined {
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      if (call.toolName !== 'shell.exec' || !call.success) continue;
      const result = call.result as {
        command?: unknown;
        stdout?: unknown;
        stderr?: unknown;
      } | undefined;
      const command = String(result?.command ?? call.params.command ?? '');
      const output = [
        String(result?.stdout ?? ''),
        String(result?.stderr ?? ''),
      ].filter(Boolean).join('\n');
      if (!command.includes('ROY_VERIFIER_PROBE=1')
        && !output.includes('VERIFIER_PROBE_')) {
        continue;
      }
      if (!output.trim()) return undefined;
      const maxChars = 2_600;
      const lines = output.split('\n').map(line => line.trim()).filter(Boolean);
      const core = lines.filter(line =>
        /^VERIFIER_PROBE_(?:EVIDENCE_VERSION|CALL|MISMATCHES|ARGS|KWARGS|RESULT|REWARD|RETAINED_DIRS)\b/.test(
          line
        )
      );
      const specs = lines.filter(line =>
        /^VERIFIER_PROBE_(?:SPEC|TASK_INPUT)\b/.test(line)
      );
      const artifacts = lines
        .filter(line => /^VERIFIER_PROBE_ARTIFACT\b/.test(line))
        .map((line, position) => {
          let evidencePath = '';
          const payloadStart = line.indexOf('{');
          if (payloadStart >= 0) {
            try {
              const payload = JSON.parse(line.slice(payloadStart)) as {
                path?: unknown;
              };
              evidencePath = String(payload.path ?? '').toLowerCase();
            } catch {
              // A compacted artifact is still lower-priority usable evidence.
            }
          }
          let priority = 0;
          if (/(?:error|failure|traceback)/.test(evidencePath)) priority += 120;
          if (/(?:^|\/)outputs?(?:\/|$)/.test(evidencePath)) priority += 60;
          if (/(?:qc|summary|report|journal|diagnostic|result|reward|log)/.test(evidencePath)) {
            priority += 50;
          }
          if (/\.json\b/.test(evidencePath)) priority += 15;
          if (/(?:manifest|ocr|token|fixture|expected)/.test(evidencePath)) priority -= 80;
          if (/(?:reconstructed|prediction)/.test(evidencePath)) priority -= 20;
          return { line, position, priority };
        })
        .sort((left, right) =>
          right.priority - left.priority || left.position - right.position
        );
      const selected: string[] = [];
      let remaining = maxChars;
      const append = (line: string, perItemLimit: number): void => {
        if (remaining <= 0) return;
        const clipped = line.length <= perItemLimit
          ? line
          : `${line.slice(0, Math.max(0, perItemLimit - 28))}[diagnostic item compacted]`;
        if (clipped.length > remaining) return;
        selected.push(clipped);
        remaining -= clipped.length + 1;
      };
      core.forEach(line => append(line, 700));
      specs.forEach(line => append(line, 1_400));
      artifacts.forEach(item => append(item.line, 1_400));
      if (selected.length > 0) return selected.join('\n');
      return output.length <= maxChars
        ? output
        : `[${output.length - maxChars} earlier diagnostic chars compacted]\n${output.slice(-maxChars)}`;
    }
    return undefined;
  }

  private latestShellFailureIndex(calls: ObservedToolCall[]): number {
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      if (call.toolName === 'shell.exec'
        && (
          (!call.success && !isUnavailableWorkspaceVerificationCall(call))
          || (
            isWorkspaceVerificationCall(call)
            && !isUnavailableWorkspaceVerificationCall(call)
            && !isSuccessfulWorkspaceVerificationCall(call)
          )
        )) {
        return index;
      }
    }
    return -1;
  }

  private normalizeWorkspacePath(value: string): string {
    return value.trim().replace(/\\/g, '/').replace(/^(?:\.\/)+/, '').replace(/\/+/g, '/');
  }

  private extractVerifierDiagnosticText(value: unknown): string[] {
    const text: string[] = [];
    const visit = (item: unknown, depth: number): void => {
      if (depth > 5 || item === null || item === undefined) return;
      if (typeof item === 'string') {
        text.push(item);
        const trimmed = item.trim();
        if ((trimmed.startsWith('{') || trimmed.startsWith('[')) && trimmed.length < 100_000) {
          try {
            visit(JSON.parse(trimmed), depth + 1);
          } catch {
            // The diagnostic may be ordinary traceback text rather than JSON.
          }
        }
        return;
      }
      if (Array.isArray(item)) {
        item.slice(0, 40).forEach(entry => visit(entry, depth + 1));
        return;
      }
      if (typeof item === 'object') {
        Object.values(item as Record<string, unknown>)
          .slice(0, 80)
          .forEach(entry => visit(entry, depth + 1));
      }
    };
    visit(value, 0);
    return text;
  }

  hasSufficientWebEvidence(task: string, calls: ObservedToolCall[]): boolean {
    return this.relevantWebDocuments(task, calls).size >= this.requiredWebSourceCount(task);
  }

  isWebCandidateAligned(task: string, candidate: string): boolean {
    const coreTerms = this.webCoreEntityTerms(task);
    if (coreTerms.length === 0) return true;
    const candidateTokens = this.webCandidateTokens(candidate);
    return coreTerms.some(term => candidateTokens.has(term)
      || (term.length >= 5 && [...candidateTokens].some(token => token.includes(term))));
  }

  private extractReferencedPaths(task: string): string[] {
    const matches = task.matchAll(/(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+(?:\/|\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh|json|jsonl|csv|md|toml|ini|cfg|txt|xml|yaml|yml)))(?=$|[.\s`'"),:;])/g);
    const fileExtension = /\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh|json|jsonl|csv|md|toml|ini|cfg|txt|xml|yaml|yml)$/i;
    return [...new Set([...matches]
      .map(match => match[1].replace(/^\.\//, ''))
      .filter(value => value.length > 0)
      .flatMap(value => {
        const segments = value.split('/');
        // Natural-language tasks commonly use "setup.py/pyproject.toml" or
        // "package.json/tsconfig.json" to mean alternatives, not a child path
        // below a file. Preserve real directory paths and split only this
        // unambiguous two-file shorthand.
        return segments.length === 2 && segments.every(segment => fileExtension.test(segment))
          ? segments
          : [value];
      }))];
  }

  private taskDeclaredDirectoryPath(task: string, rawPath: string): string | undefined {
    const normalizedPath = this.normalizeWorkspacePath(rawPath);
    if (!normalizedPath || normalizedPath.includes('/')) return undefined;
    const escapedPath = normalizedPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    for (const match of task.matchAll(new RegExp(`(?:^|\\n)[^\\n]*[\\x60'"]?${escapedPath}[\\x60'"]?(?=$|[.\\s\\x60'"),:;])`, 'gi'))) {
      const occurrence = match.index ?? 0;
      const context = task.slice(Math.max(0, occurrence - 1_200), occurrence);
      const headingBoundary = Math.max(
        context.lastIndexOf('\n## '),
        context.lastIndexOf('\n# ')
      );
      const section = context.slice(Math.max(0, headingBoundary));
      const directoryMatches = [...section.matchAll(
        /\b(?:under|inside|into|within|to)\s+[`'"]?((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)+)[`'"]?/gi
      )];
      const directory = directoryMatches.at(-1)?.[1];
      if (directory
        && /\b(?:artifacts?|outputs?|reports?|files?|write|create|produce|emit|save)\b/i.test(section)) {
        return `${directory}${normalizedPath}`;
      }
    }
    return undefined;
  }

  private extractFailureLocations(
    output: string,
    cwd: string,
    workspaceRoot?: string
  ): Array<{ path: string; line?: number }> {
    const locations: Array<{ path: string; line?: number }> = [];
    const add = (rawPath: string, rawLine?: string): void => {
      const parsedLine = rawLine === undefined ? undefined : Number(rawLine);
      const line = Number.isInteger(parsedLine) && Number(parsedLine) > 0
        ? Number(parsedLine)
        : undefined;
      let candidate = rawPath.trim().replace(/\\/g, '/');
      const normalizedCwd = cwd.trim().replace(/\\/g, '/').replace(/\/+$/, '');
      const normalizedWorkspaceRoot = String(workspaceRoot ?? '')
        .trim()
        .replace(/\\/g, '/')
        .replace(/\/+$/, '');
      const roots = [normalizedWorkspaceRoot, normalizedCwd]
        .filter(Boolean)
        .sort((left, right) => right.length - left.length);
      for (const root of roots) {
        if (candidate === root) {
          candidate = '.';
          break;
        }
        if (candidate.startsWith(`${root}/`)) {
          candidate = candidate.slice(root.length + 1);
          break;
        }
      }
      candidate = candidate.replace(/^\.\//, '');
      if (!candidate
        || candidate.startsWith('/')
        || candidate.startsWith('../')
        || !/\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php)$/i.test(candidate)) {
        return;
      }
      locations.push({ path: candidate, line });
    };
    for (const match of output.matchAll(
      /(?:^|[\s"'`([{])((?:\.{0,2}\/)?(?:[A-Za-z0-9_.@-]+\/)+[A-Za-z0-9_.@-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php))(?=$|[\s"'`,:;)\]}])/g
    )) {
      add(String(match[1]));
    }
    for (const match of output.matchAll(/\bFile\s+["']([^"']+)["'],\s+line\s+(\d+)/g)) {
      add(String(match[1]), String(match[2]));
    }
    for (const match of output.matchAll(/(?:^|\n)\s*([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php)):(\d+)(?::\d+)?/g)) {
      add(String(match[1]), String(match[2]));
    }
    for (const match of output.matchAll(/\(([^()\s]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php))\)/g)) {
      add(String(match[1]));
    }
    const seen = new Set<string>();
    return locations.reverse().filter(location => {
      const key = location.path;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  private rejectedImportConsumerPath(
    calls: ObservedToolCall[],
    rejectedPath: string,
    workspaceRoot?: string
  ): string | undefined {
    const normalizedRejectedPath = this.normalizeWorkspacePath(rejectedPath);
    if (!normalizedRejectedPath) return undefined;
    let rollbackIndex = -1;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const rollback = workspaceCandidateRollbackFromCall(calls[index]!);
      if (rollback?.restored !== true) continue;
      if (this.normalizeWorkspacePath(String(rollback.path ?? ''))
        !== normalizedRejectedPath) {
        continue;
      }
      rollbackIndex = index;
      break;
    }
    if (rollbackIndex < 0) return undefined;
    for (let index = rollbackIndex; index >= 0; index -= 1) {
      const call = calls[index]!;
      if (call.toolName !== 'shell.exec') continue;
      const shell = call.result as {
        cwd?: unknown;
        stdout?: unknown;
        stderr?: unknown;
        verifierDiagnostics?: unknown;
      } | undefined;
      const output = [
        String(shell?.stdout ?? ''),
        String(shell?.stderr ?? ''),
        String(call.error ?? ''),
        ...this.extractVerifierDiagnosticText(shell?.verifierDiagnostics),
      ].filter(Boolean).join('\n');
      const importFailure = /ImportError:\s+cannot import name\s+['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s+from\s+['"][^'"]+['"]\s+\(([^)\n]+)\)/i.exec(
        output
      );
      if (!importFailure) continue;
      const providerLocation = this.extractFailureLocations(
        importFailure[2]!,
        String(shell?.cwd ?? workspaceRoot ?? ''),
        workspaceRoot
      )[0]?.path;
      if (providerLocation && providerLocation !== normalizedRejectedPath) {
        continue;
      }
      const symbol = importFailure[1]!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const importingFrame = new RegExp(
        `File\\s+["']([^"']+\\.py)["'],\\s+line\\s+(\\d+)[^\\n]*\\n\\s*(?:from\\s+[^\\n]+\\s+import\\s+[^\\n]*\\b${symbol}\\b|import\\s+[^\\n]*\\b${symbol}\\b)`,
        'gi'
      );
      const frames = [...output.matchAll(importingFrame)];
      for (const frame of frames.reverse()) {
        const consumer = this.extractFailureLocations(
          `File "${String(frame[1])}", line ${String(frame[2])}`,
          String(shell?.cwd ?? workspaceRoot ?? ''),
          workspaceRoot
        )[0]?.path;
        if (consumer
          && consumer !== normalizedRejectedPath
          && this.isMutableImplementationPath(consumer)) {
          return consumer;
        }
      }
    }
    return undefined;
  }

  private looksLikeShellMutation(command: string): boolean {
    return /\b(?:apply_patch|touch|mkdir|cp|mv|rm|install|chmod|truncate|sed\s+-i|perl\s+-pi)\b/i.test(command)
      || /\b(?:python|python3|node)\b[\s\S]*(?:writeFile|write_text|write_bytes|open\s*\([^)]*['"][wa]['"])/i.test(command)
      || /(?:^|[;&|]\s*)(?:echo|printf)\b[^\n]*(?:>>?|tee)\s*\S+/i.test(command);
  }

  private taskRequestsWorkspaceMutation(task: string): boolean {
    return /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor|migrate|upgrade|replace|apply)\b/i.test(
      task
    ) || /(?:实现|修改|编辑|创建|写入|修复|迁移|升级|替换|应用补丁)/.test(task);
  }

  private extractImplementationContractTerms(task: string): string[] {
    const terms = [
      ...task.matchAll(
        /\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b/g
      ),
      ...task.matchAll(
        /\b(?:[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b/g
      ),
      ...task.matchAll(/[`'"]([A-Za-z_][A-Za-z0-9_.]{3,})[`'"]/g),
    ].map(match => String(match[1] ?? match[0] ?? '').replace(/[()]+$/, ''));
    const ignored = new Set([
      'agent', 'apply', 'config', 'current', 'exact', 'file',
      'implementation', 'inspect', 'modify', 'none', 'preserve', 'replace',
      'runtime', 'source', 'task', 'true', 'false', 'verify', 'workspace',
    ]);
    return [...new Set(terms)]
      .filter(term => {
        const lower = term.toLowerCase();
        if (term.length < 4 || ignored.has(lower)) return false;
        if (/\.(?:cfg|csv|ini|json|jsonl|md|py|txt|toml|xml|ya?ml)$/i.test(
          term
        )) {
          return false;
        }
        return true;
      })
      .slice(0, 80);
  }

  private recoveryDependencyManifestPaths(
    summary: string,
    task: string
  ): string[] {
    if (!/\b(?:dependenc(?:y|ies)|manifest|metadata|runtime(?:\s+version)?|version constraint|package install|pin(?:ned|ning)?)\b/i.test(
      summary
    ) || !/\b(?:fail(?:ed|ure)?|error|legacy|outdated|incorrect|wrong|incompatible|unsupported|mismatch|must|should|expect(?:ed)?|target(?:s|ed)?|upgrade|downgrade|missing|cannot|can't|not)\b/i.test(
      summary
    )) {
      return [];
    }
    const context = `${summary}\n${task}`.toLowerCase();
    if (/\b(?:python|pip|pytest|pydantic|langchain|setuptools)\b/.test(context)) {
      return ['pyproject.toml', 'requirements.txt'];
    }
    if (/\b(?:node|npm|pnpm|yarn|bun|typescript|javascript)\b/.test(context)) {
      return ['package.json'];
    }
    if (/\b(?:rust|cargo)\b/.test(context)) return ['Cargo.toml'];
    if (/\b(?:golang|go\.mod|go module)\b/.test(context)) return ['go.mod'];
    if (/\b(?:maven|gradle|java|kotlin)\b/.test(context)) {
      return ['pom.xml', 'build.gradle'];
    }
    if (/\b(?:ruby|bundler|gem)\b/.test(context)) return ['Gemfile'];
    return ['pyproject.toml', 'package.json', 'requirements.txt'];
  }

  private isEnvironmentSetupCommand(command: string): boolean {
    const normalized = command.trim();
    return /^(?:cd|pushd|popd)\b/i.test(normalized)
      || /^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|uv)\s+install\b/i.test(normalized)
      || /^(?:npm|pnpm|yarn|bun)\s+(?:install|ci)\b/i.test(normalized)
      || /^(?:apt-get|apt|apk|brew)\s+install\b/i.test(normalized);
  }

  private isDependencyInstallCommand(command: string): boolean {
    const normalized = command.trim().replace(/^(?:cd\s+\S+\s*&&\s*)+/i, '');
    return /^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|uv)\s+install\b/i.test(normalized)
      || /^(?:npm|pnpm|yarn|bun)\s+(?:install|ci)\b/i.test(normalized);
  }

  private isDependencyManifestPath(candidatePath: string): boolean {
    const path = this.normalizeWorkspacePath(candidatePath).toLowerCase();
    return /(?:^|\/)(?:pyproject\.toml|setup\.cfg|setup\.py|requirements[^/]*\.txt|package\.json|package-lock\.json|pnpm-lock\.ya?ml|yarn\.lock|bun\.lockb?|cargo\.toml|cargo\.lock|go\.mod|go\.sum|pom\.xml|build\.gradle|gemfile|gemfile\.lock)$/.test(
      path
    );
  }

  private extractExplicitShellCommands(task: string): string[] {
    const commands: string[] = [];
    const addCommand = (rawCommand: string): void => {
      const command = rawCommand.trim().replace(/^\$\s+/, '');
      if (!command
        || command.length > 1_000
        || command.endsWith('\\')
        || /\b(?:rm\s+-rf|mkfs|shutdown|reboot|halt)\b/i.test(command)
        || /^(?:the|a|an|this|that|these|those|we|you|it)\b/i.test(
          command.replace(/^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+/, '')
        )) {
        return;
      }
      commands.push(command);
    };
    for (const match of task.matchAll(/```(?:bash|sh|shell|zsh|console)\s*\n([\s\S]*?)```/gi)) {
      const index = match.index ?? 0;
      const leadIn = task.slice(Math.max(0, index - 240), index);
      if (!/(?:required|helpful|run|execute|verification|verify|test)[^\n]{0,80}(?:commands?|with|using)?/i.test(leadIn)) {
        continue;
      }
      const lines = String(match[1] ?? '')
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));
      for (const line of lines) {
        addCommand(line);
      }
    }
    for (const match of task.matchAll(
      /\b(?:run|execute|rerun|verify(?:\s+with)?|confirm(?:\s+with)?|command(?:\s+is)?|install(?:\s+it)?\s*:)\s+(?:the\s+command\s+)?(['"])([\s\S]{2,1000}?)\1/gi
    )) {
      addCommand(String(match[2] ?? ''));
    }
    return [...new Set(commands)].slice(0, 12);
  }

  private extractReferencedUrls(task: string): string[] {
    return [...new Set(task.match(/https?:\/\/[^\s`'"<>),]+/gi) ?? [])]
      .map(url => url.replace(/[.;:]+$/, ''));
  }

  private requiresWebEvidence(task: string): boolean {
    return /\b(?:web|internet|online|website|search|browse|news|up-to-date|citations?|official documentation|public documentation)\b/.test(task)
      || /\blatest\b[\s\S]*\b(?:documentation|release|version|news|announcement|api)\b/.test(task)
      || /\b(?:research|compare|verify)\b[\s\S]*\b(?:external|official|independent)\s+sources?\b/.test(task);
  }

  private buildSearchQuery(task: string): string {
    const normalized = task
      .replace(/https?:\/\/[^\s`'"<>),]+/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const withoutPrefix = normalized
      .replace(/^(?:use|consult)\s+(?:the\s+)?(?:public\s+)?(?:web|internet|online)\s+(?:sources?\s+)?to\s+/i, '')
      .replace(
        /^.*?\b(?:search|research|find|look up)\s+(?:(?:the|on)\s+)?(?:web|internet|online)?\s*(?:for\s+)?/i,
        ''
      );
    const firstObjective = withoutPrefix.split(/\.\s+/)[0];
    return firstObjective
      .replace(/^(?:compare|verify|summarize|explain|inspect|review|analy[sz]e)\s+/i, '')
      .replace(/^(?:the\s+)?(?:latest|current|up-to-date)\s+/i, '')
      .replace(/^official\s+/i, '')
      .replace(/\babout\b/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 300);
  }

  private extractIndependentWebQuestions(task: string): string[] {
    const questions: string[] = [];
    const seen = new Set<string>();
    const addQuestion = (value: string | undefined): void => {
      const question = value?.replace(/\s+/g, ' ').trim();
      if (!question || question.length < 8 || question.length > 300) return;
      const fingerprint = question.toLowerCase();
      if (seen.has(fingerprint)) return;
      seen.add(fingerprint);
      questions.push(question);
    };
    for (const match of task.matchAll(/(?:^|\s)\d{1,2}[.)]\s+([^\n?]{7,299}\?)/g)) {
      addQuestion(match[1]);
      if (questions.length >= 10) break;
    }
    if (questions.length < 10) {
      for (const line of task.split(/\r?\n/)) {
        const match = line.match(/^\s*[-*]\s+(.+\?)\s*$/);
        addQuestion(match?.[1]);
        if (questions.length >= 10) break;
      }
    }
    if (questions.length < 2
      && /\bquestions?(?:\s+\d+\s*-\s*\d+)?\s*:/i.test(task)) {
      const questionRegion = task.slice(
        task.search(/\bquestions?(?:\s+\d+\s*-\s*\d+)?\s*:/i)
      );
      for (const match of questionRegion.matchAll(/([^?\n]{7,299}\?)/g)) {
        addQuestion(String(match[1] ?? '')
          .replace(
            /^.*?\bquestions?(?:\s+\d+\s*-\s*\d+)?\s*:\s*/i,
            ''
          ));
        if (questions.length >= 10) break;
      }
    }
    return questions;
  }

  private isWebOnlyTask(task: string): boolean {
    if (!this.requiresWebEvidence(task)) return false;
    return !/\b(?:workspace|filesystem|local|project|codebase|repo|repository|source code|package\.json|files? and directories|directory tree)\b/.test(task);
  }

  webRelevanceScore(task: string, candidate: string): number {
    if (!this.isWebCandidateAligned(task, candidate)) return 0;
    const stopWords = new Set([
      'about', 'after', 'also', 'and', 'at', 'before', 'clearly', 'compare', 'concrete', 'current',
      'did', 'distinguish', 'does', 'establish', 'evidence', 'for', 'from', 'had', 'include', 'latest',
      'name', 'number', 'one', 'open', 'over', 'public', 'relevant', 'search', 'source', 'sources',
      'the', 'their', 'uncertainty', 'urls', 'use', 'using', 'verified', 'was', 'web', 'were', 'what',
      'when', 'where', 'which', 'who', 'with',
    ]);
    const terms = [...new Set(task.toLowerCase().split(/[^a-z0-9._-]+/)
      .map(term => term.replace(/^[._-]+|[._-]+$/g, ''))
      .flatMap(term => [term, ...term.split(/[._-]+/)])
      .filter(term => term.length >= 3 && !stopWords.has(term)))];
    const candidateTokens = new Set(candidate.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
    return terms.reduce((score, term) => {
      const normalizedTerm = term.replace(/[^a-z0-9]+/g, '');
      const aliases = normalizedTerm === 'documentation' ? ['documentation', 'docs', 'doc'] : [normalizedTerm];
      const exactOnly = normalizedTerm === 'official';
      const matched = aliases.some(alias => candidateTokens.has(alias)
        || (!exactOnly && [...candidateTokens].some(token => token.includes(alias))));
      if (!matched) return score;
      const weight = /^(?:abortsignal|timeout|globals)$/.test(normalizedTerm)
        ? 5
        : /^(?:api|fetch)$/.test(normalizedTerm)
          ? 4
          : /^(?:documentation|official)$/.test(normalizedTerm)
            ? 3
            : normalizedTerm === 'nodejs' ? 1 : 2;
      return score + weight;
    }, 0);
  }

  private webCoreEntityTerms(task: string): string[] {
    const toolSyntaxTerms = new Set([
      'web.search',
      'web.fetch',
      'websearch',
      'webfetch',
      'web',
      'search',
      'fetch',
    ]);
    const dotted = [...task.matchAll(/\b[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)+\b/g)]
      .flatMap(match => [match[0], ...match[0].split('.')]);
    const named = (task.match(/\b[A-Za-z][A-Za-z0-9]*\b/g) ?? [])
      .filter(term => /^[A-Z0-9]{2,}$/.test(term) || /[a-z][A-Z]/.test(term));
    return [...new Set([...dotted, ...named]
      .map(term => term.toLowerCase().replace(/[^a-z0-9]+/g, ''))
      .filter(term => !toolSyntaxTerms.has(term))
      .filter(term => term.length >= 3))];
  }

  private webCandidateTokens(candidate: string): Set<string> {
    const raw = candidate.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    return new Set([...raw, ...raw.flatMap(term => term.split(/[._-]+/))]);
  }

  webEvidenceScore(task: string, call: ObservedToolCall): number {
    if (call.toolName !== 'web.fetch' || !call.success) return 0;
    const page = call.result as { finalUrl?: unknown; title?: unknown; text?: unknown } | undefined;
    return this.webRelevanceScore(task, [
      String(page?.title ?? ''),
      String(page?.finalUrl ?? call.params.url ?? ''),
      String(page?.text ?? '').slice(0, 5000),
    ].join(' '));
  }

  private canonicalWebUrl(input: string): string {
    try {
      const url = new URL(input);
      return url.toString().replace(/\/$/, '');
    } catch {
      return input;
    }
  }

  private canonicalWebDocumentUrl(input: string): string {
    try {
      const url = new URL(input);
      url.hash = '';
      return url.toString().replace(/\/$/, '');
    } catch {
      return input;
    }
  }

  private requiredWebSourceCount(task: string): number {
    const explicitDocuments = new Set(this.extractReferencedUrls(task).map(url => this.canonicalWebDocumentUrl(url)));
    if (explicitDocuments.size >= 2) return 2;
    return /\b(?:at least|minimum of)\s+(?:two|2)\b|\b(?:two|2)\s+(?:independent|relevant|public)?\s*(?:pages?|websites?|urls?|sources?)\b/i.test(task)
      || /\bboth\s+(?:pages?|websites?|urls?|sources?|documents?)\b/i.test(task)
      ? 2
      : 1;
  }

  private relevantWebDocuments(task: string, calls: ObservedToolCall[]): Set<string> {
    const relevanceTasks = this.webEvidenceTasks(task);
    return new Set(calls
      .filter(call => call.toolName === 'web.fetch'
        && call.success
        && relevanceTasks.some(candidate => this.webEvidenceScore(candidate, call) >= 6))
      .map(call => this.canonicalWebDocumentUrl(String(
        (call.result as { finalUrl?: unknown } | undefined)?.finalUrl ?? call.params.url ?? ''
      )))
      .filter(Boolean));
  }

  private webEvidenceTasks(task: string): string[] {
    const questions = this.extractIndependentWebQuestions(task);
    return questions.length > 1 ? questions : [task];
  }
}
