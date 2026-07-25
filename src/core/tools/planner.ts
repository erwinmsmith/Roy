import {
  effectiveWorkspaceMutationCallIndices,
  isSuccessfulWorkspaceMutationCall,
  isSuccessfulWorkspaceVerificationCall,
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

export class AgentToolPlanner {
  plan(input: ToolPlanningInput): PlannedToolCall[] {
    const enabled = new Set(input.bindings.filter(binding => binding.enabled).map(binding => binding.name));
    const lower = input.task.toLowerCase();
    const plans: PlannedToolCall[] = [];
    const referencedPaths = this.extractReferencedPaths(input.task.replace(/https?:\/\/[^\s`'"<>),]+/gi, ' '));
    const referencedDirectories = referencedPaths.filter(item => item.endsWith('/'));
    const referencedFiles = referencedPaths.filter(item => !item.endsWith('/'));
    const referencedUrls = this.extractReferencedUrls(input.task);
    const explicitShellCommands = this.extractExplicitShellCommands(input.task);
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
      plans.push({
        toolName: 'web.search',
        params: { query: this.buildSearchQuery(input.task), maxResults: 5 },
        reason: 'The task requires current or externally verifiable web evidence.',
        groundingRequired: true,
      });
    }

    if (enabled.has('fs.list') && !this.isWebOnlyTask(lower) && (mutationTask || broadWorkspaceInspection)) {
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
    return plans.filter(plan => {
      const fingerprint = `${plan.toolName}:${JSON.stringify(plan.params)}`;
      if (seen.has(fingerprint)) return false;
      seen.add(fingerprint);
      return true;
    }).slice(0, 3);
  }

  planWebFollowUps(input: {
    task: string;
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    maxFetches: number;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding => binding.enabled && binding.name === 'web.fetch')) return [];
    const requiredSources = this.requiredWebSourceCount(input.task);
    const relevantDocuments = this.relevantWebDocuments(input.task, input.calls);
    const focusedSections = input.calls.filter(call => {
      if (call.toolName !== 'web.fetch' || !call.success || this.webEvidenceScore(input.task, call) < 6) return false;
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
      const resultUrls = new Set(results
        .filter(item => typeof item.url === 'string')
        .map(item => this.canonicalWebDocumentUrl(String(item.url))));
      const alreadyFetched = [...fetched].filter(url => resultUrls.has(url)).length;
      const remaining = Math.max(0, input.maxFetches - alreadyFetched);
      const searchPlans = results
        .filter(item => typeof item.url === 'string' && !fetched.has(this.canonicalWebDocumentUrl(item.url)))
        .map(item => ({ item, score: this.webRelevanceScore(input.task, `${String(item.title ?? '')} ${String(item.snippet ?? '')} ${String(item.url)}`) }))
        .filter(candidate => candidate.score >= 4)
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
      .map(link => ({ link, score: this.webRelevanceScore(input.task, `${String(link.text ?? '')} ${String(link.url)}`) }))
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
    return commands
      .filter(command => {
        const matchingCalls = input.calls.filter(call =>
          call.toolName === 'shell.exec'
          && String(call.params.command ?? '').trim() === command.trim()
        );
        if (this.isEnvironmentSetupCommand(command)
          && matchingCalls.some(call => call.success)) {
          return false;
        }
        return !input.calls.some((call, index) =>
          index > latestMutationIndex
          && call.toolName === 'shell.exec'
          && String(call.params.command ?? '').trim() === command.trim()
        );
      })
      .slice(0, 2)
      .map(command => ({
        toolName: 'shell.exec',
        params: { command },
        reason: 'Run the task-declared acceptance command against the newly mutated workspace before closing execution.',
        groundingRequired: true,
      }));
  }

  planExternalFeedbackRepair(input: {
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
    const feedbackMatch = /<official_verifier_feedback>([\s\S]*?)<\/official_verifier_feedback>/i.exec(
      input.task
    );
    const continuationMatch = /##\s+VERIFICATION FAILED\b([\s\S]*)/i.exec(input.task);
    const feedback = String(feedbackMatch?.[1] ?? continuationMatch?.[1] ?? '').trim();
    if (!feedback) return [];
    const dependencyFeedback = /\b(?:dependenc(?:y|ies)|metadata|manifest|requirements?|runtime version|version constraint|pin(?:ned|ning)?|package install)\b/i.test(
      feedback
    );
    const feedbackLocations = this.extractFailureLocations(
      feedback,
      input.workspaceRoot ?? '',
      input.workspaceRoot
    );
    const explicitFeedbackPaths = new Set([
      ...this.extractReferencedPaths(feedback),
      ...feedbackLocations.map(location => location.path),
    ].map(path => this.normalizeWorkspacePath(path)));
    const mutatedPaths = new Set(input.calls
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
      .filter(candidate => candidate.path && candidate.content && !mutatedPaths.has(candidate.path))
      .map(candidate => {
        const lowerPath = candidate.path.toLowerCase();
        let score = explicitFeedbackPaths.has(candidate.path) ? 500 : 0;
        if (dependencyFeedback) {
          if (/(?:^|\/)pyproject\.toml$/.test(lowerPath)) score += 420;
          else if (/(?:^|\/)requirements[^/]*\.txt$/.test(lowerPath)) score += 400;
          else if (/(?:^|\/)package\.json$/.test(lowerPath)) score += 380;
          else if (/(?:^|\/)(?:cargo\.toml|go\.mod|pom\.xml|build\.gradle|gemfile)$/.test(lowerPath)) score += 360;
          else if (/(?:^|\/)[^/]*(?:lock|manifest|dependencies)[^/]*\.(?:json|toml|ya?ml|txt)$/.test(lowerPath)) score += 320;
        }
        const basename = candidate.path.slice(candidate.path.lastIndexOf('/') + 1);
        if (feedback.toLowerCase().includes(basename.toLowerCase())) score += 160;
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
    return [{
      toolName: 'fs.synthesize',
      params: {
        path: candidate.path,
        instructions: [
          'Apply the smallest coherent, interface-preserving change that resolves the newest external verifier feedback.',
          `Authoritative external feedback:\n${compactFeedback}`,
          'Use the already observed current file as the patch base. Preserve unrelated working declarations and do not alter benchmark or verifier files.',
        ].join('\n\n'),
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
    const failure = input.calls[latestFailureIndex]!;
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
    const locations = this.extractFailureLocations(
      output,
      String(shell?.cwd ?? input.workspaceRoot ?? ''),
      input.workspaceRoot
    );
    return locations
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
        for (let index = input.calls.length - 1; index > latestFailureIndex; index -= 1) {
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
    const failure = input.calls[latestFailureIndex]!;
    if (!isWorkspaceVerificationCall(failure)
      || isSuccessfulWorkspaceVerificationCall(failure)) {
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
    const shell = failure.result as {
      cwd?: unknown;
      stdout?: unknown;
      stderr?: unknown;
      verifierDiagnostics?: unknown;
    } | undefined;
    const rejectedCandidate = this.latestRollbackForCurrentWorkspace(input.calls);
    const failureOutput = [
      String(shell?.stdout ?? ''),
      String(shell?.stderr ?? ''),
      String(failure.error ?? ''),
      ...this.extractVerifierDiagnosticText(shell?.verifierDiagnostics),
    ].filter(Boolean).join('\n');
    const failureLocations = this.extractFailureLocations(
      failureOutput,
      String(shell?.cwd ?? input.workspaceRoot ?? ''),
      input.workspaceRoot
    );
    if (failureLocations.length > 0 && !rejectedCandidate) {
      const targetPath = failureLocations[0]!.path;
      let sourceReadIndex = -1;
      for (let index = input.calls.length - 1; index > latestFailureIndex; index -= 1) {
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
      const localizedFailure = failureOutput.length <= 2_400
        ? failureOutput
        : `[earlier failure output compacted]\n${failureOutput.slice(-2_400)}`;
      const instructions = [
        'Apply the smallest interface-preserving patch that fixes this exact localized execution failure.',
        `Authoritative failure:\n${localizedFailure}`,
        'Use the freshly read current source as the only patch base. Initialize values on every control-flow path, preserve already working behavior, and do not broaden the change beyond the causal failure.',
      ].join('\n\n');
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
        reason: 'A fresh source read now grounds the exact traceback location; transition directly from diagnosis to a minimal local repair.',
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
      return /(?:^|\/)(?:src|lib|app|packages)\/.+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php)$/i.test(candidate)
        && !/(?:^|\/)(?:tests?|fixtures?|examples?)\//i.test(candidate);
    });
    const implementationRead = implementationReads.find(call => {
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
    const verifierGroups = {
      ...(targetRejectedCandidate?.baselineGroups
        ?? latestScorecard?.groups
        ?? {}),
    };
    for (const group of [
      ...(targetRejectedCandidate?.regressedGroups ?? []),
      ...(targetRejectedCandidate?.improvedGroups ?? []),
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
    const instructions = targetRejectedCandidate
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
        targetRejectedCandidate.regressedGroups?.length
          ? `Do not repeat the rejected regression in: ${targetRejectedCandidate.regressedGroups.map(item => item.group).join(', ')}.`
          : 'Do not repeat the rejected whole-file strategy; make the smallest coherent semantic change supported by the official verifier source.',
        diagnosticSummary
          ? `Use this newest focused expected-versus-actual verifier reproduction as the causal repair evidence:\n${diagnosticSummary}`
          : '',
        'Treat the grounded official verifier implementation as the executable specification and keep public interfaces stable.',
      ].filter(Boolean).join(' ')
      : 'Structurally repair the implementation to satisfy the grounded aggregate official-verifier failures and immutable assignment, preserving behavior already proven by passing verifier groups.';
    if (targetRejectedCandidate && input.calls.some(call =>
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
        ...(targetRejectedCandidate ? { strategy: 'patch' } : {}),
      },
      reason: targetRejectedCandidate
        ? 'The prior candidate was rolled back; change the repair hypothesis to the unresolved verifier capabilities while preserving the accepted baseline.'
        : 'The causal frontier contains a non-perfect aggregate verifier result plus complete verifier and implementation source evidence; transition from inspection to a preserving repair.',
      groundingRequired: true,
    }];
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
      const specs = lines.filter(line => /^VERIFIER_PROBE_SPEC\b/.test(line));
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
          !call.success
          || (
            isWorkspaceVerificationCall(call)
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
    return [...new Set([...matches]
      .map(match => match[1].replace(/^\.\//, ''))
      .filter(value => value.length > 0))];
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

  private looksLikeShellMutation(command: string): boolean {
    return /\b(?:apply_patch|touch|mkdir|cp|mv|rm|install|chmod|truncate|sed\s+-i|perl\s+-pi)\b/i.test(command)
      || /\b(?:python|python3|node)\b[\s\S]*(?:writeFile|write_text|write_bytes|open\s*\([^)]*['"][wa]['"])/i.test(command)
      || /(?:^|[;&|]\s*)(?:echo|printf)\b[^\n]*(?:>>?|tee)\s*\S+/i.test(command);
  }

  private isEnvironmentSetupCommand(command: string): boolean {
    const normalized = command.trim();
    return /^(?:cd|pushd|popd)\b/i.test(normalized)
      || /^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|uv)\s+install\b/i.test(normalized)
      || /^(?:npm|pnpm|yarn|bun)\s+(?:install|ci)\b/i.test(normalized)
      || /^(?:apt-get|apt|apk|brew)\s+install\b/i.test(normalized);
  }

  private extractExplicitShellCommands(task: string): string[] {
    const commands: string[] = [];
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
        const command = line.replace(/^\$\s+/, '');
        if (command.length > 1_000
          || command.endsWith('\\')
          || /\b(?:rm\s+-rf|mkfs|shutdown|reboot|halt)\b/i.test(command)) {
          continue;
        }
        commands.push(command);
      }
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

  private isWebOnlyTask(task: string): boolean {
    if (!this.requiresWebEvidence(task)) return false;
    return !/\b(?:workspace|filesystem|local|project|codebase|repo|repository|source code|package\.json|files? and directories|directory tree)\b/.test(task);
  }

  webRelevanceScore(task: string, candidate: string): number {
    if (!this.isWebCandidateAligned(task, candidate)) return 0;
    const stopWords = new Set([
      'about', 'after', 'also', 'and', 'at', 'before', 'clearly', 'compare', 'concrete', 'current',
      'distinguish', 'establish', 'evidence', 'for', 'from', 'include', 'latest',
      'open', 'public', 'relevant', 'search', 'source', 'sources', 'the', 'their', 'uncertainty',
      'urls', 'use', 'using', 'verified', 'web', 'what', 'with',
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
    const dotted = [...task.matchAll(/\b[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)+\b/g)]
      .flatMap(match => [match[0], ...match[0].split('.')]);
    const named = (task.match(/\b[A-Za-z][A-Za-z0-9]*\b/g) ?? [])
      .filter(term => /^[A-Z0-9]{2,}$/.test(term) || /[a-z][A-Z]/.test(term));
    return [...new Set([...dotted, ...named]
      .map(term => term.toLowerCase().replace(/[^a-z0-9]+/g, ''))
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
    return new Set(calls
      .filter(call => call.toolName === 'web.fetch' && call.success && this.webEvidenceScore(task, call) >= 6)
      .map(call => this.canonicalWebDocumentUrl(String(
        (call.result as { finalUrl?: unknown } | undefined)?.finalUrl ?? call.params.url ?? ''
      )))
      .filter(Boolean));
  }
}
