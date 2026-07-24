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

  planWorkspaceFailureFollowUps(input: {
    calls: ObservedToolCall[];
    bindings: ToolPlanBinding[];
    workspaceRoot?: string;
  }): PlannedToolCall[] {
    if (!input.bindings.some(binding => binding.enabled && binding.name === 'fs.read')) return [];
    let latestFailureIndex = -1;
    for (let index = input.calls.length - 1; index >= 0; index -= 1) {
      const call = input.calls[index]!;
      if (call.toolName === 'shell.exec' && !call.success) {
        latestFailureIndex = index;
        break;
      }
    }
    if (latestFailureIndex < 0) return [];
    const failure = input.calls[latestFailureIndex]!;
    const shell = failure.result as {
      cwd?: unknown;
      stdout?: unknown;
      stderr?: unknown;
    } | undefined;
    const output = [
      String(shell?.stdout ?? ''),
      String(shell?.stderr ?? ''),
      String(failure.error ?? ''),
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
          call.success && (call.toolName === 'fs.write' || call.toolName === 'fs.replace'
            || call.toolName === 'shell.exec' && this.looksLikeShellMutation(String(call.params.command ?? '')))
        );
      })
      .slice(0, 1);
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
    const matches = task.matchAll(/(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+(?:\/|\.(?:ts|tsx|js|mjs|cjs|json|md|yaml|yml)))(?=$|[.\s`'"),:;])/g);
    return [...new Set([...matches]
      .map(match => match[1].replace(/^\.\//, ''))
      .filter(value => value.length > 0))];
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

  private extractExplicitShellCommands(task: string): string[] {
    const commands: string[] = [];
    for (const match of task.matchAll(/```(?:bash|sh|shell|zsh|console)\s*\n([\s\S]*?)```/gi)) {
      const index = match.index ?? 0;
      const leadIn = task.slice(Math.max(0, index - 240), index);
      if (!/(?:required|run|execute|verification|verify|test)[^\n]{0,80}(?:command|with|using)?/i.test(leadIn)) {
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
    return [...new Set(commands)].slice(0, 3);
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
