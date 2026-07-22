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
    const runtimeApiInspection = /\bruntime\s+apis?\b[\s\S]{0,80}\b(?:exports?|surface|inspection|declarations?|signatures?|source|symbols?)\b|\bexported runtime apis?\b/.test(lower);

    if (enabled.has('web.fetch') && referencedUrls.length > 0) {
      plans.push(...referencedUrls.map(url => ({
        toolName: 'web.fetch',
        params: { url },
        reason: `The task explicitly references ${url}.`,
        groundingRequired: true,
      })));
    } else if (enabled.has('web.search') && this.requiresWebEvidence(lower)) {
      plans.push({
        toolName: 'web.search',
        params: { query: this.buildSearchQuery(input.task), maxResults: 5 },
        reason: 'The task requires current or externally verifiable web evidence.',
        groundingRequired: true,
      });
    }

    if (enabled.has('fs.list') && referencedDirectories.length > 0) {
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
      const explicitTestRun = /\b(?:run|execute)\s+(?:the\s+)?tests?\b|\bnpm test\b/.test(lower);
      const testerVerification = input.archetype === 'tester'
        && (/\b(?:verify|validate|check)\b[\s\S]*\b(?:tests?|claims?|behavio(?:u)?r|failure cases?)\b/.test(lower)
          || /\b(?:test coverage|coverage gaps?|regression risk)\b/.test(lower));
      if (explicitTestRun || testerVerification) {
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

    return plans.slice(0, 3);
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
