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

export class AgentToolPlanner {
  plan(input: ToolPlanningInput): PlannedToolCall[] {
    const enabled = new Set(input.bindings.filter(binding => binding.enabled).map(binding => binding.name));
    const lower = input.task.toLowerCase();
    const plans: PlannedToolCall[] = [];
    const referencedPaths = this.extractReferencedPaths(input.task);
    const referencedDirectories = referencedPaths.filter(item => item.endsWith('/'));
    const referencedFiles = referencedPaths.filter(item => !item.endsWith('/'));

    if (enabled.has('fs.list') && referencedDirectories.length > 0) {
      plans.push(...referencedDirectories.map(directory => ({
        toolName: 'fs.list',
        params: { path: directory.replace(/\/$/, ''), maxDepth: 3 },
        reason: `The task explicitly requests directory evidence from ${directory}.`,
        groundingRequired: true,
      })));
    } else if (enabled.has('fs.list') && /\b(inspect|analy[sz]e|review|list|structure|project|codebase|repo|repository|files?|evidence|coverage|verify)\b/.test(lower)) {
      plans.push({
        toolName: 'fs.list',
        params: { path: input.workspacePath, maxDepth: 2 },
        reason: 'The task requires concrete workspace structure evidence.',
        groundingRequired: true,
      });
    }

    const inferredFilePath = /\b(?:package exports?|export map|package manifest|package entr(?:y|ies))\b/.test(lower)
      ? 'package.json'
      : input.archetype === 'critic'
        && /\b(?:architecture|architectural|repository|codebase|dependency|coupling)\b/.test(lower)
        ? 'package.json'
      : undefined;
    const filePaths = referencedFiles.length > 0
      ? referencedFiles
      : inferredFilePath ? [inferredFilePath] : [];
    if (enabled.has('fs.read') && filePaths.length > 0 && /\b(read|inspect|review|check|open|identify|analy[sz]e)\b/.test(lower)) {
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

  private extractReferencedPaths(task: string): string[] {
    const matches = task.matchAll(/(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+(?:\/|\.(?:ts|tsx|js|mjs|cjs|json|md|yaml|yml)))(?=$|[.\s`'"),:;])/g);
    return [...new Set([...matches]
      .map(match => match[1].replace(/^\.\//, ''))
      .filter(value => value.length > 0))];
  }
}
