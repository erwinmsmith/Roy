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

    if (enabled.has('fs.list') && /\b(inspect|analy[sz]e|review|list|structure|project|codebase|repo|repository|files?|evidence|coverage|verify)\b/.test(lower)) {
      plans.push({
        toolName: 'fs.list',
        params: { path: input.workspacePath, maxDepth: 2 },
        reason: 'The task requires concrete workspace structure evidence.',
        groundingRequired: true,
      });
    }

    const explicitFilePath = input.task.match(/(?:^|\s)(\.?\.?\/[A-Za-z0-9._/@-]+|[A-Za-z0-9_/-]+\.(?:ts|tsx|js|json|md|yaml|yml))(?:\s|$)/)?.[1];
    const inferredFilePath = /\b(?:package exports?|export map|package manifest|package entr(?:y|ies))\b/.test(lower)
      ? 'package.json'
      : input.archetype === 'critic'
        && /\b(?:architecture|architectural|repository|codebase|dependency|coupling)\b/.test(lower)
        ? 'package.json'
      : undefined;
    const filePath = explicitFilePath ?? inferredFilePath;
    if (enabled.has('fs.read') && filePath && /\b(read|inspect|review|check|open|identify|analy[sz]e)\b/.test(lower)) {
      plans.push({
        toolName: 'fs.read',
        params: { path: filePath },
        reason: explicitFilePath
          ? `The task explicitly references ${filePath}.`
          : `The package export request requires evidence from ${filePath}.`,
        groundingRequired: true,
      });
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
}
