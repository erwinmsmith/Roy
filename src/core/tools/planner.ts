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
}

export class AgentToolPlanner {
  plan(input: ToolPlanningInput): PlannedToolCall[] {
    const enabled = new Set(input.bindings.filter(binding => binding.enabled).map(binding => binding.name));
    const lower = input.task.toLowerCase();
    const plans: PlannedToolCall[] = [];

    if (enabled.has('fs.list') && /\b(inspect|analy[sz]e|review|list|structure|project|codebase|repo|repository|files?)\b/.test(lower)) {
      plans.push({
        toolName: 'fs.list',
        params: { path: input.workspacePath, maxDepth: 2 },
        reason: 'The task requires concrete workspace structure evidence.',
        groundingRequired: true,
      });
    }

    const filePath = input.task.match(/(?:^|\s)(\.?\.?\/[A-Za-z0-9._/@-]+|[A-Za-z0-9_/-]+\.(?:ts|tsx|js|json|md|yaml|yml))(?:\s|$)/)?.[1];
    if (enabled.has('fs.read') && filePath && /\b(read|inspect|review|check|open)\b/.test(lower)) {
      plans.push({
        toolName: 'fs.read',
        params: { path: filePath },
        reason: `The task explicitly references ${filePath}.`,
        groundingRequired: true,
      });
    }

    if (enabled.has('shell.exec')) {
      if (/\b(?:run|execute)\s+(?:the\s+)?tests?\b|\bnpm test\b/.test(lower)) {
        plans.push({ toolName: 'shell.exec', params: { command: 'npm test' }, reason: 'The task explicitly requests the test suite.', groundingRequired: true });
      } else if (/\b(?:run|execute)\s+(?:the\s+)?build\b|\bnpm run build\b/.test(lower)) {
        plans.push({ toolName: 'shell.exec', params: { command: 'npm run build' }, reason: 'The task explicitly requests the build.', groundingRequired: true });
      }
    }

    return plans.slice(0, 3);
  }
}
