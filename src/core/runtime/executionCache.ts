export type ExecutionFeedbackKind =
  | 'tool_failure'
  | 'actor_failure'
  | 'path_observation'
  | 'workspace_mutation'
  | 'workspace_verification'
  | 'external_feedback'
  | 'unresolved_gap';

export interface ExecutionCachedActor {
  id: string;
  runtimeActorId: string;
  kind: 'agent' | 'team';
  correlationId: string;
  stepId: string;
  pathId: string;
  name: string;
  role: string;
  parentId?: string;
  teamId?: string;
  generation: number;
  task?: string;
  taskFingerprint?: string;
  definitionFingerprint?: string;
  status: 'active' | 'done' | 'failed' | 'released';
  createdAt: number;
  updatedAt: number;
}

export interface ExecutionFeedbackRecord {
  id: string;
  kind: ExecutionFeedbackKind;
  correlationId: string;
  stepId: string;
  pathId: string;
  actorId?: string;
  toolName?: string;
  path?: string;
  summary: string;
  actionable: boolean;
  createdAt: number;
}

export interface ExecutionCachedToolCall {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  success: boolean;
  error?: string;
  reason?: string;
  startedAt?: number;
  completedAt?: number;
}

export interface ExecutionCachedPath {
  id: string;
  correlationId: string;
  stepId: string;
  parentPathIds: string[];
  taskFingerprint: string;
  status: 'completed' | 'failed' | 'partial';
  actorIds: string[];
  teamIds: string[];
  observedPaths: string[];
  invalidPaths: string[];
  successfulTools: string[];
  failedTools: string[];
  mutationObserved: boolean;
  verificationObserved: boolean;
  /**
   * Bounded causal tool evidence used to resume a later runtime process
   * without replaying unchanged reads, listings, or verifier calls.
   */
  toolFrontier?: ExecutionCachedToolCall[];
  feedbackIds: string[];
  summary?: string;
  createdAt: number;
  updatedAt: number;
}

export interface ExecutionCachedStep {
  id: string;
  correlationId: string;
  stepId: string;
  index: number;
  task: string;
  taskFingerprint: string;
  pathId: string;
  dependsOn: string[];
  action: string;
  status: 'completed' | 'failed';
  actorIds: string[];
  teamIds: string[];
  feedbackIds: string[];
  resultSummary?: string;
  stateFingerprint?: string;
  createdAt: number;
  updatedAt: number;
}

export interface ExecutionCacheSnapshot {
  step: ExecutionCachedStep;
  path: ExecutionCachedPath;
  actors: ExecutionCachedActor[];
  feedback: ExecutionFeedbackRecord[];
}

export interface ExecutionKnowledgeCacheState {
  version: 1;
  updatedAt: number;
  steps: ExecutionCachedStep[];
  paths: ExecutionCachedPath[];
  actors: ExecutionCachedActor[];
  feedback: ExecutionFeedbackRecord[];
}

export const EMPTY_EXECUTION_KNOWLEDGE_CACHE: ExecutionKnowledgeCacheState = {
  version: 1,
  updatedAt: 0,
  steps: [],
  paths: [],
  actors: [],
  feedback: [],
};

export function compactExecutionKnowledgeForPrompt(
  state: ExecutionKnowledgeCacheState,
  maxFeedbackItems = 24
): Record<string, unknown> {
  const selectedFeedback = [...state.feedback]
    .sort((left, right) =>
      Number(right.actionable) - Number(left.actionable) || right.createdAt - left.createdAt
    )
    .slice(0, Math.max(1, maxFeedbackItems));
  const feedbackIds = new Set(selectedFeedback.map(item => item.id));
  const selectedPaths = state.paths.slice(-12);
  const causalToolFrontier = compactSharedToolFrontier(selectedPaths);
  return {
    updatedAt: state.updatedAt,
    steps: state.steps.slice(-12).map(step => ({
      id: step.stepId,
      index: step.index,
      action: step.action,
      status: step.status,
      dependsOn: step.dependsOn,
      pathId: step.pathId,
      resultSummary: step.resultSummary?.slice(0, 1200),
      stateFingerprint: step.stateFingerprint,
    })),
    paths: selectedPaths.map(item => ({
      id: item.id,
      parentPathIds: item.parentPathIds,
      status: item.status,
      observedPaths: item.observedPaths.slice(0, 50),
      invalidPaths: item.invalidPaths.slice(0, 30),
      successfulTools: item.successfulTools,
      failedTools: item.failedTools,
      mutationObserved: item.mutationObserved,
      verificationObserved: item.verificationObserved,
      feedbackIds: item.feedbackIds.filter(id => feedbackIds.has(id)),
    })),
    causalToolFrontier,
    actors: state.actors.slice(-30).map(item => ({
      id: item.runtimeActorId,
      kind: item.kind,
      name: item.name,
      role: item.role,
      parentId: item.parentId,
      teamId: item.teamId,
      generation: item.generation,
      task: item.task?.slice(0, 800),
      definitionFingerprint: item.definitionFingerprint,
      status: item.status,
      pathId: item.pathId,
    })),
    feedback: selectedFeedback.map(item => ({
      kind: item.kind,
      actorId: item.actorId,
      toolName: item.toolName,
      path: item.path,
      actionable: item.actionable,
      summary: item.summary,
      pathId: item.pathId,
    })),
  };
}

/**
 * Paths inherit the causal frontier from their parents, so rendering a frontier
 * inside every path multiplies identical context on long runs. Keep one
 * chronological, deduplicated frontier for the whole prompt instead. Results
 * remain in the persisted cache; the prompt only needs intent and outcome
 * references to direct attention.
 */
function compactSharedToolFrontier(
  paths: ExecutionCachedPath[],
  maxCalls = 24
): Array<Record<string, unknown>> {
  const calls = paths
    .flatMap(path => (path.toolFrontier ?? []).map(call => ({
      pathId: path.id,
      call,
      timestamp: call.completedAt ?? call.startedAt ?? 0,
    })))
    .sort((left, right) => left.timestamp - right.timestamp);
  const selected: typeof calls = [];
  const seen = new Set<string>();
  for (const item of [...calls].reverse()) {
    const fingerprint = JSON.stringify({
      toolName: item.call.toolName,
      params: item.call.params,
      success: item.call.success,
      error: item.call.error,
    });
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);
    selected.push(item);
    if (selected.length >= Math.max(1, maxCalls)) break;
  }
  return selected.reverse().map(item => ({
    pathId: item.pathId,
    toolName: item.call.toolName,
    params: item.call.params,
    success: item.call.success,
    error: item.call.error?.slice(0, 500),
  }));
}
