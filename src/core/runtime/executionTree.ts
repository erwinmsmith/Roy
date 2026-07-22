export type RootExecutionTreeStatus = 'running' | 'completed' | 'failed';
export type RootExecutionStepStatus = 'running' | 'completed' | 'failed';
export type RootExecutionNodeStatus = 'active' | 'waiting' | 'done' | 'failed' | 'released';
export type RootExecutionActivityKind =
  | 'conversation'
  | 'context'
  | 'thinking'
  | 'tool'
  | 'delegation'
  | 'agent'
  | 'team'
  | 'synthesis'
  | 'checkpoint'
  | 'control';
export type RootExecutionActivityStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface RootExecutionActivity {
  id: string;
  kind: RootExecutionActivityKind;
  status: RootExecutionActivityStatus;
  label: string;
  actorId?: string;
  parentActivityId?: string;
  messageId?: string;
  eventType?: string;
  summary?: string;
  tokenUsage?: number;
  startedAt: number;
  completedAt?: number;
  data?: Record<string, unknown>;
}

export interface RootExecutionCheckpoint {
  objective: string;
  completed: string[];
  pending: string[];
  evidence: string[];
  decisionBasis: string;
  stateFingerprint: string;
  createdAt: number;
}

export interface RootExecutionLoopState {
  iteration: number;
  maxIterations: number;
  maxWallClockMs: number;
  elapsedMs: number;
  stalledIterations: number;
  maxStalledIterations: number;
  stopReason?: 'completed' | 'clarification' | 'max_iterations' | 'timeout' | 'stalled' | 'failed';
}

export interface RootExecutionNodeSnapshot {
  id: string;
  kind: 'agent' | 'team';
  name: string;
  role: string;
  parentId?: string;
  teamId?: string;
  status: RootExecutionNodeStatus;
  createdAtStep: number;
  updatedAtStep: number;
  tokenUsage?: number;
}

export interface RootExecutionStepDecision {
  action: 'solve_directly' | 'ask_clarification' | 'delegate' | 'finalize';
  reason: string;
  agentCount: number;
}

export interface RootExecutionStep {
  id: string;
  index: number;
  dependsOn: string[];
  status: RootExecutionStepStatus;
  decision: RootExecutionStepDecision;
  actorIds: string[];
  teamIds: string[];
  resultSummary?: string;
  treeSnapshot: RootExecutionNodeSnapshot[];
  activities: RootExecutionActivity[];
  checkpoint?: RootExecutionCheckpoint;
  startedAt: number;
  completedAt?: number;
  error?: string;
}

export interface RootExecutionTreeState {
  correlationId: string;
  sessionId: string;
  task: string;
  rootAgentId: string;
  status: RootExecutionTreeStatus;
  currentStep: number;
  maxSteps: number;
  nodes: RootExecutionNodeSnapshot[];
  steps: RootExecutionStep[];
  loop: RootExecutionLoopState;
  createdAt: number;
  updatedAt: number;
  completedAt?: number;
  error?: string;
}

export interface StartRootExecutionStepInput {
  decision: RootExecutionStepDecision;
  dependsOn?: string[];
}

export interface CompleteRootExecutionStepInput {
  actorIds?: string[];
  teamIds?: string[];
  nodes?: RootExecutionNodeSnapshot[];
  resultSummary?: string;
  activities?: RootExecutionActivity[];
  checkpoint?: RootExecutionCheckpoint;
}

function cloneTree(tree: RootExecutionTreeState): RootExecutionTreeState {
  return structuredClone(tree);
}

export class RootExecutionTreeRegistry {
  private readonly trees = new Map<string, RootExecutionTreeState>();

  begin(input: {
    correlationId: string;
    sessionId: string;
    task: string;
    rootAgentId?: string;
    rootAgentName?: string;
    maxSteps: number;
    maxWallClockMs?: number;
    maxStalledIterations?: number;
  }): RootExecutionTreeState {
    const now = Date.now();
    const rootAgentId = input.rootAgentId ?? 'root';
    const tree: RootExecutionTreeState = {
      correlationId: input.correlationId,
      sessionId: input.sessionId,
      task: input.task,
      rootAgentId,
      status: 'running',
      currentStep: 0,
      maxSteps: Math.max(1, input.maxSteps),
      nodes: [{
        id: rootAgentId,
        kind: 'agent',
        name: input.rootAgentName ?? 'Roy',
        role: 'root',
        status: 'active',
        createdAtStep: 0,
        updatedAtStep: 0,
      }],
      steps: [],
      loop: {
        iteration: 0,
        maxIterations: Math.max(1, input.maxSteps),
        maxWallClockMs: Math.max(1, input.maxWallClockMs ?? 15 * 60_000),
        elapsedMs: 0,
        stalledIterations: 0,
        maxStalledIterations: Math.max(1, input.maxStalledIterations ?? 2),
      },
      createdAt: now,
      updatedAt: now,
    };
    this.trees.set(input.correlationId, tree);
    return cloneTree(tree);
  }

  startStep(correlationId: string, input: StartRootExecutionStepInput): RootExecutionStep {
    const tree = this.requireTree(correlationId);
    if (tree.status !== 'running') throw new Error(`Execution tree ${correlationId} is not running`);
    if (tree.steps.length >= tree.maxSteps) throw new Error(`Execution tree ${correlationId} reached max steps`);

    const index = tree.steps.length + 1;
    const step: RootExecutionStep = {
      id: `${correlationId}.step_${String(index).padStart(2, '0')}`,
      index,
      dependsOn: [...(input.dependsOn ?? [])],
      status: 'running',
      decision: structuredClone(input.decision),
      actorIds: [],
      teamIds: [],
      treeSnapshot: structuredClone(tree.nodes),
      activities: [],
      startedAt: Date.now(),
    };
    tree.currentStep = index;
    tree.loop.iteration = index;
    tree.steps.push(step);
    tree.updatedAt = step.startedAt;
    return structuredClone(step);
  }

  completeStep(correlationId: string, stepId: string, input: CompleteRootExecutionStepInput = {}): RootExecutionStep {
    const tree = this.requireTree(correlationId);
    const step = this.requireStep(tree, stepId);
    for (const node of input.nodes ?? []) this.upsertNode(tree, node);
    step.status = 'completed';
    step.actorIds = [...(input.actorIds ?? [])];
    step.teamIds = [...(input.teamIds ?? [])];
    step.resultSummary = input.resultSummary;
    step.activities = structuredClone(input.activities ?? step.activities);
    step.checkpoint = input.checkpoint ? structuredClone(input.checkpoint) : step.checkpoint;
    step.treeSnapshot = structuredClone(tree.nodes);
    step.completedAt = Date.now();
    tree.updatedAt = step.completedAt;
    tree.loop.elapsedMs = step.completedAt - tree.createdAt;
    if (step.checkpoint) {
      const previous = tree.steps.at(-2)?.checkpoint;
      tree.loop.stalledIterations = previous?.stateFingerprint === step.checkpoint.stateFingerprint
        ? tree.loop.stalledIterations + 1
        : 0;
    }
    return structuredClone(step);
  }

  setStepActivities(correlationId: string, stepId: string, activities: RootExecutionActivity[]): RootExecutionStep {
    const tree = this.requireTree(correlationId);
    const step = this.requireStep(tree, stepId);
    step.activities = structuredClone(activities);
    tree.updatedAt = Date.now();
    return structuredClone(step);
  }

  failStep(correlationId: string, stepId: string, error: string): RootExecutionStep {
    const tree = this.requireTree(correlationId);
    const step = this.requireStep(tree, stepId);
    step.status = 'failed';
    step.error = error;
    step.treeSnapshot = structuredClone(tree.nodes);
    step.completedAt = Date.now();
    tree.status = 'failed';
    tree.error = error;
    tree.updatedAt = step.completedAt;
    tree.loop.elapsedMs = step.completedAt - tree.createdAt;
    tree.loop.stopReason = 'failed';
    return structuredClone(step);
  }

  fail(correlationId: string, error: string): RootExecutionTreeState {
    const tree = this.requireTree(correlationId);
    const now = Date.now();
    let runningStep: RootExecutionStep | undefined;
    for (let index = tree.steps.length - 1; index >= 0; index -= 1) {
      if (tree.steps[index].status === 'running') {
        runningStep = tree.steps[index];
        break;
      }
    }
    if (runningStep) {
      runningStep.status = 'failed';
      runningStep.error = error;
      runningStep.treeSnapshot = structuredClone(tree.nodes);
      runningStep.completedAt = now;
    }
    tree.status = 'failed';
    tree.error = error;
    tree.completedAt = now;
    tree.updatedAt = now;
    tree.loop.elapsedMs = now - tree.createdAt;
    tree.loop.stopReason = 'failed';
    const root = tree.nodes.find(node => node.id === tree.rootAgentId);
    if (root) {
      root.status = 'failed';
      root.updatedAtStep = tree.currentStep;
    }
    return cloneTree(tree);
  }

  finish(
    correlationId: string,
    stopReason: RootExecutionLoopState['stopReason'] = 'completed'
  ): RootExecutionTreeState {
    const tree = this.requireTree(correlationId);
    tree.status = tree.status === 'failed' ? 'failed' : 'completed';
    tree.completedAt = Date.now();
    tree.updatedAt = tree.completedAt;
    tree.loop.elapsedMs = tree.completedAt - tree.createdAt;
    tree.loop.stopReason = tree.status === 'failed' ? 'failed' : stopReason;
    const root = tree.nodes.find(node => node.id === tree.rootAgentId);
    if (root) {
      root.status = tree.status === 'failed' ? 'failed' : 'done';
      root.updatedAtStep = tree.currentStep;
    }
    return cloneTree(tree);
  }

  get(correlationId: string): RootExecutionTreeState | undefined {
    const tree = this.trees.get(correlationId);
    return tree ? cloneTree(tree) : undefined;
  }

  latest(): RootExecutionTreeState | undefined {
    const tree = [...this.trees.values()].sort((left, right) => right.updatedAt - left.updatedAt)[0];
    return tree ? cloneTree(tree) : undefined;
  }

  list(): RootExecutionTreeState[] {
    return [...this.trees.values()]
      .sort((left, right) => right.updatedAt - left.updatedAt)
      .map(cloneTree);
  }

  restore(input: RootExecutionTreeState): RootExecutionTreeState {
    const tree = cloneTree(input);
    tree.steps = tree.steps.map(step => ({ ...step, activities: step.activities ?? [] }));
    tree.loop = tree.loop ?? {
      iteration: tree.currentStep,
      maxIterations: tree.maxSteps,
      maxWallClockMs: 15 * 60_000,
      elapsedMs: Math.max(0, tree.updatedAt - tree.createdAt),
      stalledIterations: 0,
      maxStalledIterations: 2,
      stopReason: tree.status === 'completed' ? 'completed' : tree.status === 'failed' ? 'failed' : undefined,
    };
    this.trees.set(tree.correlationId, tree);
    return cloneTree(tree);
  }

  clear(): void {
    this.trees.clear();
  }

  private upsertNode(tree: RootExecutionTreeState, input: RootExecutionNodeSnapshot): void {
    const existing = tree.nodes.find(node => node.id === input.id);
    if (existing) {
      const createdAtStep = existing.createdAtStep;
      Object.assign(existing, structuredClone(input), { createdAtStep });
    }
    else tree.nodes.push(structuredClone(input));
  }

  private requireTree(correlationId: string): RootExecutionTreeState {
    const tree = this.trees.get(correlationId);
    if (!tree) throw new Error(`Execution tree not found: ${correlationId}`);
    return tree;
  }

  private requireStep(tree: RootExecutionTreeState, stepId: string): RootExecutionStep {
    const step = tree.steps.find(item => item.id === stepId);
    if (!step) throw new Error(`Execution step not found: ${stepId}`);
    return step;
  }
}
