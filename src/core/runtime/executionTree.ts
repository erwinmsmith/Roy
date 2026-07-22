export type RootExecutionTreeStatus = 'running' | 'completed' | 'failed';
export type RootExecutionStepStatus = 'running' | 'completed' | 'failed';
export type RootExecutionNodeStatus = 'active' | 'waiting' | 'done' | 'failed' | 'released';

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
      startedAt: Date.now(),
    };
    tree.currentStep = index;
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
    step.treeSnapshot = structuredClone(tree.nodes);
    step.completedAt = Date.now();
    tree.updatedAt = step.completedAt;
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
    return structuredClone(step);
  }

  finish(correlationId: string): RootExecutionTreeState {
    const tree = this.requireTree(correlationId);
    tree.status = tree.status === 'failed' ? 'failed' : 'completed';
    tree.completedAt = Date.now();
    tree.updatedAt = tree.completedAt;
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
