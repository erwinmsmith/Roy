// Workflow state management

export interface WorkflowState {
  status: 'initialized' | 'running' | 'waiting' | 'completed' | 'failed' | 'cancelled';
  metadata: Record<string, unknown>;
  updatedAt: number;
  error?: {
    type: string;
    message: string;
    timestamp: number;
  };
}

export interface WorkflowResult<T = unknown> {
  value?: T;
  metadata: Record<string, unknown>;
  startTime?: number;
  endTime?: number;
  success: boolean;
  error?: string;
}

export function createWorkflowState(
  name?: string,
  metadata?: Record<string, unknown>
): WorkflowState {
  return {
    status: 'initialized',
    metadata: metadata || { name },
    updatedAt: Date.now(),
  };
}

export function updateWorkflowState(
  state: WorkflowState,
  updates: Partial<WorkflowState>
): WorkflowState {
  return {
    ...state,
    ...updates,
    updatedAt: Date.now(),
  };
}

export function recordWorkflowError(
  state: WorkflowState,
  error: Error
): WorkflowState {
  return updateWorkflowState(state, {
    status: 'failed',
    error: {
      type: error.constructor.name,
      message: error.message,
      timestamp: Date.now(),
    },
  });
}