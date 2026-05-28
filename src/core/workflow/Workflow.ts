// Base Workflow abstract class

import type { Executor } from '../executor/Executor.js';
import {
  WorkflowState,
  WorkflowResult,
  createWorkflowState,
  updateWorkflowState,
  recordWorkflowError,
} from './WorkflowState.js';
import { signalBus } from '../executor/SignalBus.js';

export interface WorkflowConfig {
  name?: string;
  metadata?: Record<string, unknown>;
  signalBus?: typeof signalBus;
}

export abstract class Workflow<T = unknown> {
  readonly name: string;
  protected executor: Executor;
  protected state: WorkflowState;
  protected config: WorkflowConfig;

  constructor(executor: Executor, config: WorkflowConfig = {}) {
    this.executor = executor;
    this.name = config.name || this.constructor.name;
    this.config = config;
    this.state = createWorkflowState(this.name, config.metadata);
  }

  /**
   * Main workflow implementation - must be overridden
   */
  abstract run(...args: unknown[]): Promise<WorkflowResult<T>>;

  /**
   * Get current state
   */
  getState(): WorkflowState {
    return { ...this.state };
  }

  /**
   * Update state
   */
  protected updateState(updates: Partial<WorkflowState>): void {
    this.state = updateWorkflowState(this.state, updates);
  }

  /**
   * Record error
   */
  protected recordError(error: Error): void {
    this.state = recordWorkflowError(this.state, error);
  }

  /**
   * Wait for input/signal
   */
  async waitForInput(description: string = 'Provide input'): Promise<string> {
    this.updateState({ status: 'waiting' });

    const signalName = `human_input:${this.name}`;
    const input = await this.executor.waitForSignal<string>(signalName, 60000);

    this.updateState({ status: 'running' });
    return input;
  }

  /**
   * Emit a signal
   */
  async emitSignal(signalName: string, payload?: unknown): Promise<void> {
    await this.executor.signal(signalName, payload);
  }

  /**
   * Check if workflow is in a terminal state
   */
  isTerminal(): boolean {
    return ['completed', 'failed', 'cancelled'].includes(this.state.status);
  }

  /**
   * Cancel the workflow
   */
  cancel(): void {
    this.updateState({ status: 'cancelled' });
  }

  /**
   * Get workflow metadata
   */
  getMetadata(): Record<string, unknown> {
    return { ...this.state.metadata };
  }

  /**
   * Set workflow metadata
   */
  setMetadata(key: string, value: unknown): void {
    this.state.metadata[key] = value;
    this.updateState({ metadata: this.state.metadata });
  }
}

export default Workflow;