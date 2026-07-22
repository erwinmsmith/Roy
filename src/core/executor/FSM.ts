// FSM - Finite State Machine for agent control

import type { SignalBus } from './SignalBus.js';
import { signalBus } from './SignalBus.js';

export type FSMState =
  | 'S_created'
  | 'S_context_loading'
  | 'S_ready'
  | 'S_task_received'
  | 'S_planning'
  | 'S_tool_calling'
  | 'S_reasoning'
  | 'S_delegating'
  | 'S_waiting_children'
  | 'S_synthesizing'
  | 'S_responding'
  | 'S_done'
  | 'S_failed'
  | 'S_cancelled'
  | 'S_input_received'
  | 'S_assess_task'
  | 'S_solo_reasoning'
  | 'S_delegate_planning'
  | 'S_spawn_subagents'
  | 'S_wait_subagents'
  | 'S_synthesize'
  | 'S_respond'
  | 'S_turn_done'
  | 'S_solo'
  | 'S_diagnose'
  | 'S_decide'
  | 'S_derive'
  | 'S_reuse'
  | 'S_execute'
  | 'S_merge'
  | 'S_verify'
  | 'S_backtrack'
  | 'S_final';

export interface FSMTransition {
  from: FSMState;
  to: FSMState;
  condition?: (context: FSMContext) => boolean | Promise<boolean>;
  action?: (context: FSMContext) => void | Promise<void>;
}

export interface FSMContext {
  state: FSMState;
  trace: string[];
  budget: number | null;
  cost: number;
  uncertainty: number;
  conflict: number;
  evidence: number;
  metadata: Record<string, unknown>;
}

export interface FSMConfig {
  initialState?: FSMState;
  transitions?: FSMTransition[];
  signalBus?: SignalBus;
  onTransition?: (from: FSMState, to: FSMState, context: FSMContext) => void;
  onStateChange?: (state: FSMState, context: FSMContext) => void;
  onInvalidTransition?: (from: FSMState, to: FSMState, context: FSMContext) => void;
  strict?: boolean;
}

export class InvalidFSMTransitionError extends Error {
  constructor(readonly from: FSMState, readonly to: FSMState) {
    super(`Invalid FSM transition: ${from} -> ${to}`);
    this.name = 'InvalidFSMTransitionError';
  }
}

export class FSM {
  private state: FSMState;
  private transitions: Map<string, FSMTransition[]> = new Map();
  private signalBus: SignalBus;
  private config: FSMConfig;
  private context: FSMContext;

  private onTransition?: (from: FSMState, to: FSMState, context: FSMContext) => void;
  private onStateChange?: (state: FSMState, context: FSMContext) => void;
  private onInvalidTransition?: (from: FSMState, to: FSMState, context: FSMContext) => void;

  constructor(config: FSMConfig = {}) {
    this.config = config;
    this.state = config.initialState || 'S_solo';
    this.signalBus = config.signalBus || signalBus;
    this.onTransition = config.onTransition;
    this.onStateChange = config.onStateChange;
    this.onInvalidTransition = config.onInvalidTransition;

    this.context = {
      state: this.state,
      trace: [],
      budget: null,
      cost: 0,
      uncertainty: 0,
      conflict: 0,
      evidence: 0,
      metadata: {},
    };

    // Register default transitions
    this.registerDefaultTransitions();
  }

  /**
   * Get current state
   */
  getState(): FSMState {
    return this.state;
  }

  /**
   * Get current context
   */
  getContext(): FSMContext {
    return { ...this.context };
  }

  /**
   * Update context
   */
  updateContext(updates: Partial<FSMContext>): void {
    this.context = { ...this.context, ...updates };
  }

  /**
   * Add to trace
   */
  addToTrace(entry: string): void {
    this.context.trace.push(entry);
  }

  /**
   * Transition to a new state
   */
  async transition(targetState?: FSMState): Promise<boolean> {
    const from = this.state;

    if (targetState) {
      // Direct transition to target state
      const transition = this.findTransition(from, targetState);
      if (!transition && this.config.strict) {
        this.onInvalidTransition?.(from, targetState, this.context);
        throw new InvalidFSMTransitionError(from, targetState);
      }
      if (transition) {
        if (transition.condition) {
          const result = await transition.condition(this.context);
          if (!result) return false;
        }
        if (transition.action) {
          await transition.action(this.context);
        }
      }

      this.state = targetState;
      this.context.state = targetState;
      this.onTransition?.(from, targetState, this.context);
      this.onStateChange?.(targetState, this.context);

      // Emit signal
      await this.signalBus.signal({
        name: `fsm:${targetState}`,
        payload: { from, to: targetState, context: this.context },
        timestamp: Date.now(),
      });

      return true;
    }

    // Auto-determine next state based on context
    const nextState = this.determineNextState();
    if (nextState && nextState !== this.state) {
      return this.transition(nextState);
    }

    return false;
  }

  /**
   * Determine next state based on context
   */
  private determineNextState(): FSMState | null {
    const ctx = this.context;

    // FSM Decision logic based on design document
    switch (this.state) {
      case 'S_solo':
        // If uncertainty or conflict detected, go to diagnose
        if (ctx.uncertainty > 0.5 || ctx.conflict > 0.3) {
          return 'S_diagnose';
        }
        // Otherwise continue solo
        return null;

      case 'S_diagnose':
        // After diagnosis, decide whether to expand
        return 'S_decide';

      case 'S_decide':
        // Check if expansion is worth the cost
        if (ctx.budget !== null && ctx.budget - ctx.cost < 0) {
          return 'S_final'; // Not enough budget
        }
        if (ctx.uncertainty > 0.7 || ctx.conflict > 0.5) {
          // High uncertainty/conflict - derive or reuse
          if (ctx.metadata?.useCache) {
            return 'S_reuse';
          }
          return 'S_derive';
        }
        // Low uncertainty - continue solo
        return 'S_solo';

      case 'S_derive':
      case 'S_reuse':
        // After derivation/reuse, execute
        return 'S_execute';

      case 'S_execute':
        // After execution, merge results
        return 'S_merge';

      case 'S_merge':
        // After merge, verify
        return 'S_verify';

      case 'S_verify':
        // If verification passes, finalize
        if (ctx.uncertainty < 0.2 && ctx.conflict < 0.1 && ctx.evidence > 0.7) {
          return 'S_final';
        }
        // Otherwise backtrack
        return 'S_backtrack';

      case 'S_backtrack':
        // Backtrack to derive or reuse different approach
        return 'S_derive';

      case 'S_final':
        // Terminal state
        return null;

      default:
        return null;
    }
  }

  /**
   * Register a transition
   */
  registerTransition(transition: FSMTransition): void {
    const key = `${transition.from}:${transition.to}`;
    if (!this.transitions.has(key)) {
      this.transitions.set(key, []);
    }
    this.transitions.get(key)!.push(transition);
  }

  /**
   * Register default transitions
   */
  private registerDefaultTransitions(): void {
    const paths: Array<[FSMState, FSMState]> = [
      ['S_solo', 'S_input_received'],
      ['S_input_received', 'S_assess_task'],
      ['S_assess_task', 'S_solo_reasoning'],
      ['S_assess_task', 'S_delegate_planning'],
      ['S_solo_reasoning', 'S_respond'],
      ['S_delegate_planning', 'S_spawn_subagents'],
      ['S_spawn_subagents', 'S_wait_subagents'],
      ['S_wait_subagents', 'S_synthesize'],
      ['S_wait_subagents', 'S_assess_task'],
      ['S_assess_task', 'S_synthesize'],
      ['S_synthesize', 'S_respond'],
      ['S_respond', 'S_turn_done'],
      ['S_turn_done', 'S_solo'],
      ['S_created', 'S_context_loading'],
      ['S_created', 'S_ready'],
      ['S_context_loading', 'S_ready'],
      ['S_ready', 'S_task_received'],
      ['S_task_received', 'S_context_loading'],
      ['S_context_loading', 'S_planning'],
      ['S_context_loading', 'S_tool_calling'],
      ['S_context_loading', 'S_reasoning'],
      ['S_planning', 'S_delegating'],
      ['S_planning', 'S_reasoning'],
      ['S_delegating', 'S_waiting_children'],
      ['S_waiting_children', 'S_synthesizing'],
      ['S_tool_calling', 'S_reasoning'],
      ['S_reasoning', 'S_delegating'],
      ['S_reasoning', 'S_responding'],
      ['S_synthesizing', 'S_responding'],
      ['S_responding', 'S_done'],
      ['S_done', 'S_ready'],
      ['S_failed', 'S_ready'],
      ['S_solo', 'S_diagnose'],
      ['S_diagnose', 'S_decide'],
      ['S_decide', 'S_solo'],
      ['S_decide', 'S_derive'],
      ['S_decide', 'S_reuse'],
      ['S_decide', 'S_final'],
      ['S_derive', 'S_execute'],
      ['S_reuse', 'S_execute'],
      ['S_execute', 'S_merge'],
      ['S_merge', 'S_verify'],
      ['S_verify', 'S_final'],
      ['S_verify', 'S_backtrack'],
      ['S_backtrack', 'S_derive'],
    ];
    for (const [from, to] of paths) this.registerTransition({ from, to });

    const actorStates: FSMState[] = [
      'S_created', 'S_context_loading', 'S_ready', 'S_task_received', 'S_planning',
      'S_tool_calling', 'S_reasoning', 'S_delegating', 'S_waiting_children',
      'S_synthesizing', 'S_responding', 'S_done',
    ];
    for (const state of actorStates) {
      this.registerTransition({ from: state, to: 'S_failed' });
      this.registerTransition({ from: state, to: 'S_cancelled' });
    }
  }

  /**
   * Find transition between states
   */
  private findTransition(from: FSMState, to: FSMState): FSMTransition | undefined {
    const key = `${from}:${to}`;
    return this.transitions.get(key)?.[0];
  }

  canTransition(to: FSMState): boolean {
    return this.findTransition(this.state, to) !== undefined;
  }

  /**
   * Check if FSM should transition (for external use)
   */
  shouldTransition(): boolean {
    return this.determineNextState() !== null;
  }

  /**
   * Trigger FSM processing - call this after agent steps
   */
  async trigger(): Promise<boolean> {
    const nextState = this.determineNextState();
    if (nextState) {
      return await this.transition(nextState);
    }
    return false;
  }

  /**
   * Update uncertainty metric (0-1)
   */
  setUncertainty(value: number): void {
    this.context.uncertainty = Math.max(0, Math.min(1, value));
  }

  /**
   * Update conflict metric (0-1)
   */
  setConflict(value: number): void {
    this.context.conflict = Math.max(0, Math.min(1, value));
  }

  /**
   * Update evidence metric (0-1)
   */
  setEvidence(value: number): void {
    this.context.evidence = Math.max(0, Math.min(1, value));
  }

  /**
   * Update cost (accumulates over time)
   */
  addCost(amount: number): void {
    this.context.cost += amount;
  }

  /**
   * Set budget
   */
  setBudget(budget: number): void {
    this.context.budget = budget;
  }

  /**
   * Remove budget limit.
   */
  clearBudget(): void {
    this.context.budget = null;
  }

  /**
   * Check if FSM is in terminal state
   */
  isTerminal(): boolean {
    return this.state === 'S_final';
  }

  /**
   * Get current state name
   */
  getStateName(): string {
    return this.state;
  }

  /**
   * Reset FSM to initial state
   */
  reset(): void {
    this.state = this.config.initialState || 'S_solo';
    this.context = {
      state: this.state,
      trace: [],
      budget: this.context.budget,
      cost: 0,
      uncertainty: 0,
      conflict: 0,
      evidence: 0,
      metadata: {},
    };
    this.onStateChange?.(this.state, this.context);
  }
}

export default FSM;
