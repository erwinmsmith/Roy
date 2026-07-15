import type { EvolutionFSMState } from './types.js';

const TRANSITIONS: Record<EvolutionFSMState, EvolutionFSMState[]> = {
  S_evo_idle: ['S_evo_propose'],
  S_evo_propose: ['S_evo_instantiate', 'S_evo_failed'],
  S_evo_instantiate: ['S_evo_execute', 'S_evo_failed'],
  S_evo_execute: ['S_evo_evaluate', 'S_evo_failed'],
  S_evo_evaluate: ['S_evo_select', 'S_evo_failed'],
  S_evo_select: ['S_evo_mutate', 'S_evo_integrate', 'S_evo_failed'],
  S_evo_mutate: ['S_evo_instantiate', 'S_evo_integrate', 'S_evo_failed'],
  S_evo_integrate: ['S_evo_done', 'S_evo_failed'],
  S_evo_done: ['S_evo_propose'],
  S_evo_failed: ['S_evo_propose'],
};

export class InvalidEvolutionTransitionError extends Error {
  constructor(from: EvolutionFSMState, to: EvolutionFSMState) {
    super(`Invalid evolution FSM transition: ${from} -> ${to}`);
    this.name = 'InvalidEvolutionTransitionError';
  }
}

export class EvolutionStateMachine {
  private state: EvolutionFSMState = 'S_evo_idle';

  getState(): EvolutionFSMState {
    return this.state;
  }

  transition(next: EvolutionFSMState): { from: EvolutionFSMState; to: EvolutionFSMState } {
    const from = this.state;
    if (!TRANSITIONS[from].includes(next)) throw new InvalidEvolutionTransitionError(from, next);
    this.state = next;
    return { from, to: next };
  }
}
