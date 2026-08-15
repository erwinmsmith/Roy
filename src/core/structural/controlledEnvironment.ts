import { consumeResources } from './checkpoint.js';
import type {
  ChildSpecification,
  CloneableStructuralEnvironment,
  ResourceEnvelope,
  StructuralActionKind,
  StructuralRolloutResult,
} from './types.js';

export interface ControlledDerivationTask {
  id: string;
  family: 'activation' | 'acquisition' | 'mixed';
  evidenceAvailable: boolean;
  requiredComputation: number;
  hiddenEvidenceValue: number;
  childActivationValue: number;
  directComputationValue: number;
  resources: ResourceEnvelope;
}

export interface ControlledEnvironmentSnapshot {
  task: ControlledDerivationTask;
  resources: ResourceEnvelope;
  evidenceAcquired: boolean;
  computation: number;
  terminal: boolean;
}

export class ControlledDerivationEnvironment
implements CloneableStructuralEnvironment<ControlledEnvironmentSnapshot> {
  readonly revision = 'controlled-derivation-v1';
  private state: ControlledEnvironmentSnapshot;

  constructor(task: ControlledDerivationTask) {
    this.state = {
      task: structuredClone(task),
      resources: structuredClone(task.resources),
      evidenceAcquired: task.evidenceAvailable,
      computation: 0,
      terminal: false,
    };
  }

  snapshot(): ControlledEnvironmentSnapshot {
    return structuredClone(this.state);
  }

  restore(snapshot: ControlledEnvironmentSnapshot): void {
    this.state = structuredClone(snapshot);
  }

  execute(
    action: StructuralActionKind,
    childSpecification: ChildSpecification | undefined,
    _repeat: number
  ): Omit<StructuralRolloutResult, 'action' | 'childSpecification' | 'repeat'> {
    const startedAt = Date.now();
    const resourcesBefore = structuredClone(this.state.resources);
    if (action === 'CONTINUE') {
      this.state.resources = consumeResources(this.state.resources, { computeTokens: 1 });
      this.state.computation += this.state.task.directComputationValue;
    } else if (action === 'BRANCH') {
      if (!childSpecification) throw new Error('Controlled BRANCH requires a child specification');
      this.state.resources = consumeResources(this.state.resources, {
        computeTokens: 1,
        parallelSlots: 1,
        toolCalls: this.state.task.evidenceAvailable ? 0 : 1,
      });
      if (!this.state.evidenceAcquired) this.state.evidenceAcquired = true;
      this.state.computation += this.state.task.childActivationValue;
    } else {
      this.state.terminal = true;
    }
    const utility = this.utility();
    return {
      utility,
      durationMs: Date.now() - startedAt,
      resourcesBefore,
      resourcesAfter: structuredClone(this.state.resources),
      terminal: this.state.terminal,
      output: { evidenceAcquired: this.state.evidenceAcquired, computation: this.state.computation },
    };
  }

  mechanismValues(): { noChild: number; evidenceOnly: number; fullChild: number; acquisition: number; activation: number } {
    const base = this.baseUtility(false, 0);
    const evidenceOnly = this.baseUtility(true, 0);
    const fullChild = this.baseUtility(true, this.state.task.childActivationValue);
    return {
      noChild: base,
      evidenceOnly,
      fullChild,
      acquisition: evidenceOnly - base,
      activation: fullChild - evidenceOnly,
    };
  }

  private utility(): number {
    return this.baseUtility(this.state.evidenceAcquired, this.state.computation);
  }

  private baseUtility(evidence: boolean, computation: number): number {
    const evidenceValue = evidence ? this.state.task.hiddenEvidenceValue : 0;
    const computationValue = Math.min(1, computation / Math.max(1, this.state.task.requiredComputation));
    return Math.max(0, Math.min(1, evidenceValue + computationValue * (1 - evidenceValue)));
  }
}
