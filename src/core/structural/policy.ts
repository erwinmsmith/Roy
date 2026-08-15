import type {
  ChildSpecification,
  StructuralCheckpoint,
  StructuralDecision,
  StructuralPolicy,
} from './types.js';

export class LegacyStructuralPolicyAdapter implements StructuralPolicy {
  readonly name = 'legacy-delegation-adapter';
  readonly version = '1.0.0';

  decide(checkpoint: StructuralCheckpoint): StructuralDecision {
    const branch = checkpoint.legalActions.includes('BRANCH')
      && checkpoint.eventGraph.nodes.some(node =>
        node.status === 'failed'
        || node.kind === 'dependency'
        || node.attributes?.requiresDelegation === true);
    if (branch) return { action: 'BRANCH', rationale: 'Legacy observable-gap compatibility rule.' };
    if (checkpoint.legalActions.includes('RETURN')
      && checkpoint.eventGraph.nodes.some(node => node.attributes?.outputContractSatisfied === true)) {
      return { action: 'RETURN', rationale: 'Legacy output contract is satisfied.' };
    }
    return { action: 'CONTINUE', rationale: 'Preserve the existing direct-computation path.' };
  }
}

export function validateStructuralDecision(
  checkpoint: StructuralCheckpoint,
  decision: StructuralDecision
): StructuralDecision {
  if (!checkpoint.legalActions.includes(decision.action)) {
    throw new Error(`Structural policy selected illegal action ${decision.action}`);
  }
  if (decision.action !== 'BRANCH' && decision.childSpecification) {
    throw new Error('Only BRANCH may carry a child specification');
  }
  if (decision.action === 'BRANCH' && decision.childSpecification) {
    validateChildSpecification(decision.childSpecification);
  }
  if (decision.confidence !== undefined
    && (!Number.isFinite(decision.confidence) || decision.confidence < 0 || decision.confidence > 1)) {
    throw new Error('Structural decision confidence must be in [0, 1]');
  }
  return structuredClone(decision);
}

export function validateChildSpecification(specification: ChildSpecification): void {
  if (!specification.id.trim()) throw new Error('Child specification id is required');
  if (!specification.task.trim()) throw new Error('Child specification task is required');
  if (specification.outputContract.format === 'json'
    && specification.outputContract.requiredFields.length === 0) {
    throw new Error('JSON child output contracts require at least one field');
  }
}
