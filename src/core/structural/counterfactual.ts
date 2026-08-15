import type {
  ChildSpecification,
  CloneableStructuralEnvironment,
  CounterfactualGroupResult,
  StructuralActionKind,
  StructuralCheckpoint,
  StructuralRolloutResult,
} from './types.js';
import { STRUCTURAL_SCHEMA_VERSION } from './types.js';

export interface CounterfactualGroupInput<TSnapshot> {
  checkpoint: StructuralCheckpoint;
  environment: CloneableStructuralEnvironment<TSnapshot>;
  branchSpecifications: ChildSpecification[];
  repeats: number;
  rolloutPolicy: string;
}

export async function runCounterfactualGroup<TSnapshot>(
  input: CounterfactualGroupInput<TSnapshot>
): Promise<CounterfactualGroupResult> {
  if (input.repeats < 1 || !Number.isSafeInteger(input.repeats)) {
    throw new Error('Counterfactual repeats must be a positive integer');
  }
  if (input.environment.revision !== input.checkpoint.environmentRevision) {
    throw new Error('Counterfactual environment revision does not match checkpoint');
  }
  const baseSnapshot = await input.environment.snapshot();
  const variants: Array<{ action: StructuralActionKind; child?: ChildSpecification }> = [];
  for (const action of input.checkpoint.legalActions) {
    if (action === 'BRANCH') {
      for (const child of input.branchSpecifications) variants.push({ action, child });
    } else {
      variants.push({ action });
    }
  }
  const results: StructuralRolloutResult[] = [];
  for (const variant of variants) {
    for (let repeat = 0; repeat < input.repeats; repeat += 1) {
      await input.environment.restore(structuredClone(baseSnapshot));
      const observed = await input.environment.execute(variant.action, variant.child, repeat);
      results.push({ ...observed, action: variant.action, childSpecification: variant.child, repeat });
    }
  }
  await input.environment.restore(structuredClone(baseSnapshot));

  const nonBranchValues = new Map<StructuralActionKind, number>();
  for (const action of ['CONTINUE', 'RETURN'] as const) {
    const utilities = results.filter(result => result.action === action).map(result => result.utility);
    if (utilities.length > 0) nonBranchValues.set(action, mean(utilities));
  }
  const branchBySpecification = new Map<string, number>();
  for (const child of input.branchSpecifications) {
    const utilities = results
      .filter(result => result.action === 'BRANCH' && result.childSpecification?.id === child.id)
      .map(result => result.utility);
    if (utilities.length > 0) branchBySpecification.set(child.id, mean(utilities));
  }
  const branchValue = branchBySpecification.size > 0
    ? mean([...branchBySpecification.values()])
    : undefined;
  const actionValues: CounterfactualGroupResult['actionValues'] = Object.fromEntries(nonBranchValues);
  if (branchValue !== undefined) actionValues.BRANCH = branchValue;
  const outerAdvantages = standardizedRecord(actionValues);
  const branchAdvantages = standardizedRecord(Object.fromEntries(branchBySpecification)) as Record<string, number>;
  return {
    schemaVersion: STRUCTURAL_SCHEMA_VERSION,
    checkpointId: input.checkpoint.id,
    checkpointFingerprint: input.checkpoint.fingerprint,
    rolloutPolicy: input.rolloutPolicy,
    branchAggregate: 'mean',
    results,
    actionValues,
    outerAdvantages,
    branchAdvantages,
    createdAt: Date.now(),
  };
}

export function standardizedRecord<T extends string>(
  values: Partial<Record<T, number>>,
  epsilon = 1e-8
): Partial<Record<T, number>> {
  const entries = Object.entries(values) as Array<[T, number]>;
  if (entries.length === 0) return {};
  const average = mean(entries.map(([, value]) => value));
  const variance = mean(entries.map(([, value]) => (value - average) ** 2));
  const deviation = Math.sqrt(variance);
  if (deviation <= epsilon) return Object.fromEntries(entries.map(([key]) => [key, 0])) as Partial<Record<T, number>>;
  return Object.fromEntries(entries.map(([key, value]) => [key, (value - average) / (deviation + epsilon)])) as Partial<Record<T, number>>;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}
