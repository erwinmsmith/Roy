import type {
  ReasoningInvestmentEstimate,
  ReasoningInvestmentInput,
  ReasoningInvestmentModel,
  ReasoningUtilitySignals,
} from './types.js';

export interface WeightedReasoningInvestmentModelOptions {
  benefitWeights?: Partial<Record<BenefitSignal, number>>;
  riskWeights?: Partial<Record<RiskSignal, number>>;
  riskPenalty?: number;
  tokenScale?: number;
  contextTokenScale?: number;
  toolCallScale?: number;
  latencyScaleMs?: number;
}

type BenefitSignal =
  | 'rootUtility'
  | 'parentUtility'
  | 'historicalUtility'
  | 'evidenceGain'
  | 'uncertaintyReduction'
  | 'conflictResolution'
  | 'verificationGain'
  | 'cacheConfidence';

type RiskSignal = 'duplicationRisk' | 'executionRisk';

const DEFAULT_BENEFIT_WEIGHTS: Record<BenefitSignal, number> = {
  rootUtility: 0.12,
  parentUtility: 0.12,
  historicalUtility: 0.12,
  evidenceGain: 0.18,
  uncertaintyReduction: 0.16,
  conflictResolution: 0.1,
  verificationGain: 0.12,
  cacheConfidence: 0.08,
};

const DEFAULT_RISK_WEIGHTS: Record<RiskSignal, number> = {
  duplicationRisk: 0.45,
  executionRisk: 0.55,
};

export class WeightedReasoningInvestmentModel implements ReasoningInvestmentModel {
  readonly id = 'weighted_reasoning_investment_v1';
  private readonly benefits: Record<BenefitSignal, number>;
  private readonly risks: Record<RiskSignal, number>;
  private readonly riskPenalty: number;
  private readonly tokenScale: number;
  private readonly contextTokenScale: number;
  private readonly toolCallScale: number;
  private readonly latencyScaleMs: number;

  constructor(options: WeightedReasoningInvestmentModelOptions = {}) {
    this.benefits = normalizeWeights({ ...DEFAULT_BENEFIT_WEIGHTS, ...options.benefitWeights });
    this.risks = normalizeWeights({ ...DEFAULT_RISK_WEIGHTS, ...options.riskWeights });
    this.riskPenalty = clamp(options.riskPenalty ?? 0.55);
    this.tokenScale = positive(options.tokenScale, 10_000);
    this.contextTokenScale = positive(options.contextTokenScale, 8_000);
    this.toolCallScale = positive(options.toolCallScale, 10);
    this.latencyScaleMs = positive(options.latencyScaleMs, 60_000);
  }

  estimate(input: ReasoningInvestmentInput): ReasoningInvestmentEstimate {
    const signals = this.normalizeSignals(input.signals);
    const expectedUtility = sumWeighted(signals, this.benefits);
    const risk = sumWeighted(signals, this.risks);
    const confidence = clamp(signals.confidence ?? 0.65);
    const riskAdjustedUtility = clamp(expectedUtility * (1 - risk * this.riskPenalty) * (0.7 + confidence * 0.3));
    const tokenCost = clamp(input.resources.tokens / this.tokenScale);
    const contextCost = clamp((input.resources.contextTokens ?? input.resources.inputTokens ?? 0) / this.contextTokenScale);
    const toolCost = clamp((input.resources.toolCalls ?? 0) / this.toolCallScale);
    const latencyCost = clamp((input.resources.latencyMs ?? 0) / this.latencyScaleMs);
    const costScore = clamp(tokenCost * 0.65 + contextCost * 0.15 + toolCost * 0.1 + latencyCost * 0.1);
    const expectedReturn = Number((riskAdjustedUtility / Math.max(0.05, costScore)).toFixed(4));

    return {
      model: this.id,
      expectedUtility: round(expectedUtility),
      riskAdjustedUtility: round(riskAdjustedUtility),
      costScore: round(costScore),
      expectedReturn,
      confidence: round(confidence),
      components: {
        ...Object.fromEntries((Object.keys(this.benefits) as BenefitSignal[]).map(key => [key, round(signals[key] ?? 0)])),
        duplicationRisk: round(signals.duplicationRisk ?? 0),
        executionRisk: round(signals.executionRisk ?? 0),
        tokenCost: round(tokenCost),
        contextCost: round(contextCost),
        toolCost: round(toolCost),
        latencyCost: round(latencyCost),
      },
      rationale: this.rationale(input, signals, riskAdjustedUtility, costScore),
    };
  }

  private normalizeSignals(signals: ReasoningUtilitySignals): Required<ReasoningUtilitySignals> {
    return {
      rootUtility: clamp(signals.rootUtility ?? 0.5),
      parentUtility: clamp(signals.parentUtility ?? signals.rootUtility ?? 0.5),
      historicalUtility: clamp(signals.historicalUtility ?? 0.5),
      evidenceGain: clamp(signals.evidenceGain ?? 0),
      uncertaintyReduction: clamp(signals.uncertaintyReduction ?? 0),
      conflictResolution: clamp(signals.conflictResolution ?? 0),
      verificationGain: clamp(signals.verificationGain ?? 0),
      cacheConfidence: clamp(signals.cacheConfidence ?? 0),
      duplicationRisk: clamp(signals.duplicationRisk ?? 0),
      executionRisk: clamp(signals.executionRisk ?? 0),
      confidence: clamp(signals.confidence ?? 0.65),
    };
  }

  private rationale(
    input: ReasoningInvestmentInput,
    signals: Required<ReasoningUtilitySignals>,
    utility: number,
    cost: number
  ): string[] {
    const reasons = [`${input.kind} expected utility ${round(utility)} at normalized cost ${round(cost)}.`];
    if (signals.evidenceGain >= 0.65) reasons.push('The investment is expected to add grounded evidence.');
    if (signals.uncertaintyReduction >= 0.65) reasons.push('The investment targets material unresolved uncertainty.');
    if (signals.conflictResolution >= 0.65) reasons.push('The investment can reconcile conflicting actor beliefs.');
    if (signals.cacheConfidence >= 0.65) reasons.push('Prior cached execution evidence supports reuse.');
    if (signals.executionRisk >= 0.6 || signals.duplicationRisk >= 0.6) reasons.push('Risk adjustment reduced the expected return.');
    return reasons;
  }
}

function normalizeWeights<T extends string>(weights: Record<T, number>): Record<T, number> {
  const entries = Object.entries(weights) as Array<[T, number]>;
  const total = entries.reduce((sum, [, value]) => sum + Math.max(0, value), 0) || 1;
  return Object.fromEntries(entries.map(([key, value]) => [key, Math.max(0, value) / total])) as Record<T, number>;
}

function sumWeighted<T extends string>(signals: Record<T, number>, weights: Record<T, number>): number {
  return (Object.keys(weights) as T[]).reduce((sum, key) => sum + signals[key] * weights[key], 0);
}

function positive(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : fallback;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function round(value: number): number {
  return Number(value.toFixed(4));
}
