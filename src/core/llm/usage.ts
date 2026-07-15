import type { LLMMessage, NormalizedModelTokenUsage, TokenMetricAvailability } from './types.js';

type RawUsage = Record<string, unknown>;

export interface TokenUsageNormalizationInput {
  provider: string;
  model?: string;
  usage?: unknown;
  messages?: LLMMessage[];
  output?: string;
}

export interface TokenUsageNormalizer {
  readonly provider: string;
  normalize(input: TokenUsageNormalizationInput): NormalizedModelTokenUsage | undefined;
}

export interface TokenEstimator {
  supports(provider: string, model?: string): boolean;
  estimateText(text: string, provider: string, model?: string): number;
}

const numberAt = (value: unknown, ...path: string[]): number | undefined => {
  let current: unknown = value;
  for (const key of path) {
    if (!current || typeof current !== 'object') return undefined;
    current = (current as RawUsage)[key];
  }
  return typeof current === 'number' && Number.isFinite(current) ? Math.max(0, Math.floor(current)) : undefined;
};

const firstNumber = (value: unknown, paths: string[][]): number | undefined => {
  for (const path of paths) {
    const found = numberAt(value, ...path);
    if (found !== undefined) return found;
  }
  return undefined;
};

const availability = (value: number | undefined): TokenMetricAvailability => value === undefined ? 'unavailable' : 'reported';

function canonicalUsage(input: {
  provider: string;
  model?: string;
  input?: number;
  output?: number;
  total?: number;
  thinking?: number;
  cachedInput?: number;
  cacheCreationInput?: number;
  source?: 'provider' | 'estimated';
  estimatedAvailability?: boolean;
}): NormalizedModelTokenUsage {
  const inputTokens = input.input ?? 0;
  const outputTokens = input.output ?? 0;
  const metricAvailability: TokenMetricAvailability = input.estimatedAvailability ? 'estimated' : 'reported';
  return {
    promptTokens: inputTokens,
    completionTokens: outputTokens,
    totalTokens: input.total ?? inputTokens + outputTokens,
    inputTokens,
    outputTokens,
    thinkingTokens: input.thinking ?? null,
    cachedInputTokens: input.cachedInput ?? null,
    cacheCreationInputTokens: input.cacheCreationInput ?? null,
    provider: input.provider,
    model: input.model,
    source: input.source ?? 'provider',
    availability: {
      input: input.estimatedAvailability ? metricAvailability : availability(input.input),
      output: input.estimatedAvailability ? metricAvailability : availability(input.output),
      thinking: availability(input.thinking),
      cachedInput: availability(input.cachedInput),
      cacheCreationInput: availability(input.cacheCreationInput),
    },
  };
}

export class OpenAICompatibleUsageNormalizer implements TokenUsageNormalizer {
  constructor(readonly provider: string) {}

  normalize(input: TokenUsageNormalizationInput): NormalizedModelTokenUsage | undefined {
    if (!input.usage) return undefined;
    const prompt = firstNumber(input.usage, [['prompt_tokens'], ['input_tokens']]);
    const completion = firstNumber(input.usage, [['completion_tokens'], ['output_tokens']]);
    if (prompt === undefined && completion === undefined) return undefined;
    return canonicalUsage({
      provider: input.provider,
      model: input.model,
      input: prompt,
      output: completion,
      total: firstNumber(input.usage, [['total_tokens']]),
      thinking: firstNumber(input.usage, [
        ['completion_tokens_details', 'reasoning_tokens'],
        ['reasoning_tokens'],
        ['thinking_tokens'],
      ]),
      cachedInput: firstNumber(input.usage, [
        ['prompt_tokens_details', 'cached_tokens'],
        ['cached_input_tokens'],
        ['cache_read_input_tokens'],
      ]),
      cacheCreationInput: firstNumber(input.usage, [['cache_creation_input_tokens']]),
    });
  }
}

export class AnthropicUsageNormalizer implements TokenUsageNormalizer {
  readonly provider = 'anthropic';

  normalize(input: TokenUsageNormalizationInput): NormalizedModelTokenUsage | undefined {
    if (!input.usage) return undefined;
    const inputTokens = firstNumber(input.usage, [['input_tokens'], ['prompt_tokens']]);
    const outputTokens = firstNumber(input.usage, [['output_tokens'], ['completion_tokens']]);
    if (inputTokens === undefined && outputTokens === undefined) return undefined;
    return canonicalUsage({
      provider: input.provider,
      model: input.model,
      input: inputTokens,
      output: outputTokens,
      thinking: firstNumber(input.usage, [['thinking_tokens'], ['reasoning_tokens']]),
      cachedInput: firstNumber(input.usage, [['cache_read_input_tokens'], ['cached_input_tokens']]),
      cacheCreationInput: firstNumber(input.usage, [['cache_creation_input_tokens']]),
    });
  }
}

export class CharacterTokenEstimator implements TokenEstimator {
  supports(): boolean { return true; }

  estimateText(text: string, provider: string, model?: string): number {
    const family = `${provider}/${model ?? ''}`.toLowerCase();
    const charsPerToken = family.includes('deepseek') ? 3.5 : family.includes('anthropic') || family.includes('claude') ? 3.8 : 4;
    return Math.max(0, Math.ceil(text.length / charsPerToken));
  }
}

export class TokenUsageRegistry {
  private normalizers = new Map<string, TokenUsageNormalizer>();
  private estimators: TokenEstimator[] = [new CharacterTokenEstimator()];

  constructor() {
    this.registerNormalizer(new OpenAICompatibleUsageNormalizer('openai'));
    this.registerNormalizer(new OpenAICompatibleUsageNormalizer('deepseek'));
    this.registerNormalizer(new AnthropicUsageNormalizer());
  }

  registerNormalizer(normalizer: TokenUsageNormalizer): void {
    this.normalizers.set(normalizer.provider.toLowerCase(), normalizer);
  }

  registerEstimator(estimator: TokenEstimator): void {
    this.estimators.unshift(estimator);
  }

  estimateText(text: string, provider: string, model?: string): number {
    const estimator = this.estimators.find(item => item.supports(provider, model));
    return estimator?.estimateText(text, provider, model) ?? Math.max(0, Math.ceil(text.length / 4));
  }

  normalize(input: TokenUsageNormalizationInput): NormalizedModelTokenUsage | undefined {
    const provider = input.provider.toLowerCase();
    const normalizer = this.normalizers.get(provider) ?? new OpenAICompatibleUsageNormalizer(provider);
    const reported = normalizer.normalize(input);
    if (reported) return reported;
    if (!input.messages && input.output === undefined) return undefined;
    const prompt = this.estimateText((input.messages ?? []).map(message => `${message.role}:${message.content}`).join('\n'), provider, input.model);
    const completion = this.estimateText(input.output ?? '', provider, input.model);
    return canonicalUsage({
      provider,
      model: input.model,
      input: prompt,
      output: completion,
      source: 'estimated',
      estimatedAvailability: true,
    });
  }
}

export const tokenUsageRegistry = new TokenUsageRegistry();

export function emptyModelTokenUsage(provider?: string, model?: string): NormalizedModelTokenUsage {
  return canonicalUsage({ provider: provider ?? 'unknown', model, input: 0, output: 0 });
}
