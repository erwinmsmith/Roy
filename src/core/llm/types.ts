// LLM Provider types and interfaces

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMCompletionOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  stop?: string[];
  stream?: boolean;
}

export type TokenMetricAvailability = 'reported' | 'estimated' | 'unavailable';

/** Canonical token accounting shared by every provider and the budget system. */
export interface ModelTokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  inputTokens?: number;
  outputTokens?: number;
  thinkingTokens?: number | null;
  cachedInputTokens?: number | null;
  cacheCreationInputTokens?: number | null;
  provider?: string;
  model?: string;
  source?: 'provider' | 'estimated';
  availability?: {
    input: TokenMetricAvailability;
    output: TokenMetricAvailability;
    thinking: TokenMetricAvailability;
    cachedInput: TokenMetricAvailability;
    cacheCreationInput: TokenMetricAvailability;
  };
}

export interface NormalizedModelTokenUsage extends ModelTokenUsage {
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number | null;
  cachedInputTokens: number | null;
  cacheCreationInputTokens: number | null;
  provider?: string;
  model?: string;
  source: 'provider' | 'estimated';
  availability: {
    input: TokenMetricAvailability;
    output: TokenMetricAvailability;
    thinking: TokenMetricAvailability;
    cachedInput: TokenMetricAvailability;
    cacheCreationInput: TokenMetricAvailability;
  };
}

export interface LLMCompletionResult {
  content: string;
  usage?: ModelTokenUsage;
  model?: string;
  finishReason?: string;
}

export interface LLMStreamChunk {
  content: string;
  done: boolean;
  usage?: ModelTokenUsage;
  finishReason?: string;
}

export interface LLMJSONCompletionResult<T> {
  value: T;
  completion: LLMCompletionResult;
}

/**
 * Base interface for LLM providers
 * Implement this interface to add support for new LLM providers
 */
export interface LLMProvider {
  readonly name: string;
  readonly defaultModel: string;

  /**
   * Generate a completion
   */
  complete(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<LLMCompletionResult>;

  /**
   * Generate a streaming completion
   */
  stream(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown>;

  /**
   * Generate a JSON completion
   */
  completeJSON<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<T>;

  /** Optional richer JSON API used when callers need provider token accounting. */
  completeJSONWithUsage?<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<LLMJSONCompletionResult<T>>;

  /**
   * Check if the provider is configured
   */
  isConfigured(): boolean;
}

/**
 * Provider configuration
 */
export interface ProviderConfig {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

/**
 * LLM Provider factory
 */
export interface LLMProviderFactory {
  create(config: ProviderConfig): LLMProvider;
}
