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

export interface LLMCompletionResult {
  content: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  model?: string;
  finishReason?: string;
}

export interface LLMStreamChunk {
  content: string;
  done: boolean;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
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