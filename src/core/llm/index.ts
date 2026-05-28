// LLM module exports

export type {
  LLMProvider,
  LLMMessage,
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMStreamChunk,
  ProviderConfig,
} from './types.js';

export { AnthropicProvider, createAnthropicProvider } from './providers/anthropic.js';
export { OpenAIProvider, DeepSeekProvider, createOpenAIProvider, createDeepSeekProvider } from './providers/openai.js';

import type { LLMProvider } from './types.js';
import { createAnthropicProvider } from './providers/anthropic.js';
import { createOpenAIProvider, createDeepSeekProvider } from './providers/openai.js';

/**
 * LLM Provider factory - creates provider based on model name
 */
export class LLMFactory {
  private providers: Map<string, LLMProvider> = new Map();

  constructor() {
    // Register default providers
    this.providers.set('anthropic', createAnthropicProvider());
    this.providers.set('openai', createOpenAIProvider());
    this.providers.set('deepseek', createDeepSeekProvider());
  }

  /**
   * Register a provider factory
   */
  register(name: string, factory: () => LLMProvider): void {
    this.providers.set(name, factory());
  }

  /**
   * Get provider by name
   */
  get(name: string): LLMProvider | undefined {
    return this.providers.get(name);
  }

  /**
   * Get or create default provider
   */
  getDefault(): LLMProvider {
    // Try Anthropic first if configured
    const anthropic = this.providers.get('anthropic');
    if (anthropic?.isConfigured()) {
      return anthropic;
    }

    // Fall back to OpenAI
    const openai = this.providers.get('openai');
    if (openai?.isConfigured()) {
      return openai;
    }

    const deepseek = this.providers.get('deepseek');
    if (deepseek?.isConfigured()) {
      return deepseek;
    }

    // Return first available provider (may not be configured)
    const first = Array.from(this.providers.values())[0];
    if (first) return first;

    throw new Error('No LLM providers configured');
  }

  /**
   * List all available providers
   */
  listProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  /**
   * List configured providers
   */
  listConfigured(): string[] {
    return this.listProviders().filter(name => {
      const provider = this.providers.get(name);
      return provider?.isConfigured();
    });
  }
}

export const llmFactory = new LLMFactory();
