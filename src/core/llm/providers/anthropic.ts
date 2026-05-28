// Anthropic LLM provider implementation

import Anthropic from '@anthropic-ai/sdk';
import type {
  LLMProvider,
  LLMMessage,
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMStreamChunk,
  ProviderConfig,
} from '../types.js';
import { logger } from '../../utils/logger.js';

export class AnthropicProvider implements LLMProvider {
  readonly name = 'anthropic';
  readonly defaultModel = 'claude-sonnet-4-20250514';
  private client: Anthropic | null = null;
  private config: ProviderConfig;

  constructor(config?: ProviderConfig) {
    this.config = config || {};
    this.initialize();
  }

  private initialize(): void {
    if (this.config.apiKey) {
      this.client = new Anthropic({
        apiKey: this.config.apiKey,
        baseURL: this.config.baseUrl,
      });
    }
  }

  isConfigured(): boolean {
    return this.client !== null && !!this.config.apiKey;
  }

  async complete(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<LLMCompletionResult> {
    if (!this.client) {
      throw new Error('Anthropic provider not configured');
    }

    const model = options?.model || this.config.model || this.defaultModel;
    const systemMessage = messages.find(m => m.role === 'system');
    const filteredMessages = messages.filter(m => m.role !== 'system');

    const response = await this.client.messages.create({
      model,
      max_tokens: options?.maxTokens || this.config.maxTokens || 4096,
      temperature: options?.temperature ?? this.config.temperature ?? 0.5,
      system: systemMessage?.content,
      messages: filteredMessages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
    });

    return {
      content: response.content[0].type === 'text' ? response.content[0].text : '',
      usage: {
        promptTokens: response.usage.input_tokens,
        completionTokens: response.usage.output_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens,
      },
      model,
      finishReason: response.stop_reason ?? undefined,
    };
  }

  async *stream(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    if (!this.client) {
      throw new Error('Anthropic provider not configured');
    }

    const model = options?.model || this.config.model || this.defaultModel;
    const systemMessage = messages.find(m => m.role === 'system');
    const filteredMessages = messages.filter(m => m.role !== 'system');

    const stream = await this.client.messages.stream({
      model,
      max_tokens: options?.maxTokens || this.config.maxTokens || 4096,
      temperature: options?.temperature ?? this.config.temperature ?? 0.5,
      system: systemMessage?.content,
      messages: filteredMessages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
    });

    let fullContent = '';
    let totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    for await (const event of stream) {
      if (event.type === 'content_block_delta') {
        if (event.delta.type === 'text_delta') {
          fullContent += event.delta.text;
          yield {
            content: event.delta.text,
            done: false,
          };
        }
      } else if (event.type === 'message_delta') {
        if (event.usage && event.usage.input_tokens !== null && event.usage.output_tokens !== null) {
          totalUsage = {
            promptTokens: event.usage.input_tokens,
            completionTokens: event.usage.output_tokens,
            totalTokens: event.usage.input_tokens + event.usage.output_tokens,
          };
        }
      }
    }

    yield {
      content: '',
      done: true,
      usage: totalUsage,
    };
  }

  async completeJSON<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<T> {
    const result = await this.complete(messages, options);
    try {
      return JSON.parse(result.content) as T;
    } catch {
      throw new Error(`Failed to parse JSON response: ${result.content}`);
    }
  }
}

/**
 * Create Anthropic provider from environment
 */
export function createAnthropicProvider(): LLMProvider {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  const baseUrl = process.env.ANTHROPIC_BASE_URL;
  const model = process.env.DEFAULT_MODEL;

  return new AnthropicProvider({
    apiKey: apiKey || undefined,
    baseUrl: baseUrl || undefined,
    model: model || undefined,
  });
}