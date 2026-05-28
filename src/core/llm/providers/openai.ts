// OpenAI LLM provider implementation

import OpenAI from 'openai';
import type {
  LLMProvider,
  LLMMessage,
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMStreamChunk,
  ProviderConfig,
} from '../types.js';

export class OpenAIProvider implements LLMProvider {
  readonly name = 'openai';
  readonly defaultModel = 'gpt-4o';
  private client: OpenAI | null = null;
  private config: ProviderConfig;

  constructor(config?: ProviderConfig) {
    this.config = config || {};
    this.initialize();
  }

  private initialize(): void {
    if (this.config.apiKey) {
      this.client = new OpenAI({
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
      throw new Error('OpenAI provider not configured');
    }

    const model = options?.model || this.config.model || this.defaultModel;

    const response = await this.client.chat.completions.create({
      model,
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      temperature: options?.temperature ?? this.config.temperature ?? 0.5,
      max_tokens: options?.maxTokens || this.config.maxTokens || 4096,
      top_p: options?.topP,
      stop: options?.stop,
    });

    const choice = response.choices[0];
    return {
      content: choice.message.content || '',
      usage: response.usage ? {
        promptTokens: response.usage.prompt_tokens,
        completionTokens: response.usage.completion_tokens,
        totalTokens: response.usage.total_tokens,
      } : undefined,
      model,
      finishReason: choice.finish_reason ?? undefined,
    };
  }

  async *stream(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    if (!this.client) {
      throw new Error('OpenAI provider not configured');
    }

    const model = options?.model || this.config.model || this.defaultModel;

    const stream = await this.client.chat.completions.create({
      model,
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      temperature: options?.temperature ?? this.config.temperature ?? 0.5,
      max_tokens: options?.maxTokens || this.config.maxTokens || 4096,
      top_p: options?.topP,
      stop: options?.stop,
      stream: true,
      stream_options: { include_usage: true },
    });

    let totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';

      if (chunk.usage) {
        totalUsage = {
          promptTokens: chunk.usage.prompt_tokens,
          completionTokens: chunk.usage.completion_tokens,
          totalTokens: chunk.usage.total_tokens,
        };
      }

      if (content) {
        yield {
          content,
          done: false,
        };
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
    if (!this.client) {
      throw new Error('OpenAI provider not configured');
    }

    const model = options?.model || this.config.model || this.defaultModel;

    const response = await this.client.chat.completions.create({
      model,
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      temperature: options?.temperature ?? this.config.temperature ?? 0.5,
      max_tokens: options?.maxTokens || this.config.maxTokens || 4096,
      response_format: { type: 'json_object' },
    });

    const content = response.choices[0].message.content || '';
    try {
      return JSON.parse(content) as T;
    } catch {
      throw new Error(`Failed to parse JSON response: ${content}`);
    }
  }
}

/**
 * Create OpenAI provider from environment
 */
export function createOpenAIProvider(): LLMProvider {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL;
  const model = process.env.DEFAULT_MODEL;

  return new OpenAIProvider({
    apiKey: apiKey || undefined,
    baseUrl: baseUrl || undefined,
    model: model || undefined,
  });
}