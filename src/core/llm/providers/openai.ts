// OpenAI LLM provider implementation

import OpenAI from 'openai';
import type {
  LLMProvider,
  LLMMessage,
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMStreamChunk,
  LLMJSONCompletionResult,
  ProviderConfig,
} from '../types.js';
import { tokenUsageRegistry } from '../usage.js';

export class OpenAIProvider implements LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  private client: OpenAI | null = null;
  private config: ProviderConfig;

  constructor(config?: ProviderConfig, name = 'openai', defaultModel = 'gpt-4o') {
    this.name = name;
    this.defaultModel = defaultModel;
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
      usage: tokenUsageRegistry.normalize({ provider: this.name, model, usage: response.usage, messages, output: choice.message.content || '' }),
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

    let rawUsage: unknown;

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';

      if (chunk.usage) rawUsage = chunk.usage;

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
      usage: tokenUsageRegistry.normalize({ provider: this.name, model, usage: rawUsage, messages }),
    };
  }

  async completeJSON<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<T> {
    return (await this.completeJSONWithUsage<T>(messages, options)).value;
  }

  async completeJSONWithUsage<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<LLMJSONCompletionResult<T>> {
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
      return {
        value: JSON.parse(content) as T,
        completion: {
          content,
          usage: tokenUsageRegistry.normalize({ provider: this.name, model, usage: response.usage, messages, output: content }),
          model,
          finishReason: response.choices[0].finish_reason ?? undefined,
        },
      };
    } catch {
      throw new Error(`Failed to parse JSON response: ${content}`);
    }
  }
}

export class DeepSeekProvider extends OpenAIProvider {
  constructor(config?: ProviderConfig) {
    super(config, 'deepseek', 'deepseek-v4-flash');
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

/**
 * Create DeepSeek provider using the OpenAI-compatible API.
 */
export function createDeepSeekProvider(): LLMProvider {
  return new DeepSeekProvider({
    apiKey: process.env.DEEPSEEK_API_KEY || undefined,
    baseUrl: process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com',
    model: process.env.DEFAULT_MODEL || 'deepseek-v4-flash',
  });
}
