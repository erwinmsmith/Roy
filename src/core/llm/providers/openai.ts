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
        timeout: this.config.timeoutMs ?? 120_000,
        maxRetries: this.config.maxRetries ?? 0,
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
      max_tokens: options?.maxTokens ?? this.config.maxTokens ?? 4096,
      top_p: options?.topP,
      stop: options?.stop,
    }, requestOptions(options, this.config.timeoutMs));

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
      max_tokens: options?.maxTokens ?? this.config.maxTokens ?? 4096,
      top_p: options?.topP,
      stop: options?.stop,
      stream: true,
      stream_options: { include_usage: true },
    }, requestOptions(options, this.config.timeoutMs));

    let rawUsage: unknown;
    let finishReason: string | undefined;

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      const chunkFinishReason = chunk.choices[0]?.finish_reason;

      if (chunk.usage) rawUsage = chunk.usage;
      if (chunkFinishReason) finishReason = chunkFinishReason;

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
      finishReason,
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
      max_tokens: options?.maxTokens ?? this.config.maxTokens ?? 4096,
      response_format: { type: 'json_object' },
    }, requestOptions(options, this.config.timeoutMs));

    const message = response.choices[0].message as unknown as {
      content?: string | null;
      reasoning_content?: string | null;
    };
    const content = message.content || message.reasoning_content || '';
    try {
      return {
        value: parseJSONResponse<T>(content),
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

function parseJSONResponse<T>(content: string): T {
  const trimmed = content.trim();
  if (!trimmed) throw new Error('Empty JSON response');
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  const candidates = [trimmed, fenced, extractJSONObject(trimmed)].filter(
    (value): value is string => Boolean(value)
  );
  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate) as T;
    } catch {
      // Try the next structured representation.
    }
  }
  throw new Error(`Failed to parse JSON response: ${content}`);
}

function extractJSONObject(content: string): string | undefined {
  const start = content.indexOf('{');
  if (start < 0) return undefined;
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < content.length; index += 1) {
    const char = content[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
      continue;
    }
    if (char === '"') quoted = true;
    else if (char === '{') depth += 1;
    else if (char === '}') {
      depth -= 1;
      if (depth === 0) return content.slice(start, index + 1);
    }
  }
  return undefined;
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
    timeoutMs: readPositiveInteger(
      process.env.OPENAI_REQUEST_TIMEOUT_MS ?? process.env.ROY_LLM_REQUEST_TIMEOUT_MS,
      120_000
    ),
    maxRetries: readNonNegativeInteger(
      process.env.OPENAI_PROVIDER_MAX_RETRIES ?? process.env.ROY_LLM_PROVIDER_MAX_RETRIES,
      0
    ),
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
    timeoutMs: readPositiveInteger(
      process.env.DEEPSEEK_REQUEST_TIMEOUT_MS ?? process.env.ROY_LLM_REQUEST_TIMEOUT_MS,
      120_000
    ),
    maxRetries: readNonNegativeInteger(
      process.env.DEEPSEEK_PROVIDER_MAX_RETRIES ?? process.env.ROY_LLM_PROVIDER_MAX_RETRIES,
      0
    ),
  });
}

function readPositiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function readNonNegativeInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

function requestOptions(
  options: LLMCompletionOptions | undefined,
  configuredTimeoutMs: number | undefined
): { timeout: number } {
  const configured = configuredTimeoutMs ?? 120_000;
  const requested = options?.timeoutMs;
  return {
    timeout: Number.isFinite(requested) && Number(requested) > 0
      ? Math.max(1, Math.min(configured, Math.floor(Number(requested))))
      : configured,
  };
}
