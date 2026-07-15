import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BudgetMarket } from '../src/core/budget/index.js';
import {
  AnthropicUsageNormalizer,
  OpenAICompatibleUsageNormalizer,
  TokenUsageRegistry,
} from '../src/core/llm/usage.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';

class MeteredLLM implements LLMProvider {
  readonly name = 'deepseek';
  readonly defaultModel = 'deepseek-test';
  lastMaxTokens?: number;

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'ok', usage: this.usage() };
  }

  async *stream(_messages: LLMMessage[], options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk> {
    this.lastMaxTokens = options?.maxTokens;
    yield { content: 'Roy metered response.', done: true, usage: this.usage() };
  }

  async completeJSON<T>(): Promise<T> {
    return { action: 'solve_directly', reason: 'Bounded direct request.' } satisfies DelegationDecision as T;
  }

  isConfigured(): boolean { return true; }

  private usage() {
    return {
      promptTokens: 120,
      completionTokens: 80,
      totalTokens: 200,
      inputTokens: 120,
      outputTokens: 80,
      thinkingTokens: 30,
      cachedInputTokens: 20,
      cacheCreationInputTokens: null,
      provider: 'deepseek',
      model: 'deepseek-test',
      source: 'provider' as const,
      availability: {
        input: 'reported' as const,
        output: 'reported' as const,
        thinking: 'reported' as const,
        cachedInput: 'reported' as const,
        cacheCreationInput: 'unavailable' as const,
      },
    };
  }
}

describe('Phase 5 budget market', () => {
  it('allocates competing bids by priority and utility through a pluggable market policy', () => {
    const market = new BudgetMarket(() => 0, { mode: 'market', minimumGrantTokens: 100 });
    market.configure(1000);
    const [low, critical] = market.requestMany([
      { requesterId: 'low-agent', parentId: 'root', requestedTokens: 800, minimumTokens: 100, priority: 'low', expectedUtility: 0.4, purpose: 'optional review' },
      { requesterId: 'critical-agent', parentId: 'root', requestedTokens: 800, minimumTokens: 100, priority: 'critical', expectedUtility: 0.95, purpose: 'required synthesis' },
    ]);
    expect(critical.allocatedTokens).toBeGreaterThan(low.allocatedTokens);
    expect(low.allocatedTokens + critical.allocatedTokens).toBeLessThanOrEqual(1000);
    expect(market.getState()).toMatchObject({ policy: 'market', limitMode: 'limited', sessionLimit: 1000 });
  });

  it('records canonical usage, efficiency, overrun, and ledger transitions', () => {
    const market = new BudgetMarket(() => 0, { mode: 'fixed' });
    market.configure(1000);
    const allocation = market.request({
      requesterId: 'researcher-1', parentId: 'root', requestedTokens: 400, minimumTokens: 100,
      expectedUtility: 0.8, purpose: 'inspection',
    });
    const settled = market.settle(allocation.id, {
      promptTokens: 250, completionTokens: 200, totalTokens: 450,
      inputTokens: 250, outputTokens: 200, thinkingTokens: 75,
      cachedInputTokens: 50, cacheCreationInputTokens: null,
    });
    expect(settled).toMatchObject({ status: 'exceeded', consumedTokens: 450, actualTokens: 450 });
    expect(settled?.usage?.thinkingTokens).toBe(75);
    expect(settled?.efficiency).toBeGreaterThan(0);
    expect(market.getState().ledger.map(entry => entry.type)).toEqual(
      expect.arrayContaining(['requested', 'allocated', 'consumed', 'exceeded', 'settled'])
    );
  });

  it('rebalances active allocations after the available session supply changes', () => {
    const market = new BudgetMarket(() => 0, { mode: 'market', minimumGrantTokens: 100 });
    market.configure(2000);
    market.requestMany([
      { requesterId: 'a', parentId: 'root', requestedTokens: 1000, minimumTokens: 100, priority: 'medium', purpose: 'a' },
      { requesterId: 'b', parentId: 'root', requestedTokens: 1000, minimumTokens: 100, priority: 'high', purpose: 'b' },
    ]);
    market.configure(1200);
    const result = market.rebalance();
    expect(result.changed.length).toBeGreaterThan(0);
    expect(result.reservedTokens).toBeLessThanOrEqual(1200);
    expect(market.getState().ledger.some(entry => entry.type === 'rebalanced')).toBe(true);
  });

  it('counts externally consumed runtime allocations without double-counting agent usage', () => {
    const market = new BudgetMarket(() => 0, { mode: 'fixed' });
    market.configure(1000);
    const allocation = market.request({
      requesterId: 'external-worker', parentId: 'root', actorType: 'runtime',
      requestedTokens: 1000, minimumTokens: 100, purpose: 'external evaluation',
    });
    market.consume(allocation.id, 250);
    expect(market.getState()).toMatchObject({ usedTokens: 250, reservedTokens: 750, availableTokens: 0 });
  });

  it('normalizes provider-specific reasoning and cache token fields', () => {
    const deepseek = new OpenAICompatibleUsageNormalizer('deepseek').normalize({
      provider: 'deepseek', model: 'deepseek-reasoner',
      usage: {
        prompt_tokens: 100, completion_tokens: 60, total_tokens: 160,
        completion_tokens_details: { reasoning_tokens: 25 },
        prompt_tokens_details: { cached_tokens: 40 },
      },
    });
    expect(deepseek).toMatchObject({ inputTokens: 100, outputTokens: 60, thinkingTokens: 25, cachedInputTokens: 40 });

    const anthropic = new AnthropicUsageNormalizer().normalize({
      provider: 'anthropic', model: 'claude-test',
      usage: { input_tokens: 90, output_tokens: 30, cache_read_input_tokens: 45, cache_creation_input_tokens: 10 },
    });
    expect(anthropic).toMatchObject({ inputTokens: 90, outputTokens: 30, cachedInputTokens: 45, cacheCreationInputTokens: 10 });
    expect(anthropic?.thinkingTokens).toBeNull();

    const estimated = new TokenUsageRegistry().normalize({
      provider: 'deepseek', model: 'unknown', messages: [{ role: 'user', content: 'x'.repeat(350) }], output: 'x'.repeat(70),
    });
    expect(estimated?.source).toBe('estimated');
    expect(estimated?.availability.thinking).toBe('unavailable');
  });

  it('meters root input, output, thinking, and cache usage through runtime events', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-budget-market-'));
    const runtime = new Runtime();
    const llm = new MeteredLLM();
    await runtime.initialize({ sessionId: 'budget-market-runtime', workspaceCwd: cwd, llmProvider: llm });
    await runtime.handleUserTurn('Reply directly and briefly.');
    const root = runtime.getBudgetState().perAgent.root;
    expect(root.inputTokens).toBeGreaterThan(120);
    expect(root.outputTokens).toBeGreaterThan(80);
    expect(root).toMatchObject({ thinkingTokens: 30, cachedInputTokens: 20 });
    const allocation = runtime.getBudgetMarketState().allocations.find(item => item.request.requesterId === 'root' && item.usage?.thinkingTokens === 30);
    expect(allocation).toMatchObject({ status: 'settled', consumedTokens: 200 });
    expect(allocation?.usage?.thinkingTokens).toBe(30);
    const events = runtime.getEvents().map(event => event.type);
    expect(events).toEqual(expect.arrayContaining(['budget.requested', 'budget.allocated', 'budget.consumed', 'budget.settled']));
    expect(llm.lastMaxTokens).toBeGreaterThan(0);
    expect(llm.lastMaxTokens).toBeLessThanOrEqual(2400);
    await runtime.shutdown();
  });
});
