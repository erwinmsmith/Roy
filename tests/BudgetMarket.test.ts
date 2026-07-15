import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BudgetMarket, WeightedReasoningInvestmentModel } from '../src/core/budget/index.js';
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
  it('estimates multidimensional reasoning return through a replaceable utility model', () => {
    const model = new WeightedReasoningInvestmentModel();
    const strong = model.estimate({
      kind: 'evidence_review', requesterId: 'researcher', parentId: 'root', purpose: 'inspect evidence',
      resources: { tokens: 2000, contextTokens: 500, toolCalls: 2 },
      signals: {
        rootUtility: 0.85, parentUtility: 0.8, historicalUtility: 0.75,
        evidenceGain: 0.95, uncertaintyReduction: 0.85, cacheConfidence: 0.7,
        duplicationRisk: 0.05, executionRisk: 0.1, confidence: 0.9,
      },
    });
    const risky = model.estimate({
      kind: 'duplicate_review', requesterId: 'critic', parentId: 'root', purpose: 'duplicate work',
      resources: { tokens: 6000, contextTokens: 3000, toolCalls: 6 },
      signals: {
        rootUtility: 0.6, parentUtility: 0.5, evidenceGain: 0.2,
        duplicationRisk: 0.9, executionRisk: 0.8, confidence: 0.4,
      },
    });
    expect(strong.riskAdjustedUtility).toBeGreaterThan(risky.riskAdjustedUtility);
    expect(strong.expectedReturn).toBeGreaterThan(risky.expectedReturn);
    expect(strong.components).toMatchObject({ evidenceGain: 0.95, toolCost: 0.2 });
  });

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

  it('records realized outcomes and recomputes efficiency from observed utility', () => {
    const market = new BudgetMarket(() => 0, { mode: 'fixed' });
    market.configure(1000);
    const allocation = market.request({
      requesterId: 'reviewer', parentId: 'root', requestedTokens: 500, minimumTokens: 100,
      expectedUtility: 0.9, purpose: 'review',
    });
    market.settle(allocation.id, 400);
    const before = market.getAllocation(allocation.id)!;
    const after = market.recordOutcome(allocation.id, {
      success: false,
      quality: 0.2,
      evidenceGain: 0.1,
      error: 'unsupported conclusion',
    })!;
    expect(after.outcome).toMatchObject({ success: false, error: 'unsupported conclusion' });
    expect(after.efficiency).toBeLessThan(before.efficiency!);
    expect(market.getState().ledger.at(-1)?.type).toBe('outcome');

    const next = market.request({
      requesterId: 'reviewer-2', parentId: 'root', requestedTokens: 400, minimumTokens: 100,
      expectedUtility: 0.9, purpose: 'review',
    });
    expect(next.request.investment?.components.historicalUtility).toBeLessThan(0.2);
    expect(market.getState().outcomeHistory).toMatchObject([
      { key: 'review', count: 1, successCount: 0 },
    ]);
    expect(() => market.recordOutcome(allocation.id, { success: true })).toThrow('already has a recorded outcome');
  });

  it('settles against the configured accounting dimension while retaining full model usage', () => {
    const market = new BudgetMarket(() => 0, { mode: 'fixed', accountingDimension: 'thinking_tokens' });
    market.configure(1000);
    const allocation = market.request({
      requesterId: 'reasoner-1', parentId: 'root', requestedTokens: 200, minimumTokens: 50,
      purpose: 'reasoning allocation',
    });
    const settled = market.settle(allocation.id, {
      promptTokens: 250, completionTokens: 200, totalTokens: 450,
      inputTokens: 250, outputTokens: 200, thinkingTokens: 75,
    });
    expect(settled?.consumedTokens).toBe(75);
    expect(settled?.usage?.totalTokens).toBe(450);
    expect(settled?.usage?.thinkingTokens).toBe(75);
    expect(market.getState().usedTokens).toBe(0);
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

  it('rebalances duplicate requester purposes by allocation order instead of conflating bids', () => {
    const market = new BudgetMarket(() => 0, { mode: 'market', minimumGrantTokens: 100 });
    market.configure(2000);
    const allocations = market.requestMany([
      { requesterId: 'same-agent', parentId: 'root', requestedTokens: 500, minimumTokens: 100, purpose: 'same-purpose' },
      { requesterId: 'same-agent', parentId: 'root', requestedTokens: 1500, minimumTokens: 100, purpose: 'same-purpose' },
    ]);
    market.configure(1000);
    market.rebalance();
    const first = market.getAllocation(allocations[0].id);
    const second = market.getAllocation(allocations[1].id);
    expect(first?.allocatedTokens).not.toBe(second?.allocatedTokens);
    expect((first?.allocatedTokens ?? 0) + (second?.allocatedTokens ?? 0)).toBeLessThanOrEqual(1000);
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

  it('uses thinking tokens for runtime supply and completion capacity when configured', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-thinking-budget-'));
    const bootstrapRuntime = new Runtime();
    await bootstrapRuntime.initialize({ sessionId: 'thinking-config-bootstrap', workspaceCwd: cwd, llmProvider: new MeteredLLM() });
    await bootstrapRuntime.shutdown();

    const configPath = path.join(cwd, '.roy', 'config.json');
    const workspaceConfig = JSON.parse(await readFile(configPath, 'utf8')) as {
      budgetMarket: { accountingDimension: string };
    };
    workspaceConfig.budgetMarket.accountingDimension = 'thinking_tokens';
    await writeFile(configPath, `${JSON.stringify(workspaceConfig, null, 2)}\n`, 'utf8');

    const runtime = new Runtime();
    const llm = new MeteredLLM();
    await runtime.initialize({ sessionId: 'thinking-budget-runtime', workspaceCwd: cwd, llmProvider: llm });
    await runtime.handleUserTurn('Reply directly and briefly.');

    const soloAllocation = runtime.getBudgetMarketState().allocations.find(
      allocation => allocation.request.purpose === 'root.solo_reasoning'
    );
    expect(runtime.getBudgetMarketState().accountingDimension).toBe('thinking_tokens');
    expect(soloAllocation?.consumedTokens).toBe(30);
    expect(soloAllocation?.usage?.totalTokens).toBe(200);
    expect(runtime.getBudgetMarketState().usedTokens).toBeGreaterThan(30);
    expect(llm.lastMaxTokens).toBe(2400);
    await runtime.shutdown();
  });
});
