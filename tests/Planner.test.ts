import { describe, expect, it } from 'vitest';
import { CompositePlanner, LLMPlanner } from '../src/core/actions/Planner.js';
import type { Plan, PlanContext, Planner } from '../src/core/actions/Planner.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

const context: PlanContext = {
  agentInfo: { name: 'Roy' },
  observation: 'inspect the project',
  availableActions: ['fs.list'],
};

class PlannerLLM implements LLMProvider {
  readonly name = 'planner-test';
  readonly defaultModel = 'test';

  constructor(
    private readonly result: unknown,
    private readonly streamContent = JSON.stringify(result)
  ) {}

  async complete(
    _messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): Promise<LLMCompletionResult> {
    return { content: this.streamContent };
  }

  async *stream(
    _messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: this.streamContent, done: true };
  }

  async completeJSON<T>(): Promise<T> {
    return this.result as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

const fixedPlanner = (name: string, result: Plan | null): Planner => ({
  name,
  async plan() {
    return result;
  },
  async *planStream() {
    if (result) yield JSON.stringify(result);
    return result;
  },
});

describe('LLMPlanner', () => {
  it('rejects actions outside the advertised capability set', async () => {
    const planner = new LLMPlanner({
      name: 'validated',
      llm: new PlannerLLM({ action: 'shell.exec', params: {} }),
    });

    await expect(planner.plan(context)).resolves.toBeNull();
  });

  it('normalizes params and confidence from fenced streaming JSON', async () => {
    const planner = new LLMPlanner({
      name: 'streaming',
      llm: new PlannerLLM({}, '```json\n{"action":"fs.list","params":null,"confidence":2}\n```'),
    });
    const iterator = planner.planStream(context);
    let step = await iterator.next();
    const chunks: string[] = [];
    while (!step.done) {
      chunks.push(step.value);
      step = await iterator.next();
    }

    expect(chunks.join('')).toContain('fs.list');
    expect(step.value).toEqual({
      action: 'fs.list',
      params: {},
      reasoning: undefined,
      confidence: 1,
    });
  });
});

describe('CompositePlanner', () => {
  it('honors fallback configuration for direct and streaming plans', async () => {
    const expected: Plan = { action: 'fs.list', params: { path: '.' } };
    const primary = fixedPlanner('primary', null);
    const fallback = fixedPlanner('fallback', expected);

    await expect(
      new CompositePlanner('strict', [primary, fallback], false).plan(context)
    ).resolves.toBeNull();
    await expect(
      new CompositePlanner('fallback', [primary, fallback], true).plan(context)
    ).resolves.toEqual(expected);

    const iterator = new CompositePlanner('stream', [primary, fallback], true).planStream(context);
    let step = await iterator.next();
    while (!step.done) step = await iterator.next();
    expect(step.value).toEqual(expected);
  });
});
