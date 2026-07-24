import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMJSONCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class OutputContractLLM implements LLMProvider {
  readonly name = 'output-contract-test';
  readonly defaultModel = 'test-model';
  repairs = 0;

  async complete(): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
    };
  }

  async *stream(): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield {
      content: 'The best hint is cinema because it links the intended concepts, but',
      done: true,
      finishReason: 'length',
      usage: { promptTokens: 8, completionTokens: 8, totalTokens: 16 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('repair a root agent response')) {
      this.repairs += 1;
      return { finalResponse: 'FINAL_HINT: cinema' } as T;
    }
    return {
      action: 'solve_directly',
      reason: 'The bounded task can be answered directly.',
    } satisfies DelegationDecision as T;
  }

  async completeJSONWithUsage<T>(
    messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages);
    return {
      value,
      completion: {
        content: JSON.stringify(value),
        usage: { promptTokens: 6, completionTokens: 3, totalTokens: 9 },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('root output contracts', () => {
  it('repairs a truncated response and records the repair as a cached step', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-output-contract-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      tools: {
        approval: { readOnly: 'deny', write: 'deny', execute: 'deny' },
        web: { enabled: false },
        executionLoop: { enabled: false },
      },
    }, null, 2));
    const llm = new OutputContractLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'root-output-contract-test',
      workspaceCwd,
      llmProvider: llm,
    });

    const result = await runtime.handleUserTurn(
      'Choose one concise hint. End with exactly: FINAL_HINT: <one word>'
    );

    expect(result.finalResponse).toBe('FINAL_HINT: cinema');
    expect(llm.repairs).toBe(1);
    expect(result.executionTree.steps).toHaveLength(2);
    expect(result.executionTree.steps[1]).toMatchObject({
      status: 'completed',
      decision: { action: 'finalize' },
      resultSummary: 'FINAL_HINT: cinema',
    });
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'llm.stream.truncated',
      'root.output_contract.repair.started',
      'root.output_contract.repair.completed',
      'execution.cache.snapshot.recorded',
    ]));
    await runtime.shutdown();
  });
});
