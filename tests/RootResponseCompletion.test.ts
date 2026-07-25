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

async function createWorkspace(prefix: string): Promise<string> {
  const workspaceCwd = await mkdtemp(path.join(tmpdir(), prefix));
  await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
  await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
    tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    tools: {
      approval: { readOnly: 'deny', write: 'deny', execute: 'deny' },
      web: { enabled: false },
      executionLoop: { enabled: false },
    },
  }, null, 2));
  return workspaceCwd;
}

class ContinuationLLM implements LLMProvider {
  readonly name = 'continuation-test';
  readonly defaultModel = 'test-model';
  streamCalls = 0;

  async complete(): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
    };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    this.streamCalls += 1;
    const isContinuation = messages.some(message =>
      message.content.includes('Continue a response that the provider cut off')
    );
    yield isContinuation
      ? {
        content: ' beta omega',
        done: true,
        finishReason: 'stop',
        usage: { promptTokens: 12, completionTokens: 3, totalTokens: 15 },
      }
      : {
        content: 'FINAL_TEXT: Alpha',
        done: true,
        finishReason: 'length',
        usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 },
      };
  }

  async completeJSON<T>(): Promise<T> {
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

class AcceptanceRepairLLM implements LLMProvider {
  readonly name = 'acceptance-repair-test';
  readonly defaultModel = 'test-model';
  acceptanceAudits = 0;
  repairCalls = 0;

  async complete(): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
    };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('Repair the candidate into one complete final response')) {
      this.repairCalls += 1;
      yield {
        content: 'Alpha, beta, gamma, and delta are all covered.\nFINAL_REPORT: complete',
        done: true,
        finishReason: 'stop',
        usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 },
      };
      return;
    }
    yield {
      content: 'Alpha and beta are covered, but gamma and delta are omitted requirements.\nFINAL_REPORT: partial',
      done: true,
      finishReason: 'stop',
      usage: { promptTokens: 12, completionTokens: 10, totalTokens: 22 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('final-response acceptance auditor')) {
      this.acceptanceAudits += 1;
      return (this.acceptanceAudits === 1
        ? {
          complete: false,
          unmetRequirements: ['Include gamma', 'Include delta'],
          reason: 'Two explicit numbered obligations are missing.',
        }
        : {
          complete: true,
          unmetRequirements: [],
          reason: 'All four explicit obligations are present.',
        }) as T;
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
        usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('root response completion', () => {
  it('continues a provider-truncated root response without repeating its prefix', async () => {
    const workspaceCwd = await createWorkspace('roy-root-continuation-');
    const llm = new ContinuationLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'root-continuation-test',
      workspaceCwd,
      llmProvider: llm,
    });

    const result = await runtime.handleUserTurn(
      'Return a short phrase. End with exactly: FINAL_TEXT: <phrase>'
    );

    expect(result.finalResponse).toBe('FINAL_TEXT: Alpha beta omega');
    expect(llm.streamCalls).toBe(2);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'llm.stream.truncated',
      'llm.stream.continuation.started',
      'llm.stream.continuation.completed',
    ]));
    await runtime.shutdown();
  });

  it('audits and repairs an answer that leaves explicit obligations unmet', async () => {
    const workspaceCwd = await createWorkspace('roy-root-acceptance-');
    const llm = new AcceptanceRepairLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'root-acceptance-test',
      workspaceCwd,
      llmProvider: llm,
    });

    const result = await runtime.handleUserTurn([
      'Produce one report covering every numbered requirement.',
      '1. Include alpha.',
      '2. Include beta.',
      '3. Include gamma.',
      '4. Include delta.',
      'End with exactly: FINAL_REPORT: <status>',
    ].join('\n'));

    expect(result.finalResponse).toContain('Alpha, beta, gamma, and delta');
    expect(result.finalResponse).toContain('FINAL_REPORT: complete');
    expect(llm.acceptanceAudits).toBe(2);
    expect(llm.repairCalls).toBe(1);
    expect(result.executionTree.steps.at(-1)).toMatchObject({
      status: 'completed',
      decision: { action: 'finalize' },
    });
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'root.response.acceptance.audit.started',
      'root.response.acceptance.unmet',
      'root.response.acceptance.repair.started',
      'root.response.acceptance.repair.completed',
      'execution.feedback.captured',
      'execution.cache.snapshot.recorded',
    ]));
    await runtime.shutdown();
  });
});
