import { describe, expect, it } from 'vitest';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class TerminalTaskLLM implements LLMProvider {
  readonly name = 'terminal-task-test';
  readonly defaultModel = 'test-model';

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'Created artifact.txt and verified its contents.' };
  }

  async *stream(): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'Created artifact.txt and verified its contents.', done: true };
  }

  async completeJSON<T>(messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes("root delegation controller")) {
      return { action: 'solve_directly', reason: 'The root has the required terminal capability.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('Completed tool round: 0')) {
        return {
          action: 'call_tools',
          reason: 'Create and verify the requested artifact.',
          calls: [{
            toolName: 'shell.exec',
            params: {
              command: "printf 'roy-terminal-ready' > artifact.txt && test \"$(cat artifact.txt)\" = roy-terminal-ready",
            },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'The artifact was created and verified.', calls: [] } as T;
    }
    return {} as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('benchmark terminal capability', () => {
  it('runs an explicitly authorized shell loop and persists its execution tree', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-terminal-task-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
      delegation: {
        candidateScoring: {
          enabledScorers: ['heuristic', 'cost', 'tom', 'cache_evolution'],
          minimumScore: 0.05,
        },
      },
      tools: {
        approval: {
          readOnly: 'auto',
          write: 'auto',
          execute: 'auto',
          overrides: {},
        },
        shell: {
          mode: 'unrestricted',
          shell: '/bin/sh',
          defaultTimeoutMs: 10_000,
          maxTimeoutMs: 60_000,
          defaultMaxOutputBytes: 40_000,
          maxCallsPerAgent: 10,
        },
        executionLoop: {
          enabled: true,
          maxRounds: 4,
          maxCallsPerRun: 6,
          maxConsecutiveFailures: 2,
          maxWallClockMs: 30_000,
          maxFetchesAfterSearch: 2,
          llmReplanning: true,
        },
      },
    }, null, 2));

    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'terminal-capability-test',
      workspaceCwd: workspace,
      llmProvider: new TerminalTaskLLM(),
    });
    const result = await runtime.handleUserTurn(
      'Use the terminal in this workspace to create artifact.txt, verify it, and report completion.'
    );

    expect(await readFile(path.join(workspace, 'artifact.txt'), 'utf8')).toBe('roy-terminal-ready');
    expect(result.finalResponse).toContain('Created artifact.txt');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.call',
      data: expect.objectContaining({ toolName: 'shell.exec' }),
    }));
    expect(result.executionTree.status).toBe('completed');
    expect(JSON.parse(await readFile(
      path.join(workspace, '.roy', 'execution-trees', 'terminal-capability-test', `${result.correlationId}.json`),
      'utf8'
    ))).toMatchObject({ correlationId: result.correlationId, status: 'completed' });

    await runtime.shutdown();
  });
});
