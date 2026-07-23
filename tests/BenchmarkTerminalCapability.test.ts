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

class DelegatedTerminalTaskLLM extends TerminalTaskLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('<root_execution_report>')
      ? 'Applied the delegated workspace change and verified delegated.txt.'
      : 'Delegated analysis identified the requested artifact but did not write it.';
    yield { content, done: true };
  }

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes("root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Inspect before applying the requested workspace change.',
        continuationPolicy: 'finalize_after_round',
        agents: [{
          archetype: 'researcher',
          name: 'ArtifactInspector-1',
          task: 'Inspect the workspace and report what is needed for delegated.txt without modifying files.',
          tools: ['fs.list', 'fs.read'],
          tomLevel: 0,
        }],
      } as T;
    }
    if (system.includes("delegation controller")) {
      return { action: 'solve_directly', reason: 'The child should inspect directly.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('[runtime_execution_phase]') && user.includes('Completed tool round: 0')) {
        return {
          action: 'call_tools',
          reason: 'Apply and verify the delegated workspace change.',
          calls: [{
            toolName: 'shell.exec',
            params: {
              command: "printf 'delegation-closed' > delegated.txt && test \"$(cat delegated.txt)\" = delegation-closed",
            },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'The requested change is applied and verified.', calls: [] } as T;
    }
    return {} as T;
  }
}

class FailedDelegationRecoveryLLM extends TerminalTaskLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('<root_execution_report>')
      ? 'Recovered the failed delegation, created failed-delegation.txt, and verified it.'
      : '<tool_call><tool_name>fs.read</tool_name><path>missing.txt</path></tool_call>';
    yield { content, done: true };
  }

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes("root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Try one delegated inspection before implementation.',
        continuationPolicy: 'finalize_after_round',
        agents: [{
          archetype: 'custom',
          name: 'FailingInspector',
          task: 'Return invalid unresolved tool markup without executing any tool.',
          tomLevel: 0,
        }],
      } as T;
    }
    if (system.includes("delegation controller")) {
      return { action: 'solve_directly', reason: 'The child should answer directly.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('[runtime_execution_phase]') && user.includes('Completed tool round: 0')) {
        return {
          action: 'call_tools',
          reason: 'Recover by applying and verifying the requested workspace change.',
          calls: [{
            toolName: 'shell.exec',
            params: {
              command: "printf 'recovered' > failed-delegation.txt && test \"$(cat failed-delegation.txt)\" = recovered",
            },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'No child tool call is required.', calls: [] } as T;
    }
    return {} as T;
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

  it('turns delegated analysis into a root workspace mutation and verification', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-delegated-terminal-task-'));
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
          maxCallsPerRun: 8,
          maxConsecutiveFailures: 2,
          maxWallClockMs: 30_000,
          maxFetchesAfterSearch: 2,
          llmReplanning: true,
        },
      },
    }, null, 2));

    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegated-terminal-capability-test',
      workspaceCwd: workspace,
      llmProvider: new DelegatedTerminalTaskLLM(),
    });
    const result = await runtime.handleUserTurn(
      'Implement a workspace change by creating delegated.txt, then verify the file.'
    );

    expect(result.decision.action).toBe('spawn_subagents');
    expect(await readFile(path.join(workspace, 'delegated.txt'), 'utf8')).toBe('delegation-closed');
    expect(result.finalResponse).toContain('Applied the delegated workspace change');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.required.completed',
      data: expect.objectContaining({
        mutationApplied: true,
        verificationRan: true,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.call',
      agentId: 'root',
      data: expect.objectContaining({ toolName: 'shell.exec' }),
    }));

    await runtime.shutdown();
  });

  it('recovers an initial failed delegation through root execution', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-failed-delegation-recovery-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 4,
          maxDelegationRounds: 2,
          reassessAfterDelegation: true,
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
          maxCallsPerRun: 8,
          maxConsecutiveFailures: 2,
          maxWallClockMs: 30_000,
          maxFetchesAfterSearch: 2,
          llmReplanning: true,
        },
      },
    }, null, 2));

    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'failed-delegation-recovery-test',
      workspaceCwd: workspace,
      llmProvider: new FailedDelegationRecoveryLLM(),
    });
    const result = await runtime.handleUserTurn(
      'Implement a workspace change by creating failed-delegation.txt, then verify the file.'
    );

    expect(await readFile(path.join(workspace, 'failed-delegation.txt'), 'utf8')).toBe('recovered');
    expect(result.executionTree.status).toBe('completed');
    expect(result.executionTree.steps.map(step => step.status)).toEqual([
      'failed',
      'completed',
      'completed',
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.step.recovered',
      data: expect.objectContaining({ recovery: 'root_execution_after_failed_delegation' }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.required.completed',
      data: expect.objectContaining({ mutationApplied: true, verificationRan: true }),
    }));

    await runtime.shutdown();
  });
});
