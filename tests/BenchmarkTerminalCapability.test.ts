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
  rootDecisionPrompts: string[] = [];

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'Created artifact.txt and verified its contents.' };
  }

  async *stream(): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'Created artifact.txt and verified its contents.', done: true };
  }

  async completeJSON<T>(messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes('final acceptance auditor')) {
      return {
        items: Array.from({ length: 10 }, (_, index) => ({
          id: `acceptance_${String(index + 1).padStart(2, '0')}`,
          status: 'verified',
          evidence: 'The final read-only audit observed the artifact and a passing executable check.',
        })),
        reason: 'All supplied acceptance items have direct audit evidence.',
      } as T;
    }
    if (system.includes("root delegation controller")) {
      this.rootDecisionPrompts.push(user);
      return { action: 'solve_directly', reason: 'The root has the required terminal capability.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('[runtime_acceptance_audit_phase]') && user.includes('Completed tool round:')) {
        const file = user.includes('failed-delegation.txt')
          ? 'failed-delegation.txt'
          : user.includes('delegated.txt')
            ? 'delegated.txt'
            : 'artifact.txt';
        return {
          action: 'call_tools',
          reason: 'Verify the final artifact without changing it.',
          calls: [{
            toolName: 'shell.exec',
            params: { command: `test -f ${file} && test -s ${file}` },
          }],
        } as T;
      }
      if (user.includes('Completed tool round:')) {
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
    if (system.includes('final acceptance auditor') || user.includes('[runtime_acceptance_audit_phase]')) {
      return super.completeJSON<T>(messages);
    }
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
      if (user.includes('[runtime_execution_phase]') && user.includes('Completed tool round:')) {
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
    if (system.includes('final acceptance auditor') || user.includes('[runtime_acceptance_audit_phase]')) {
      return super.completeJSON<T>(messages);
    }
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
      if (user.includes('[runtime_execution_phase]') && user.includes('Completed tool round:')) {
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

class DelegatedMutationRootVerificationLLM extends TerminalTaskLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('<root_execution_report>')
      ? 'Accepted the delegated mutation after an independent root verification.'
      : 'Created delegated-global.txt for root verification.';
    yield { content, done: true };
  }

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes("root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Delegate the implementation, then have root verify the resulting global state.',
        continuationPolicy: 'finalize_after_round',
        agents: [{
          archetype: 'coder',
          name: 'DelegatedWriter',
          task: 'Create delegated-global.txt in the workspace with the exact content delegated.',
          tools: ['fs.list', 'shell.exec'],
          tomLevel: 0,
        }],
      } as T;
    }
    if (system.includes("delegation controller")) {
      return { action: 'solve_directly', reason: 'The child should implement its assigned change.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('Completed tool round:')
        && !user.includes("test \"$(cat delegated-global.txt)\" = delegated")) {
        return {
          action: 'call_tools',
          reason: 'Independently verify the delegated workspace mutation.',
          calls: [{
            toolName: 'shell.exec',
            params: {
              command: 'test -f delegated-global.txt && test "$(cat delegated-global.txt)" = delegated',
            },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'Root verification passed.', calls: [] } as T;
    }
    return {} as T;
  }
}

class RetryingDirectExecutionLLM extends TerminalTaskLLM {
  override async *stream(): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'Recovered the incomplete execution and verified artifact.txt.', done: true };
  }

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes('final acceptance auditor') || user.includes('[runtime_acceptance_audit_phase]')) {
      return super.completeJSON<T>(messages);
    }
    if (system.includes("root delegation controller")) {
      return { action: 'solve_directly', reason: 'Execute the bounded workspace task directly.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('[runtime_execution_phase]') && user.includes('Completed tool round:')) {
        return {
          action: 'call_tools',
          reason: 'Apply an initial incomplete edit.',
          calls: [{
            toolName: 'fs.write',
            params: { path: 'artifact.txt', content: 'incomplete' },
          }],
        } as T;
      }
      if (user.includes('[runtime_execution_repair_phase]') && user.includes('Completed tool round:')) {
        return {
          action: 'call_tools',
          reason: 'Repair and verify the incomplete edit.',
          calls: [{
            toolName: 'shell.exec',
            params: {
              command: "printf 'repaired' > artifact.txt && test \"$(cat artifact.txt)\" = repaired",
            },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'No additional call selected in this attempt.', calls: [] } as T;
    }
    return {} as T;
  }
}

describe('benchmark terminal capability', () => {
  it('requires verification at or after the latest successful mutation', () => {
    const runtime = new Runtime();
    const analyze = (runtime as unknown as {
      analyzeWorkspaceExecutionClosure: (calls: Array<{
        toolName: string;
        params: Record<string, unknown>;
        success: boolean;
      }>) => {
        closed: boolean;
        verificationAttemptedAfterMutation: boolean;
        verificationPassed: boolean;
      };
    }).analyzeWorkspaceExecutionClosure.bind(runtime);

    expect(analyze([
      { toolName: 'shell.exec', params: { command: 'npm test' }, success: true },
      { toolName: 'fs.write', params: { path: 'artifact.txt', content: 'changed' }, success: true },
    ])).toMatchObject({
      closed: false,
      verificationAttemptedAfterMutation: false,
      verificationPassed: false,
    });
    expect(analyze([
      { toolName: 'fs.write', params: { path: 'artifact.txt', content: 'changed' }, success: true },
      { toolName: 'shell.exec', params: { command: 'npm test' }, success: false },
    ])).toMatchObject({
      closed: false,
      verificationAttemptedAfterMutation: true,
      verificationPassed: false,
    });
    expect(analyze([
      {
        toolName: 'shell.exec',
        params: { command: "printf 'fixed' > artifact.txt && npm test" },
        success: true,
      },
    ])).toMatchObject({
      closed: true,
      verificationAttemptedAfterMutation: true,
      verificationPassed: true,
    });
  });

  it('reuses persisted invalid-path knowledge in a later correlation', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-persisted-path-cache-'));
    const cacheDirectory = path.join(workspace, '.roy', 'cache');
    await mkdir(cacheDirectory, { recursive: true });
    const now = Date.now();
    await writeFile(path.join(cacheDirectory, 'execution-knowledge.json'), JSON.stringify({
      version: 1,
      updatedAt: now,
      steps: [{
        id: 'prior.step.cache',
        correlationId: 'prior-correlation',
        stepId: 'prior.step',
        index: 1,
        task: 'Inspect the workspace.',
        taskFingerprint: 'prior-task',
        pathId: 'prior.step.path',
        dependsOn: [],
        action: 'delegate',
        status: 'completed',
        actorIds: [],
        teamIds: [],
        feedbackIds: [],
        createdAt: now,
        updatedAt: now,
      }],
      paths: [{
        id: 'prior.step.path',
        correlationId: 'prior-correlation',
        stepId: 'prior.step',
        parentPathIds: [],
        taskFingerprint: 'prior-task',
        status: 'partial',
        actorIds: [],
        teamIds: [],
        observedPaths: ['src/actual.txt'],
        invalidPaths: ['missing.txt'],
        successfulTools: ['fs.list'],
        failedTools: ['fs.read'],
        mutationObserved: false,
        verificationObserved: false,
        feedbackIds: [],
        createdAt: now,
        updatedAt: now,
      }],
      actors: [],
      feedback: [],
    }, null, 2));

    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'persisted-path-cache-test',
      workspaceCwd: workspace,
    });
    const rejected = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'missing.txt' },
      { correlationId: 'later-correlation' }
    );

    expect(rejected).toMatchObject({
      success: false,
      metadata: { cacheRejected: true, path: 'missing.txt' },
    });
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.path.cache_rejected',
      correlationId: 'later-correlation',
      data: expect.objectContaining({ source: 'persisted execution knowledge' }),
    }));
    expect(runtime.getEvents().filter(event =>
      event.type === 'tool.call' && event.correlationId === 'later-correlation'
    )).toHaveLength(0);
    await runtime.shutdown();
  });

  it('rejects repeated reads of a cached invalid path until the hypothesis changes', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-invalid-path-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'invalid-path-cache-test',
      workspaceCwd: workspace,
    });

    const first = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: path.join(workspace, 'missing.txt') },
      { correlationId: 'invalid-path-turn' }
    );
    const repeated = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'missing.txt' },
      { correlationId: 'invalid-path-turn' }
    );

    expect(first.success).toBe(false);
    expect(repeated).toMatchObject({
      success: false,
      metadata: {
        cacheRejected: true,
        path: 'missing.txt',
      },
    });
    expect(runtime.getEvents().filter(event =>
      event.type === 'tool.call'
      && event.correlationId === 'invalid-path-turn'
      && event.data.toolName === 'fs.read'
    )).toHaveLength(1);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.path.cache_rejected',
      correlationId: 'invalid-path-turn',
      data: expect.objectContaining({ path: 'missing.txt' }),
    }));

    await writeFile(path.join(workspace, 'missing.txt'), 'created after the failed observation');
    const retriedAfterChange = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'missing.txt' },
      {
        correlationId: 'invalid-path-turn',
        reason: 'Retry after mutation: path created by the current execution.',
      }
    );
    expect(retriedAfterChange).toMatchObject({ success: true });

    const missingDirectory = await runtime.executeToolForAgent(
      'root',
      'fs.list',
      { path: 'outputs' },
      { correlationId: 'invalid-path-turn' }
    );
    expect(missingDirectory.success).toBe(false);
    await mkdir(path.join(workspace, 'outputs'));
    (runtime as unknown as {
      recordToolPathOutcome: (
        agentId: string,
        toolName: string,
        params: Record<string, unknown>,
        result: { success: boolean; result?: Record<string, unknown> },
        correlationId: string
      ) => void;
    }).recordToolPathOutcome(
      'root',
      'shell.exec',
      { command: `mkdir -p ${path.join(workspace, 'outputs')}` },
      { success: true, result: {} },
      'invalid-path-turn'
    );
    const retriedAfterShellMutation = await runtime.executeToolForAgent(
      'root',
      'fs.list',
      { path: 'outputs' },
      { correlationId: 'invalid-path-turn' }
    );
    expect(retriedAfterShellMutation).toMatchObject({ success: true });

    await runtime.shutdown();
  });

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
    const llm = new TerminalTaskLLM();
    await runtime.initialize({
      sessionId: 'terminal-capability-test',
      workspaceCwd: workspace,
      llmProvider: llm,
    });
    const result = await runtime.handleUserTurn(
      'Use the terminal in this workspace to create artifact.txt, verify it, and report completion.'
    );

    expect(llm.rootDecisionPrompts[0]).toContain('<acceptance_checklist>');
    expect(llm.rootDecisionPrompts[0]).toContain('"status": "unverified"');
    expect(await readFile(path.join(workspace, 'artifact.txt'), 'utf8')).toBe('roy-terminal-ready');
    expect(result.finalResponse).toContain('Created artifact.txt');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.call',
      data: expect.objectContaining({ toolName: 'shell.exec' }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.required.completed',
      data: expect.objectContaining({
        source: 'solve_directly',
        mutationApplied: true,
        verificationRan: true,
      }),
    }));
    expect(result.executionTree.status).toBe('completed');
    expect(JSON.parse(await readFile(
      path.join(workspace, '.roy', 'execution-trees', 'terminal-capability-test', `${result.correlationId}.json`),
      'utf8'
    ))).toMatchObject({ correlationId: result.correlationId, status: 'completed' });
    const executionKnowledge = JSON.parse(
      await readFile(path.join(workspace, '.roy', 'cache', 'execution-knowledge.json'), 'utf8')
    );
    expect(executionKnowledge.paths).toContainEqual(expect.objectContaining({
      mutationObserved: true,
      verificationObserved: true,
      successfulTools: expect.arrayContaining(['shell.exec']),
    }));
    expect(executionKnowledge.feedback).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: 'workspace_mutation' }),
      expect.objectContaining({ kind: 'workspace_verification' }),
    ]));

    await runtime.handleUserTurn(
      'Summarize the cached execution state for the previously created artifact.txt.'
    );
    expect(llm.rootDecisionPrompts.at(-1)).toContain('<execution_knowledge>');
    expect(llm.rootDecisionPrompts.at(-1)).toContain('"mutationObserved": true');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'execution.cache.hit',
      data: expect.objectContaining({ scope: 'root.delegation' }),
    }));

    await runtime.shutdown();
  });

  it('resumes an open persisted execution path without rebuilding the initial team', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-terminal-resume-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
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
    const llm = new TerminalTaskLLM();
    const task = [
      'This is a long-horizon terminal benchmark task.',
      'Use the terminal to create artifact.txt, verify it, and continue until the task is complete.',
      '<official_verifier_feedback>Previous verifier failure: artifact.txt is missing.</official_verifier_feedback>',
    ].join('\n');
    await runtime.initialize({
      sessionId: 'terminal-resume-test',
      workspaceCwd: workspace,
      llmProvider: llm,
    });
    const now = Date.now();
    await writeFile(
      path.join(workspace, '.roy', 'cache', 'execution-knowledge.json'),
      JSON.stringify({
        version: 1,
        updatedAt: now,
        steps: [{
          id: 'prior-step.cache',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          index: 1,
          task,
          taskFingerprint: 'prior-task',
          pathId: 'prior-path',
          dependsOn: [],
          action: 'delegate',
          status: 'failed',
          actorIds: ['prior-agent'],
          teamIds: ['prior-team'],
          feedbackIds: ['prior-feedback'],
          resultSummary: 'The prior team did not create artifact.txt.',
          createdAt: now - 1000,
          updatedAt: now,
        }],
        paths: [{
          id: 'prior-path',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          parentPathIds: [],
          taskFingerprint: 'prior-task',
          status: 'partial',
          actorIds: ['prior-agent'],
          teamIds: ['prior-team'],
          observedPaths: [],
          invalidPaths: ['artifact.txt'],
          successfulTools: ['fs.list'],
          failedTools: ['fs.read'],
          mutationObserved: false,
          verificationObserved: false,
          feedbackIds: ['prior-feedback'],
          summary: 'artifact.txt is missing',
          createdAt: now - 1000,
          updatedAt: now,
        }],
        actors: [],
        feedback: [{
          id: 'prior-feedback',
          kind: 'external_feedback',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          pathId: 'prior-path',
          path: 'artifact.txt',
          summary: 'Official verifier reports artifact.txt is missing.',
          actionable: true,
          createdAt: now,
        }],
      }, null, 2)
    );

    const result = await runtime.handleUserTurn(task);

    expect(llm.rootDecisionPrompts).toHaveLength(0);
    expect(await readFile(path.join(workspace, 'artifact.txt'), 'utf8')).toBe('roy-terminal-ready');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.task_loop.resumed',
      data: expect.objectContaining({
        sourceCorrelationId: 'prior-correlation',
        anchorPathId: 'prior-path',
      }),
    }));
    expect(result.executionTree.steps[0].cache?.path.parentPathIds).toContain('prior-path');
    expect(result.decision).toMatchObject({ action: 'solve_directly' });

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

  it('closes execution from a delegated mutation followed by root verification', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-delegated-global-closure-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, 'delegated-global.txt'), 'delegated');
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
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
      sessionId: 'delegated-global-closure-test',
      workspaceCwd: workspace,
      llmProvider: new DelegatedMutationRootVerificationLLM(),
    });
    const now = Date.now();
    const delegatedResults = [{
      node: { identity: { archetype: 'coder' } },
      agent: { identity: { name: 'DelegatedWriter' } },
      subagentResult: {
        toolCalls: [{
          toolName: 'shell.exec',
          params: { command: "printf 'delegated' > delegated-global.txt" },
          result: { exitCode: 0, stdout: '', stderr: '' },
          success: true,
          startedAt: now - 20,
          completedAt: now - 10,
        }],
        grounded: true,
        warnings: [],
        context: '',
        evidence: {
          toolGrounded: true,
          outputGrounded: true,
          observedPaths: ['delegated-global.txt'],
          toolResultSummary: 'DelegatedWriter wrote delegated-global.txt.',
        },
        toolLoop: {
          rounds: [],
          totalCalls: 1,
          successfulCalls: 1,
          failedCalls: 0,
          stopReason: 'completed',
          startedAt: now - 20,
          completedAt: now - 10,
        },
        result: 'Created delegated-global.txt.',
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        agent: { identity: { name: 'DelegatedWriter' } },
      },
    }];
    const runRequiredRootExecution = (runtime as unknown as {
      runRequiredRootExecution: (
        task: string,
        subagents: unknown[],
        teams: unknown[],
        correlationId: string
      ) => Promise<{
        toolCalls: Array<{
          toolName: string;
          params: Record<string, unknown>;
          success: boolean;
        }>;
      }>;
    }).runRequiredRootExecution.bind(runtime);
    const execution = await runRequiredRootExecution(
      'Implement and verify delegated-global.txt in the workspace with the exact content delegated.',
      delegatedResults,
      [],
      'delegated-global-closure-correlation'
    );
    const analyze = (runtime as unknown as {
      analyzeWorkspaceExecutionClosure: (calls: Array<{
        toolName: string;
        params: Record<string, unknown>;
        success: boolean;
      }>) => {
        closed: boolean;
        mutationApplied: boolean;
        verificationPassed: boolean;
        lastMutationCallIndex: number;
        lastVerificationCallIndex: number;
      };
    }).analyzeWorkspaceExecutionClosure.bind(runtime);

    expect(analyze(execution.toolCalls)).toMatchObject({
      closed: true,
      mutationApplied: true,
      verificationPassed: true,
      lastMutationCallIndex: 0,
    });
    const rootShellCommands = runtime.getEvents()
      .filter(event =>
        event.type === 'tool.call'
        && event.agentId === 'root'
        && event.data?.toolName === 'shell.exec'
      )
      .map(event => String((event.data?.params as { command?: unknown } | undefined)?.command ?? ''));
    expect(rootShellCommands).toContain(
      'test -f delegated-global.txt && test "$(cat delegated-global.txt)" = delegated'
    );
    expect(rootShellCommands.some(command => command.includes('> delegated-global.txt'))).toBe(false);

    await runtime.shutdown();
  });

  it('re-enters root execution when the first direct attempt is not verified', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-direct-execution-retry-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
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
      sessionId: 'direct-execution-retry-test',
      workspaceCwd: workspace,
      llmProvider: new RetryingDirectExecutionLLM(),
    });
    await runtime.handleUserTurn(
      'Implement the workspace change in artifact.txt and run a verification.'
    );

    expect(await readFile(path.join(workspace, 'artifact.txt'), 'utf8')).toBe('repaired');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.attempt.completed',
      data: expect.objectContaining({
        attempt: 1,
        mutationApplied: true,
        verificationRan: false,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.attempt.completed',
      data: expect.objectContaining({
        attempt: 2,
        mutationApplied: true,
        verificationRan: true,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.required.completed',
      data: expect.objectContaining({
        source: 'solve_directly',
        mutationApplied: true,
        verificationRan: true,
      }),
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
