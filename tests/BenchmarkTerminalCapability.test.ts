import { describe, expect, it } from 'vitest';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { compactExecutionKnowledgeForPrompt } from '../src/core/runtime/executionCache.js';
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

class NoProgressDirectExecutionLLM extends TerminalTaskLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const system = messages.find(message => message.role === 'system')?.content ?? '';
    const user = messages.findLast(message => message.role === 'user')?.content ?? '';
    if (system.includes("root delegation controller")) {
      return { action: 'solve_directly', reason: 'Execute the workspace task directly.' } as T;
    }
    if (system.includes('plan authorized tool calls')) {
      if (user.includes('[runtime_execution_phase]')
        && user.includes('Completed tool round:')) {
        return {
          action: 'call_tools',
          reason: 'Apply one incomplete edit.',
          calls: [{
            toolName: 'fs.write',
            params: { path: 'artifact.txt', content: 'incomplete' },
          }],
        } as T;
      }
      return { action: 'finish', reason: 'No repair action can be selected.', calls: [] } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

describe('benchmark terminal capability', () => {
  it('treats an identical failed closure action as stalled rather than progress', () => {
    const runtime = new Runtime();
    const advanced = (runtime as unknown as {
      rootExecutionAttemptAdvanced: (
        current: {
          toolCalls: Array<{
            toolName: string;
            params: Record<string, unknown>;
            success: boolean;
            error?: string;
            result?: unknown;
          }>;
        },
        prior?: {
          toolCalls: Array<{
            toolName: string;
            params: Record<string, unknown>;
            success: boolean;
            error?: string;
            result?: unknown;
          }>;
        }
      ) => boolean;
    }).rootExecutionAttemptAdvanced.bind(runtime);
    const rejected = {
      toolName: 'fs.synthesize',
      params: { path: '.roy/official-verifier/test_outputs.py' },
      success: false,
      error: '.roy/official-verifier is immutable runtime evidence',
    };

    expect(advanced(
      { toolCalls: [rejected] },
      { toolCalls: [rejected] }
    )).toBe(false);
    expect(advanced(
      {
        toolCalls: [{
          ...rejected,
          error: 'requirements.txt still contains langchain==0.0.1',
        }],
      },
      { toolCalls: [rejected] }
    )).toBe(true);
    expect(advanced(
      { toolCalls: [{ ...rejected, success: true, error: undefined }] },
      { toolCalls: [rejected] }
    )).toBe(true);
  });

  it('requires verification at or after the latest successful mutation', () => {
    const runtime = new Runtime();
    const analyze = (runtime as unknown as {
      analyzeWorkspaceExecutionClosure: (calls: Array<{
        toolName: string;
        params: Record<string, unknown>;
        success: boolean;
        result?: unknown;
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
    expect(analyze([
      {
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'candidate' },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          exitCode: 0,
          stdout: '0.5\n',
          candidateRollback: {
            restored: true,
            path: 'artifact.txt',
            reason: 'no_objective_gain',
          },
        },
      },
    ])).toMatchObject({
      closed: false,
      verificationAttemptedAfterMutation: false,
      verificationPassed: false,
    });
  });

  it('resumes rejected repairs by causal recovery round and reuses fresh diagnostics', () => {
    const runtime = new Runtime();
    const now = Date.now();
    const buildRecovery = (runtime as unknown as {
      buildVerifierGuidedResumeRecoveryDecision: (
        task: string,
        state: Record<string, unknown>
      ) => {
        action: string;
        agents: Array<{ name?: string; role?: string; tools?: string[] }>;
        coordination?: string;
        team?: { name?: string; executionPolicy?: { mode?: string } };
      } | undefined;
    }).buildVerifierGuidedResumeRecoveryDecision.bind(runtime);
    const state = {
      sourceCorrelationId: 'prior-correlation',
      anchorPathId: 'prior-path',
      openPaths: 1,
      actionableFeedback: 1,
      knowledge: {
        version: 1,
        updatedAt: now,
        steps: [{
          id: 'prior-step.cache',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          index: 1,
          task: 'Repair implementation.py.',
          taskFingerprint: 'task',
          pathId: 'prior-path',
          dependsOn: [],
          action: 'solve_directly',
          status: 'failed',
          actorIds: [],
          teamIds: [],
          feedbackIds: [],
          createdAt: now,
          updatedAt: now,
        }],
        paths: [{
          id: 'prior-path',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          parentPathIds: [],
          taskFingerprint: 'task',
          status: 'partial',
          actorIds: [],
          teamIds: [],
          observedPaths: ['implementation.py'],
          invalidPaths: [],
          successfulTools: ['fs.synthesize'],
          failedTools: [],
          mutationObserved: true,
          verificationObserved: true,
          toolFrontier: [{
            toolName: 'shell.exec',
            params: { command: 'python .roy/official-verifier/grade.py' },
            success: true,
            result: {
              stdout: '0.4\n',
              candidateRollback: {
                restored: true,
                path: 'implementation.py',
                reason: 'reward_regression',
                baselineReward: 0.5,
                candidateReward: 0.4,
              },
            },
            startedAt: now - 100,
            completedAt: now,
          }],
          feedbackIds: [],
          createdAt: now,
          updatedAt: now,
        }],
        actors: [] as Array<Record<string, unknown>>,
        feedback: [],
      },
    };

    const recovery = buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      state
    );
    expect(recovery).toMatchObject({
      action: 'spawn_subagents',
      coordination: 'team',
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-1', role: 'verifier-guided diagnostic probe' }),
        expect.objectContaining({ name: 'FocusedRepairer-2', role: 'verifier-guided recovery executor' }),
        expect.objectContaining({ name: 'RecoveryVerifier-3', role: 'independent recovery verifier' }),
      ],
      team: expect.objectContaining({
        name: 'VerifierGuidedRecoveryTeam',
        memberDelegationPolicy: 'deny',
        executionPolicy: expect.objectContaining({ mode: 'sequential' }),
      }),
    });
    expect(recovery?.agents[0]?.task).toContain('[runtime_verifier_diagnostic_probe]');
    expect(recovery?.agents[0]?.task).toContain('<recovery_capsule>');
    expect(recovery?.agents[0]?.task).not.toContain('<resume_ledger>');
    expect(recovery?.agents[1]?.task).toContain(
      'python .roy/official-verifier/grade.py'
    );
    expect(recovery?.agents[2]?.task).toContain(
      'python .roy/official-verifier/grade.py'
    );
    expect(recovery?.agents.every(agent =>
      agent.memoryScope?.public === false
      && agent.memoryScope.private === false
      && agent.memoryScope.sessionWindowTurns === 0
      && !agent.skills?.includes('delegate_to_subagent')
    )).toBe(true);
    const failedAttempt = structuredClone(state);
    failedAttempt.knowledge.actors.push({
      id: 'actor',
      runtimeActorId: 'actor',
      kind: 'agent',
      correlationId: 'prior-correlation',
      stepId: 'prior-step',
      pathId: 'prior-path',
      name: 'VerifierProbe-1',
      role: 'verifier-guided diagnostic probe',
      generation: 1,
      status: 'failed',
      createdAt: now,
      updatedAt: now,
    });
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      failedAttempt
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-4' }),
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
      team: expect.objectContaining({ name: 'VerifierGuidedRecoveryTeam-2' }),
    });

    const cachedDiagnostic = structuredClone(failedAttempt);
    const diagnosticFrontier = cachedDiagnostic.knowledge.paths[0]!.toolFrontier;
    diagnosticFrontier.push({
      toolName: 'shell.exec',
      params: {
        command: 'ROY_VERIFIER_PROBE=1 python -c "print(1)"',
      },
      success: true,
      result: {
        stdout: 'VERIFIER_PROBE_EVIDENCE_VERSION 3\nVERIFIER_PROBE_RESULT 1.0\nVERIFIER_PROBE_ARTIFACT {"path":"outputs/layout_qc.json"}',
      },
      startedAt: now + 1,
      completedAt: now + 2,
    } as never);
    diagnosticFrontier.push({
      toolName: 'fs.synthesize',
      params: { path: 'implementation.py', strategy: 'patch' },
      success: true,
      result: { path: 'implementation.py', synthesized: true },
      startedAt: now + 3,
      completedAt: now + 4,
    } as never);
    diagnosticFrontier.push({
      toolName: 'shell.exec',
      params: { command: 'python .roy/official-verifier/grade.py' },
      success: true,
      result: {
        stdout: '0.4\n',
        candidateRollback: {
          restored: true,
          path: 'implementation.py',
          reason: 'reward_regression',
          baselineReward: 0.5,
          candidateReward: 0.4,
        },
      },
      startedAt: now + 5,
      completedAt: now + 6,
    } as never);
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      cachedDiagnostic
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-4' }),
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
      team: expect.objectContaining({ name: 'VerifierGuidedRecoveryTeam-2' }),
    });

    const freshDiagnostic = structuredClone(failedAttempt);
    freshDiagnostic.knowledge.paths[0]!.toolFrontier.push(
      diagnosticFrontier.at(-3)!
    );
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      freshDiagnostic
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
      team: expect.objectContaining({ name: 'VerifierGuidedRecoveryTeam-2' }),
    });

    const diagnosticThenInvalidPatch = structuredClone(failedAttempt);
    diagnosticThenInvalidPatch.knowledge.paths[0]!.toolFrontier.push(
      diagnosticFrontier.at(-3)!,
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: false,
        result: {
          path: 'implementation.py',
          synthesisRejected: true,
          reason: 'focused patch source anchor does not match',
        },
        startedAt: now + 3,
        completedAt: now + 4,
      } as never
    );
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      diagnosticThenInvalidPatch
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
      team: expect.objectContaining({ name: 'VerifierGuidedRecoveryTeam-2' }),
    });

    const completedRepairOnlyRound = structuredClone(failedAttempt);
    completedRepairOnlyRound.knowledge.actors.push({
      id: 'repair-actor',
      runtimeActorId: 'repair-actor',
      kind: 'agent',
      correlationId: 'repair-correlation',
      stepId: 'repair-step',
      pathId: 'repair-path',
      name: 'FocusedRepairer-5',
      role: 'coder',
      generation: 1,
      status: 'completed',
      createdAt: now + 10,
      updatedAt: now + 20,
    });
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      completedRepairOnlyRound
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-7' }),
        expect.objectContaining({ name: 'FocusedRepairer-8' }),
        expect.objectContaining({ name: 'RecoveryVerifier-9' }),
      ],
      team: expect.objectContaining({ name: 'VerifierGuidedRecoveryTeam-3' }),
    });

    const staleAfterAcceptedProgress = structuredClone(failedAttempt);
    const progressedFrontier = staleAfterAcceptedProgress.knowledge.paths[0]!.toolFrontier;
    progressedFrontier.push(
      diagnosticFrontier.at(-3)!,
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: true,
        result: { path: 'implementation.py', synthesized: true },
        startedAt: now + 3,
        completedAt: now + 4,
      } as never,
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: { stdout: '0.6\n' },
        startedAt: now + 5,
        completedAt: now + 6,
      } as never,
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: true,
        result: { path: 'implementation.py', synthesized: true },
        startedAt: now + 7,
        completedAt: now + 8,
      } as never,
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          stdout: '0.5\n',
          candidateRollback: {
            restored: true,
            path: 'implementation.py',
            reason: 'reward_regression',
            baselineReward: 0.6,
            candidateReward: 0.5,
          },
        },
        startedAt: now + 9,
        completedAt: now + 10,
      } as never
    );
    expect(buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      staleAfterAcceptedProgress
    )).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-4' }),
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
    });

    const obsoleteRejectionAfterAcceptedProgress = structuredClone(failedAttempt);
    obsoleteRejectionAfterAcceptedProgress.knowledge.paths[0]!.toolFrontier.push(
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: true,
        result: { path: 'implementation.py', synthesized: true },
        startedAt: now + 3,
        completedAt: now + 4,
      } as never,
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          stdout: '0.8\n',
          verifierDiagnostics: [{
            path: '/logs/verifier/scorecard.json',
            content: JSON.stringify({
              reward: 0.8,
              groups: { accepted_capability: 1, unresolved_capability: 0.5 },
            }),
          }],
        },
        startedAt: now + 5,
        completedAt: now + 6,
      } as never
    );
    const scorecardRecovery = buildRecovery(
      'Repair implementation.py until the official verifier succeeds.',
      obsoleteRejectionAfterAcceptedProgress
    );
    expect(scorecardRecovery).toMatchObject({
      agents: [
        expect.objectContaining({ name: 'VerifierProbe-4' }),
        expect.objectContaining({ name: 'FocusedRepairer-5' }),
        expect.objectContaining({ name: 'RecoveryVerifier-6' }),
      ],
    });
    expect(scorecardRecovery?.agents[0]?.task).toContain(
      '"recoveryTrigger": "incomplete_accepted_scorecard_after_progress"'
    );
    expect(scorecardRecovery?.agents[0]?.task).toContain(
      '"unresolvedGroups": [\n    "unresolved_capability=0.500"'
    );
    expect(scorecardRecovery?.agents[0]?.task).not.toContain('"rejectedCandidate"');
  });

  it('expires rejected synthesis paths after a newer mutation changes the hypothesis', () => {
    const runtime = new Runtime();
    const unresolved = (runtime as unknown as {
      hasUnresolvedSynthesisRejection: (
        calls: Array<Record<string, unknown>>
      ) => boolean;
    }).hasUnresolvedSynthesisRejection.bind(runtime);

    expect(unresolved([
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: false,
        result: { synthesisRejected: true },
      },
    ])).toBe(true);
    expect(unresolved([
      {
        toolName: 'fs.synthesize',
        params: { path: 'implementation.py', strategy: 'patch' },
        success: false,
        result: { synthesisRejected: true },
      },
      {
        toolName: 'fs.replace',
        params: {
          path: 'implementation.py',
          oldText: 'VALUE = 1',
          newText: 'VALUE = 2',
        },
        success: true,
        result: { path: 'implementation.py', replacements: 1 },
      },
    ])).toBe(false);
  });

  it('deduplicates inherited tool frontiers in prompt context', () => {
    const now = Date.now();
    const sharedCalls = [{
      toolName: 'fs.read',
      params: { path: 'implementation.py' },
      success: true,
      result: { path: 'implementation.py', content: 'VALUE = 1\n' },
      completedAt: now,
    }, {
      toolName: 'shell.exec',
      params: { command: 'python .roy/official-verifier/grade.py' },
      success: true,
      result: { stdout: '0.5\n' },
      completedAt: now + 1,
    }];
    const pathRecord = (id: string) => ({
      id,
      correlationId: 'correlation',
      stepId: `${id}.step`,
      parentPathIds: [],
      taskFingerprint: 'task',
      status: 'partial' as const,
      actorIds: [],
      teamIds: [],
      observedPaths: ['implementation.py'],
      invalidPaths: [],
      successfulTools: ['fs.read', 'shell.exec'],
      failedTools: [],
      mutationObserved: true,
      verificationObserved: true,
      toolFrontier: sharedCalls,
      feedbackIds: [],
      createdAt: now,
      updatedAt: now + 1,
    });
    const compacted = compactExecutionKnowledgeForPrompt({
      version: 1,
      updatedAt: now + 1,
      steps: [],
      paths: [pathRecord('path-a'), pathRecord('path-b')],
      actors: [],
      feedback: [],
    }) as {
      paths: Array<Record<string, unknown>>;
      causalToolFrontier: Array<Record<string, unknown>>;
    };

    expect(compacted.paths.every(item => item.toolFrontier === undefined)).toBe(true);
    expect(compacted.causalToolFrontier).toHaveLength(2);
  });

  it('retains semantic workspace anchors when bounding a long tool frontier', () => {
    const runtime = new Runtime();
    const bound = (runtime as unknown as {
      boundExecutionToolFrontier: (
        calls: Array<Record<string, unknown>>,
        maxCalls: number,
        maxSerializedChars: number
      ) => Array<Record<string, unknown>>;
    }).boundExecutionToolFrontier.bind(runtime);
    const calls: Array<Record<string, unknown>> = [
      {
        toolName: 'fs.read',
        params: { path: 'implementation.py' },
        success: true,
        result: { path: 'implementation.py', content: 'VALUE = 1\n' },
      },
      {
        toolName: 'fs.replace',
        params: {
          path: 'implementation.py',
          oldText: 'VALUE = 1',
          newText: 'VALUE = 2',
        },
        success: true,
        result: { path: 'implementation.py', replacements: 1 },
      },
      ...Array.from({ length: 60 }, (_, index) => ({
        toolName: 'shell.exec',
        params: { command: `printf diagnostic-${index}` },
        success: true,
        result: { stdout: `diagnostic-${index}` },
      })),
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          stdout: '0.8\n',
          verifierDiagnostics: [{
            path: '/logs/verifier/scorecard.json',
            content: '{"reward":0.8,"groups":{"remaining":0.5}}',
          }],
        },
      },
    ];

    const selected = bound(calls, 8, 100_000);
    expect(selected).toEqual(expect.arrayContaining([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'implementation.py' },
      }),
      expect.objectContaining({
        toolName: 'fs.replace',
        params: expect.objectContaining({ path: 'implementation.py' }),
      }),
      expect.objectContaining({
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
      }),
    ]));
  });

  it('omits opaque runtime-generated verifier commands from model evidence', () => {
    const runtime = new Runtime();
    const compact = (runtime as unknown as {
      compactShellCommandForEvidence: (command: string) => string;
    }).compactShellCommandForEvidence.bind(runtime);
    const opaquePayload = 'A'.repeat(20_000);
    const command = `ROY_VERIFIER_PROBE=1 python -c "import base64;exec(base64.b64decode('${opaquePayload}'))"`;
    const result = compact(command);

    expect(result).toContain('runtime-generated verifier probe omitted');
    expect(result).toContain(`chars=${command.length}`);
    expect(result).not.toContain(opaquePayload.slice(0, 200));
    expect(result.length).toBeLessThan(120);
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

  it('does not cache a valid file as an invalid path when only its requested line range is stale', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-stale-read-range-'));
    await writeFile(path.join(workspace, 'implementation.py'), 'line one\nline two\n');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'stale-read-range-test',
      workspaceCwd: workspace,
    });

    const staleRange = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'implementation.py', startLine: 40, endLine: 60 },
      { correlationId: 'stale-read-range-turn' }
    );
    const correctedRange = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'implementation.py', startLine: 1, endLine: 2 },
      { correlationId: 'stale-read-range-turn' }
    );

    expect(staleRange).toMatchObject({
      success: false,
      error: expect.stringContaining('exceeds'),
    });
    expect(correctedRange).toMatchObject({
      success: true,
      result: expect.objectContaining({ content: 'line one\nline two' }),
    });
    expect(runtime.getEvents().filter(event =>
      event.type === 'tool.path.cache_rejected'
      && event.correlationId === 'stale-read-range-turn'
    )).toHaveLength(0);
    expect(runtime.getEvents().filter(event =>
      event.type === 'tool.call'
      && event.correlationId === 'stale-read-range-turn'
      && event.data.toolName === 'fs.read'
    )).toHaveLength(2);

    await runtime.shutdown();
  });

  it('attaches standard verifier scorecards to an opaque numeric verification result', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-verifier-diagnostics-'));
    await mkdir(path.join(workspace, '.roy', 'official-verifier'), { recursive: true });
    await mkdir(path.join(workspace, 'logs', 'verifier'), { recursive: true });
    await writeFile(
      path.join(workspace, '.roy', 'official-verifier', 'grade.py'),
      'print("0.250000000000")\n'
    );
    await writeFile(
      path.join(workspace, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: { readOnly: 'auto', execute: 'auto' },
          shell: { mode: 'unrestricted', shell: '/bin/sh' },
        },
      })
    );
    await writeFile(
      path.join(workspace, 'logs', 'verifier', 'scorecard.json'),
      JSON.stringify({
        groups: { schema: 1, hidden_stress: 0 },
        reward: 0.25,
      })
    );
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'verifier-diagnostics-test',
      workspaceCwd: workspace,
    });

    const result = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command: 'python3 .roy/official-verifier/grade.py' },
      { correlationId: 'verifier-diagnostics-turn' }
    );

    expect(result).toMatchObject({
      success: true,
      result: expect.objectContaining({
        stdout: expect.stringContaining('0.250000000000'),
        verifierDiagnostics: expect.arrayContaining([
          expect.objectContaining({
            path: path.join(workspace, 'logs', 'verifier', 'scorecard.json'),
            content: expect.stringContaining('"hidden_stress":0'),
          }),
        ]),
      }),
    });
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.verifier_diagnostics.attached',
      data: expect.objectContaining({
        files: expect.arrayContaining([
          path.join(workspace, 'logs', 'verifier', 'scorecard.json'),
        ]),
      }),
    }));
    await runtime.shutdown();
  });

  it('protects the first resumed mutation with the persisted verifier baseline', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-resumed-verifier-baseline-'));
    await writeFile(path.join(workspace, 'implementation.py'), 'VALUE = 41\n');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'resumed-verifier-baseline-test',
      workspaceCwd: workspace,
    });
    const correlationId = 'resumed-verifier-baseline-turn';
    (runtime as unknown as {
      resumedExecutionByCorrelation: Map<string, unknown>;
    }).resumedExecutionByCorrelation.set(correlationId, {
      sourceCorrelationId: 'prior-turn',
      anchorPathId: 'prior-path',
      knowledge: {
        paths: [{
          toolFrontier: [{
            toolName: 'shell.exec',
            params: { command: 'python .roy/official-verifier/grade.py' },
            success: true,
            result: {
              verifierDiagnostics: [{
                path: '/logs/verifier/scorecard.json',
                content: JSON.stringify({
                  reward: 0.75,
                  groups: { public: 1, hidden: 0.5 },
                }),
              }],
            },
            startedAt: Date.now() - 10,
            completedAt: Date.now() - 5,
          }],
        }],
      },
    });

    const checkpoint = await (runtime as unknown as {
      captureWorkspaceMutationCheckpoint: (
        toolName: string,
        params: Record<string, unknown>,
        calls: Array<Record<string, unknown>>,
        correlationId: string
      ) => Promise<{
        baseline?: { reward: number; groups: Record<string, number> };
      } | undefined>;
    }).captureWorkspaceMutationCheckpoint(
      'fs.replace',
      { path: 'implementation.py' },
      [],
      correlationId
    );

    expect(checkpoint?.baseline).toEqual({
      reward: 0.75,
      groups: { public: 1, hidden: 0.5 },
    });
    await runtime.shutdown();
  });

  it('rolls back an existing source mutation when the verifier reward regresses', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-verifier-regression-'));
    await mkdir(path.join(workspace, '.roy', 'official-verifier'), { recursive: true });
    await mkdir(path.join(workspace, 'logs', 'verifier'), { recursive: true });
    await writeFile(path.join(workspace, 'implementation.py'), 'VALUE = 41\n');
    await writeFile(
      path.join(workspace, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: { readOnly: 'auto', write: 'auto', execute: 'auto' },
          shell: { mode: 'unrestricted', shell: '/bin/sh' },
        },
      })
    );
    const gradePath = path.join(workspace, '.roy', 'official-verifier', 'grade.py');
    const scorecardPath = path.join(workspace, 'logs', 'verifier', 'scorecard.json');
    await writeFile(gradePath, 'print("0.500000000000")\n');
    await writeFile(
      scorecardPath,
      JSON.stringify({ groups: { public: 1, hidden: 0 }, reward: 0.5 })
    );
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'verifier-regression-test',
      workspaceCwd: workspace,
    });

    const baseline = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command: 'python3 .roy/official-verifier/grade.py' },
      { correlationId: 'verifier-regression-turn' }
    );
    const baselineCall = {
      toolName: 'shell.exec',
      params: { command: 'python3 .roy/official-verifier/grade.py' },
      reason: 'Establish the verifier baseline.',
      groundingRequired: true,
      success: baseline.success,
      result: baseline.result,
      error: baseline.error,
    };
    const mutation = await runtime.executeToolForAgent(
      'root',
      'fs.replace',
      {
        path: 'implementation.py',
        oldText: 'VALUE = 41',
        newText: 'VALUE = 0',
        expectedReplacements: 1,
      },
      {
        correlationId: 'verifier-regression-turn',
        groundingCalls: [baselineCall],
      }
    );
    expect(await readFile(path.join(workspace, 'implementation.py'), 'utf8')).toBe(
      'VALUE = 0\n'
    );
    const followUpMutation = await runtime.executeToolForAgent(
      'root',
      'fs.replace',
      {
        path: 'implementation.py',
        oldText: 'VALUE = 0',
        newText: 'VALUE = -1',
        expectedReplacements: 1,
      },
      {
        correlationId: 'verifier-regression-turn',
        groundingCalls: [
          baselineCall,
          {
            toolName: 'fs.replace',
            params: { path: 'implementation.py' },
            reason: 'Apply the first unverified candidate slice.',
            groundingRequired: true,
            success: mutation.success,
            result: mutation.result,
            error: mutation.error,
          },
        ],
      }
    );
    expect(followUpMutation.success).toBe(true);
    expect(await readFile(path.join(workspace, 'implementation.py'), 'utf8')).toBe(
      'VALUE = -1\n'
    );
    await writeFile(gradePath, 'print("0.100000000000")\n');
    await writeFile(
      scorecardPath,
      JSON.stringify({ groups: { public: 0, hidden: 0.1 }, reward: 0.1 })
    );

    const regressed = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command: 'python3 .roy/official-verifier/grade.py' },
      {
        correlationId: 'verifier-regression-turn',
        groundingCalls: [
          baselineCall,
          {
            toolName: 'fs.replace',
            params: { path: 'implementation.py' },
            reason: 'Apply a candidate repair.',
            groundingRequired: true,
            success: mutation.success,
            result: mutation.result,
            error: mutation.error,
          },
        ],
      }
    );

    expect(regressed).toMatchObject({
      success: true,
      result: expect.objectContaining({
        regressionRollback: expect.objectContaining({
          restored: true,
          path: 'implementation.py',
          baselineReward: 0.5,
          regressedReward: 0.1,
          baselineGroups: { public: 1, hidden: 0 },
          candidateGroups: { public: 0, hidden: 0.1 },
        }),
      }),
    });
    const restoredScorecard = (runtime as unknown as {
      verifierScorecardFromToolResult: (
        result: unknown
      ) => { reward: number; groups: Record<string, number> } | undefined;
    }).verifierScorecardFromToolResult(regressed.result);
    expect(restoredScorecard).toEqual({
      reward: 0.5,
      groups: { public: 1, hidden: 0 },
    });
    expect(await readFile(path.join(workspace, 'implementation.py'), 'utf8')).toBe(
      'VALUE = 41\n'
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'workspace.mutation.regression_rolled_back',
      data: expect.objectContaining({
        path: 'implementation.py',
        baselineReward: 0.5,
        regressedReward: 0.1,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'workspace.mutation.checkpoint.retained',
      data: expect.objectContaining({
        path: 'implementation.py',
        reason: 'preserve_accepted_snapshot_across_unverified_mutation_chain',
      }),
    }));

    await writeFile(gradePath, 'print("0.500000000000")\n');
    await writeFile(
      scorecardPath,
      JSON.stringify({ groups: { public: 1, hidden: 0 }, reward: 0.5 })
    );
    const equalBaseline = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command: 'python3 .roy/official-verifier/grade.py' },
      { correlationId: 'verifier-no-gain-turn' }
    );
    const equalBaselineCall = {
      toolName: 'shell.exec',
      params: { command: 'python3 .roy/official-verifier/grade.py' },
      reason: 'Establish the equal-score verifier baseline.',
      groundingRequired: true,
      success: equalBaseline.success,
      result: equalBaseline.result,
      error: equalBaseline.error,
    };
    const equalMutation = await runtime.executeToolForAgent(
      'root',
      'fs.replace',
      {
        path: 'implementation.py',
        oldText: 'VALUE = 41',
        newText: 'VALUE = 42',
        expectedReplacements: 1,
      },
      {
        correlationId: 'verifier-no-gain-turn',
        groundingCalls: [equalBaselineCall],
      }
    );
    const noGain = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command: 'python3 .roy/official-verifier/grade.py' },
      {
        correlationId: 'verifier-no-gain-turn',
        groundingCalls: [
          equalBaselineCall,
          {
            toolName: 'fs.replace',
            params: { path: 'implementation.py' },
            reason: 'Apply an equal-score candidate.',
            groundingRequired: true,
            success: equalMutation.success,
            result: equalMutation.result,
            error: equalMutation.error,
          },
        ],
      }
    );
    expect(noGain).toMatchObject({
      success: true,
      result: expect.objectContaining({
        candidateRollback: expect.objectContaining({
          restored: true,
          path: 'implementation.py',
          reason: 'no_objective_gain',
          baselineReward: 0.5,
          candidateReward: 0.5,
        }),
      }),
    });
    expect((noGain.result as { regressionRollback?: unknown }).regressionRollback)
      .toBeUndefined();
    expect(await readFile(path.join(workspace, 'implementation.py'), 'utf8')).toBe(
      'VALUE = 41\n'
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'workspace.mutation.candidate_rolled_back',
      data: expect.objectContaining({
        reason: 'no_objective_gain',
        baselineReward: 0.5,
        candidateReward: 0.5,
      }),
    }));
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
      toolFrontier: expect.arrayContaining([
        expect.objectContaining({
          toolName: 'shell.exec',
          success: true,
        }),
      ]),
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
        }, {
          id: 'ancestor-step.cache',
          correlationId: 'ancestor-correlation',
          stepId: 'ancestor-step',
          index: 1,
          task,
          taskFingerprint: 'ancestor-task',
          pathId: 'ancestor-path',
          dependsOn: [],
          action: 'delegate',
          status: 'failed',
          actorIds: [],
          teamIds: [],
          feedbackIds: [],
          resultSummary: 'The workspace root was already listed.',
          createdAt: now - 3000,
          updatedAt: now - 2000,
        }],
        paths: [{
          id: 'prior-path',
          correlationId: 'prior-correlation',
          stepId: 'prior-step',
          parentPathIds: ['ancestor-path'],
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
          toolFrontier: [{
            toolName: 'fs.list',
            params: { path: '.', maxDepth: 2 },
            result: { root: '.', entries: ['.roy/config.json'] },
            success: true,
            startedAt: now - 2900,
            completedAt: now - 2800,
          }, {
            toolName: 'fs.list',
            params: { path: '.', maxDepth: 2 },
            result: { root: '.', entries: ['.roy/config.json', 'task-note.txt'] },
            success: true,
            startedAt: now - 1900,
            completedAt: now - 1800,
          }],
          feedbackIds: ['prior-feedback'],
          summary: 'artifact.txt is missing',
          createdAt: now - 1000,
          updatedAt: now,
        }, {
          id: 'ancestor-path',
          correlationId: 'ancestor-correlation',
          stepId: 'ancestor-step',
          parentPathIds: [],
          taskFingerprint: 'ancestor-task',
          status: 'partial',
          actorIds: [],
          teamIds: [],
          observedPaths: ['.roy/config.json'],
          invalidPaths: [],
          successfulTools: ['fs.list'],
          failedTools: [],
          mutationObserved: false,
          verificationObserved: false,
          toolFrontier: [{
            toolName: 'fs.list',
            params: { path: '.', maxDepth: 2 },
            result: { root: '.', entries: ['.roy/config.json'] },
            success: true,
            startedAt: now - 2900,
            completedAt: now - 2800,
          }],
          feedbackIds: [],
          summary: 'Workspace root listing is authoritative.',
          createdAt: now - 3000,
          updatedAt: now - 2000,
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
        paths: 2,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'execution.tool_frontier.resumed',
      data: expect.objectContaining({
        sourceCorrelationId: 'prior-correlation',
        paths: 2,
        toolCalls: 2,
      }),
    }));
    expect(result.executionTree.steps[0].cache?.path.parentPathIds).toContain('prior-path');
    expect(result.executionTree.steps[0].cache?.path.toolFrontier).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          toolName: 'fs.list',
          success: true,
        }),
      ])
    );
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

  it('reuses a delegated mutation and fresh verification without making root repeat the work', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-delegated-closed-reuse-'));
    await writeFile(path.join(workspace, 'delegated-global.txt'), 'delegated', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegated-closed-reuse-test',
      workspaceCwd: workspace,
      llmProvider: new TerminalTaskLLM(),
    });
    const now = Date.now();
    const delegatedResults = [{
      node: { identity: { archetype: 'coder' } },
      agent: { identity: { name: 'DelegatedWriter' } },
      subagentResult: {
        toolCalls: [
          {
            toolName: 'shell.exec',
            params: { command: "printf 'delegated' > delegated-global.txt" },
            result: { exitCode: 0, stdout: '', stderr: '' },
            success: true,
            startedAt: now - 40,
            completedAt: now - 30,
          },
          {
            toolName: 'shell.exec',
            params: { command: 'test "$(cat delegated-global.txt)" = delegated' },
            result: { exitCode: 0, stdout: '', stderr: '' },
            success: true,
            startedAt: now - 20,
            completedAt: now - 10,
          },
        ],
        grounded: true,
        warnings: [],
        context: '',
        evidence: {
          toolGrounded: true,
          outputGrounded: true,
          observedPaths: ['delegated-global.txt'],
          toolResultSummary: 'DelegatedWriter created and freshly verified delegated-global.txt.',
        },
        toolLoop: {
          rounds: [],
          totalCalls: 2,
          successfulCalls: 2,
          failedCalls: 0,
          stopReason: 'completed',
          startedAt: now - 40,
          completedAt: now - 10,
        },
        result: 'Created and verified delegated-global.txt.',
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        agent: { identity: { name: 'DelegatedWriter' } },
      },
    }];
    const execute = (runtime as unknown as {
      runRequiredRootExecution: (
        task: string,
        subagents: unknown[],
        teams: unknown[],
        correlationId: string
      ) => Promise<{ toolCalls: Array<{ toolName: string; success: boolean }> }>;
    }).runRequiredRootExecution.bind(runtime);

    const execution = await execute(
      'Implement and verify delegated-global.txt in the workspace.',
      delegatedResults,
      [],
      'delegated-closed-reuse-correlation'
    );

    expect(execution.toolCalls).toHaveLength(2);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.delegated.closure.reused',
      data: expect.objectContaining({
        mutationApplied: true,
        verificationPassed: true,
        auditRequired: false,
      }),
    }));
    expect(runtime.getEvents()).not.toContainEqual(expect.objectContaining({
      type: 'tool.call',
      agentId: 'root',
    }));
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
      type: 'root.execution.progress_horizon.extended',
      data: expect.objectContaining({
        attempt: 1,
        reason: expect.stringContaining('successful workspace mutation'),
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

  it('stops repeating root closure attempts after a fully corrected planner returns no action', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-direct-execution-no-progress-'));
    await mkdir(path.join(workspace, '.roy'), { recursive: true });
    await writeFile(path.join(workspace, '.roy', 'config.json'), JSON.stringify({
      delegation: {
        rootSteps: {
          maxExecutionClosureAttempts: 8,
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
      sessionId: 'direct-execution-no-progress-test',
      workspaceCwd: workspace,
      llmProvider: new NoProgressDirectExecutionLLM(),
    });
    await runtime.handleUserTurn(
      'Implement the workspace change in artifact.txt and run a verification.'
    );

    const attempts = runtime.getEvents().filter(event =>
      event.type === 'root.execution.attempt.completed'
    );
    expect(attempts).toHaveLength(3);
    const stalls = runtime.getEvents().filter(event =>
      event.type === 'root.execution.no_progress.detected'
    );
    expect(stalls).toHaveLength(1);
    expect(stalls.at(-1)?.data).toMatchObject({
      stalledIterations: 1,
      maxStalledIterations: 2,
      stateUnchanged: true,
      reason: expect.stringContaining('unchanged causal frontier'),
    });

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
