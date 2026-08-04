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

class AutonomousStructureLLM implements LLMProvider {
  readonly name = 'autonomous-structure-test';
  readonly defaultModel = 'test-model';
  readonly rootDelegationPrompts: string[] = [];
  private childDelegationUsed = false;

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('formal subteam actor')
      ? 'The ReleaseReadinessCell reconciled evidence coverage and operational risk.'
      : text.includes('Synthesize their results into one final user-facing response')
        ? 'Roy synthesized the autonomously selected runtime structure.'
        : text.includes('EvidenceMapper-1')
          ? 'EvidenceMapper-1 reported bounded project evidence after child verification.'
          : text.includes('AssumptionBreaker-1')
            ? 'AssumptionBreaker-1 reported concrete operational failure modes.'
            : text.includes('EvidenceVerifier-1')
              ? 'EvidenceVerifier-1 verified the evidence boundary.'
              : text.includes('DecisionComposer-1')
                ? 'DecisionComposer-1 converted prior findings into release criteria.'
                : 'Roy answered directly from the accumulated session context.';
    yield { content, done: true, usage: { promptTokens: 18, completionTokens: 8, totalTokens: 26 } };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes("Roy's root delegation controller")) this.rootDelegationPrompts.push(text);
    if (text.includes('delegation candidate evaluator')) {
      const input = JSON.parse(messages[1].content) as { candidates: Array<{ id: string }> };
      return {
        scores: input.candidates.map(candidate => ({
          candidateId: candidate.id,
          score: candidate.id === 'candidate_full_plan' ? 1 : 0.1,
        })),
      } as T;
    }
    if (text.includes("Roy's root delegation controller") && text.includes('<user_task>THIRD_AUTONOMOUS_TURN')) {
      return { action: 'solve_directly', reason: 'The prior turns already established the required structure and evidence.' } satisfies DelegationDecision as T;
    }
    if (text.includes("Roy's root delegation controller") && text.includes('<user_task>SECOND_AUTONOMOUS_TURN')) {
      return {
        action: 'spawn_subagents',
        reason: 'One task-specific decision actor is sufficient after the prior team findings.',
        coordination: 'independent',
        agents: [{
          archetype: 'custom',
          name: 'DecisionComposer-1',
          role: 'release decision composer',
          task: 'Use prior-turn findings to produce explicit release criteria.',
          tomLevel: 1,
          existenceReason: 'Transform accumulated findings into a decision contract.',
        }],
      } satisfies DelegationDecision as T;
    }
    if (text.includes("Roy's root delegation controller") && text.includes('<user_task>FIRST_AUTONOMOUS_TURN')) {
      return {
        action: 'spawn_subagents',
        reason: 'The task requires coordinated evidence mapping and assumption testing.',
        coordination: 'team',
        team: {
          name: 'ReleaseReadinessCell',
          description: 'A task-specific cell designed from the current release-readiness gaps.',
          task: 'Establish grounded release readiness.',
          synthesisPolicy: 'Reconcile evidence gaps against operational failure modes and preserve unresolved uncertainty.',
          executionPolicy: {
            mode: 'sequential',
            failureMode: 'best_effort',
            maxConcurrency: 1,
            minimumSuccessfulMembers: 1,
          },
        },
        agents: [
          {
            archetype: 'researcher',
            name: 'EvidenceMapper-1',
            role: 'release evidence mapper',
            task: 'Map the bounded evidence and delegate one verification gap if useful.',
            tools: ['fs.list', 'not.registered'],
            skills: ['use_tool_when_needed', 'delegate_to_subagent', 'not_registered'],
            tomLevel: 1,
            existenceReason: 'Close the factual evidence gap.',
          },
          {
            archetype: 'critic',
            name: 'AssumptionBreaker-1',
            role: 'operational assumption breaker',
            task: 'Identify failure modes not covered by the evidence map.',
            tools: ['fs.read'],
            skills: ['use_tool_when_needed'],
            tomLevel: 2,
            existenceReason: 'Challenge unsupported release assumptions.',
          },
        ],
      } satisfies DelegationDecision as T;
    }
    if (text.includes("Roy's dynamic root-step controller")) {
      return { action: 'finalize', reason: 'The current autonomous structure completed its bounded objective.' } as T;
    }
    if (text.includes("delegation controller") && !text.includes("Roy's root delegation controller")) {
      if (!this.childDelegationUsed) {
        this.childDelegationUsed = true;
        return {
          action: 'spawn_subagents',
          reason: 'A separate verifier can close one evidence boundary.',
          coordination: 'independent',
          agents: [{
            archetype: 'tester',
            name: 'EvidenceVerifier-1',
            role: 'evidence boundary verifier',
            task: 'Verify the bounded evidence claims and return discrepancies.',
            tomLevel: 1,
            existenceReason: 'Verify the parent member evidence before upward synthesis.',
          }],
        } satisfies DelegationDecision as T;
      }
      return { action: 'solve_directly', reason: 'This bounded actor can complete without another child.' } satisfies DelegationDecision as T;
    }
    return {} as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[]): Promise<LLMJSONCompletionResult<T>> {
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

class StreamResilienceLLM implements LLMProvider {
  readonly name = 'deepseek-stream-resilience-test';
  readonly defaultModel = 'test-model';
  private retryTurnAttempts = 0;
  private connectionRetryAttempts = 0;
  private recoveredTurnAttempts = 0;
  private decisionConnectionAttempts = 0;
  private decisionAbortAttempts = 0;
  private jsonRetryAttempts = 0;
  lastJSONMaxTokens?: number;

  async complete(_messages: LLMMessage[]): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('FAIL_TURN')) throw new Error('Premature close');
    if (text.includes('CONNECTION_RETRY')) {
      this.connectionRetryAttempts += 1;
      if (this.connectionRetryAttempts === 1) throw new Error('Premature close');
      if (this.connectionRetryAttempts === 2) throw new Error('Connection error.');
    }
    if (text.includes('TURN_LEVEL_RETRY') && this.recoveredTurnAttempts++ === 0) {
      throw new Error('Connection error.');
    }
    if (text.includes('RETRY_TURN') && this.retryTurnAttempts++ === 0) {
      yield { content: 'discard this partial response', done: false };
      throw new Error('Premature close');
    }
    yield {
      content: text.includes('SECOND_TURN') ? 'The second turn completed after recovery.' : 'The retried turn completed once.',
      done: true,
      usage: { promptTokens: 5, completionTokens: 3, totalTokens: 8 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes('DECISION_CONNECTION_RETRY')
      && this.decisionConnectionAttempts++ === 0) {
      throw new Error('Connection error.');
    }
    if (text.includes('DECISION_ABORT_RETRY')
      && this.decisionAbortAttempts++ === 0) {
      throw new Error('Request was aborted.');
    }
    if (text.includes('JSON_RETRY') && this.jsonRetryAttempts++ === 0) {
      throw new Error('Failed to parse JSON response: {"action":"solve_directly"');
    }
    return { action: 'solve_directly', reason: 'The bounded test task requires no delegation.' } as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    this.lastJSONMaxTokens = options?.maxTokens;
    const value = await this.completeJSON<T>(messages);
    return {
      value,
      completion: {
        content: JSON.stringify(value),
        usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

class InferredToolBindingLLM implements LLMProvider {
  readonly name = 'inferred-tool-binding-test';
  readonly defaultModel = 'test-model';

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => message.content).join('\n');
    const content = text.includes('Synthesize their results into one final user-facing response')
      ? 'Roy compared the grounded runtime API evidence.'
      : 'SourceInspector reported src/index.ts and src/core/runtime/index.ts evidence.';
    yield { content, done: true, usage: { promptTokens: 8, completionTokens: 4, totalTokens: 12 } };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => message.content).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'A task-specific source inspector should collect the API evidence.',
        coordination: 'independent',
        continuationPolicy: 'finalize_after_round',
        agents: [{
          archetype: 'custom',
          name: 'SourceInspector',
          role: 'runtime API source inspector',
          task: 'Read src/index.ts and src/core/runtime/index.ts, then compare their exports.',
        }],
      } satisfies DelegationDecision as T;
    }
    return { action: 'solve_directly', reason: 'No child is needed.' } as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[]): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages);
    return {
      value,
      completion: {
        content: JSON.stringify(value),
        usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Autonomous multi-turn actor design', () => {
  it('infers the minimum read-only tool binding for a model-generated custom source task', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-inferred-tool-binding-'));
    await mkdir(path.join(workspaceCwd, 'src/core/runtime'), { recursive: true });
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, 'src/index.ts'), "export { Runtime } from './core/runtime/index.js';\n");
    await writeFile(path.join(workspaceCwd, 'src/core/runtime/index.ts'), 'export class Runtime {}\n');
    await writeFile(path.join(workspaceCwd, '.roy/config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'inferred-tool-binding-test',
      workspaceCwd,
      llmProvider: new InferredToolBindingLLM(),
    });

    const result = await runtime.handleUserTurn('Inspect the exported runtime API with an autonomous actor.');

    expect(result.subagents).toHaveLength(1);
    expect(result.subagents[0].node.capabilities.tools).toContain('fs.read');
    expect(result.subagents[0].node.capabilities.skills).toContain('use_tool_when_needed');
    expect(result.subagents[0].subagentResult.toolCalls.map(call => call.params.path)).toEqual([
      'src/index.ts',
      'src/core/runtime/index.ts',
    ]);
    expect(result.subagents[0].subagentResult.evidence.observedPaths).toEqual([
      'src/index.ts',
      'src/core/runtime/index.ts',
    ]);
    await runtime.shutdown();
  });

  it('lets the model design a team, recursively add a child, then choose a different structure on later turns', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-autonomous-multi-turn-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: {
        maxChildrenPerParent: 5,
        maxDepth: 3,
        maxTotalAgentsPerTurn: 8,
        rootSteps: { maxDelegationRounds: 1 },
      },
      lifecycle: {
        manual: 'retain_session',
        automaticDelegation: 'retain_session',
        teamMember: 'retain_session',
        evolutionCandidate: 'release',
        retainFailures: true,
        cascade: true,
      },
    }, null, 2));
    const runtime = new Runtime();
    const llm = new AutonomousStructureLLM();
    await runtime.initialize({
      sessionId: 'autonomous-multi-turn-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: llm,
    });

    const experiment = await runtime.runMultiTurnExperiment({
      turns: [
        'FIRST_AUTONOMOUS_TURN: determine release readiness using whatever team structure is useful.',
        'SECOND_AUTONOMOUS_TURN: use the previous result and autonomously choose the smallest useful actor structure.',
        'THIRD_AUTONOMOUS_TURN: state the final decision from the accumulated conversation.',
      ],
    });

    expect(experiment.completedTurns).toBe(3);
    expect(experiment.failedTurns).toBe(0);
    expect(experiment.turns.map(turn => turn.result?.decision.action)).toEqual([
      'spawn_subagents',
      'spawn_subagents',
      'solve_directly',
    ]);

    const first = experiment.turns[0].result!;
    expect(first.decision.action).toBe('spawn_subagents');
    if (first.decision.action !== 'spawn_subagents') throw new Error('Expected delegated first turn');
    expect(first.decision.coordination).toBe('team');
    expect(first.decision.team?.name).toBe('ReleaseReadinessCell');
    expect(first.teams).toHaveLength(1);
    expect(first.teams[0].team.synthesisPolicy).toContain('operational failure modes');
    expect(first.teams[0].team.executionPolicy).toMatchObject({ mode: 'sequential', failureMode: 'best_effort' });
    expect(first.teams[0].members.map(member => member.agent.identity.name)).toEqual([
      'EvidenceMapper-1',
      'AssumptionBreaker-1',
    ]);
    expect(first.teams[0].team.memberAgentIds).toHaveLength(2);

    expect(experiment.turns[0].eventTypes).toContain('delegation.team.designed');
    const verifierNode = first.executionTree.nodes.find(node => node.name === 'EvidenceVerifier-1');
    expect(verifierNode?.parentId).toBe(first.teams[0].members[0].agent.identity.id);
    const teamPattern = (await runtime.getCachePatterns('teams')).find(pattern => pattern.name === 'ReleaseReadinessCell');
    expect(teamPattern).toMatchObject({
      synthesisPolicy: expect.stringContaining('operational failure modes'),
      memberArchetypes: ['researcher', 'critic'],
    });

    const second = experiment.turns[1].result!;
    expect(second.teams).toHaveLength(0);
    expect(second.subagents).toHaveLength(1);
    expect(second.subagents[0].agent.identity.name).toBe('DecisionComposer-1');
    expect(second.subagents[0].node.identity.role).toBe('release decision composer');
    expect(experiment.turns[2].agentIds).toHaveLength(0);
    expect(experiment.turns[2].teamIds).toHaveLength(0);
    expect(llm.rootDelegationPrompts[2]).toContain('SECOND_AUTONOMOUS_TURN');

    const conversation = await runtime.getConversation(undefined, 20);
    expect(conversation.filter(entry => entry.role === 'user')).toHaveLength(3);
    expect(conversation.filter(entry => entry.role === 'assistant')).toHaveLength(3);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'experiment.multi_turn.started',
      'experiment.turn.completed',
      'experiment.multi_turn.completed',
    ]));
    expect(experiment.totalUsage.totalTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('retries a transient root stream without exposing discarded partial output', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-stream-retry-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: { streamMaxAttempts: 2, retryInitialDelayMs: 0, retryMaxDelayMs: 0 },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'stream-retry-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const result = await runtime.handleUserTurn('RETRY_TURN: answer this bounded task directly.');

    expect(result.finalResponse).toBe('The retried turn completed once.');
    expect(result.finalResponse).not.toContain('discard this partial response');
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'llm.stream.retrying',
      'llm.stream.recovered',
    ]));
    expect(runtime.getContext().fsm.getState()).toBe('S_solo');
    await runtime.shutdown();
  });

  it('treats a provider Connection error as retryable after a prematurely closed stream', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-connection-retry-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: { streamMaxAttempts: 3, retryInitialDelayMs: 0, retryMaxDelayMs: 0 },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'connection-retry-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const result = await runtime.handleUserTurn(
      'CONNECTION_RETRY: complete this bounded task after transient provider failures.'
    );

    expect(result.finalResponse).toBe('The retried turn completed once.');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.stream.retrying',
      data: expect.objectContaining({
        attempt: 2,
        error: 'Connection error.',
        retryable: true,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.stream.recovered',
      data: expect.objectContaining({ attempt: 3 }),
    }));
    await runtime.shutdown();
  });

  it('resumes the same persisted task after stream retries are exhausted', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-transient-turn-recovery-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: {
        streamMaxAttempts: 1,
        turnMaxAttempts: 2,
        retryInitialDelayMs: 0,
        retryMaxDelayMs: 0,
      },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'transient-turn-recovery-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const recovery = await runtime.handleUserTurnWithRecovery(
      'TURN_LEVEL_RETRY: resume this exact task after the provider recovers.'
    );

    expect(recovery.result.finalResponse).toBe('The retried turn completed once.');
    expect(recovery.attempts).toBe(2);
    expect(recovery.recovered).toBe(true);
    expect(recovery.correlationIds).toHaveLength(2);
    expect(runtime.listRootExecutionTrees().map(tree => tree.status).sort()).toEqual([
      'completed',
      'failed',
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'runtime.transient_turn.retrying',
      data: expect.objectContaining({
        attempt: 1,
        retryable: true,
        persistedState: true,
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'runtime.transient_turn.recovered',
      correlationId: recovery.result.correlationId,
      data: expect.objectContaining({ attempt: 2 }),
    }));
    await runtime.shutdown();
  });

  it('marks exhausted transient turn recovery as a persisted retryable handoff', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-transient-turn-exhausted-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: {
        streamMaxAttempts: 1,
        turnMaxAttempts: 2,
        retryInitialDelayMs: 0,
        retryMaxDelayMs: 0,
      },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'transient-turn-exhausted-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    await expect(runtime.handleUserTurnWithRecovery(
      'FAIL_TURN: preserve the failed execution state after all provider retries.'
    )).rejects.toThrow('Premature close');

    expect(runtime.listRootExecutionTrees().map(tree => tree.status)).toEqual([
      'failed',
      'failed',
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'runtime.transient_turn.failed',
      data: expect.objectContaining({
        attempt: 2,
        maxAttempts: 2,
        retryable: true,
        persistedState: true,
      }),
    }));
    await runtime.shutdown();
  });

  it('propagates a root decision transport failure to turn recovery instead of returning a fallback', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-decision-transport-recovery-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: {
        jsonMaxAttempts: 1,
        turnMaxAttempts: 2,
        retryInitialDelayMs: 0,
        retryMaxDelayMs: 0,
      },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'decision-transport-recovery-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const recovery = await runtime.handleUserTurnWithRecovery(
      'DECISION_CONNECTION_RETRY: answer this concrete bounded task directly.'
    );

    expect(recovery.recovered).toBe(true);
    expect(recovery.attempts).toBe(2);
    expect(recovery.result.finalResponse).toBe('The retried turn completed once.');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'runtime.transient_turn.retrying',
      data: expect.objectContaining({
        attempt: 1,
        error: 'Connection error.',
      }),
    }));
    expect(runtime.getEvents()).not.toContainEqual(expect.objectContaining({
      type: 'delegation.decision.fallback',
      data: expect.objectContaining({ reason: 'llm_decision_failed' }),
    }));
    await runtime.shutdown();
  });

  it('retries transport failures inside a JSON decision even when reasoning budget is limited', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-limited-json-transport-retry-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: {
        jsonMaxAttempts: 2,
        turnMaxAttempts: 1,
        retryInitialDelayMs: 0,
        retryMaxDelayMs: 0,
      },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'limited-json-transport-retry-test',
      workspaceCwd,
      budget: 30_000,
      llmProvider: new StreamResilienceLLM(),
    });

    const recovery = await runtime.handleUserTurnWithRecovery(
      'DECISION_CONNECTION_RETRY: answer this concrete bounded task directly.'
    );

    expect(recovery.recovered).toBe(false);
    expect(recovery.attempts).toBe(1);
    expect(recovery.result.finalResponse).toBe('The retried turn completed once.');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.retrying',
      data: expect.objectContaining({
        purpose: 'root.delegation_decision',
        attempt: 1,
        retryable: true,
        wallClockMs: expect.any(Number),
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.attempt.started',
      data: expect.objectContaining({
        purpose: 'root.delegation_decision',
        attempt: 1,
        maxAttempts: 2,
        estimatedInputTokens: expect.any(Number),
        maxOutputTokens: expect.any(Number),
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.attempt.completed',
      data: expect.objectContaining({
        purpose: 'root.delegation_decision',
        attempt: 2,
        maxAttempts: 2,
        wallClockMs: expect.any(Number),
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.recovered',
      data: expect.objectContaining({
        purpose: 'root.delegation_decision',
        attempt: 2,
      }),
    }));
    expect(runtime.getEvents()).not.toContainEqual(expect.objectContaining({
      type: 'runtime.transient_turn.retrying',
    }));
    await runtime.shutdown();
  });

  it('retries a request-deadline abort as a transient JSON transport failure', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-json-abort-retry-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: {
        jsonMaxAttempts: 2,
        turnMaxAttempts: 1,
        retryInitialDelayMs: 0,
        retryMaxDelayMs: 0,
      },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'json-abort-retry-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const recovery = await runtime.handleUserTurnWithRecovery(
      'DECISION_ABORT_RETRY: answer this concrete bounded task directly.'
    );

    expect(recovery.result.finalResponse).toBe('The retried turn completed once.');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.retrying',
      data: expect.objectContaining({
        purpose: 'root.delegation_decision',
        attempt: 1,
        retryable: true,
        error: 'Request was aborted.',
      }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'llm.json.recovered',
      data: expect.objectContaining({ attempt: 2 }),
    }));
    await runtime.shutdown();
  });

  it('reserves reasoning tokens and retries an incomplete structured delegation response', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-json-retry-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: { jsonMaxAttempts: 2, retryInitialDelayMs: 0, retryMaxDelayMs: 0 },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const llm = new StreamResilienceLLM();
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'json-retry-test', workspaceCwd, llmProvider: llm });

    const result = await runtime.handleUserTurn('JSON_RETRY: answer this bounded task directly.');

    expect(result.decision.action).toBe('solve_directly');
    expect(llm.lastJSONMaxTokens).toBeGreaterThan(1800);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'llm.json.retrying',
      'llm.json.recovered',
    ]));
    await runtime.shutdown();
  });

  it('restores the root FSM and continues later turns after an unrecoverable stream failure', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-turn-recovery-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      llm: { streamMaxAttempts: 2, retryInitialDelayMs: 0, retryMaxDelayMs: 0 },
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
    }, null, 2));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'turn-recovery-test',
      workspaceCwd,
      llmProvider: new StreamResilienceLLM(),
    });

    const experiment = await runtime.runMultiTurnExperiment({
      stopOnError: false,
      turns: [
        'FAIL_TURN: simulate an exhausted transient stream failure.',
        'SECOND_TURN: continue this session after the prior failure.',
      ],
    });

    expect(experiment.failedTurns).toBe(1);
    expect(experiment.completedTurns).toBe(1);
    expect(experiment.turns[1].result?.finalResponse).toBe('The second turn completed after recovery.');
    expect(runtime.getContext().fsm.getState()).toBe('S_solo');
    expect(runtime.listRootExecutionTrees().map(tree => tree.status)).toEqual(['completed', 'failed']);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'llm.stream.failed',
      'root.turn.failed',
      'root.turn.recovered',
      'experiment.turn.completed',
    ]));
    await runtime.shutdown();
  });
});
