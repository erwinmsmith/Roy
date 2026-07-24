import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
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

class DynamicStepLLM implements LLMProvider {
  readonly name = 'dynamic-step-test';
  readonly defaultModel = 'test-model';
  private continuationCalls = 0;
  continuationPrompt = '';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'complete', usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 } };
  }

  async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    const content = text.includes('Synthesize their results into one final user-facing response')
      ? 'Roy final answer based on staged research and verification.'
      : text.includes('Verify the prior researcher result')
        ? 'Tester report: the prior observation is verified.'
        : 'Researcher report: observed the project structure and identified a verification gap.';
    yield {
      content,
      done: true,
      usage: { promptTokens: 20, completionTokens: 8, totalTokens: 28 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Start with grounded inspection before deciding verification.',
        agents: [{
          archetype: 'researcher',
          name: 'Researcher-1',
          task: 'Inspect the project structure and identify what must be verified next.',
          tomLevel: 0,
        }],
      } satisfies DelegationDecision as T;
    }
    if (text.includes("Roy's dynamic root-step controller")) {
      this.continuationPrompt = text;
      this.continuationCalls += 1;
      if (this.continuationCalls === 1) {
        return {
          action: 'delegate_more',
          reason: 'The research result exposed a verification dependency.',
          agents: [{
            archetype: 'tester',
            name: 'Tester-1',
            task: 'Verify the prior researcher result and report concrete discrepancies.',
            tomLevel: 0,
          }],
        } as T;
      }
      return { action: 'finalize', reason: 'Research and verification are both complete.' } as T;
    }
    return { action: 'none', params: {} } as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[]): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages);
    return {
      value,
      completion: {
        content: JSON.stringify(value),
        usage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 },
      },
    };
  }

  isConfigured(): boolean {
    return true;
  }
}

class MalformedRootDecisionLLM extends DynamicStepLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      throw new Error('provider returned reasoning without structured JSON');
    }
    if (text.includes("Roy's dynamic root-step controller")) {
      return { action: 'finalize', reason: 'The first grounded step is sufficient.' } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class ContinuingStepLLM extends DynamicStepLLM {
  private step = 0;

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's dynamic root-step controller")) {
      this.step += 1;
      return {
        action: 'delegate_more',
        reason: `Continue bounded task iteration ${this.step}.`,
        agents: [{
          archetype: 'tester',
          name: `Tester-${this.step}`,
          task: `Verify unique checkpoint ${this.step}.`,
          tomLevel: 0,
        }],
      } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class RecoverableStepFailureLLM extends DynamicStepLLM {
  private continuationCalls = 0;

  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    const content = text.includes('Synthesize their results into one final user-facing response')
      ? 'Roy recovered the prior completed checkpoint and produced a final response.'
      : text.includes('Emit an unresolved tool request')
        ? '<tool_call><tool_name>fs.read</tool_name><path>missing.txt</path></tool_call>'
        : 'First checkpoint completed with usable evidence.';
    yield {
      content,
      done: true,
      usage: { promptTokens: 20, completionTokens: 8, totalTokens: 28 },
    };
  }

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's dynamic root-step controller")) {
      this.continuationCalls += 1;
      return {
        action: 'delegate_more',
        reason: 'Try one additional bounded checkpoint.',
        agents: [{
          archetype: 'custom',
          name: 'FailingCheckpoint',
          task: 'Emit an unresolved tool request without executing it.',
          tomLevel: 0,
        }],
      } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class DirectInitialDecisionLLM extends DynamicStepLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return { action: 'solve_directly', reason: 'Initially classified as direct.' } as T;
    }
    if (text.includes("Roy's dynamic root-step controller")) {
      return { action: 'finalize', reason: 'The planning checkpoint is sufficient.' } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class FinalizeAfterRoundLLM extends DynamicStepLLM {
  continuationCalls = 0;

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'A bounded two-view team should complete this task in one round.',
        coordination: 'team',
        continuationPolicy: 'finalize_after_round',
        team: {
          name: 'BoundedDecisionCell',
          description: 'Produces two bounded views and stops after synthesis.',
          synthesisPolicy: 'Reconcile both views once, then return the result.',
        },
        agents: [
          { archetype: 'custom', name: 'ConstraintView', task: 'State the decision constraints.' },
          { archetype: 'custom', name: 'ChallengeView', task: 'Challenge one decision assumption.' },
        ],
      } satisfies DelegationDecision as T;
    }
    if (text.includes("Roy's dynamic root-step controller")) {
      this.continuationCalls += 1;
      return { action: 'delegate_more', reason: 'This must not be called.', agents: [] } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

describe('Root dynamic execution tree', () => {
  it('honors a model-selected finalize-after-round policy for a formal team', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-finalize-after-team-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: { rootSteps: { maxDelegationRounds: 4, reassessAfterDelegation: true } },
    }));
    const llm = new FinalizeAfterRoundLLM();
    const runtime = new Runtime();
    await runtime.initialize({ sessionId: 'finalize-after-team-test', workspaceCwd, llmProvider: llm });

    const result = await runtime.handleUserTurn('Use one team and finalize immediately after its synthesis.');

    expect(result.decision).toMatchObject({
      action: 'spawn_subagents',
      coordination: 'team',
      continuationPolicy: 'finalize_after_round',
    });
    expect(result.teams).toHaveLength(1);
    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual(['delegate', 'finalize']);
    expect(llm.continuationCalls).toBe(0);
    await runtime.shutdown();
  });

  it('reassesses prior state, grows the tree in a dependent step, and lets Roy finalize', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-dynamic-tree-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 1 },
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 4,
          maxDelegationRounds: 3,
          reassessAfterDelegation: true,
        },
      },
    }));
    const runtime = new Runtime();
    const llm = new DynamicStepLLM();
    await runtime.initialize({
      sessionId: 'dynamic-tree-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: llm,
    });

    const result = await runtime.handleUserTurn('Inspect this project, then verify any gap before answering.');

    expect(result.correlationId).toContain('dynamic-tree-test');
    expect(result.finalResponse).toBe('Roy final answer based on staged research and verification.');
    expect(result.subagents.map(item => item.node.identity.archetype)).toEqual(expect.arrayContaining(['researcher', 'tester']));
    expect(result.executionTree.status).toBe('completed');
    expect(result.executionTree.steps).toHaveLength(3);
    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual([
      'delegate',
      'delegate',
      'finalize',
    ]);
    expect(result.executionTree.steps[1].dependsOn).toEqual([result.executionTree.steps[0].id]);
    expect(result.executionTree.steps[2].dependsOn).toEqual([result.executionTree.steps[1].id]);
    expect(result.executionTree.nodes.map(node => node.role)).toEqual(expect.arrayContaining([
      'root',
      'researcher',
      'tester',
    ]));
    expect(result.executionTree.nodes.find(node => node.role === 'researcher')?.createdAtStep).toBe(1);
    expect(result.executionTree.nodes.find(node => node.name === 'CheckpointVerifier-2')?.createdAtStep).toBe(1);
    expect(result.executionTree.nodes
      .filter(node => node.name === 'Tester-1')
      .some(node => node.createdAtStep === 2)).toBe(true);
    expect(result.executionTree.steps[1].actorIds.filter(
      actorId => result.executionTree.steps[0].actorIds.includes(actorId)
    )).toEqual([]);
    expect(result.executionTree.loop).toMatchObject({
      iteration: 3,
      stopReason: 'completed',
      maxIterations: 4,
    });
    expect(result.executionTree.steps.every(step => step.activities.length > 0)).toBe(true);
    expect(result.executionTree.steps.every(step => step.checkpoint?.stateFingerprint)).toBe(true);
    expect(result.executionTree.steps[0].activities.map(activity => activity.kind)).toEqual(expect.arrayContaining([
      'conversation',
      'thinking',
      'agent',
      'checkpoint',
    ]));

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.filter(message => message.kind === 'root.step.plan')).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'root.step.result')).toHaveLength(3);
    const events = runtime.getEvents().filter(event => event.correlationId === result.correlationId);
    expect(events.filter(event => event.type === 'root.step.started')).toHaveLength(3);
    expect(events.filter(event => event.type === 'root.step.tree.updated')).toHaveLength(3);
    expect(events).toContainEqual(expect.objectContaining({ type: 'root.execution_tree.completed' }));
    expect(events).toContainEqual(expect.objectContaining({ type: 'execution.cache.snapshot.recorded' }));
    expect(events).toContainEqual(expect.objectContaining({ type: 'execution.path.updated' }));
    expect(llm.continuationPrompt).toContain('<execution_knowledge>');
    expect(llm.continuationPrompt).toContain('<acceptance_checklist>');
    expect(llm.continuationPrompt).toContain('LongHorizonCheckpointTeam');
    expect(llm.continuationPrompt).toContain('strict JSON object');

    const executionKnowledge = JSON.parse(
      await readFile(path.join(workspaceCwd, '.roy', 'cache', 'execution-knowledge.json'), 'utf8')
    );
    expect(executionKnowledge.steps).toHaveLength(3);
    expect(executionKnowledge.paths).toHaveLength(3);
    expect(executionKnowledge.actors).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: 'team', generation: 1 }),
      expect.objectContaining({ kind: 'agent', role: 'researcher' }),
      expect.objectContaining({ kind: 'agent', role: 'tester' }),
    ]));
    expect(result.executionTree.steps[0].cache).toMatchObject({
      path: expect.objectContaining({
        id: `${result.executionTree.steps[0].id}.path`,
      }),
    });

    const persisted = await runtime.listPersistedRootExecutionTrees('dynamic-tree-test');
    expect(persisted).toHaveLength(1);
    const persistedTree = JSON.parse(await readFile(persisted[0].path, 'utf8'));
    expect(persistedTree.steps).toHaveLength(3);
    expect(persistedTree.steps[1].dependsOn).toEqual([result.executionTree.steps[0].id]);

    await runtime.shutdown();

    const resumedRuntime = new Runtime();
    await resumedRuntime.initialize({
      sessionId: 'dynamic-tree-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new DynamicStepLLM(),
    });
    const restored = await resumedRuntime.loadRootExecutionTree(result.correlationId);
    expect(restored?.steps).toHaveLength(3);
    expect(restored?.steps[0].activities.length).toBeGreaterThan(0);
    await resumedRuntime.shutdown();
  });

  it('uses file-aware fallback and keeps staged work out of the initial ToM expansion', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-dynamic-fallback-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'dynamic-fallback-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new MalformedRootDecisionLLM(),
    });

    const result = await runtime.handleUserTurn(
      'Inspect package.json using filesystem evidence. After the first result, decide whether a separate verifier is needed.'
    );

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.executionTree.steps[0].decision).toMatchObject({ action: 'delegate', agentCount: 2 });
    expect(result.executionTree.steps[0].actorIds).toHaveLength(2);
    expect(result.executionTree.steps[0].teamIds).toHaveLength(1);
    expect(result.subagents[0].node.identity.archetype).toBe('researcher');
    expect(result.executionTree.steps[0].activities).toContainEqual(expect.objectContaining({
      kind: 'tool',
      status: 'completed',
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'delegation.decision.fallback',
      data: expect.objectContaining({ reason: 'llm_decision_failed' }),
    }));

    await runtime.shutdown();
  });

  it('reserves a final step and stops a long task at the configured loop boundary', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-bounded-loop-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 4,
          maxDelegationRounds: 10,
          reassessAfterDelegation: true,
          maxWallClockMs: 60000,
          maxStalledIterations: 5,
          persistEveryStep: true,
        },
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'bounded-loop-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new ContinuingStepLLM(),
    });

    const result = await runtime.handleUserTurn('Inspect this project through several dependent verification steps.');

    expect(result.executionTree.steps).toHaveLength(4);
    expect(result.executionTree.steps.slice(0, 3).every(step => step.decision.action === 'delegate')).toBe(true);
    expect(result.executionTree.steps[3].decision.action).toBe('finalize');
    expect(result.executionTree.loop.stopReason).toBe('max_iterations');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.step.limit_reached',
      data: expect.objectContaining({ reason: 'max_iterations' }),
    }));
    await runtime.shutdown();
  });

  it('reserves both execution and finalization steps for workspace mutations', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-bounded-mutation-loop-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 4,
          maxDelegationRounds: 10,
          reassessAfterDelegation: true,
          maxWallClockMs: 60000,
          maxStalledIterations: 5,
        },
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'bounded-mutation-loop-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new ContinuingStepLLM(),
    });

    const result = await runtime.handleUserTurn('Modify the project code and run tests.');

    expect(result.executionTree.steps).toHaveLength(4);
    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual([
      'delegate',
      'delegate',
      'solve_directly',
      'finalize',
    ]);
    expect(result.executionTree.loop.stopReason).toBe('closure_unmet');
    expect(result.finalResponse).toContain('[runtime_execution_closure_unmet]');
    await runtime.shutdown();
  });

  it('hands a mutation task to root execution after bounded exploratory delegation', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-mutation-handoff-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 10,
          maxDelegationRounds: 10,
          reassessAfterDelegation: true,
          maxWallClockMs: 60000,
          maxStalledIterations: 8,
        },
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'mutation-handoff-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new ContinuingStepLLM(),
    });

    const result = await runtime.handleUserTurn('Modify the project code and run tests.');

    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual([
      'delegate',
      'delegate',
      'solve_directly',
      'finalize',
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.execution.handoff.required',
      data: expect.objectContaining({
        delegationRounds: 2,
        reason: 'delegation_round_cap_without_mutation',
      }),
    }));
    await runtime.shutdown();
  });

  it('keeps the execution tree running when a later delegated step is recoverable', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-recoverable-step-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      delegation: {
        rootSteps: {
          enabled: true,
          maxStepsPerTurn: 5,
          maxDelegationRounds: 4,
          reassessAfterDelegation: true,
        },
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'recoverable-step-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RecoverableStepFailureLLM(),
    });

    const result = await runtime.handleUserTurn(
      'Analyze this staged question, then recover from a failed optional checkpoint.'
    );

    expect(result.finalResponse).toContain('recovered the prior completed checkpoint');
    expect(result.executionTree.status).toBe('completed');
    expect(result.executionTree.steps.map(step => step.status)).toEqual([
      'completed',
      'failed',
      'completed',
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.step.recovered',
      data: expect.objectContaining({ recovery: 'synthesize_completed_prior_steps' }),
    }));
    expect(result.executionTree.steps[1].cache).toMatchObject({
      step: expect.objectContaining({ status: 'failed' }),
      path: expect.objectContaining({ status: expect.stringMatching(/failed|partial/) }),
    });
    expect(result.executionTree.steps[1].cache?.feedback).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: 'actor_failure', actionable: true }),
    ]));
    await runtime.shutdown();
  });

  it('promotes an explicit long-horizon task into a checkpointed loop', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-long-horizon-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'long-horizon-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new DirectInitialDecisionLLM(),
    });

    const result = await runtime.handleUserTurn('Execute this multi-step task progressively with checkpoints until complete.');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.teams).toHaveLength(1);
    expect(result.subagents.map(item => item.node.identity.archetype)).toEqual([
      'researcher',
      'tester',
    ]);
    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual(['delegate', 'finalize']);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({ type: 'root.task_loop.promoted' }));
    await runtime.shutdown();
  });

  it('orders long-horizon evidence, implementation, and verification without duplicating a custom executor', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-long-horizon-dependencies-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'long-horizon-dependencies-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new DirectInitialDecisionLLM(),
    });
    const normalize = (runtime as unknown as {
      ensureLongHorizonTeamDecision: (
        decision: Extract<DelegationDecision, { action: 'spawn_subagents' }>,
        task: string,
        requiresWorkspaceMutation: boolean,
        correlationId: string
      ) => Extract<DelegationDecision, { action: 'spawn_subagents' }>;
    }).ensureLongHorizonTeamDecision;
    const decision = normalize.call(runtime, {
      action: 'spawn_subagents',
      reason: 'Build and verify the pipeline.',
      coordination: 'team',
      agents: [
        {
          archetype: 'custom',
          name: 'PipelineCoder',
          role: 'implementation engineer',
          task: 'Implement the data pipeline in the authoritative source tree.',
          tools: ['fs.read', 'fs.replace', 'fs.write', 'shell.exec'],
        },
        {
          archetype: 'tester',
          name: 'PipelineTester',
          role: 'acceptance verifier',
          task: 'After PipelineCoder completes, run the pipeline and verify every artifact.',
          tools: ['fs.read', 'shell.exec'],
        },
      ],
      team: {
        name: 'PipelineTeam',
        description: 'Build and verify the pipeline.',
        executionPolicy: {
          mode: 'parallel',
          maxConcurrency: 2,
          minimumSuccessfulMembers: 1,
        },
      },
    }, 'Implement the complete long-horizon workspace pipeline and verify it.', true, 'dependency-order-test');

    expect(decision.agents.map(agent => agent.name)).toEqual([
      'PathSteward-1',
      'PipelineCoder',
      'PipelineTester',
    ]);
    expect(decision.agents.filter(agent => agent.archetype === 'coder')).toHaveLength(0);
    expect(decision.team?.executionPolicy).toMatchObject({
      mode: 'sequential',
      maxConcurrency: 1,
    });
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'delegation.team.dependencies.normalized',
      correlationId: 'dependency-order-test',
    }));
    await runtime.shutdown();
  });

  it('rejects premature finalize when a long-horizon mutation path is still unverified', () => {
    const runtime = new Runtime();
    const ensureRecovery = (runtime as unknown as {
      ensureLongHorizonRecoveryContinuation: (
        continuation: { action: 'finalize'; reason: string },
        task: string,
        step: unknown,
        delegationRound: number,
        maxDelegationRounds: number,
        requiresLongHorizon: boolean,
        requiresWorkspaceMutation: boolean,
        correlationId: string
      ) => {
        action: string;
        agents?: Array<{ archetype: string; name: string }>;
        coordination?: string;
        team?: { memberDelegationPolicy?: string };
      };
    }).ensureLongHorizonRecoveryContinuation;
    const continuation = ensureRecovery.call(
      runtime,
      { action: 'finalize', reason: 'The first edit appears sufficient.' },
      'Continue until the workspace migration is implemented and verified.',
      {
        id: 'step_01',
        index: 1,
        cache: {
          path: {
            id: 'step_01.path',
            status: 'partial',
            mutationObserved: true,
            verificationObserved: false,
          },
          feedback: [{
            actionable: true,
            summary: 'The verification command has not run.',
          }],
        },
      },
      1,
      4,
      true,
      true,
      'long-horizon-recovery-test'
    );

    expect(continuation).toMatchObject({
      action: 'delegate_more',
      coordination: 'team',
      agents: [
        { archetype: 'coder', name: 'RecoveryExecutor-2' },
        { archetype: 'tester', name: 'RecoveryVerifier-2' },
      ],
      team: { memberDelegationPolicy: 'allow' },
    });
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'root.step.long_horizon_recovery.required',
      correlationId: 'long-horizon-recovery-test',
      data: expect.objectContaining({
        pathStatus: 'partial',
        mutationObserved: true,
        verificationObserved: false,
      }),
    }));
  });

  it('hands a long-horizon task back to root as soon as a delegated mutation occurs', () => {
    const runtime = new Runtime();
    const shouldHandoff = (runtime as unknown as {
      shouldHandoffToRootExecution: (input: {
        requiresWorkspaceMutation: boolean;
        requiresLongHorizon: boolean;
        roundMutationApplied: boolean;
        delegationRounds: number;
        maxRounds: number;
        exploratoryDelegationLimit: number;
      }) => boolean;
    }).shouldHandoffToRootExecution;

    expect(shouldHandoff.call(runtime, {
      requiresWorkspaceMutation: true,
      requiresLongHorizon: true,
      roundMutationApplied: true,
      delegationRounds: 1,
      maxRounds: 12,
      exploratoryDelegationLimit: 4,
    })).toBe(true);
    expect(shouldHandoff.call(runtime, {
      requiresWorkspaceMutation: true,
      requiresLongHorizon: true,
      roundMutationApplied: false,
      delegationRounds: 1,
      maxRounds: 12,
      exploratoryDelegationLimit: 4,
    })).toBe(false);
    expect(shouldHandoff.call(runtime, {
      requiresWorkspaceMutation: true,
      requiresLongHorizon: true,
      roundMutationApplied: false,
      delegationRounds: 4,
      maxRounds: 12,
      exploratoryDelegationLimit: 4,
    })).toBe(true);
  });

  it('does not treat mutation-task output paths as missing input evidence', () => {
    const runtime = new Runtime();
    const buildFollowUp = (runtime as unknown as {
      buildRequiredEvidenceFollowUp: (
        task: string,
        subagents: unknown[]
      ) => unknown;
    }).buildRequiredEvidenceFollowUp;

    expect(buildFollowUp.call(
      runtime,
      'Implement the pipeline and create outputs/validation_report.json, then verify it.',
      []
    )).toBeUndefined();
  });
});
