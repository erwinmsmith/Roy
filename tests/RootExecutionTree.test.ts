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

describe('Root dynamic execution tree', () => {
  it('reassesses prior state, grows the tree in a dependent step, and lets Roy finalize', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-dynamic-tree-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
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
    await runtime.initialize({
      sessionId: 'dynamic-tree-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new DynamicStepLLM(),
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
    expect(result.executionTree.nodes.find(node => node.role === 'tester')?.createdAtStep).toBe(2);
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
    expect(result.executionTree.steps[0].decision).toMatchObject({ action: 'delegate', agentCount: 1 });
    expect(result.executionTree.steps[0].actorIds).toHaveLength(1);
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
    expect(result.subagents[0].node.identity.archetype).toBe('planner');
    expect(result.executionTree.steps.map(step => step.decision.action)).toEqual(['delegate', 'finalize']);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({ type: 'root.task_loop.promoted' }));
    await runtime.shutdown();
  });
});
