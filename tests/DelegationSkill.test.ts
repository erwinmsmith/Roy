import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { skillRegistry } from '../src/core/skills/index.js';
import { DelegateToSubagentSkill } from '../src/core/skills/delegation.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class DelegationLLM implements LLMProvider {
  readonly name = 'delegation-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
    };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    const content = text.includes('Produce the final response to the user as Roy')
      ? 'Roy synthesized the researcher result.'
      : 'Observed project paths:\n- README.md\n- package.json\n- src/\n- tests/';
    yield {
      content,
      done: true,
      usage: { promptTokens: 12, completionTokens: 5, totalTokens: 17 },
    };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

class FailingDelegationLLM extends DelegationLLM {
  override async *stream(): AsyncGenerator<LLMStreamChunk, void, unknown> {
    throw new Error('simulated_llm_failure');
  }
}

describe('delegate_to_subagent skill', () => {
  it('is registered as a system skill with agent creation permissions', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-manifest-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-manifest-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const manifest = skillRegistry.getManifest('delegate_to_subagent');
    expect(manifest?.scope).toBe('system');
    expect(manifest?.permissions).toEqual(['agent.create', 'agent.delegate']);
    expect(skillRegistry.unregister('delegate_to_subagent')).toBe(false);
    expect(skillRegistry.has('delegate_to_subagent')).toBe(true);
    expect(() => skillRegistry.register(new DelegateToSubagentSkill(runtime)))
      .toThrow('cannot be overwritten by an extension');

    await runtime.shutdown();
  });

  it('delegates through the runtime message-mediated subagent flow', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-skill-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-skill-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const output = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: {
          archetype: 'researcher',
          task: 'Inspect the project structure',
          parentId: 'root',
          requireRootSynthesis: true,
          lifecycle: { mode: 'retain_session' },
        },
      },
      {
        agentId: 'root',
        sessionId: 'delegation-skill-test',
        variables: {},
      }
    );

    expect(output.success).toBe(true);
    const result = output.result as Record<string, any>;
    expect(result.correlationId).toMatch(/^del_/);
    expect(result.agentId).toBe('agent_researcher_001');
    expect(result.agentName).toBe('Researcher-1');
    expect(result.node.nodeId).toMatch(/^node_del_/);
    expect(result.node.parentId).toBe('root');
    expect(result.node.sessionId).toBe('delegation-skill-test');
    expect(result.node.definitionFingerprint).toMatch(/^[a-f0-9]{64}$/);
    expect(result.node.invocationFingerprint).toMatch(/^[a-f0-9]{64}$/);
    expect(result.node.capabilities.tools).toEqual(['fs.list', 'fs.read']);
    expect(result.node.capabilities.skills).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);
    expect(result.node.assignment.outputContract.groundingRequired).toBe(true);
    expect(result.node.governance.lifecycle).toEqual({ mode: 'retain_session' });
    expect(result.agentResult.toolCalls.map((call: any) => call.toolName)).toContain('fs.list');
    expect(result.rootSynthesis).toBe('Roy synthesized the researcher result.');
    expect(result.tokenUsage.root.totalTokens).toBeGreaterThan(0);
    expect(result.tokenUsage.subagent.totalTokens).toBeGreaterThan(0);
    expect(result.events.map((event: any) => event.type)).toContain('agent.spawned');
    expect(result.events.map((event: any) => event.type)).toContain('agent.run.completed');
    expect(result.events.map((event: any) => event.type)).toContain('agent.node.resolved');
    expect(result.events.map((event: any) => event.type)).toContain('agent.node.execution.completed');
    expect(result.events.every((event: any) => (event.correlationId ?? event.data?.correlationId) === result.correlationId)).toBe(true);

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toEqual([
      'agent.control',
      'agent.create.request',
      'budget.request',
      'budget.grant',
      'agent.create.approved',
      'agent.task',
      'tool.approval.request',
      'tool.approval.resolved',
      'tool.call',
      'tool.result',
      'agent.result',
      'root.synthesis',
      'budget.request',
      'budget.grant',
      'root.final_response',
    ]);
    expect(messages.find(message => message.kind === 'agent.create.request')?.payload).toMatchObject({
      nodeId: result.node.nodeId,
      definitionFingerprint: result.node.definitionFingerprint,
      creationMode: 'generated',
    });
    const conversation = await runtime.getConversation('delegation-skill-test', 20);
    expect(conversation.some(entry => entry.content.startsWith('/spawn '))).toBe(false);
    expect(conversation.find(entry => entry.metadata?.kind === 'agent.result')?.metadata).toMatchObject({
      nodeId: result.node.nodeId,
      definitionFingerprint: result.node.definitionFingerprint,
      creationMode: 'generated',
    });

    await runtime.shutdown();
  });

  it('rejects parent impersonation and cross-session execution', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-boundary-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-boundary-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const impersonation = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: { parentId: 'root', archetype: 'researcher', task: 'Inspect the project' },
      },
      { agentId: 'agent_other_001', sessionId: 'delegation-boundary-test', variables: {} }
    );
    expect(impersonation.success).toBe(false);
    expect(impersonation.error).toContain('parent mismatch');

    const wrongSession = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: { parentId: 'root', archetype: 'researcher', task: 'Inspect the project' },
      },
      { agentId: 'root', sessionId: 'different-session', variables: {} }
    );
    expect(wrongSession.success).toBe(false);
    expect(wrongSession.error).toContain('session mismatch');

    const failedCorrelationId = 'del_require_cache_failure';
    await expect(runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Inspect the project',
      reuse: { mode: 'require_cache' },
    }, {
      agentId: 'root',
      sessionId: 'delegation-boundary-test',
      source: 'test',
    }, failedCorrelationId)).rejects.toThrow('requires a cached pattern');
    expect(runtime.getEvents().find(event => event.type === 'agent.node.resolve.failed'
      && event.correlationId === failedCorrelationId)?.data?.error).toContain('requires a cached pattern');

    await expect(runtime.createAgentComputeNode({
      archetype: 'custom',
      task: 'Use an unknown capability',
      tools: ['unknown.tool'],
    }, {
      agentId: 'root',
      sessionId: 'delegation-boundary-test',
      source: 'test',
    })).rejects.toThrow('requested unknown tool');

    await runtime.shutdown();
  });

  it('reuses a cached definition while creating a fresh runtime instance', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-cache-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });
    const execute = (task: string, reuseMode: string = 'prefer_cache', overrides: Record<string, unknown> = {}) => skillRegistry.execute(
      'delegate_to_subagent',
      { action: 'delegate_to_subagent', params: { archetype: 'researcher', task, reuseMode, ...overrides } },
      { agentId: 'root', sessionId: 'delegation-cache-test', variables: {} }
    );

    const first = (await execute('Inspect the project structure')).result as Record<string, any>;
    const second = (await execute('Inspect the project structure')).result as Record<string, any>;
    expect(first.node.reuse.creationMode).toBe('generated');
    expect(second.node.reuse.creationMode).toBe('cache_hit');
    expect(first.node.definitionFingerprint).toBe(second.node.definitionFingerprint);
    expect(first.node.invocationFingerprint).not.toBe(second.node.invocationFingerprint);
    expect(first.agentId).not.toBe(second.agentId);
    expect(second.node.reuse.cacheHits).toContain('agent_pattern_researcher_v1');
    expect(second.events.some((event: any) => event.type === 'cache.hit')).toBe(true);

    const cache = JSON.parse(await readFile(path.join(workspaceCwd, '.roy', 'cache', 'agent-patterns.json'), 'utf8'));
    const pattern = cache.patterns.find((item: any) => item.id === 'agent_pattern_researcher_v1');
    expect(pattern.definitionFingerprint).toBe(second.node.definitionFingerprint);
    expect(pattern.memoryScope).toEqual(second.node.context.memoryScope);
    expect(pattern.outputContract).toEqual(second.node.assignment.outputContract);
    expect(pattern.usage.count).toBe(2);

    const mutated = (await execute('Inspect only package metadata', 'prefer_cache', { tools: ['fs.read'] })).result as Record<string, any>;
    expect(mutated.node.reuse.creationMode).toBe('mutated_from_cache');
    expect(mutated.node.definitionFingerprint).not.toBe(second.node.definitionFingerprint);
    expect(mutated.creationUsage.definitionTokens).toBeGreaterThan(0);

    const cacheAfterMutation = JSON.parse(await readFile(path.join(workspaceCwd, '.roy', 'cache', 'agent-patterns.json'), 'utf8'));
    const canonicalAfterMutation = cacheAfterMutation.patterns.find((item: any) => item.id === 'agent_pattern_researcher_v1');
    const mutationPattern = cacheAfterMutation.patterns.find((item: any) => item.id === mutated.node.reuse.targetPatternId);
    expect(canonicalAfterMutation.tools).toEqual(['fs.list', 'fs.read']);
    expect(canonicalAfterMutation.definitionFingerprint).toBe(second.node.definitionFingerprint);
    expect(mutationPattern.basePatternId).toBe('agent_pattern_researcher_v1');
    expect(mutationPattern.tools).toEqual(['fs.read']);
    expect(mutationPattern.status).toBe('candidate');
    expect(mutationPattern.evaluation.runs).toBe(1);

    const fresh = (await execute('Inspect another project area', 'fresh')).result as Record<string, any>;
    expect(fresh.node.reuse.creationMode).toBe('generated');
    expect(fresh.node.reuse.cacheHits).toEqual([]);

    await runtime.shutdown();
  });

  it('scopes repeated-correlation executions to the current compute node', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-event-scope-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-event-scope-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });
    const invocation = { agentId: 'root', sessionId: 'delegation-event-scope-test', source: 'test' };
    const correlationId = 'del_shared_correlation';
    const first = await runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    }, invocation, correlationId);
    const second = await runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Inspect the project structure again',
    }, invocation, correlationId);

    expect(first.delegation.agent.identity.id).not.toBe(second.delegation.agent.identity.id);
    expect(second.events.filter(event => event.type === 'agent.spawned').map(event => event.agentId))
      .toEqual([second.delegation.agent.identity.id]);
    expect(second.events.some(event => event.agentId === first.delegation.agent.identity.id)).toBe(false);

    await runtime.shutdown();
  });

  it('constrains child memory and spawn policy to parent and workspace limits', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-policy-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-policy-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });
    const invocation = { agentId: 'root', sessionId: 'delegation-policy-test', source: 'test' };
    const execution = await runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Inspect the project structure',
      memoryScope: { public: true, private: true, parentContext: true, sessionWindowTurns: 999 },
      spawnPolicy: {
        canSpawn: true,
        maxChildren: 999,
        maxDepth: 99,
        maxTotalAgentsPerTurn: 999,
        allowCustomAgents: true,
        budgetAware: false,
        allowedStates: ['S_planning', 'S_delegating'],
      },
    }, invocation);

    expect(execution.node.context.memoryScope.sessionWindowTurns).toBe(10);
    expect(execution.node.governance.spawnPolicy).toMatchObject({
      maxChildren: 5,
      maxDepth: 3,
      maxTotalAgentsPerTurn: 10,
      budgetAware: true,
      allowedStates: ['S_planning', 'S_delegating'],
    });

    await expect(runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Use an invalid child state',
      spawnPolicy: { allowedStates: ['S_done'] },
    }, invocation)).rejects.toThrow('allowedStates may only contain');

    await runtime.shutdown();
  });

  it('fails the task message and releases reserved budget when execution fails', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-failure-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-failure-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new FailingDelegationLLM(),
    });
    runtime.setBudget(5000);
    const correlationId = 'del_execution_failure';

    await expect(runtime.createAgentComputeNode({
      archetype: 'researcher',
      task: 'Inspect the project structure',
      execution: { disableRecursiveDelegation: true },
    }, {
      agentId: 'root',
      sessionId: 'delegation-failure-test',
      source: 'test',
    }, correlationId)).rejects.toThrow('simulated_llm_failure');

    const messages = await runtime.getMessages({ correlationId });
    expect(messages.find(message => message.kind === 'agent.task')?.status).toBe('failed');
    expect(runtime.getBudgetMarketState().reservedTokens).toBe(0);
    expect(runtime.getBudgetMarketState().allocations.some(allocation => allocation.status === 'released')).toBe(true);
    expect(runtime.getEvents().some(event => event.type === 'agent.task.failed' && event.correlationId === correlationId)).toBe(true);
    expect(runtime.getEvents().some(event => event.type === 'agent.node.execution.failed' && event.correlationId === correlationId)).toBe(true);

    await runtime.shutdown();
  });

  it('validates required delegation inputs', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-delegation-skill-invalid-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'delegation-skill-invalid',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: new DelegationLLM(),
    });

    const output = await skillRegistry.execute(
      'delegate_to_subagent',
      {
        action: 'delegate_to_subagent',
        params: { archetype: 'unknown' },
      },
      {
        agentId: 'root',
        sessionId: 'delegation-skill-invalid',
        variables: {},
      }
    );

    expect(output.success).toBe(false);
    expect(output.error).toContain('Validation failed');

    await runtime.shutdown();
  });
});
