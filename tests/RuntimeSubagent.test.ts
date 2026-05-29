import { describe, it, expect } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type { LLMProvider, LLMMessage, LLMCompletionOptions, LLMCompletionResult, LLMStreamChunk } from '../src/core/llm/types.js';

class EchoLLM implements LLMProvider {
  readonly name = 'echo-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'echo complete',
      usage: {
        promptTokens: 5,
        completionTokens: 3,
        totalTokens: 8,
      },
    };
  }

  async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'subagent ', done: false };
    yield {
      content: 'result',
      done: true,
      usage: {
        promptTokens: 7,
        completionTokens: 2,
        totalTokens: 9,
      },
    };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('Runtime controlled subagent spawning', () => {
  it('spawns, registers, runs, and tracks a subagent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-subagent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'subagent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const spawned = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 2,
      description: 'Inspect runtime state',
      task: 'Inspect runtime state',
      budgetTokens: 8000,
    });

    expect(spawned.identity.id).toBe('agent_researcher_001');
    expect(spawned.identity.parentId).toBe('root');

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].agent.identity.id).toBe(spawned.identity.id);

    const result = await runtime.runAgent(spawned.identity.id, 'Check token accounting');
    expect(result.result).toBe('subagent result');
    expect(result.usage.totalTokens).toBe(9);

    const budget = runtime.getBudgetState();
    expect(budget.usedTokens).toBe(9);
    expect(budget.perAgent[spawned.identity.id].totalTokens).toBe(9);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.spawned');
    expect(eventTypes).toContain('budget.allocated');
    expect(eventTypes).toContain('agent.run.completed');

    await runtime.shutdown();
  });

  it('runs controlled spawn through root-mediated messages and synthesis', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-mediated-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'mediated-spawn-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });

    expect(result.correlationId).toMatch(/^del_/);
    expect(result.agent.identity.tomProfile.level).toBe(0);
    expect(result.subagentResult.grounded).toBe(true);
    expect(result.subagentResult.toolCalls.map(call => call.toolName)).toContain('fs.list');
    expect(result.finalResponse).toBe('subagent result');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toEqual([
      'user.command.spawn',
      'agent.task',
      'tool.call',
      'tool.result',
      'agent.result',
      'root.synthesis',
      'root.final_response',
    ]);

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBe(9);
    expect(budget.perAgent[result.agent.identity.id].totalTokens).toBe(9);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');
    expect(eventTypes).toContain('agent.result.sent');
    expect(eventTypes).toContain('memory.pattern.updated');
    expect((await runtime.getConversation(undefined, 20)).some(entry => entry.role === 'agent')).toBe(true);
    const memoryState = await runtime.getMemoryState();
    expect(memoryState.agentMemories.map(memory => memory.id)).toContain('researcher');
    expect(memoryState.patterns.agents).toBe(1);
    expect(memoryState.patterns.delegations).toBe(1);
    const signals = await runtime.collectMemorySignals();
    expect(signals.counts.agentResults).toBe(1);
    expect(signals.candidateSignals).toContain('researcher.tool_policy');
    expect(signals.candidateSignals).toContain('researcher.failure_case');
    expect(signals.candidateSignals).toContain('roy.delegation_lesson');
    const proposals = await runtime.listMemoryProposals();
    expect(proposals.map(proposal => proposal.target.section)).toContain('tool-policy');
    expect(proposals.map(proposal => proposal.target.section)).toContain('failure-cases');
    expect(proposals.map(proposal => proposal.target.section)).toContain('delegation-lessons');
    expect(proposals[0].id).toMatch(/^mem_prop_\d{17}_[a-f0-9]{4}$/);

    const prompt = await readFile(path.join(workspaceCwd, '.roy', 'agents', 'researcher', 'prompt.md'), 'utf8');
    expect(prompt).toContain('{{public_context}}');
    expect(prompt).toContain('{{agent_private_memory}}');
    expect(prompt).toContain('{{agent_identity}}');
    expect(prompt).toContain('{{tom_profile}}');
    expect(prompt).toContain('{{available_skills}}');
    expect(prompt).toContain('{{available_tools}}');
    expect(prompt).toContain('{{parent_context}}');
    expect(prompt).toContain('{{task}}');
    expect(result.subagentResult.evidence.toolGrounded).toBe(true);
    expect(result.subagentResult.evidence.outputGrounded).toBe(false);
    expect(result.creationUsage.promptDefinitionTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('emits cache hits on repeated controlled spawn', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'cache-hit-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });
    const second = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure again',
    });

    const hits = runtime.getEvents()
      .filter(event => event.type === 'cache.hit' && event.data?.correlationId === second.correlationId)
      .map(event => event.data?.patternId);
    expect(hits).toContain('agent_pattern_researcher_v1');
    expect(hits).toContain('delegation_project_inspection_researcher_v1');
    expect(second.creationUsage.cacheHits).toHaveLength(2);
    expect(second.creationUsage.promptDefinitionTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });
});
