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
      'agent.create.request',
      'agent.create.approved',
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
    expect(result.creationUsage.mode).toBe('generated');
    expect(result.creationUsage.definitionTokens).toBeGreaterThan(0);
    expect(result.creationUsage.renderedPromptTokens).toBeGreaterThan(0);

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
    expect(second.creationUsage.mode).toBe('cache_hit');
    expect(second.creationUsage.definitionTokens).toBe(0);
    expect(second.creationUsage.renderedPromptTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('injects custom agent name and role into rendered prompts', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-custom-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'custom-agent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const rendered = await runtime.renderAgentPrompt({
      agentKey: 'custom',
      name: 'Singer-1',
      role: 'performer',
      task: 'Introduce yourself briefly.',
      archetype: 'custom',
    });

    expect(rendered.prompt).toContain('Singer-1');
    expect(rendered.prompt).toContain('performer');
    expect(rendered.prompt).toContain('Introduce yourself briefly.');

    await runtime.shutdown();
  });

  it('exposes built-in archetype skills, tools, and spawn policies', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-archetypes-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'archetype-policy-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const profiles = runtime.getAgentArchetypeProfiles();
    const researcher = profiles.find(profile => profile.archetype === 'researcher');
    const critic = profiles.find(profile => profile.archetype === 'critic');

    expect(researcher?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read']);
    expect(researcher?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(critic?.tools.map(tool => tool.name)).toEqual(['fs.read']);
    expect(critic?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(researcher?.spawnPolicy.maxChildren).toBe(5);
    expect(researcher?.spawnPolicy.maxDepth).toBe(3);

    await runtime.shutdown();
  });

  it('binds parent-approved tools and skills, and stores them in cache patterns', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-bindings-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'binding-cache-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });
    const policy = runtime.getAgentPolicy(result.agent.identity.id);
    expect(policy?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read']);
    expect(policy?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);

    const agentPatterns = await runtime.getCachePatterns('agents');
    const researcherPattern = agentPatterns.find(pattern => pattern.id === 'agent_pattern_researcher_v1');
    expect(researcherPattern?.tools).toEqual(['fs.list', 'fs.read']);
    expect(researcherPattern?.skills).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);
    expect(researcherPattern?.spawnPolicy).toMatchObject({
      maxChildren: 5,
      maxDepth: 3,
      budgetAware: true,
    });

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.create.requested');
    expect(eventTypes).toContain('spawn.policy.checked');
    expect(eventTypes).toContain('agent.create.approved');
    expect(eventTypes).toContain('agent.instance.created');
    expect(eventTypes).toContain('agent.tool.bound');
    expect(eventTypes).toContain('agent.skill.bound');

    await runtime.shutdown();
  });

  it('creates custom agents with custom identity, role, and explicit bindings', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-custom-spawn-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'custom-spawn-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'custom',
      name: 'PromptAuditor-1',
      customRole: 'prompt inspector',
      task: 'Introduce yourself briefly.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
    });

    expect(result.agent.identity.name).toBe('PromptAuditor-1');
    expect(result.agent.identity.description).toContain('Introduce yourself briefly.');
    const policy = runtime.getAgentPolicy(result.agent.identity.id);
    expect(policy?.tools.map(tool => tool.name)).toEqual(['fs.read']);
    expect(policy?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed']);

    const prompt = await readFile(path.join(workspaceCwd, '.roy', 'agents', 'promptauditor-1', 'prompt.md'), 'utf8');
    expect(prompt).toContain('{{agent_identity}}');

    await runtime.shutdown();
  });

  it('rejects the sixth direct child under the default parent child limit', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-child-limit-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'child-limit-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    for (let index = 1; index <= 5; index += 1) {
      await runtime.spawnAgent({
        parentId: 'root',
        archetype: 'researcher',
        name: `Researcher-${index}`,
        tomLevel: 0,
        description: `task ${index}`,
        task: `task ${index}`,
      });
    }

    await expect(runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-6',
      tomLevel: 0,
      description: 'task 6',
      task: 'task 6',
    })).rejects.toThrow('max_children_exceeded');

    const rejected = runtime.getEvents().find(event => event.type === 'spawn.policy.rejected');
    expect(rejected?.data?.reason).toBe('max_children_exceeded');
    expect(runtime.getChildren('root')).toHaveLength(5);

    await runtime.shutdown();
  });

  it('supports creating a subsubagent under a subagent parent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-subsubagent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'subsubagent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const critic = await runtime.spawnAgent({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      tomLevel: 2,
      description: 'Review Researcher-1 output',
      task: 'Review Researcher-1 output',
    });

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].agent.identity.id).toBe(researcher.identity.id);
    expect(tree.children[0].children).toHaveLength(1);
    expect(tree.children[0].children[0].agent.identity.id).toBe(critic.identity.id);

    await runtime.shutdown();
  });

  it('routes subsubagent results through parent synthesis before root final synthesis', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-parent-synthesis-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'parent-synthesis-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const result = await runtime.handleSpawnCommand({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      task: 'Review Researcher-1 output',
    });

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('agent.synthesis');
    const parentResult = messages.find(message => message.kind === 'agent.result' && message.from === researcher.identity.id && message.to === 'root');
    expect(parentResult).toBeDefined();

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.synthesis.started');
    expect(eventTypes).toContain('agent.synthesis.completed');
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');

    const parentEvents = runtime.getEvents().filter(event => event.agentId === researcher.identity.id);
    expect(parentEvents.some(event => event.type === 'agent.fsm.state' && event.data?.state === 'S_synthesize')).toBe(true);
    expect(runtime.getBudgetState().perAgent[researcher.identity.id].totalTokens).toBeGreaterThan(0);
    expect(result.finalResponse).toBe('subagent result');

    await runtime.shutdown();
  });

  it('rejects child creation when the parent is failed', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-invalid-fsm-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'invalid-fsm-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const agent = runtime.getContext().manager.getAgentById(researcher.identity.id);
    agent?.setRuntimeState('failed');

    await expect(runtime.spawnAgent({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      tomLevel: 2,
      description: 'Review failed researcher',
      task: 'Review failed researcher',
    })).rejects.toThrow('invalid_fsm_state');

    const rejected = runtime.getEvents().find(event => event.type === 'delegation.rejected');
    expect(rejected?.data?.reason).toBe('invalid_fsm_state');

    await runtime.shutdown();
  });
});
