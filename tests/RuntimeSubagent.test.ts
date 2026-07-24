import { describe, it, expect } from 'vitest';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import type { LLMProvider, LLMMessage, LLMCompletionOptions, LLMCompletionResult, LLMJSONCompletionResult, LLMStreamChunk } from '../src/core/llm/types.js';

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

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages, options);
    return { value, completion: { content: JSON.stringify(value), usage: { promptTokens: 3, completionTokens: 1, totalTokens: 4 } } };
  }

  isConfigured(): boolean {
    return true;
  }
}

class ContradictoryArchitectureLLM extends EchoLLM {
  override async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'This is a Rust project built around Cargo.toml, Cargo.lock, and src/main.rs.', done: false };
    yield {
      content: '',
      done: true,
      usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 },
    };
  }
}

class FabricatedPathsLLM extends EchoLLM {
  override async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'The repository contains `package.json`, `src/fabricated/worker.ts`, and `config/missing.yaml`.', done: false };
    yield { content: '', done: true, usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 } };
  }
}

class MarkdownToolIntentLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The runtime evidence confirms package.json and the project source tree.'
      : '```tool\nfs.read\n{"path":"package.json"}\n```';
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

class XmlToolIntentRecoveryLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The recovered runtime call read evidence.txt and confirmed the value.'
      : '<tool_call><tool_name>fs.read</tool_name><path>evidence.txt</path></tool_call>';
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

describe('Runtime controlled subagent spawning', () => {
  it('does not route a local repair through web tools because verifier logs contain URLs', () => {
    const runtime = new Runtime();
    const needsWeb = (task: string) => (runtime as unknown as {
      taskNeedsWebAccess: (value: string) => boolean;
    }).taskNeedsWebAccess(task);

    expect(needsWeb([
      'Work directly in /app. Repair src/dq_audit/audit.py and run the local verifier.',
      '---',
      '## VERIFICATION FAILED — CONTINUE WORKING',
      '<official_verifier_feedback>',
      'WARNING: see https://docs.pytest.org/en/stable/how-to/capture-warnings.html',
      '</official_verifier_feedback>',
    ].join('\n'))).toBe(false);
    expect(needsWeb([
      'Repair the current workspace package and rerun its tests.',
      'Latest command output:',
      'WARNING: use a virtual environment: https://pip.pypa.io/warnings/venv',
    ].join('\n'))).toBe(false);
    expect(needsWeb(
      'Use public web sources to compare the official Node.js and MDN documentation.'
    )).toBe(true);
    expect(needsWeb(
      'Read https://nodejs.org/api/globals.html and summarize the fetch section.'
    )).toBe(true);
  });

  it('removes the tool-use skill from a delegation plan that has no tools', () => {
    const runtime = new Runtime();
    const normalized = (runtime as unknown as {
      normalizeDelegationAgentPlan: (
        plan: Record<string, unknown>,
        fallbackTask: string
      ) => { tools?: string[]; skills?: string[] };
    }).normalizeDelegationAgentPlan({
      archetype: 'custom',
      task: 'Answer the supplied trivia questions from model knowledge.',
      tools: [],
      skills: ['use_tool_when_needed'],
    }, 'Answer the question.');

    expect(normalized.tools).toBeUndefined();
    expect(normalized.skills).toBeUndefined();
  });

  it('allows a semantic researcher to reason without pretending it has an external tool path', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-semantic-researcher-'));
    await writeFile(path.join(workspaceCwd, '.roy-config-placeholder'), '');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'semantic-researcher-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'SemanticResearcher-1',
      tomLevel: 0,
      description: 'Reason over the supplied word list only.',
      task: 'Infer which supplied words match the clue using only the prompt.',
      tools: [],
      skills: [],
      outputContract: { format: 'markdown', groundingRequired: false },
    });

    const result = await runtime.runAgent(
      researcher.identity.id,
      'Given only these words and the clue, rank the most likely matches.',
      { disableRecursiveDelegation: true, archetype: 'researcher' }
    );

    expect(result.toolCalls).toHaveLength(0);
    expect(result.grounded).toBe(true);
    expect(result.result).toBe('subagent result');
    expect(result.warnings).not.toContain(expect.stringContaining('no authorized tool call'));
    await runtime.shutdown();
  });

  it('keeps web tool enablement scoped to each runtime workspace', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-web-scope-'));
    const bootstrap = new Runtime();
    await bootstrap.initialize({
      sessionId: 'web-scope-bootstrap',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    expect(bootstrap.getAgentPolicy('root')?.tools.map(tool => tool.name)).toEqual(expect.arrayContaining([
      'web.search', 'web.fetch',
    ]));
    await bootstrap.shutdown();

    const configPath = path.join(workspaceCwd, '.roy', 'config.json');
    const workspaceConfig = JSON.parse(await readFile(configPath, 'utf8')) as { tools: { web: { enabled: boolean } } };
    workspaceConfig.tools.web.enabled = false;
    await writeFile(configPath, `${JSON.stringify(workspaceConfig, null, 2)}\n`, 'utf8');

    const disabled = new Runtime();
    await disabled.initialize({
      sessionId: 'web-scope-disabled',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    expect(disabled.getAgentPolicy('root')?.tools.map(tool => tool.name)).not.toContain('web.search');
    expect(disabled.getAgentPolicy('root')?.tools.map(tool => tool.name)).not.toContain('web.fetch');
    await expect(disabled.executeToolForAgent('root', 'web.fetch', { url: 'https://example.com' })).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining('not authorized'),
    });
    await disabled.shutdown();
  });

  it('repairs Markdown tool requests instead of presenting them as final output', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-markdown-tool-intent-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"tool-repair-test"}\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'markdown-tool-intent-test',
      llmProvider: new MarkdownToolIntentLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect package exports.', task: 'Inspect package exports.',
    });

    const result = await runtime.runAgent(agent.identity.id, 'Inspect package exports.', {
      archetype: 'researcher', disableRecursiveDelegation: true,
    });

    expect(result.result).not.toContain('```tool');
    expect(result.result).toContain('package.json');
    expect(runtime.getEvents().map(event => event.type)).toContain('agent.output.repair.completed');
    await runtime.shutdown();
  });

  it('executes an authorized unresolved tool intent before repairing the answer', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-tool-intent-recovery-'));
    await writeFile(path.join(workspaceCwd, 'evidence.txt'), 'runtime-grounded\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'tool-intent-recovery-test',
      llmProvider: new XmlToolIntentRecoveryLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      name: 'IntentRecovery-1',
      tomLevel: 0,
      description: 'Resolve the attached fact.',
      task: 'Resolve the attached fact.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(agent.identity.id, 'Resolve the attached fact.', {
      archetype: 'custom',
      disableRecursiveDelegation: true,
    });

    expect(result.toolCalls).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        success: true,
        params: { path: 'evidence.txt' },
      }),
    ]);
    expect(result.result).toContain('evidence.txt');
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'agent.output.tool_intent.recovery.started',
      'agent.output.tool_intent.recovery.completed',
      'agent.output.repair.completed',
    ]));
    await runtime.shutdown();
  });

  it('rejects a model report that contradicts runtime filesystem evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-contradiction-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"typescript-project"}\n', 'utf8');
    await writeFile(path.join(workspaceCwd, 'index.ts'), 'export const value = 1;\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-contradiction-test',
      llmProvider: new ContradictoryArchitectureLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: 'Inspect repository architecture from filesystem evidence.',
      task: 'Inspect repository architecture from filesystem evidence.',
      outputContract: { format: 'markdown', groundingRequired: true },
    });
    const result = await runtime.runAgent(
      agent.identity.id,
      'Inspect repository architecture from filesystem evidence.',
      { archetype: 'researcher' }
    );

    expect(result.evidence.toolGrounded).toBe(true);
    expect(result.evidence.outputGrounded).toBe(false);
    expect(result.grounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('claims a Rust/Cargo project'));
    expect(runtime.getEvents().map(event => event.type)).toContain('agent.grounding.contradiction');

    await runtime.shutdown();
  });

  it('rejects multiple concrete project paths that are absent from runtime evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-path-contradiction-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"typescript-project"}\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-path-contradiction-test',
      llmProvider: new FabricatedPathsLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect repository architecture from filesystem evidence.',
      task: 'Inspect repository architecture from filesystem evidence.',
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(agent.identity.id, agent.identity.description ?? 'Inspect repository.', {
      archetype: 'researcher', disableRecursiveDelegation: true,
    });

    expect(result.grounded).toBe(false);
    expect(result.evidence.outputGrounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('src/fabricated/worker.ts'));
    await runtime.shutdown();
  });

  it('does not mark a grounding-required agent as grounded when no tool can be planned', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-required-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-required-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'tester',
      tomLevel: 0,
      description: 'Verify behavior without an authorized tool.',
      task: 'Verify behavior against tests and failure cases.',
      tools: [],
      outputContract: { format: 'markdown', groundingRequired: true },
    });
    const result = await runtime.runAgent(
      agent.identity.id,
      'Verify behavior against tests and failure cases.',
      { archetype: 'tester', disableRecursiveDelegation: true }
    );

    expect(result.toolCalls).toHaveLength(0);
    expect(result.evidence).toMatchObject({ toolGrounded: false, outputGrounded: false });
    expect(result.grounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('no authorized tool call'));

    await runtime.shutdown();
  });

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
    expect(result.usage.totalTokens).toBeGreaterThanOrEqual(13);

    const budget = runtime.getBudgetState();
    expect(budget.usedTokens).toBe(result.usage.totalTokens);
    expect(budget.perAgent[spawned.identity.id].totalTokens).toBe(result.usage.totalTokens);

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

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBe(9);
    expect(budget.perAgent[result.agent.identity.id].totalTokens).toBe(result.subagentResult.usage.totalTokens);

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
    expect(signals.candidateSignals).toContain('public.project_structure');
    expect(signals.candidateSignals).toContain('roy.delegation_lesson');
    const proposals = await runtime.listMemoryProposals();
    expect(proposals.map(proposal => proposal.target.section)).toContain('tool-policy');
    expect(proposals.map(proposal => proposal.target.section)).toContain('project-structure');
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
    expect(result.subagentResult.evidence.outputGrounded).toBe(true);
    expect(result.subagentResult.result).toContain('## Runtime-Verified Evidence');
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
    expect(rendered.prompt.match(/Introduce yourself briefly\./g)).toHaveLength(1);
    expect(rendered.prompt.match(/<execution_knowledge>/g)).toHaveLength(1);
    expect(rendered.prompt.match(/<agent_memory_file/g)).toBeNull();

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

    expect(researcher?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(researcher?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(critic?.tools.map(tool => tool.name)).toEqual(['fs.read', 'fs.search']);
    expect(critic?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(researcher?.spawnPolicy.maxChildren).toBe(5);
    expect(researcher?.spawnPolicy.maxDepth).toBe(3);

    await runtime.shutdown();
  });

  it('grants web tools only to an agent whose assigned task requires web evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-web-capability-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'web-capability-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const localAgent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect local project files.', task: 'Inspect local project files.',
    });
    const webAgent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Search the web for the latest official Node.js documentation.',
      task: 'Search the web for the latest official Node.js documentation.',
    });

    expect(runtime.getAgentPolicy(localAgent.identity.id)?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(runtime.getAgentPolicy(webAgent.identity.id)?.tools.map(tool => tool.name)).toEqual([
      'fs.list', 'fs.read', 'fs.search', 'web.search', 'web.fetch',
    ]);
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
    expect(policy?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(policy?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);

    const agentPatterns = await runtime.getCachePatterns('agents');
    const researcherPattern = agentPatterns.find(pattern => pattern.id === 'agent_pattern_researcher_v1');
    expect(researcherPattern?.tools).toEqual(['fs.list', 'fs.read', 'fs.search']);
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

  it('keeps a custom agent name from colliding with a built-in archetype pattern', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-pattern-namespace-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'pattern-namespace-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    await runtime.handleSpawnCommand({
      archetype: 'custom',
      name: 'Critic',
      task: 'Recommend semantic candidates from the supplied prompt.',
      tools: [],
      skills: [],
    });
    await expect(runtime.handleSpawnCommand({
      archetype: 'critic',
      name: 'ArchitectureCritic',
      task: 'Critique the supplied architecture evidence.',
    })).resolves.toBeDefined();

    const patterns = await runtime.getCachePatterns('agents');
    expect(patterns.map(pattern => pattern.id)).toEqual(expect.arrayContaining([
      'agent_pattern_custom-critic_v1',
      'agent_pattern_critic_v1',
    ]));
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
    expect(parentEvents.some(event => event.type === 'agent.fsm.state' && event.data?.state === 'S_synthesizing')).toBe(true);
    expect(runtime.getBudgetState().perAgent[researcher.identity.id].totalTokens).toBeGreaterThan(0);
    expect(result.finalResponse).toBe('subagent result');

    await runtime.shutdown();
  });

  it('lets a non-root agent recursively delegate to a direct child during its run', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-recursive-delegation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'recursive-delegation-test',
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
    const result = await runtime.runAgent(
      researcher.identity.id,
      'Review project risks and grounding gaps with a direct child critic.',
      { correlationId: 'del_recursive_test', archetype: 'researcher' }
    );

    const tree = runtime.getAgentTree();
    expect(tree.children[0].agent.identity.id).toBe(researcher.identity.id);
    expect(tree.children[0].children).toHaveLength(3);
    expect(tree.children[0].children[0].agent.identity.id).toBe('agent_critic_002');
    expect(tree.children[0].children.map(child => child.agent.identity.tomProfile.cognitiveGaps).flat().length).toBeGreaterThan(0);

    expect(result.agent.identity.id).toBe(researcher.identity.id);
    expect(result.result).toBe('subagent result');
    expect(result.usage.totalTokens).toBeGreaterThan(0);

    const messages = await runtime.getMessages({ correlationId: 'del_recursive_test' });
    expect(messages.map(message => message.kind)).toContain('agent.create.request');
    expect(messages.map(message => message.kind)).toContain('agent.task');
    expect(messages.map(message => message.kind)).toContain('agent.result');
    expect(messages.map(message => message.kind)).toContain('agent.synthesis');

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('delegation.decision');
    expect(eventTypes).toContain('delegation.plan.created');
    expect(eventTypes).toContain('delegation.completed');
    expect(eventTypes).toContain('agent.synthesis.completed');
    expect(runtime.getEvents().some(event =>
      event.type === 'budget.rebalanced'
      && event.agentId === researcher.identity.id
      && event.data?.purpose === 'agent.multi_child_synthesis'
    )).toBe(true);
    expect(runtime.getEvents().some(event =>
      event.type === 'budget.context.truncated'
      && event.agentId === researcher.identity.id
      && event.data?.purpose === 'agent.multi_child_synthesis'
    )).toBe(true);

    await runtime.shutdown();
  });

  it('lets a non-root parent aggregate multiple direct children', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-multi-child-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'multi-child-delegation-test',
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
    const result = await runtime.runAgent(
      researcher.identity.id,
      'Delegate project risk review and test-coverage verification to direct children, then aggregate them.',
      { correlationId: 'del_multi_child_test', archetype: 'researcher' }
    );

    const children = runtime.getChildren(researcher.identity.id);
    expect(children).toHaveLength(3);
    expect(children.map(child => child.identity.name)).toEqual(expect.arrayContaining([
      expect.stringMatching(/^Critic-/),
      expect.stringMatching(/^Tester-/),
      expect.stringMatching(/^Researcher-/),
    ]));
    expect(result.result).toBe('subagent result');

    const messages = await runtime.getMessages({ correlationId: 'del_multi_child_test' });
    const team = runtime.getTeams()[0];
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'team.result' && message.to === researcher.identity.id)).toHaveLength(1);
    expect(messages.filter(message => message.kind === 'agent.synthesis' && message.from === researcher.identity.id)).toHaveLength(1);

    const synthesisEvent = runtime.getEvents().find(event => event.type === 'agent.synthesis.completed' && event.agentId === researcher.identity.id);
    expect(synthesisEvent?.data?.childIds).toEqual(children.map(child => child.identity.id));
    expect(runtime.getBudgetState().perAgent[researcher.identity.id].totalTokens).toBeGreaterThan(0);

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
