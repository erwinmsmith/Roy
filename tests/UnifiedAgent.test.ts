import { describe, it, expect, beforeEach } from 'vitest';
import { UnifiedAgent } from '../src/core/agent/UnifiedAgent.js';
import { toolRegistry } from '../src/core/tools/index.js';
import { skillRegistry, UseToolWhenNeededSkill } from '../src/core/skills/index.js';
import { actionRegistry } from '../src/core/actions/index.js';
import { MessageQueue } from '../src/core/message/MessageQueue.js';
import type { LLMProvider, LLMMessage, LLMCompletionOptions, LLMCompletionResult, LLMStreamChunk } from '../src/core/llm/types.js';
import type { Tool } from '../src/core/tools/types.js';
import type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput, SkillManifest } from '../src/core/skills/types.js';

class PlanningLLM implements LLMProvider {
  readonly name = 'planning-test';
  readonly defaultModel = 'test-model';
  jsonCalls = 0;

  constructor(private readonly action: string) {}

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return { content: 'unused' };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const result = messages.findLast(message => message.content.includes('Capability result:'))?.content ?? '';
    yield { content: `synthesized:${result}`, done: true };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    this.jsonCalls += 1;
    return {
      action: this.action,
      params: { value: 'hello' },
      reasoning: 'test plan',
    } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

class NonJSONWebPlanningLLM extends PlanningLLM {
  constructor() {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    throw new Error([
      'Failed to parse JSON response. Fetch these likely official sources:',
      'https://nodejs.org/docs/latest/api/globals.html#fetch',
      'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static',
    ].join('\n'));
  }
}

class ReadOnlyThenMutationPlanningLLM extends PlanningLLM {
  constructor() {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    if (this.jsonCalls === 1) {
      return {
        action: 'call_tools',
        reason: 'Repeat inspection instead of implementing.',
        calls: [{ toolName: 'fs.read', params: { path: 'artifact.txt' } }],
      } as T;
    }
    return {
      action: 'call_tools',
      reason: 'Apply the required workspace change.',
      calls: [{
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'implemented' },
      }],
    } as T;
  }
}

class MutationRepairPlanningLLM extends PlanningLLM {
  constructor(private readonly repeatVerificationFirst = false) {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    if (this.repeatVerificationFirst && this.jsonCalls === 1) {
      return {
        action: 'call_tools',
        reason: 'Repeat the already failed verification.',
        calls: [{
          toolName: 'shell.exec',
          params: { command: 'pytest -q', timeoutMs: 60_000 },
        }],
      } as T;
    }
    return {
      action: 'call_tools',
      reason: 'Repair the implementation using the failed verification output.',
      calls: [{
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'repaired' },
      }],
    } as T;
  }
}

class FragileWriterThenReplacePlanningLLM extends PlanningLLM {
  constructor() {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    if (this.jsonCalls === 1) {
      return {
        action: 'call_tools',
        reason: 'Rewrite a source line through an inline interpreter.',
        calls: [{
          toolName: 'shell.exec',
          params: {
            command: "python -c \"from pathlib import Path; p=Path('src/app.py'); p.write_text('fixed')\"",
          },
        }],
      } as T;
    }
    return {
      action: 'call_tools',
      reason: 'Use the dedicated exact replacement tool.',
      calls: [{
        toolName: 'fs.replace',
        params: {
          path: 'src/app.py',
          oldText: 'broken',
          newText: 'fixed',
        },
      }],
    } as T;
  }
}

class DestructiveRepairThenReplacePlanningLLM extends PlanningLLM {
  constructor() {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    if (this.jsonCalls === 1) {
      return {
        action: 'call_tools',
        reason: 'Replace the entire existing source after one verifier failure.',
        calls: [{
          toolName: 'fs.write',
          params: {
            path: 'src/app.py',
            content: 'full rewrite that discards working behavior',
            mode: 'overwrite',
          },
        }],
      } as T;
    }
    return {
      action: 'call_tools',
      reason: 'Preserve the implementation and repair only the reported line.',
      calls: [{
        toolName: 'fs.replace',
        params: {
          path: 'src/app.py',
          oldText: 'broken assertion',
          newText: 'fixed assertion',
          expectedReplacements: 1,
        },
      }],
    } as T;
  }
}

class NoisyFailureInspectionPlanningLLM extends PlanningLLM {
  constructor() {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    return {
      action: 'call_tools',
      reason: 'Inspect the reported line and several unrelated paths.',
      calls: [
        {
          toolName: 'fs.read',
          params: { path: 'src/app.py', startLine: 40, endLine: 70 },
        },
        {
          toolName: 'shell.exec',
          params: { command: 'head -20 unrelated.csv' },
        },
        {
          toolName: 'fs.list',
          params: { path: '.', maxDepth: 5 },
        },
      ],
    } as T;
  }
}

class EchoTool implements Tool {
  readonly name = 'echo-tool';
  readonly description = 'Echoes a value';

  async execute(params: Record<string, unknown>) {
    return {
      success: true,
      result: `tool:${params.value}`,
    };
  }
}

class LargeResultTool implements Tool {
  readonly name = 'large-result-tool';
  readonly description = 'Returns a large evidence payload';

  async execute() {
    return { success: true, result: `observed-evidence\n${'entry\n'.repeat(5000)}` };
  }
}

class EchoSkill implements Skill {
  readonly name = 'echo-skill';
  readonly description = 'Echoes a value';
  readonly version = '1.0.0';

  getManifest(): SkillManifest {
    return {
      name: this.name,
      version: this.version,
      description: this.description,
      tags: ['test'],
    };
  }

  async execute(input: SkillInput, _context: SkillContext): Promise<SkillOutput> {
    return {
      success: true,
      result: `skill:${input.params.value}`,
    };
  }

  async initialize(_config: SkillConfig): Promise<void> {}
}

class ContextIdentitySkill extends EchoSkill {
  override readonly name = 'context-identity-skill';

  override async execute(_input: SkillInput, context: SkillContext): Promise<SkillOutput> {
    return { success: true, result: context.agentId };
  }
}

class CapturingToolPlanningLLM extends PlanningLLM {
  messages: LLMMessage[] = [];
  options?: LLMCompletionOptions;

  constructor(private readonly failure?: Error) {
    super('none');
  }

  override async completeJSON<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<T> {
    this.messages = messages;
    this.options = options;
    if (this.failure) throw this.failure;
    return { action: 'finish', reason: 'No more calls are needed.', calls: [] } as T;
  }
}

class TruncatedMutationPlanningLLM extends PlanningLLM {
  messagesByAttempt: LLMMessage[][] = [];
  optionsByAttempt: Array<LLMCompletionOptions | undefined> = [];

  constructor() {
    super('none');
  }

  override async completeJSON<T>(
    messages: LLMMessage[],
    options?: LLMCompletionOptions
  ): Promise<T> {
    this.jsonCalls += 1;
    this.messagesByAttempt.push(messages);
    this.optionsByAttempt.push(options);
    if (this.jsonCalls === 1) {
      throw new Error('Failed to parse JSON response: {"action":"call_tools","calls":[{"toolName":"fs.write"');
    }
    return {
      action: 'call_tools',
      reason: 'Apply the first bounded implementation chunk.',
      calls: [{
        toolName: 'fs.write',
        params: {
          path: 'src/implementation.py',
          content: 'first bounded chunk',
          mode: 'overwrite',
        },
      }],
    } as T;
  }
}

class RecoverableTruncatedWritePlanningLLM extends PlanningLLM {
  constructor(private readonly content = 'first line\\nsecond line\\nthird partial') {
    super('none');
  }

  override async completeJSON<T>(): Promise<T> {
    this.jsonCalls += 1;
    throw new Error(
      'Failed to parse JSON response: '
      + '{"action":"call_tools","reason":"write source","calls":[{"toolName":"fs.write",'
      + '"params":{"path":"src/app.py","content":"'
      + this.content
    );
  }
}

describe('UnifiedAgent capability execution', () => {
  beforeEach(() => {
    actionRegistry.clear();
    toolRegistry.clear();
    skillRegistry.clear();
  });

  it('executes registered tools through the main step flow', async () => {
    toolRegistry.register(new EchoTool());

    const agent = new UnifiedAgent({
      name: 'agent',
      goal: 'test',
      llm: new PlanningLLM('echo-tool'),
      mode: 'action',
    });
    const queue = new MessageQueue(['env', 'agent']);
    agent.setMessageQueue(queue);
    await agent.initialize('session');

    await agent.step('run the echo tool');

    const output = await queue.receive('env');
    expect(output?.content).toContain('tool:hello');
    expect(output?.metadata?.done).toBe(true);
  });

  it('executes registered skills through the main step flow', async () => {
    skillRegistry.register(new EchoSkill());

    const agent = new UnifiedAgent({
      name: 'agent',
      goal: 'test',
      llm: new PlanningLLM('echo-skill'),
      mode: 'action',
    });
    const queue = new MessageQueue(['env', 'agent']);
    agent.setMessageQueue(queue);
    await agent.initialize('session');

    await agent.step('run the echo skill');

    const output = await queue.receive('env');
    expect(output?.content).toContain('skill:hello');
    expect(output?.metadata?.done).toBe(true);
  });

  it('falls back to reasoning when an optional tool action omits the tool name', async () => {
    skillRegistry.register(new UseToolWhenNeededSkill());
    const agent = new UnifiedAgent({
      name: 'knowledge-agent',
      goal: 'answer from available knowledge',
      llm: new PlanningLLM('use_tool_when_needed'),
      mode: 'action',
      allowedSkills: ['use_tool_when_needed'],
    });
    const queue = new MessageQueue(['env', 'knowledge-agent']);
    agent.setMessageQueue(queue);
    await agent.initialize('knowledge-session');

    await agent.step('check these facts and answer directly');

    expect((await queue.receive('env'))?.content).not.toContain('Action error');
    expect(agent.getInfo().error).toBeUndefined();
  });

  it('passes the stable agent id to system skills instead of the display name', async () => {
    skillRegistry.register(new ContextIdentitySkill());

    const agent = new UnifiedAgent({
      id: 'agent_researcher_001',
      name: 'Researcher-1',
      goal: 'test',
      llm: new PlanningLLM('context-identity-skill'),
      mode: 'action',
      allowedSkills: ['context-identity-skill'],
    });
    const queue = new MessageQueue(['env', 'Researcher-1']);
    agent.setMessageQueue(queue);
    await agent.initialize('session');

    await agent.step('delegate a child task');

    expect((await queue.receive('env'))?.content).toContain('agent_researcher_001');
  });

  it('rejects globally registered skills that were not authorized for the agent', async () => {
    skillRegistry.register(new EchoSkill());
    const agent = new UnifiedAgent({
      id: 'agent_restricted_001',
      name: 'Restricted-1',
      goal: 'test',
      llm: new PlanningLLM('echo-skill'),
      mode: 'action',
      allowedSkills: [],
    });
    const queue = new MessageQueue(['env', 'Restricted-1']);
    agent.setMessageQueue(queue);
    await agent.initialize('session');

    await agent.step('attempt an unauthorized skill');

    expect((await queue.receive('env'))?.content).toContain('is not authorized');
  });

  it('does not treat cognitive stress-testing as a command to execute a tool', async () => {
    toolRegistry.register(new EchoTool());
    const agent = new UnifiedAgent({
      name: 'critic',
      goal: 'critique evidence',
      llm: new PlanningLLM('echo-tool'),
      mode: 'hybrid',
    });
    const queue = new MessageQueue(['env', 'critic']);
    agent.setMessageQueue(queue);
    await agent.initialize('critic-session');

    await agent.step('Stress-test the evidence and proposed conclusions.');

    expect((await queue.receive('env'))?.content).not.toContain('tool:hello');
  });

  it('compacts large capability results to fit the active synthesis allocation', async () => {
    toolRegistry.register(new LargeResultTool());
    const agent = new UnifiedAgent({
      name: 'bounded-agent',
      goal: 'summarize evidence',
      llm: new PlanningLLM('large-result-tool'),
      mode: 'action',
    });
    const queue = new MessageQueue(['env', 'bounded-agent']);
    agent.setMessageQueue(queue);
    agent.setCompletionTokenLimit(800, 'total_tokens');
    await agent.initialize('bounded-session');

    await expect(agent.step('run the evidence inspection')).resolves.toBeUndefined();
    expect((await queue.receive('env'))?.content).toContain('observed-evidence');
  });

  it('uses runtime grounding directly and compacts an oversized system prompt within allocation', async () => {
    toolRegistry.register(new EchoTool());
    const llm = new PlanningLLM('echo-tool');
    const agent = new UnifiedAgent({
      name: 'grounded-agent',
      goal: `preserve identity and constraints\n${'large cached context\n'.repeat(1200)}`,
      llm,
      mode: 'hybrid',
    });
    const queue = new MessageQueue(['env', 'grounded-agent']);
    agent.setMessageQueue(queue);
    agent.setCompletionTokenLimit(2200, 'total_tokens');
    await agent.initialize('grounded-session');

    await expect(agent.step([
      '[runtime_grounding_provided]',
      'Inspect the package exports.',
      'Grounding context:',
      'Filesystem listing:',
      'package.json',
      'src',
    ].join('\n'))).resolves.toBeUndefined();

    expect(llm.jsonCalls).toBe(0);
    expect((await queue.receive('env'))?.content).not.toContain('tool:hello');
    expect(agent.getInfo().error).toBeUndefined();
  });

  it('recovers authorized public URLs from a non-JSON tool-planning response', async () => {
    const agent = new UnifiedAgent({
      name: 'web-researcher',
      goal: 'collect web evidence',
      llm: new NonJSONWebPlanningLLM(),
      mode: 'hybrid',
      allowedTools: ['web.search', 'web.fetch'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Open official Node.js and MDN documentation.',
      round: 1,
      remainingCalls: 2,
      tools: [{ name: 'web.search' }, { name: 'web.fetch' }],
      calls: [{
        toolName: 'web.search', params: { query: 'Node.js fetch' }, reason: 'search',
        groundingRequired: true, success: true, result: { results: [] },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' },
      }),
      expect.objectContaining({
        toolName: 'web.fetch',
        params: { url: 'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static' },
      }),
    ]);
  });

  it('rejects a read-only continuation when a workspace mutation is still required', async () => {
    const llm = new ReadOnlyThenMutationPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'implementation-agent',
      goal: 'implement workspace changes',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement the project change by creating artifact.txt and run tests.',
      round: 1,
      remainingCalls: 4,
      tools: [{ name: 'fs.read' }, { name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [{
        toolName: 'fs.read',
        params: { path: 'artifact.txt' },
        reason: 'initial inspection',
        groundingRequired: true,
        success: false,
        error: 'missing',
      }],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'implemented' },
      }),
    ]);
  });

  it('preserves execution intent when Runtime wraps the original mutation task', async () => {
    const llm = new ReadOnlyThenMutationPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'wrapped-repair-agent',
      goal: 'continue the persisted implementation',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: '[runtime_execution_repair_phase]\nUse cached failure evidence and continue.',
      executionRequired: true,
      round: 3,
      remainingCalls: 4,
      tools: [{ name: 'fs.read' }, { name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [{
        toolName: 'fs.read',
        params: { path: 'artifact.txt' },
        reason: 'inspect the current implementation',
        groundingRequired: true,
        success: true,
        result: { content: 'incomplete' },
      }],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'implemented' },
      }),
    ]);
  });

  it('allows another workspace repair after a mutation when verification failed', async () => {
    const llm = new MutationRepairPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'repair-agent',
      goal: 'repair failed implementation',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement the project change and run tests.',
      round: 2,
      remainingCalls: 4,
      tools: [{ name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [
        {
          toolName: 'fs.write',
          params: { path: 'artifact.txt', content: 'initial' },
          reason: 'initial implementation',
          groundingRequired: true,
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q' },
          reason: 'verify implementation',
          groundingRequired: true,
          success: false,
          error: 'tests failed',
        },
      ],
    });

    expect(llm.jsonCalls).toBe(1);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'repaired' },
      }),
    ]);
  });

  it('rejects an equivalent failed verification call before planning a repair', async () => {
    const llm = new MutationRepairPlanningLLM(true);
    const agent = new UnifiedAgent({
      name: 'deduplicating-repair-agent',
      goal: 'repair failed implementation',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement the project change and run tests.',
      round: 2,
      remainingCalls: 4,
      tools: [{ name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [
        {
          toolName: 'fs.write',
          params: { path: 'artifact.txt', content: 'initial' },
          reason: 'initial implementation',
          groundingRequired: true,
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q' },
          reason: 'verify implementation',
          groundingRequired: true,
          success: false,
          error: 'tests failed',
        },
      ],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'repaired' },
      }),
    ]);
  });

  it('allows the same verification command after a newer repair mutation', async () => {
    const llm = new MutationRepairPlanningLLM(true);
    const agent = new UnifiedAgent({
      name: 'post-repair-verifier',
      goal: 'verify the newest workspace state',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement the project change and run tests.',
      round: 3,
      remainingCalls: 2,
      tools: [{ name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q' },
          reason: 'verify initial implementation',
          groundingRequired: true,
          success: false,
          error: 'tests failed',
        },
        {
          toolName: 'fs.write',
          params: { path: 'artifact.txt', content: 'repaired' },
          reason: 'repair the failure',
          groundingRequired: true,
          success: true,
        },
      ],
    });

    expect(llm.jsonCalls).toBe(1);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: expect.objectContaining({ command: 'pytest -q' }),
      }),
    ]);
  });

  it('rejects fragile inline shell writers when dedicated file tools are available', async () => {
    const llm = new FragileWriterThenReplacePlanningLLM();
    const agent = new UnifiedAgent({
      name: 'safe-source-repair-agent',
      goal: 'repair source without shell quoting corruption',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.replace', 'fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Repair src/app.py and run its verification.',
      executionRequired: true,
      round: 2,
      remainingCalls: 3,
      tools: [
        { name: 'fs.read' },
        { name: 'fs.replace' },
        { name: 'fs.write' },
        { name: 'shell.exec' },
      ],
      calls: [{
        toolName: 'fs.read',
        params: { path: 'src/app.py', startLine: 10, endLine: 20 },
        reason: 'Inspect the reported failure.',
        groundingRequired: true,
        success: true,
        result: { content: 'broken' },
      }],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.replace',
        params: {
          path: 'src/app.py',
          oldText: 'broken',
          newText: 'fixed',
        },
      }),
    ]);
  });

  it('rejects destructive overwrites of existing files during verifier repair', async () => {
    const llm = new DestructiveRepairThenReplacePlanningLLM();
    const agent = new UnifiedAgent({
      name: 'focused-verifier-repair-agent',
      goal: 'repair without regressing working behavior',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.replace', 'fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Repair src/app.py and run the official tests.',
      executionRequired: true,
      round: 4,
      remainingCalls: 3,
      tools: [
        { name: 'fs.read' },
        { name: 'fs.replace' },
        { name: 'fs.write' },
        { name: 'shell.exec' },
      ],
      calls: [
        {
          toolName: 'fs.write',
          params: { path: 'src/app.py', content: 'working base', mode: 'overwrite' },
          reason: 'Initial implementation.',
          groundingRequired: true,
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q /tests/test_outputs.py' },
          reason: 'Run official verifier.',
          groundingRequired: true,
          success: false,
          error: 'line 52: broken assertion',
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/app.py', startLine: 45, endLine: 60 },
          reason: 'Inspect failing source.',
          groundingRequired: true,
          success: true,
          result: { content: 'broken assertion' },
        },
      ],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.replace',
        params: expect.objectContaining({
          path: 'src/app.py',
          oldText: 'broken assertion',
          newText: 'fixed assertion',
        }),
      }),
    ]);
  });

  it('executes only one targeted inspection after a verifier failure', async () => {
    const llm = new NoisyFailureInspectionPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'targeted-verifier-inspection-agent',
      goal: 'inspect only the causal failure location',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.list', 'fs.read', 'fs.replace', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Repair src/app.py and run the official tests.',
      executionRequired: true,
      round: 3,
      remainingCalls: 3,
      tools: [
        { name: 'fs.list' },
        { name: 'fs.read' },
        { name: 'fs.replace' },
        { name: 'shell.exec' },
      ],
      calls: [
        {
          toolName: 'fs.replace',
          params: { path: 'src/app.py', oldText: 'broken', newText: 'almost fixed' },
          reason: 'Apply initial repair.',
          groundingRequired: true,
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q /tests/test_outputs.py' },
          reason: 'Run official verifier.',
          groundingRequired: true,
          success: false,
          error: 'src/app.py:52 assertion failed',
        },
      ],
    });

    expect(llm.jsonCalls).toBe(1);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/app.py',
          startLine: 40,
          endLine: 70,
        },
      }),
    ]);
  });

  it('compacts long tool-planning tasks and propagates the planning deadline', async () => {
    const llm = new CapturingToolPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'bounded-tool-planner',
      goal: 'plan bounded evidence collection',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read'],
    });
    const longTask = `TASK_HEAD\n${'middle-context\n'.repeat(2_000)}TASK_TAIL`;

    await agent.planNextToolRound({
      task: longTask,
      round: 1,
      remainingCalls: 1,
      requestTimeoutMs: 1_234,
      tools: [{ name: 'fs.read' }],
      calls: [],
    });

    const userPrompt = llm.messages.find(message =>
      message.role === 'user' && message.content.includes('Tool observations:')
    )?.content ?? '';
    expect(userPrompt).toContain('TASK_HEAD');
    expect(userPrompt).toContain('TASK_TAIL');
    expect(userPrompt).toContain('[runtime_compacted_middle_for_tool_planning]');
    expect(userPrompt.length).toBeLessThan(22_000);
    expect(llm.options?.timeoutMs).toBeGreaterThan(0);
    expect(llm.options?.timeoutMs).toBeLessThanOrEqual(1_234);
  });

  it('keeps one detailed tool frontier without replaying every historical log', async () => {
    const llm = new CapturingToolPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'delta-observation-planner',
      goal: 'repair from bounded verifier evidence',
      llm,
      mode: 'hybrid',
      allowedTools: ['shell.exec'],
    });
    const calls = Array.from({ length: 10 }, (_, index) => ({
      toolName: 'shell.exec',
      params: { command: `pytest -q tests/test_${index}.py ${'x'.repeat(2_000)}` },
      reason: 'Run a verifier.',
      groundingRequired: true,
      success: false,
      error: `command failure ${'e'.repeat(10_000)}`,
      result: {
        command: `pytest -q tests/test_${index}.py`,
        stdout: `stdout ${'o'.repeat(10_000)}`,
        stderr: `stderr ${'s'.repeat(10_000)}`,
        exitCode: 1,
      },
    }));

    await agent.planNextToolRound({
      task: `TASK_HEAD\n${'requirements\n'.repeat(2_000)}LATEST_FAILURE`,
      round: 10,
      remainingCalls: 1,
      tools: [{ name: 'shell.exec' }],
      calls,
    });

    const userPrompt = llm.messages.find(message =>
      message.role === 'user' && message.content.includes('Tool observations:')
    )?.content ?? '';
    expect(userPrompt).toContain('TASK_HEAD');
    expect(userPrompt).toContain('LATEST_FAILURE');
    expect(userPrompt).toContain('earlier chars compacted');
    expect(userPrompt.length).toBeLessThan(38_000);
  });

  it('retains the newest failed verifier frontier after a later source read', async () => {
    const llm = new CapturingToolPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'causal-verifier-planner',
      goal: 'repair from the official verifier failure',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.replace', 'shell.exec'],
    });
    const verifierTail = `FAILED hidden dataset\n${'diagnostic-context-'.repeat(120)}`;

    await agent.planNextToolRound({
      task: 'Repair src/app.py and run the official tests.',
      round: 3,
      remainingCalls: 2,
      tools: [{ name: 'fs.read' }, { name: 'fs.replace' }, { name: 'shell.exec' }],
      calls: [
        {
          toolName: 'fs.replace',
          params: { path: 'src/app.py', oldText: 'broken', newText: 'almost fixed' },
          reason: 'Apply an initial repair.',
          groundingRequired: true,
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'pytest -q /tests/test_outputs.py' },
          reason: 'Run the official verifier.',
          groundingRequired: true,
          success: false,
          error: verifierTail,
          result: { stderr: verifierTail, exitCode: 1 },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/app.py', startLine: 40, endLine: 70 },
          reason: 'Inspect the reported source location.',
          groundingRequired: true,
          success: true,
          result: { content: 'almost fixed' },
        },
      ],
    });

    const userPrompt = llm.messages.find(message =>
      message.role === 'user' && message.content.includes('Tool observations:')
    )?.content ?? '';
    expect(userPrompt).toContain('FAILED hidden dataset');
    expect(userPrompt).toContain('diagnostic-context-diagnostic-context');
    expect(userPrompt).toContain('"latestVerificationFailed":true');
    expect(userPrompt).toContain('"inspectedAfterLatestFailure":true');
  });

  it('requires a new mutation after an unmet acceptance audit even when prior verification passed', async () => {
    const llm = new ReadOnlyThenMutationPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'acceptance-repair-planner',
      goal: 'repair unmet acceptance items',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.write', 'shell.exec'],
    });
    const calls = [
      {
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'placeholder' },
        reason: 'Write the initial implementation.',
        groundingRequired: true,
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'test -f artifact.txt' },
        reason: 'Run the declared command.',
        groundingRequired: true,
        success: true,
      },
      {
        toolName: 'fs.read',
        params: { path: 'artifact.txt' },
        reason: 'Acceptance audit observed placeholder content.',
        groundingRequired: true,
        success: true,
        result: { content: 'placeholder' },
      },
    ];

    const plans = await agent.planNextToolRound({
      task: '[runtime_execution_repair_phase]\nAcceptance audit failed: placeholder content remains.',
      executionRequired: true,
      requiredMutationAfterCallIndex: calls.length - 1,
      round: 3,
      remainingCalls: 2,
      tools: [{ name: 'fs.read' }, { name: 'fs.write' }, { name: 'shell.exec' }],
      calls,
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: { path: 'artifact.txt', content: 'implemented' },
      }),
    ]);
  });

  it('classifies a tool-planning request timeout for runtime telemetry', async () => {
    const llm = new CapturingToolPlanningLLM(new Error('Request timed out after 250ms'));
    const agent = new UnifiedAgent({
      name: 'timeout-tool-planner',
      goal: 'plan bounded evidence collection',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read'],
    });

    await expect(agent.planNextToolRound({
      task: 'Inspect the current file.',
      round: 1,
      remainingCalls: 1,
      requestTimeoutMs: 250,
      tools: [{ name: 'fs.read' }],
      calls: [],
    })).resolves.toEqual([]);
    expect(agent.getLastToolPlanningFailure()).toMatchObject({
      timedOut: true,
      message: 'Request timed out after 250ms',
    });
  });

  it('retries a truncated mutation plan with bounded chunking guidance', async () => {
    const llm = new TruncatedMutationPlanningLLM();
    const agent = new UnifiedAgent({
      name: 'chunked-mutation-planner',
      goal: 'implement a large source repair',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.read', 'fs.replace', 'fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Repair the project implementation and run its verification suite.',
      round: 1,
      remainingCalls: 4,
      requestTimeoutMs: 5_000,
      tools: [
        { name: 'fs.read' },
        { name: 'fs.replace' },
        { name: 'fs.write' },
        { name: 'shell.exec' },
      ],
      calls: [{
        toolName: 'fs.read',
        params: { path: 'src/implementation.py' },
        reason: 'Inspect the authoritative source.',
        groundingRequired: true,
        success: true,
        result: { content: 'broken implementation' },
      }],
    });

    expect(llm.jsonCalls).toBe(2);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: expect.objectContaining({
          path: 'src/implementation.py',
          content: 'first bounded chunk',
        }),
      }),
    ]);
    const firstSystemPrompt = llm.messagesByAttempt[0]
      ?.find(message => message.role === 'system')?.content ?? '';
    expect(firstSystemPrompt).toContain('Limit one fs.write/fs.replace payload to 6000 characters');
    expect(firstSystemPrompt).toContain('Do not embed multiline source');
    const retryPrompt = llm.messagesByAttempt[1]
      ?.findLast(message => message.role === 'user')?.content ?? '';
    expect(retryPrompt).toContain('one complete compact JSON object');
    expect(retryPrompt).toContain('mode=append');
    expect(llm.optionsByAttempt[0]?.maxTokens).toBe(4096);
    expect(llm.optionsByAttempt[1]?.maxTokens).toBe(8192);
    expect(agent.getLastToolPlanningFailure()).toBeUndefined();
  });

  it('executes a bounded prefix recovered from a truncated fs.write response', async () => {
    const llm = new RecoverableTruncatedWritePlanningLLM();
    const agent = new UnifiedAgent({
      name: 'recover-truncated-source-writer',
      goal: 'preserve generated implementation chunks',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement src/app.py and run verification.',
      executionRequired: true,
      round: 1,
      remainingCalls: 2,
      tools: [{ name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [],
    });

    expect(llm.jsonCalls).toBe(1);
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: {
          path: 'src/app.py',
          content: 'first line\nsecond line\n',
          mode: 'overwrite',
          createDirectories: true,
        },
        reason: expect.stringContaining('truncated structured fs.write'),
      }),
    ]);
    expect(agent.getLastToolPlanningFailure()).toBeUndefined();
  });

  it('appends the next recovered source chunk after an earlier recovered write', async () => {
    const llm = new RecoverableTruncatedWritePlanningLLM(
      'continuation helper line\\nnext partial'
    );
    const agent = new UnifiedAgent({
      name: 'continue-truncated-source-writer',
      goal: 'continue a recovered implementation',
      llm,
      mode: 'hybrid',
      allowedTools: ['fs.write', 'shell.exec'],
    });

    const plans = await agent.planNextToolRound({
      task: 'Implement src/app.py and run verification.',
      executionRequired: true,
      round: 2,
      remainingCalls: 2,
      tools: [{ name: 'fs.write' }, { name: 'shell.exec' }],
      calls: [{
        toolName: 'fs.write',
        params: {
          path: 'src/app.py',
          content: 'first line\nsecond line\n',
          mode: 'overwrite',
        },
        reason: 'Recovered a bounded source chunk from a truncated structured fs.write response.',
        groundingRequired: true,
        success: true,
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.write',
        params: expect.objectContaining({
          path: 'src/app.py',
          content: 'continuation helper line\n',
          mode: 'append',
        }),
      }),
    ]);
  });
});
