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
        calls: [{ toolName: 'shell.exec', params: { command: 'pytest -q' } }],
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
});
