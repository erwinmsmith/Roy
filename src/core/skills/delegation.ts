import type { Runtime, RuntimeEvent, RootMediatedSpawnResult, SpawnCommandPayload, SubAgentArchetype, TokenUsage } from '../runtime/Runtime.js';
import type { Skill, SkillContext, SkillInput, SkillManifest, SkillOutput } from './types.js';

export interface DelegateToSubagentParams {
  parentId?: string;
  archetype: SubAgentArchetype;
  task: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  tools?: string[];
  budgetTokens?: number;
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
}

export interface DelegateToSubagentResult {
  correlationId: string;
  agentId: string;
  agentName: string;
  agentResult: RootMediatedSpawnResult['subagentResult'];
  rootSynthesis?: string;
  tokenUsage: {
    root: TokenUsage;
    subagent: TokenUsage;
    total: TokenUsage;
  };
  events: RuntimeEvent[];
}

export class DelegateToSubagentSkill implements Skill {
  readonly name = 'delegate_to_subagent';
  readonly description = 'Delegate a task to a root-managed subagent through the Roy runtime message chain.';
  readonly version = '0.1.0';

  constructor(private readonly getRuntime: () => Runtime) {}

  getManifest(): SkillManifest {
    return {
      name: this.name,
      version: this.version,
      description: this.description,
      tags: ['delegation', 'subagent', 'runtime', 'message-queue'],
    };
  }

  validate(input: SkillInput): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    const params = input.params as Partial<DelegateToSubagentParams>;

    if (!this.isArchetype(params.archetype)) {
      errors.push('archetype must be one of researcher, critic, planner, coder, summarizer, tester, or custom');
    }
    if (typeof params.task !== 'string' || params.task.trim().length === 0) {
      errors.push('task is required');
    }
    if (params.parentId !== undefined && typeof params.parentId !== 'string') {
      errors.push('parentId must be a string when provided');
    }
    if (params.name !== undefined && typeof params.name !== 'string') {
      errors.push('name must be a string when provided');
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
    };
  }

  async execute(input: SkillInput, _context: SkillContext): Promise<SkillOutput> {
    const validation = this.validate(input);
    if (!validation.valid) {
      return {
        success: false,
        error: `Validation failed: ${validation.errors?.join(', ')}`,
      };
    }

    const runtime = this.getRuntime();
    const params = input.params as unknown as DelegateToSubagentParams;
    const eventsBefore = runtime.getEvents().length;
    const result = await runtime.handleSpawnCommand(this.toSpawnCommand(params));
    const budget = runtime.getBudgetState();
    const rootUsage = budget.perAgent.root ?? this.zeroUsage();
    const subagentUsage = budget.perAgent[result.agent.identity.id] ?? this.zeroUsage();
    const totalUsage = this.addUsage(rootUsage, subagentUsage);

    const output: DelegateToSubagentResult = {
      correlationId: result.correlationId,
      agentId: result.agent.identity.id,
      agentName: result.agent.identity.name,
      agentResult: result.subagentResult,
      rootSynthesis: result.finalResponse,
      tokenUsage: {
        root: rootUsage,
        subagent: subagentUsage,
        total: totalUsage,
      },
      events: runtime.getEvents().slice(eventsBefore),
    };

    return {
      success: true,
      result: output,
      metadata: {
        skill: this.name,
        correlationId: result.correlationId,
        agentId: result.agent.identity.id,
        messageMediated: true,
      },
    };
  }

  private toSpawnCommand(params: DelegateToSubagentParams): SpawnCommandPayload {
    return {
      parentId: params.parentId,
      archetype: params.archetype,
      task: params.task,
      name: params.name,
      customRole: params.customRole,
      customStyle: params.customStyle,
      requireRootSynthesis: params.requireRootSynthesis ?? true,
      showSubagentOutput: params.showSubagentOutput ?? false,
    };
  }

  private isArchetype(value: unknown): value is SubAgentArchetype {
    return value === 'researcher'
      || value === 'critic'
      || value === 'planner'
      || value === 'coder'
      || value === 'summarizer'
      || value === 'tester'
      || value === 'custom';
  }

  private zeroUsage(): TokenUsage {
    return {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      thinkingTokens: null,
    };
  }

  private addUsage(left: TokenUsage, right: TokenUsage): TokenUsage {
    return {
      llmCalls: left.llmCalls + right.llmCalls,
      promptTokens: left.promptTokens + right.promptTokens,
      completionTokens: left.completionTokens + right.completionTokens,
      totalTokens: left.totalTokens + right.totalTokens,
      thinkingTokens: left.thinkingTokens === null && right.thinkingTokens === null
        ? null
        : Number(left.thinkingTokens ?? 0) + Number(right.thinkingTokens ?? 0),
    };
  }
}
