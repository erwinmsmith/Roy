import type { RuntimeEvent, RootMediatedSpawnResult, SubAgentArchetype, TokenUsage } from '../runtime/Runtime.js';
import type {
  AgentComputeNodeDefinition,
  AgentComputeNodeRequest,
  AgentCreationGateway,
  AgentNodeOutputContract,
  AgentNodeReuseMode,
} from './agentCreation.js';
import type { Skill, SkillContext, SkillInput, SkillManifest, SkillOutput } from './types.js';

export interface DelegateToSubagentParams {
  parentId?: string;
  archetype: SubAgentArchetype;
  task: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  description?: string;
  tools?: string[];
  skills?: string[];
  budgetTokens?: number;
  memoryScope?: AgentComputeNodeRequest['memoryScope'];
  spawnPolicy?: AgentComputeNodeRequest['spawnPolicy'];
  tomProfile?: AgentComputeNodeRequest['tomProfile'];
  tomProfileMode?: AgentComputeNodeRequest['tomProfileMode'];
  cognitiveGapIds?: string[];
  existenceReason?: string;
  reuseMode?: AgentNodeReuseMode;
  outputContract?: AgentNodeOutputContract;
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
  lifecycle?: AgentComputeNodeRequest['lifecycle'];
}

export interface DelegateToSubagentResult {
  correlationId: string;
  agentId: string;
  agentName: string;
  node: AgentComputeNodeDefinition;
  creationUsage: RootMediatedSpawnResult['creationUsage'];
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
  readonly description = 'Compile and execute a parent-authorized child agent node through the Roy runtime message chain.';
  readonly version = '0.1.0';

  constructor(private readonly gateway: AgentCreationGateway) {}

  getManifest(): SkillManifest {
    return {
      name: this.name,
      version: this.version,
      description: this.description,
      tags: ['delegation', 'subagent', 'runtime', 'message-queue'],
      scope: 'system',
      permissions: ['agent.create', 'agent.delegate'],
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
    const lifecycleMode = params.lifecycle?.mode;
    if (lifecycleMode !== undefined
      && lifecycleMode !== 'adaptive'
      && lifecycleMode !== 'release'
      && lifecycleMode !== 'retain_session'
      && lifecycleMode !== 'persist') {
      errors.push('lifecycle.mode must be adaptive, release, retain_session, or persist');
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
    };
  }

  async execute(input: SkillInput, context: SkillContext): Promise<SkillOutput> {
    const validation = this.validate(input);
    if (!validation.valid) {
      return {
        success: false,
        error: `Validation failed: ${validation.errors?.join(', ')}`,
      };
    }

    const params = input.params as unknown as DelegateToSubagentParams;
    const execution = await this.gateway.createAgentComputeNode(this.toNodeRequest(params), {
      agentId: context.agentId,
      sessionId: context.sessionId,
      source: `skill:${this.name}`,
    });
    const result = execution.delegation;

    const output: DelegateToSubagentResult = {
      correlationId: result.correlationId,
      agentId: result.agent.identity.id,
      agentName: result.agent.identity.name,
      node: execution.node,
      creationUsage: result.creationUsage,
      agentResult: result.subagentResult,
      rootSynthesis: result.finalResponse,
      tokenUsage: execution.tokenUsage,
      events: execution.events,
    };

    return {
      success: true,
      result: output,
      metadata: {
        skill: this.name,
        correlationId: result.correlationId,
        agentId: result.agent.identity.id,
        nodeId: execution.node.nodeId,
        definitionFingerprint: execution.node.definitionFingerprint,
        messageMediated: true,
      },
    };
  }

  private toNodeRequest(params: DelegateToSubagentParams): AgentComputeNodeRequest {
    return {
      parentId: params.parentId,
      archetype: params.archetype,
      task: params.task,
      name: params.name,
      role: params.customRole,
      style: params.customStyle,
      description: params.description,
      tools: params.tools,
      skills: params.skills,
      budgetTokens: params.budgetTokens,
      memoryScope: params.memoryScope,
      spawnPolicy: params.spawnPolicy,
      tomProfile: params.tomProfile,
      tomProfileMode: params.tomProfileMode ?? 'definition_override',
      cognitiveGapIds: params.cognitiveGapIds,
      existenceReason: params.existenceReason,
      reuse: { mode: params.reuseMode ?? 'prefer_cache' },
      outputContract: params.outputContract,
      lifecycle: params.lifecycle,
      execution: {
        requireParentSynthesis: params.requireRootSynthesis ?? true,
        showSubagentOutput: params.showSubagentOutput ?? false,
      },
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

}
