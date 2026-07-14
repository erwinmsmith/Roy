import type { ToMProfile } from '../agent/BaseAgent.js';
import type {
  AgentMemoryScope,
  AgentSpawnPolicy,
  RootMediatedSpawnResult,
  RuntimeEvent,
  SubAgentArchetype,
  TokenUsage,
} from '../runtime/Runtime.js';

export type AgentNodeReuseMode = 'prefer_cache' | 'require_cache' | 'fresh' | 'mutate_cache';
export type AgentNodeCreationMode = 'generated' | 'cache_hit' | 'custom' | 'mutated_from_cache';

export interface AgentNodeOutputContract {
  format: 'markdown' | 'json' | 'structured_report';
  requiredFields?: string[];
  groundingRequired?: boolean;
}

export interface AgentComputeNodeRequest {
  parentId?: string;
  archetype: SubAgentArchetype;
  task: string;
  name?: string;
  role?: string;
  style?: string;
  description?: string;
  tools?: string[];
  skills?: string[];
  budgetTokens?: number;
  memoryScope?: AgentMemoryScope;
  spawnPolicy?: Partial<AgentSpawnPolicy>;
  tomProfile?: ToMProfile;
  reuse?: {
    mode?: AgentNodeReuseMode;
    agentPatternId?: string;
    delegationPatternId?: string;
  };
  execution?: {
    requireParentSynthesis?: boolean;
    showSubagentOutput?: boolean;
    disableRecursiveDelegation?: boolean;
    teamId?: string;
  };
  outputContract?: AgentNodeOutputContract;
}

export interface AgentCreationInvocation {
  agentId: string;
  sessionId: string;
  source: string;
}

export interface AgentComputeNodeDefinition {
  nodeId: string;
  sessionId: string;
  correlationId: string;
  parentId: string;
  depth: number;
  definitionFingerprint: string;
  invocationFingerprint: string;
  identity: {
    archetype: SubAgentArchetype;
    name?: string;
    role: string;
    style?: string;
    description: string;
    tomProfile?: ToMProfile;
  };
  assignment: {
    task: string;
    outputContract: AgentNodeOutputContract;
  };
  capabilities: {
    tools: string[];
    skills: string[];
  };
  context: {
    memoryScope: AgentMemoryScope;
  };
  resources: {
    budgetTokens?: number;
  };
  governance: {
    spawnPolicy: AgentSpawnPolicy;
  };
  execution: {
    requireParentSynthesis: boolean;
    showSubagentOutput: boolean;
    disableRecursiveDelegation: boolean;
    teamId?: string;
  };
  reuse: {
    mode: AgentNodeReuseMode;
    creationMode: AgentNodeCreationMode;
    cacheHits: string[];
    agentPatternId?: string;
    delegationPatternId?: string;
  };
  source: string;
}

export interface AgentComputeNodeExecution {
  node: AgentComputeNodeDefinition;
  delegation: RootMediatedSpawnResult;
  tokenUsage: {
    root: TokenUsage;
    subagent: TokenUsage;
    total: TokenUsage;
  };
  events: RuntimeEvent[];
}

export interface AgentCreationGateway {
  createAgentComputeNode(
    request: AgentComputeNodeRequest,
    invocation: AgentCreationInvocation
  ): Promise<AgentComputeNodeExecution>;
}
