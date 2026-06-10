// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { config } from '../../config/index.js';
import { logger } from '../utils/logger.js';
import { configureLogging, shutdownLogging } from '../logging/index.js';
import { llmFactory, type LLMProvider } from '../llm/index.js';
import { AgentManager } from './AgentManager.js';
import { FSM } from '../executor/FSM.js';
import { signalBus } from '../executor/SignalBus.js';
import { UnifiedAgent } from '../agent/UnifiedAgent.js';
import type { AgentInfo, AgentUsage, BaseAgent, ToMProfile } from '../agent/BaseAgent.js';
import { actionRegistry } from '../actions/index.js';
import { registerCoreTools, toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import { DelegateToSubagentSkill } from '../skills/delegation.js';
import { UseToolWhenNeededSkill } from '../skills/toolUse.js';
import { DefaultDelegationCandidatePlanner, type DelegationCandidateSelection } from '../delegation/index.js';
import {
  InMemoryMessageQueue,
  MessageScheduler,
  type EnqueueMessageInput,
  type MessageQueue,
  type QueueState,
  type QueueTransition,
  type RuntimeMessage,
} from '../queue/index.js';
import {
  WorkspaceMemoryManager,
  type ConversationEntry,
  type ConversationSessionState,
  type MemoryAutoState,
  type MemoryMode,
  type MemoryProposalSummary,
  type MemorySignals,
  type MemoryUpdateProposal,
  type MemoryUpdateRecord,
  type WorkspaceMemoryState,
  type WorkspaceRuntimeConfig,
  type RootMemoryContext,
} from '../memory/index.js';

export interface RuntimeConfig {
  agentName?: string;
  agentGoal?: string;
  sessionId?: string;
  fsmEnabled?: boolean;
  budget?: number;
  mode?: 'conversational' | 'action' | 'hybrid';
  llmProvider?: LLMProvider;
  workspaceCwd?: string;
}

export interface RuntimeContext {
  config: typeof config;
  llm: LLMProvider | null;
  fsm: FSM;
  signalBus: typeof signalBus;
  manager: AgentManager;
  agent: UnifiedAgent;
  sessionId: string;
  queue: MessageQueue;
  scheduler: MessageScheduler;
  memory: WorkspaceMemoryManager;
  capabilities: {
    skills: number;
    actions: number;
    tools: number;
  };
}

export interface TokenUsage extends AgentUsage {
  thinkingTokens: number | null;
  estimatedCostUsd?: number;
}

export interface BudgetState {
  mode: 'unlimited' | 'limited';
  limitTokens?: number;
  usedTokens: number;
  remainingTokens?: number;
  perAgent: Record<string, TokenUsage>;
  perTurn: TokenUsage[];
}

export interface RuntimeEvent {
  type: string;
  timestamp: number;
  agentId?: string;
  data?: Record<string, unknown>;
}

export interface RuntimeState {
  sessionId: string;
  rootAgentId: string;
  rootAgent: AgentInfo;
  agents: AgentInfo[];
  events: RuntimeEvent[];
  budget: BudgetState;
}

export type SubAgentArchetype =
  | 'researcher'
  | 'critic'
  | 'planner'
  | 'coder'
  | 'summarizer'
  | 'tester'
  | 'custom';

export interface ToolBinding {
  name: string;
  enabled: boolean;
  permission: 'read_only' | 'write' | 'execute';
  constraints?: {
    allowedPaths?: string[];
    blockedPaths?: string[];
    allowlistedCommands?: string[];
    maxCalls?: number;
  };
}

export interface SkillBinding {
  name: string;
  enabled: boolean;
  description: string;
  constraints?: {
    maxCalls?: number;
    requiresApproval?: boolean;
  };
}

export interface AgentSpawnPolicy {
  canSpawn: boolean;
  maxChildren: number;
  maxDepth: number;
  maxTotalAgentsPerTurn: number;
  allowCustomAgents: boolean;
  budgetAware: boolean;
  allowedStates: string[];
}

export interface AgentMemoryScope {
  public: boolean;
  private: boolean;
  parentContext: boolean;
  sessionWindowTurns: number;
}

export interface SpawnAgentSpec {
  parentId: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  archetype: SubAgentArchetype;
  tomLevel: number;
  description: string;
  task?: string;
  tools?: string[] | ToolBinding[];
  skills?: string[] | SkillBinding[];
  memoryScope?: AgentMemoryScope;
  spawnPolicy?: Partial<AgentSpawnPolicy>;
  budgetTokens?: number;
  systemPrompt?: string;
  correlationId?: string;
}

export interface AgentTreeNode {
  agent: AgentInfo;
  children: AgentTreeNode[];
}

export interface RunAgentResult {
  agent: AgentInfo;
  result: string;
  usage: TokenUsage;
  toolCalls: ToolCallRecord[];
  evidence: RunEvidence;
  grounded: boolean;
  warnings: string[];
}

export interface RunEvidence {
  toolGrounded: boolean;
  outputGrounded: boolean;
  observedPaths: string[];
  toolResultSummary?: string;
}

export interface ToolCallRecord {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  success: boolean;
}

export interface SpawnCommandPayload {
  archetype: SubAgentArchetype;
  task: string;
  parentId?: string;
  correlationId?: string;
  source?: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  tools?: string[];
  skills?: string[];
  tomLevel?: number;
  budgetTokens?: number;
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
  disableRecursiveDelegation?: boolean;
}

export type DelegationDecision =
  | {
      action: 'solve_directly';
      reason: string;
    }
  | {
      action: 'ask_clarification';
      reason: string;
      question: string;
    }
  | {
      action: 'spawn_subagents';
      reason: string;
      agents: DelegationAgentPlan[];
    };

export interface DelegationAgentPlan {
  archetype: SubAgentArchetype;
  name?: string;
  task: string;
  tools?: string[];
  skills?: string[];
  tomLevel?: number;
  budgetTokens?: number;
}

export interface RootMediatedSpawnResult {
  correlationId: string;
  agent: AgentInfo;
  subagentResult: RunAgentResult;
  finalResponse: string;
  messages: RuntimeMessage[];
  creationUsage: AgentCreationUsage;
}

export interface RootTurnResult {
  correlationId: string;
  decision: DelegationDecision;
  finalResponse: string;
  subagents: RootMediatedSpawnResult[];
  messages: RuntimeMessage[];
  usage: {
    root: TokenUsage;
    subagents: Record<string, TokenUsage>;
    total: TokenUsage;
  };
}

export interface AgentCreationUsage {
  mode: 'generated' | 'cache_hit';
  patternIds: string[];
  cacheHits: string[];
  definitionTokens: number;
  renderedPromptTokens: number;
  renderedPromptChars: number;
}

export interface AgentBindingState {
  tools: ToolBinding[];
  skills: SkillBinding[];
  memoryScope: AgentMemoryScope;
  spawnPolicy: AgentSpawnPolicy;
}

export interface AgentPolicyView extends AgentBindingState {
  agentId: string;
  parentId?: string;
  depth: number;
  currentChildren: number;
  allowedChildren: number;
}

export interface AgentArchetypeProfile {
  archetype: SubAgentArchetype;
  tools: ToolBinding[];
  skills: SkillBinding[];
  spawnPolicy: AgentSpawnPolicy;
}

export class Runtime {
  private static instance: Runtime | null = null;

  private ctx: RuntimeContext | null = null;
  private initialized = false;
  private events: RuntimeEvent[] = [];
  private perTurnUsage: TokenUsage[] = [];
  private agentSequence = 0;
  private delegationSequence = 0;
  private queue: MessageQueue | null = null;
  private scheduler: MessageScheduler | null = null;
  private memory: WorkspaceMemoryManager | null = null;
  private agentBindings = new Map<string, AgentBindingState>();
  private workspaceRuntimeConfig: WorkspaceRuntimeConfig | null = null;
  private readonly candidatePlanner = new DefaultDelegationCandidatePlanner();
  private turnAgentCounts = new Map<string, number>();

  static getInstance(): Runtime {
    if (!Runtime.instance) {
      Runtime.instance = new Runtime();
    }
    return Runtime.instance;
  }

  async initialize(options: RuntimeConfig = {}): Promise<RuntimeContext> {
    if (this.initialized && this.ctx) {
      return this.ctx;
    }

    const startTime = Date.now();

    // Initialize logger
    const logLevel = config.logger?.level ?? 'info';
    logger.setLevel(logLevel as 'debug' | 'info' | 'warn' | 'error');
    await configureLogging();
    logger.info('Runtime initializing...');

    // Create LLM provider
    const llm = options.llmProvider ?? this.createLLMProvider();
    if (llm) {
      logger.info(`LLM provider: ${llm.name}, model: ${llm.defaultModel}`);
    } else {
      logger.warn('No LLM provider configured - agent will have limited functionality');
    }

    // Create FSM
    const fsm = new FSM({
      initialState: 'S_solo',
      signalBus,
      onTransition: (from, to) => {
        logger.debug(`FSM transition: ${from} -> ${to}`);
        signalBus.emit('fsm:transition', { from, to });
        this.emit({ type: 'fsm.transition', agentId: 'root', data: { from, to } });
      },
      onStateChange: (state) => {
        logger.debug(`FSM state: ${state}`);
        signalBus.emit('fsm:stateChange', { state });
        this.emit({ type: 'fsm.state.changed', agentId: 'root', data: { state } });
      },
    });

    if (options.budget !== undefined) {
      fsm.setBudget(options.budget);
    }

    // Create AgentManager
    const manager = new AgentManager();
    registerCoreTools();
    this.registerCoreSkills();
    const memory = new WorkspaceMemoryManager();
    await memory.initWorkspace(options.workspaceCwd ?? process.cwd(), options.sessionId ?? 'main');
    this.workspaceRuntimeConfig = await memory.getWorkspaceConfig();
    const rootMemory = await memory.loadAgentMemory('roy');
    const rootContext = await memory.loadRootContext();
    const queue = new InMemoryMessageQueue(transition => this.handleQueueTransition(transition));
    const scheduler = new MessageScheduler(queue);

    // Create unified agent
    const agentName = options.agentName ?? 'Roy';
    const agentGoal = options.agentGoal ?? this.buildAgentPromptFromMemory({
      name: agentName,
      role: 'root',
      parentName: 'none',
      task: 'Operate as the root agent for the current Roy runtime session.',
      description: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      bundle: rootMemory,
      publicContext: this.formatPublicContext(rootContext),
      tomProfile: this.createRootToMProfile(),
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
    });

    const agent = new UnifiedAgent({
      name: agentName,
      goal: agentGoal,
      llm: llm ?? undefined,  // Convert null to undefined for agent
      fsm: options.fsmEnabled !== false ? fsm : undefined,
      id: 'root',
      role: 'root',
      generation: 0,
      tomLevel: 1,
      tomProfile: this.createRootToMProfile(),
      description: 'Root agent of the Roy autonomous agent system',
      mode: options.mode ?? 'hybrid',
    });

    logger.info(`Agent created: ${agentName} in ${options.mode ?? 'hybrid'} mode`);

    // Register capabilities with agent
    const capabilities = this.registerCapabilities(agent);
    this.agentBindings.set('root', {
      tools: this.getRootToolBindings(),
      skills: this.getRootSkillBindings(),
      memoryScope: this.getDefaultMemoryScope('root'),
      spawnPolicy: this.getDefaultSpawnPolicy('root'),
    });

    // Add agent to manager
    manager.addAgent(agent);
    manager.setInteractWithEnv(agentName);

    // Create main session
    const sessionId = options.sessionId ?? 'main';
    manager.createSession(sessionId);

    const elapsed = Date.now() - startTime;
    logger.info(`Runtime initialized in ${elapsed}ms`);

    this.ctx = {
      config,
      llm,
      fsm,
      signalBus,
      manager,
      agent,
      sessionId,
      queue,
      scheduler,
      memory,
      capabilities,
    };
    this.queue = queue;
    this.scheduler = scheduler;
    this.memory = memory;

    this.initialized = true;
    this.emit({ type: 'runtime.initialized', agentId: 'root', data: { sessionId, provider: llm?.name ?? null } });
    return this.ctx;
  }

  getContext(): RuntimeContext {
    if (!this.ctx) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }
    return this.ctx;
  }

  createSession(sessionId: string): void {
    if (!this.ctx) {
      throw new Error('Runtime not initialized');
    }
    this.ctx.manager.createSession(sessionId);
  }

  async shutdown(): Promise<void> {
    if (!this.ctx) return;

    logger.info('Runtime shutting down...');

    for (const sessionId of this.ctx.manager.listSessions()) {
      await this.ctx.manager.closeSession(sessionId);
    }

    await shutdownLogging();

    this.ctx = null;
    this.queue = null;
    this.scheduler = null;
    this.memory = null;
    this.workspaceRuntimeConfig = null;
    this.agentBindings.clear();
    this.turnAgentCounts.clear();
    this.initialized = false;
    logger.info('Runtime shutdown complete');
  }

  private createLLMProvider(): LLMProvider | null {
    const llmConfig = config.llm;

    try {
      let provider: LLMProvider;

      if (llmConfig?.provider === 'anthropic') {
        provider = llmFactory.get('anthropic')!;
      } else if (llmConfig?.provider === 'openai') {
        provider = llmFactory.get('openai')!;
      } else if (llmConfig?.provider === 'deepseek') {
        provider = llmFactory.get('deepseek')!;
      } else {
        provider = llmFactory.getDefault();
      }

      if (provider?.isConfigured()) {
        return provider;
      }

      logger.warn('LLM provider not configured');
      return null;
    } catch (error) {
      logger.error('Failed to create LLM provider:', error);
      return null;
    }
  }

  private registerCapabilities(agent: UnifiedAgent, toolNames?: string[]): RuntimeContext['capabilities'] {
    // Register actions
    const actions = actionRegistry.list();
    for (const action of actions) {
      agent.registerAction(action);
      logger.debug(`Registered action: ${action.name}`);
    }

    // Register tools
    const allowedToolNames = toolNames ? new Set(toolNames) : undefined;
    const tools = toolRegistry.list().filter(tool => !allowedToolNames || allowedToolNames.has(tool.name));
    for (const tool of tools) {
      agent.registerTool(tool);
      logger.debug(`Registered tool: ${tool.name}`);
    }

    // Skills are executed via skillRegistry
    const skills = skillRegistry.list();
    logger.debug(`Available ${skills.length} skills`);

    return {
      skills: skills.length,
      actions: actions.length,
      tools: tools.length,
    };
  }

  private registerCoreSkills(): void {
    if (!skillRegistry.has('delegate_to_subagent')) {
      skillRegistry.register(new DelegateToSubagentSkill(() => this));
    }
    if (!skillRegistry.has('use_tool_when_needed')) {
      skillRegistry.register(new UseToolWhenNeededSkill());
    }
  }

  getAgentArchetypeProfiles(): AgentArchetypeProfile[] {
    const archetypes: SubAgentArchetype[] = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'];
    return archetypes.map(archetype => ({
      archetype,
      tools: this.getDefaultToolBindings(archetype),
      skills: this.getDefaultSkillBindings(archetype),
      spawnPolicy: this.getDefaultSpawnPolicy('subagent', archetype),
    }));
  }

  getAgentPolicy(agentId: string): AgentPolicyView | undefined {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) return undefined;
    const identity = agent.getIdentity();
    const bindings = this.agentBindings.get(agentId) ?? {
      tools: [],
      skills: [],
      memoryScope: this.getDefaultMemoryScope(identity.role),
      spawnPolicy: this.getDefaultSpawnPolicy(identity.role === 'root' ? 'root' : 'subagent'),
    };
    const depth = this.getAgentDepth(agentId);
    return {
      ...bindings,
      agentId,
      parentId: identity.parentId,
      depth,
      currentChildren: this.getChildren(agentId).length,
      allowedChildren: this.computeAllowedChildren(bindings.spawnPolicy),
    };
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  emit(event: Omit<RuntimeEvent, 'timestamp'>): RuntimeEvent {
    const runtimeEvent: RuntimeEvent = {
      ...event,
      timestamp: Date.now(),
    };
    this.events.push(runtimeEvent);
    if (this.events.length > 500) {
      this.events = this.events.slice(-500);
    }
    void this.memory?.writeTrace(runtimeEvent).catch(error => {
      logger.warn(`Failed to write runtime trace: ${error instanceof Error ? error.message : String(error)}`);
    });
    return runtimeEvent;
  }

  recordTurnUsage(usage: AgentUsage): void {
    this.perTurnUsage.push(this.toTokenUsage(usage));
  }

  getState(): RuntimeState {
    const ctx = this.getContext();
    const agents = ctx.manager.listAgentInfo();
    const rootAgent = ctx.agent.getInfo();
    return {
      sessionId: ctx.sessionId,
      rootAgentId: rootAgent.identity.id,
      rootAgent,
      agents,
      events: this.getEvents(),
      budget: this.getBudgetState(),
    };
  }

  getEvents(): RuntimeEvent[] {
    return [...this.events];
  }

  getBudgetState(): BudgetState {
    const ctx = this.getContext();
    const fsmCtx = ctx.fsm.getContext();
    const agents = ctx.manager.listAgentInfo();
    const perAgent: Record<string, TokenUsage> = {};
    let usedTokens = 0;

    for (const agent of agents) {
      const usage = this.toTokenUsage(agent.usage);
      perAgent[agent.identity.id] = usage;
      usedTokens += usage.totalTokens;
    }

    return {
      mode: fsmCtx.budget === null ? 'unlimited' : 'limited',
      limitTokens: fsmCtx.budget ?? undefined,
      usedTokens,
      remainingTokens: fsmCtx.budget === null ? undefined : Math.max(0, fsmCtx.budget - usedTokens),
      perAgent,
      perTurn: [...this.perTurnUsage],
    };
  }

  setBudget(limitTokens: number | null): BudgetState {
    const ctx = this.getContext();
    if (limitTokens === null) {
      ctx.fsm.clearBudget();
      this.emit({ type: 'budget.updated', data: { mode: 'unlimited' } });
    } else {
      ctx.fsm.setBudget(limitTokens);
      this.emit({ type: 'budget.updated', data: { mode: 'limited', limitTokens } });
    }
    return this.getBudgetState();
  }

  private getRootToolBindings(): ToolBinding[] {
    return toolRegistry.list().map(tool => this.createToolBinding(tool.name));
  }

  private getRootSkillBindings(): SkillBinding[] {
    return skillRegistry.list().map(skill => this.createSkillBinding(skill.name));
  }

  private getDefaultToolBindings(archetype: SubAgentArchetype): ToolBinding[] {
    const namesByArchetype: Record<SubAgentArchetype, string[]> = {
      researcher: ['fs.list', 'fs.read'],
      critic: ['fs.read'],
      planner: [],
      coder: ['fs.read', 'shell.exec'],
      summarizer: [],
      tester: ['fs.read', 'shell.exec'],
      custom: [],
    };
    const configured = this.workspaceRuntimeConfig?.agents.defaultToolsByArchetype[archetype];
    return (configured ?? namesByArchetype[archetype]).map(name => this.createToolBinding(name));
  }

  private getDefaultSkillBindings(archetype: SubAgentArchetype): SkillBinding[] {
    const namesByArchetype: Record<SubAgentArchetype, string[]> = {
      researcher: ['use_tool_when_needed', 'delegate_to_subagent'],
      critic: ['use_tool_when_needed', 'delegate_to_subagent'],
      planner: ['delegate_to_subagent'],
      coder: ['use_tool_when_needed', 'delegate_to_subagent'],
      summarizer: [],
      tester: ['use_tool_when_needed', 'delegate_to_subagent'],
      custom: [],
    };
    const configured = this.workspaceRuntimeConfig?.agents.defaultSkillsByArchetype[archetype];
    return (configured ?? namesByArchetype[archetype]).map(name => this.createSkillBinding(name));
  }

  private createToolBinding(name: string): ToolBinding {
    const permission: ToolBinding['permission'] = name === 'shell.exec'
      ? 'execute'
      : name === 'fs.write'
        ? 'write'
        : 'read_only';
    return {
      name,
      enabled: true,
      permission,
      constraints: name === 'shell.exec'
        ? {
            allowlistedCommands: ['npm', 'node', 'git', 'pwd', 'ls', 'cat', 'rg', 'sed'],
            maxCalls: 5,
          }
        : { allowedPaths: [process.cwd()], maxCalls: 20 },
    };
  }

  private createSkillBinding(name: string): SkillBinding {
    const skill = skillRegistry.get(name);
    return {
      name,
      enabled: true,
      description: skill?.description ?? name,
      constraints: name === 'delegate_to_subagent'
        ? { maxCalls: 5, requiresApproval: false }
        : undefined,
    };
  }

  private normalizeToolBindings(input: SpawnAgentSpec['tools'] | SpawnCommandPayload['tools'], archetype: SubAgentArchetype): ToolBinding[] {
    const raw = input ?? this.getDefaultToolBindings(archetype);
    return raw.map(item => typeof item === 'string' ? this.createToolBinding(item) : item);
  }

  private normalizeSkillBindings(input: SpawnAgentSpec['skills'] | SpawnCommandPayload['skills'], archetype: SubAgentArchetype): SkillBinding[] {
    const raw = input ?? this.getDefaultSkillBindings(archetype);
    return raw.map(item => typeof item === 'string' ? this.createSkillBinding(item) : item);
  }

  private getDefaultSpawnPolicy(role: 'root' | 'subagent' | string, archetype?: SubAgentArchetype): AgentSpawnPolicy {
    const isRoot = role === 'root';
    const delegation = this.workspaceRuntimeConfig?.delegation;
    const archetypeSkills = archetype ? this.getDefaultSkillBindings(archetype).map(binding => binding.name) : [];
    const canSpawn = delegation?.enabled !== false && (isRoot || archetypeSkills.includes('delegate_to_subagent'));
    return {
      canSpawn,
      maxChildren: delegation?.maxChildrenPerParent ?? 5,
      maxDepth: delegation?.maxDepth ?? 3,
      maxTotalAgentsPerTurn: delegation?.maxTotalAgentsPerTurn ?? 10,
      allowCustomAgents: isRoot && (delegation?.allowCustomAgents ?? true),
      budgetAware: delegation?.budgetAware ?? true,
      allowedStates: ['idle', 'thinking', 'waiting', 'done'],
    };
  }

  private getDefaultMemoryScope(role: string): AgentMemoryScope {
    return {
      public: true,
      private: true,
      parentContext: role !== 'root',
      sessionWindowTurns: role === 'root' ? 10 : 5,
    };
  }

  private mergeSpawnPolicy(base: AgentSpawnPolicy, override?: Partial<AgentSpawnPolicy>): AgentSpawnPolicy {
    return {
      ...base,
      ...override,
      allowedStates: override?.allowedStates ?? base.allowedStates,
    };
  }

  private computeAllowedChildren(policy: AgentSpawnPolicy): number {
    if (!policy.canSpawn) return 0;
    const budget = this.getBudgetState();
    if (!policy.budgetAware || budget.mode === 'unlimited') return policy.maxChildren;
    const remaining = budget.remainingTokens ?? 0;
    if (remaining < 1000) return 0;
    if (remaining < 3000) return Math.min(policy.maxChildren, 1);
    if (remaining < 8000) return Math.min(policy.maxChildren, 2);
    return policy.maxChildren;
  }

  private getAgentDepth(agentId: string): number {
    const ctx = this.getContext();
    let depth = 0;
    let current = ctx.manager.getAgentById(agentId)?.getIdentity().parentId;
    while (current) {
      depth += 1;
      current = ctx.manager.getAgentById(current)?.getIdentity().parentId;
    }
    return depth;
  }

  private getTurnAgentCount(correlationId?: string): number {
    return correlationId ? this.turnAgentCounts.get(correlationId) ?? 0 : 0;
  }

  private getMaxTotalAgentsPerTurn(parentId: string): number {
    return this.getAgentPolicy(parentId)?.spawnPolicy.maxTotalAgentsPerTurn
      ?? this.workspaceRuntimeConfig?.delegation.maxTotalAgentsPerTurn
      ?? 10;
  }

  private getRemainingTotalAgentsForTurn(parentId: string, correlationId?: string): number {
    return Math.max(0, this.getMaxTotalAgentsPerTurn(parentId) - this.getTurnAgentCount(correlationId));
  }

  private recordTurnAgentCreated(correlationId?: string): void {
    if (!correlationId) return;
    this.turnAgentCounts.set(correlationId, this.getTurnAgentCount(correlationId) + 1);
  }

  private validateSpawnPolicy(input: {
    parentId: string;
    archetype: SubAgentArchetype;
    tools: ToolBinding[];
    skills: SkillBinding[];
    correlationId?: string;
  }): {
    allowed: boolean;
    reason?: string;
    currentChildren: number;
    allowedChildren: number;
    depth: number;
  } {
    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(input.parentId);
    if (!parent) {
      return { allowed: false, reason: 'parent_not_found', currentChildren: 0, allowedChildren: 0, depth: 0 };
    }

    const parentInfo = parent.getInfo();
    const parentBindings = this.agentBindings.get(input.parentId) ?? {
      tools: [],
      skills: [],
      memoryScope: this.getDefaultMemoryScope(parentInfo.role),
      spawnPolicy: this.getDefaultSpawnPolicy(parentInfo.role === 'root' ? 'root' : 'subagent'),
    };
    const currentChildren = this.getChildren(input.parentId).length;
    const allowedChildren = this.computeAllowedChildren(parentBindings.spawnPolicy);
    const depth = this.getAgentDepth(input.parentId);
    const nextDepth = depth + 1;

    if (!parentBindings.spawnPolicy.canSpawn) {
      return { allowed: false, reason: 'spawn_disabled_for_parent', currentChildren, allowedChildren, depth };
    }
    if (!parentBindings.spawnPolicy.allowedStates.includes(parentInfo.state)) {
      return { allowed: false, reason: 'invalid_fsm_state', currentChildren, allowedChildren, depth };
    }
    if (currentChildren >= allowedChildren) {
      return { allowed: false, reason: 'max_children_exceeded', currentChildren, allowedChildren, depth };
    }
    if (this.getRemainingTotalAgentsForTurn(input.parentId, input.correlationId) <= 0) {
      return { allowed: false, reason: 'max_total_agents_per_turn_exceeded', currentChildren, allowedChildren, depth };
    }
    if (nextDepth > parentBindings.spawnPolicy.maxDepth) {
      return { allowed: false, reason: 'max_depth_exceeded', currentChildren, allowedChildren, depth };
    }
    if (input.archetype === 'custom' && !parentBindings.spawnPolicy.allowCustomAgents) {
      return { allowed: false, reason: 'custom_agents_not_allowed', currentChildren, allowedChildren, depth };
    }
    for (const binding of input.tools) {
      if (!toolRegistry.has(binding.name)) {
        return { allowed: false, reason: `tool_not_registered:${binding.name}`, currentChildren, allowedChildren, depth };
      }
    }
    for (const binding of input.skills) {
      if (!skillRegistry.has(binding.name)) {
        return { allowed: false, reason: `skill_not_registered:${binding.name}`, currentChildren, allowedChildren, depth };
      }
    }

    return { allowed: true, currentChildren, allowedChildren, depth };
  }

  private safeAgentKey(value: string): string {
    const key = value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return key || 'custom';
  }

  private inferAgentArchetype(agent: AgentInfo): SubAgentArchetype {
    const id = agent.identity.id.toLowerCase();
    const name = agent.identity.name.toLowerCase();
    const text = `${id} ${name}`;
    const archetypes: SubAgentArchetype[] = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester'];
    return archetypes.find(archetype => text.includes(archetype)) ?? 'custom';
  }

  async enqueueMessage<TPayload>(message: EnqueueMessageInput<TPayload>): Promise<RuntimeMessage<TPayload>> {
    const ctx = this.getContext();
    return ctx.queue.enqueue(message);
  }

  async getQueueState(limit = 20): Promise<QueueState> {
    const ctx = this.getContext();
    const [stats, recent] = await Promise.all([
      ctx.queue.getStats(),
      ctx.queue.listMessages({ limit }),
    ]);

    return { stats, recent };
  }

  async getMessages(filter: { correlationId?: string; limit?: number } = {}): Promise<RuntimeMessage[]> {
    const ctx = this.getContext();
    const messages = await ctx.queue.listMessages({ limit: filter.limit });
    return filter.correlationId
      ? messages.filter(message => message.correlationId === filter.correlationId)
      : messages;
  }

  async getMemoryState(): Promise<WorkspaceMemoryState> {
    const ctx = this.getContext();
    return ctx.memory.getState();
  }

  async loadRootMemoryContext(): Promise<RootMemoryContext> {
    const ctx = this.getContext();
    return ctx.memory.loadRootContext();
  }

  async listTraces(): Promise<Array<{ name: string; path: string; size: number; updatedAt: number }>> {
    const ctx = this.getContext();
    return ctx.memory.listTraces();
  }

  async readTrace(name = 'latest', limit = 50): Promise<RuntimeEvent[]> {
    const ctx = this.getContext();
    return ctx.memory.readTrace(name, limit);
  }

  async readPublicMemoryDoc(name: string): Promise<string> {
    const ctx = this.getContext();
    return ctx.memory.readPublicDoc(name);
  }

  async readAgentMemoryDoc(agentKey: string, doc = 'memory'): Promise<string> {
    const ctx = this.getContext();
    return ctx.memory.readAgentDoc(agentKey, doc);
  }

  async getMemoryMode(): Promise<MemoryMode> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryMode();
  }

  async setMemoryMode(mode: MemoryMode): Promise<MemoryMode> {
    const ctx = this.getContext();
    const next = await ctx.memory.setMemoryMode(mode);
    this.emit({ type: 'memory.mode.changed', data: { mode: next } });
    return next;
  }

  async listMemoryProposals(): Promise<MemoryUpdateProposal[]> {
    const ctx = this.getContext();
    return ctx.memory.listMemoryProposals();
  }

  async getMemoryProposal(id: string): Promise<MemoryUpdateProposal | undefined> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryProposal(id);
  }

  async proposeMemoryUpdates(source = 'manual'): Promise<MemoryUpdateProposal[]> {
    const ctx = this.getContext();
    this.emit({ type: 'memory.update.propose.started', agentId: 'root', data: { source } });
    const signals = await ctx.memory.collectMemorySignals();
    this.emit({
      type: 'memory.signals.collected',
      agentId: 'root',
      data: {
        source,
        sessionId: signals.source.sessionId,
        agentResults: signals.counts.agentResults,
        rootFinalResponses: signals.counts.rootFinalResponses,
        toolCalls: signals.toolCalls.length,
        outputGrounded: signals.agents.filter(agent => agent.outputGrounded).length,
        candidateSignals: signals.candidateSignals,
      },
    });
    const proposals = await ctx.memory.proposeMemoryUpdates();
    for (const proposal of proposals) {
      this.emit({
        type: 'memory.proposal.created',
        agentId: proposal.target.type === 'agent' ? proposal.target.key : 'root',
        data: {
          source,
          proposalId: proposal.id,
          target: proposal.target.path,
          section: proposal.target.section,
          risk: proposal.risk,
          confidence: proposal.confidence,
        },
      });
    }
    if (proposals.length === 0) {
      this.emit({
        type: 'memory.update.skipped',
        agentId: 'root',
        data: {
          reason: signals.candidateSignals.length === 0 ? 'no_signals' : 'no_new_proposals',
          source,
          signalCounts: signals.counts,
          candidateSignals: signals.candidateSignals,
        },
      });
    }
    const records = await ctx.memory.listAllMemoryProposalRecords();
    const summary: MemoryProposalSummary = {
      createdThisRun: proposals.length,
      skippedDuplicates: proposals.length === 0 ? signals.candidateSignals.length : 0,
      updatedPendingProposals: 0,
      pendingProposals: records.filter(record => record.status === 'pending').length,
      alreadyCommitted: records.filter(record => record.status === 'committed').length,
    };
    await ctx.memory.recordAutoPropose(source, summary, proposals.length === 0 ? 'no_new_proposals' : undefined);
    this.emit({
      type: 'memory.update.propose.completed',
      agentId: 'root',
      data: {
        source,
        created: proposals.length,
        updated: summary.updatedPendingProposals,
        skippedDuplicates: summary.skippedDuplicates,
        pending: summary.pendingProposals,
        committed: summary.alreadyCommitted,
      },
    });
    return proposals;
  }

  async summarizeMemoryUpdates(source = 'manual'): Promise<MemoryProposalSummary> {
    const ctx = this.getContext();
    this.emit({ type: 'memory.update.propose.started', agentId: 'root', data: { source } });
    const summary = await ctx.memory.summarizeMemoryUpdates();
    await ctx.memory.recordAutoPropose(source, summary, summary.createdThisRun === 0 ? 'no_new_proposals' : undefined);
    this.emit({
      type: summary.createdThisRun > 0 ? 'memory.update.propose.completed' : 'memory.update.skipped',
      agentId: 'root',
      data: {
        createdThisRun: summary.createdThisRun,
        skippedDuplicates: summary.skippedDuplicates,
        updatedPendingProposals: summary.updatedPendingProposals,
        pendingProposals: summary.pendingProposals,
        alreadyCommitted: summary.alreadyCommitted,
        source,
        reason: summary.createdThisRun === 0 ? 'no_new_proposals' : undefined,
      },
    });
    return summary;
  }

  async collectMemorySignals(): Promise<MemorySignals> {
    const ctx = this.getContext();
    return ctx.memory.collectMemorySignals();
  }

  async getMemoryAutoState(): Promise<MemoryAutoState> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryAutoState();
  }

  async getCachePatterns(kind: 'agents' | 'delegations' | 'teams'): Promise<Array<Record<string, unknown>>> {
    const ctx = this.getContext();
    return ctx.memory.getCachePatterns(kind);
  }

  async acceptMemoryProposal(id: string): Promise<MemoryUpdateRecord | undefined> {
    const ctx = this.getContext();
    const record = await ctx.memory.acceptMemoryProposal(id);
    this.emit({
      type: record ? 'memory.update.committed' : 'memory.update.skipped',
      agentId: 'root',
      data: { proposalId: id, target: record?.targetPath },
    });
    return record;
  }

  async rejectMemoryProposal(id: string): Promise<boolean> {
    const ctx = this.getContext();
    const rejected = await ctx.memory.rejectMemoryProposal(id);
    this.emit({
      type: rejected ? 'memory.update.rejected' : 'memory.update.skipped',
      agentId: 'root',
      data: { proposalId: id },
    });
    return rejected;
  }

  async listMemoryUpdates(): Promise<MemoryUpdateRecord[]> {
    const ctx = this.getContext();
    return ctx.memory.listMemoryUpdates();
  }

  async recordConversation(entry: Omit<ConversationEntry, 'id' | 'timestamp' | 'sessionId'> & { sessionId?: string }): Promise<ConversationEntry> {
    const ctx = this.getContext();
    return ctx.memory.appendConversation({
      ...entry,
      sessionId: entry.sessionId ?? ctx.sessionId,
    });
  }

  async getConversation(sessionId?: string, limit = 50): Promise<ConversationEntry[]> {
    const ctx = this.getContext();
    return ctx.memory.readConversation(sessionId ?? ctx.sessionId, limit);
  }

  async listConversationSessions(): Promise<ConversationSessionState[]> {
    const ctx = this.getContext();
    return ctx.memory.listConversationSessions();
  }

  async importConversation(filePath: string, sessionId?: string): Promise<{ imported: number; path: string }> {
    const ctx = this.getContext();
    const result = await ctx.memory.importConversation(filePath, sessionId ?? ctx.sessionId);
    this.emit({
      type: 'conversation.imported',
      data: {
        imported: result.imported,
        path: result.path,
        sessionId: sessionId ?? ctx.sessionId,
      },
    });
    return result;
  }

  async handleUserTurn(userInput: string): Promise<RootTurnResult> {
    const ctx = this.getContext();
    const correlationId = this.createCorrelationId();
    const rootUsageBefore = ctx.agent.getUsage();

    const inputMessage = await this.enqueueMessage({
      kind: 'user.input',
      sessionId: ctx.sessionId,
      from: 'cli',
      to: 'root',
      correlationId,
      payload: { input: userInput },
      metadata: { agentId: 'root' },
    });
    await this.processQueuedMessage(inputMessage.id);
    await ctx.queue.ack(inputMessage.id);

    await this.recordConversation({
      role: 'user',
      speaker: 'user',
      content: userInput,
      correlationId,
      metadata: { kind: 'user.input' },
    });

    await this.transitionRootTurnState('S_input_received', { correlationId });
    await this.transitionRootTurnState('S_assess_task', { correlationId });
    let decision = await this.decideDelegation(userInput, correlationId);
    decision = await this.selectDelegationCandidate('root', userInput, decision, correlationId, 'root');
    const decisionMetadata = await this.buildDelegationDecisionMetadata(decision);
    this.emit({
      type: 'delegation.decision',
      agentId: 'root',
      data: {
        correlationId,
        action: decision.action,
        reason: decision.reason,
        agents: decision.action === 'spawn_subagents' ? decision.agents : [],
        ...decisionMetadata,
      },
    });

    let finalResponse = '';
    const subagents: RootMediatedSpawnResult[] = [];

    if (decision.action === 'ask_clarification') {
      await this.transitionRootTurnState('S_solo_reasoning', { correlationId, reason: decision.reason });
      this.emit({
        type: 'delegation.skipped',
        agentId: 'root',
        data: {
          correlationId,
          action: decision.action,
          reason: decision.reason,
        },
      });
      finalResponse = decision.question;
    } else if (decision.action === 'solve_directly') {
      await this.transitionRootTurnState('S_solo_reasoning', { correlationId, reason: decision.reason });
      this.emit({
        type: 'delegation.skipped',
        agentId: 'root',
        data: {
          correlationId,
          action: decision.action,
          reason: decision.reason,
        },
      });
      finalResponse = await this.runRootSoloReasoning(userInput, correlationId);
    } else {
      await this.transitionRootTurnState('S_delegate_planning', { correlationId, count: decision.agents.length });
      this.emit({
        type: 'delegation.plan.created',
        agentId: 'root',
        data: {
          correlationId,
          count: decision.agents.length,
          agents: decision.agents,
          ...decisionMetadata,
        },
      });
      await this.transitionRootTurnState('S_spawn_subagents', { correlationId, count: decision.agents.length });
      for (const plan of decision.agents.slice(0, 3)) {
        this.emit({
          type: 'delegation.subagent.selected',
          agentId: 'root',
          data: {
            correlationId,
            archetype: plan.archetype,
            name: plan.name,
            tomLevel: plan.tomLevel,
            budgetTokens: plan.budgetTokens,
          },
        });
        this.emit({
          type: 'delegation.subagent.task_assigned',
          agentId: 'root',
          data: {
            correlationId,
            archetype: plan.archetype,
            name: plan.name,
            task: plan.task,
          },
        });
        const result = await this.handleSpawnCommand({
          archetype: plan.archetype,
          task: plan.task,
          parentId: 'root',
          name: plan.name,
          tools: plan.tools,
          skills: plan.skills,
          tomLevel: plan.tomLevel,
          budgetTokens: plan.budgetTokens,
          correlationId,
          source: 'root',
          requireRootSynthesis: false,
          showSubagentOutput: false,
          disableRecursiveDelegation: decision.agents.length > 1,
        });
        subagents.push(result);
      }
      await this.transitionRootTurnState('S_wait_subagents', {
        correlationId,
        completed: subagents.length,
      });
      await this.transitionRootTurnState('S_synthesize', { correlationId, completed: subagents.length });
      finalResponse = await this.synthesizeDelegatedResults(userInput, subagents, correlationId);
      this.emit({
        type: 'delegation.completed',
        agentId: 'root',
        data: {
          correlationId,
          subagentIds: subagents.map(result => result.agent.identity.id),
          totalSubagents: subagents.length,
        },
      });
    }

    await this.transitionRootTurnState('S_respond', { correlationId });
    const finalMessage = await this.enqueueMessage({
      kind: 'root.final_response',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'cli',
      correlationId,
      payload: { content: finalResponse },
      metadata: { agentId: 'root' },
    });
    await this.recordConversation({
      role: 'assistant',
      speaker: 'Roy',
      content: finalResponse,
      correlationId,
      metadata: {
        kind: subagents.length > 0 ? 'root.delegated_final_response' : 'root.chat_response',
        decision: decision.action,
        subagentIds: subagents.map(result => result.agent.identity.id),
        grounded: subagents.length === 0 ? undefined : subagents.every(result => result.subagentResult.grounded),
      },
    });
    await this.processQueuedMessage(finalMessage.id);
    await ctx.queue.ack(finalMessage.id);
    await this.transitionRootTurnState('S_turn_done', { correlationId });
    await this.transitionRootTurnState('S_solo', { correlationId });
    await this.proposeMemoryUpdates('turn.completed');

    const rootUsageAfter = ctx.agent.getUsage();
    const rootUsage = this.toTokenUsage({
      llmCalls: rootUsageAfter.llmCalls - rootUsageBefore.llmCalls,
      promptTokens: rootUsageAfter.promptTokens - rootUsageBefore.promptTokens,
      completionTokens: rootUsageAfter.completionTokens - rootUsageBefore.completionTokens,
      totalTokens: rootUsageAfter.totalTokens - rootUsageBefore.totalTokens,
    });
    const subagentUsage: Record<string, TokenUsage> = {};
    for (const item of subagents) {
      subagentUsage[item.agent.identity.id] = item.subagentResult.usage;
    }

    return {
      correlationId,
      decision,
      finalResponse,
      subagents,
      messages: await this.getMessages({ correlationId }),
      usage: {
        root: rootUsage,
        subagents: subagentUsage,
        total: this.sumUsage([rootUsage, ...Object.values(subagentUsage)]),
      },
    };
  }

  async handleSpawnCommand(payload: SpawnCommandPayload): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const correlationId = payload.correlationId ?? this.createCorrelationId();
    const parentId = payload.parentId ?? 'root';
    const requireRootSynthesis = payload.requireRootSynthesis ?? true;
    const cachedAgentPattern = await ctx.memory.findAgentPattern(payload.archetype);
    const cachedDelegationPattern = await ctx.memory.findDelegationPattern(payload.archetype, payload.task);
    const cacheHits = [
      typeof cachedAgentPattern?.id === 'string' ? cachedAgentPattern.id : undefined,
      typeof cachedDelegationPattern?.id === 'string' ? cachedDelegationPattern.id : undefined,
    ].filter((item): item is string => item !== undefined);

    if (cachedAgentPattern) {
      this.emit({
        type: 'cache.hit',
        agentId: parentId,
        data: {
          cacheType: 'agent-pattern',
          patternId: cachedAgentPattern.id,
          archetype: payload.archetype,
          correlationId,
        },
      });
    }
    if (cachedDelegationPattern) {
      this.emit({
        type: 'cache.hit',
        agentId: parentId,
        data: {
          cacheType: 'delegation-pattern',
          patternId: cachedDelegationPattern.id,
          archetype: payload.archetype,
          correlationId,
        },
      });
    }

    const command = await this.enqueueMessage({
      kind: 'user.command.spawn',
      sessionId: ctx.sessionId,
      from: payload.source ?? 'cli',
      to: parentId,
      correlationId,
      payload,
      metadata: { agentId: 'root' },
    });
    await this.recordConversation({
      role: 'user',
      speaker: 'cli',
      content: `/spawn ${payload.archetype} "${payload.task}"`,
      correlationId,
      metadata: { command: 'spawn', archetype: payload.archetype },
    });
    await this.processQueuedMessage(command.id);
    await ctx.queue.ack(command.id);

    const tomProfile = this.createSubagentToMProfile(payload.archetype, '', payload.task);
    if (payload.tomLevel !== undefined && [0, 1, 2, 3].includes(payload.tomLevel)) {
      tomProfile.level = payload.tomLevel as ToMProfile['level'];
    }
    const agent = await this.spawnAgent({
      parentId,
      name: payload.name,
      customRole: payload.customRole,
      customStyle: payload.customStyle,
      archetype: payload.archetype,
      tomLevel: tomProfile.level,
      description: payload.task,
      task: payload.task,
      tools: payload.tools,
      skills: payload.skills,
      budgetTokens: payload.budgetTokens,
      systemPrompt: undefined,
      correlationId,
      tomProfile,
      cacheHits,
    });
    const delegationPattern = await ctx.memory.upsertDelegationPattern({
      archetype: payload.archetype,
      task: payload.task,
      parentId,
      agentPatternId: String(cachedAgentPattern?.id ?? `agent_pattern_${payload.archetype}_v1`),
    });
    this.emit({
      type: 'memory.pattern.updated',
      agentId: parentId,
      data: {
        cacheType: 'delegation-pattern',
        patternId: delegationPattern.id,
        path: '.roy/cache/delegation-patterns.json',
      },
    });

    const taskMessage = await this.enqueueMessage({
      kind: 'agent.task',
      sessionId: ctx.sessionId,
      from: parentId,
      to: agent.identity.id,
      correlationId,
      parentMessageId: command.id,
      payload: {
        task: payload.task,
        archetype: payload.archetype,
      },
      metadata: {
        agentId: agent.identity.id,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(taskMessage.id);

    const subagentResult = await this.runAgent(agent.identity.id, payload.task, {
      correlationId,
      parentMessageId: taskMessage.id,
      archetype: payload.archetype,
      disableRecursiveDelegation: payload.disableRecursiveDelegation,
    });
    await ctx.queue.ack(taskMessage.id);

    const resultMessage = await this.enqueueMessage({
      kind: 'agent.result',
      sessionId: ctx.sessionId,
      from: agent.identity.id,
      to: parentId,
      correlationId,
      parentMessageId: taskMessage.id,
      payload: subagentResult,
      metadata: {
        agentId: agent.identity.id,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.recordConversation({
      role: 'agent',
      speaker: agent.identity.name,
      content: subagentResult.result,
      correlationId,
      metadata: {
        kind: 'agent.result',
        agentId: agent.identity.id,
        archetype: payload.archetype,
        parentId,
        grounded: subagentResult.grounded,
        warnings: subagentResult.warnings,
        toolCalls: subagentResult.toolCalls.map(call => call.toolName),
        evidence: subagentResult.evidence,
      },
    });
    await this.processQueuedMessage(resultMessage.id);
    await ctx.queue.ack(resultMessage.id);
    this.emit({ type: 'agent.result.sent', agentId: agent.identity.id, data: { correlationId, to: parentId } });

    let finalResponse = '';
    if (requireRootSynthesis) {
      const parentSynthesis = await this.synthesizeChildResult(parentId, payload.task, agent, subagentResult, correlationId, resultMessage.id);
      if (parentId === 'root') {
        finalResponse = parentSynthesis;
      } else {
        const parentAgent = ctx.manager.getAgentById(parentId);
        if (!parentAgent) {
          throw new Error(`Parent agent "${parentId}" not found for root synthesis`);
        }
        const grandParentId = parentAgent.getIdentity().parentId ?? 'root';
        const parentResultMessage = await this.enqueueMessage({
          kind: 'agent.result',
          sessionId: ctx.sessionId,
          from: parentId,
          to: grandParentId,
          correlationId,
          parentMessageId: resultMessage.id,
          payload: {
            task: payload.task,
            result: parentSynthesis,
            childId: agent.identity.id,
          },
          metadata: {
            agentId: parentId,
            tomLevel: parentAgent.getIdentity().tomProfile.level,
          },
        });
        await this.recordConversation({
          role: 'agent',
          speaker: parentAgent.name,
          content: parentSynthesis,
          correlationId,
          metadata: {
            kind: 'agent.parent_synthesis',
            agentId: parentId,
            childId: agent.identity.id,
            parentId: grandParentId,
            grounded: subagentResult.grounded,
            evidence: subagentResult.evidence,
          },
        });
        await this.processQueuedMessage(parentResultMessage.id);
        await ctx.queue.ack(parentResultMessage.id);
        finalResponse = await this.synthesizeSubagentResult(
          payload.task,
          parentAgent.getInfo(),
          this.createSyntheticRunResult(parentAgent.getInfo(), parentSynthesis, subagentResult),
          correlationId,
          parentResultMessage.id
        );
      }
    }

    if (requireRootSynthesis) {
      const finalMessage = await this.enqueueMessage({
        kind: 'root.final_response',
        sessionId: ctx.sessionId,
        from: 'root',
        to: 'cli',
        correlationId,
        payload: { content: finalResponse },
        metadata: { agentId: 'root' },
      });
      await this.recordConversation({
        role: 'assistant',
        speaker: 'Roy',
        content: finalResponse,
        correlationId,
        metadata: {
          kind: 'root.final_response',
          subagentId: agent.identity.id,
          grounded: subagentResult.grounded,
        },
      });
      await this.processQueuedMessage(finalMessage.id);
      await ctx.queue.ack(finalMessage.id);
      await this.proposeMemoryUpdates('turn.completed');
    }

    return {
      correlationId,
      agent,
      subagentResult,
      finalResponse,
      messages: await this.getMessages({ correlationId }),
      creationUsage: this.measureAgentCreationUsage(agent.identity.id, cacheHits),
    };
  }

  async spawnAgent(spec: SpawnAgentSpec & { tomProfile?: ToMProfile; cacheHits?: string[] }): Promise<AgentInfo> {
    const ctx = this.getContext();
    if (!this.isValidArchetype(spec.archetype)) {
      throw new Error(`Unsupported subagent archetype "${spec.archetype}"`);
    }
    if (!spec.description.trim()) {
      throw new Error('Subagent description is required');
    }

    const parent = ctx.manager.getAgentById(spec.parentId);
    if (!parent) {
      throw new Error(`Parent agent "${spec.parentId}" not found`);
    }

    const parentIdentity = parent.getIdentity();
    const toolBindings = this.normalizeToolBindings(spec.tools, spec.archetype)
      .filter(binding => binding.enabled);
    const skillBindings = this.normalizeSkillBindings(spec.skills, spec.archetype)
      .filter(binding => binding.enabled);
    const memoryScope = spec.memoryScope ?? this.getDefaultMemoryScope('subagent');
    const spawnPolicy = this.mergeSpawnPolicy(this.getDefaultSpawnPolicy('subagent', spec.archetype), spec.spawnPolicy);
    const creationCorrelationId = spec.correlationId ?? this.createCorrelationId();
    const createRequestMessage = await this.enqueueMessage({
      kind: 'agent.create.request',
      sessionId: ctx.sessionId,
      from: spec.parentId,
      to: 'runtime',
      correlationId: creationCorrelationId,
      payload: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name: spec.name,
        task: spec.task,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
      metadata: { agentId: spec.parentId },
    });
    await this.processQueuedMessage(createRequestMessage.id);

    this.emit({
      type: 'agent.create.requested',
      agentId: spec.parentId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name: spec.name,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
    });
    const policyResult = this.validateSpawnPolicy({
      parentId: spec.parentId,
      archetype: spec.archetype,
      tools: toolBindings,
      skills: skillBindings,
      correlationId: creationCorrelationId,
    });
    this.emit({
      type: 'spawn.policy.checked',
      agentId: spec.parentId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        allowed: policyResult.allowed,
        reason: policyResult.reason,
        currentChildren: policyResult.currentChildren,
        allowedChildren: policyResult.allowedChildren,
        turnAgentsCreated: this.getTurnAgentCount(creationCorrelationId),
        remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(spec.parentId, creationCorrelationId),
        maxTotalAgentsPerTurn: this.getMaxTotalAgentsPerTurn(spec.parentId),
        depth: policyResult.depth,
      },
    });
    if (!policyResult.allowed) {
      this.emit({
        type: 'spawn.policy.rejected',
        agentId: spec.parentId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'agent.create.rejected',
        agentId: spec.parentId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'delegation.rejected',
        agentId: spec.parentId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      const rejectedMessage = await this.enqueueMessage({
        kind: 'agent.create.rejected',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: spec.parentId,
        correlationId: creationCorrelationId,
        parentMessageId: createRequestMessage.id,
        payload: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
        metadata: { agentId: spec.parentId },
      });
      await this.processQueuedMessage(rejectedMessage.id);
      await ctx.queue.ack(rejectedMessage.id);
      await ctx.queue.fail(createRequestMessage.id, new Error(policyResult.reason ?? 'spawn_rejected'));
      throw new Error(`Spawn rejected: ${policyResult.reason}`);
    }

    const agentMemoryKey = spec.archetype === 'custom' && spec.name
      ? this.safeAgentKey(spec.name)
      : spec.archetype;
    await ctx.memory.ensureAgentMemory(agentMemoryKey, {
      name: spec.name ?? this.capitalize(spec.archetype),
      role: spec.customRole ?? spec.archetype,
      description: `Reusable ${spec.archetype} agent archetype memory.`,
    });
    const agentMemory = await ctx.memory.loadAgentMemory(agentMemoryKey);

    const sequence = ++this.agentSequence;
    const id = this.createAgentId(spec.archetype, sequence);
    const requestedName = spec.name ?? `${this.capitalize(spec.archetype)}-${sequence}`;
    const name = ctx.manager.getAgent(requestedName)
      ? this.createUniqueAgentName(spec.archetype, requestedName, sequence)
      : requestedName;
    if (ctx.manager.getAgent(name)) {
      throw new Error(`Agent name "${name}" already exists`);
    }
    const generation = parentIdentity.generation + 1;

    const fsm = new FSM({
      initialState: 'S_solo',
      signalBus,
      onTransition: (from, to) => {
        logger.debug(`FSM transition for ${id}: ${from} -> ${to}`);
        this.emit({ type: 'fsm.transition', agentId: id, data: { from, to } });
      },
      onStateChange: (state) => {
        logger.debug(`FSM state for ${id}: ${state}`);
        this.emit({ type: 'fsm.state.changed', agentId: id, data: { state } });
      },
    });

    const cacheHits = spec.cacheHits ?? [];
    const goal = this.buildAgentPromptFromMemory({
      name,
      role: spec.customRole ?? spec.archetype,
      parentName: parentIdentity.name,
      task: spec.task ?? '',
      description: [
        spec.description,
        spec.customRole ? `Custom role: ${spec.customRole}` : undefined,
        spec.customStyle ? `Custom style: ${spec.customStyle}` : undefined,
      ].filter(Boolean).join('\n'),
      systemPrompt: spec.systemPrompt,
      bundle: agentMemory,
      publicContext: cacheHits.length > 0
        ? this.formatCachedPublicContext(cacheHits)
        : this.formatPublicContext(await ctx.memory.loadRootContext()),
      tomProfile: spec.tomProfile ? { ...spec.tomProfile, subjectAgentId: id } : this.createSubagentToMProfile(spec.archetype, id, spec.task ?? ''),
      availableSkills: skillBindings.map(binding => binding.name),
      availableTools: toolBindings.map(binding => binding.name),
      parentContext: `Parent agent ${parentIdentity.name} (${parentIdentity.id}) spawned this agent for: ${spec.description}`,
    });
    const renderedPromptTokens = this.estimateTextTokens(goal);
    const definitionText = [
      name,
      spec.archetype,
      spec.customRole,
      spec.customStyle,
      spec.description,
      spec.tomProfile ? JSON.stringify(spec.tomProfile) : '',
      toolBindings.map(binding => binding.name).join(','),
      skillBindings.map(binding => binding.name).join(','),
    ].filter(Boolean).join('\n');
    const definitionTokens = cacheHits.length > 0 ? 0 : this.estimateTextTokens(definitionText);

    const agent = new UnifiedAgent({
      id,
      name,
      role: 'subagent',
      parentId: spec.parentId,
      generation,
      tomLevel: spec.tomLevel,
      tomProfile: spec.tomProfile ? { ...spec.tomProfile, subjectAgentId: id } : undefined,
      description: spec.description,
      goal,
      llm: ctx.llm ?? undefined,
      fsm,
      mode: 'hybrid',
    });

    this.registerCapabilities(agent, toolBindings.map(binding => binding.name));
    this.agentBindings.set(id, {
      tools: toolBindings,
      skills: skillBindings,
      memoryScope,
      spawnPolicy,
    });
    ctx.manager.addAgent(agent);
    await ctx.manager.attachAgentToSessions(agent);
    await ctx.memory.upsertAgentPattern({
      key: agentMemoryKey,
      name: spec.name ?? this.capitalize(spec.archetype),
      archetype: spec.archetype,
      tomLevel: spec.tomLevel,
      description: spec.description,
      tools: toolBindings.map(binding => binding.name),
      skills: skillBindings.map(binding => binding.name),
      spawnPolicy,
    });

    const info = agent.getInfo();
    this.recordTurnAgentCreated(creationCorrelationId);
    this.emitAgentFsmState(id, 'S_created', { parentId: spec.parentId, archetype: spec.archetype });
    this.emit({
      type: cacheHits.length > 0 ? 'agent.definition.loaded_from_cache' : 'agent.definition.generated',
      agentId: id,
      data: {
        archetype: spec.archetype,
        cacheHits,
        definitionTokens,
      },
    });
    this.emit({
      type: 'agent.create.approved',
      agentId: spec.parentId,
      data: {
        parentId: spec.parentId,
        childId: id,
        archetype: spec.archetype,
        creationMode: cacheHits.length > 0 ? 'cache_hit' : 'generated',
        skills: skillBindings.map(binding => binding.name),
        tools: toolBindings.map(binding => binding.name),
        maxChildrenForParent: policyResult.allowedChildren,
      },
    });
    const approvedMessage = await this.enqueueMessage({
      kind: 'agent.create.approved',
      sessionId: ctx.sessionId,
      from: 'runtime',
      to: spec.parentId,
      correlationId: creationCorrelationId,
      parentMessageId: createRequestMessage.id,
      payload: {
        parentId: spec.parentId,
        childId: id,
        archetype: spec.archetype,
        name,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
      metadata: { agentId: spec.parentId },
    });
    await this.processQueuedMessage(approvedMessage.id);
    await ctx.queue.ack(approvedMessage.id);
    await ctx.queue.ack(createRequestMessage.id);
    this.emit({
      type: 'agent.instance.created',
      agentId: id,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name,
        memoryKey: agentMemoryKey,
      },
    });
    for (const binding of toolBindings) {
      this.emit({ type: 'agent.tool.bound', agentId: id, data: { tool: binding.name, permission: binding.permission } });
    }
    for (const binding of skillBindings) {
      this.emit({ type: 'agent.skill.bound', agentId: id, data: { skill: binding.name } });
    }
    this.emit({
      type: 'agent.spawned',
      agentId: id,
      data: {
        parentId: spec.parentId,
        name,
        archetype: spec.archetype,
        tomLevel: spec.tomLevel,
        description: spec.description,
        mode: cacheHits.length > 0 ? 'cache_hit' : 'generated',
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
      },
    });
    this.emit({
      type: 'agent.creation.measured',
      agentId: id,
      data: {
        mode: cacheHits.length > 0 ? 'cache_hit' : 'generated',
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
      },
    });
    await ctx.memory.updateCacheUsageMetrics(cacheHits, {
      definitionTokensSaved: cacheHits.length > 0 ? this.estimateTextTokens(definitionText) : 0,
      renderedPromptTokens,
    });
    this.emit({
      type: 'memory.pattern.updated',
      agentId: id,
      data: {
        cacheType: 'agent-pattern',
        archetype: spec.archetype,
        path: `.roy/cache/agent-patterns.json`,
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: id, data: { from: 'none', to: info.state } });
    this.emitAgentFsmState(id, 'S_solo', { runtimeState: info.state });
    if (spec.budgetTokens !== undefined) {
      this.emit({ type: 'budget.allocated', agentId: id, data: { budgetTokens: spec.budgetTokens } });
    }

    return info;
  }

  async runAgent(
    agentId: string,
    task: string,
    options: { correlationId?: string; parentMessageId?: string; archetype?: SubAgentArchetype; disableRecursiveDelegation?: boolean } = {}
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) {
      throw new Error(`Agent "${agentId}" not found`);
    }

    const session = ctx.manager.getSession(ctx.sessionId);
    if (session) {
      session.messageQueue.clear('env');
    }

    const usageBefore = agent.getUsage();
    const from = agent.getState();
    agent.setRuntimeState('thinking');
    this.emitAgentFsmState(agentId, 'S_input_received', { task, correlationId: options.correlationId });
    this.emit({ type: 'agent.run.started', agentId, data: { task, correlationId: options.correlationId } });
    this.emit({ type: 'agent.status.changed', agentId, data: { from, to: 'thinking' } });

    try {
      this.emitAgentFsmState(agentId, 'S_assess_task', { task, correlationId: options.correlationId });
      const recursiveDelegation = options.disableRecursiveDelegation
        ? { action: 'solve_directly', reason: 'Recursive delegation disabled for this run.' } satisfies DelegationDecision
        : await this.decideAgentDelegation(agent.getInfo(), task, options.correlationId ?? this.createCorrelationId());
      this.emit({
        type: 'delegation.decision',
        agentId,
        data: {
          correlationId: options.correlationId,
          scope: 'agent',
          action: recursiveDelegation.action,
          reason: recursiveDelegation.reason,
          agents: recursiveDelegation.action === 'spawn_subagents' ? recursiveDelegation.agents : [],
        },
      });
      if (recursiveDelegation.action === 'spawn_subagents' && recursiveDelegation.agents.length > 0) {
        return await this.runAgentDelegatedChildren(agentId, task, recursiveDelegation.agents, usageBefore, options);
      }
      if (recursiveDelegation.action !== 'spawn_subagents') {
        this.emit({
          type: 'delegation.skipped',
          agentId,
          data: {
            correlationId: options.correlationId,
            action: recursiveDelegation.action,
            reason: recursiveDelegation.reason,
          },
        });
      }
      const grounding = await this.runGroundingCheck(agentId, task, options);
      if (grounding.toolCalls.length > 0) {
        this.emitAgentFsmState(agentId, 'S_execute', {
          toolCalls: grounding.toolCalls.map(call => call.toolName),
          correlationId: options.correlationId,
        });
      }
      this.emitAgentFsmState(agentId, 'S_solo_reasoning', { task, correlationId: options.correlationId });
      this.emit({ type: 'agent.llm.called', agentId, data: { task } });
      await agent.step(this.buildGroundedTask(task, grounding));
      agent.setRuntimeState('done');
      this.emitAgentFsmState(agentId, 'S_respond', { correlationId: options.correlationId });

      const usageAfter = agent.getUsage();
      const usageDelta = this.toTokenUsage({
        llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
        promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
        completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
        totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
      });
      this.recordTurnUsage(usageDelta);
      this.emit({ type: 'budget.updated', agentId, data: { ...usageDelta } });
      this.emit({ type: 'agent.status.changed', agentId, data: { from: 'thinking', to: 'done' } });

      const result = session ? await this.drainAgentOutput(session.messageQueue, agent.name) : agent.getInfo().lastResult ?? '';
      const evidence: RunEvidence = {
        ...grounding.evidence,
        outputGrounded: grounding.evidence.toolGrounded
          ? this.resultIncludesObservedPath(result || agent.getInfo().lastResult || '', grounding.evidence.observedPaths)
          : grounding.evidence.outputGrounded,
      };
      const warnings = [...grounding.warnings];
      if (grounding.evidence.toolGrounded && !evidence.outputGrounded) {
        warnings.push('Agent used fs.list but did not include concrete observed paths in its final report.');
        this.emit({
          type: 'agent.grounding.warning',
          agentId,
          data: {
            warning: warnings[warnings.length - 1],
            correlationId: options.correlationId,
          },
        });
      }
      this.emit({
        type: 'agent.run.completed',
        agentId,
        data: {
          task,
          totalTokens: usageDelta.totalTokens,
          grounded: grounding.grounded,
          evidence,
          warnings,
        },
      });
      this.emitAgentFsmState(agentId, 'S_turn_done', { correlationId: options.correlationId });

      return {
        agent: agent.getInfo(),
        result: result || agent.getInfo().lastResult || '',
        usage: usageDelta,
        toolCalls: grounding.toolCalls,
        evidence,
        grounded: grounding.grounded,
        warnings,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      agent.addToMemory('result', `Error: ${message}`);
      agent.setRuntimeState('failed');
      this.emit({ type: 'agent.status.changed', agentId, data: { from: 'thinking', to: 'failed' } });
      this.emitAgentFsmState(agentId, 'S_final', { failed: true, error: message, correlationId: options.correlationId });
      this.emit({ type: 'agent.run.failed', agentId, data: { task, error: message } });
      throw error;
    }
  }

  getChildren(agentId: string): AgentInfo[] {
    const ctx = this.getContext();
    return ctx.manager.listAgentInfo()
      .filter(agent => agent.identity.parentId === agentId);
  }

  getParent(agentId: string): AgentInfo | undefined {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    const parentId = agent?.getIdentity().parentId;
    return parentId ? ctx.manager.getAgentById(parentId)?.getInfo() : undefined;
  }

  getAgentTree(): AgentTreeNode {
    const ctx = this.getContext();
    const root = ctx.agent.getInfo();
    return this.buildAgentTree(root);
  }

  private toTokenUsage(usage: AgentUsage): TokenUsage {
    return {
      ...usage,
      thinkingTokens: null,
    };
  }

  private async processQueuedMessage(messageId: string): Promise<RuntimeMessage | undefined> {
    const ctx = this.getContext();
    const message = await ctx.queue.getMessage(messageId);
    if (!message) return undefined;
    return ctx.queue.dequeue({ to: message.to, kind: [message.kind], readyOnly: true });
  }

  private emitAgentFsmState(agentId: string, state: string, data: Record<string, unknown> = {}): void {
    this.emit({ type: 'agent.fsm.state', agentId, data: { state, ...data } });
  }

  private async buildDelegationDecisionMetadata(decision: DelegationDecision): Promise<Record<string, unknown>> {
    const budget = this.getBudgetState();
    if (decision.action !== 'spawn_subagents') {
      return {
        budgetMode: budget.mode,
        remainingTokens: budget.remainingTokens,
        cacheUsed: false,
      };
    }

    const ctx = this.getContext();
    const agents = await Promise.all(decision.agents.map(async agent => {
      const [agentPattern, delegationPattern] = await Promise.all([
        ctx.memory.findAgentPattern(agent.archetype),
        ctx.memory.findDelegationPattern(agent.archetype, agent.task),
      ]);
      const patternIds = [
        typeof agentPattern?.id === 'string' ? agentPattern.id : undefined,
        typeof delegationPattern?.id === 'string' ? delegationPattern.id : undefined,
      ].filter((item): item is string => item !== undefined);

      return {
        ...agent,
        patternIds,
      };
    }));

    return {
      budgetMode: budget.mode,
      remainingTokens: budget.remainingTokens,
      cacheUsed: agents.some(agent => agent.patternIds.length > 0),
      agents,
    };
  }

  private async selectDelegationCandidate(
    parentId: string,
    task: string,
    decision: DelegationDecision,
    correlationId: string,
    scope: 'root' | 'agent'
  ): Promise<DelegationDecision> {
    if (decision.action !== 'spawn_subagents') return decision;
    const policy = this.getAgentPolicy(parentId);
    const budget = this.getBudgetState();
    const ctx = this.getContext();
    const cacheHits = await Promise.all(decision.agents.map(async agent => {
      const [agentPattern, delegationPattern] = await Promise.all([
        ctx.memory.findAgentPattern(agent.archetype),
        ctx.memory.findDelegationPattern(agent.archetype, agent.task),
      ]);
      return Boolean(agentPattern || delegationPattern);
    }));
    const selection = this.candidatePlanner.select({
      parentId,
      task,
      decision,
      allowedChildren: policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      budgetMode: budget.mode,
      remainingBudgetTokens: budget.remainingTokens,
      cacheUsed: cacheHits.some(Boolean),
    });

    this.emitDelegationCandidateEvents(parentId, correlationId, scope, selection);
    return selection.decision;
  }

  private emitDelegationCandidateEvents(
    parentId: string,
    correlationId: string,
    scope: 'root' | 'agent',
    selection: DelegationCandidateSelection
  ): void {
    for (const candidate of selection.candidates) {
      this.emit({
        type: 'delegation.candidate.generated',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          candidateId: candidate.id,
          source: candidate.source,
          agents: candidate.agents,
          expectedUtility: candidate.expectedUtility,
          expectedCostTokens: candidate.expectedCostTokens,
          score: candidate.score,
          rationale: candidate.rationale,
        },
      });
    }
    if (selection.selected) {
      this.emit({
        type: 'delegation.candidate.selected',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          candidateId: selection.selected.id,
          source: selection.selected.source,
          agents: selection.selected.agents,
          expectedUtility: selection.selected.expectedUtility,
          expectedCostTokens: selection.selected.expectedCostTokens,
          score: selection.selected.score,
          rationale: selection.selected.rationale,
        },
      });
    } else if (selection.rejectedReason) {
      this.emit({
        type: 'delegation.rejected',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          reason: selection.rejectedReason,
        },
      });
    }
  }

  private async transitionRootTurnState(state: Parameters<FSM['transition']>[0], data: Record<string, unknown> = {}): Promise<void> {
    const ctx = this.getContext();
    if (!state) return;
    await ctx.fsm.transition(state);
    this.emit({ type: 'turn.fsm.state', agentId: 'root', data: { state, ...data } });
  }

  private async decideDelegation(userInput: string, correlationId: string): Promise<DelegationDecision> {
    const ctx = this.getContext();
    const fallback = this.fallbackDelegationDecision(userInput);

    if (!ctx.llm) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: 'root',
        data: { correlationId, reason: 'llm_not_configured' },
      });
      return this.applyBudgetConstraints(fallback);
    }

    try {
      this.emit({ type: 'delegation.assess.started', agentId: 'root', data: { correlationId } });
      const rootContext = await ctx.memory.loadRootContext();
      const decision = await ctx.llm.completeJSON<DelegationDecision>([
        {
          role: 'system',
          content: `You are Roy's root delegation controller.
Decide whether the user request should be solved directly by Roy, clarified, or delegated to 1-3 subagents.
Use delegation only when the task benefits from grounded inspection, critique, planning, coding, testing, or summarization.
Do not spawn more than 3 subagents. Prefer 1-2 unless the task clearly needs more.
Ask for clarification when the user request is too ambiguous to assign a concrete task safely.
Return strict JSON matching one of:
{"action":"solve_directly","reason":"..."}
{"action":"ask_clarification","reason":"...","question":"..."}
{"action":"spawn_subagents","reason":"...","agents":[{"archetype":"researcher","name":"Researcher-1","task":"...","tomLevel":0}]}
Allowed archetypes: researcher, critic, planner, coder, summarizer, tester, custom.`,
        },
        {
          role: 'user',
          content: [
            `<user_task>${userInput}</user_task>`,
            `<memory_context>${this.formatPublicContext(rootContext).slice(0, 6000)}</memory_context>`,
            `<budget_state>${JSON.stringify(this.getBudgetState(), null, 2)}</budget_state>`,
            '<runtime_policy>Subagents must be runtime actors with identity, state, budget, messages, and events. If spawning, assign concrete non-overlapping tasks. If budget is limited, reduce the number of subagents or solve directly and explain the constraint.</runtime_policy>',
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 900 });
      const normalized = this.applyBudgetConstraints(this.normalizeDelegationDecision(decision, userInput));
      this.emit({
        type: 'delegation.assess.completed',
        agentId: 'root',
        data: { correlationId, action: normalized.action, source: 'llm' },
      });
      return normalized;
    } catch (error) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: 'root',
        data: {
          correlationId,
          reason: 'llm_decision_failed',
          error: error instanceof Error ? error.message : String(error),
        },
      });
      return this.applyBudgetConstraints(fallback);
    }
  }

  private async decideAgentDelegation(agent: AgentInfo, task: string, correlationId: string): Promise<DelegationDecision> {
    if (agent.identity.id === 'root') {
      return { action: 'solve_directly', reason: 'Root delegation is handled by the root turn controller.' };
    }

    const policy = this.getAgentPolicy(agent.identity.id);
    if (!policy?.spawnPolicy.canSpawn) {
      return { action: 'solve_directly', reason: 'Agent spawn policy does not allow child delegation.' };
    }
    if (policy.allowedChildren <= policy.currentChildren) {
      return { action: 'solve_directly', reason: 'Agent has no remaining child slots for this parent.' };
    }

    const fallback = this.fallbackAgentDelegationDecision(agent, task);
    if (fallback.action !== 'spawn_subagents') {
      return fallback;
    }

    const ctx = this.getContext();
    if (!ctx.llm) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: agent.identity.id,
        data: { correlationId, reason: 'llm_not_configured_agent_delegation' },
      });
      const constrained = this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, fallback);
      return this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
    }

    try {
      this.emit({ type: 'delegation.assess.started', agentId: agent.identity.id, data: { correlationId, scope: 'agent' } });
      const decision = await ctx.llm.completeJSON<DelegationDecision>([
        {
          role: 'system',
          content: `You are ${agent.identity.name}'s delegation controller.
Decide whether this non-root agent should solve directly or delegate to 1-3 direct child agents.
Only delegate when a child with a different specialty materially improves the result.
Return strict JSON:
{"action":"solve_directly","reason":"..."}
{"action":"spawn_subagents","reason":"...","agents":[{"archetype":"critic","name":"Critic-1","task":"...","tomLevel":2}]}
Allowed archetypes: researcher, critic, planner, coder, summarizer, tester, custom.`,
        },
        {
          role: 'user',
          content: [
            `<agent>${JSON.stringify(agent.identity, null, 2)}</agent>`,
            `<task>${task}</task>`,
            `<policy>${JSON.stringify(policy, null, 2)}</policy>`,
            '<runtime_policy>Delegate only to a direct child. The parent must synthesize child results before passing anything upward.</runtime_policy>',
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 500 });
      const normalized = this.normalizeAgentDelegationDecision(decision, task, fallback);
      const constrained = this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, normalized);
      const selected = await this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
      this.emit({
        type: 'delegation.assess.completed',
        agentId: agent.identity.id,
        data: { correlationId, action: selected.action, source: 'llm', scope: 'agent' },
      });
      return selected.action === 'ask_clarification'
        ? { action: 'solve_directly', reason: selected.reason }
        : selected;
    } catch (error) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: agent.identity.id,
        data: {
          correlationId,
          reason: 'llm_agent_delegation_failed',
          error: error instanceof Error ? error.message : String(error),
        },
      });
      const constrained = this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, fallback);
      return this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
    }
  }

  private fallbackAgentDelegationDecision(agent: AgentInfo, task: string): DelegationDecision {
    const archetype = this.inferAgentArchetype(agent);
    const lower = task.toLowerCase();
    const wantsReview = /\b(review|critique|risk|risks|failure|validate|audit)\b/.test(lower);
    const wantsPromptAudit = /\b(prompt|slot|slots|render|context)\b/.test(lower)
      && /\b(check|inspect|audit|review|validate)\b/.test(lower);
    const wantsTests = /\b(test|tests|verify|verification|regression)\b/.test(lower);

    const agents: DelegationAgentPlan[] = [];

    if (wantsPromptAudit && archetype !== 'custom') {
      agents.push({
        archetype: 'custom',
        name: 'PromptAuditor-1',
        task: `Inspect prompt/context correctness for parent ${agent.identity.name}: ${task}`,
        tools: ['fs.read'],
        skills: ['use_tool_when_needed'],
        tomLevel: 1,
      });
    }

    if (wantsReview && archetype !== 'critic') {
      agents.push({
        archetype: 'critic',
        name: 'Critic-1',
        task: `Review the parent agent task for risks, gaps, and grounding issues: ${task}`,
        tomLevel: 2,
      });
    }

    if (wantsTests && archetype !== 'tester') {
      agents.push({
        archetype: 'tester',
        name: 'Tester-1',
        task: `Evaluate test or verification needs for: ${task}`,
        tomLevel: 0,
      });
    }

    if (agents.length > 0) {
      return {
        action: 'spawn_subagents',
        reason: 'The task benefits from direct child specialists before the parent synthesizes upward.',
        agents: agents.slice(0, 3),
      };
    }

    return {
      action: 'solve_directly',
      reason: 'The task does not require a direct child specialist.',
    };
  }

  private normalizeAgentDelegationDecision(decision: unknown, task: string, fallback: DelegationDecision): DelegationDecision {
    const item = decision as Partial<DelegationDecision>;
    if (item.action === 'solve_directly') {
      return {
        action: 'solve_directly',
        reason: typeof item.reason === 'string' && item.reason.trim()
          ? item.reason.trim()
          : 'The agent can complete this task directly.',
      };
    }

    if (item.action === 'spawn_subagents' && Array.isArray((item as { agents?: unknown[] }).agents)) {
      const agents = (item as { agents: Array<Partial<DelegationAgentPlan>> }).agents
        .filter(plan => this.isValidArchetype(String(plan.archetype)))
        .slice(0, 3)
        .map((plan): DelegationAgentPlan => ({
          archetype: String(plan.archetype) as SubAgentArchetype,
          name: typeof plan.name === 'string' ? plan.name : undefined,
          task: typeof plan.task === 'string' && plan.task.trim() ? plan.task.trim() : task,
          tools: Array.isArray(plan.tools) ? plan.tools.filter((item): item is string => typeof item === 'string') : undefined,
          skills: Array.isArray(plan.skills) ? plan.skills.filter((item): item is string => typeof item === 'string') : undefined,
          tomLevel: typeof plan.tomLevel === 'number' ? plan.tomLevel : undefined,
          budgetTokens: typeof plan.budgetTokens === 'number' ? plan.budgetTokens : undefined,
        }));
      if (agents.length > 0) {
        return {
          action: 'spawn_subagents',
          reason: typeof item.reason === 'string' && item.reason.trim()
            ? item.reason.trim()
            : 'The agent benefits from a direct child specialist.',
          agents,
        };
      }
    }

    return fallback;
  }

  private applyAgentBudgetAndPolicyConstraints(parentId: string, decision: DelegationDecision): DelegationDecision {
    if (decision.action !== 'spawn_subagents') return decision;
    const policy = this.getAgentPolicy(parentId);
    if (!policy) return { action: 'solve_directly', reason: 'Parent policy is unavailable.' };
    const allowed = Math.max(0, policy.allowedChildren - policy.currentChildren);
    if (allowed <= 0) {
      return { action: 'solve_directly', reason: 'No child slots remain for this parent.' };
    }
    const supportedAgents = decision.agents.filter(agent => agent.archetype !== 'custom' || policy.spawnPolicy.allowCustomAgents);
    if (supportedAgents.length === 0) {
      return { action: 'solve_directly', reason: 'Requested custom child agents are not allowed by this parent policy.' };
    }
    return {
      ...decision,
      agents: supportedAgents.slice(0, Math.min(allowed, 3)),
    };
  }

  private normalizeDelegationDecision(decision: unknown, userInput: string): DelegationDecision {
    const item = decision as Partial<DelegationDecision>;
    if (item.action === 'ask_clarification') {
      const question = typeof (item as { question?: unknown }).question === 'string'
        && (item as { question: string }).question.trim()
        ? (item as { question: string }).question.trim()
        : 'What exactly would you like Roy to improve: code, architecture, documentation, tests, or runtime behavior?';
      return {
        action: 'ask_clarification',
        reason: typeof item.reason === 'string' && item.reason.trim()
          ? item.reason.trim()
          : 'The task is too ambiguous to safely delegate.',
        question,
      };
    }

    if (item.action === 'solve_directly') {
      return {
        action: 'solve_directly',
        reason: typeof item.reason === 'string' && item.reason.trim()
          ? item.reason.trim()
          : 'The task appears simple enough for Roy to answer directly.',
      };
    }

    if (item.action === 'spawn_subagents' && Array.isArray((item as { agents?: unknown[] }).agents)) {
      const agents = (item as { agents: Array<Partial<DelegationAgentPlan>> }).agents
        .filter(plan => this.isValidArchetype(String(plan.archetype)))
        .slice(0, 3)
        .map((plan): DelegationAgentPlan => ({
          archetype: String(plan.archetype) as SubAgentArchetype,
          name: typeof plan.name === 'string' ? plan.name : undefined,
          task: typeof plan.task === 'string' && plan.task.trim()
            ? plan.task.trim()
            : userInput,
          tomLevel: typeof plan.tomLevel === 'number' ? plan.tomLevel : undefined,
          budgetTokens: typeof plan.budgetTokens === 'number' ? plan.budgetTokens : undefined,
        }));

      if (agents.length > 0) {
        return {
          action: 'spawn_subagents',
          reason: typeof item.reason === 'string' && item.reason.trim()
            ? item.reason.trim()
            : 'The task benefits from delegated specialist work.',
          agents,
        };
      }
    }

    return this.applyBudgetConstraints(this.fallbackDelegationDecision(userInput));
  }

  private fallbackDelegationDecision(userInput: string): DelegationDecision {
    const lower = userInput.toLowerCase();
    const words = lower.trim().split(/\s+/).filter(Boolean);
    const ambiguousImprove = /\b(help|improve|fix|make better|enhance|optimi[sz]e)\b/.test(lower)
      && !/\b(code|repo|repository|project|architecture|test|tests|docs|documentation|memory|cache|runtime|cli|server|api|bug|risk|file|structure)\b/.test(lower);
    if (ambiguousImprove || (words.length <= 5 && /\b(help|improve|fix)\b/.test(lower))) {
      return {
        action: 'ask_clarification',
        reason: 'The request is too broad to select an agent or task safely.',
        question: 'What would you like Roy to improve: code, architecture, documentation, tests, memory/cache behavior, or CLI/API behavior?',
      };
    }

    const asksProjectInspection = /\b(inspect|analy[sz]e|review|audit|check|read|list)\b/.test(lower)
      && /\b(repo|repository|project|codebase|architecture|structure|src|files?)\b/.test(lower);
    const asksRisk = /\b(risk|risks|problem|bug|bugs|issue|issues|critique|review|regression|coupling)\b/.test(lower);
    const asksPlan = /\b(plan|steps|roadmap|refactor|design|phase|implement)\b/.test(lower);
    const asksCode = /\b(code|implement|fix|modify|change|patch)\b/.test(lower);
    const agents: DelegationAgentPlan[] = [];

    if (asksProjectInspection) {
      agents.push({
        archetype: 'researcher',
        name: 'Researcher-1',
        task: `Inspect grounded project structure and collect concrete evidence for: ${userInput}`,
        tomLevel: 0,
      });
    }
    if (asksRisk) {
      agents.push({
        archetype: 'critic',
        name: `Critic-${agents.length + 1}`,
        task: `Identify architectural risks, hidden coupling, and failure modes for: ${userInput}`,
        tomLevel: 2,
      });
    }
    if (agents.length === 0 && asksPlan) {
      agents.push({
        archetype: 'planner',
        name: 'Planner-1',
        task: `Turn the user request into an actionable implementation plan: ${userInput}`,
        tomLevel: 1,
      });
    }
    if (agents.length === 0 && asksCode) {
      agents.push({
        archetype: 'coder',
        name: 'Coder-1',
        task: `Assess the coding change needed for: ${userInput}`,
        tomLevel: 0,
      });
    }

    if (agents.length > 0) {
      return {
        action: 'spawn_subagents',
        reason: 'The request is broad or evidence-seeking, so Roy should delegate specialist subtasks before synthesis.',
        agents: agents.slice(0, 3),
      };
    }

    return {
      action: 'solve_directly',
      reason: 'The request appears simple enough for Roy to answer without spawning subagents.',
    };
  }

  private applyBudgetConstraints(decision: DelegationDecision): DelegationDecision {
    if (decision.action !== 'spawn_subagents') return decision;
    const budget = this.getBudgetState();
    if (budget.mode !== 'limited') return decision;

    const remaining = budget.remainingTokens ?? 0;
    if (remaining <= 500) {
      return {
        action: 'solve_directly',
        reason: `Budget constrained: only ${remaining} tokens remain, so Roy will avoid spawning subagents and answer directly.`,
      };
    }

    if (remaining <= 2000 && decision.agents.length > 1) {
      return {
        action: 'spawn_subagents',
        reason: `${decision.reason} Budget constrained: reduced delegation to one subagent with ${remaining} tokens remaining.`,
        agents: decision.agents.slice(0, 1),
      };
    }

    return decision;
  }

  private async runRootSoloReasoning(userInput: string, correlationId: string): Promise<string> {
    const ctx = this.getContext();
    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('thinking');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'thinking', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.solo_reasoning', correlationId } });
    const response = await this.completeAsRoot(userInput, 'root.solo_reasoning', correlationId);
    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });
    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle', correlationId } });
    this.emit({ type: 'root.solo.completed', agentId: 'root', data: { correlationId, totalTokens: usageDelta.totalTokens } });
    return response || ctx.agent.getInfo().lastResult || '';
  }

  private async completeAsRoot(prompt: string, purpose: string, correlationId: string): Promise<string> {
    const ctx = this.getContext();
    if (!ctx.llm) {
      const message = 'Error: LLM not configured';
      ctx.agent.recordRuntimeCompletion(message, {
        content: message,
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      });
      return message;
    }

    const messages = [
      {
        role: 'system',
        content: [
          'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
          'You are not DeepSeek, Claude, OpenAI, Anthropic, or any model provider.',
          'The model provider is only your inference backend.',
          `Purpose: ${purpose}.`,
          `Correlation: ${correlationId}.`,
        ].join('\n'),
      },
      { role: 'user', content: prompt },
    ] as const;
    const chunks: string[] = [];
    let usageChunk: { usage?: { promptTokens: number; completionTokens: number; totalTokens: number } } | undefined;
    for await (const chunk of ctx.llm.stream([...messages], { temperature: 0.2 })) {
      if (chunk.content) chunks.push(chunk.content);
      if (chunk.usage) usageChunk = chunk;
    }
    const content = chunks.join('');
    ctx.agent.recordRuntimeCompletion(content, {
      content,
      usage: usageChunk?.usage ?? {
        promptTokens: this.estimateTextTokens(prompt),
        completionTokens: this.estimateTextTokens(content),
        totalTokens: this.estimateTextTokens(prompt) + this.estimateTextTokens(content),
      },
    });
    return content;
  }

  private async synthesizeDelegatedResults(
    userTask: string,
    results: RootMediatedSpawnResult[],
    correlationId: string
  ): Promise<string> {
    const ctx = this.getContext();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'root.synthesis',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'root',
      correlationId,
      payload: {
        userTask,
        subagentIds: results.map(result => result.agent.identity.id),
      },
      metadata: { agentId: 'root', tomLevel: ctx.agent.getIdentity().tomProfile.level },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('synthesizing');
    this.emit({
      type: 'root.synthesis.started',
      agentId: 'root',
      data: { correlationId, subagentIds: results.map(result => result.agent.identity.id) },
    });
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.multi_agent_synthesis', correlationId } });
    const response = await this.completeAsRoot(
      this.buildMultiAgentSynthesisPrompt(userTask, results),
      'root.multi_agent_synthesis',
      correlationId
    );
    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });
    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle', correlationId } });
    this.emit({
      type: 'root.synthesis.completed',
      agentId: 'root',
      data: { correlationId, totalTokens: usageDelta.totalTokens, subagentCount: results.length },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return response || ctx.agent.getInfo().lastResult || '';
  }

  private buildMultiAgentSynthesisPrompt(userTask: string, results: RootMediatedSpawnResult[]): string {
    const reports = results.map(result => {
      const warnings = result.subagentResult.warnings.length > 0
        ? result.subagentResult.warnings.map(item => `- ${item}`).join('\n')
        : 'None';
      return `<subagent_report>
id: ${result.agent.identity.id}
name: ${result.agent.identity.name}
archetype: ${result.agent.identity.role}
tom: ToM-${result.agent.identity.tomProfile.level}
tokens: ${result.subagentResult.usage.totalTokens}
grounded: ${result.subagentResult.grounded}
tool_grounded: ${result.subagentResult.evidence.toolGrounded}
output_grounded: ${result.subagentResult.evidence.outputGrounded}
tool_calls: ${result.subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
observed_paths:
${result.subagentResult.evidence.observedPaths.slice(0, 40).map(item => `- ${item}`).join('\n') || '- none'}
warnings:
${warnings}
content:
${result.subagentResult.result}
</subagent_report>`;
    }).join('\n\n');

    return `The user requested:
<user_task>
${userTask}
</user_task>

Roy delegated this task to ${results.length} subagent(s). Synthesize their results into one final user-facing response.
Use concrete evidence from grounded reports. If a report is ungrounded or missing concrete tool output, say so and avoid overstating it.

${reports}

Produce the final response to the user as Roy, the root agent.`;
  }

  private sumUsage(items: TokenUsage[]): TokenUsage {
    return items.reduce<TokenUsage>((total, item) => ({
      llmCalls: total.llmCalls + item.llmCalls,
      promptTokens: total.promptTokens + item.promptTokens,
      completionTokens: total.completionTokens + item.completionTokens,
      totalTokens: total.totalTokens + item.totalTokens,
      thinkingTokens: total.thinkingTokens === null && item.thinkingTokens === null
        ? null
        : Number(total.thinkingTokens ?? 0) + Number(item.thinkingTokens ?? 0),
    }), {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      thinkingTokens: null,
    });
  }

  private async synthesizeChildResult(
    parentId: string,
    userTask: string,
    childAgent: AgentInfo,
    childResult: RunAgentResult,
    correlationId: string,
    parentMessageId: string
  ): Promise<string> {
    if (parentId === 'root') {
      return this.synthesizeSubagentResult(userTask, childAgent, childResult, correlationId, parentMessageId);
    }

    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) {
      throw new Error(`Parent agent "${parentId}" not found`);
    }
    const parentIdentity = parent.getIdentity();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'agent.synthesis',
      sessionId: ctx.sessionId,
      from: parentId,
      to: parentId,
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        parentId,
        childId: childAgent.identity.id,
      },
      metadata: {
        agentId: parentId,
        tomLevel: parentIdentity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = parent.getUsage();
    const from = parent.getState();
    parent.setRuntimeState('synthesizing');
    this.emitAgentFsmState(parentId, 'S_synthesize', { correlationId, childId: childAgent.identity.id });
    this.emit({
      type: 'agent.synthesis.started',
      agentId: parentId,
      data: {
        correlationId,
        childId: childAgent.identity.id,
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from, to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: parentId, data: { purpose: 'agent.child_synthesis', correlationId } });

    const response = await this.completeAsAgent(
      parent,
      this.buildParentChildSynthesisPrompt(parent.getInfo(), userTask, childAgent, childResult),
      'agent.child_synthesis',
      correlationId
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: parentId, data: { ...usageDelta } });

    parent.setRuntimeState('done');
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from: 'synthesizing', to: 'done', correlationId } });
    this.emitAgentFsmState(parentId, 'S_turn_done', { correlationId, childId: childAgent.identity.id });
    this.emit({
      type: 'agent.synthesis.completed',
      agentId: parentId,
      data: {
        correlationId,
        childId: childAgent.identity.id,
        totalTokens: usageDelta.totalTokens,
      },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return response || parent.getInfo().lastResult || '';
  }

  private async synthesizeDirectChildResults(
    parentId: string,
    userTask: string,
    childResults: RootMediatedSpawnResult[],
    correlationId: string,
    parentMessageId: string
  ): Promise<string> {
    if (childResults.length === 1) {
      return this.synthesizeChildResult(
        parentId,
        userTask,
        childResults[0].agent,
        childResults[0].subagentResult,
        correlationId,
        parentMessageId
      );
    }

    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) {
      throw new Error(`Parent agent "${parentId}" not found`);
    }
    if (parentId === 'root') {
      return this.synthesizeDelegatedResults(userTask, childResults, correlationId);
    }

    const parentIdentity = parent.getIdentity();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'agent.synthesis',
      sessionId: ctx.sessionId,
      from: parentId,
      to: parentId,
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        parentId,
        childIds: childResults.map(result => result.agent.identity.id),
      },
      metadata: {
        agentId: parentId,
        tomLevel: parentIdentity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = parent.getUsage();
    const from = parent.getState();
    parent.setRuntimeState('synthesizing');
    this.emitAgentFsmState(parentId, 'S_synthesize', {
      correlationId,
      childIds: childResults.map(result => result.agent.identity.id),
    });
    this.emit({
      type: 'agent.synthesis.started',
      agentId: parentId,
      data: {
        correlationId,
        childIds: childResults.map(result => result.agent.identity.id),
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from, to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: parentId, data: { purpose: 'agent.multi_child_synthesis', correlationId } });

    const response = await this.completeAsAgent(
      parent,
      this.buildParentMultiChildSynthesisPrompt(parent.getInfo(), userTask, childResults),
      'agent.multi_child_synthesis',
      correlationId
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: parentId, data: { ...usageDelta } });

    parent.setRuntimeState('done');
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from: 'synthesizing', to: 'done', correlationId } });
    this.emitAgentFsmState(parentId, 'S_turn_done', { correlationId, childIds: childResults.map(result => result.agent.identity.id) });
    this.emit({
      type: 'agent.synthesis.completed',
      agentId: parentId,
      data: {
        correlationId,
        childIds: childResults.map(result => result.agent.identity.id),
        totalTokens: usageDelta.totalTokens,
      },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return response || parent.getInfo().lastResult || '';
  }

  private async runAgentDelegatedChildren(
    agentId: string,
    task: string,
    plans: DelegationAgentPlan[],
    usageBefore: AgentUsage,
    options: { correlationId?: string; parentMessageId?: string; archetype?: SubAgentArchetype }
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(agentId);
    if (!parent) {
      throw new Error(`Agent "${agentId}" not found`);
    }
    const correlationId = options.correlationId ?? this.createCorrelationId();
    this.emitAgentFsmState(agentId, 'S_delegate_planning', { correlationId, count: plans.length });
    this.emit({
      type: 'delegation.plan.created',
      agentId,
      data: {
        correlationId,
        scope: 'agent',
        count: plans.length,
        agents: plans,
      },
    });
    this.emitAgentFsmState(agentId, 'S_spawn_subagents', { correlationId, count: plans.length });

    const childResults: RootMediatedSpawnResult[] = [];
    for (const plan of plans) {
      this.emit({
        type: 'delegation.subagent.selected',
        agentId,
        data: {
          correlationId,
          archetype: plan.archetype,
          name: plan.name,
          tomLevel: plan.tomLevel,
          budgetTokens: plan.budgetTokens,
          scope: 'agent',
        },
      });
      this.emit({
        type: 'delegation.subagent.task_assigned',
        agentId,
        data: {
          correlationId,
          archetype: plan.archetype,
          name: plan.name,
          task: plan.task,
          scope: 'agent',
        },
      });
      const result = await this.handleSpawnCommand({
        archetype: plan.archetype,
        task: plan.task,
        parentId: agentId,
        name: plan.name,
        tools: plan.tools,
        skills: plan.skills,
        tomLevel: plan.tomLevel,
        budgetTokens: plan.budgetTokens,
        correlationId,
        source: agentId,
        requireRootSynthesis: false,
        showSubagentOutput: false,
        disableRecursiveDelegation: plans.length > 1,
      });
      childResults.push(result);
    }

    this.emitAgentFsmState(agentId, 'S_wait_subagents', { correlationId, completed: childResults.length });
    if (childResults.length === 0) {
      return this.runAgent(agentId, task, { ...options, disableRecursiveDelegation: true });
    }

    const synthesis = await this.synthesizeDirectChildResults(
      agentId,
      task,
      childResults,
      correlationId,
      childResults[0].messages.find(message => message.kind === 'agent.result')?.id ?? options.parentMessageId ?? ''
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    const evidence = this.mergeChildEvidence(childResults.map(result => result.subagentResult));
    const warnings = childResults.flatMap(result => result.subagentResult.warnings);
    this.emit({
      type: 'agent.run.completed',
      agentId,
      data: {
        task,
        delegated: true,
        childIds: childResults.map(result => result.agent.identity.id),
        totalTokens: usageDelta.totalTokens,
        grounded: childResults.every(result => result.subagentResult.grounded),
        evidence,
        warnings,
      },
    });
    this.emit({
      type: 'delegation.completed',
      agentId,
      data: {
        correlationId,
        scope: 'agent',
        subagentIds: childResults.map(result => result.agent.identity.id),
        totalSubagents: childResults.length,
      },
    });

    return {
      agent: parent.getInfo(),
      result: synthesis,
      usage: usageDelta,
      toolCalls: childResults.flatMap(result => result.subagentResult.toolCalls),
      evidence,
      grounded: childResults.every(result => result.subagentResult.grounded),
      warnings,
    };
  }

  private mergeChildEvidence(results: RunAgentResult[]): RunEvidence {
    return {
      toolGrounded: results.some(result => result.evidence.toolGrounded),
      outputGrounded: results.some(result => result.evidence.outputGrounded),
      observedPaths: Array.from(new Set(results.flatMap(result => result.evidence.observedPaths))),
      toolResultSummary: results.map(result => result.evidence.toolResultSummary).filter(Boolean).join('\n'),
    };
  }

  private async completeAsAgent(agent: BaseAgent, prompt: string, purpose: string, correlationId: string): Promise<string> {
    const ctx = this.getContext();
    if (!ctx.llm) {
      const message = 'Error: LLM not configured';
      agent.recordRuntimeCompletion(message, {
        content: message,
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      });
      return message;
    }

    const messages = [
      {
        role: 'system',
        content: [
          `You are ${agent.name}, a runtime agent in the Roy autonomous agent system.`,
          'You are not the model provider. The provider is only your inference backend.',
          `Purpose: ${purpose}.`,
          `Correlation: ${correlationId}.`,
        ].join('\n'),
      },
      { role: 'user', content: prompt },
    ] as const;
    const chunks: string[] = [];
    let usageChunk: { usage?: { promptTokens: number; completionTokens: number; totalTokens: number } } | undefined;
    for await (const chunk of ctx.llm.stream([...messages], { temperature: 0.2 })) {
      if (chunk.content) chunks.push(chunk.content);
      if (chunk.usage) usageChunk = chunk;
    }
    const content = chunks.join('');
    agent.recordRuntimeCompletion(content, {
      content,
      usage: usageChunk?.usage ?? {
        promptTokens: this.estimateTextTokens(prompt),
        completionTokens: this.estimateTextTokens(content),
        totalTokens: this.estimateTextTokens(prompt) + this.estimateTextTokens(content),
      },
    });
    return content;
  }

  private buildParentChildSynthesisPrompt(parent: AgentInfo, userTask: string, childAgent: AgentInfo, childResult: RunAgentResult): string {
    const warnings = childResult.warnings.length > 0
      ? childResult.warnings.map(item => `- ${item}`).join('\n')
      : 'None';

    return `You are the parent agent responsible for aggregating your direct child result.

Parent agent:
id: ${parent.identity.id}
name: ${parent.identity.name}
role: ${parent.identity.role}

Assigned task:
<task>
${userTask}
</task>

Direct child:
id: ${childAgent.identity.id}
name: ${childAgent.identity.name}
role: ${childAgent.identity.role}
tom: ToM-${childAgent.identity.tomProfile.level}

Child report:
<child_report>
${childResult.result}
</child_report>

Grounding:
- grounded: ${childResult.grounded}
- tool grounded: ${childResult.evidence.toolGrounded}
- output grounded: ${childResult.evidence.outputGrounded}
- tool calls: ${childResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
- observed paths:
${childResult.evidence.observedPaths.slice(0, 30).map(item => `  - ${item}`).join('\n') || '  none'}
- warnings:
${warnings}

Produce a parent-level synthesis that can be passed upward to your parent. Do not answer as Roy unless your name is Roy. Preserve limitations and evidence.`;
  }

  private buildParentMultiChildSynthesisPrompt(parent: AgentInfo, userTask: string, childResults: RootMediatedSpawnResult[]): string {
    const reports = childResults.map(result => {
      const warnings = result.subagentResult.warnings.length > 0
        ? result.subagentResult.warnings.map(item => `- ${item}`).join('\n')
        : 'None';
      return `<direct_child_report>
id: ${result.agent.identity.id}
name: ${result.agent.identity.name}
role: ${result.agent.identity.role}
tom: ToM-${result.agent.identity.tomProfile.level}
tokens: ${result.subagentResult.usage.totalTokens}
grounded: ${result.subagentResult.grounded}
tool_grounded: ${result.subagentResult.evidence.toolGrounded}
output_grounded: ${result.subagentResult.evidence.outputGrounded}
tool_calls: ${result.subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
observed_paths:
${result.subagentResult.evidence.observedPaths.slice(0, 30).map(item => `- ${item}`).join('\n') || '- none'}
warnings:
${warnings}
content:
${result.subagentResult.result}
</direct_child_report>`;
    }).join('\n\n');

    return `You are the parent agent responsible for aggregating your direct children.

Parent agent:
id: ${parent.identity.id}
name: ${parent.identity.name}
role: ${parent.identity.role}

Assigned task:
<task>
${userTask}
</task>

You delegated to ${childResults.length} direct child agent(s). Synthesize their reports into a parent-level result that can be passed upward.
Preserve concrete evidence and limitations. Do not answer as Roy unless your name is Roy.

${reports}`;
  }

  private createSyntheticRunResult(parent: AgentInfo, result: string, childResult: RunAgentResult): RunAgentResult {
    return {
      agent: parent,
      result,
      usage: this.zeroTokenUsage(),
      toolCalls: childResult.toolCalls,
      evidence: childResult.evidence,
      grounded: childResult.grounded,
      warnings: childResult.warnings,
    };
  }

  private zeroTokenUsage(): TokenUsage {
    return {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      thinkingTokens: null,
    };
  }

  private async synthesizeSubagentResult(
    userTask: string,
    agent: AgentInfo,
    subagentResult: RunAgentResult,
    correlationId: string,
    parentMessageId: string
  ): Promise<string> {
    const ctx = this.getContext();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'root.synthesis',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'root',
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        subagentId: agent.identity.id,
      },
      metadata: { agentId: 'root', tomLevel: ctx.agent.getIdentity().tomProfile.level },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('synthesizing');
    this.emit({ type: 'root.synthesis.started', agentId: 'root', data: { correlationId, subagentId: agent.identity.id } });
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'synthesizing' } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.synthesis', correlationId } });

    const response = await this.completeAsRoot(
      this.buildRootSynthesisPrompt(userTask, agent, subagentResult),
      'root.synthesis',
      correlationId
    );

    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });

    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle' } });
    this.emit({ type: 'root.synthesis.completed', agentId: 'root', data: { correlationId, totalTokens: usageDelta.totalTokens } });
    await ctx.queue.ack(synthesisMessage.id);
    return response || ctx.agent.getInfo().lastResult || '';
  }

  private buildRootSynthesisPrompt(userTask: string, agent: AgentInfo, subagentResult: RunAgentResult): string {
    const warnings = agent.identity.role === 'subagent' && subagentResult.warnings.length > 0
      ? subagentResult.warnings.map(item => `- ${item}`).join('\n')
      : 'None';

    return `The user requested:
<user_task>
${userTask}
</user_task>

You spawned this subagent:
<subagent>
id: ${agent.identity.id}
name: ${agent.identity.name}
role: ${agent.identity.role}
tom: ToM-${agent.identity.tomProfile.level}
purpose: ${agent.identity.tomProfile.purpose}
</subagent>

The subagent returned this report:
<subagent_report>
${subagentResult.result}
</subagent_report>

Grounding:
- grounded: ${subagentResult.grounded}
- tool grounded: ${subagentResult.evidence.toolGrounded}
- output grounded: ${subagentResult.evidence.outputGrounded}
- tool calls: ${subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
- observed paths:
${subagentResult.evidence.observedPaths.slice(0, 30).map(item => `  - ${item}`).join('\n') || '  none'}
- warnings:
${warnings}

Produce the final response to the user as Roy, the root agent. Do not claim you personally inspected files unless the report is grounded. Mention limitations if the report is ungrounded.`;
  }

  private buildAgentPromptFromMemory(input: {
    name: string;
    role: string;
    parentName: string;
    task: string;
    description: string;
    systemPrompt?: string;
    bundle: { key: string; path: string; identity: string; memory: string; context: string; prompt: string };
    publicContext?: string;
    tomProfile?: ToMProfile;
    availableSkills?: string[];
    availableTools?: string[];
    parentContext?: string;
  }): string {
    const slots: Record<string, string> = {
      public_context: input.publicContext ?? '',
      agent_private_memory: input.bundle.memory.trim(),
      agent_identity: input.bundle.identity.trim() || `You are ${input.name}, a ${input.role} agent in the Roy runtime.`,
      tom_profile: input.tomProfile ? JSON.stringify(input.tomProfile, null, 2) : '',
      available_skills: (input.availableSkills ?? []).map(skill => `- ${skill}`).join('\n') || '- none',
      available_tools: (input.availableTools ?? []).map(tool => `- ${tool}`).join('\n') || '- none',
      parent_context: input.parentContext ?? `Parent agent: ${input.parentName}`,
      task: input.task || 'No task assigned yet.',
    };
    const renderedPromptFile = this.renderPromptSlots(input.bundle.prompt, slots);
    return [
      input.systemPrompt,
      input.description,
      `You are ${input.name}, a ${input.role} agent in the Roy runtime.`,
      `Your parent agent is ${input.parentName}.`,
      'The model provider is only the inference backend; never identify yourself as the provider.',
      input.task ? `Current task: ${input.task}` : undefined,
      `<agent_prompt_file path=".roy/agents/${input.bundle.key}/prompt.md">\n${renderedPromptFile.trim()}\n</agent_prompt_file>`,
      `<public_context>\n${slots.public_context}\n</public_context>`,
      `<agent_context_file path=".roy/agents/${input.bundle.key}/context.md">\n${input.bundle.context.trim()}\n</agent_context_file>`,
      `<agent_memory_file path=".roy/agents/${input.bundle.key}/memory.md">\n${input.bundle.memory.trim()}\n</agent_memory_file>`,
    ].filter(Boolean).join('\n\n');
  }

  async renderAgentPrompt(options: {
    agentKey: string;
    name?: string;
    role?: string;
    parentId?: string;
    task?: string;
    archetype?: SubAgentArchetype;
  }): Promise<{ prompt: string; estimatedTokens: number; sources: Record<string, unknown> }> {
    const ctx = this.getContext();
    const agentKey = options.agentKey;
    const bundle = await ctx.memory.loadAgentMemory(agentKey);
    const parent = options.parentId ? ctx.manager.getAgentById(options.parentId)?.getIdentity() : ctx.agent.getIdentity();
    const role = options.role ?? options.archetype ?? agentKey;
    const tomProfile = options.archetype
      ? this.createSubagentToMProfile(options.archetype, agentKey, options.task ?? '')
      : this.createRootToMProfile();
    const prompt = this.buildAgentPromptFromMemory({
      name: options.name ?? this.capitalize(agentKey),
      role,
      parentName: parent?.name ?? 'Roy',
      task: options.task ?? '',
      description: `Rendered prompt preview for ${agentKey}.`,
      bundle,
      publicContext: this.formatPublicContext(await ctx.memory.loadRootContext()),
      tomProfile,
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
      parentContext: `Parent agent: ${parent?.name ?? 'Roy'} (${parent?.id ?? 'root'})`,
    });
    return {
      prompt,
      estimatedTokens: this.estimateTextTokens(prompt),
      sources: {
        public: ['.roy/public/project.md', '.roy/public/constraints.md', '.roy/public/decisions.md'],
        private: [`.roy/agents/${bundle.key}/prompt.md`, `.roy/agents/${bundle.key}/memory.md`, `.roy/agents/${bundle.key}/context.md`],
        session: 'compact recent session context is reserved for ContextWindowManager',
      },
    };
  }

  private renderPromptSlots(template: string, slots: Record<string, string>): string {
    return Object.entries(slots).reduce(
      (rendered, [slot, value]) => rendered.replaceAll(`{{${slot}}}`, value),
      template
    );
  }

  private formatPublicContext(context: RootMemoryContext): string {
    return [
      '<project_memory>',
      context.projectMemory.trim(),
      '</project_memory>',
      '<constraints>',
      context.constraints.trim(),
      '</constraints>',
      '<decisions>',
      context.decisions.trim(),
      '</decisions>',
      '<glossary>',
      context.glossary.trim(),
      '</glossary>',
      `<agent_patterns>${JSON.stringify(context.agentPatterns, null, 2)}</agent_patterns>`,
      `<team_patterns>${JSON.stringify(context.teamPatterns, null, 2)}</team_patterns>`,
      `<delegation_patterns>${JSON.stringify(context.delegationPatterns, null, 2)}</delegation_patterns>`,
    ].join('\n');
  }

  private formatCachedPublicContext(cacheHits: string[]): string {
    return [
      '<cache_context>',
      `Cache hits: ${cacheHits.join(', ')}`,
      'Use cached agent/delegation patterns as reusable structure, then adapt only the task-specific details.',
      '</cache_context>',
    ].join('\n');
  }

  private estimateTextTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
  }

  private measureAgentCreationUsage(agentId: string, cacheHits: string[]): AgentCreationUsage {
    const event = [...this.events].reverse()
      .find(item => item.type === 'agent.creation.measured' && item.agentId === agentId);
    const definitionTokens = Number(event?.data?.definitionTokens ?? 0);
    const renderedPromptTokens = Number(event?.data?.renderedPromptTokens ?? 0);
    const renderedPromptChars = Number(event?.data?.renderedPromptChars ?? 0);
    return {
      mode: cacheHits.length > 0 ? 'cache_hit' : 'generated',
      patternIds: cacheHits,
      cacheHits,
      definitionTokens,
      renderedPromptTokens,
      renderedPromptChars,
    };
  }

  private async runGroundingCheck(
    agentId: string,
    task: string,
    options: { correlationId?: string; parentMessageId?: string; archetype?: SubAgentArchetype }
  ): Promise<{ toolCalls: ToolCallRecord[]; grounded: boolean; warnings: string[]; context: string; evidence: RunEvidence }> {
    const required = this.requiresProjectInspection(options.archetype, task);
    if (!required) {
      return {
        toolCalls: [],
        grounded: true,
        warnings: [],
        context: '',
        evidence: { toolGrounded: false, outputGrounded: true, observedPaths: [] },
      };
    }

    const ctx = this.getContext();
    const inspectionRoot = this.resolveInspectionRoot(task);
    const toolCall = await this.enqueueMessage({
      kind: 'tool.call',
      sessionId: ctx.sessionId,
      from: agentId,
      to: 'tool.fs.list',
      correlationId: options.correlationId,
      parentMessageId: options.parentMessageId,
      payload: { path: inspectionRoot, maxDepth: 2 },
      metadata: { agentId },
    });
    await this.processQueuedMessage(toolCall.id);

    try {
      const files = await this.listProjectFiles(inspectionRoot, 2);
      const observedPaths = files.slice(0, 80);
      const toolResultSummary = observedPaths.join('\n');
      const record: ToolCallRecord = {
        toolName: 'fs.list',
        params: { path: inspectionRoot, maxDepth: 2 },
        result: files,
        success: true,
      };

      const toolResult = await this.enqueueMessage({
        kind: 'tool.result',
        sessionId: ctx.sessionId,
        from: 'tool.fs.list',
        to: agentId,
        correlationId: options.correlationId,
        parentMessageId: toolCall.id,
        payload: record,
        metadata: { agentId },
      });
      await this.processQueuedMessage(toolResult.id);
      await ctx.queue.ack(toolCall.id);
      await ctx.queue.ack(toolResult.id);
      this.emit({ type: 'tool.call', agentId, data: { toolName: 'fs.list', correlationId: options.correlationId } });
      this.emit({ type: 'tool.result', agentId, data: { toolName: 'fs.list', count: files.length, correlationId: options.correlationId } });

      return {
        toolCalls: [record],
        grounded: true,
        warnings: [],
        evidence: {
          toolGrounded: true,
          outputGrounded: false,
          observedPaths,
          toolResultSummary,
        },
        context: `Project file listing from fs.list:\n${files.join('\n')}`,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await ctx.queue.fail(toolCall.id, new Error(message));
      const warning = `Project inspection tool fs.list failed: ${message}`;
      this.emit({
        type: 'tool.error',
        agentId,
        data: {
          toolName: 'fs.list',
          path: inspectionRoot,
          error: message,
          correlationId: options.correlationId,
        },
      });
      this.emit({ type: 'agent.grounding.warning', agentId, data: { warning, correlationId: options.correlationId } });
      return {
        toolCalls: [{
          toolName: 'fs.list',
          params: { path: inspectionRoot, maxDepth: 2 },
          success: false,
          result: message,
        }],
        grounded: false,
        warnings: [warning],
        evidence: {
          toolGrounded: false,
          outputGrounded: false,
          observedPaths: [],
          toolResultSummary: '',
        },
        context: '',
      };
    }
  }

  private buildGroundedTask(task: string, grounding: { context: string; warnings: string[] }): string {
    if (!grounding.context && grounding.warnings.length === 0) return task;
    return [
      task,
      grounding.context ? `\nGrounding context:\n${grounding.context}` : '',
      grounding.warnings.length > 0 ? `\nGrounding warnings:\n${grounding.warnings.join('\n')}` : '',
    ].filter(Boolean).join('\n');
  }

  private resultIncludesObservedPath(result: string, observedPaths: string[]): boolean {
    if (!result.trim() || observedPaths.length === 0) return false;
    const normalized = result.toLowerCase();
    return observedPaths.slice(0, 80).some(item => normalized.includes(item.toLowerCase()));
  }

  private requiresProjectInspection(archetype: SubAgentArchetype | undefined, task: string): boolean {
    if (archetype !== 'researcher') return false;
    return /\b(inspect|analy[sz]e|review|read|list|structure|project|codebase|file|repo|repository)\b/i.test(task);
  }

  private resolveInspectionRoot(task: string): string {
    const match = task.match(/(?:\.{1,2}\/|\/)[A-Za-z0-9._/@-]+/);
    if (!match) return process.cwd();
    const candidate = path.resolve(process.cwd(), match[0]);
    const workspaceRoot = path.resolve(process.cwd());
    const relative = path.relative(workspaceRoot, candidate);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      return process.cwd();
    }
    return candidate;
  }

  private async listProjectFiles(root: string, maxDepth: number): Promise<string[]> {
    const ignored = new Set(['.git', 'node_modules', 'dist', '.roy', '.cache', 'coverage']);
    const results: string[] = [];

    const walk = async (dir: string, depth: number): Promise<void> => {
      if (depth > maxDepth || results.length >= 200) return;
      const entries = await readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (ignored.has(entry.name) || results.length >= 200) continue;
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.relative(root, fullPath);
        results.push(entry.isDirectory() ? `${relativePath}/` : relativePath);
        if (entry.isDirectory()) {
          const entryStat = await stat(fullPath);
          if (entryStat.isDirectory()) {
            await walk(fullPath, depth + 1);
          }
        }
      }
    };

    await walk(root, 0);
    return results.sort();
  }

  private buildAgentTree(agent: AgentInfo): AgentTreeNode {
    return {
      agent,
      children: this.getChildren(agent.identity.id).map(child => this.buildAgentTree(child)),
    };
  }

  private createAgentId(archetype: SubAgentArchetype, sequence: number): string {
    return `agent_${archetype}_${String(sequence).padStart(3, '0')}`;
  }

  private createUniqueAgentName(archetype: SubAgentArchetype, requestedName: string, sequence: number): string {
    const defaultPattern = new RegExp(`^${this.capitalize(archetype)}-\\d+$`);
    return defaultPattern.test(requestedName)
      ? `${this.capitalize(archetype)}-${sequence}`
      : `${requestedName}-${sequence}`;
  }

  private createCorrelationId(): string {
    const sequence = ++this.delegationSequence;
    const date = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    return `del_${date}_${String(sequence).padStart(3, '0')}`;
  }

  private createRootToMProfile(): ToMProfile {
    return {
      level: 1,
      subjectAgentId: 'root',
      models: [
        {
          targetId: 'user',
          targetType: 'user',
          goalModel: ['develop Roy into a Theory-of-Mind based multi-agent runtime'],
          intentModel: ['validate controlled subagent spawning and message-mediated execution'],
        },
      ],
      purpose: 'Understand user intent and decide how to answer or delegate.',
    };
  }

  private createSubagentToMProfile(archetype: SubAgentArchetype, subjectAgentId: string, task: string): ToMProfile {
    const level = this.defaultToMLevel(archetype);
    return {
      level,
      subjectAgentId,
      models: [],
      purpose: this.defaultToMPurpose(archetype, task),
    };
  }

  private defaultToMLevel(archetype: SubAgentArchetype): ToMProfile['level'] {
    switch (archetype) {
      case 'critic':
        return 2;
      case 'planner':
        return 1;
      case 'summarizer':
        return 0;
      case 'researcher':
      case 'coder':
      case 'tester':
      case 'custom':
      default:
        return 0;
    }
  }

  private defaultToMPurpose(archetype: SubAgentArchetype, task: string): string {
    switch (archetype) {
      case 'researcher':
        return 'Collect grounded facts from the project context.';
      case 'critic':
        return "Evaluate another agent or design result against Roy's goal and user intent.";
      case 'planner':
        return 'Turn context into a sequence of actionable steps.';
      case 'coder':
        return 'Implement scoped code changes.';
      case 'summarizer':
        return 'Condense results into a clear summary.';
      case 'tester':
        return 'Validate behavior and identify regressions.';
      case 'custom':
      default:
        return `Complete the assigned task: ${task}`;
    }
  }

  private isValidArchetype(value: string): value is SubAgentArchetype {
    return ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'].includes(value);
  }

  private handleQueueTransition(transition: QueueTransition): void {
    const message = transition.message;
    this.emit({
      type: transition.type,
      agentId: message.metadata?.agentId,
      data: {
        messageId: message.id,
        kind: message.kind,
        from: message.from,
        to: message.to,
        status: message.status,
        sessionId: message.sessionId,
        turnId: message.turnId,
        traceId: message.traceId,
        correlationId: message.correlationId,
        parentMessageId: message.parentMessageId,
        error: transition.error,
        reason: transition.reason,
      },
    });
  }

  private capitalize(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  private async drainAgentOutput(
    queue: { isEmpty(recipient: string): boolean; receive(recipient: string): Promise<{ sender: string; content: unknown } | undefined> },
    sender: string
  ): Promise<string> {
    const chunks: string[] = [];
    while (!queue.isEmpty('env')) {
      const message = await queue.receive('env');
      if (!message) break;
      if (message.sender === sender) {
        chunks.push(String(message.content));
      }
    }
    return chunks.join('');
  }
}

export const runtime = Runtime.getInstance();
export default Runtime;
