// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import path from 'node:path';
import { createHash } from 'node:crypto';
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
import type {
  AgentComputeNodeDefinition,
  AgentComputeNodeExecution,
  AgentComputeNodeRequest,
  AgentCreationInvocation,
  AgentNodeCreationMode,
} from '../skills/agentCreation.js';
import { UseToolWhenNeededSkill } from '../skills/toolUse.js';
import { DefaultDelegationCandidatePlanner, type DelegationCandidateSelection } from '../delegation/index.js';
import { ContextWindowManager, type ContextWindow } from '../context/index.js';
import { BudgetMarket, type BudgetAllocation, type BudgetMarketState } from '../budget/index.js';
import { TeamRegistry, type TeamFSMState, type TeamRuntimeState } from '../team/index.js';
import { ToolApprovalManager, type ToolApprovalRequest } from '../tools/approval.js';
import type { ToolResult } from '../tools/types.js';
import { AgentToolPlanner } from '../tools/planner.js';
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
  perTeam: Record<string, TokenUsage>;
  perTurn: TokenUsage[];
}

export interface RuntimeEvent {
  type: string;
  timestamp: number;
  agentId?: string;
  sessionId?: string;
  correlationId?: string;
  nodeId?: string;
  data?: Record<string, unknown>;
}

export interface RuntimeState {
  sessionId: string;
  rootAgentId: string;
  rootAgent: AgentInfo;
  agents: AgentInfo[];
  teams: TeamRuntimeState[];
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
  outputContract?: AgentComputeNodeRequest['outputContract'];
  correlationId?: string;
  teamId?: string;
  nodeDefinition?: AgentComputeNodeDefinition;
}

export interface AgentTreeNode {
  agent: AgentInfo;
  children: AgentTreeNode[];
}

export interface TeamMemberSpec {
  archetype: SubAgentArchetype;
  task: string;
  name?: string;
  role?: string;
  style?: string;
  tools?: string[];
  skills?: string[];
  budgetTokens?: number;
  tomLevel?: number;
  lead?: boolean;
}

export interface SpawnTeamSpec {
  parentAgentId?: string;
  name: string;
  description: string;
  tomLevel?: number;
  leadAgentId?: string;
  task?: string;
  members?: TeamMemberSpec[];
  correlationId?: string;
}

export interface TeamTreeNode {
  team: TeamRuntimeState;
  members: AgentTreeNode[];
}

export type RuntimeActorNode = RuntimeAgentActorNode | RuntimeTeamActorNode;

export interface RuntimeAgentActorNode {
  type: 'agent';
  agent: AgentInfo;
  children: RuntimeActorNode[];
}

export interface RuntimeTeamActorNode {
  type: 'team';
  team: TeamRuntimeState;
  children: RuntimeActorNode[];
}

export interface RuntimeActorTree {
  root: AgentInfo;
  teams: TeamTreeNode[];
  hierarchy: RuntimeAgentActorNode;
}

export interface TeamRunResult {
  team: TeamRuntimeState;
  result: string;
  members: RunAgentResult[];
  memberExecutions: RootMediatedSpawnResult[];
  correlationId: string;
  messages: RuntimeMessage[];
  usage: TokenUsage;
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
  memoryScope?: AgentMemoryScope;
  spawnPolicy?: Partial<AgentSpawnPolicy>;
  tomProfile?: ToMProfile;
  reuseMode?: NonNullable<AgentComputeNodeRequest['reuse']>['mode'];
  outputContract?: AgentComputeNodeRequest['outputContract'];
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
  disableRecursiveDelegation?: boolean;
  teamId?: string;
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
  node: AgentComputeNodeDefinition;
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
  teams: TeamRunResult[];
  messages: RuntimeMessage[];
  usage: {
    root: TokenUsage;
    subagents: Record<string, TokenUsage>;
    teamSynthesis: Record<string, TokenUsage>;
    total: TokenUsage;
  };
}

export interface AgentCreationUsage {
  mode: AgentNodeCreationMode;
  nodeId?: string;
  definitionFingerprint?: string;
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
  private contextWindowManager: ContextWindowManager | null = null;
  private agentFsms = new Map<string, FSM>();
  private budgetMarket: BudgetMarket | null = null;
  private agentBudgetAllocations = new Map<string, string>();
  private agentBudgetLimits = new Map<string, number>();
  private toolApprovalManager: ToolApprovalManager | null = null;
  private toolCallCounts = new Map<string, number>();
  private readonly teams = new TeamRegistry();
  private teamMemberPlans = new Map<string, TeamMemberSpec[]>();
  private readonly toolPlanner = new AgentToolPlanner();
  private candidatePlanner: DefaultDelegationCandidatePlanner | null = null;
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
      strict: true,
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
      onInvalidTransition: (from, to) => {
        this.emit({ type: 'fsm.invalid_transition', agentId: 'root', data: { from, to } });
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
    this.candidatePlanner = new DefaultDelegationCandidatePlanner({
      llm,
      enabledScorers: this.workspaceRuntimeConfig.delegation.candidateScoring.enabledScorers,
      minimumScore: this.workspaceRuntimeConfig.delegation.candidateScoring.minimumScore,
    });
    this.budgetMarket = new BudgetMarket(() => this.ctx ? this.getBudgetState().usedTokens : 0);
    this.budgetMarket.configure(options.budget ?? null);
    this.toolApprovalManager = new ToolApprovalManager(this.workspaceRuntimeConfig.tools.approval);
    const contextWindowManager = new ContextWindowManager(memory, this.workspaceRuntimeConfig.context);
    this.contextWindowManager = contextWindowManager;
    const rootMemory = await memory.loadAgentMemory('roy');
    const rootWindow = await contextWindowManager.build({
      sessionId: options.sessionId ?? 'main',
      agentId: 'root',
      agentKey: 'roy',
      role: 'root',
      task: 'Operate as the root agent for the current Roy runtime session.',
      memoryScope: this.getDefaultMemoryScope('root'),
    });
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
      publicContext: [rootWindow.publicContext, rootWindow.sessionContext].filter(Boolean).join('\n\n'),
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
    this.agentFsms.set('root', fsm);

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
    this.contextWindowManager = null;
    this.candidatePlanner = null;
    this.agentBindings.clear();
    this.agentFsms.clear();
    this.agentBudgetAllocations.clear();
    this.agentBudgetLimits.clear();
    this.budgetMarket = null;
    this.toolApprovalManager = null;
    this.toolCallCounts.clear();
    this.teams.clear();
    this.teamMemberPlans.clear();
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
    skillRegistry.unregister('use_tool_when_needed');
    skillRegistry.registerSystem(new DelegateToSubagentSkill(this));
    skillRegistry.register(new UseToolWhenNeededSkill(
      (agentId, toolName, params, reason) => this.executeToolForAgent(agentId, toolName, params, { reason })
    ));
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
      teams: this.teams.list(),
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
    const perTeam: Record<string, TokenUsage> = {};
    let usedTokens = 0;

    for (const agent of agents) {
      const usage = this.toTokenUsage(agent.usage);
      perAgent[agent.identity.id] = usage;
      usedTokens += usage.totalTokens;
    }
    for (const team of this.teams.list()) {
      perTeam[team.identity.id] = { ...team.tokenUsage };
      usedTokens += team.synthesisUsage.totalTokens;
    }

    return {
      mode: fsmCtx.budget === null ? 'unlimited' : 'limited',
      limitTokens: fsmCtx.budget ?? undefined,
      usedTokens,
      remainingTokens: fsmCtx.budget === null ? undefined : Math.max(0, fsmCtx.budget - usedTokens),
      perAgent,
      perTeam,
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
    this.budgetMarket?.configure(limitTokens);
    return this.getBudgetState();
  }

  getBudgetMarketState(): BudgetMarketState {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    return this.budgetMarket.getState();
  }

  getTeams(): TeamRuntimeState[] {
    return this.teams.list();
  }

  getTeam(teamId: string): TeamRuntimeState | undefined {
    return this.teams.get(teamId);
  }

  getTeamState(teamId: string): TeamRuntimeState | undefined {
    return this.getTeam(teamId);
  }

  getTeamTree(teamId: string): TeamTreeNode | undefined {
    const team = this.teams.get(teamId);
    if (!team) return undefined;
    const ctx = this.getContext();
    return {
      team,
      members: team.memberAgentIds
        .map(agentId => ctx.manager.getAgentById(agentId)?.getInfo())
        .filter((agent): agent is AgentInfo => Boolean(agent))
        .map(agent => this.buildAgentTree(agent)),
    };
  }

  getTeamActorTree(): RuntimeActorTree {
    return {
      root: this.getContext().agent.getInfo(),
      teams: this.teams.list()
        .map(team => this.getTeamTree(team.identity.id))
        .filter((team): team is TeamTreeNode => Boolean(team)),
      hierarchy: this.buildRuntimeAgentActorTree(this.getContext().agent.getInfo(), new Set()),
    };
  }

  async spawnTeam(spec: SpawnTeamSpec): Promise<TeamRuntimeState> {
    const ctx = this.getContext();
    const parentAgentId = spec.parentAgentId ?? 'root';
    const correlationId = spec.correlationId ?? this.createCorrelationId();
    const request = await this.enqueueMessage({
      kind: 'team.create.request',
      sessionId: ctx.sessionId,
      from: parentAgentId,
      to: 'runtime',
      correlationId,
      payload: { ...spec, parentAgentId },
      metadata: { agentId: parentAgentId, tomLevel: spec.tomLevel ?? 2 },
    });
    await this.processQueuedMessage(request.id);

    let createdTeamId: string | undefined;
    try {
      if (this.workspaceRuntimeConfig?.teams.enabled === false) throw new Error('Subteams are disabled by workspace policy');
      const parent = ctx.manager.getAgentById(parentAgentId);
      if (!parent) throw new Error(`Parent agent "${parentAgentId}" not found`);
      if (!spec.name?.trim()) throw new Error('Team name is required');
      if (!spec.description?.trim()) throw new Error('Team description is required');
      if (spec.task !== undefined && (typeof spec.task !== 'string' || !spec.task.trim())) {
        throw new Error('Team task must be a non-empty string when provided');
      }
      if (spec.tomLevel !== undefined
        && (!Number.isInteger(spec.tomLevel) || spec.tomLevel < 0 || spec.tomLevel > 3)) {
        throw new Error('Team tomLevel must be an integer from 0 to 3');
      }
      const requestedMembers = spec.members ?? [];
      for (const member of requestedMembers) {
        if (!member || typeof member !== 'object') throw new Error('Every planned team member must be an object');
        if (!this.isValidArchetype(member.archetype)) throw new Error(`Unsupported team member archetype "${member.archetype}"`);
        if (typeof member.task !== 'string' || !member.task.trim()) throw new Error('Every planned team member requires a task');
        if (member.name !== undefined && (typeof member.name !== 'string' || !member.name.trim())) {
          throw new Error('Team member name must be a non-empty string when provided');
        }
        if (member.tools !== undefined
          && (!Array.isArray(member.tools) || member.tools.some(tool => typeof tool !== 'string' || !tool.trim()))) {
          throw new Error('Team member tools must be an array of non-empty strings');
        }
        if (member.skills !== undefined
          && (!Array.isArray(member.skills) || member.skills.some(skill => typeof skill !== 'string' || !skill.trim()))) {
          throw new Error('Team member skills must be an array of non-empty strings');
        }
        if (member.budgetTokens !== undefined
          && (!Number.isFinite(member.budgetTokens) || member.budgetTokens <= 0)) {
          throw new Error('Team member budgetTokens must be a positive number when provided');
        }
        if (member.tomLevel !== undefined
          && (!Number.isInteger(member.tomLevel) || member.tomLevel < 0 || member.tomLevel > 3)) {
          throw new Error('Team member tomLevel must be an integer from 0 to 3');
        }
      }
      const members = requestedMembers.map(member => ({
        ...member,
        tools: member.tools ?? this.getDefaultToolBindings(member.archetype).map(binding => binding.name),
        skills: member.skills ?? this.getDefaultSkillBindings(member.archetype).map(binding => binding.name),
        tomLevel: member.tomLevel ?? this.createSubagentToMProfile(
          member.archetype,
          '',
          member.task,
          parentAgentId
        ).level,
      }));
      const maxMembers = this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5;
      if (members.length > maxMembers) {
        throw new Error(`Team member limit exceeded: requested ${members.length}, maximum ${maxMembers}`);
      }
      const parentPolicy = this.getAgentPolicy(parentAgentId);
      if (!parentPolicy?.spawnPolicy.canSpawn) {
        throw new Error(`Agent "${parentAgentId}" is not authorized to create subteams`);
      }
      const parentFsmState = parentAgentId === 'root'
        ? ctx.fsm.getState()
        : this.agentFsms.get(parentAgentId)?.getState();
      if (!parentFsmState || !parentPolicy.spawnPolicy.allowedStates.includes(parentFsmState)) {
        throw new Error(`Agent "${parentAgentId}" cannot create a subteam in FSM state "${parentFsmState ?? 'unknown'}"`);
      }
      const team = this.teams.create({
        name: spec.name,
        parentAgentId,
        description: spec.description,
        generation: parent.getIdentity().generation + 1,
        tomLevel: spec.tomLevel ?? 2,
        leadAgentId: spec.leadAgentId,
        task: spec.task,
        correlationId,
      });
      createdTeamId = team.identity.id;
      this.teamMemberPlans.set(team.identity.id, members.map(member => ({ ...member })));
      const teamKey = this.safeAgentKey(spec.name);
      await ctx.memory.ensureTeamMemory(teamKey, { name: spec.name, purpose: spec.description });
      const cachedTeamPattern = (await ctx.memory.getCachePatterns('teams'))
        .find(item => item.id === `team_pattern_${teamKey}_v1` || item.key === teamKey);
      if (cachedTeamPattern) {
        this.emit({
          type: 'cache.hit',
          agentId: parentAgentId,
          sessionId: ctx.sessionId,
          correlationId,
          data: { cacheType: 'team-pattern', patternId: cachedTeamPattern.id, teamKey },
        });
      }
      const pattern = await ctx.memory.upsertTeamPattern({
        key: teamKey,
        name: spec.name,
        purpose: spec.description,
        parentId: parentAgentId,
        memberArchetypes: members.map(member => member.archetype),
        tomLevel: spec.tomLevel ?? 2,
        leadArchetype: members.find(member => member.lead)?.archetype,
        members: members.map(member => ({
          archetype: member.archetype,
          name: member.name,
          role: member.role,
          tools: member.tools,
          skills: member.skills,
          tomLevel: member.tomLevel,
          lead: member.lead ?? false,
        })),
      });
      await ctx.memory.writeTeamTopology(teamKey, {
        type: 'subteam',
        teamId: team.identity.id,
        parentAgentId,
        leadAgentId: spec.leadAgentId,
        members: [],
        plannedMembers: members,
        tomLevel: team.identity.tomLevel,
        updatedAt: new Date().toISOString(),
      });
      const approved = await this.enqueueMessage({
        kind: 'team.create.approved',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: parentAgentId,
        correlationId,
        parentMessageId: request.id,
        payload: team,
        metadata: { agentId: parentAgentId, teamId: team.identity.id, tomLevel: team.identity.tomLevel },
      });
      await this.processQueuedMessage(approved.id);
      await ctx.queue.ack(request.id);
      await ctx.queue.ack(approved.id);
      this.emit({
        type: 'team.created',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          name: spec.name,
          description: spec.description,
          tomLevel: team.identity.tomLevel,
          plannedMembers: members.length,
          patternId: pattern.id,
          parentAgentId,
        },
      });
      return this.teams.get(team.identity.id)!;
    } catch (error) {
      const failure = error instanceof Error ? error : new Error(String(error));
      const current = await ctx.queue.getMessage(request.id);
      if (current?.status === 'processing' || current?.status === 'pending') {
        await ctx.queue.fail(request.id, failure);
      }
      if (createdTeamId) {
        try {
          await this.transitionTeamFsm(createdTeamId, 'S_team_failed', { error: failure.message });
        } catch {
          // The original creation failure is the actionable error.
        }
      }
      const rejected = await this.enqueueMessage({
        kind: 'team.create.rejected',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: parentAgentId,
        correlationId,
        parentMessageId: request.id,
        payload: { parentAgentId, teamId: createdTeamId, reason: failure.message },
        metadata: { agentId: createdTeamId ?? parentAgentId, teamId: createdTeamId },
      });
      await this.processQueuedMessage(rejected.id);
      await ctx.queue.ack(rejected.id);
      this.emit({
        type: createdTeamId ? 'team.create.failed' : 'team.create.rejected',
        agentId: createdTeamId ?? parentAgentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: createdTeamId, parentAgentId, error: failure.message },
      });
      throw failure;
    }
  }

  async createSubteam(input: {
    parentId: string;
    name: string;
    purpose: string;
    memberArchetypes: SubAgentArchetype[];
    correlationId?: string;
  }): Promise<TeamRuntimeState> {
    return this.spawnTeam({
      parentAgentId: input.parentId,
      name: input.name,
      description: input.purpose,
      members: input.memberArchetypes.map(archetype => ({ archetype, task: input.purpose })),
      correlationId: input.correlationId,
    });
  }

  async spawnAgentIntoTeam(teamId: string, spec: TeamMemberSpec): Promise<TeamRuntimeState> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    if (team.status === 'done' || team.status === 'failed') {
      throw new Error(`Team "${teamId}" cannot add members while status is ${team.status}`);
    }
    const maxMembers = this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5;
    const plans = this.teamMemberPlans.get(teamId) ?? [];
    if (team.memberAgentIds.length + plans.length >= maxMembers) {
      throw new Error(`Team "${teamId}" already has the maximum ${maxMembers} members`);
    }
    if (!this.isValidArchetype(spec.archetype)) throw new Error(`Unsupported team member archetype "${spec.archetype}"`);
    if (!spec.task.trim()) throw new Error('Team member task is required');
    const normalizedSpec: TeamMemberSpec = {
      ...spec,
      tools: spec.tools ?? this.getDefaultToolBindings(spec.archetype).map(binding => binding.name),
      skills: spec.skills ?? this.getDefaultSkillBindings(spec.archetype).map(binding => binding.name),
      tomLevel: spec.tomLevel ?? this.createSubagentToMProfile(
        spec.archetype,
        '',
        spec.task,
        team.identity.parentAgentId
      ).level,
    };
    const nextPlans = [...plans, normalizedSpec];
    this.teamMemberPlans.set(teamId, nextPlans);
    await this.getContext().memory.updateTeamPatternMembers(this.safeAgentKey(team.identity.name), {
      memberArchetypes: nextPlans.map(member => member.archetype),
      leadArchetype: nextPlans.find(member => member.lead)?.archetype,
      tomLevel: team.identity.tomLevel,
      members: nextPlans.map(member => ({
        archetype: member.archetype,
        name: member.name,
        role: member.role,
        tools: member.tools,
        skills: member.skills,
        tomLevel: member.tomLevel,
        lead: member.lead ?? false,
      })),
    });
    await this.persistTeamTopology(team);
    this.emit({
      type: 'team.member.planned',
      agentId: team.identity.id,
      sessionId: this.getContext().sessionId,
      correlationId: team.correlationId,
      data: { teamId, archetype: spec.archetype, name: spec.name, task: spec.task, lead: spec.lead ?? false },
    });
    return this.teams.get(teamId)!;
  }

  private async executeTeamMember(teamId: string, spec: TeamMemberSpec): Promise<RootMediatedSpawnResult> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    if (team.fsmState !== 'S_member_execute') {
      throw new Error(`Team "${teamId}" cannot execute a member in FSM state "${team.fsmState}"`);
    }
    const execution = await this.createAgentComputeNode({
      parentId: team.identity.parentAgentId,
      archetype: spec.archetype,
      task: spec.task,
      name: spec.name,
      role: spec.role,
      style: spec.style,
      tools: spec.tools,
      skills: spec.skills,
      budgetTokens: spec.budgetTokens,
      tomProfile: spec.tomLevel === undefined
        ? undefined
        : {
          ...this.createSubagentToMProfile(spec.archetype, '', spec.task, team.identity.parentAgentId),
          level: spec.tomLevel as ToMProfile['level'],
        },
      execution: {
        requireParentSynthesis: false,
        showSubagentOutput: false,
        disableRecursiveDelegation: true,
        teamId,
      },
    }, {
      agentId: team.identity.parentAgentId,
      sessionId: this.getContext().sessionId,
      source: teamId,
    }, team.correlationId);
    const result = execution.delegation;
    const updated = this.teams.addMember(teamId, result.agent.identity.id, spec.task, spec.lead);
    this.teams.recordMemberResult(
      teamId,
      result.agent.identity.id,
      spec.task,
      result.subagentResult.result,
      result.subagentResult.usage
    );
    await this.persistTeamTopology(updated);
    this.emit({
      type: 'team.member.completed',
      agentId: result.agent.identity.id,
      sessionId: this.getContext().sessionId,
      correlationId: result.correlationId,
      data: {
        teamId,
        parentAgentId: team.identity.parentAgentId,
        task: spec.task,
        totalTokens: result.subagentResult.usage.totalTokens,
      },
    });
    return result;
  }

  async runTeam(teamId: string, task: string): Promise<TeamRunResult> {
    const ctx = this.getContext();
    const initial = this.teams.get(teamId);
    if (!initial) throw new Error(`Team "${teamId}" not found`);
    if (!task.trim()) throw new Error('Team task is required');
    const usageBefore = { ...initial.tokenUsage };
    const correlationId = initial.correlationId ?? this.createCorrelationId();
    this.teams.setTask(teamId, task, correlationId);
    await this.transitionTeamFsm(teamId, 'S_team_plan', { task });

    const taskMessage = await this.enqueueMessage({
      kind: 'team.task',
      sessionId: ctx.sessionId,
      from: initial.identity.parentAgentId,
      to: teamId,
      correlationId,
      payload: { teamId, task },
      metadata: { agentId: initial.identity.parentAgentId, teamId, tomLevel: initial.identity.tomLevel },
    });
    await this.processQueuedMessage(taskMessage.id);

    const memberExecutions: RootMediatedSpawnResult[] = [];
    const members: RunAgentResult[] = [];
    try {
      const plans = this.teamMemberPlans.get(teamId) ?? [];
      if (plans.length > 0) {
        await this.transitionTeamFsm(teamId, 'S_member_spawn', { count: plans.length });
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: plans.length });
        for (const plan of plans) {
          const execution = await this.executeTeamMember(teamId, plan);
          memberExecutions.push(execution);
          members.push(execution.subagentResult);
        }
        this.teamMemberPlans.set(teamId, []);
      } else {
        const team = this.teams.get(teamId)!;
        if (team.memberAgentIds.length === 0) throw new Error(`Team "${teamId}" has no members or member plans`);
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: team.memberAgentIds.length });
        for (const agentId of team.memberAgentIds) {
          const memberTask = team.memberTasks[agentId] ?? task;
          const result = await this.runAgent(agentId, memberTask, {
            correlationId,
            disableRecursiveDelegation: true,
          });
          members.push(result);
          this.teams.recordMemberResult(teamId, agentId, memberTask, result.result, result.usage);
          this.emit({
            type: 'team.member.completed',
            agentId,
            sessionId: ctx.sessionId,
            correlationId,
            data: { teamId, task: memberTask, totalTokens: result.usage.totalTokens },
          });
        }
      }

      await this.transitionTeamFsm(teamId, 'S_member_aggregate', { completed: members.length });
      await this.transitionTeamFsm(teamId, 'S_team_synthesize', { completed: members.length });
      const synthesis = await this.completeAsTeam(this.teams.get(teamId)!, task, members, correlationId);
      this.teams.recordSynthesis(teamId, synthesis.content, synthesis.usage);

      const resultMessage = await this.enqueueMessage({
        kind: 'team.result',
        sessionId: ctx.sessionId,
        from: teamId,
        to: initial.identity.parentAgentId,
        correlationId,
        parentMessageId: taskMessage.id,
        payload: {
          teamId,
          task,
          result: synthesis.content,
          memberAgentIds: this.teams.get(teamId)!.memberAgentIds,
        },
        metadata: { agentId: initial.identity.parentAgentId, teamId, tomLevel: initial.identity.tomLevel },
      });
      await this.processQueuedMessage(resultMessage.id);
      await ctx.queue.ack(taskMessage.id);
      await ctx.queue.ack(resultMessage.id);
      await this.transitionTeamFsm(teamId, 'S_team_done', { totalTokens: this.teams.get(teamId)!.tokenUsage.totalTokens });
      const completedTeam = this.teams.get(teamId)!;
      const runUsage = this.subtractTokenUsage(completedTeam.tokenUsage, usageBefore);
      await this.persistTeamRunArtifacts({
        team: completedTeam,
        task,
        result: synthesis.content,
        correlationId,
        usage: runUsage,
        success: true,
      });
      this.emit({
        type: 'team.completed',
        agentId: teamId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId,
          result: synthesis.content,
          memberAgentIds: this.teams.get(teamId)!.memberAgentIds,
          totalTokens: runUsage.totalTokens,
          cumulativeTokens: this.teams.get(teamId)!.tokenUsage.totalTokens,
          parentAgentId: initial.identity.parentAgentId,
        },
      });
      return {
        team: this.teams.get(teamId)!,
        result: synthesis.content,
        members,
        memberExecutions,
        correlationId,
        messages: await this.getMessages({ correlationId }),
        usage: runUsage,
      };
    } catch (error) {
      const current = await ctx.queue.getMessage(taskMessage.id);
      if (current?.status === 'pending' || current?.status === 'processing') {
        await ctx.queue.fail(taskMessage.id, error instanceof Error ? error : new Error(String(error)));
      }
      await this.transitionTeamFsm(teamId, 'S_team_failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      const failedTeam = this.teams.get(teamId)!;
      await this.persistTeamRunArtifacts({
        team: failedTeam,
        task,
        correlationId,
        usage: this.subtractTokenUsage(failedTeam.tokenUsage, usageBefore),
        success: false,
      });
      this.emit({
        type: 'team.failed',
        agentId: teamId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId, parentAgentId: initial.identity.parentAgentId, error: error instanceof Error ? error.message : String(error) },
      });
      throw error;
    }
  }

  getToolApprovals(status?: ToolApprovalRequest['status']): ToolApprovalRequest[] {
    if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
    return this.toolApprovalManager.list(status);
  }

  async executeToolForAgent(
    agentId: string,
    toolName: string,
    params: Record<string, unknown>,
    options: { reason?: string; approvalId?: string; correlationId?: string; nodeId?: string } = {}
  ): Promise<ToolResult> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) return { success: false, error: `Agent "${agentId}" not found` };
    const binding = (this.agentBindings.get(agentId)?.tools ?? []).find(item => item.name === toolName && item.enabled);
    if (!binding) return { success: false, error: `Tool "${toolName}" is not authorized for agent "${agentId}"` };
    if (!toolRegistry.has(toolName)) return { success: false, error: `Tool "${toolName}" not found` };

    const callKey = `${agentId}:${toolName}`;
    const calls = this.toolCallCounts.get(callKey) ?? 0;
    const maxCalls = binding.constraints?.maxCalls;
    if (maxCalls !== undefined && calls >= maxCalls) {
      return { success: false, error: `Tool call limit reached for ${toolName}`, metadata: { maxCalls } };
    }

    let approved = false;
    if (options.approvalId) {
      const prior = this.toolApprovalManager?.get(options.approvalId);
      approved = prior?.status === 'approved' && prior.agentId === agentId && prior.toolName === toolName;
    }
    if (!approved) {
      if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
      const authorization = this.toolApprovalManager.authorize({
        agentId,
        toolName,
        permission: binding.permission,
        params,
        reason: options.reason,
      });
      const approvalMessage = await this.enqueueMessage({
        kind: 'tool.approval.request',
        sessionId: ctx.sessionId,
        from: agentId,
        to: 'runtime.approval',
        correlationId: options.correlationId,
        payload: authorization.request,
        metadata: { agentId, nodeId: options.nodeId },
      });
      await this.processQueuedMessage(approvalMessage.id);
      await ctx.queue.ack(approvalMessage.id);
      this.emit({
        type: 'tool.approval.requested',
        agentId,
        data: {
          approvalId: authorization.request.id,
          toolName,
          permission: binding.permission,
          decision: authorization.decision,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
        },
      });
      if (authorization.decision !== 'pending') {
        const resolvedMessage = await this.enqueueMessage({
          kind: 'tool.approval.resolved',
          sessionId: ctx.sessionId,
          from: 'runtime.approval',
          to: agentId,
          correlationId: options.correlationId,
          parentMessageId: approvalMessage.id,
          payload: authorization.request,
          metadata: { agentId, nodeId: options.nodeId },
        });
        await this.processQueuedMessage(resolvedMessage.id);
        await ctx.queue.ack(resolvedMessage.id);
      }
      if (authorization.decision === 'pending') {
        return {
          success: false,
          error: `Tool approval required for ${toolName}`,
          metadata: { pendingApproval: true, approvalId: authorization.request.id },
        };
      }
      if (authorization.decision === 'denied') {
        return { success: false, error: `Tool policy denied ${toolName}`, metadata: { approvalId: authorization.request.id } };
      }
    }

    const toolCall = await this.enqueueMessage({
      kind: 'tool.call',
      sessionId: ctx.sessionId,
      from: agentId,
      to: `tool.${toolName}`,
      correlationId: options.correlationId,
      payload: { toolName, params, reason: options.reason },
      metadata: { agentId, nodeId: options.nodeId },
    });
    await this.processQueuedMessage(toolCall.id);
    this.emit({
      type: 'tool.call',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: { toolName, params, correlationId: options.correlationId },
    });
    const result = await toolRegistry.execute(toolName, params);
    this.toolCallCounts.set(callKey, calls + 1);
    const resultMessage = await this.enqueueMessage({
      kind: 'tool.result',
      sessionId: ctx.sessionId,
      from: `tool.${toolName}`,
      to: agentId,
      correlationId: options.correlationId,
      parentMessageId: toolCall.id,
      payload: result,
      metadata: { agentId, nodeId: options.nodeId },
    });
    await this.processQueuedMessage(resultMessage.id);
    if (result.success) await ctx.queue.ack(toolCall.id);
    else await ctx.queue.fail(toolCall.id, new Error(result.error ?? 'tool_failed'));
    await ctx.queue.ack(resultMessage.id);
    this.emit({
      type: result.success ? 'tool.result' : 'tool.error',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: { toolName, correlationId: options.correlationId, success: result.success, error: result.error },
    });
    return result;
  }

  async resolveToolApproval(id: string, decision: 'approved' | 'denied'): Promise<ToolApprovalRequest | undefined> {
    if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
    const request = this.toolApprovalManager.resolve(id, decision);
    if (request) {
      const ctx = this.getContext();
      const message = await this.enqueueMessage({
        kind: 'tool.approval.resolved',
        sessionId: ctx.sessionId,
        from: 'runtime.approval',
        to: request.agentId,
        payload: request,
        metadata: { agentId: request.agentId },
      });
      await this.processQueuedMessage(message.id);
      await ctx.queue.ack(message.id);
      this.emit({
        type: 'tool.approval.resolved',
        agentId: request.agentId,
        data: { approvalId: id, decision, toolName: request.toolName },
      });
    }
    return request;
  }

  private async requestAgentBudget(input: {
    parentId: string;
    archetype: SubAgentArchetype;
    correlationId?: string;
    nodeId?: string;
    requestedTokens?: number;
    purpose: string;
  }): Promise<BudgetAllocation | undefined> {
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = input.requestedTokens ?? this.estimateAgentBudget(input.archetype);
    const requestMessage = await this.enqueueMessage({
      kind: 'budget.request',
      sessionId: this.getContext().sessionId,
      from: input.parentId,
      to: 'budget.market',
      correlationId: input.correlationId,
      payload: { ...input, requestedTokens },
      metadata: { agentId: input.parentId, budgetTokens: requestedTokens, nodeId: input.nodeId },
    });
    await this.processQueuedMessage(requestMessage.id);
    this.emit({
      type: 'budget.requested',
      agentId: input.parentId,
      correlationId: input.correlationId,
      nodeId: input.nodeId,
      data: { correlationId: input.correlationId, archetype: input.archetype, requestedTokens, purpose: input.purpose },
    });
    const allocation = this.budgetMarket.request({
      requesterId: `${input.parentId}:${input.archetype}`,
      parentId: input.parentId,
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens: this.workspaceRuntimeConfig?.budgetMarket.minimumGrantTokens ?? 256,
      purpose: input.purpose,
    });
    const responseMessage = await this.enqueueMessage({
      kind: allocation.status === 'granted' ? 'budget.grant' : 'budget.denied',
      sessionId: this.getContext().sessionId,
      from: 'budget.market',
      to: input.parentId,
      correlationId: input.correlationId,
      parentMessageId: requestMessage.id,
      payload: allocation,
      metadata: { agentId: input.parentId, budgetTokens: allocation.grantedTokens, nodeId: input.nodeId },
    });
    await this.processQueuedMessage(responseMessage.id);
    await this.getContext().queue.ack(requestMessage.id);
    await this.getContext().queue.ack(responseMessage.id);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.granted' : 'budget.denied',
      agentId: input.parentId,
      correlationId: input.correlationId,
      nodeId: input.nodeId,
      data: {
        correlationId: input.correlationId,
        allocationId: allocation.id,
        requestedTokens,
        grantedTokens: allocation.grantedTokens,
        reason: allocation.reason,
      },
    });
    return allocation;
  }

  private async requestTeamSynthesisBudget(input: {
    team: TeamRuntimeState;
    correlationId: string;
    promptTokens: number;
    completionTokens: number;
  }): Promise<BudgetAllocation | undefined> {
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = input.promptTokens + input.completionTokens;
    const minimumTokens = Math.max(
      input.promptTokens + 1,
      this.workspaceRuntimeConfig?.budgetMarket.minimumGrantTokens ?? 256
    );
    const requestMessage = await this.enqueueMessage({
      kind: 'budget.request',
      sessionId: this.getContext().sessionId,
      from: input.team.identity.id,
      to: 'budget.market',
      correlationId: input.correlationId,
      payload: {
        requesterId: input.team.identity.id,
        parentId: input.team.identity.parentAgentId,
        requestedTokens,
        minimumTokens,
        purpose: 'team_synthesis',
      },
      metadata: {
        agentId: input.team.identity.id,
        teamId: input.team.identity.id,
        budgetTokens: requestedTokens,
      },
    });
    await this.processQueuedMessage(requestMessage.id);
    this.emit({
      type: 'budget.requested',
      agentId: input.team.identity.id,
      correlationId: input.correlationId,
      data: {
        teamId: input.team.identity.id,
        requestedTokens,
        minimumTokens,
        purpose: 'team_synthesis',
      },
    });
    const allocation = this.budgetMarket.request({
      requesterId: input.team.identity.id,
      parentId: input.team.identity.parentAgentId,
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens,
      purpose: 'team_synthesis',
    });
    const responseMessage = await this.enqueueMessage({
      kind: allocation.status === 'granted' ? 'budget.grant' : 'budget.denied',
      sessionId: this.getContext().sessionId,
      from: 'budget.market',
      to: input.team.identity.id,
      correlationId: input.correlationId,
      parentMessageId: requestMessage.id,
      payload: allocation,
      metadata: {
        agentId: input.team.identity.id,
        teamId: input.team.identity.id,
        budgetTokens: allocation.grantedTokens,
      },
    });
    await this.processQueuedMessage(responseMessage.id);
    await this.getContext().queue.ack(requestMessage.id);
    await this.getContext().queue.ack(responseMessage.id);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.granted' : 'budget.denied',
      agentId: input.team.identity.id,
      correlationId: input.correlationId,
      data: {
        teamId: input.team.identity.id,
        allocationId: allocation.id,
        requestedTokens,
        grantedTokens: allocation.grantedTokens,
        reason: allocation.reason,
      },
    });
    return allocation;
  }

  private settleTeamSynthesisBudget(
    teamId: string,
    allocation: BudgetAllocation | undefined,
    actualTokens: number,
    correlationId: string
  ): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const settled = this.budgetMarket.settle(allocation.id, actualTokens);
    if (!settled) return;
    this.emit({
      type: 'budget.settled',
      agentId: teamId,
      correlationId,
      data: {
        teamId,
        allocationId: allocation.id,
        grantedTokens: allocation.grantedTokens,
        actualTokens,
      },
    });
    if (actualTokens > allocation.grantedTokens) {
      this.emit({
        type: 'budget.overrun',
        agentId: teamId,
        correlationId,
        data: {
          teamId,
          allocationId: allocation.id,
          grantedTokens: allocation.grantedTokens,
          actualTokens,
        },
      });
    }
  }

  private releaseTeamSynthesisBudget(
    teamId: string,
    allocation: BudgetAllocation | undefined,
    correlationId: string,
    reason: string
  ): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const released = this.budgetMarket.release(allocation.id, reason);
    if (released) {
      this.emit({
        type: 'budget.released',
        agentId: teamId,
        correlationId,
        data: { teamId, allocationId: allocation.id, reason },
      });
    }
  }

  private settleAgentBudget(agentId: string, actualTokens: number): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.settle(allocationId, actualTokens);
    this.agentBudgetAllocations.delete(agentId);
    if (allocation) {
      this.emit({
        type: 'budget.settled',
        agentId,
        data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens },
      });
      if (actualTokens > allocation.grantedTokens) {
        this.emit({
          type: 'budget.overrun',
          agentId,
          data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens },
        });
      }
    }
  }

  private releaseAgentBudget(agentId: string, reason: string): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.release(allocationId, reason);
    this.agentBudgetAllocations.delete(agentId);
    if (allocation) this.emit({ type: 'budget.released', agentId, data: { allocationId, reason } });
  }

  private estimateAgentBudget(archetype: SubAgentArchetype): number {
    return {
      researcher: 2200,
      critic: 1600,
      planner: 1400,
      coder: 2600,
      summarizer: 1000,
      tester: 1800,
      custom: 1800,
    }[archetype];
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
      allowedStates: isRoot
        ? ['S_solo', 'S_delegate_planning', 'S_spawn_subagents']
        : ['S_planning', 'S_delegating'],
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

    if (parentInfo.state === 'failed' || parentInfo.state === 'stopped') {
      return { allowed: false, reason: 'invalid_fsm_state', currentChildren, allowedChildren, depth };
    }

    if (!parentBindings.spawnPolicy.canSpawn) {
      return { allowed: false, reason: 'spawn_disabled_for_parent', currentChildren, allowedChildren, depth };
    }
    const parentFsmState = input.parentId === 'root'
      ? ctx.fsm.getState()
      : this.agentFsms.get(input.parentId)?.getState();
    if (!parentFsmState || !parentBindings.spawnPolicy.allowedStates.includes(parentFsmState)) {
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

  private deriveTeamName(plans: DelegationAgentPlan[]): string {
    const archetypes = Array.from(new Set(plans.map(plan => plan.archetype)));
    const has = (archetype: SubAgentArchetype) => archetypes.includes(archetype);
    if (has('coder') && has('tester')) return 'ImplementationTeam';
    if (has('researcher') && has('critic') && has('summarizer')) return 'AnalysisTeam';
    if (has('researcher') && has('critic')) return 'ReviewTeam';
    if (has('researcher') && has('planner')) return 'DiscoveryTeam';
    const label = archetypes
      .map(archetype => archetype.charAt(0).toUpperCase() + archetype.slice(1))
      .join('');
    return `${label || 'Delegation'}Team`;
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

  async readTeamMemoryDoc(teamKey: string, doc = 'memory'): Promise<string> {
    return this.getContext().memory.readTeamDoc(teamKey, doc);
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

  async getEvolutionHistory(limit = 50): Promise<Array<Record<string, unknown>>> {
    return this.getContext().memory.readEvolutionHistory(limit);
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

    let finalResponse: string;
    const subagents: RootMediatedSpawnResult[] = [];
    const teamResults: TeamRunResult[] = [];

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
      }
      const plans = decision.agents.slice(0, 3);
      if (plans.length > 1 && this.workspaceRuntimeConfig?.teams.createForMultipleAgents !== false) {
        const team = await this.spawnTeam({
          parentAgentId: 'root',
          name: this.deriveTeamName(plans),
          description: userInput,
          task: userInput,
          members: plans.map((plan, index) => ({ ...plan, lead: index === 0 })),
          correlationId,
        });
        const teamResult = await this.runTeam(team.identity.id, userInput);
        teamResults.push(teamResult);
        subagents.push(...teamResult.memberExecutions);
      } else {
        for (const plan of plans) {
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
            disableRecursiveDelegation: this.getBudgetState().mode === 'limited',
          });
          subagents.push(result);
        }
      }
      await this.transitionRootTurnState('S_wait_subagents', {
        correlationId,
        completed: subagents.length,
      });
      await this.transitionRootTurnState('S_synthesize', { correlationId, completed: subagents.length });
      finalResponse = await this.synthesizeDelegatedResults(userInput, subagents, correlationId, teamResults);
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
    const teamSynthesisUsage = Object.fromEntries(
      teamResults.map(result => [result.team.identity.id, { ...result.team.synthesisUsage }])
    );

    return {
      correlationId,
      decision,
      finalResponse,
      subagents,
      teams: teamResults,
      messages: await this.getMessages({ correlationId }),
      usage: {
        root: rootUsage,
        subagents: subagentUsage,
        teamSynthesis: teamSynthesisUsage,
        total: this.sumUsage([rootUsage, ...Object.values(subagentUsage), ...Object.values(teamSynthesisUsage)]),
      },
    };
  }

  async handleSpawnCommand(payload: SpawnCommandPayload): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const execution = await this.createAgentComputeNode({
      parentId: payload.parentId,
      archetype: payload.archetype,
      task: payload.task,
      name: payload.name,
      role: payload.customRole,
      style: payload.customStyle,
      tools: payload.tools,
      skills: payload.skills,
      budgetTokens: payload.budgetTokens,
      memoryScope: payload.memoryScope,
      spawnPolicy: payload.spawnPolicy,
      tomProfile: payload.tomProfile ?? (payload.tomLevel === undefined
        ? undefined
        : { ...this.createSubagentToMProfile(payload.archetype, '', payload.task, payload.parentId ?? 'root'), level: payload.tomLevel as ToMProfile['level'] }),
      reuse: { mode: payload.reuseMode ?? 'prefer_cache' },
      outputContract: payload.outputContract,
      execution: {
        requireParentSynthesis: payload.requireRootSynthesis ?? true,
        showSubagentOutput: payload.showSubagentOutput ?? false,
        disableRecursiveDelegation: payload.disableRecursiveDelegation ?? false,
        teamId: payload.teamId,
      },
    }, {
      agentId: payload.parentId ?? 'root',
      sessionId: ctx.sessionId,
      source: payload.source ?? 'cli',
    }, payload.correlationId);
    return execution.delegation;
  }

  async createAgentComputeNode(
    request: AgentComputeNodeRequest,
    invocation: AgentCreationInvocation,
    requestedCorrelationId?: string
  ): Promise<AgentComputeNodeExecution> {
    const ctx = this.getContext();
    if (invocation.sessionId !== ctx.sessionId) {
      throw new Error(`Agent creation session mismatch: expected "${ctx.sessionId}", received "${invocation.sessionId}"`);
    }
    const parentId = request.parentId ?? invocation.agentId;
    if (parentId !== invocation.agentId) {
      throw new Error(`Agent creation parent mismatch: "${invocation.agentId}" cannot create a child for "${parentId}"`);
    }
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) throw new Error(`Parent agent "${parentId}" not found`);
    const parentBindings = this.agentBindings.get(parentId);
    const delegationBinding = parentBindings?.skills.find(binding => binding.name === 'delegate_to_subagent' && binding.enabled);
    if (!delegationBinding) {
      throw new Error(`Agent "${parentId}" is not authorized to use delegate_to_subagent`);
    }

    const correlationId = requestedCorrelationId ?? this.createCorrelationId();
    const eventStart = this.events.length;
    let node: AgentComputeNodeDefinition;
    try {
      node = await this.resolveAgentComputeNode(request, invocation, correlationId);
    } catch (error) {
      this.emit({
        type: 'agent.node.resolve.failed',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          sessionId: ctx.sessionId,
          correlationId,
          archetype: request.archetype,
          error: error instanceof Error ? error.message : String(error),
        },
      });
      throw error;
    }
    const rootUsageBefore = ctx.agent.getUsage();
    this.emitNodeEvent('agent.node.execution.started', node, parentId, {
      archetype: node.identity.archetype,
      creationMode: node.reuse.creationMode,
    });

    try {
      const delegation = await this.executeAgentComputeNode(node);
      const rootUsageAfter = ctx.agent.getUsage();
      const rootUsage = this.toTokenUsage({
        llmCalls: rootUsageAfter.llmCalls - rootUsageBefore.llmCalls,
        promptTokens: rootUsageAfter.promptTokens - rootUsageBefore.promptTokens,
        completionTokens: rootUsageAfter.completionTokens - rootUsageBefore.completionTokens,
        totalTokens: rootUsageAfter.totalTokens - rootUsageBefore.totalTokens,
      });
      const subagentUsage = delegation.subagentResult.usage;
      const totalUsage = this.sumUsage([rootUsage, subagentUsage]);
      this.emitNodeEvent('agent.node.execution.completed', node, delegation.agent.identity.id, {
        totalTokens: totalUsage.totalTokens,
        grounded: delegation.subagentResult.grounded,
      });
      return {
        node,
        delegation,
        tokenUsage: { root: rootUsage, subagent: subagentUsage, total: totalUsage },
        events: this.getEvents().slice(eventStart).filter(event => this.eventCorrelationId(event) === correlationId),
      };
    } catch (error) {
      this.emitNodeEvent('agent.node.execution.failed', node, parentId, {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  private async resolveAgentComputeNode(
    request: AgentComputeNodeRequest,
    invocation: AgentCreationInvocation,
    correlationId: string
  ): Promise<AgentComputeNodeDefinition> {
    const ctx = this.getContext();
    if (!this.isValidArchetype(request.archetype)) {
      throw new Error(`Unsupported subagent archetype "${request.archetype}"`);
    }
    if (!request.task.trim()) throw new Error('Agent compute node task is required');
    this.validateAgentNodeRequest(request);

    const parentId = request.parentId ?? invocation.agentId;
    const reuseMode = request.reuse?.mode ?? 'prefer_cache';
    const patternKey = request.archetype === 'custom' && request.name
      ? this.safeAgentKey(request.name)
      : request.archetype;
    this.emit({
      type: 'agent.node.resolve.started',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { archetype: request.archetype, reuseMode, source: invocation.source },
    });

    const canReadCache = reuseMode !== 'fresh';
    const canonicalAgentPattern = await ctx.memory.findAgentPattern(patternKey);
    const cachedAgentPattern = canReadCache
      ? request.reuse?.agentPatternId
        ? await ctx.memory.findAgentPatternById(request.reuse.agentPatternId)
        : canonicalAgentPattern
      : undefined;
    const cachedDelegationPattern = canReadCache
      ? await ctx.memory.findDelegationPattern(request.archetype, request.task)
      : undefined;
    if (reuseMode === 'require_cache' && !cachedAgentPattern) {
      throw new Error(`Agent creation requires a cached pattern for archetype "${request.archetype}"`);
    }
    if (request.reuse?.agentPatternId && cachedAgentPattern?.id !== request.reuse.agentPatternId) {
      throw new Error(`Requested agent pattern "${request.reuse.agentPatternId}" was not found`);
    }
    if (cachedAgentPattern && cachedAgentPattern.archetype !== request.archetype) {
      throw new Error(`Agent pattern "${String(cachedAgentPattern.id)}" does not match archetype "${request.archetype}"`);
    }
    if (request.reuse?.delegationPatternId && cachedDelegationPattern?.id !== request.reuse.delegationPatternId) {
      throw new Error(`Requested delegation pattern "${request.reuse.delegationPatternId}" was not found`);
    }

    const cachedTools = this.stringArray(cachedAgentPattern?.tools);
    const cachedSkills = this.stringArray(cachedAgentPattern?.skills);
    const tools = request.tools ?? (cachedTools.length > 0 ? cachedTools : this.getDefaultToolBindings(request.archetype).map(item => item.name));
    const skills = request.skills ?? (cachedSkills.length > 0 ? cachedSkills : this.getDefaultSkillBindings(request.archetype).map(item => item.name));
    this.validateDelegatedCapabilities(parentId, tools, skills);

    const cachedMemoryScope = this.agentMemoryScope(cachedAgentPattern?.memoryScope);
    const cachedSpawnPolicy = this.partialSpawnPolicy(cachedAgentPattern?.spawnPolicy);
    const memoryScope = this.constrainMemoryScope(
      request.memoryScope ?? cachedMemoryScope ?? this.getDefaultMemoryScope('subagent')
    );
    const requestedSpawnPolicy = this.mergeSpawnPolicy(
      this.getDefaultSpawnPolicy('subagent', request.archetype),
      { ...cachedSpawnPolicy, ...request.spawnPolicy }
    );
    const spawnPolicy = this.constrainChildSpawnPolicy(parentId, requestedSpawnPolicy, skills);
    const agentPatternId = typeof cachedAgentPattern?.id === 'string' ? cachedAgentPattern.id : undefined;
    const delegationPatternId = typeof cachedDelegationPattern?.id === 'string' ? cachedDelegationPattern.id : undefined;
    const cacheHits = [agentPatternId, delegationPatternId].filter((item): item is string => Boolean(item));
    const hasDefinitionOverrides = request.name !== undefined
      || request.role !== undefined
      || request.style !== undefined
      || request.description !== undefined
      || request.tools !== undefined
      || request.skills !== undefined
      || request.memoryScope !== undefined
      || request.spawnPolicy !== undefined
      || request.tomProfile !== undefined
      || request.outputContract !== undefined;
    const creationMode: AgentNodeCreationMode = cachedAgentPattern && (reuseMode === 'mutate_cache' || hasDefinitionOverrides)
      ? 'mutated_from_cache'
      : cachedAgentPattern
        ? 'cache_hit'
        : request.archetype === 'custom'
          ? 'custom'
          : 'generated';
    const outputContract = request.outputContract ?? {
      format: 'markdown',
      groundingRequired: request.archetype === 'researcher',
    };
    const description = request.description ?? `Reusable ${request.archetype} agent compute node.`;
    const definitionSeed = {
      archetype: request.archetype,
      name: request.name,
      role: request.role ?? request.archetype,
      style: request.style,
      description,
      tools,
      skills,
      memoryScope,
      spawnPolicy,
      tomProfile: request.tomProfile,
      outputContract,
    };
    const definitionFingerprint = this.fingerprint(definitionSeed);
    const invocationFingerprint = this.fingerprint({
      definitionFingerprint,
      parentId,
      task: request.task,
      sessionId: ctx.sessionId,
      correlationId,
    });
    const canonicalPatternId = `agent_pattern_${this.safeAgentKey(patternKey)}_v1`;
    const targetPatternId = ((hasDefinitionOverrides && canonicalAgentPattern)
      || creationMode === 'mutated_from_cache')
      ? `agent_pattern_${this.safeAgentKey(patternKey)}_${definitionFingerprint.slice(0, 12)}_v1`
      : canonicalPatternId;
    const nodeId = `node_${correlationId}_${definitionFingerprint.slice(0, 10)}`;
    const node: AgentComputeNodeDefinition = {
      nodeId,
      sessionId: ctx.sessionId,
      correlationId,
      parentId,
      depth: this.getAgentDepth(parentId) + 1,
      definitionFingerprint,
      invocationFingerprint,
      identity: {
        archetype: request.archetype,
        name: request.name,
        role: request.role ?? request.archetype,
        style: request.style,
        description,
        tomProfile: request.tomProfile,
      },
      assignment: { task: request.task, outputContract },
      capabilities: { tools: [...tools], skills: [...skills] },
      context: { memoryScope },
      resources: { budgetTokens: request.budgetTokens },
      governance: { spawnPolicy },
      execution: {
        requireParentSynthesis: request.execution?.requireParentSynthesis ?? true,
        showSubagentOutput: request.execution?.showSubagentOutput ?? false,
        disableRecursiveDelegation: request.execution?.disableRecursiveDelegation ?? false,
        teamId: request.execution?.teamId,
      },
      reuse: {
        mode: reuseMode,
        creationMode,
        definitionOverrides: hasDefinitionOverrides,
        cacheHits,
        agentPatternId,
        basePatternId: targetPatternId === canonicalPatternId
          ? undefined
          : typeof canonicalAgentPattern?.id === 'string' ? canonicalAgentPattern.id : undefined,
        delegationPatternId,
        targetPatternId,
      },
      source: invocation.source,
    };

    this.emitNodeEvent('agent.node.cache.evaluated', node, parentId, {
      reuseMode,
      cacheHits,
      creationMode,
      hasDefinitionOverrides,
    });
    for (const patternId of cacheHits) {
      this.emitNodeEvent('cache.hit', node, parentId, {
        cacheType: patternId.startsWith('agent_pattern_') ? 'agent-pattern' : 'delegation-pattern',
        patternId,
        archetype: request.archetype,
      });
    }
    this.emitNodeEvent('agent.node.resolved', node, parentId, {
      definitionFingerprint,
      invocationFingerprint,
      tools,
      skills,
      depth: node.depth,
    });
    return node;
  }

  private async executeAgentComputeNode(node: AgentComputeNodeDefinition): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const payload: SpawnCommandPayload = {
      parentId: node.parentId,
      archetype: node.identity.archetype,
      task: node.assignment.task,
      name: node.identity.name,
      customRole: node.identity.role,
      customStyle: node.identity.style,
      tools: node.capabilities.tools,
      skills: node.capabilities.skills,
      budgetTokens: node.resources.budgetTokens,
      memoryScope: node.context.memoryScope,
      spawnPolicy: node.governance.spawnPolicy,
      tomProfile: node.identity.tomProfile,
      outputContract: node.assignment.outputContract,
      correlationId: node.correlationId,
      source: node.source,
      requireRootSynthesis: node.execution.requireParentSynthesis,
      showSubagentOutput: node.execution.showSubagentOutput,
      disableRecursiveDelegation: node.execution.disableRecursiveDelegation,
      teamId: node.execution.teamId,
    };
    const correlationId = node.correlationId;
    const parentId = node.parentId;
    const requireRootSynthesis = node.execution.requireParentSynthesis;
    const cacheHits = node.reuse.cacheHits;

    const externalCommand = payload.source === 'cli' || payload.source === 'server';
    const command = await this.enqueueMessage({
      kind: externalCommand ? 'user.command.spawn' : 'agent.control',
      sessionId: ctx.sessionId,
      from: externalCommand ? payload.source ?? 'cli' : parentId,
      to: externalCommand ? parentId : 'runtime',
      correlationId,
      payload: externalCommand ? payload : { action: 'delegate_to_subagent', node },
      metadata: { agentId: parentId, nodeId: node.nodeId },
    });
    if (externalCommand) {
      await this.recordConversation({
        role: 'user',
        speaker: payload.source ?? 'cli',
        content: `/spawn ${payload.archetype} "${payload.task}"`,
        correlationId,
        metadata: { command: 'spawn', archetype: payload.archetype },
      });
    }
    await this.processQueuedMessage(command.id);
    await ctx.queue.ack(command.id);

    if (parentId !== 'root') {
      await this.prepareParentForDelegation(parentId, correlationId, payload.task);
    }

    const tomProfile = payload.tomProfile
      ? { ...payload.tomProfile }
      : this.createSubagentToMProfile(payload.archetype, '', payload.task, parentId);
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
      memoryScope: node.context.memoryScope,
      spawnPolicy: node.governance.spawnPolicy,
      budgetTokens: payload.budgetTokens,
      systemPrompt: undefined,
      outputContract: node.assignment.outputContract,
      correlationId,
      tomProfile,
      cacheHits,
      nodeDefinition: node,
      teamId: payload.teamId,
    });
    if (payload.teamId) {
      const team = this.teams.addMember(payload.teamId, agent.identity.id);
      try {
        await this.persistTeamTopology(team);
      } catch (error) {
        this.emit({
          type: 'team.persistence.failed',
          agentId: team.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: team.identity.id,
            parentAgentId: team.identity.parentAgentId,
            operation: 'persist_member_addition',
            error: error instanceof Error ? error.message : String(error),
          },
        });
      }
      this.emit({ type: 'team.member.added', agentId: agent.identity.id, data: { teamId: payload.teamId, parentId } });
    }
    const delegationPattern = await ctx.memory.upsertDelegationPattern({
      archetype: payload.archetype,
      task: payload.task,
      parentId,
      agentPatternId: node.reuse.targetPatternId,
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

    const taskSender = payload.teamId ?? parentId;
    const resultRecipient = payload.teamId ?? parentId;
    const taskMessage = await this.enqueueMessage({
      kind: 'agent.task',
      sessionId: ctx.sessionId,
      from: taskSender,
      to: agent.identity.id,
      correlationId,
      parentMessageId: command.id,
      payload: {
        task: payload.task,
        archetype: payload.archetype,
      },
      metadata: {
        agentId: agent.identity.id,
        nodeId: node.nodeId,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(taskMessage.id);

    let subagentResult: RunAgentResult;
    try {
      subagentResult = await this.runAgent(agent.identity.id, payload.task, {
        correlationId,
        parentMessageId: taskMessage.id,
        archetype: payload.archetype,
        disableRecursiveDelegation: payload.disableRecursiveDelegation,
        nodeId: node.nodeId,
        patternId: node.reuse.targetPatternId,
      });
      await ctx.queue.ack(taskMessage.id);
    } catch (error) {
      const currentTask = await ctx.queue.getMessage(taskMessage.id);
      if (currentTask?.status === 'pending' || currentTask?.status === 'processing') {
        await ctx.queue.fail(taskMessage.id, error instanceof Error ? error : new Error(String(error)));
      }
      this.emitNodeEvent('agent.task.failed', node, agent.identity.id, {
        task: payload.task,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }

    const resultMessage = await this.enqueueMessage({
      kind: 'agent.result',
      sessionId: ctx.sessionId,
      from: agent.identity.id,
      to: resultRecipient,
      correlationId,
      parentMessageId: taskMessage.id,
      payload: subagentResult,
      metadata: {
        agentId: agent.identity.id,
        nodeId: node.nodeId,
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
        teamId: payload.teamId,
        grounded: subagentResult.grounded,
        warnings: subagentResult.warnings,
        toolCalls: subagentResult.toolCalls.map(call => call.toolName),
        evidence: subagentResult.evidence,
        nodeId: node.nodeId,
        definitionFingerprint: node.definitionFingerprint,
        invocationFingerprint: node.invocationFingerprint,
        creationMode: node.reuse.creationMode,
      },
    });
    await this.processQueuedMessage(resultMessage.id);
    await ctx.queue.ack(resultMessage.id);
    this.emit({ type: 'agent.result.sent', agentId: agent.identity.id, data: { correlationId, to: resultRecipient, teamId: payload.teamId } });

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
          nodeId: node.nodeId,
          definitionFingerprint: node.definitionFingerprint,
        },
      });
      await this.processQueuedMessage(finalMessage.id);
      await ctx.queue.ack(finalMessage.id);
      await this.proposeMemoryUpdates('turn.completed');
    }

    return {
      correlationId,
      node,
      agent,
      subagentResult,
      finalResponse,
      messages: await this.getMessages({ correlationId }),
      creationUsage: this.measureAgentCreationUsage(agent.identity.id, node),
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
    const creationCorrelationId = spec.correlationId ?? this.createCorrelationId();
    if (spec.parentId !== 'root') {
      await this.prepareParentForDelegation(spec.parentId, creationCorrelationId, spec.task ?? spec.description);
    }
    const toolBindings = this.normalizeToolBindings(spec.tools, spec.archetype)
      .filter(binding => binding.enabled);
    const skillBindings = this.normalizeSkillBindings(spec.skills, spec.archetype)
      .filter(binding => binding.enabled);
    const memoryScope = spec.memoryScope ?? this.getDefaultMemoryScope('subagent');
    const spawnPolicy = this.mergeSpawnPolicy(this.getDefaultSpawnPolicy('subagent', spec.archetype), spec.spawnPolicy);
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
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        invocationFingerprint: spec.nodeDefinition?.invocationFingerprint,
        creationMode: spec.nodeDefinition?.reuse.creationMode,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
      metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
    });
    await this.processQueuedMessage(createRequestMessage.id);

    this.emit({
      type: 'agent.create.requested',
      agentId: spec.parentId,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name: spec.name,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        creationMode: spec.nodeDefinition?.reuse.creationMode,
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
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
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
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'agent.create.rejected',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'delegation.rejected',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
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
        metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
      });
      await this.processQueuedMessage(rejectedMessage.id);
      await ctx.queue.ack(rejectedMessage.id);
      await ctx.queue.fail(createRequestMessage.id, new Error(policyResult.reason ?? 'spawn_rejected'));
      throw new Error(`Spawn rejected: ${policyResult.reason}`);
    }

    const budgetAllocation = await this.requestAgentBudget({
      parentId: spec.parentId,
      archetype: spec.archetype,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      requestedTokens: spec.budgetTokens,
      purpose: spec.task ?? spec.description,
    });
    if (budgetAllocation?.status === 'denied') {
      await ctx.queue.fail(createRequestMessage.id, new Error('budget_request_denied'));
      this.emit({
        type: 'agent.create.rejected',
        agentId: spec.parentId,
        data: { parentId: spec.parentId, archetype: spec.archetype, reason: 'budget_request_denied' },
      });
      throw new Error('Spawn rejected: budget_request_denied');
    }

    const sequence = ++this.agentSequence;
    const id = this.createAgentId(spec.archetype, sequence);
    const requestedName = spec.name ?? `${this.capitalize(spec.archetype)}-${sequence}`;
    const name = ctx.manager.getAgent(requestedName)
      ? this.createUniqueAgentName(spec.archetype, requestedName, sequence)
      : requestedName;
    if (ctx.manager.getAgent(name)) {
      if (budgetAllocation?.status === 'granted') {
        this.agentBudgetAllocations.set(id, budgetAllocation.id);
        this.releaseAgentBudget(id, 'agent_creation_failed');
      }
      throw new Error(`Agent name "${name}" already exists`);
    }
    if (budgetAllocation?.status === 'granted') {
      this.agentBudgetAllocations.set(id, budgetAllocation.id);
    }
    if (spec.budgetTokens !== undefined) {
      this.agentBudgetLimits.set(id, Math.max(0, Math.floor(spec.budgetTokens)));
    }

    try {
      const agentMemoryKey =
        spec.archetype === 'custom' && spec.name ? this.safeAgentKey(spec.name) : spec.archetype;
      await ctx.memory.ensureAgentMemory(agentMemoryKey, {
        name: spec.name ?? this.capitalize(spec.archetype),
        role: spec.customRole ?? spec.archetype,
        description: `Reusable ${spec.archetype} agent archetype memory.`,
      });
      const agentMemory = await ctx.memory.loadAgentMemory(agentMemoryKey);

      const generation = parentIdentity.generation + 1;

      const fsm = new FSM({
      initialState: 'S_created',
      strict: true,
      signalBus,
      onTransition: (from, to) => {
        logger.debug(`FSM transition for ${id}: ${from} -> ${to}`);
        this.emit({ type: 'fsm.transition', agentId: id, data: { from, to } });
      },
      onStateChange: (state) => {
        logger.debug(`FSM state for ${id}: ${state}`);
        this.emit({ type: 'fsm.state.changed', agentId: id, data: { state } });
      },
      onInvalidTransition: (from, to) => {
        this.emit({ type: 'fsm.invalid_transition', agentId: id, data: { from, to } });
      },
      });

      const cacheHits = spec.cacheHits ?? [];
      const creationMode =
        spec.nodeDefinition?.reuse.creationMode ??
        (cacheHits.length > 0 ? 'cache_hit' : spec.archetype === 'custom' ? 'custom' : 'generated');
      const resolvedTomProfile = spec.tomProfile
        ? { ...spec.tomProfile, subjectAgentId: id }
        : this.createSubagentToMProfile(spec.archetype, id, spec.task ?? '', spec.parentId);
    const contextWindow = await this.requireContextWindowManager().build({
      sessionId: ctx.sessionId,
      agentId: id,
      agentKey: agentMemoryKey,
      role: 'subagent',
      task: spec.task ?? '',
      parentContext: `Parent agent ${parentIdentity.name} (${parentIdentity.id}) spawned this agent for: ${spec.description}`,
      memoryScope,
    });
    const goal = this.buildAgentPromptFromMemory({
      name,
      role: spec.customRole ?? spec.archetype,
      parentName: parentIdentity.name,
      task: spec.task ?? '',
      description: [
          spec.description,
          spec.customRole ? `Custom role: ${spec.customRole}` : undefined,
          spec.customStyle ? `Custom style: ${spec.customStyle}` : undefined,
          spec.outputContract
            ? `Output contract: ${JSON.stringify(spec.outputContract)}`
            : undefined,
        ]
          .filter(Boolean)
          .join('\n'),
        systemPrompt: spec.systemPrompt,
        bundle: agentMemory,
        publicContext: [
          contextWindow.publicContext,
          contextWindow.sessionContext,
          cacheHits.length > 0 ? this.formatCachedPublicContext(cacheHits) : '',
        ]
          .filter(Boolean)
          .join('\n\n'),
        tomProfile: resolvedTomProfile,
        availableSkills: skillBindings.map((binding) => binding.name),
        availableTools: toolBindings.map((binding) => binding.name),
        parentContext: contextWindow.parentContext,
      });
      const renderedPromptTokens = this.estimateTextTokens(goal);
    const definitionText = [
      name,
      spec.archetype,
      spec.customRole,
        spec.customStyle,
        spec.description,
        spec.tomProfile ? JSON.stringify(spec.tomProfile) : '',
        toolBindings.map((binding) => binding.name).join(','),
        skillBindings.map((binding) => binding.name).join(','),
      ]
        .filter(Boolean)
        .join('\n');
      const definitionTokens =
        creationMode === 'cache_hit' ? 0 : this.estimateTextTokens(definitionText);

      const agent = new UnifiedAgent({
        id,
      name,
      role: 'subagent',
      parentId: spec.parentId,
      teamId: spec.teamId,
      generation,
      tomLevel: spec.tomLevel,
      tomProfile: resolvedTomProfile,
      description: spec.description,
      goal,
        llm: ctx.llm ?? undefined,
        fsm,
        mode: 'hybrid',
        allowedTools: toolBindings.map(binding => binding.name),
        allowedSkills: skillBindings.map(binding => binding.name),
      });

      this.registerCapabilities(
        agent,
        toolBindings.map((binding) => binding.name)
      );
      this.agentBindings.set(id, {
        tools: toolBindings,
        skills: skillBindings,
      memoryScope,
        spawnPolicy,
      });
      this.agentFsms.set(id, fsm);
      ctx.manager.addAgent(agent);
      await ctx.manager.attachAgentToSessions(agent);
      await ctx.memory.upsertAgentPattern({
        key: agentMemoryKey,
        patternId: spec.nodeDefinition?.reuse.targetPatternId,
        basePatternId: spec.nodeDefinition?.reuse.basePatternId,
        status: spec.nodeDefinition?.reuse.basePatternId ? 'candidate' : undefined,
        name: spec.name ?? this.capitalize(spec.archetype),
        archetype: spec.archetype,
        tomLevel: spec.tomLevel,
        description: spec.description,
        tools: toolBindings.map((binding) => binding.name),
        skills: skillBindings.map((binding) => binding.name),
        spawnPolicy,
        memoryScope,
        outputContract: spec.outputContract,
      definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
      creationMode,
    });

    const info = agent.getInfo();
    this.recordTurnAgentCreated(creationCorrelationId);
    this.emit({
      type: 'context.loaded',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        sources: contextWindow.sources,
        tokenUsage: contextWindow.tokenUsage,
        sessionWindowTurns: memoryScope.sessionWindowTurns,
        },
      });
      this.emit({
        type:
          creationMode === 'cache_hit'
            ? 'agent.definition.loaded_from_cache'
            : 'agent.definition.generated',
        agentId: id,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        archetype: spec.archetype,
        cacheHits,
        creationMode,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
      },
    });
    this.emit({
      type: 'agent.create.approved',
      agentId: spec.parentId,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        childId: id,
        archetype: spec.archetype,
          creationMode,
          nodeId: spec.nodeDefinition?.nodeId,
          definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
          skills: skillBindings.map((binding) => binding.name),
          tools: toolBindings.map((binding) => binding.name),
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
        nodeId: spec.nodeDefinition?.nodeId,
          definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
          invocationFingerprint: spec.nodeDefinition?.invocationFingerprint,
          creationMode,
          tools: toolBindings.map((binding) => binding.name),
          skills: skillBindings.map((binding) => binding.name),
        },
        metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
      });
      await this.processQueuedMessage(approvedMessage.id);
      await ctx.queue.ack(approvedMessage.id);
    await ctx.queue.ack(createRequestMessage.id);
    this.emit({
      type: 'agent.instance.created',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name,
        memoryKey: agentMemoryKey,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        },
      });
      for (const binding of toolBindings) {
        this.emit({
          type: 'agent.tool.bound',
          agentId: id,
          data: { tool: binding.name, permission: binding.permission },
        });
      }
      for (const binding of skillBindings) {
        this.emit({ type: 'agent.skill.bound', agentId: id, data: { skill: binding.name } });
    }
    this.emit({
      type: 'agent.spawned',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        name,
        archetype: spec.archetype,
        tomLevel: spec.tomLevel,
        description: spec.description,
        mode: creationMode,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
      },
    });
    this.emit({
      type: 'agent.creation.measured',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        mode: creationMode,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
        },
      });
      await ctx.memory.updateCacheUsageMetrics(cacheHits, {
        definitionTokensSaved:
          creationMode === 'cache_hit' ? this.estimateTextTokens(definitionText) : 0,
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
      this.emit({
        type: 'agent.status.changed',
        agentId: id,
        data: { from: 'none', to: info.state },
      });
      await this.transitionAgentFsm(id, 'S_ready', { runtimeState: info.state });
      if (budgetAllocation?.status === 'granted') {
        this.emit({
        type: 'budget.allocated',
        agentId: id,
        data: { allocationId: budgetAllocation.id, budgetTokens: budgetAllocation.grantedTokens },
      });
      }

      return info;
    } catch (error) {
      ctx.manager.removeAgent(name);
      this.agentBindings.delete(id);
      this.agentFsms.delete(id);
      this.agentBudgetLimits.delete(id);
      this.releaseAgentBudget(id, 'agent_creation_failed');
      const currentRequest = await ctx.queue.getMessage(createRequestMessage.id);
      if (currentRequest?.status === 'pending' || currentRequest?.status === 'processing') {
        await ctx.queue.fail(
          createRequestMessage.id,
          error instanceof Error ? error : new Error(String(error))
        );
      }
      this.emit({
        type: 'agent.create.failed',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          childId: id,
          archetype: spec.archetype,
          error: error instanceof Error ? error.message : String(error),
        },
      });
      throw error;
    }
  }

  async runAgent(
    agentId: string,
    task: string,
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      disableRecursiveDelegation?: boolean;
      nodeId?: string;
      patternId?: string;
    } = {}
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) {
      throw new Error(`Agent "${agentId}" not found`);
    }

    if (!this.agentBudgetAllocations.has(agentId)) {
      const archetype = options.archetype ?? this.inferAgentArchetype(agent.getInfo());
      const allocation = await this.requestAgentBudget({
        parentId: agent.getIdentity().parentId ?? 'root',
        archetype,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        requestedTokens: this.agentBudgetLimits.get(agentId),
        purpose: task,
      });
      if (allocation?.status === 'denied') {
        this.emit({
          type: 'agent.run.rejected',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: { task, reason: 'budget_request_denied' },
        });
        throw new Error('Agent run rejected: budget_request_denied');
      }
      if (allocation?.status === 'granted') {
        this.agentBudgetAllocations.set(agentId, allocation.id);
      }
    }

    const session = ctx.manager.getSession(ctx.sessionId);
    if (session) {
      session.messageQueue.clear('env');
    }

    const usageBefore = agent.getUsage();
    const from = agent.getState();
    const actorFsm = this.requireAgentFsm(agentId);
    if (actorFsm.getState() === 'S_done' || actorFsm.getState() === 'S_failed') {
      await this.transitionAgentFsm(agentId, 'S_ready', { reason: 'new_task' });
    }
    agent.setRuntimeState('thinking');
    await this.transitionAgentFsm(agentId, 'S_task_received', { task, correlationId: options.correlationId });
    this.emit({
      type: 'agent.run.started',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: { task, correlationId: options.correlationId },
    });
    this.emit({ type: 'agent.status.changed', agentId, data: { from, to: 'thinking' } });

    try {
      await this.transitionAgentFsm(agentId, 'S_context_loading', { task, correlationId: options.correlationId });
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
        await this.transitionAgentFsm(agentId, 'S_tool_calling', {
          toolCalls: grounding.toolCalls.map(call => call.toolName),
          correlationId: options.correlationId,
        });
      }
      await this.transitionAgentFsm(agentId, 'S_reasoning', { task, correlationId: options.correlationId });
      this.emit({ type: 'agent.llm.called', agentId, data: { task } });
      await agent.step(this.buildGroundedTask(task, grounding));
      const stepError = agent.getInfo().error;
      if (stepError) {
        throw new Error(stepError.replace(/^Error:\s*/, ''));
      }
      agent.setRuntimeState('done');
      await this.transitionAgentFsm(agentId, 'S_responding', { correlationId: options.correlationId });

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
          ? this.resultIncludesEvidence(result || agent.getInfo().lastResult || '', grounding.evidence)
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
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          task,
          correlationId: options.correlationId,
          totalTokens: usageDelta.totalTokens,
          grounded: grounding.grounded,
          evidence,
          warnings,
        },
      });
      await this.transitionAgentFsm(agentId, 'S_done', { correlationId: options.correlationId });
      this.settleAgentBudget(agentId, usageDelta.totalTokens);
      await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(agent.getInfo()), {
        success: true,
        grounded: grounding.grounded && evidence.outputGrounded,
        totalTokens: usageDelta.totalTokens,
      }, options.patternId);

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
      if (this.requireAgentFsm(agentId).getState() !== 'S_failed') {
        await this.transitionAgentFsm(agentId, 'S_failed', { failed: true, error: message, correlationId: options.correlationId });
      }
      this.emit({
        type: 'agent.run.failed',
        agentId,
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: { task, error: message, correlationId: options.correlationId },
      });
      this.releaseAgentBudget(agentId, 'agent_run_failed');
      await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(agent.getInfo()), {
        success: false,
        grounded: false,
        totalTokens: 0,
      }, options.patternId);
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

  private async transitionAgentFsm(
    agentId: string,
    state: Parameters<FSM['transition']>[0],
    data: Record<string, unknown> = {}
  ): Promise<void> {
    if (!state) return;
    const fsm = this.requireAgentFsm(agentId);
    const from = fsm.getState();
    try {
      await fsm.transition(state);
      this.emit({ type: 'agent.fsm.state', agentId, data: { from, state, ...data } });
    } catch (error) {
      this.emit({
        type: 'delegation.rejected',
        agentId,
        data: {
          reason: 'invalid_fsm_state',
          from,
          to: state,
          error: error instanceof Error ? error.message : String(error),
          ...data,
        },
      });
      throw error;
    }
  }

  private requireAgentFsm(agentId: string): FSM {
    const fsm = this.agentFsms.get(agentId);
    if (!fsm) throw new Error(`FSM for agent "${agentId}" not found`);
    return fsm;
  }

  private requireContextWindowManager(): ContextWindowManager {
    if (!this.contextWindowManager) throw new Error('ContextWindowManager is not initialized');
    return this.contextWindowManager;
  }

  private requireCandidatePlanner(): DefaultDelegationCandidatePlanner {
    if (!this.candidatePlanner) throw new Error('Delegation candidate planner is not initialized');
    return this.candidatePlanner;
  }

  private async prepareParentForDelegation(parentId: string, correlationId: string, task: string): Promise<void> {
    const parent = this.getContext().manager.getAgentById(parentId);
    if (!parent || parent.getState() === 'failed' || parent.getState() === 'stopped') return;
    const fsm = this.requireAgentFsm(parentId);
    if (fsm.getState() === 'S_done') {
      await this.transitionAgentFsm(parentId, 'S_ready', { correlationId, reason: 'manual_child_delegation' });
    }
    if (fsm.getState() === 'S_ready') {
      await this.transitionAgentFsm(parentId, 'S_task_received', { correlationId, task });
    }
    if (fsm.getState() === 'S_task_received') {
      await this.transitionAgentFsm(parentId, 'S_context_loading', { correlationId, task });
    }
    if (fsm.getState() === 'S_context_loading') {
      await this.transitionAgentFsm(parentId, 'S_planning', { correlationId, task });
    }
    if (fsm.getState() === 'S_planning') {
      await this.transitionAgentFsm(parentId, 'S_delegating', { correlationId, task });
    }
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
    const [agentPatterns, delegationPatterns] = await Promise.all([
      ctx.memory.getCachePatterns('agents'),
      ctx.memory.getCachePatterns('delegations'),
    ]);
    const selection = await this.requireCandidatePlanner().select({
      parentId,
      task,
      decision,
      allowedChildren: policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      budgetMode: budget.mode,
      remainingBudgetTokens: budget.remainingTokens,
      cacheUsed: cacheHits.some(Boolean),
      cachedPatterns: [...agentPatterns, ...delegationPatterns],
      parentToMProfile: ctx.manager.getAgentById(parentId)?.getIdentity().tomProfile,
    });

    await this.recordEvolutionLifecycle(parentId, correlationId, scope, selection);
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
          scoreBreakdown: candidate.scoreBreakdown,
          lineage: candidate.lineage,
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
          scoreBreakdown: selection.selected.scoreBreakdown,
          lineage: selection.selected.lineage,
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

  private async recordEvolutionLifecycle(
    parentId: string,
    correlationId: string,
    scope: 'root' | 'agent',
    selection: DelegationCandidateSelection
  ): Promise<void> {
    const ctx = this.getContext();
    const proposed = await this.enqueueMessage({
      kind: 'evo.propose',
      sessionId: ctx.sessionId,
      from: parentId,
      to: 'delegation.evolution',
      correlationId,
      payload: { scope, candidates: selection.candidates.map(candidate => ({ id: candidate.id, source: candidate.source, lineage: candidate.lineage })) },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(proposed.id);
    const evaluated = await this.enqueueMessage({
      kind: 'evo.evaluate',
      sessionId: ctx.sessionId,
      from: 'delegation.evolution',
      to: parentId,
      correlationId,
      parentMessageId: proposed.id,
      payload: { candidates: selection.candidates.map(candidate => ({ id: candidate.id, score: candidate.score, scoreBreakdown: candidate.scoreBreakdown })) },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(evaluated.id);
    const selected = await this.enqueueMessage({
      kind: 'evo.select',
      sessionId: ctx.sessionId,
      from: 'delegation.evolution',
      to: parentId,
      correlationId,
      parentMessageId: evaluated.id,
      payload: { selected: selection.selected?.id, rejectedReason: selection.rejectedReason },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(selected.id);
    await ctx.queue.ack(proposed.id);
    await ctx.queue.ack(evaluated.id);
    await ctx.queue.ack(selected.id);
    await ctx.memory.recordEvolutionRun({
      correlationId,
      parentId,
      scope,
      proposed: selection.candidates.map(candidate => ({
        id: candidate.id,
        source: candidate.source,
        score: candidate.score,
        scoreBreakdown: candidate.scoreBreakdown,
        lineage: candidate.lineage,
      })),
      selected: selection.selected?.id,
      rejectedReason: selection.rejectedReason,
    });
    this.emit({ type: 'evo.proposed', agentId: parentId, data: { correlationId, count: selection.candidates.length, scope } });
    this.emit({ type: 'evo.evaluated', agentId: parentId, data: { correlationId, count: selection.candidates.length, scope } });
    this.emit({ type: 'evo.selected', agentId: parentId, data: { correlationId, candidateId: selection.selected?.id, scope } });
    if (selection.selected?.source === 'mutated_from_cache') {
      this.emit({
        type: 'cache.mutated',
        agentId: parentId,
        data: { correlationId, candidateId: selection.selected.id, lineage: selection.selected.lineage },
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
    const grounding = await this.runGroundingCheck('root', userInput, { correlationId, archetype: 'custom' });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.solo_reasoning', correlationId } });
    const response = await this.completeAsRoot(this.buildGroundedTask(userInput, grounding), 'root.solo_reasoning', correlationId);
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
    correlationId: string,
    teamResults: TeamRunResult[] = []
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
        teamIds: teamResults.map(result => result.team.identity.id),
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
      this.buildMultiAgentSynthesisPrompt(userTask, results, teamResults),
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

  private buildMultiAgentSynthesisPrompt(
    userTask: string,
    results: RootMediatedSpawnResult[],
    teamResults: TeamRunResult[] = []
  ): string {
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

    const teamReports = teamResults.map(result => `<team_report>
id: ${result.team.identity.id}
name: ${result.team.identity.name}
tom: ToM-${result.team.identity.tomLevel}
members: ${result.team.memberAgentIds.join(', ')}
tokens: ${result.usage.totalTokens}
content:
${result.result}
</team_report>`).join('\n\n');

    return `The user requested:
<user_task>
${userTask}
</user_task>

Roy delegated this task to ${results.length} subagent(s). Synthesize their results into one final user-facing response.
Use concrete evidence from grounded reports. If a report is ungrounded or missing concrete tool output, say so and avoid overstating it.

${teamReports ? `The following subteam reports have already aggregated their direct members. Treat them as the primary delegation result.\n\n${teamReports}` : ''}

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

  private subtractTokenUsage(after: TokenUsage, before: TokenUsage): TokenUsage {
    const thinkingTokens = after.thinkingTokens === null && before.thinkingTokens === null
      ? null
      : Math.max(0, Number(after.thinkingTokens ?? 0) - Number(before.thinkingTokens ?? 0));
    const estimatedCostUsd = Math.max(0, Number(after.estimatedCostUsd ?? 0) - Number(before.estimatedCostUsd ?? 0));
    return {
      llmCalls: Math.max(0, after.llmCalls - before.llmCalls),
      promptTokens: Math.max(0, after.promptTokens - before.promptTokens),
      completionTokens: Math.max(0, after.completionTokens - before.completionTokens),
      totalTokens: Math.max(0, after.totalTokens - before.totalTokens),
      thinkingTokens,
      estimatedCostUsd: estimatedCostUsd > 0 ? estimatedCostUsd : undefined,
    };
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
    if (this.requireAgentFsm(parentId).getState() === 'S_delegating') {
      await this.transitionAgentFsm(parentId, 'S_waiting_children', { correlationId, childId: childAgent.identity.id });
    }
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
    await this.transitionAgentFsm(parentId, 'S_synthesizing', { correlationId, childId: childAgent.identity.id });
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
    await this.transitionAgentFsm(parentId, 'S_responding', { correlationId, childId: childAgent.identity.id });
    await this.transitionAgentFsm(parentId, 'S_done', { correlationId, childId: childAgent.identity.id });
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
    parentMessageId: string,
    teamResult?: TeamRunResult
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
      return this.synthesizeDelegatedResults(userTask, childResults, correlationId, teamResult ? [teamResult] : []);
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
    await this.transitionAgentFsm(parentId, 'S_synthesizing', {
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
      this.buildParentMultiChildSynthesisPrompt(parent.getInfo(), userTask, childResults, teamResult),
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
    await this.transitionAgentFsm(parentId, 'S_responding', { correlationId, childIds: childResults.map(result => result.agent.identity.id) });
    await this.transitionAgentFsm(parentId, 'S_done', { correlationId, childIds: childResults.map(result => result.agent.identity.id) });
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
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      disableRecursiveDelegation?: boolean;
      nodeId?: string;
      patternId?: string;
    }
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(agentId);
    if (!parent) {
      throw new Error(`Agent "${agentId}" not found`);
    }
    const correlationId = options.correlationId ?? this.createCorrelationId();
    await this.transitionAgentFsm(agentId, 'S_planning', { correlationId, count: plans.length });
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
    await this.transitionAgentFsm(agentId, 'S_delegating', { correlationId, count: plans.length });

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
    }

    let teamResult: TeamRunResult | undefined;
    if (plans.length > 1 && this.workspaceRuntimeConfig?.teams.createForMultipleAgents !== false) {
      const team = await this.spawnTeam({
        parentAgentId: agentId,
        name: this.deriveTeamName(plans),
        description: task,
        task,
        members: plans.map((plan, index) => ({ ...plan, lead: index === 0 })),
        correlationId,
      });
      teamResult = await this.runTeam(team.identity.id, task);
      childResults.push(...teamResult.memberExecutions);
    } else {
      for (const plan of plans) {
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
          disableRecursiveDelegation: false,
        });
        childResults.push(result);
      }
    }

    await this.transitionAgentFsm(agentId, 'S_waiting_children', { correlationId, completed: childResults.length });
    if (childResults.length === 0) {
      return this.runAgent(agentId, task, { ...options, disableRecursiveDelegation: true });
    }

    const synthesis = await this.synthesizeDirectChildResults(
      agentId,
      task,
      childResults,
      correlationId,
      childResults[0].messages.find(message => message.kind === 'agent.result')?.id ?? options.parentMessageId ?? '',
      teamResult
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
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        task,
        correlationId,
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
    this.settleAgentBudget(agentId, usageDelta.totalTokens);
    await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(parent.getInfo()), {
      success: true,
      grounded: childResults.every(result => result.subagentResult.grounded),
      totalTokens: usageDelta.totalTokens,
    }, options.patternId);

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

  private async transitionTeamFsm(
    teamId: string,
    state: TeamFSMState,
    data: Record<string, unknown> = {}
  ): Promise<TeamRuntimeState> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    const from = team.fsmState;
    try {
      const updated = this.teams.transitionFsm(teamId, state, typeof data.error === 'string' ? data.error : undefined);
      this.emit({
        type: 'team.fsm.transition',
        agentId: updated.identity.id,
        sessionId: this.getContext().sessionId,
        correlationId: updated.correlationId,
        data: { teamId, from, to: state, status: updated.status, ...data },
      });
      return updated;
    } catch (error) {
      this.emit({
        type: 'team.fsm.invalid_transition',
        agentId: team.identity.id,
        sessionId: this.getContext().sessionId,
        correlationId: team.correlationId,
        data: { teamId, from, to: state, error: error instanceof Error ? error.message : String(error) },
      });
      throw error;
    }
  }

  private async persistTeamTopology(team: TeamRuntimeState): Promise<void> {
    const ctx = this.getContext();
    const plannedMembers = (this.teamMemberPlans.get(team.identity.id) ?? []).map(member => ({
      archetype: member.archetype,
      name: member.name,
      role: member.role,
      task: member.task,
      tools: member.tools,
      skills: member.skills,
      tomLevel: member.tomLevel,
      lead: member.lead ?? false,
    }));
    const members = team.memberAgentIds.map(agentId => {
      const agent = ctx.manager.getAgentById(agentId)?.getInfo();
      return {
        id: agentId,
        name: agent?.identity.name,
        role: agent?.identity.role,
        parentId: agent?.identity.parentId,
        teamId: agent?.identity.teamId,
        task: team.memberTasks[agentId],
      };
    });
    await ctx.memory.writeTeamTopology(this.safeAgentKey(team.identity.name), {
      type: 'subteam',
      teamId: team.identity.id,
      parentAgentId: team.identity.parentAgentId,
      leadAgentId: team.leadAgentId,
      tomLevel: team.identity.tomLevel,
      status: team.status,
      fsmState: team.fsmState,
      members,
      plannedMembers,
      tokenUsage: team.tokenUsage,
      updatedAt: new Date().toISOString(),
    });
  }

  private async persistTeamRunArtifacts(input: {
    team: TeamRuntimeState;
    task: string;
    result?: string;
    correlationId: string;
    usage: TokenUsage;
    success: boolean;
  }): Promise<void> {
    const ctx = this.getContext();
    const teamKey = this.safeAgentKey(input.team.identity.name);
    const operations: Array<{ name: string; run: () => Promise<void> }> = [];
    if (input.success && input.result !== undefined) {
      operations.push({
        name: 'append_team_session',
        run: () => ctx.memory.appendTeamSession(teamKey, {
          timestamp: Date.now(),
          sessionId: ctx.sessionId,
          correlationId: input.correlationId,
          teamId: input.team.identity.id,
          task: input.task,
          result: input.result,
          memberAgentIds: input.team.memberAgentIds,
          tokenUsage: input.usage,
          cumulativeTokenUsage: input.team.tokenUsage,
        }),
      });
    }
    operations.push(
      {
        name: 'persist_team_topology',
        run: () => this.persistTeamTopology(input.team),
      },
      {
        name: 'record_team_pattern_outcome',
        run: () => ctx.memory.recordTeamPatternOutcome(teamKey, {
          success: input.success,
          totalTokens: input.usage.totalTokens,
          memberCount: input.team.memberAgentIds.length,
        }),
      }
    );
    for (const operation of operations) {
      try {
        await operation.run();
      } catch (error) {
        this.emit({
          type: 'team.persistence.failed',
          agentId: input.team.identity.id,
          sessionId: ctx.sessionId,
          correlationId: input.correlationId,
          data: {
            teamId: input.team.identity.id,
            parentAgentId: input.team.identity.parentAgentId,
            operation: operation.name,
            error: error instanceof Error ? error.message : String(error),
          },
        });
      }
    }
  }

  private async completeAsTeam(
    team: TeamRuntimeState,
    task: string,
    members: RunAgentResult[],
    correlationId: string
  ): Promise<{ content: string; usage: TokenUsage }> {
    const ctx = this.getContext();
    const teamKey = this.safeAgentKey(team.identity.name);
    const [teamDefinition, teamMemory, rootContext] = await Promise.all([
      ctx.memory.readTeamDoc(teamKey, 'team'),
      ctx.memory.readTeamDoc(teamKey, 'memory'),
      ctx.memory.loadRootContext(),
    ]);
    const publicContext = [rootContext.projectMemory, rootContext.constraints, rootContext.decisions]
      .filter(Boolean)
      .join('\n\n')
      .slice(0, 6000);
    this.emit({
      type: 'team.context.loaded',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        teamId: team.identity.id,
        teamDefinitionChars: teamDefinition.length,
        teamMemoryChars: teamMemory.length,
        publicContextChars: publicContext.length,
      },
    });
    const reports = members.map(member => [
      `<member id="${member.agent.identity.id}" name="${member.agent.identity.name}">`,
      `tokens: ${member.usage.totalTokens}`,
      `grounded: ${member.grounded}`,
      member.result,
      '</member>',
    ].join('\n')).join('\n\n');
    const prompt = [
      `Team task: ${task}`,
      `You are ${team.identity.name}, a subteam actor in Roy.`,
      `Description: ${team.identity.description}`,
      `ToM level: ${team.identity.tomLevel}`,
      `<team_definition>\n${teamDefinition}\n</team_definition>`,
      `<team_private_memory>\n${teamMemory}\n</team_private_memory>`,
      `<public_context>\n${publicContext}\n</public_context>`,
      'Aggregate direct member reports into one grounded result for the parent agent.',
      'Do not claim evidence that is absent from member reports.',
      reports,
    ].join('\n\n');
    const systemPrompt = [
      `You are ${team.identity.name}, a formal subteam actor in the Roy autonomous agent system.`,
      'You are not Roy and you are not the model provider.',
      `Parent agent: ${team.identity.parentAgentId}.`,
      `Correlation: ${correlationId}.`,
    ].join('\n');
    this.emit({
      type: 'team.synthesis.started',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId: team.identity.id, memberAgentIds: team.memberAgentIds },
    });
    if (!ctx.llm) {
      const usage = this.zeroTokenUsage();
      this.emit({
        type: 'team.synthesis.completed',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: team.identity.id, totalTokens: 0, limited: true, reason: 'llm_not_configured' },
      });
      return {
        content: `Team ${team.identity.name} could not synthesize results because no LLM is configured.`,
        usage,
      };
    }
    const estimatedPromptTokens = this.estimateTextTokens(`${systemPrompt}\n${prompt}`);
    const allocation = await this.requestTeamSynthesisBudget({
      team,
      correlationId,
      promptTokens: estimatedPromptTokens,
      completionTokens: 1024,
    });
    if (allocation?.status === 'denied') {
      throw new Error(`Team synthesis rejected: ${allocation.reason}`);
    }
    const maxCompletionTokens = allocation
      ? Math.max(1, allocation.grantedTokens - estimatedPromptTokens)
      : 1024;
    const chunks: string[] = [];
    let providerUsage: { promptTokens: number; completionTokens: number; totalTokens: number } | undefined;
    try {
      for await (const chunk of ctx.llm.stream([
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt },
      ], { temperature: 0.2, maxTokens: maxCompletionTokens })) {
        if (chunk.content) chunks.push(chunk.content);
        if (chunk.usage) providerUsage = chunk.usage;
      }
    } catch (error) {
      this.releaseTeamSynthesisBudget(team.identity.id, allocation, correlationId, 'team_synthesis_failed');
      throw error;
    }
    const content = chunks.join('');
    const usage = this.toTokenUsage({
      llmCalls: 1,
      promptTokens: providerUsage?.promptTokens ?? estimatedPromptTokens,
      completionTokens: providerUsage?.completionTokens ?? this.estimateTextTokens(content),
      totalTokens: providerUsage?.totalTokens
        ?? estimatedPromptTokens + this.estimateTextTokens(content),
    });
    this.settleTeamSynthesisBudget(team.identity.id, allocation, usage.totalTokens, correlationId);
    this.recordTurnUsage(usage);
    this.emit({
      type: 'budget.updated',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId: team.identity.id, ...usage },
    });
    this.emit({
      type: 'team.synthesis.completed',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId: team.identity.id, totalTokens: usage.totalTokens },
    });
    return { content, usage };
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

  private buildParentMultiChildSynthesisPrompt(
    parent: AgentInfo,
    userTask: string,
    childResults: RootMediatedSpawnResult[],
    teamResult?: TeamRunResult
  ): string {
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

${teamResult ? `Your subteam already aggregated its members into this report:\n<team_report>\n${teamResult.result}\n</team_report>\nUse this report as the primary result and verify it against the direct child reports below.` : ''}

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
    const contextWindow = await this.renderAgentContext({
      agentKey,
      agentId: agentKey === 'roy' ? 'root' : agentKey,
      role: agentKey === 'roy' ? 'root' : 'subagent',
      parentId: options.parentId,
      task: options.task ?? '',
    });
    const tomProfile = options.archetype
      ? this.createSubagentToMProfile(options.archetype, agentKey, options.task ?? '', options.parentId ?? 'root')
      : this.createRootToMProfile();
    const prompt = this.buildAgentPromptFromMemory({
      name: options.name ?? this.capitalize(agentKey),
      role,
      parentName: parent?.name ?? 'Roy',
      task: options.task ?? '',
      description: `Rendered prompt preview for ${agentKey}.`,
      bundle,
      publicContext: [contextWindow.publicContext, contextWindow.sessionContext].filter(Boolean).join('\n\n'),
      tomProfile,
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
      parentContext: contextWindow.parentContext || `Parent agent: ${parent?.name ?? 'Roy'} (${parent?.id ?? 'root'})`,
    });
    return {
      prompt,
      estimatedTokens: this.estimateTextTokens(prompt),
      sources: {
        ...contextWindow.sources,
        prompt: [`.roy/agents/${bundle.key}/prompt.md`],
        tokenUsage: contextWindow.tokenUsage,
      },
    };
  }

  async renderAgentContext(options: {
    agentKey: string;
    agentId?: string;
    role?: 'root' | 'subagent';
    parentId?: string;
    task?: string;
  }): Promise<ContextWindow> {
    const ctx = this.getContext();
    const role = options.role ?? (options.agentKey === 'roy' ? 'root' : 'subagent');
    const parent = options.parentId
      ? ctx.manager.getAgentById(options.parentId)?.getIdentity()
      : role === 'subagent'
        ? ctx.agent.getIdentity()
        : undefined;
    const runtimeAgent = options.agentId ? ctx.manager.getAgentById(options.agentId) : undefined;
    const memoryScope = runtimeAgent
      ? this.getAgentPolicy(runtimeAgent.id)?.memoryScope ?? this.getDefaultMemoryScope(role)
      : this.getDefaultMemoryScope(role);
    return this.requireContextWindowManager().build({
      sessionId: ctx.sessionId,
      agentId: options.agentId ?? (role === 'root' ? 'root' : options.agentKey),
      agentKey: options.agentKey,
      role,
      task: options.task ?? '',
      parentContext: parent ? `Parent agent: ${parent.name} (${parent.id})` : undefined,
      memoryScope,
    });
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

  private validateDelegatedCapabilities(parentId: string, tools: string[], skills: string[]): void {
    const bindings = this.agentBindings.get(parentId);
    if (!bindings) throw new Error(`Agent bindings for parent "${parentId}" were not found`);
    if (!bindings.skills.some(item => item.name === 'delegate_to_subagent' && item.enabled)) {
      throw new Error(`Agent "${parentId}" is not authorized to delegate`);
    }
    for (const tool of tools) {
      if (!toolRegistry.has(tool)) throw new Error(`Parent agent "${parentId}" requested unknown tool "${tool}"`);
    }
    for (const skill of skills) {
      if (!skillRegistry.has(skill)) throw new Error(`Parent agent "${parentId}" requested unknown skill "${skill}"`);
    }
  }

  private validateAgentNodeRequest(request: AgentComputeNodeRequest): void {
    const reuseModes = new Set(['prefer_cache', 'require_cache', 'fresh', 'mutate_cache']);
    if (request.reuse?.mode && !reuseModes.has(request.reuse.mode)) {
      throw new Error(`Unsupported agent cache reuse mode "${String(request.reuse.mode)}"`);
    }
    if (request.budgetTokens !== undefined
      && (!Number.isFinite(request.budgetTokens) || request.budgetTokens <= 0)) {
      throw new Error('Agent node budgetTokens must be a positive finite number');
    }
    if (request.outputContract
      && !['markdown', 'json', 'structured_report'].includes(request.outputContract.format)) {
      throw new Error(`Unsupported agent output format "${String(request.outputContract.format)}"`);
    }
    if (request.memoryScope
      && (!Number.isFinite(request.memoryScope.sessionWindowTurns) || request.memoryScope.sessionWindowTurns < 0)) {
      throw new Error('Agent memory sessionWindowTurns must be a non-negative finite number');
    }
    for (const [field, value] of Object.entries(request.spawnPolicy ?? {})) {
      if (['maxChildren', 'maxDepth', 'maxTotalAgentsPerTurn'].includes(field)
        && (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || !Number.isInteger(value))) {
        throw new Error(`Agent spawn policy ${field} must be a non-negative integer`);
      }
    }
    const validChildStates = new Set(['S_planning', 'S_delegating']);
    if (request.spawnPolicy?.allowedStates?.some(state => !validChildStates.has(state))) {
      throw new Error('Child agent spawn policy allowedStates may only contain S_planning or S_delegating');
    }
  }

  private constrainMemoryScope(scope: AgentMemoryScope): AgentMemoryScope {
    const configuredTurns = this.workspaceRuntimeConfig?.context.sessionWindowTurns ?? 10;
    return {
      public: scope.public,
      private: scope.private,
      parentContext: scope.parentContext,
      sessionWindowTurns: Math.min(configuredTurns, Math.max(0, Math.floor(scope.sessionWindowTurns))),
    };
  }

  private constrainChildSpawnPolicy(
    parentId: string,
    requested: AgentSpawnPolicy,
    childSkills: string[]
  ): AgentSpawnPolicy {
    const parentPolicy = this.agentBindings.get(parentId)?.spawnPolicy;
    if (!parentPolicy) throw new Error(`Spawn policy for parent "${parentId}" was not found`);
    const delegation = this.workspaceRuntimeConfig?.delegation;
    const maxChildren = Math.min(
      requested.maxChildren,
      parentPolicy.maxChildren,
      delegation?.maxChildrenPerParent ?? requested.maxChildren
    );
    const maxDepth = Math.min(
      requested.maxDepth,
      parentPolicy.maxDepth,
      delegation?.maxDepth ?? requested.maxDepth
    );
    const maxTotalAgentsPerTurn = Math.min(
      requested.maxTotalAgentsPerTurn,
      parentPolicy.maxTotalAgentsPerTurn,
      delegation?.maxTotalAgentsPerTurn ?? requested.maxTotalAgentsPerTurn
    );
    const validStates = new Set(['S_planning', 'S_delegating']);
    const allowedStates = requested.allowedStates.filter(state => validStates.has(state));
    return {
      canSpawn: requested.canSpawn
        && parentPolicy.canSpawn
        && childSkills.includes('delegate_to_subagent'),
      maxChildren,
      maxDepth,
      maxTotalAgentsPerTurn,
      allowCustomAgents: requested.allowCustomAgents
        && parentPolicy.allowCustomAgents
        && (delegation?.allowCustomAgents ?? true),
      budgetAware: (delegation?.budgetAware ?? true) || requested.budgetAware,
      allowedStates: allowedStates.length > 0 ? allowedStates : ['S_planning', 'S_delegating'],
    };
  }

  private stringArray(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.filter((item): item is string => typeof item === 'string' && item.length > 0)));
  }

  private agentMemoryScope(value: unknown): AgentMemoryScope | undefined {
    if (!value || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    if (typeof record.public !== 'boolean'
      || typeof record.private !== 'boolean'
      || typeof record.parentContext !== 'boolean'
      || typeof record.sessionWindowTurns !== 'number') {
      return undefined;
    }
    return {
      public: record.public,
      private: record.private,
      parentContext: record.parentContext,
      sessionWindowTurns: Math.max(0, Math.floor(record.sessionWindowTurns)),
    };
  }

  private partialSpawnPolicy(value: unknown): Partial<AgentSpawnPolicy> | undefined {
    if (!value || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    const policy: Partial<AgentSpawnPolicy> = {};
    if (typeof record.canSpawn === 'boolean') policy.canSpawn = record.canSpawn;
    if (typeof record.maxChildren === 'number') policy.maxChildren = Math.max(0, Math.floor(record.maxChildren));
    if (typeof record.maxDepth === 'number') policy.maxDepth = Math.max(0, Math.floor(record.maxDepth));
    if (typeof record.maxTotalAgentsPerTurn === 'number') {
      policy.maxTotalAgentsPerTurn = Math.max(0, Math.floor(record.maxTotalAgentsPerTurn));
    }
    if (typeof record.allowCustomAgents === 'boolean') policy.allowCustomAgents = record.allowCustomAgents;
    if (typeof record.budgetAware === 'boolean') policy.budgetAware = record.budgetAware;
    const allowedStates = this.stringArray(record.allowedStates);
    if (allowedStates.length > 0) policy.allowedStates = allowedStates;
    return policy;
  }

  private fingerprint(value: unknown): string {
    const normalize = (input: unknown): unknown => {
      if (Array.isArray(input)) return input.map(normalize);
      if (!input || typeof input !== 'object') return input;
      return Object.fromEntries(
        Object.entries(input as Record<string, unknown>)
          .filter(([, item]) => item !== undefined)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, item]) => [key, normalize(item)])
      );
    };
    return createHash('sha256').update(JSON.stringify(normalize(value))).digest('hex');
  }

  private emitNodeEvent(
    type: string,
    node: AgentComputeNodeDefinition,
    agentId: string,
    data: Record<string, unknown> = {}
  ): RuntimeEvent {
    return this.emit({
      type,
      agentId,
      sessionId: node.sessionId,
      correlationId: node.correlationId,
      nodeId: node.nodeId,
      data: {
        ...data,
        sessionId: node.sessionId,
        correlationId: node.correlationId,
        nodeId: node.nodeId,
      },
    });
  }

  private eventCorrelationId(event: RuntimeEvent): string | undefined {
    return event.correlationId
      ?? (typeof event.data?.correlationId === 'string' ? event.data.correlationId : undefined);
  }

  private estimateTextTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
  }

  private measureAgentCreationUsage(agentId: string, node: AgentComputeNodeDefinition): AgentCreationUsage {
    const event = [...this.events].reverse()
      .find(item => item.type === 'agent.creation.measured' && item.agentId === agentId);
    const definitionTokens = Number(event?.data?.definitionTokens ?? 0);
    const renderedPromptTokens = Number(event?.data?.renderedPromptTokens ?? 0);
    const renderedPromptChars = Number(event?.data?.renderedPromptChars ?? 0);
    return {
      mode: node.reuse.creationMode,
      nodeId: node.nodeId,
      definitionFingerprint: node.definitionFingerprint,
      patternIds: node.reuse.cacheHits,
      cacheHits: node.reuse.cacheHits,
      definitionTokens,
      renderedPromptTokens,
      renderedPromptChars,
    };
  }

  private async runGroundingCheck(
    agentId: string,
    task: string,
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      nodeId?: string;
      patternId?: string;
    }
  ): Promise<{ toolCalls: ToolCallRecord[]; grounded: boolean; warnings: string[]; context: string; evidence: RunEvidence }> {
    const bindings = this.agentBindings.get(agentId)?.tools ?? [];
    const inspectionRoot = this.resolveInspectionRoot(task);
    const plans = this.toolPlanner.plan({ task, workspacePath: inspectionRoot, bindings });
    if (plans.length === 0) {
      return {
        toolCalls: [],
        grounded: true,
        warnings: [],
        context: '',
        evidence: { toolGrounded: false, outputGrounded: true, observedPaths: [] },
      };
    }

    const toolCalls: ToolCallRecord[] = [];
    const warnings: string[] = [];
    const observedPaths: string[] = [];
    const summaries: string[] = [];
    const contexts: string[] = [];

    for (const plan of plans) {
      const result = await this.executeToolForAgent(agentId, plan.toolName, plan.params, {
        reason: plan.reason,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
      });
      toolCalls.push({ toolName: plan.toolName, params: plan.params, result: result.result, success: result.success });

      if (!result.success) {
        const warning = `${plan.toolName === 'fs.list' ? 'Project inspection tool ' : 'Tool '}${plan.toolName} failed: ${result.error ?? 'unknown error'}`;
        warnings.push(warning);
        this.emit({ type: 'agent.grounding.warning', agentId, data: { warning, correlationId: options.correlationId } });
        continue;
      }

      if (plan.toolName === 'fs.list') {
        const entries = Array.isArray((result.result as { entries?: unknown } | undefined)?.entries)
          ? (result.result as { entries: unknown[] }).entries.filter((item): item is string => typeof item === 'string')
          : [];
        observedPaths.push(...entries.slice(0, 80));
        summaries.push(entries.slice(0, 80).join('\n'));
        contexts.push(`Filesystem listing:\n${entries.join('\n')}`);
      } else if (plan.toolName === 'fs.read') {
        const read = result.result as { path?: unknown; content?: unknown } | undefined;
        if (typeof read?.path === 'string') observedPaths.push(read.path);
        const content = typeof read?.content === 'string' ? read.content.slice(0, 8000) : '';
        summaries.push(`${String(read?.path ?? 'file')}: ${content.slice(0, 1000)}`);
        contexts.push(`File read result for ${String(read?.path ?? 'file')}:\n${content}`);
      } else if (plan.toolName === 'shell.exec') {
        const shell = result.result as { command?: unknown; stdout?: unknown; stderr?: unknown } | undefined;
        const output = [shell?.stdout, shell?.stderr].filter(value => typeof value === 'string' && value).join('\n');
        summaries.push(`${String(shell?.command ?? 'command')}: ${output.slice(0, 1600)}`);
        contexts.push(`Command result for ${String(shell?.command ?? 'command')}:\n${output.slice(0, 8000)}`);
      }
    }

    const successful = toolCalls.filter(call => call.success);
    return {
      toolCalls,
      grounded: plans.every((plan, index) => !plan.groundingRequired || toolCalls[index]?.success),
      warnings,
      evidence: {
        toolGrounded: successful.length > 0,
        outputGrounded: false,
        observedPaths: Array.from(new Set(observedPaths)),
        toolResultSummary: summaries.filter(Boolean).join('\n\n'),
      },
      context: contexts.join('\n\n'),
    };
  }

  private buildGroundedTask(task: string, grounding: { context: string; warnings: string[] }): string {
    if (!grounding.context && grounding.warnings.length === 0) return task;
    return [
      task,
      grounding.context ? `\nGrounding context:\n${grounding.context}` : '',
      grounding.warnings.length > 0 ? `\nGrounding warnings:\n${grounding.warnings.join('\n')}` : '',
    ].filter(Boolean).join('\n');
  }

  private resultIncludesEvidence(result: string, evidence: RunEvidence): boolean {
    if (!result.trim()) return false;
    const normalized = result.toLowerCase();
    if (evidence.observedPaths.slice(0, 80).some(item => normalized.includes(item.toLowerCase()))) return true;
    const evidenceTerms = (evidence.toolResultSummary ?? '')
      .toLowerCase()
      .split(/[^a-z0-9._/-]+/)
      .filter(term => term.length >= 4)
      .slice(0, 30);
    return evidenceTerms.some(term => normalized.includes(term));
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

  private buildAgentTree(agent: AgentInfo): AgentTreeNode {
    return {
      agent,
      children: this.getChildren(agent.identity.id).map(child => this.buildAgentTree(child)),
    };
  }

  private buildRuntimeAgentActorTree(agent: AgentInfo, visited: Set<string>): RuntimeAgentActorNode {
    const key = `agent:${agent.identity.id}`;
    if (visited.has(key)) return { type: 'agent', agent, children: [] };
    const nextVisited = new Set(visited).add(key);
    const teamChildren = this.teams.list()
      .filter(team => team.identity.parentAgentId === agent.identity.id)
      .map(team => this.buildRuntimeTeamActorTree(team, nextVisited));
    const agentChildren = this.getChildren(agent.identity.id)
      .filter(child => !child.identity.teamId)
      .map(child => this.buildRuntimeAgentActorTree(child, nextVisited));
    return {
      type: 'agent',
      agent,
      children: [...teamChildren, ...agentChildren],
    };
  }

  private buildRuntimeTeamActorTree(team: TeamRuntimeState, visited: Set<string>): RuntimeTeamActorNode {
    const key = `team:${team.identity.id}`;
    if (visited.has(key)) return { type: 'team', team, children: [] };
    const nextVisited = new Set(visited).add(key);
    const ctx = this.getContext();
    const children = team.memberAgentIds
      .map(agentId => ctx.manager.getAgentById(agentId)?.getInfo())
      .filter((agent): agent is AgentInfo => Boolean(agent))
      .map(agent => this.buildRuntimeAgentActorTree(agent, nextVisited));
    return { type: 'team', team, children };
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

  private createSubagentToMProfile(archetype: SubAgentArchetype, subjectAgentId: string, task: string, parentId = 'root'): ToMProfile {
    const level = this.defaultToMLevel(archetype);
    const models: ToMProfile['models'] = [];
    const recursiveModels: NonNullable<ToMProfile['recursiveModels']> = [];
    if (level >= 1) {
      models.push({
        targetId: parentId,
        targetType: 'agent',
        goalModel: ['receive a reliable result that advances the delegated task'],
        intentModel: ['delegate a bounded specialist task and synthesize the result upward'],
      });
    }
    if (archetype === 'critic') {
      models.push({
        targetId: 'user',
        targetType: 'user',
        goalModel: ['receive a grounded result with explicit limitations'],
        uncertaintyModel: ['the delegated report may omit evidence or overstate conclusions'],
      });
      recursiveModels.push({
        observerId: parentId,
        targetId: subjectAgentId || 'critic',
        relation: 'parent expects evidence-aware critique',
        description: 'Evaluate whether the candidate result satisfies the parent goal and user intent.',
      });
    }
    return {
      level,
      subjectAgentId,
      models,
      recursiveModels: recursiveModels.length > 0 ? recursiveModels : undefined,
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
      sessionId: message.sessionId,
      correlationId: message.correlationId,
      nodeId: message.metadata?.nodeId,
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
