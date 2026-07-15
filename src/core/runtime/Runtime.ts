// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { config } from '../../config/index.js';
import { logger } from '../utils/logger.js';
import { configureLogging, shutdownLogging } from '../logging/index.js';
import {
  llmFactory,
  tokenUsageRegistry,
  type LLMCompletionOptions,
  type LLMCompletionResult,
  type LLMMessage,
  type LLMProvider,
  type ModelTokenUsage,
} from '../llm/index.js';
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
import {
  DefaultDelegationCandidatePlanner,
  HashTaskEmbeddingProvider,
  type DelegationCandidateInput,
  type DelegationCandidateSelection,
  type LLMDelegationScorerInvocation,
} from '../delegation/index.js';
import {
  CompositeEvolutionEvaluator,
  EvolutionLifecycleEngine,
  TeamFirstGenomePlanner,
  WeightedTopKSelectionPolicy,
  defaultMutationOperators,
  type EvolutionAblations,
  type EvolutionCandidate,
  type EvolutionEvaluationDimensions,
  type EvolutionEvaluationResult,
  type EvolutionExecutionArtifact,
  type EvolutionLifecycleResult,
  type EvolutionJudge,
  type EvolutionPattern,
  type EvolutionProfile,
  type EvolutionRunOptions,
  type EvolutionRunResult,
  type EvolutionSeedAgent,
  type GenomeToMProfile,
} from '../evolution/index.js';
import {
  normalizeToMProfile,
  ToMDelegationPlanner,
  type ToMAnalysisSignals,
  type ToMDelegationEngine,
  type ToMTaskAnalysis,
} from '../tom/index.js';
import {
  AgentCommunicationManager,
  type AgentCommunicationProtocol,
  type CommunicationState,
  type MultiPartyTrace,
} from '../communication/index.js';
import { ContextWindowManager, type ContextWindow } from '../context/index.js';
import {
  BudgetMarket,
  WeightedReasoningInvestmentModel,
  type BudgetAllocation,
  type BudgetMarketState,
  type BudgetOutcome,
  type BudgetPriority,
  type BudgetRebalanceResult,
  type BudgetRequest,
  type ReasoningInvestmentModel,
} from '../budget/index.js';
import {
  executeTeamItems,
  normalizeTeamExecutionPolicy,
  TeamRegistry,
  type TeamExecutionOutcome,
  type TeamExecutionPolicy,
  type TeamFSMState,
  type TeamRuntimeState,
} from '../team/index.js';
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
  communicationProtocols?: AgentCommunicationProtocol[];
  tomPlanner?: ToMDelegationEngine;
  reasoningInvestmentModel?: ReasoningInvestmentModel;
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
  communication: AgentCommunicationManager;
  capabilities: {
    skills: number;
    actions: number;
    tools: number;
  };
}

export interface TokenUsage extends AgentUsage {
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number | null;
  cachedInputTokens: number | null;
  cacheCreationInputTokens: number | null;
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

interface DelegationScorerBudgetContext {
  parentId: string;
  correlationId?: string;
  usageBefore: AgentUsage;
  allocation?: BudgetAllocation;
  ownsAllocation: boolean;
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

export interface ToMRuntimeState {
  analyses: ToMTaskAnalysis[];
  agents: Array<{
    agentId: string;
    name: string;
    parentId?: string;
    teamId?: string;
    profile: ToMProfile;
  }>;
  teams: Array<{
    teamId: string;
    name: string;
    parentAgentId: string;
    profile: ToMProfile;
  }>;
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
  cognitiveGapIds?: string[];
  existenceReason?: string;
  communicationProtocol?: string;
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
  tomProfile?: ToMProfile;
  cognitiveGapIds?: string[];
  existenceReason?: string;
  lead?: boolean;
  communicationProtocol?: string;
  systemPrompt?: string;
}

export interface SpawnTeamSpec {
  parentAgentId?: string;
  name: string;
  description: string;
  tomLevel?: number;
  tomProfile?: ToMProfile;
  tomAnalysis?: ToMTaskAnalysis;
  leadAgentId?: string;
  task?: string;
  members?: TeamMemberSpec[];
  correlationId?: string;
  executionPolicy?: Partial<TeamExecutionPolicy>;
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
  memberOutcomes: TeamMemberRunOutcome[];
  correlationId: string;
  messages: RuntimeMessage[];
  usage: TokenUsage;
}

export interface TeamMemberRunOutcome {
  key: string;
  agentId?: string;
  status: TeamExecutionOutcome<unknown>['status'];
  error?: string;
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
  cognitiveGapIds?: string[];
  existenceReason?: string;
  communicationProtocol?: string;
  systemPrompt?: string;
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
  tomProfile?: ToMProfile;
  cognitiveGapIds?: string[];
  existenceReason?: string;
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
  evolution?: EvolutionRunResult;
  messages: RuntimeMessage[];
  usage: {
    root: TokenUsage;
    subagents: Record<string, TokenUsage>;
    teamSynthesis: Record<string, TokenUsage>;
    total: TokenUsage;
  };
}

export interface RunEvolutionInput {
  task: string;
  parentId?: string;
  correlationId?: string;
  seedAgents?: EvolutionSeedAgent[];
  profile?: EvolutionProfile;
  options?: Partial<Omit<EvolutionRunOptions, 'ablations'>> & {
    ablations?: Partial<EvolutionAblations>;
  };
}

export interface EvolutionBenchmarkResult {
  task: string;
  profiles: EvolutionProfile[];
  runs: EvolutionRunResult[];
  comparison: Array<{
    profile: EvolutionProfile;
    success: boolean;
    score: number;
    totalTokens: number;
    thinkingTokens: number | null;
    wallClockMs: number;
    agentsSpawned: number;
    teamsSpawned: number;
  }>;
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
  private tomPlanner: ToMDelegationEngine = new ToMDelegationPlanner();
  private reasoningInvestmentModel: ReasoningInvestmentModel = new WeightedReasoningInvestmentModel();
  private readonly tomAnalyses = new Map<string, ToMTaskAnalysis>();
  private communicationManager: AgentCommunicationManager | null = null;
  private evolutionSequence = 0;
  private readonly evolutionRuns: EvolutionRunResult[] = [];
  private readonly archivedAgentUsage = new Map<string, TokenUsage>();
  private readonly evolutionBudgetBypassCorrelations = new Set<string>();

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
    this.tomPlanner = options.tomPlanner ?? new ToMDelegationPlanner();
    this.reasoningInvestmentModel = options.reasoningInvestmentModel ?? new WeightedReasoningInvestmentModel();
    const communication = new AgentCommunicationManager(
      this.workspaceRuntimeConfig.communication,
      options.communicationProtocols
    );
    this.communicationManager = communication;
    this.candidatePlanner = new DefaultDelegationCandidatePlanner({
      llm,
      enabledScorers: this.workspaceRuntimeConfig.delegation.candidateScoring.enabledScorers,
      minimumScore: this.workspaceRuntimeConfig.delegation.candidateScoring.minimumScore,
      minimumToMCoverage: this.workspaceRuntimeConfig.tom.minimumCoverage,
      investmentModel: this.reasoningInvestmentModel,
      llmHooks: {
        before: (input, messages, completionOptions) => this.beforeDelegationScorerCall(input, messages, completionOptions),
        after: (completion, input, hookContext) => this.afterDelegationScorerCall(completion, input, hookContext),
        failed: (_error, _input, hookContext) => this.releaseDelegationScorerBudget(hookContext),
      },
    });
    this.budgetMarket = new BudgetMarket(() => this.ctx ? this.getAccountedRuntimeUsedTokens() : 0, {
      mode: this.workspaceRuntimeConfig.budgetMarket.mode,
      minimumGrantTokens: this.workspaceRuntimeConfig.budgetMarket.minimumGrantTokens,
      accountingDimension: this.workspaceRuntimeConfig.budgetMarket.accountingDimension,
      priorityWeights: this.workspaceRuntimeConfig.budgetMarket.priorityWeights,
      investmentModel: this.reasoningInvestmentModel,
    });
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
      communicationProtocol: communication.getDefaultProtocolId(),
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
      communication,
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
    this.communicationManager = null;
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
    this.tomAnalyses.clear();
    this.evolutionRuns.length = 0;
    this.archivedAgentUsage.clear();
    this.evolutionBudgetBypassCorrelations.clear();
    this.evolutionSequence = 0;
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

  getToMState(correlationId?: string): ToMRuntimeState {
    const ctx = this.getContext();
    const analyses = correlationId
      ? [this.tomAnalyses.get(correlationId)].filter((item): item is ToMTaskAnalysis => Boolean(item))
      : [...this.tomAnalyses.values()];
    return {
      analyses: analyses.map(analysis => ({
        ...analysis,
        parentBeliefs: [...analysis.parentBeliefs],
        parentGoals: [...analysis.parentGoals],
        parentUncertainties: [...analysis.parentUncertainties],
        gaps: analysis.gaps.map(gap => ({
          ...gap,
          beliefScope: [...gap.beliefScope],
          uncertainty: [...gap.uncertainty],
          requiredCapabilities: [...gap.requiredCapabilities],
          modelsTargets: [...gap.modelsTargets],
        })),
      })),
      agents: ctx.manager.listAgentInfo().map(agent => ({
        agentId: agent.identity.id,
        name: agent.identity.name,
        parentId: agent.identity.parentId,
        teamId: agent.identity.teamId,
        profile: normalizeToMProfile(agent.identity.tomProfile, agent.identity.tomProfile),
      })),
      teams: this.teams.list().map(team => ({
        teamId: team.identity.id,
        name: team.identity.name,
        parentAgentId: team.identity.parentAgentId,
        profile: normalizeToMProfile(team.identity.tomProfile, team.identity.tomProfile),
      })),
    };
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
    for (const [agentId, usage] of this.archivedAgentUsage) {
      perAgent[agentId] = { ...usage };
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

  private getAccountedRuntimeUsedTokens(): number {
    const budget = this.getBudgetState();
    const dimension = this.workspaceRuntimeConfig?.budgetMarket.accountingDimension ?? 'total_tokens';
    // Team tokenUsage includes member usage, which is already present in perAgent.
    // Only team synthesis is an additional model call at this aggregation level.
    const usage = [
      ...Object.values(budget.perAgent),
      ...this.teams.list().map(team => team.synthesisUsage),
    ];
    return usage.reduce((sum, item) => {
      if (dimension === 'output_tokens') return sum + item.outputTokens;
      if (dimension === 'thinking_tokens') {
        return sum + (item.thinkingAccountingTokens ?? item.thinkingTokens ?? item.totalTokens);
      }
      return sum + item.totalTokens;
    }, 0);
  }

  rebalanceBudgetMarket(): BudgetRebalanceResult {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const result = this.budgetMarket.rebalance();
    this.emit({
      type: 'budget.rebalanced',
      data: {
        changed: result.changed.map(item => item.id),
        releasedTokens: result.releasedTokens,
        reservedTokens: result.reservedTokens,
      },
    });
    return result;
  }

  getBudgetAllocation(allocationId: string): BudgetAllocation | undefined {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    return this.budgetMarket.getAllocation(allocationId);
  }

  allocateBudget(request: BudgetRequest): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    this.emit({
      type: 'budget.requested',
      agentId: request.requesterId,
      correlationId: request.correlationId,
      data: { ...request },
    });
    const allocation = this.budgetMarket.request(request);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.allocated' : 'budget.denied',
      agentId: request.requesterId,
      correlationId: request.correlationId,
      data: {
        allocationId: allocation.id,
        requestedTokens: request.requestedTokens,
        allocatedTokens: allocation.allocatedTokens,
        policy: allocation.policy,
        score: allocation.score,
        rationale: allocation.rationale,
      },
    });
    return allocation;
  }

  allocateBudgets(requests: BudgetRequest[]): BudgetAllocation[] {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    for (const request of requests) {
      this.emit({ type: 'budget.requested', agentId: request.requesterId, correlationId: request.correlationId, data: { ...request } });
    }
    const allocations = this.budgetMarket.requestMany(requests);
    for (const allocation of allocations) {
      this.emit({
        type: allocation.status === 'granted' ? 'budget.allocated' : 'budget.denied',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: {
          allocationId: allocation.id,
          requestedTokens: allocation.request.requestedTokens,
          allocatedTokens: allocation.allocatedTokens,
          policy: allocation.policy,
          score: allocation.score,
          rationale: allocation.rationale,
        },
      });
    }
    return allocations;
  }

  consumeBudget(allocationId: string, usage: number | ModelTokenUsage): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.consume(allocationId, usage);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.consumed',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        consumedTokens: allocation.consumedTokens,
        utilization: allocation.utilization,
      },
    });
    if (allocation.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens },
      });
    }
    return allocation;
  }

  settleBudget(allocationId: string, usage: number | ModelTokenUsage): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.settle(allocationId, usage);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.settled',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        allocatedTokens: allocation.allocatedTokens,
        consumedTokens: allocation.consumedTokens,
        utilization: allocation.utilization,
        efficiency: allocation.efficiency,
        status: allocation.status,
      },
    });
    if (allocation.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens },
      });
    }
    return allocation;
  }

  recordBudgetOutcome(allocationId: string, outcome: BudgetOutcome): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.recordOutcome(allocationId, outcome);
    if (!allocation) throw new Error(`Budget allocation "${allocationId}" not found or cannot accept an outcome`);
    this.emit({
      type: 'budget.outcome.recorded',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        success: allocation.outcome?.success,
        realizedUtility: allocation.outcome?.realizedUtility,
        efficiency: allocation.efficiency,
      },
    });
    return allocation;
  }

  releaseBudget(allocationId: string, reason = 'released_by_controller'): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.release(allocationId, reason);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.released',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: { allocationId, reason },
    });
    return allocation;
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
      const teamKey = this.safeAgentKey(spec.name);
      const cachedTeamPattern = (await ctx.memory.getCachePatterns('teams'))
        .find(item => item.id === `team_pattern_${teamKey}_v1` || item.key === teamKey);
      const cachedMembers = Array.isArray(cachedTeamPattern?.members)
        ? cachedTeamPattern.members as TeamMemberSpec[]
        : [];
      const cachedExecutionPolicy = cachedTeamPattern?.executionPolicy
        && typeof cachedTeamPattern.executionPolicy === 'object'
        ? cachedTeamPattern.executionPolicy as Partial<TeamExecutionPolicy>
        : {};
      const configuredTeamPolicy = this.workspaceRuntimeConfig?.teams;
      const executionPolicy = normalizeTeamExecutionPolicy({
        mode: configuredTeamPolicy?.executionMode,
        failureMode: configuredTeamPolicy?.failureMode,
        maxConcurrency: configuredTeamPolicy?.maxConcurrency,
        minimumSuccessfulMembers: configuredTeamPolicy?.minimumSuccessfulMembers,
        ...cachedExecutionPolicy,
        ...spec.executionPolicy,
      });
      const requestedMembers = spec.members ?? cachedMembers;
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
        tomProfile: member.tomProfile ?? this.createSubagentToMProfile(
          member.archetype,
          member.name ?? member.archetype,
          member.task,
          parentAgentId
        ),
      }));
      const maxMembers = this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5;
      if (members.length > maxMembers) {
        throw new Error(`Team member limit exceeded: requested ${members.length}, maximum ${maxMembers}`);
      }
      if (members.length > 0 && executionPolicy.minimumSuccessfulMembers > members.length) {
        throw new Error(
          `Team minimumSuccessfulMembers ${executionPolicy.minimumSuccessfulMembers} exceeds planned members ${members.length}`
        );
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
      const teamTomProfile = spec.tomProfile ?? this.tomPlanner.createTeamProfile({
        teamId: 'pending-team',
        parentId: parentAgentId,
        task: spec.task ?? spec.description,
        members,
      });
      const team = this.teams.create({
        name: spec.name,
        parentAgentId,
        description: spec.description,
        generation: parent.getIdentity().generation + 1,
        tomLevel: teamTomProfile.level,
        tomProfile: teamTomProfile,
        leadAgentId: spec.leadAgentId,
        task: spec.task,
        correlationId,
        executionPolicy: { ...executionPolicy },
      });
      createdTeamId = team.identity.id;
      this.teamMemberPlans.set(team.identity.id, members.map(member => ({ ...member })));
      await ctx.memory.ensureTeamMemory(teamKey, { name: spec.name, purpose: spec.description });
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
        tomLevel: team.identity.tomLevel,
        tomProfile: team.identity.tomProfile,
        tomAnalysis: spec.tomAnalysis,
        leadArchetype: members.find(member => member.lead)?.archetype,
        members: members.map(member => ({
          archetype: member.archetype,
          name: member.name,
          role: member.role,
          task: member.task,
          tools: member.tools,
          skills: member.skills,
          tomLevel: member.tomLevel,
          tomProfile: member.tomProfile,
          cognitiveGapIds: member.cognitiveGapIds,
          existenceReason: member.existenceReason,
          systemPrompt: member.systemPrompt,
          lead: member.lead ?? false,
        })),
        executionPolicy: { ...executionPolicy },
      });
      await ctx.memory.writeTeamTopology(teamKey, {
        type: 'subteam',
        teamId: team.identity.id,
        parentAgentId,
        leadAgentId: spec.leadAgentId,
        members: [],
        plannedMembers: members,
        tomLevel: team.identity.tomLevel,
        tomProfile: team.identity.tomProfile,
        tomAnalysis: spec.tomAnalysis,
        executionPolicy,
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
          tomProfile: team.identity.tomProfile,
          plannedMembers: members.length,
          patternId: pattern.id,
          parentAgentId,
          executionPolicy,
          cognitiveGapIds: team.identity.tomProfile.cognitiveGaps,
          perspective: team.identity.tomProfile.perspective,
        },
      });
      this.emit({
        type: 'tom.team.profile.created',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          parentAgentId,
          profile: team.identity.tomProfile,
          analysisId: spec.tomAnalysis?.id,
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
      tomProfile: spec.tomProfile ?? this.createSubagentToMProfile(
        spec.archetype,
        spec.name ?? spec.archetype,
        spec.task,
        team.identity.parentAgentId
      ),
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
        task: member.task,
        tools: member.tools,
        skills: member.skills,
        tomLevel: member.tomLevel,
        tomProfile: member.tomProfile,
        cognitiveGapIds: member.cognitiveGapIds,
        existenceReason: member.existenceReason,
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
    const membersBefore = new Set(team.memberAgentIds);
    let execution: Awaited<ReturnType<Runtime['createAgentComputeNode']>>;
    try {
      execution = await this.createAgentComputeNode({
        parentId: team.identity.parentAgentId,
        archetype: spec.archetype,
        task: spec.task,
        name: spec.name,
        role: spec.role,
        style: spec.style,
        tools: spec.tools,
        skills: spec.skills,
        budgetTokens: spec.budgetTokens,
        tomProfile: spec.tomProfile ?? (spec.tomLevel === undefined
          ? undefined
          : {
            ...this.createSubagentToMProfile(spec.archetype, '', spec.task, team.identity.parentAgentId),
            level: spec.tomLevel as ToMProfile['level'],
          }),
        tomProfileMode: 'runtime_assignment',
        cognitiveGapIds: spec.cognitiveGapIds,
        existenceReason: spec.existenceReason,
        systemPrompt: spec.systemPrompt,
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
    } catch (error) {
      const createdAgentId = this.teams.get(teamId)?.memberAgentIds.find(agentId => !membersBefore.has(agentId));
      const failure = error instanceof Error ? error : new Error(String(error));
      if (createdAgentId) Object.assign(failure, { teamMemberAgentId: createdAgentId });
      throw failure;
    }
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

  async runTeam(teamId: string, task: string, options: { correlationId?: string } = {}): Promise<TeamRunResult> {
    const ctx = this.getContext();
    const initial = this.teams.get(teamId);
    if (!initial) throw new Error(`Team "${teamId}" not found`);
    if (!task.trim()) throw new Error('Team task is required');
    const usageBefore = { ...initial.tokenUsage };
    // A team definition can run repeatedly, but every execution is a distinct trace.
    const correlationId = options.correlationId ?? this.createCorrelationId();
    this.teams.setTask(teamId, task, correlationId);
    await this.transitionTeamFsm(teamId, 'S_team_plan', { task });
    this.emit({
      type: 'team.execution.policy.applied',
      agentId: teamId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId, ...initial.executionPolicy },
    });

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

    type TeamWorkValue = {
      agentId: string;
      result: RunAgentResult;
      execution?: RootMediatedSpawnResult;
    };
    let executionOutcomes: TeamExecutionOutcome<TeamWorkValue>[];
    try {
      const plans = this.teamMemberPlans.get(teamId) ?? [];
      if (plans.length > 0) {
        await this.transitionTeamFsm(teamId, 'S_member_spawn', { count: plans.length });
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: plans.length });
        executionOutcomes = await executeTeamItems(plans.map((plan, index) => {
          const key = `planned:${index + 1}:${plan.archetype}`;
          return {
            key,
            execute: async (): Promise<TeamWorkValue> => {
              this.teams.markMemberRunning(teamId, key);
              this.emit({
                type: 'team.member.started',
                agentId: teamId,
                sessionId: ctx.sessionId,
                correlationId,
                data: { teamId, memberKey: key, archetype: plan.archetype, task: plan.task },
              });
              const execution = await this.executeTeamMember(teamId, plan);
              this.teams.clearMemberTracking(teamId, key);
              return {
                agentId: execution.agent.identity.id,
                result: execution.subagentResult,
                execution,
              };
            },
          };
        }), initial.executionPolicy);
        this.teamMemberPlans.set(teamId, []);
      } else {
        const team = this.teams.get(teamId)!;
        if (team.memberAgentIds.length === 0) throw new Error(`Team "${teamId}" has no members or member plans`);
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: team.memberAgentIds.length });
        executionOutcomes = await executeTeamItems(team.memberAgentIds.map(agentId => {
          const memberTask = team.memberTasks[agentId] ?? task;
          return {
            key: agentId,
            execute: async (): Promise<TeamWorkValue> => {
              this.teams.markMemberRunning(teamId, agentId);
              this.emit({
                type: 'team.member.started',
                agentId,
                sessionId: ctx.sessionId,
                correlationId,
                data: { teamId, memberKey: agentId, task: memberTask },
              });
              const result = await this.runAgent(agentId, memberTask, {
                correlationId,
                disableRecursiveDelegation: true,
              });
              this.teams.recordMemberResult(teamId, agentId, memberTask, result.result, result.usage);
              this.emit({
                type: 'team.member.completed',
                agentId,
                sessionId: ctx.sessionId,
                correlationId,
                data: { teamId, task: memberTask, totalTokens: result.usage.totalTokens },
              });
              return { agentId, result };
            },
          };
        }), initial.executionPolicy);
      }

      for (const outcome of executionOutcomes) {
        if (outcome.status === 'completed') continue;
        if (outcome.status === 'failed') {
          const failedAgentId = outcome.cause instanceof Error
            && typeof (outcome.cause as Error & { teamMemberAgentId?: unknown }).teamMemberAgentId === 'string'
            ? (outcome.cause as Error & { teamMemberAgentId: string }).teamMemberAgentId
            : undefined;
          const memberKey = failedAgentId ?? outcome.key;
          if (memberKey !== outcome.key) this.teams.clearMemberTracking(teamId, outcome.key);
          this.teams.recordMemberFailure(teamId, memberKey, outcome.error ?? 'unknown member execution failure');
          this.emit({
            type: 'team.member.failed',
            agentId: failedAgentId ?? outcome.value?.agentId ?? teamId,
            sessionId: ctx.sessionId,
            correlationId,
            data: { teamId, memberKey, error: outcome.error },
          });
        } else {
          this.teams.markMemberSkipped(teamId, outcome.key);
          this.emit({
            type: 'team.member.skipped',
            agentId: teamId,
            sessionId: ctx.sessionId,
            correlationId,
            data: { teamId, memberKey: outcome.key, reason: 'fail_fast' },
          });
        }
      }
      const completedOutcomes = executionOutcomes.filter(
        (outcome): outcome is TeamExecutionOutcome<TeamWorkValue> & { status: 'completed'; value: TeamWorkValue } =>
          outcome.status === 'completed' && outcome.value !== undefined
      );
      const failedOutcomes = executionOutcomes.filter(outcome => outcome.status === 'failed');
      if (initial.executionPolicy.failureMode === 'fail_fast' && failedOutcomes.length > 0) {
        throw new Error(`Team member execution failed: ${failedOutcomes[0].error ?? failedOutcomes[0].key}`);
      }
      if (completedOutcomes.length < initial.executionPolicy.minimumSuccessfulMembers) {
        throw new Error(
          `Team completed ${completedOutcomes.length} members, below minimum ${initial.executionPolicy.minimumSuccessfulMembers}`
        );
      }
      const members = completedOutcomes.map(outcome => outcome.value.result);
      const memberExecutions = completedOutcomes
        .map(outcome => outcome.value.execution)
        .filter((execution): execution is RootMediatedSpawnResult => execution !== undefined);

      await this.transitionTeamFsm(teamId, 'S_member_aggregate', {
        completed: members.length,
        failed: failedOutcomes.length,
      });
      await this.transitionTeamFsm(teamId, 'S_team_synthesize', {
        completed: members.length,
        failed: failedOutcomes.length,
      });
      const synthesis = await this.completeAsTeam(
        this.teams.get(teamId)!,
        task,
        members,
        failedOutcomes,
        correlationId
      );
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
          failedMembers: failedOutcomes.length,
          partial: failedOutcomes.length > 0,
        },
      });
      return {
        team: this.teams.get(teamId)!,
        result: synthesis.content,
        members,
        memberExecutions,
        memberOutcomes: executionOutcomes.map(outcome => ({
          key: outcome.key,
          agentId: outcome.value?.agentId ?? (
            outcome.cause instanceof Error
            && typeof (outcome.cause as Error & { teamMemberAgentId?: unknown }).teamMemberAgentId === 'string'
              ? (outcome.cause as Error & { teamMemberAgentId: string }).teamMemberAgentId
              : undefined
          ),
          status: outcome.status,
          error: outcome.error,
        })),
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
    requesterId?: string;
    archetype: SubAgentArchetype;
    correlationId?: string;
    nodeId?: string;
    requestedTokens?: number;
    minimumTokens?: number;
    priority?: BudgetPriority;
    expectedUtility?: number;
    purpose: string;
  }): Promise<BudgetAllocation | undefined> {
    if (input.correlationId && this.evolutionBudgetBypassCorrelations.has(input.correlationId)) {
      this.emit({
        type: 'budget.bypassed',
        agentId: input.requesterId ?? input.parentId,
        correlationId: input.correlationId,
        data: { reason: 'evolution_ablation_without_budget_market', requestedTokens: input.requestedTokens, purpose: input.purpose },
      });
      return undefined;
    }
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = input.requestedTokens ?? this.estimateAgentBudget(input.archetype);
    const requesterId = input.requesterId ?? `${input.parentId}:${input.archetype}`;
    const analysis = input.correlationId ? this.tomAnalyses.get(input.correlationId) : undefined;
    const tools = this.getDefaultToolBindings(input.archetype).filter(binding => binding.enabled).length;
    const cachedPattern = await this.getContext().memory.findAgentPattern(input.archetype);
    const evaluation = cachedPattern?.evaluation && typeof cachedPattern.evaluation === 'object'
      ? cachedPattern.evaluation as Record<string, unknown>
      : undefined;
    const historicalUtility = averageDefined([
      finiteRatio(evaluation?.successRate),
      finiteRatio(evaluation?.groundingRate),
    ]) ?? 0.5;
    const evaluationRuns = finiteNumber(evaluation?.runs) ?? 0;
    const cacheConfidence = cachedPattern
      ? Math.min(0.95, 0.5 + evaluationRuns * 0.08)
      : 0;
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
      requesterId,
      parentId: input.parentId,
      actorType: 'agent',
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens: input.minimumTokens ?? this.workspaceRuntimeConfig?.budgetMarket.minimumGrantTokens ?? 256,
      priority: input.priority ?? this.workspaceRuntimeConfig?.budgetMarket.defaultPriority ?? 'medium',
      expectedUtility: input.expectedUtility ?? this.defaultBudgetUtility(input.archetype),
      purpose: input.purpose,
      resourceEstimate: {
        tokens: requestedTokens,
        contextTokens: Math.min(4000, Math.round(requestedTokens * 0.35)),
        toolCalls: tools,
      },
      metadata: {
        investmentKind: input.purpose,
        evidenceGain: analysis?.gaps.some(gap => gap.kind === 'evidence') ? 0.82 : 0.25,
        uncertaintyReduction: analysis ? Math.min(1, analysis.gaps.length / 4) : 0.4,
        conflictResolution: analysis?.signals.conflictLevel ?? 0,
        verificationGain: input.archetype === 'tester' ? 0.9 : 0,
        historicalUtility,
        cacheConfidence,
        executionRisk: analysis ? Math.min(1, (1 - analysis.confidence) + analysis.signals.failedTraceCount * 0.08) : 0.25,
        confidence: analysis?.confidence ?? 0.62,
        investmentHistoryKey: typeof cachedPattern?.id === 'string'
          ? `${cachedPattern.id}:${input.purpose}`
          : `${input.archetype}:${input.purpose}`,
      },
    });
    if (this.workspaceRuntimeConfig?.budgetMarket.rebalanceOnRequest && allocation.status === 'denied') {
      this.rebalanceBudgetMarket();
    }
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
        allocatedTokens: allocation.allocatedTokens,
        policy: allocation.policy,
        score: allocation.score,
        reason: allocation.reason,
      },
    });
    if (allocation.status === 'granted') {
      this.emit({
        type: 'budget.allocated',
        agentId: requesterId,
        correlationId: input.correlationId,
        nodeId: input.nodeId,
        data: {
          allocationId: allocation.id,
          requestedTokens,
          allocatedTokens: allocation.allocatedTokens,
          policy: allocation.policy,
          rationale: allocation.rationale,
        },
      });
    }
    return allocation;
  }

  private budgetAccountingDimension(): 'total_tokens' | 'output_tokens' | 'thinking_tokens' {
    return this.workspaceRuntimeConfig?.budgetMarket.accountingDimension ?? 'total_tokens';
  }

  private budgetRequestTokens(inputTokens: number, completionTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens'
      ? inputTokens + completionTokens
      : completionTokens;
  }

  private budgetMinimumTokens(inputTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens' ? inputTokens + 1 : 1;
  }

  private completionCapacity(allocatedTokens: number, inputTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens'
      ? Math.max(0, allocatedTokens - inputTokens)
      : Math.max(0, allocatedTokens);
  }

  private async requestTeamSynthesisBudget(input: {
    team: TeamRuntimeState;
    correlationId: string;
    promptTokens: number;
    completionTokens: number;
  }): Promise<BudgetAllocation | undefined> {
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = this.budgetRequestTokens(input.promptTokens, input.completionTokens);
    const minimumTokens = this.budgetMinimumTokens(input.promptTokens);
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
      actorType: 'team',
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens,
      priority: 'high',
      expectedUtility: 0.85,
      purpose: 'team_synthesis',
      resourceEstimate: {
        tokens: requestedTokens,
        inputTokens: input.promptTokens,
        outputTokens: input.completionTokens,
        contextTokens: input.promptTokens,
      },
      metadata: {
        investmentKind: 'team_synthesis',
        parentUtility: teamCompletionRatio(input.team),
        conflictResolution: input.team.identity.tomProfile.level >= 2 ? 0.9 : 0.55,
        uncertaintyReduction: Math.min(1, input.team.memberAgentIds.length / 4),
        executionRisk: Object.values(input.team.memberStatuses).some(status => status === 'failed') ? 0.65 : 0.2,
        confidence: Object.values(input.team.memberStatuses).some(status => status === 'failed') ? 0.55 : 0.8,
      },
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
    actual: number | TokenUsage,
    correlationId: string
  ): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const modelTotalTokens = typeof actual === 'number' ? actual : actual.totalTokens;
    const settled = this.budgetMarket.settle(allocation.id, actual);
    if (!settled) return;
    const latestTeam = this.teams.get(teamId);
    const completedMembers = latestTeam
      ? Object.values(latestTeam.memberStatuses).filter(status => status === 'completed').length
      : 0;
    this.recordBudgetOutcome(allocation.id, {
      success: true,
      conflictResolution: latestTeam?.memberAgentIds.length
        ? completedMembers / latestTeam.memberAgentIds.length
        : 0.5,
      quality: latestTeam ? teamCompletionRatio(latestTeam) : undefined,
      metadata: { teamId, phase: 'synthesis' },
    });
    this.emit({
      type: 'budget.settled',
      agentId: teamId,
      correlationId,
      data: {
        teamId,
        allocationId: allocation.id,
        grantedTokens: allocation.grantedTokens,
        actualTokens: settled.consumedTokens,
        modelTotalTokens,
      },
    });
    this.emit({
      type: 'budget.consumed',
      agentId: teamId,
      correlationId,
      data: {
        teamId,
        allocationId: allocation.id,
        consumedTokens: settled.consumedTokens,
        modelTotalTokens,
        inputTokens: typeof actual === 'number' ? undefined : actual.inputTokens,
        outputTokens: typeof actual === 'number' ? undefined : actual.outputTokens,
        thinkingTokens: typeof actual === 'number' ? undefined : actual.thinkingTokens,
        utilization: settled.utilization,
      },
    });
    if (settled.status === 'exceeded') {
      this.emit({
        type: 'budget.overrun',
        agentId: teamId,
        correlationId,
        data: {
          teamId,
          allocationId: allocation.id,
          grantedTokens: allocation.grantedTokens,
          actualTokens: settled.consumedTokens,
          modelTotalTokens,
        },
      });
      this.emit({
        type: 'budget.exceeded',
        agentId: teamId,
        correlationId,
        data: { allocationId: allocation.id, allocatedTokens: allocation.allocatedTokens, consumedTokens: settled.consumedTokens, modelTotalTokens },
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
      this.recordBudgetOutcome(allocation.id, {
        success: false,
        error: reason,
        metadata: { teamId, phase: 'synthesis' },
      });
      this.emit({
        type: 'budget.released',
        agentId: teamId,
        correlationId,
        data: { teamId, allocationId: allocation.id, reason },
      });
    }
  }

  private settleAgentBudget(
    agentId: string,
    actual: number | TokenUsage,
    outcome: BudgetOutcome = { success: true }
  ): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.settle(allocationId, actual);
    this.agentBudgetAllocations.delete(agentId);
    this.getContext().manager.getAgentById(agentId)?.setCompletionTokenLimit(undefined);
    if (allocation) {
      this.recordBudgetOutcome(allocationId, {
        ...outcome,
        metadata: { ...outcome.metadata, agentId, phase: 'agent_run' },
      });
      const modelTotalTokens = typeof actual === 'number' ? actual : actual.totalTokens;
      this.emit({
        type: 'budget.settled',
        agentId,
        data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens: allocation.consumedTokens, modelTotalTokens },
      });
      this.emit({
        type: 'budget.consumed',
        agentId,
        data: {
          allocationId,
          consumedTokens: allocation.consumedTokens,
          modelTotalTokens,
          inputTokens: typeof actual === 'number' ? undefined : actual.inputTokens,
          outputTokens: typeof actual === 'number' ? undefined : actual.outputTokens,
          thinkingTokens: typeof actual === 'number' ? undefined : actual.thinkingTokens,
          utilization: allocation.utilization,
        },
      });
      if (allocation.status === 'exceeded') {
        this.emit({
          type: 'budget.overrun',
          agentId,
          data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens: allocation.consumedTokens, modelTotalTokens },
        });
        this.emit({
          type: 'budget.exceeded',
          agentId,
          data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens, modelTotalTokens },
        });
      }
    }
  }

  private settleDirectBudget(agentId: string, allocation: BudgetAllocation | undefined, usage: TokenUsage, correlationId?: string): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const settled = this.budgetMarket.settle(allocation.id, usage);
    if (!settled) return;
    this.recordBudgetOutcome(allocation.id, {
      success: true,
      metadata: { agentId, phase: 'direct_reasoning' },
    });
    this.emit({
      type: 'budget.consumed',
      agentId,
      correlationId,
      data: {
        allocationId: allocation.id,
        consumedTokens: settled.consumedTokens,
        modelTotalTokens: usage.totalTokens,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        cachedInputTokens: usage.cachedInputTokens,
        utilization: settled.utilization,
      },
    });
    this.emit({
      type: 'budget.settled',
      agentId,
      correlationId,
      data: {
        allocationId: allocation.id,
        allocatedTokens: allocation.allocatedTokens,
        actualTokens: settled.consumedTokens,
        modelTotalTokens: usage.totalTokens,
      },
    });
    if (settled.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId,
        correlationId,
        data: { allocationId: allocation.id, allocatedTokens: allocation.allocatedTokens, consumedTokens: usage.totalTokens },
      });
      this.emit({
        type: 'budget.overrun',
        agentId,
        correlationId,
        data: {
          allocationId: allocation.id,
          grantedTokens: allocation.grantedTokens,
          actualTokens: settled.consumedTokens,
          modelTotalTokens: usage.totalTokens,
        },
      });
    }
  }

  private consumeActiveAgentBudget(
    agentId: string,
    usage: TokenUsage,
    correlationId?: string,
    purpose?: string
  ): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const before = this.budgetMarket.getAllocation(allocationId);
    const consumed = this.budgetMarket.consume(allocationId, usage);
    if (!consumed) return;
    const consumedDelta = Math.max(0, consumed.consumedTokens - (before?.consumedTokens ?? 0));
    this.emit({
      type: 'budget.consumed',
      agentId,
      correlationId,
      data: {
        allocationId,
        purpose,
        consumedTokens: consumedDelta,
        cumulativeConsumedTokens: consumed.consumedTokens,
        modelTotalTokens: usage.totalTokens,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        utilization: consumed.utilization,
      },
    });
    if (consumed.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId,
        correlationId,
        data: {
          allocationId,
          purpose,
          allocatedTokens: consumed.allocatedTokens,
          consumedTokens: consumed.consumedTokens,
          modelTotalTokens: usage.totalTokens,
        },
      });
    }
  }

  private releaseAgentBudget(agentId: string, reason: string): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.release(allocationId, reason);
    this.agentBudgetAllocations.delete(agentId);
    this.getContext().manager.getAgentById(agentId)?.setCompletionTokenLimit(undefined);
    if (allocation) {
      this.recordBudgetOutcome(allocationId, { success: false, error: reason, metadata: { agentId } });
      this.emit({ type: 'budget.released', agentId, data: { allocationId, reason } });
    }
  }

  private estimateAgentBudget(archetype: SubAgentArchetype): number {
    return this.workspaceRuntimeConfig?.budgetMarket.defaultRequestsByArchetype[archetype] ?? ({
      researcher: 2200,
      critic: 1600,
      planner: 1400,
      coder: 2600,
      summarizer: 1000,
      tester: 1800,
      custom: 1800,
    }[archetype]);
  }

  private defaultBudgetUtility(archetype: SubAgentArchetype): number {
    return ({ researcher: 0.82, critic: 0.72, planner: 0.76, coder: 0.88, summarizer: 0.65, tester: 0.8, custom: 0.7 })[archetype];
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
    const recipientProtocol = ctx.manager.getAgentById(message.to)?.getIdentity().communicationProtocol;
    return ctx.queue.enqueue({
      ...message,
      metadata: {
        ...message.metadata,
        communicationProtocol:
          message.metadata?.communicationProtocol ?? recipientProtocol ?? ctx.communication.getDefaultProtocolId(),
      },
    });
  }

  registerCommunicationProtocol(protocol: AgentCommunicationProtocol): void {
    this.requireCommunicationManager().registerProtocol(protocol);
    this.emit({
      type: 'communication.protocol.registered',
      agentId: 'runtime',
      data: { protocolId: protocol.id, version: protocol.version },
    });
  }

  setDefaultCommunicationProtocol(protocolId: string): void {
    const ctx = this.getContext();
    const previous = ctx.communication.getDefaultProtocolId();
    ctx.communication.setDefaultProtocol(protocolId);
    for (const info of ctx.manager.listAgentInfo()) {
      if (info.identity.communicationProtocol !== previous) continue;
      ctx.manager.getAgentById(info.identity.id)?.setCommunicationProtocol(protocolId);
    }
    this.emit({
      type: 'communication.protocol.default.changed',
      agentId: 'runtime',
      data: { protocolId },
    });
  }

  getCommunicationState(): CommunicationState {
    return this.requireCommunicationManager().getState();
  }

  getCommunicationTraces(options: {
    correlationId?: string;
    agentId?: string;
    limit?: number;
  } = {}): MultiPartyTrace[] {
    const ctx = this.getContext();
    return ctx.communication.traces.list({
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      actorId: options.agentId,
      limit: options.limit,
    });
  }

  injectSystemTrace(agentId: string | 'broadcast', trace: MultiPartyTrace): void {
    const ctx = this.getContext();
    if (!ctx.communication.registry.get(trace.protocolId)) {
      throw new Error(`Communication protocol "${trace.protocolId}" is not registered`);
    }
    if (trace.sessionId !== ctx.sessionId) {
      throw new Error(`Trace session "${trace.sessionId}" does not match active session "${ctx.sessionId}"`);
    }
    ctx.communication.traces.append(trace);
    const targets = agentId === 'broadcast'
      ? ctx.manager.listAgentInfo().map(info => ctx.manager.getAgentById(info.identity.id)!).filter(Boolean)
      : [ctx.manager.getAgentById(agentId)].filter((agent): agent is BaseAgent => Boolean(agent));
    if (targets.length === 0) throw new Error(`Agent "${agentId}" not found`);
    for (const agent of targets) agent.receiveSystemTrace(trace);
    this.emit({
      type: 'communication.trace.injected',
      agentId: agentId === 'broadcast' ? 'runtime' : agentId,
      sessionId: trace.sessionId,
      correlationId: trace.correlationId,
      data: { traceId: trace.id, kind: trace.kind, protocolId: trace.protocolId },
    });
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

  async getEvolutionPatterns(): Promise<EvolutionPattern[]> {
    return this.getContext().memory.getEvolutionPatterns();
  }

  getEvolutionRuns(limit = 20): EvolutionRunResult[] {
    return this.evolutionRuns.slice(-Math.max(1, limit)).map(run => structuredClone(run));
  }

  getEvolutionConfig(): WorkspaceRuntimeConfig['evolution'] {
    if (!this.workspaceRuntimeConfig) throw new Error('Runtime workspace config is not initialized');
    return structuredClone(this.workspaceRuntimeConfig.evolution);
  }

  async updateEvolutionConfig(
    patch: Partial<Omit<WorkspaceRuntimeConfig['evolution'], 'ablations'>> & {
      ablations?: Partial<WorkspaceRuntimeConfig['evolution']['ablations']>;
    }
  ): Promise<WorkspaceRuntimeConfig['evolution']> {
    this.validateEvolutionConfigPatch(patch);
    const next = await this.getContext().memory.updateEvolutionConfig(patch);
    if (!this.workspaceRuntimeConfig) throw new Error('Runtime workspace config is not initialized');
    this.workspaceRuntimeConfig.evolution = next;
    this.emit({ type: 'evo.config.updated', agentId: 'root', data: next as unknown as Record<string, unknown> });
    return structuredClone(next);
  }

  async runEvolution(input: RunEvolutionInput): Promise<EvolutionRunResult> {
    const ctx = this.getContext();
    if (!input.task?.trim()) throw new Error('Evolution task is required');
    if (this.workspaceRuntimeConfig?.evolution.enabled === false) throw new Error('Evolution is disabled by workspace policy');
    const parentId = input.parentId ?? 'root';
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) throw new Error(`Evolution parent agent "${parentId}" not found`);
    const correlationId = input.correlationId ?? this.createCorrelationId();
    const runId = `evo_run_${Date.now()}_${(++this.evolutionSequence).toString(36)}`;
    const options = this.resolveEvolutionRunOptions(input.profile, input.options);
    if (options.profile === 'solo' || options.ablations.withoutSubagents) {
      return this.runSoloEvolutionBaseline(runId, correlationId, input.task, options);
    }

    const startedAt = Date.now();
    if (options.ablations.withoutBudgetMarket) this.evolutionBudgetBypassCorrelations.add(correlationId);
    const initialBudgetMarket = this.getBudgetMarketState();
    const patterns = options.ablations.withoutPatternMemory
      ? []
      : await this.findRelevantEvolutionPatterns(input.task, options.patternSimilarityThreshold);
    for (const pattern of patterns) {
      this.emit({
        type: 'cache.hit',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { cacheType: 'evolution-pattern', patternId: pattern.id, similarityThreshold: options.patternSimilarityThreshold },
      });
    }
    const seedAgents = input.seedAgents?.length
      ? input.seedAgents
      : await this.createEvolutionSeeds(input.task, parentId, correlationId, options);
    const policy = this.getAgentPolicy(parentId);
    const availableAgentSlots = Math.max(0, Math.min(
      policy?.allowedChildren ?? 0,
      this.getRemainingTotalAgentsForTurn(parentId, correlationId)
    ));
    if (availableAgentSlots <= 0) throw new Error('Evolution cannot execute because no agent slots remain for the parent or turn');
    const proposalInput = {
      runId,
      task: input.task,
      parentId,
      agents: seedAgents,
      patterns,
      availableTokens: options.ablations.withoutBudgetMarket ? undefined : this.getBudgetState().remainingTokens,
      availableAgentSlots,
      options,
    };

    const instantiated = new Map<string, { kind: 'agent' | 'team'; actorId?: string; error?: string }>();
    const details = new Map<string, { agent?: RunAgentResult; team?: TeamRunResult }>();
    let evaluationUsage = this.sumUsage([]);
    const judge = options.useLlmJudge && ctx.llm
      ? this.createEvolutionJudge(parentId, correlationId, (usage: TokenUsage) => {
        evaluationUsage = this.sumUsage([evaluationUsage, usage]);
      })
      : undefined;
    const engine = new EvolutionLifecycleEngine(
      new TeamFirstGenomePlanner(),
      new CompositeEvolutionEvaluator(judge),
      new WeightedTopKSelectionPolicy(),
      defaultMutationOperators(),
      {
        onTransition: async (from, to, data) => {
          this.emit({ type: 'evo.fsm.transition', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, from, to, ...data } });
          await this.recordEvolutionLifecycleMessage(to, parentId, correlationId, { runId, from, to, ...data });
        },
        instantiate: async candidate => {
          try {
            const actor = await this.instantiateEvolutionCandidate(candidate, parentId, correlationId, options);
            instantiated.set(candidate.id, actor);
            this.emit({
              type: 'evo.candidate.spawned', agentId: actor.actorId ?? parentId, sessionId: ctx.sessionId, correlationId,
              data: {
                runId, candidateId: candidate.id, genomeId: candidate.genome.id,
                actorKind: actor.kind, actorId: actor.actorId, memberCount: candidate.genome.members.length,
                degeneratedToAgent: candidate.genome.members.length === 1, source: candidate.source,
              },
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            instantiated.set(candidate.id, { kind: candidate.genome.members.length === 1 ? 'agent' : 'team', error: message });
            this.emit({ type: 'evo.candidate.instantiate.failed', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, candidateId: candidate.id, error: message } });
          }
        },
        execute: async candidate => {
          const actor = instantiated.get(candidate.id);
          if (!actor || actor.error || !actor.actorId) {
            return this.failedEvolutionArtifact(candidate, actor?.actorId ?? 'not-created', actor?.error ?? 'candidate_not_instantiated');
          }
          const artifact = await this.executeEvolutionCandidate(candidate, actor, correlationId, details);
          this.emit({
            type: artifact.success ? 'evo.candidate.executed' : 'evo.candidate.failed',
            agentId: artifact.actorId,
            sessionId: ctx.sessionId,
            correlationId,
            data: {
              runId, candidateId: candidate.id, actorKind: artifact.actorKind,
              totalTokens: artifact.usage.totalTokens, wallClockMs: artifact.wallClockMs,
              toolSuccessRate: artifact.toolCalls > 0 ? artifact.successfulToolCalls / artifact.toolCalls : null,
              failedActors: artifact.failedActors,
            },
          });
          this.archiveEvolutionCandidateAgents(artifact.agentIds, runId, candidate.id);
          return artifact;
        },
        integrate: async (selected, evaluation, execution) => {
          if (options.ablations.withoutPatternMemory) return undefined;
          return this.integrateEvolutionPattern(selected, evaluation, execution);
        },
      }
    );

    try {
      const lifecycle = await engine.run(proposalInput);
      for (const evaluation of lifecycle.evaluations) {
        this.emit({
          type: 'evo.candidate.evaluated', agentId: parentId, sessionId: ctx.sessionId, correlationId,
          data: { runId, candidateId: evaluation.candidateId, score: evaluation.score, dimensions: evaluation.dimensions, evaluator: evaluation.evaluator },
        });
      }
      if (lifecycle.selected) {
        this.emit({
          type: 'evo.candidate.selected', agentId: parentId, sessionId: ctx.sessionId, correlationId,
          data: {
            runId, candidateId: lifecycle.selected.id, genomeId: lifecycle.selected.genome.id,
            score: lifecycle.selectedEvaluation?.score, source: lifecycle.selected.source,
            lineage: lifecycle.selected.lineage,
          },
        });
      }
      const completedAt = Date.now();
      const executionsUsage = this.sumUsage(lifecycle.executions.map(execution => this.evolutionUsageToTokenUsage(execution.usage)));
      const totalUsage = this.sumUsage([executionsUsage, evaluationUsage]);
      const finalBudgetMarket = this.getBudgetMarketState();
      const runAllocations = finalBudgetMarket.allocations.filter(allocation => allocation.request.correlationId === correlationId);
      const metrics = this.buildEvolutionMetrics({
        lifecycle,
        usage: totalUsage,
        startedAt,
        completedAt,
        cacheHits: patterns.length,
        budgetRequested: runAllocations.reduce((sum, allocation) => sum + allocation.request.requestedTokens, 0),
        budgetAllocated: runAllocations.reduce((sum, allocation) => sum + allocation.allocatedTokens, 0),
      });
      const run: EvolutionRunResult = {
        id: runId,
        correlationId,
        task: input.task,
        profile: options.profile,
        state: lifecycle.state,
        candidates: lifecycle.candidates,
        executions: lifecycle.executions,
        evaluations: lifecycle.evaluations,
        selected: lifecycle.selected,
        selectedExecution: lifecycle.selectedExecution,
        selectedEvaluation: lifecycle.selectedEvaluation,
        integratedPatternId: lifecycle.integratedPatternId,
        metrics,
        ablations: options.ablations,
        startedAt,
        completedAt,
      };
      this.evolutionRuns.push(run);
      if (this.evolutionRuns.length > 100) this.evolutionRuns.splice(0, this.evolutionRuns.length - 100);
      await ctx.memory.recordEvolutionRun({ ...run, initialBudgetMarket, finalBudgetMarket });
      this.emit({
        type: 'evo.run.completed', agentId: parentId, sessionId: ctx.sessionId, correlationId,
        data: { runId, profile: options.profile, selected: run.selected?.id, score: run.selectedEvaluation?.score, metrics },
      });
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
      return structuredClone(run);
    } catch (error) {
      const completedAt = Date.now();
      const failed: EvolutionRunResult = {
        id: runId, correlationId, task: input.task, profile: options.profile, state: 'S_evo_failed',
        candidates: [], executions: [], evaluations: [], selected: undefined,
        selectedExecution: undefined, selectedEvaluation: undefined,
        metrics: {
          taskSuccess: false, answerQuality: 0, toolSuccessRate: 0, agentsSpawned: 0, teamsSpawned: 0,
          totalTokens: evaluationUsage.totalTokens, thinkingTokens: evaluationUsage.thinkingTokens,
          wallClockMs: completedAt - startedAt, budgetRequested: 0, budgetAllocated: 0,
          failureRecoveryCount: 0, candidateCount: 0, executedCandidateCount: 0,
          cacheHits: patterns.length, mutationsApplied: 0,
        },
        ablations: options.ablations, startedAt, completedAt,
        error: error instanceof Error ? error.message : String(error),
      };
      this.evolutionRuns.push(failed);
      await ctx.memory.recordEvolutionRun(failed as unknown as Record<string, unknown>);
      this.emit({ type: 'evo.run.failed', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, error: failed.error } });
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
      throw error;
    }
  }

  async runEvolutionBenchmark(
    task: string,
    profiles: EvolutionProfile[] = ['solo', 'fixed_subagents', 'tom_subteam', 'budget_market', 'evo_team']
  ): Promise<EvolutionBenchmarkResult> {
    const runs: EvolutionRunResult[] = [];
    for (const profile of profiles) {
      runs.push(await this.runEvolution({ task, profile }));
    }
    return {
      task,
      profiles,
      runs,
      comparison: runs.map(run => ({
        profile: run.profile,
        success: run.metrics.taskSuccess,
        score: run.selectedEvaluation?.score ?? run.metrics.answerQuality,
        totalTokens: run.metrics.totalTokens,
        thinkingTokens: run.metrics.thinkingTokens,
        wallClockMs: run.metrics.wallClockMs,
        agentsSpawned: run.metrics.agentsSpawned,
        teamsSpawned: run.metrics.teamsSpawned,
      })),
    };
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
    let evolution: EvolutionRunResult | undefined;

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
            cognitiveGapIds: plan.cognitiveGapIds,
            existenceReason: plan.existenceReason,
            tomProfile: plan.tomProfile,
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
            cognitiveGapIds: plan.cognitiveGapIds,
            existenceReason: plan.existenceReason,
          },
        });
      }
      const plans = decision.agents.slice(0, 3);
      if (this.workspaceRuntimeConfig?.evolution.enabled
        && this.workspaceRuntimeConfig.evolution.mode === 'auto') {
        evolution = await this.runEvolution({
          task: userInput,
          parentId: 'root',
          correlationId,
          profile: this.workspaceRuntimeConfig.evolution.profile,
          seedAgents: plans.map(plan => ({
            archetype: plan.archetype,
            name: plan.name,
            role: plan.existenceReason ?? plan.archetype,
            task: plan.task,
            tools: plan.tools,
            skills: plan.skills,
            budgetTokens: plan.budgetTokens,
            tomLevel: plan.tomLevel,
            perspective: plan.tomProfile?.perspective,
            groundingRequired: plan.archetype === 'researcher' || plan.archetype === 'tester',
          })),
        });
      } else if (plans.length > 1 && this.workspaceRuntimeConfig?.teams.createForMultipleAgents !== false) {
        const team = await this.spawnTeam({
          parentAgentId: 'root',
          name: this.deriveTeamName(plans),
          description: userInput,
          task: userInput,
          members: plans.map((plan, index) => ({ ...plan, lead: index === 0 })),
          tomAnalysis: this.tomAnalyses.get(correlationId),
          correlationId,
        });
        const teamResult = await this.runTeam(team.identity.id, userInput, { correlationId });
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
            tomProfile: plan.tomProfile,
            cognitiveGapIds: plan.cognitiveGapIds,
            existenceReason: plan.existenceReason,
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
        completed: evolution?.metrics.agentsSpawned ?? subagents.length,
      });
      await this.transitionRootTurnState('S_synthesize', {
        correlationId,
        completed: evolution?.metrics.agentsSpawned ?? subagents.length,
      });
      finalResponse = evolution
        ? await this.synthesizeEvolutionResult(userInput, evolution, correlationId)
        : await this.synthesizeDelegatedResults(userInput, subagents, correlationId, teamResults);
      this.emit({
        type: 'delegation.completed',
        agentId: 'root',
        data: {
          correlationId,
          subagentIds: subagents.map(result => result.agent.identity.id),
          totalSubagents: evolution?.metrics.agentsSpawned ?? subagents.length,
          evolutionRunId: evolution?.id,
          selectedGenomeId: evolution?.selected?.genome.id,
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
        kind: subagents.length > 0 || evolution ? 'root.delegated_final_response' : 'root.chat_response',
        decision: decision.action,
        subagentIds: subagents.map(result => result.agent.identity.id),
        grounded: evolution
          ? Boolean(evolution.selectedExecution && evolution.selectedExecution.groundedResults > 0)
          : subagents.length === 0 ? undefined : subagents.every(result => result.subagentResult.grounded),
        evolutionRunId: evolution?.id,
      },
    });
    await this.processQueuedMessage(finalMessage.id);
    await ctx.queue.ack(finalMessage.id);
    await this.transitionRootTurnState('S_turn_done', { correlationId });
    await this.transitionRootTurnState('S_solo', { correlationId });
    await this.proposeMemoryUpdates('turn.completed');

    const rootUsageAfter = ctx.agent.getUsage();
    const rootUsage = this.usageDifference(rootUsageBefore, rootUsageAfter);
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
      evolution,
      messages: await this.getMessages({ correlationId }),
      usage: {
        root: rootUsage,
        subagents: subagentUsage,
        teamSynthesis: teamSynthesisUsage,
        total: this.sumUsage([
          rootUsage,
          ...Object.values(subagentUsage),
          ...Object.values(teamSynthesisUsage),
          ...(evolution ? [this.sumUsage(evolution.executions.map(item => this.evolutionUsageToTokenUsage(item.usage)))] : []),
        ]),
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
      tomProfileMode: payload.source && !['cli', 'server'].includes(payload.source)
        ? 'runtime_assignment'
        : 'definition_override',
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
      communicationProtocol: payload.communicationProtocol,
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
      const rootUsage = this.usageDifference(rootUsageBefore, rootUsageAfter);
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
    const baseTools = cachedTools.length > 0 ? cachedTools : this.getDefaultToolBindings(request.archetype).map(item => item.name);
    const baseSkills = cachedSkills.length > 0 ? cachedSkills : this.getDefaultSkillBindings(request.archetype).map(item => item.name);
    const tools = request.tools ?? baseTools;
    const skills = request.skills ?? baseSkills;
    const toolsOverrideDefinition = request.tools !== undefined && !this.sameStringSet(request.tools, baseTools);
    const skillsOverrideDefinition = request.skills !== undefined && !this.sameStringSet(request.skills, baseSkills);
    this.validateDelegatedCapabilities(parentId, tools, skills);

    const cachedMemoryScope = this.agentMemoryScope(cachedAgentPattern?.memoryScope);
    const cachedSpawnPolicy = this.partialSpawnPolicy(cachedAgentPattern?.spawnPolicy);
    const memoryScope = this.constrainMemoryScope(
      request.memoryScope ?? cachedMemoryScope ?? this.getDefaultMemoryScope('subagent')
    );
    const cachedCommunicationProtocol = typeof cachedAgentPattern?.communicationProtocol === 'string'
      ? cachedAgentPattern.communicationProtocol
      : undefined;
    const communicationProtocol = request.communicationProtocol
      ?? cachedCommunicationProtocol
      ?? ctx.communication.getDefaultProtocolId();
    if (!ctx.communication.registry.get(communicationProtocol)) {
      throw new Error(`Communication protocol "${communicationProtocol}" is not registered`);
    }
    const requestedSpawnPolicy = this.mergeSpawnPolicy(
      this.getDefaultSpawnPolicy('subagent', request.archetype),
      { ...cachedSpawnPolicy, ...request.spawnPolicy }
    );
    const spawnPolicy = this.constrainChildSpawnPolicy(parentId, requestedSpawnPolicy, skills);
    const agentPatternId = typeof cachedAgentPattern?.id === 'string' ? cachedAgentPattern.id : undefined;
    const delegationPatternId = typeof cachedDelegationPattern?.id === 'string' ? cachedDelegationPattern.id : undefined;
    const cacheHits = [agentPatternId, delegationPatternId].filter((item): item is string => Boolean(item));
    const hasDefinitionOverrides = (request.archetype === 'custom' && request.name !== undefined)
      || request.role !== undefined
      || request.style !== undefined
      || request.description !== undefined
      || request.systemPrompt !== undefined
      || toolsOverrideDefinition
      || skillsOverrideDefinition
      || request.memoryScope !== undefined
      || request.spawnPolicy !== undefined
      || (request.tomProfile !== undefined && request.tomProfileMode !== 'runtime_assignment')
      || (request.communicationProtocol !== undefined && request.communicationProtocol !== cachedCommunicationProtocol)
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
      name: request.archetype === 'custom' ? request.name : undefined,
      role: request.role ?? request.archetype,
      style: request.style,
      description,
      systemPrompt: request.systemPrompt,
      tools,
      skills,
      memoryScope,
      spawnPolicy,
      tomProfile: request.tomProfileMode === 'definition_override' ? request.tomProfile : undefined,
      communicationProtocol,
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
        systemPrompt: request.systemPrompt,
        tomProfile: request.tomProfile,
        tomProfileMode: request.tomProfileMode ?? 'definition_override',
        cognitiveGapIds: [...(request.cognitiveGapIds ?? [])],
        existenceReason: request.existenceReason,
      },
      assignment: { task: request.task, outputContract },
      capabilities: { tools: [...tools], skills: [...skills] },
      context: { memoryScope, communicationProtocol },
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
      cognitiveGapIds: node.identity.cognitiveGapIds,
      existenceReason: node.identity.existenceReason,
      systemPrompt: node.identity.systemPrompt,
      communicationProtocol: node.context.communicationProtocol,
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
      ? normalizeToMProfile(payload.tomProfile, payload.tomProfile)
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
      systemPrompt: payload.systemPrompt,
      outputContract: node.assignment.outputContract,
      correlationId,
      tomProfile,
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
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
      tomProfile: agent.identity.tomProfile,
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
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
    if (budgetAllocation?.status === 'granted') {
      this.budgetMarket?.assignRequester(budgetAllocation.id, id, 'agent');
    }
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
        ? normalizeToMProfile({ ...spec.tomProfile, subjectAgentId: id }, {
          level: spec.tomProfile.level,
          subjectAgentId: id,
          purpose: spec.tomProfile.purpose,
        })
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
    let goal = this.buildAgentPromptFromMemory({
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
        communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
        availableSkills: skillBindings.map((binding) => binding.name),
        availableTools: toolBindings.map((binding) => binding.name),
        parentContext: contextWindow.parentContext,
      });
      const originalRenderedPromptTokens = this.estimateTextTokens(goal);
      if (budgetAllocation?.status === 'granted' && this.budgetAccountingDimension() === 'total_tokens') {
        const outputReserve = Math.min(512, Math.max(128, Math.floor(budgetAllocation.allocatedTokens * 0.2)));
        const executionOverhead = Math.min(512, Math.max(128, Math.floor(budgetAllocation.allocatedTokens * 0.15)));
        const promptBudget = Math.max(64, budgetAllocation.allocatedTokens - outputReserve - executionOverhead);
        if (originalRenderedPromptTokens > promptBudget) {
          goal = this.truncateTextToTokenBudget(goal, promptBudget);
          this.emit({
            type: 'budget.context.truncated',
            agentId: id,
            sessionId: ctx.sessionId,
            correlationId: creationCorrelationId,
            nodeId: spec.nodeDefinition?.nodeId,
            data: {
              purpose: 'agent.system_prompt',
              contextType: 'rendered_agent_prompt',
              originalTokens: originalRenderedPromptTokens,
              allowedTokens: promptBudget,
              outputReserve,
              executionOverhead,
            },
          });
        }
      }
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
      tomLevel: resolvedTomProfile.level,
      tomProfile: resolvedTomProfile,
      communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
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
        tomLevel: resolvedTomProfile.level,
        tomProfile: resolvedTomProfile,
        cognitiveGapIds: spec.cognitiveGapIds ?? resolvedTomProfile.cognitiveGaps,
        existenceReason: spec.existenceReason,
        description: spec.description,
        tools: toolBindings.map((binding) => binding.name),
        skills: skillBindings.map((binding) => binding.name),
        spawnPolicy,
        memoryScope,
        outputContract: spec.outputContract,
        communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
      definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
      creationMode,
    });

    const info = agent.getInfo();
    this.recordTurnAgentCreated(creationCorrelationId);
    this.emit({
      type: 'tom.profile.assigned',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        level: resolvedTomProfile.level,
        perspective: resolvedTomProfile.perspective,
        beliefScope: resolvedTomProfile.beliefScope,
        goalModel: resolvedTomProfile.goalModel,
        uncertainty: resolvedTomProfile.uncertainty,
        observesAgents: resolvedTomProfile.observesAgents,
        modelsAgents: resolvedTomProfile.modelsAgents,
        cognitiveGapIds: spec.cognitiveGapIds ?? resolvedTomProfile.cognitiveGaps,
        existenceReason: spec.existenceReason,
      },
    });
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
        tomLevel: resolvedTomProfile.level,
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

    const activeAllocationId = this.agentBudgetAllocations.get(agentId);
    const activeAllocation = activeAllocationId ? this.budgetMarket?.getAllocation(activeAllocationId) : undefined;
    agent.setCompletionTokenLimit(activeAllocation?.allocatedTokens, this.budgetAccountingDimension());

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
      const communicationContext = agent.getCommunicationContext();
      const rawObservation = [
        this.buildGroundedTask(task, grounding),
        communicationContext
          ? `<system_communication_context protocol="${communicationContext.protocolId}">\n${communicationContext.rendered}\n</system_communication_context>`
          : '',
      ].filter(Boolean).join('\n\n');
      const observation = this.constrainAgentObservation(
        agent,
        rawObservation,
        options.correlationId,
        'agent.task_execution'
      );
      await agent.step(observation);
      const stepError = agent.getInfo().error;
      if (stepError) {
        throw new Error(stepError.replace(/^Error:\s*/, ''));
      }
      let result = session ? await this.drainAgentOutput(session.messageQueue, agent.name) : agent.getInfo().lastResult ?? '';
      if (this.containsUnresolvedToolIntent(result)) {
        if (!grounding.evidence.toolGrounded) {
          throw new Error('Agent returned an unexecuted tool request without runtime grounding evidence');
        }
        this.emit({
          type: 'agent.output.repair.started',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          data: { reason: 'unexecuted_tool_intent', task },
        });
        result = await this.completeAsAgent(
          agent,
          [
            `Task:\n${task}`,
            `Runtime-provided evidence:\n${grounding.evidence.toolResultSummary ?? grounding.context}`,
            'Produce the final task result from the evidence above.',
            'Do not emit tool-call markup, JSON tool requests, or claim that a tool still needs to run.',
          ].join('\n\n'),
          'agent.output_repair',
          options.correlationId ?? this.createCorrelationId()
        );
        if (this.containsUnresolvedToolIntent(result)) {
          throw new Error('Agent output repair still contained an unexecuted tool request');
        }
        this.emit({
          type: 'agent.output.repair.completed',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          data: { task },
        });
      }
      agent.setRuntimeState('done');
      await this.transitionAgentFsm(agentId, 'S_responding', { correlationId: options.correlationId });

      const usageAfter = agent.getUsage();
      const usageDelta = this.usageDifference(usageBefore, usageAfter);
      this.recordTurnUsage(usageDelta);
      this.emit({ type: 'budget.updated', agentId, data: { ...usageDelta } });
      this.emit({ type: 'agent.status.changed', agentId, data: { from: 'thinking', to: 'done' } });

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
      this.settleAgentBudget(agentId, usageDelta, {
        success: true,
        evidenceGain: evidence.outputGrounded ? 0.95 : evidence.toolGrounded ? 0.5 : 0.15,
        uncertaintyReduction: grounding.grounded && evidence.outputGrounded ? 0.8 : 0.35,
        verificationGain: options.archetype === 'tester' ? (evidence.outputGrounded ? 0.9 : 0.45) : undefined,
      });
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
      inputTokens: usage.inputTokens ?? usage.promptTokens,
      outputTokens: usage.outputTokens ?? usage.completionTokens,
      thinkingTokens: usage.thinkingTokens ?? null,
      cachedInputTokens: usage.cachedInputTokens ?? null,
      cacheCreationInputTokens: usage.cacheCreationInputTokens ?? null,
    };
  }

  private usageDifference(before: AgentUsage, after: AgentUsage): TokenUsage {
    const nullableDelta = (previous: number | null | undefined, current: number | null | undefined): number | null => {
      if (previous === null && current === null) return null;
      if (previous === undefined && current === undefined) return null;
      return Math.max(0, (current ?? 0) - (previous ?? 0));
    };
    return this.toTokenUsage({
      llmCalls: after.llmCalls - before.llmCalls,
      promptTokens: after.promptTokens - before.promptTokens,
      completionTokens: after.completionTokens - before.completionTokens,
      totalTokens: after.totalTokens - before.totalTokens,
      inputTokens: (after.inputTokens ?? after.promptTokens) - (before.inputTokens ?? before.promptTokens),
      outputTokens: (after.outputTokens ?? after.completionTokens) - (before.outputTokens ?? before.completionTokens),
      thinkingTokens: nullableDelta(before.thinkingTokens, after.thinkingTokens),
      thinkingAccountingTokens: Math.max(
        0,
        (after.thinkingAccountingTokens ?? after.thinkingTokens ?? after.totalTokens)
          - (before.thinkingAccountingTokens ?? before.thinkingTokens ?? before.totalTokens)
      ),
      cachedInputTokens: nullableDelta(before.cachedInputTokens, after.cachedInputTokens),
      cacheCreationInputTokens: nullableDelta(before.cacheCreationInputTokens, after.cacheCreationInputTokens),
    });
  }

  private async processQueuedMessage(messageId: string): Promise<RuntimeMessage | undefined> {
    const ctx = this.getContext();
    const message = await ctx.queue.getMessage(messageId);
    if (!message) return undefined;
    const dequeued = await ctx.queue.dequeue({ to: message.to, kind: [message.kind], readyOnly: true });
    if (dequeued) this.deliverCommunicationContext(dequeued);
    return dequeued;
  }

  private deliverCommunicationContext(message: RuntimeMessage): void {
    const ctx = this.getContext();
    const recipient = ctx.manager.getAgentById(message.to);
    if (!recipient) return;
    const participantIds = new Set<string>([message.from, message.to]);
    for (const trace of ctx.communication.traces.list({
      sessionId: message.sessionId,
      correlationId: message.correlationId,
    })) {
      participantIds.add(trace.from.id);
      for (const actor of trace.to) participantIds.add(actor.id);
    }
    const participants = ctx.manager.listAgentInfo().filter(agent => participantIds.has(agent.identity.id));
    const task = this.extractMessageTask(message);
    const communicationContext = ctx.communication.buildContext({
      message,
      recipient: recipient.getInfo(),
      participants,
      task,
    });
    recipient.receiveCommunicationContext(communicationContext);
    this.emit({
      type: 'communication.context.delivered',
      agentId: recipient.id,
      sessionId: message.sessionId,
      correlationId: message.correlationId,
      nodeId: message.metadata?.nodeId,
      data: {
        messageId: message.id,
        protocolId: communicationContext.protocolId,
        traceCount: communicationContext.traces.length,
        participantIds: participants.map(agent => agent.identity.id),
      },
    });
  }

  private extractMessageTask(message: RuntimeMessage): string | undefined {
    if (!message.payload || typeof message.payload !== 'object') return undefined;
    const payload = message.payload as Record<string, unknown>;
    const value = payload.task ?? payload.input ?? payload.userTask;
    return typeof value === 'string' ? value : undefined;
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

  private requireCommunicationManager(): AgentCommunicationManager {
    if (!this.communicationManager) throw new Error('AgentCommunicationManager is not initialized');
    return this.communicationManager;
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
    const parentToMProfile = ctx.manager.getAgentById(parentId)?.getIdentity().tomProfile;
    const tomSignals = this.deriveToMAnalysisSignals(parentId, correlationId, parentToMProfile);
    this.emit({
      type: 'tom.signals.collected',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { ...tomSignals },
    });
    const tomAnalysis = this.tomPlanner.analyzeTask({
      task,
      parentId,
      parentProfile: parentToMProfile,
      signals: tomSignals,
    });
    const remainingAgentSlots = Math.min(
      policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      this.workspaceRuntimeConfig?.tom.maxAgentsPerDecision ?? 3
    );
    const tomEnabled = this.workspaceRuntimeConfig?.tom.enabled !== false;
    const completedPlans = tomEnabled
      ? this.tomPlanner.completePlans(
        tomAnalysis,
        decision.agents,
        this.workspaceRuntimeConfig?.tom.autoCompleteGaps === false
          ? Math.min(decision.agents.length, remainingAgentSlots)
          : remainingAgentSlots
      )
      : decision.agents.slice(0, remainingAgentSlots);
    if (completedPlans.length === 0) {
      return {
        action: 'solve_directly',
        reason: `${decision.reason} Delegation skipped because no policy-approved agent slots remain.`,
      };
    }
    const enrichedDecision: DelegationDecision = {
      ...decision,
      agents: completedPlans,
    };
    this.tomAnalyses.set(correlationId, tomAnalysis);
    this.emit({
      type: 'tom.task.analyzed',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        analysisId: tomAnalysis.id,
        rationale: tomAnalysis.rationale,
        requiresHigherOrderToM: tomAnalysis.requiresHigherOrderToM,
        gapCount: tomAnalysis.gaps.length,
        parentBeliefs: tomAnalysis.parentBeliefs,
        parentGoals: tomAnalysis.parentGoals,
        parentUncertainties: tomAnalysis.parentUncertainties,
        source: tomAnalysis.source,
        confidence: tomAnalysis.confidence,
        signals: tomAnalysis.signals,
      },
    });
    for (const gap of tomAnalysis.gaps) {
      this.emit({
        type: 'tom.gap.identified',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { ...gap, analysisId: tomAnalysis.id },
      });
    }
    if (tomAnalysis.requiresHigherOrderToM) {
      this.emit({
        type: 'tom.higher_order.required',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { analysisId: tomAnalysis.id, reason: tomAnalysis.rationale },
      });
    }

    const cacheHits = await Promise.all(completedPlans.map(async agent => {
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
      correlationId,
      task,
      decision: enrichedDecision,
      allowedChildren: policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      budgetMode: budget.mode,
      remainingBudgetTokens: budget.remainingTokens,
      cacheUsed: cacheHits.some(Boolean),
      cachedPatterns: [...agentPatterns, ...delegationPatterns],
      parentToMProfile,
      tomAnalysis,
    });

    await this.recordEvolutionLifecycle(parentId, correlationId, scope, selection);
    this.emitDelegationCandidateEvents(parentId, correlationId, scope, selection);
    if (selection.selected?.tomCoverage) {
      this.emit({
        type: 'tom.delegation.coverage.evaluated',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          analysisId: tomAnalysis.id,
          candidateId: selection.selected.id,
          ...selection.selected.tomCoverage,
        },
      });
    }
    return selection.decision;
  }

  private deriveToMAnalysisSignals(
    parentId: string,
    correlationId: string,
    parentProfile?: ToMProfile
  ): ToMAnalysisSignals {
    const ctx = this.getContext();
    const current = this.communicationManager?.traces.list({
      sessionId: ctx.sessionId,
      correlationId,
      limit: this.workspaceRuntimeConfig?.communication.traceWindowSize ?? 200,
    }) ?? [];
    const traces = current;
    const participants = new Set(traces.flatMap(trace => [trace.from.id, ...trace.to.map(actor => actor.id)]));
    const failed = traces.filter(trace => trace.phase === 'failed');
    const cancelled = traces.filter(trace => trace.phase === 'cancelled');
    const toolResults = traces.filter(trace => trace.kind === 'tool.result' && trace.phase === 'completed');
    const evidence = traces.filter(trace =>
      trace.phase === 'completed'
      && ['tool.result', 'agent.result', 'team.result'].includes(trace.kind)
      && Boolean(trace.content?.trim())
    );
    const conflicting = traces.filter(trace =>
      /\b(conflict|contradict|disagree|inconsistent|unsupported|mismatch)\b|冲突|矛盾|不一致/.test(trace.content?.toLowerCase() ?? '')
    );
    const reliabilityConcerns = [
      ...(failed.length > 0 ? [`${failed.length} observable message(s) failed`] : []),
      ...(cancelled.length > 0 ? [`${cancelled.length} observable message(s) were cancelled`] : []),
    ];
    const evidenceOpportunity = traces.filter(trace =>
      ['agent.task', 'team.task', 'tool.call', 'tool.result', 'agent.result', 'team.result'].includes(trace.kind)
    ).length;
    const evidenceCoverage = evidenceOpportunity === 0
      ? 1
      : Math.min(1, (evidence.length + toolResults.length * 0.5) / evidenceOpportunity);
    const conflictLevel = participants.size === 0 ? 0 : Math.min(1, conflicting.length / participants.size);
    const profileUncertainty = parentProfile?.uncertainty.length ?? 0;
    const uncertaintyLevel = Math.min(1, profileUncertainty * 0.2 + failed.length * 0.15 + (1 - evidenceCoverage) * 0.35);
    return {
      traceCount: traces.length,
      participantCount: participants.size,
      failedTraceCount: failed.length,
      cancelledTraceCount: cancelled.length,
      toolResultCount: toolResults.length,
      evidenceTraceCount: evidence.length,
      conflictingTraceCount: conflicting.length,
      evidenceCoverage: Number(evidenceCoverage.toFixed(4)),
      conflictLevel: Number(conflictLevel.toFixed(4)),
      uncertaintyLevel: Number(uncertaintyLevel.toFixed(4)),
      observedKinds: [...new Set(traces.map(trace => trace.kind))],
      reliabilityConcerns,
    };
  }

  private async beforeDelegationScorerCall(
    input: DelegationCandidateInput,
    messages: LLMMessage[],
    options: LLMCompletionOptions
  ): Promise<LLMDelegationScorerInvocation> {
    const parent = this.getContext().manager.getAgentById(input.parentId);
    if (!parent) return { skip: true };

    const estimatedInputTokens = this.estimateTextTokens(
      messages.map(message => `${message.role}:${message.content}`).join('\n')
    );
    const activeAllocationId = this.agentBudgetAllocations.get(parent.id);
    const activeAllocation = activeAllocationId
      ? this.budgetMarket?.getAllocation(activeAllocationId)
      : undefined;
    const ownsAllocation = !activeAllocation || activeAllocation.status !== 'granted';
    const allocation = ownsAllocation
      ? await this.requestAgentBudget({
        parentId: parent.getIdentity().parentId ?? parent.id,
        requesterId: parent.id,
        archetype: parent.id === 'root' ? 'custom' : this.inferAgentArchetype(parent.getInfo()),
        correlationId: input.correlationId,
        requestedTokens: this.budgetRequestTokens(estimatedInputTokens, options.maxTokens ?? 700),
        minimumTokens: this.budgetMinimumTokens(estimatedInputTokens),
        priority: 'high',
        expectedUtility: 0.84,
        purpose: 'delegation.candidate_scoring',
      })
      : activeAllocation;

    if (allocation?.status === 'denied') {
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'budget_request_denied', allocationId: allocation.id },
      });
      return { skip: true };
    }

    const allocationRemaining = allocation?.status === 'granted'
      ? Math.max(0, allocation.allocatedTokens - allocation.consumedTokens)
      : undefined;
    const agentRemaining = parent.getCompletionTokenLimit();
    const availableTokens = [allocationRemaining, agentRemaining]
      .filter((value): value is number => value !== undefined)
      .reduce<number | undefined>((minimum, value) => minimum === undefined ? value : Math.min(minimum, value), undefined);
    if (availableTokens !== undefined && availableTokens <= 0) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'delegation_scorer_allocation_exhausted');
      }
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'allocation_exhausted', availableTokens },
      });
      return { skip: true };
    }
    if (this.budgetAccountingDimension() === 'total_tokens'
      && availableTokens !== undefined
      && availableTokens <= estimatedInputTokens) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'delegation_scorer_input_exceeds_allocation');
      }
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'insufficient_tokens_for_input', estimatedInputTokens, availableTokens },
      });
      return { skip: true };
    }

    const maxTokens = availableTokens === undefined
      ? options.maxTokens
      : Math.max(1, Math.min(
        options.maxTokens ?? 700,
        this.completionCapacity(availableTokens, estimatedInputTokens)
      ));
    return {
      options: { ...options, maxTokens },
      context: {
        parentId: parent.id,
        correlationId: input.correlationId,
        usageBefore: parent.getUsage(),
        allocation,
        ownsAllocation,
      } satisfies DelegationScorerBudgetContext,
    };
  }

  private afterDelegationScorerCall(
    completion: LLMCompletionResult,
    input: DelegationCandidateInput,
    hookContext?: unknown
  ): void {
    const context = hookContext as DelegationScorerBudgetContext | undefined;
    if (!context) return;
    const parent = this.getContext().manager.getAgentById(context.parentId);
    if (!parent) {
      this.releaseDelegationScorerBudget(context);
      return;
    }

    parent.recordRuntimeUsage(completion);
    const usage = this.usageDifference(context.usageBefore, parent.getUsage());
    if (context.ownsAllocation) {
      this.settleDirectBudget(parent.id, context.allocation, usage, context.correlationId);
    } else {
      this.consumeActiveAgentBudget(parent.id, usage, context.correlationId, 'delegation.candidate_scoring');
    }
    if (parent.id === 'root') this.recordTurnUsage(usage);
    this.emit({
      type: 'agent.llm.called',
      agentId: parent.id,
      correlationId: context.correlationId,
      data: {
        purpose: 'delegation.candidate_scoring',
        provider: this.getContext().llm?.name,
        model: completion.model ?? this.getContext().llm?.defaultModel,
        source: completion.usage?.source ?? 'estimated',
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        totalTokens: usage.totalTokens,
      },
    });
    this.emit({
      type: 'budget.updated',
      agentId: parent.id,
      correlationId: input.correlationId,
      data: { purpose: 'delegation.candidate_scoring', ...usage },
    });
  }

  private releaseDelegationScorerBudget(hookContext?: unknown): void {
    const context = hookContext as DelegationScorerBudgetContext | undefined;
    if (!context?.ownsAllocation || context.allocation?.status !== 'granted' || !this.budgetMarket) return;
    const released = this.budgetMarket.release(context.allocation.id, 'delegation_candidate_scoring_failed');
    if (released) {
      this.emit({
        type: 'budget.released',
        agentId: context.parentId,
        correlationId: context.correlationId,
        data: { allocationId: context.allocation.id, reason: 'delegation_candidate_scoring_failed' },
      });
    }
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
          investment: candidate.investment,
          lineage: candidate.lineage,
          rationale: candidate.rationale,
          tomCoverage: candidate.tomCoverage,
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
          tomCoverage: selection.selected.tomCoverage,
          investment: selection.selected.investment,
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
      payload: {
        candidates: selection.candidates.map(candidate => ({
          id: candidate.id,
          score: candidate.score,
          scoreBreakdown: candidate.scoreBreakdown,
          investment: candidate.investment,
        })),
      },
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
        investment: candidate.investment,
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
      const decision = await this.completeJSONAsAgent<DelegationDecision>(ctx.agent, [
        {
          role: 'system',
          content: `You are Roy's root delegation controller.
Decide whether the user request should be solved directly by Roy, clarified, or delegated to 1-3 subagents.
Reason in terms of cognitive gaps: missing evidence, missing perspective, failure-mode uncertainty, implementation capability, verification, or belief reconciliation.
Use delegation only when the task benefits from grounded inspection, critique, planning, coding, testing, or summarization.
Do not spawn more than 3 subagents. Prefer 1-2 unless the task clearly needs more.
Ask for clarification when the user request is too ambiguous to assign a concrete task safely.
Return strict JSON matching one of:
{"action":"solve_directly","reason":"..."}
{"action":"ask_clarification","reason":"...","question":"..."}
{"action":"spawn_subagents","reason":"...","agents":[{"archetype":"researcher","name":"Researcher-1","task":"...","tomLevel":0,"existenceReason":"which cognitive gap this agent fills"}]}
Allowed archetypes: researcher, critic, planner, coder, summarizer, tester, custom.`,
        },
        {
          role: 'user',
          content: [
            `<user_task>${userInput}</user_task>`,
            `<memory_context>${this.formatPublicContext(rootContext).slice(0, 6000)}</memory_context>`,
            `<budget_state>${JSON.stringify(this.getBudgetState(), null, 2)}</budget_state>`,
            '<runtime_policy>Subagents must be runtime actors with identity, state, budget, messages, and events. Each agent must fill a distinct cognitive gap, receive a concrete non-overlapping task, and expose why it exists. The runtime will validate and enrich the ToM profiles. If budget is limited, reduce the number of subagents or solve directly and explain the constraint.</runtime_policy>',
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 900 }, 'root.delegation_decision', correlationId);
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
      const decision = await this.completeJSONAsAgent<DelegationDecision>(
        ctx.manager.getAgentById(agent.identity.id)!,
        [
        {
          role: 'system',
          content: `You are ${agent.identity.name}'s delegation controller.
Decide whether this non-root agent should solve directly or delegate to 1-3 direct child agents. Delegate only to close an explicit evidence, perspective, risk, planning, implementation, verification, or synthesis gap in the parent agent's current model.
Only delegate when a child with a different specialty materially improves the result.
Return strict JSON:
{"action":"solve_directly","reason":"..."}
{"action":"spawn_subagents","reason":"...","agents":[{"archetype":"critic","name":"Critic-1","task":"...","tomLevel":2,"existenceReason":"which cognitive gap this child fills"}]}
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
        ],
        { temperature: 0.1, maxTokens: 500 },
        'agent.delegation_decision',
        correlationId
      );
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
      cognitiveGapIds: Array.isArray(plan.cognitiveGapIds)
            ? plan.cognitiveGapIds.filter((item): item is string => typeof item === 'string')
            : undefined,
          existenceReason: typeof plan.existenceReason === 'string' ? plan.existenceReason : undefined,
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
          cognitiveGapIds: Array.isArray(plan.cognitiveGapIds)
            ? plan.cognitiveGapIds.filter((item): item is string => typeof item === 'string')
            : undefined,
          existenceReason: typeof plan.existenceReason === 'string' ? plan.existenceReason : undefined,
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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

    const communicationSummary = ctx.agent.getCommunicationContext()?.rendered;
    const systemContent = [
      'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      'You are not DeepSeek, Claude, OpenAI, Anthropic, or any model provider.',
      'The model provider is only your inference backend.',
      `Purpose: ${purpose}.`,
      `Correlation: ${correlationId}.`,
      communicationSummary && communicationSummary.length > 1200
        ? `${communicationSummary.slice(0, 1200)}\n...[communication context truncated]`
        : communicationSummary,
    ].join('\n');
    let effectivePrompt = prompt;
    const availableTokens = this.budgetMarket?.getState().availableTokens;
    if (availableTokens !== undefined && this.budgetAccountingDimension() === 'total_tokens') {
      const outputReserve = Math.min(512, Math.max(64, Math.floor(availableTokens * 0.2)));
      const systemEstimate = this.estimateTextTokens(systemContent);
      const promptBudget = Math.max(0, availableTokens - outputReserve - systemEstimate);
      const promptEstimate = this.estimateTextTokens(effectivePrompt);
      if (promptEstimate > promptBudget && promptBudget > 0) {
        const maxChars = Math.max(64, Math.floor(effectivePrompt.length * promptBudget / promptEstimate));
        effectivePrompt = `${effectivePrompt.slice(0, maxChars)}\n...[budget-constrained context truncation]`;
        this.emit({
          type: 'budget.context.truncated',
          agentId: 'root',
          correlationId,
          data: { purpose, originalTokens: promptEstimate, allowedTokens: promptBudget, outputReserve },
        });
      }
    }

    const messages: LLMMessage[] = [
      { role: 'system', content: systemContent },
      { role: 'user', content: effectivePrompt },
    ];
    const estimatedInputTokens = this.estimateTextTokens(messages.map(message => message.content).join('\n'));
    const allocation = await this.requestAgentBudget({
      parentId: 'root',
      requesterId: 'root',
      archetype: 'custom',
      correlationId,
      requestedTokens: Math.max(
        this.budgetRequestTokens(estimatedInputTokens, 512),
        this.workspaceRuntimeConfig?.budgetMarket.defaultRequestsByArchetype.root ?? 2400
      ),
      minimumTokens: this.budgetMinimumTokens(estimatedInputTokens),
      priority: purpose.includes('synthesis') ? 'high' : 'medium',
      expectedUtility: purpose.includes('synthesis') ? 0.9 : 0.78,
      purpose,
    });
    if (allocation?.status === 'denied') throw new Error(`Root completion rejected by budget market: ${allocation.reason}`);
    const usageBefore = ctx.agent.getUsage();
    const chunks: string[] = [];
    let usageChunk: { usage?: ModelTokenUsage } | undefined;
    try {
      const maxTokens = allocation?.status === 'granted'
        ? Math.max(1, this.completionCapacity(allocation.allocatedTokens, estimatedInputTokens))
        : undefined;
      for await (const chunk of ctx.llm.stream([...messages], { temperature: 0.2, maxTokens })) {
        if (chunk.content) chunks.push(chunk.content);
        if (chunk.usage) usageChunk = chunk;
      }
    } catch (error) {
      if (allocation?.status === 'granted') this.budgetMarket?.release(allocation.id, 'root_completion_failed');
      throw error;
    }
    const content = chunks.join('');
    ctx.agent.recordRuntimeCompletion(content, {
      content,
      usage: usageChunk?.usage ?? this.estimateModelUsage(messages, content),
    });
    this.settleDirectBudget('root', allocation, this.usageDifference(usageBefore, ctx.agent.getUsage()), correlationId);
    return content;
  }

  private async completeJSONAsAgent<T>(
    agent: BaseAgent,
    messages: LLMMessage[],
    options: LLMCompletionOptions,
    purpose: string,
    correlationId: string
  ): Promise<T> {
    const ctx = this.getContext();
    if (!ctx.llm) throw new Error('LLM not configured');
    const usageBefore = agent.getUsage();
    const estimatedInput = this.estimateTextTokens(messages.map(message => `${message.role}:${message.content}`).join('\n'));
    const ownsAllocation = agent.id === 'root' || !this.agentBudgetAllocations.has(agent.id);
    const allocation = ownsAllocation
      ? await this.requestAgentBudget({
        parentId: agent.getIdentity().parentId ?? agent.id,
        requesterId: agent.id,
        archetype: agent.id === 'root' ? 'custom' : this.inferAgentArchetype(agent.getInfo()),
        correlationId,
        requestedTokens: this.budgetRequestTokens(estimatedInput, options.maxTokens ?? 512),
        minimumTokens: this.budgetMinimumTokens(estimatedInput),
        priority: purpose.includes('delegation') ? 'high' : 'medium',
        expectedUtility: 0.82,
        purpose,
      })
      : undefined;
    if (allocation?.status === 'denied') throw new Error(`JSON completion rejected by budget market: ${allocation.reason}`);

    const activeAllocationId = this.agentBudgetAllocations.get(agent.id);
    const activeAllocation = activeAllocationId ? this.budgetMarket?.getAllocation(activeAllocationId) : undefined;
    const effectiveAllocation = allocation ?? activeAllocation;
    const allocationRemaining = effectiveAllocation?.status === 'granted'
      ? Math.max(0, effectiveAllocation.allocatedTokens - effectiveAllocation.consumedTokens)
      : undefined;
    const agentRemaining = agent.getCompletionTokenLimit();
    const availableTokens = [allocationRemaining, agentRemaining]
      .filter((value): value is number => value !== undefined)
      .reduce<number | undefined>(
        (minimum, value) => minimum === undefined ? value : Math.min(minimum, value),
        undefined
      );
    if (availableTokens !== undefined && availableTokens <= 0) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'json_completion_allocation_exhausted');
      }
      throw new Error('JSON completion rejected: active allocation is exhausted');
    }
    if (this.budgetAccountingDimension() === 'total_tokens'
      && availableTokens !== undefined
      && availableTokens <= estimatedInput) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'json_completion_input_exceeds_allocation');
      }
      throw new Error(`JSON completion rejected: estimated input ${estimatedInput} exceeds remaining allocation ${availableTokens}`);
    }
    const boundedOptions: LLMCompletionOptions = {
      ...options,
      maxTokens: availableTokens === undefined
        ? options.maxTokens
        : Math.max(1, Math.min(
          options.maxTokens ?? availableTokens,
          this.completionCapacity(availableTokens, estimatedInput)
        )),
    };

    try {
      let value: T;
      if (ctx.llm.completeJSONWithUsage) {
        const result = await ctx.llm.completeJSONWithUsage<T>(messages, boundedOptions);
        value = result.value;
        agent.recordRuntimeUsage(result.completion);
      } else {
        value = await ctx.llm.completeJSON<T>(messages, boundedOptions);
        const output = JSON.stringify(value);
        agent.recordRuntimeUsage({
          content: output,
          usage: this.estimateModelUsage(messages, output),
        });
      }
      const usage = this.usageDifference(usageBefore, agent.getUsage());
      if (ownsAllocation) {
        this.settleDirectBudget(agent.id, allocation, usage, correlationId);
      } else {
        this.consumeActiveAgentBudget(agent.id, usage, correlationId, purpose);
      }
      if (agent.id === 'root') {
        this.recordTurnUsage(usage);
        this.emit({ type: 'budget.updated', agentId: agent.id, correlationId, data: { purpose, ...usage } });
      }
      return value;
    } catch (error) {
      if (allocation?.status === 'granted') this.budgetMarket?.release(allocation.id, 'json_completion_failed');
      throw error;
    }
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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
tom_profile: ${JSON.stringify(result.agent.identity.tomProfile)}
existence_reason: ${result.node.identity.existenceReason ?? 'bounded specialist contribution'}
cognitive_gaps: ${result.node.identity.cognitiveGapIds.join(', ') || 'none'}
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
tom_profile: ${JSON.stringify(result.team.identity.tomProfile)}
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
Compare each agent's belief scope and perspective against the cognitive gaps it was created to fill. Preserve unresolved uncertainty instead of forcing agreement.

${teamReports ? `The following subteam reports have already aggregated their direct members. Treat them as the primary delegation result.\n\n${teamReports}` : ''}

${reports}

Produce the final response to the user as Roy, the root agent.`;
  }

  private async synthesizeEvolutionResult(
    task: string,
    run: EvolutionRunResult,
    correlationId: string
  ): Promise<string> {
    const selected = run.selected;
    const execution = run.selectedExecution;
    const evaluation = run.selectedEvaluation;
    if (!execution) {
      return `Roy could not complete the evolutionary delegation run. Run ${run.id} produced no executable result.`;
    }
    return this.completeAsRoot(
      [
        'Synthesize the selected evolutionary agent/team result into the final answer to the user.',
        'Do not expose hidden chain-of-thought. Explain observable evidence, limitations, and relevant disagreements.',
        `<user_task>${task}</user_task>`,
        `<evolution_run>${JSON.stringify({
          runId: run.id,
          profile: run.profile,
          candidateCount: run.metrics.candidateCount,
          selectedGenome: selected?.genome,
          evaluation,
          metrics: run.metrics,
        }, null, 2)}</evolution_run>`,
        `<selected_result>${execution.result}</selected_result>`,
        `<warnings>${execution.warnings.join('\n') || 'none'}</warnings>`,
      ].join('\n\n'),
      'root.evolution_synthesis',
      correlationId
    );
  }

  private resolveEvolutionRunOptions(
    profileOverride?: EvolutionProfile,
    override?: RunEvolutionInput['options']
  ): EvolutionRunOptions {
    const configured = this.workspaceRuntimeConfig?.evolution;
    if (!configured) throw new Error('Evolution workspace configuration is unavailable');
    const profile = profileOverride ?? override?.profile ?? configured.profile;
    const profileAblations: Record<EvolutionProfile, EvolutionAblations> = {
      solo: {
        withoutSubagents: true, withoutToMProfile: true, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      fixed_subagents: {
        withoutSubagents: false, withoutToMProfile: true, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      tom_subteam: {
        withoutSubagents: false, withoutToMProfile: false, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      budget_market: {
        withoutSubagents: false, withoutToMProfile: false, withoutBudgetMarket: false,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      evo_team: { ...configured.ablations },
    };
    const ablations = { ...profileAblations[profile], ...override?.ablations };
    return {
      profile,
      populationSize: clampInteger(override?.populationSize ?? configured.populationSize, 1, 10),
      generations: ablations.withoutEvoMutation ? 0 : clampInteger(override?.generations ?? configured.generations, 0, 5),
      topK: clampInteger(override?.topK ?? configured.topK, 1, 5),
      maxExecutedCandidates: clampInteger(override?.maxExecutedCandidates ?? configured.maxExecutedCandidates, 1, 10),
      integrationMinimumScore: clamp01(override?.integrationMinimumScore ?? configured.integrationMinimumScore),
      patternSimilarityThreshold: clamp01(override?.patternSimilarityThreshold ?? configured.patternSimilarityThreshold),
      useLlmJudge: override?.useLlmJudge ?? configured.useLlmJudge,
      ablations,
    };
  }

  private validateEvolutionConfigPatch(
    patch: Partial<Omit<WorkspaceRuntimeConfig['evolution'], 'ablations'>> & {
      ablations?: Partial<WorkspaceRuntimeConfig['evolution']['ablations']>;
    }
  ): void {
    if (patch.enabled !== undefined && typeof patch.enabled !== 'boolean') throw new Error('Evolution enabled must be boolean');
    if (patch.mode !== undefined && patch.mode !== 'manual' && patch.mode !== 'auto') throw new Error('Evolution mode must be manual or auto');
    if (patch.profile !== undefined
      && !['solo', 'fixed_subagents', 'tom_subteam', 'budget_market', 'evo_team'].includes(patch.profile)) {
      throw new Error('Unsupported evolution profile');
    }
    for (const key of ['populationSize', 'generations', 'topK', 'maxExecutedCandidates'] as const) {
      const value = patch[key];
      if (value !== undefined && (!Number.isInteger(value) || value < (key === 'generations' ? 0 : 1))) {
        throw new Error(`Evolution ${key} must be ${key === 'generations' ? 'a non-negative' : 'a positive'} integer`);
      }
    }
    for (const key of ['integrationMinimumScore', 'patternSimilarityThreshold'] as const) {
      const value = patch[key];
      if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 1)) {
        throw new Error(`Evolution ${key} must be between 0 and 1`);
      }
    }
    if (patch.useLlmJudge !== undefined && typeof patch.useLlmJudge !== 'boolean') {
      throw new Error('Evolution useLlmJudge must be boolean');
    }
    if (patch.ablations) {
      for (const [key, value] of Object.entries(patch.ablations)) {
        if (!['withoutSubagents', 'withoutToMProfile', 'withoutBudgetMarket', 'withoutEvoMutation', 'withoutPatternMemory'].includes(key)
          || typeof value !== 'boolean') {
          throw new Error(`Invalid evolution ablation ${key}`);
        }
      }
    }
  }

  private async runSoloEvolutionBaseline(
    runId: string,
    correlationId: string,
    task: string,
    options: EvolutionRunOptions
  ): Promise<EvolutionRunResult> {
    const ctx = this.getContext();
    const startedAt = Date.now();
    if (options.ablations.withoutBudgetMarket) this.evolutionBudgetBypassCorrelations.add(correlationId);
    const usageBefore = ctx.agent.getUsage();
    this.emit({ type: 'evo.fsm.transition', agentId: 'root', correlationId, data: { runId, from: 'S_evo_idle', to: 'S_evo_execute', profile: 'solo' } });
    try {
      const result = await this.completeAsRoot(
        `Solve the following task directly as Roy without creating subagents or teams. Return a concrete answer and state limitations.\n\n${task}`,
        'evo.solo_baseline',
        correlationId
      );
      const usage = this.usageDifference(usageBefore, ctx.agent.getUsage());
      const completedAt = Date.now();
      const artifact: EvolutionExecutionArtifact = {
        candidateId: 'solo_root', actorKind: 'agent', actorId: 'root', success: Boolean(result.trim()), result,
        usage: this.tokenUsageToEvolutionUsage(usage), wallClockMs: completedAt - startedAt,
        agentIds: [], teamIds: [], toolCalls: 0, successfulToolCalls: 0, unresolvedToolIntents: 0,
        groundedResults: 0, totalResults: 1, failedActors: result.trim() ? 0 : 1,
        recoveredFailures: 0, warnings: [],
      };
      const answerQuality = result.trim() ? Math.min(1, 0.5 + Math.log10(result.length + 1) / 5) : 0;
      const run: EvolutionRunResult = {
        id: runId, correlationId, task, profile: 'solo', state: 'S_evo_done',
        candidates: [], executions: [artifact], evaluations: [], selected: undefined,
        selectedExecution: artifact, selectedEvaluation: undefined,
        metrics: {
          taskSuccess: Boolean(result.trim()), answerQuality, toolSuccessRate: 0,
          agentsSpawned: 0, teamsSpawned: 0, totalTokens: usage.totalTokens,
          thinkingTokens: usage.thinkingTokens, wallClockMs: completedAt - startedAt,
          budgetRequested: 0, budgetAllocated: 0, failureRecoveryCount: 0,
          candidateCount: 0, executedCandidateCount: 1, cacheHits: 0, mutationsApplied: 0,
        },
        ablations: options.ablations, startedAt, completedAt,
      };
      this.evolutionRuns.push(run);
      await ctx.memory.recordEvolutionRun(run as unknown as Record<string, unknown>);
      this.emit({ type: 'evo.run.completed', agentId: 'root', correlationId, data: { runId, profile: 'solo', metrics: run.metrics } });
      return structuredClone(run);
    } finally {
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
    }
  }

  private async findRelevantEvolutionPatterns(task: string, threshold: number): Promise<EvolutionPattern[]> {
    const patterns = await this.getContext().memory.getEvolutionPatterns();
    const embeddings = new HashTaskEmbeddingProvider();
    return patterns
      .map(pattern => ({ pattern, similarity: embeddings.similarity(task, pattern.taskSignature) }))
      .filter(item => item.pattern.status !== 'deprecated' && item.similarity >= threshold)
      .sort((left, right) => {
        const leftScore = left.similarity * 0.7 + left.pattern.averageScore * 0.3;
        const rightScore = right.similarity * 0.7 + right.pattern.averageScore * 0.3;
        return rightScore - leftScore;
      })
      .map(item => item.pattern);
  }

  private async createEvolutionSeeds(
    task: string,
    parentId: string,
    correlationId: string,
    options: EvolutionRunOptions
  ): Promise<EvolutionSeedAgent[]> {
    let decision: DelegationDecision;
    if (parentId === 'root') {
      decision = await this.decideDelegation(task, correlationId);
      decision = await this.selectDelegationCandidate(parentId, task, decision, correlationId, 'root');
    } else {
      const parent = this.getContext().manager.getAgentById(parentId)?.getInfo();
      if (!parent) throw new Error(`Evolution parent agent "${parentId}" not found`);
      decision = await this.decideAgentDelegation(parent, task, correlationId);
    }
    const plans = decision.action === 'spawn_subagents' && decision.agents.length > 0
      ? decision.agents
      : this.defaultEvolutionPlans(task);
    const limit = Math.max(1, Math.min(options.populationSize, this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5));
    return plans.slice(0, limit).map(plan => ({
      archetype: plan.archetype,
      name: plan.name,
      role: plan.existenceReason ?? plan.archetype,
      task: plan.task,
      tools: plan.tools ?? this.getDefaultToolBindings(plan.archetype).map(binding => binding.name),
      skills: plan.skills ?? this.getDefaultSkillBindings(plan.archetype).map(binding => binding.name),
      budgetTokens: plan.budgetTokens,
      tomLevel: options.ablations.withoutToMProfile ? 0 : plan.tomLevel,
      perspective: options.ablations.withoutToMProfile ? undefined : plan.tomProfile?.perspective,
      groundingRequired: plan.archetype === 'researcher' || plan.archetype === 'tester',
    }));
  }

  private defaultEvolutionPlans(task: string): DelegationAgentPlan[] {
    const lower = task.toLowerCase();
    if (/\b(implement|code|fix|patch|refactor)\b/.test(lower)) {
      return [
        { archetype: 'planner', task: `Decompose the implementation and its dependencies: ${task}`, tomLevel: 1, existenceReason: 'implementation planning' },
        { archetype: 'coder', task: `Produce the bounded implementation analysis or change: ${task}`, tomLevel: 0, existenceReason: 'implementation capability' },
        { archetype: 'tester', task: `Verify behavior, regressions, and failure paths for: ${task}`, tomLevel: 0, existenceReason: 'verification capability' },
      ];
    }
    if (/\b(inspect|analy[sz]e|review|risk|architecture|project|repo|structure)\b/.test(lower)) {
      return [
        { archetype: 'researcher', task: `Collect grounded project evidence for: ${task}`, tomLevel: 0, existenceReason: 'missing project evidence' },
        { archetype: 'critic', task: `Stress-test claims, risks, and evidence gaps for: ${task}`, tomLevel: 2, existenceReason: 'missing failure-mode perspective' },
      ];
    }
    return [{ archetype: 'custom', name: 'TaskSpecialist', task, tomLevel: 1, existenceReason: 'bounded specialist capability not covered by a built-in role' }];
  }

  private async instantiateEvolutionCandidate(
    candidate: EvolutionCandidate,
    parentId: string,
    correlationId: string,
    options: EvolutionRunOptions
  ): Promise<{ kind: 'agent' | 'team'; actorId: string }> {
    const members = candidate.genome.members;
    if (members.length === 1) {
      const member = members[0];
      const agent = await this.spawnAgent({
        parentId,
        name: member.name,
        customRole: member.role,
        archetype: member.archetype,
        tomLevel: member.tomProfile.level,
        description: candidate.genome.purpose,
        task: member.task,
        tools: member.toolPolicy.map(tool => tool.name),
        skills: member.skills,
        budgetTokens: options.ablations.withoutBudgetMarket ? undefined : member.budgetPolicy.requestedTokens,
        systemPrompt: member.rolePrompt,
        outputContract: {
          format: member.outputContract.format,
          requiredFields: member.outputContract.requiredFields,
          groundingRequired: member.outputContract.groundingRequired,
        },
        correlationId,
        tomProfile: this.genomeToRuntimeToM(member.tomProfile, member.id, parentId, member.role, member.toolPolicy.map(tool => tool.name)),
        cacheHits: candidate.lineage.parentPatternIds,
        cognitiveGapIds: member.tomProfile.uncertainty.map((_, index) => `${member.id}_gap_${index + 1}`),
        existenceReason: candidate.rationale,
      });
      return { kind: 'agent', actorId: agent.identity.id };
    }
    const team = await this.spawnTeam({
      parentAgentId: parentId,
      name: candidate.genome.name,
      description: candidate.genome.purpose,
      task: candidate.genome.taskSignature,
      tomLevel: options.ablations.withoutToMProfile ? 0 : candidate.genome.tomLevel,
      correlationId,
      members: members.map((member, index) => ({
        archetype: member.archetype,
        name: member.name,
        role: member.role,
        task: member.task,
        tools: member.toolPolicy.map(tool => tool.name),
        skills: member.skills,
        budgetTokens: options.ablations.withoutBudgetMarket ? undefined : member.budgetPolicy.requestedTokens,
        tomLevel: member.tomProfile.level,
        tomProfile: this.genomeToRuntimeToM(member.tomProfile, member.id, parentId, member.role, member.toolPolicy.map(tool => tool.name)),
        systemPrompt: member.rolePrompt,
        existenceReason: candidate.rationale,
        lead: index === 0,
      })),
      executionPolicy: {
        mode: candidate.genome.coordinationPolicy === 'parallel' ? 'parallel' : 'sequential',
        failureMode: 'best_effort',
        maxConcurrency: Math.min(3, members.length),
        minimumSuccessfulMembers: 1,
      },
    });
    return { kind: 'team', actorId: team.identity.id };
  }

  private async executeEvolutionCandidate(
    candidate: EvolutionCandidate,
    actor: { kind: 'agent' | 'team'; actorId?: string },
    correlationId: string,
    details: Map<string, { agent?: RunAgentResult; team?: TeamRunResult }>
  ): Promise<EvolutionExecutionArtifact> {
    const startedAt = Date.now();
    try {
      if (!actor.actorId) throw new Error('Evolution candidate actor id is missing');
      if (actor.kind === 'agent') {
        const member = candidate.genome.members[0];
        const result = await this.runAgent(actor.actorId, member.task, {
          correlationId,
          archetype: member.archetype,
          disableRecursiveDelegation: true,
        });
        details.set(candidate.id, { agent: result });
        return {
          candidateId: candidate.id, actorKind: 'agent', actorId: actor.actorId,
          success: Boolean(result.result.trim()), result: result.result,
          usage: this.tokenUsageToEvolutionUsage(result.usage), wallClockMs: Date.now() - startedAt,
          agentIds: [actor.actorId], teamIds: [], toolCalls: result.toolCalls.length,
          successfulToolCalls: result.toolCalls.filter(call => call.success).length,
          unresolvedToolIntents: this.containsUnresolvedToolIntent(result.result) ? 1 : 0,
          groundedResults: result.grounded ? 1 : 0, totalResults: 1,
          failedActors: result.agent.state === 'failed' ? 1 : 0, recoveredFailures: 0,
          warnings: [...result.warnings],
        };
      }
      const result = await this.runTeam(actor.actorId, candidate.genome.taskSignature, { correlationId });
      details.set(candidate.id, { team: result });
      const toolCalls = result.members.flatMap(member => member.toolCalls);
      const unresolvedToolIntents = result.members.filter(member => this.containsUnresolvedToolIntent(member.result)).length
        + (this.containsUnresolvedToolIntent(result.result) ? 1 : 0);
      const failedActors = result.memberOutcomes.filter(outcome => outcome.status === 'failed').length;
      return {
        candidateId: candidate.id, actorKind: 'team', actorId: actor.actorId,
        success: result.team.status === 'done' && Boolean(result.result.trim()), result: result.result,
        usage: this.tokenUsageToEvolutionUsage(result.usage), wallClockMs: Date.now() - startedAt,
        agentIds: result.members.map(member => member.agent.identity.id), teamIds: [actor.actorId],
        toolCalls: toolCalls.length, successfulToolCalls: toolCalls.filter(call => call.success).length,
        unresolvedToolIntents,
        groundedResults: result.members.filter(member => member.grounded).length,
        totalResults: result.members.length, failedActors,
        recoveredFailures: result.team.status === 'done' ? failedActors : 0,
        warnings: result.members.flatMap(member => member.warnings),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const team = actor.actorId ? this.getTeamState(actor.actorId) : undefined;
      const agent = actor.actorId ? this.getContext().manager.getAgentById(actor.actorId)?.getInfo() : undefined;
      return {
        ...this.failedEvolutionArtifact(candidate, actor.actorId ?? 'execution-failed', message),
        actorKind: actor.kind,
        wallClockMs: Date.now() - startedAt,
        agentIds: team?.memberAgentIds ?? (agent ? [agent.identity.id] : []),
        teamIds: team ? [team.identity.id] : [],
      };
    }
  }

  private failedEvolutionArtifact(candidate: EvolutionCandidate, actorId: string, error: string): EvolutionExecutionArtifact {
    return {
      candidateId: candidate.id,
      actorKind: candidate.genome.members.length === 1 ? 'agent' : 'team',
      actorId,
      success: false,
      result: '',
      usage: this.tokenUsageToEvolutionUsage(this.sumUsage([])),
      wallClockMs: 0,
      agentIds: [], teamIds: [], toolCalls: 0, successfulToolCalls: 0, unresolvedToolIntents: 0,
      groundedResults: 0, totalResults: candidate.genome.members.length,
      failedActors: candidate.genome.members.length, recoveredFailures: 0,
      warnings: [error],
    };
  }

  private archiveEvolutionCandidateAgents(agentIds: string[], runId: string, candidateId: string): void {
    const ctx = this.getContext();
    for (const agentId of agentIds) {
      const agent = ctx.manager.getAgentById(agentId);
      if (!agent || agentId === 'root') continue;
      const info = agent.getInfo();
      this.archivedAgentUsage.set(agentId, this.toTokenUsage(info.usage));
      ctx.manager.removeAgent(info.name);
      this.agentBindings.delete(agentId);
      this.agentFsms.delete(agentId);
      this.agentBudgetAllocations.delete(agentId);
      this.agentBudgetLimits.delete(agentId);
      this.toolCallCounts.delete(agentId);
      this.emit({
        type: 'evo.candidate.actor.archived', agentId, sessionId: ctx.sessionId,
        data: { runId, candidateId, name: info.identity.name, totalTokens: info.usage.totalTokens },
      });
    }
  }

  private createEvolutionJudge(
    parentId: string,
    correlationId: string,
    onUsage: (usage: TokenUsage) => void
  ): EvolutionJudge {
    return {
      name: 'llm_judge',
      evaluate: async (task, candidate, execution) => {
        const agent = this.getContext().manager.getAgentById(parentId) ?? this.getContext().agent;
        const usageBefore = agent.getUsage();
        this.emit({ type: 'evo.judge.started', agentId: agent.id, correlationId, data: { candidateId: candidate.id } });
        const judged = await this.completeJSONAsAgent<Partial<EvolutionEvaluationDimensions> & { rationale?: string }>(
          agent,
          [
            {
              role: 'system',
              content: 'Evaluate an executed agent/team candidate. Return strict JSON with optional 0..1 fields taskSuccess, answerQuality, completeness, costEfficiency, novelty, toolUse, consistency, tomCoverage, plus rationale. Judge only observable output and metrics.',
            },
            {
              role: 'user',
              content: JSON.stringify({ task, genome: candidate.genome, execution: { ...execution, result: execution.result.slice(0, 8000) } }),
            },
          ],
          { temperature: 0, maxTokens: 600 },
          'evo.candidate_evaluation',
          correlationId
        );
        const usage = this.usageDifference(usageBefore, agent.getUsage());
        onUsage(usage);
        this.emit({ type: 'evo.judge.completed', agentId: agent.id, correlationId, data: { candidateId: candidate.id, usage } });
        return this.normalizeEvolutionJudgeResult(judged);
      },
    };
  }

  private normalizeEvolutionJudgeResult(
    value: Partial<EvolutionEvaluationDimensions> & { rationale?: string }
  ): Partial<EvolutionEvaluationDimensions> & { rationale?: string } {
    const result: Partial<EvolutionEvaluationDimensions> & { rationale?: string } = {};
    for (const key of ['taskSuccess', 'answerQuality', 'completeness', 'costEfficiency', 'novelty', 'toolUse', 'consistency', 'tomCoverage'] as const) {
      const score = value[key];
      if (typeof score === 'number' && Number.isFinite(score)) result[key] = clamp01(score);
    }
    if (typeof value.rationale === 'string') result.rationale = value.rationale;
    return result;
  }

  private async integrateEvolutionPattern(
    selected: EvolutionCandidate,
    evaluation: EvolutionEvaluationResult,
    execution: EvolutionExecutionArtifact
  ): Promise<string> {
    const ctx = this.getContext();
    const structure = selected.genome.members.map(member => ({
      archetype: member.archetype,
      role: member.role,
      tools: member.toolPolicy.map(tool => tool.name),
    }));
    const patternId = `evo_pattern_${this.fingerprint({ task: selected.genome.taskSignature, structure }).slice(0, 16)}_v1`;
    const [agentPatterns, teamPatterns] = await Promise.all([
      ctx.memory.getCachePatterns('agents'),
      ctx.memory.getCachePatterns('teams'),
    ]);
    const agentPatternIds = agentPatterns
      .filter(pattern => selected.genome.members.some(member => pattern.archetype === member.archetype))
      .map(pattern => String(pattern.id ?? ''))
      .filter(Boolean);
    const teamPatternIds = execution.actorKind === 'team'
      ? teamPatterns
        .filter(pattern => pattern.name === selected.genome.name || pattern.key === this.safeAgentKey(selected.genome.name))
        .map(pattern => String(pattern.id ?? ''))
        .filter(Boolean)
      : [];
    const links = { agentPatternIds, teamPatternIds };
    const pattern = await ctx.memory.upsertEvolutionPattern({
      id: patternId,
      name: selected.genome.name,
      taskSignature: selected.genome.taskSignature,
      genome: selected.genome,
      historicalScores: [],
      lineage: selected.lineage,
      linkedPatterns: links,
      evaluation,
      tokenCost: execution.usage.totalTokens,
    });
    await ctx.memory.linkEvolutionPattern(pattern.id, links);
    this.emit({
      type: 'evo.candidate.integrated', agentId: execution.actorId,
      data: { candidateId: selected.id, patternId: pattern.id, score: evaluation.score, linkedPatterns: links },
    });
    return pattern.id;
  }

  private async recordEvolutionLifecycleMessage(
    state: string,
    parentId: string,
    correlationId: string,
    payload: Record<string, unknown>
  ): Promise<void> {
    const kind = state === 'S_evo_propose' ? 'evo.propose'
      : state === 'S_evo_instantiate' ? 'evo.instantiate'
        : state === 'S_evo_execute' ? 'evo.execute'
          : state === 'S_evo_evaluate' ? 'evo.evaluate'
            : state === 'S_evo_select' ? 'evo.select'
              : state === 'S_evo_mutate' ? 'evo.mutate'
                : state === 'S_evo_integrate' || state === 'S_evo_done' ? 'evo.integrate'
                  : undefined;
    if (!kind) return;
    const message = await this.enqueueMessage({
      kind,
      sessionId: this.getContext().sessionId,
      from: parentId,
      to: 'runtime.evolution',
      correlationId,
      payload,
      metadata: { agentId: parentId, tags: ['evolution'] },
    });
    await this.processQueuedMessage(message.id);
    await this.getContext().queue.ack(message.id);
  }

  private genomeToRuntimeToM(
    profile: GenomeToMProfile,
    subjectAgentId: string,
    parentId: string,
    purpose: string,
    capabilities: string[]
  ): ToMProfile {
    return normalizeToMProfile({
      level: profile.level,
      subjectAgentId,
      beliefScope: [...profile.beliefScope],
      goalModel: [...profile.goalModel],
      uncertainty: [...profile.uncertainty],
      perspective: profile.perspective,
      observesAgents: [...profile.observesAgents ?? []],
      modelsAgents: [...profile.modelsAgents ?? []],
      capabilityScope: [...capabilities],
      cognitiveGaps: profile.uncertainty.map((_, index) => `${subjectAgentId}_gap_${index + 1}`),
      models: profile.level >= 1 ? [{
        targetId: parentId,
        targetType: 'agent',
        goalModel: [...profile.goalModel],
        uncertaintyModel: [...profile.uncertainty],
      }] : [],
      purpose,
    }, { level: profile.level, subjectAgentId, purpose });
  }

  private buildEvolutionMetrics(input: {
    lifecycle: EvolutionLifecycleResult;
    usage: TokenUsage;
    startedAt: number;
    completedAt: number;
    cacheHits: number;
    budgetRequested: number;
    budgetAllocated: number;
  }): EvolutionRunResult['metrics'] {
    const selectedEvaluation = input.lifecycle.selectedEvaluation;
    const toolCalls = input.lifecycle.executions.reduce((sum, execution) => sum + execution.toolCalls, 0);
    const successfulToolCalls = input.lifecycle.executions.reduce((sum, execution) => sum + execution.successfulToolCalls, 0);
    return {
      taskSuccess: selectedEvaluation?.success ?? false,
      answerQuality: selectedEvaluation?.dimensions.answerQuality ?? 0,
      toolSuccessRate: toolCalls > 0 ? successfulToolCalls / toolCalls : 0,
      agentsSpawned: input.lifecycle.executions.reduce((sum, execution) => sum + execution.agentIds.length, 0),
      teamsSpawned: input.lifecycle.executions.reduce((sum, execution) => sum + execution.teamIds.length, 0),
      totalTokens: input.usage.totalTokens,
      thinkingTokens: input.usage.thinkingTokens,
      wallClockMs: input.completedAt - input.startedAt,
      budgetRequested: input.budgetRequested,
      budgetAllocated: input.budgetAllocated,
      failureRecoveryCount: input.lifecycle.executions.reduce((sum, execution) => sum + execution.recoveredFailures, 0),
      candidateCount: input.lifecycle.candidates.length,
      executedCandidateCount: input.lifecycle.executions.length,
      selectedGenomeId: input.lifecycle.selected?.genome.id,
      selectedGenomeScore: selectedEvaluation?.score,
      cacheHits: input.cacheHits,
      mutationsApplied: input.lifecycle.candidates.filter(candidate => candidate.lineage.operators.length > 0).length,
    };
  }

  private tokenUsageToEvolutionUsage(usage: TokenUsage): EvolutionExecutionArtifact['usage'] {
    return {
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thinkingTokens: usage.thinkingTokens,
      totalTokens: usage.totalTokens,
      cachedInputTokens: usage.cachedInputTokens,
      cacheCreationInputTokens: usage.cacheCreationInputTokens,
    };
  }

  private evolutionUsageToTokenUsage(usage: EvolutionExecutionArtifact['usage']): TokenUsage {
    return {
      llmCalls: usage.totalTokens > 0 ? 1 : 0,
      promptTokens: usage.inputTokens,
      completionTokens: usage.outputTokens,
      totalTokens: usage.totalTokens,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thinkingTokens: usage.thinkingTokens,
      thinkingAccountingTokens: usage.thinkingTokens ?? usage.totalTokens,
      cachedInputTokens: usage.cachedInputTokens ?? null,
      cacheCreationInputTokens: usage.cacheCreationInputTokens ?? null,
    };
  }

  private sumUsage(items: TokenUsage[]): TokenUsage {
    return items.reduce<TokenUsage>((total, item) => ({
      llmCalls: total.llmCalls + item.llmCalls,
      promptTokens: total.promptTokens + item.promptTokens,
      completionTokens: total.completionTokens + item.completionTokens,
      totalTokens: total.totalTokens + item.totalTokens,
      inputTokens: total.inputTokens + item.inputTokens,
      outputTokens: total.outputTokens + item.outputTokens,
      thinkingTokens: total.thinkingTokens === null && item.thinkingTokens === null
        ? null
        : Number(total.thinkingTokens ?? 0) + Number(item.thinkingTokens ?? 0),
      thinkingAccountingTokens: (total.thinkingAccountingTokens ?? 0) + (item.thinkingAccountingTokens ?? item.thinkingTokens ?? item.totalTokens),
      cachedInputTokens: total.cachedInputTokens === null && item.cachedInputTokens === null
        ? null
        : Number(total.cachedInputTokens ?? 0) + Number(item.cachedInputTokens ?? 0),
      cacheCreationInputTokens: total.cacheCreationInputTokens === null && item.cacheCreationInputTokens === null
        ? null
        : Number(total.cacheCreationInputTokens ?? 0) + Number(item.cacheCreationInputTokens ?? 0),
    }), {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      thinkingTokens: null,
      thinkingAccountingTokens: 0,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
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
      inputTokens: Math.max(0, after.inputTokens - before.inputTokens),
      outputTokens: Math.max(0, after.outputTokens - before.outputTokens),
      thinkingTokens,
      thinkingAccountingTokens: Math.max(
        0,
        (after.thinkingAccountingTokens ?? after.thinkingTokens ?? after.totalTokens)
          - (before.thinkingAccountingTokens ?? before.thinkingTokens ?? before.totalTokens)
      ),
      cachedInputTokens: after.cachedInputTokens === null && before.cachedInputTokens === null
        ? null
        : Math.max(0, Number(after.cachedInputTokens ?? 0) - Number(before.cachedInputTokens ?? 0)),
      cacheCreationInputTokens: after.cacheCreationInputTokens === null && before.cacheCreationInputTokens === null
        ? null
        : Math.max(0, Number(after.cacheCreationInputTokens ?? 0) - Number(before.cacheCreationInputTokens ?? 0)),
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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
          tomProfile: plan.tomProfile,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
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
        tomAnalysis: this.tomAnalyses.get(correlationId),
        correlationId,
      });
      teamResult = await this.runTeam(team.identity.id, task, { correlationId });
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
          tomProfile: plan.tomProfile,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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
    const groundedChildren = childResults.filter(result => result.subagentResult.grounded).length;
    const groundedRatio = childResults.length > 0 ? groundedChildren / childResults.length : 0;
    this.settleAgentBudget(agentId, usageDelta, {
      success: true,
      evidenceGain: evidence.outputGrounded ? 0.9 : evidence.toolGrounded ? 0.5 : 0.15,
      uncertaintyReduction: groundedRatio,
      conflictResolution: childResults.length > 1 ? groundedRatio : undefined,
    });
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
      tomProfile: member.tomProfile,
      cognitiveGapIds: member.cognitiveGapIds,
      existenceReason: member.existenceReason,
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
      tomProfile: team.identity.tomProfile,
      status: team.status,
      fsmState: team.fsmState,
      members,
      plannedMembers,
      tokenUsage: team.tokenUsage,
      executionPolicy: team.executionPolicy,
      memberStatuses: team.memberStatuses,
      memberErrors: team.memberErrors,
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
          memberStatuses: input.team.memberStatuses,
          memberErrors: input.team.memberErrors,
          partial: Object.keys(input.team.memberErrors).length > 0,
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
          failedMemberCount: Object.keys(input.team.memberErrors).length,
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
    failures: Array<TeamExecutionOutcome<unknown>>,
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
    const orderedMembers = [...members].sort((left, right) => {
      if (left.agent.identity.id === team.leadAgentId) return -1;
      if (right.agent.identity.id === team.leadAgentId) return 1;
      return 0;
    });
    const reports = orderedMembers.map(member => [
      `<member id="${member.agent.identity.id}" name="${member.agent.identity.name}">`,
      `team_role: ${member.agent.identity.id === team.leadAgentId ? 'lead' : 'member'}`,
      `tokens: ${member.usage.totalTokens}`,
      `grounded: ${member.grounded}`,
      `tom_profile: ${JSON.stringify(member.agent.identity.tomProfile)}`,
      member.result,
      '</member>',
    ].join('\n')).join('\n\n');
    const failureReports = failures.map(failure => [
      `<member_failure key="${failure.key}">`,
      failure.error ?? 'unknown member execution failure',
      '</member_failure>',
    ].join('\n')).join('\n\n');
    const prompt = [
      `Team task: ${task}`,
      `You are ${team.identity.name}, a subteam actor in Roy.`,
      `Description: ${team.identity.description}`,
      `ToM level: ${team.identity.tomLevel}`,
      `ToM profile: ${JSON.stringify(team.identity.tomProfile, null, 2)}`,
      `Lead agent: ${team.leadAgentId ?? 'not assigned'}`,
      `<team_definition>\n${teamDefinition}\n</team_definition>`,
      `<team_private_memory>\n${teamMemory}\n</team_private_memory>`,
      `<public_context>\n${publicContext}\n</public_context>`,
      'Aggregate direct member reports into one grounded result for the parent agent.',
      'Reconcile member beliefs explicitly, preserve unresolved uncertainty, and explain how the final result covers the team cognitive gaps.',
      'Give the lead report coordination priority, but verify it against all available member evidence.',
      'If member failures are present, state their impact and do not imply full team completion.',
      'Do not claim evidence that is absent from member reports.',
      reports,
      failureReports,
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
      data: {
        teamId: team.identity.id,
        memberAgentIds: team.memberAgentIds,
        leadAgentId: team.leadAgentId,
        failedMembers: failures.length,
      },
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
      ? Math.max(1, this.completionCapacity(allocation.grantedTokens, estimatedPromptTokens))
      : 1024;
    const chunks: string[] = [];
    let providerUsage: ModelTokenUsage | undefined;
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
    const normalizedUsage = providerUsage ?? this.estimateModelUsage([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt },
    ], content);
    const usage = this.toTokenUsage({
      llmCalls: 1,
      promptTokens: normalizedUsage.promptTokens,
      completionTokens: normalizedUsage.completionTokens,
      totalTokens: normalizedUsage.totalTokens,
      inputTokens: normalizedUsage.inputTokens ?? normalizedUsage.promptTokens,
      outputTokens: normalizedUsage.outputTokens ?? normalizedUsage.completionTokens,
      thinkingTokens: normalizedUsage.thinkingTokens ?? null,
      cachedInputTokens: normalizedUsage.cachedInputTokens ?? null,
      cacheCreationInputTokens: normalizedUsage.cacheCreationInputTokens ?? null,
    });
    this.settleTeamSynthesisBudget(team.identity.id, allocation, usage, correlationId);
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

    const remainingTokens = agent.getCompletionTokenLimit();
    if (remainingTokens !== undefined && remainingTokens <= 0) {
      this.emit({
        type: 'budget.exceeded',
        agentId: agent.id,
        correlationId,
        data: { purpose, remainingTokens, reason: 'agent_allocation_exhausted' },
      });
      throw new Error('Agent completion rejected: active allocation is exhausted');
    }
    const systemBase = [
      `You are ${agent.name}, a runtime agent in the Roy autonomous agent system.`,
      'You are not the model provider. The provider is only your inference backend.',
      `Purpose: ${purpose}.`,
      `Correlation: ${correlationId}.`,
    ].join('\n');
    let communicationContext = agent.getCommunicationContext()?.rendered ?? '';
    let effectivePrompt = prompt;
    if (remainingTokens !== undefined && this.budgetAccountingDimension() === 'total_tokens') {
      const outputReserve = Math.min(512, Math.max(64, Math.floor(remainingTokens * 0.2)));
      const inputBudget = Math.max(0, remainingTokens - outputReserve);
      const baseSystemTokens = this.estimateTextTokens(`system:${systemBase}`);
      const communicationTokens = communicationContext
        ? this.estimateTextTokens(communicationContext)
        : 0;
      const communicationBudget = Math.max(0, Math.min(
        communicationTokens,
        Math.floor(Math.max(0, inputBudget - baseSystemTokens) * 0.25)
      ));
      if (communicationTokens > communicationBudget) {
        communicationContext = communicationBudget > 0
          ? this.truncateTextToTokenBudget(communicationContext, communicationBudget)
          : '';
        this.emit({
          type: 'budget.context.truncated',
          agentId: agent.id,
          correlationId,
          data: {
            purpose,
            contextType: 'communication',
            originalTokens: communicationTokens,
            allowedTokens: communicationBudget,
            outputReserve,
          },
        });
      }
      const systemTokens = this.estimateTextTokens(`system:${systemBase}\n${communicationContext}`);
      const promptBudget = Math.max(0, inputBudget - systemTokens);
      const promptTokens = this.estimateTextTokens(`user:${effectivePrompt}`);
      if (promptTokens > promptBudget && promptBudget > 0) {
        effectivePrompt = this.truncateTextToTokenBudget(effectivePrompt, promptBudget);
        this.emit({
          type: 'budget.context.truncated',
          agentId: agent.id,
          correlationId,
          data: { purpose, originalTokens: promptTokens, allowedTokens: promptBudget, outputReserve },
        });
      }
    }
    const systemContent = [systemBase, communicationContext].filter(Boolean).join('\n');
    const messages: LLMMessage[] = [
      { role: 'system', content: systemContent },
      { role: 'user', content: effectivePrompt },
    ];
    const estimatedInputTokens = this.estimateTextTokens(
      messages.map(message => `${message.role}:${message.content}`).join('\n')
    );
    if (this.budgetAccountingDimension() === 'total_tokens'
      && remainingTokens !== undefined
      && remainingTokens <= estimatedInputTokens) {
      this.emit({
        type: 'budget.exceeded',
        agentId: agent.id,
        correlationId,
        data: {
          purpose,
          estimatedInputTokens,
          remainingTokens,
          reason: 'input_exceeds_remaining_agent_allocation',
        },
      });
      throw new Error(`Agent completion rejected: estimated input ${estimatedInputTokens} exceeds remaining allocation ${remainingTokens}`);
    }
    const maxTokens = remainingTokens === undefined
      ? undefined
      : Math.max(1, this.completionCapacity(remainingTokens, estimatedInputTokens));
    const chunks: string[] = [];
    let usageChunk: { usage?: ModelTokenUsage } | undefined;
    for await (const chunk of ctx.llm.stream(messages, { temperature: 0.2, maxTokens })) {
      if (chunk.content) chunks.push(chunk.content);
      if (chunk.usage) usageChunk = chunk;
    }
    const content = chunks.join('');
    agent.recordRuntimeCompletion(content, {
      content,
      usage: usageChunk?.usage ?? this.estimateModelUsage(messages, content),
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
      inputTokens: 0,
      outputTokens: 0,
      thinkingTokens: null,
      thinkingAccountingTokens: 0,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
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
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
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
profile: ${JSON.stringify(agent.identity.tomProfile, null, 2)}
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
    communicationProtocol?: string;
    communicationContext?: string;
    multiPartyTraces?: MultiPartyTrace[];
    multiPartyTraceContext?: string;
  }): string {
    const slots: Record<string, string> = {
      public_context: input.publicContext ?? '',
      agent_private_memory: input.bundle.memory.trim(),
      agent_identity: input.bundle.identity.trim() || `You are ${input.name}, a ${input.role} agent in the Roy runtime.`,
      tom_profile: input.tomProfile ? JSON.stringify(input.tomProfile, null, 2) : '',
      communication_context: input.communicationContext
        ?? `Protocol: ${input.communicationProtocol ?? this.communicationManager?.getDefaultProtocolId() ?? 'tom'}. Runtime messages are rendered through the selected communication protocol.`,
      multi_party_traces: input.multiPartyTraceContext ?? (input.multiPartyTraces?.length
        ? input.multiPartyTraces.map(trace => `[${trace.phase}] ${trace.from.id} -> ${trace.to.map(actor => actor.id).join(',')}: ${trace.kind}`).join('\n')
        : 'Observable multi-party traces are injected by the runtime for each message.'),
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
    const runtimeAgent = agentKey === 'roy' ? ctx.agent : ctx.manager.getAgentById(agentKey);
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
      communicationProtocol: runtimeAgent?.getIdentity().communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
      communicationContext: contextWindow.communicationContext,
      multiPartyTraceContext: contextWindow.multiPartyTraceContext,
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
      communicationContext: runtimeAgent?.getCommunicationContext()?.rendered,
      systemTraces: runtimeAgent?.getSystemTraces({ limit: 50 }),
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
    if (request.tomProfileMode
      && request.tomProfileMode !== 'runtime_assignment'
      && request.tomProfileMode !== 'definition_override') {
      throw new Error(`Unsupported ToM profile mode "${String(request.tomProfileMode)}"`);
    }
    if (request.budgetTokens !== undefined
      && (!Number.isFinite(request.budgetTokens) || request.budgetTokens <= 0)) {
      throw new Error('Agent node budgetTokens must be a positive finite number');
    }
    if (request.outputContract
      && !['markdown', 'json', 'structured_report'].includes(request.outputContract.format)) {
      throw new Error(`Unsupported agent output format "${String(request.outputContract.format)}"`);
    }
    if (request.tomProfile) {
      if (!Number.isInteger(request.tomProfile.level) || request.tomProfile.level < 0 || request.tomProfile.level > 3) {
        throw new Error('Agent ToM profile level must be an integer from 0 to 3');
      }
      for (const field of ['beliefScope', 'goalModel', 'uncertainty', 'observesAgents', 'modelsAgents', 'capabilityScope', 'cognitiveGaps'] as const) {
        const value = request.tomProfile[field];
        if (!Array.isArray(value) || value.some(item => typeof item !== 'string' || !item.trim())) {
          throw new Error(`Agent ToM profile ${field} must be an array of non-empty strings`);
        }
      }
      if (!request.tomProfile.purpose?.trim()) throw new Error('Agent ToM profile purpose is required');
    }
    if (request.cognitiveGapIds?.some(item => typeof item !== 'string' || !item.trim())) {
      throw new Error('Agent cognitiveGapIds must contain non-empty strings');
    }
    if (request.existenceReason !== undefined && !request.existenceReason.trim()) {
      throw new Error('Agent existenceReason must be a non-empty string when provided');
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

  private sameStringSet(left: string[], right: string[]): boolean {
    const a = [...new Set(left)].sort();
    const b = [...new Set(right)].sort();
    return a.length === b.length && a.every((value, index) => value === b[index]);
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

  private truncateTextToTokenBudget(text: string, tokenBudget: number): string {
    const targetChars = Math.max(64, tokenBudget * 4);
    if (text.length <= targetChars) return text;
    const marker = '\n...[budget-constrained context truncation]...\n';
    const availableChars = Math.max(32, targetChars - marker.length);
    const headChars = Math.floor(availableChars * 0.4);
    const tailChars = availableChars - headChars;
    return `${text.slice(0, headChars)}${marker}${text.slice(-tailChars)}`;
  }

  private constrainAgentObservation(
    agent: BaseAgent,
    observation: string,
    correlationId?: string,
    purpose = 'agent.task_execution'
  ): string {
    const remainingTokens = agent.getCompletionTokenLimit();
    if (remainingTokens === undefined || this.budgetAccountingDimension() !== 'total_tokens') return observation;
    const outputReserve = Math.min(512, Math.max(128, Math.floor(remainingTokens * 0.2)));
    const promptWrapperReserve = Math.min(512, Math.max(256, Math.floor(remainingTokens * 0.12)));
    const goalTokens = this.estimateTextTokens(agent.getInfo().goal ?? '');
    const observationBudget = Math.max(64, remainingTokens - outputReserve - promptWrapperReserve - goalTokens);
    const observationTokens = this.estimateTextTokens(observation);
    if (observationTokens <= observationBudget) return observation;
    this.emit({
      type: 'budget.context.truncated',
      agentId: agent.id,
      correlationId,
      data: {
        purpose,
        contextType: 'agent_observation',
        originalTokens: observationTokens,
        allowedTokens: observationBudget,
        outputReserve,
        promptWrapperReserve,
      },
    });
    return this.truncateTextToTokenBudget(observation, observationBudget);
  }

  private estimateModelUsage(messages: LLMMessage[], output: string): ModelTokenUsage {
    const llm = this.getContext().llm;
    const normalized = tokenUsageRegistry.normalize({
      provider: llm?.name ?? 'unknown',
      model: llm?.defaultModel,
      messages,
      output,
    });
    if (normalized) return normalized;

    const prompt = messages.map(message => `${message.role}:${message.content}`).join('\n');
    const promptTokens = this.estimateTextTokens(prompt);
    const completionTokens = this.estimateTextTokens(output);
    return {
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      inputTokens: promptTokens,
      outputTokens: completionTokens,
      thinkingTokens: null,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
      provider: llm?.name,
      model: llm?.defaultModel,
      source: 'estimated',
      availability: {
        input: 'estimated',
        output: 'estimated',
        thinking: 'unavailable',
        cachedInput: 'unavailable',
        cacheCreationInput: 'unavailable',
      },
    };
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

  private containsUnresolvedToolIntent(result: string): boolean {
    if (!result.trim()) return false;
    return /<tool_call>[\s\S]*?<\/tool_call>/i.test(result)
      || /<tool_name>[\s\S]*?<\/tool_name>/i.test(result)
      || /\{\s*"(?:tool_name|tool|function)"\s*:\s*"[^"\n]+"[\s\S]*?\}/i.test(result);
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
    return normalizeToMProfile({
      level: 1,
      subjectAgentId: 'root',
      beliefScope: ['user intent', 'runtime state', 'available agent capabilities', 'budget and policy constraints'],
      goalModel: ['Provide a reliable final response and delegate only when another perspective closes a cognitive gap.'],
      uncertainty: ['Which evidence or specialist perspective is missing for the current task?'],
      perspective: 'root coordinator and user-intent modeler',
      observesAgents: [],
      modelsAgents: [],
      capabilityScope: ['task assessment', 'delegation', 'team synthesis', 'memory coordination'],
      cognitiveGaps: [],
      models: [
        {
          targetId: 'user',
          targetType: 'user',
          goalModel: ['develop Roy into a Theory-of-Mind based multi-agent runtime'],
          intentModel: ['validate controlled subagent spawning and message-mediated execution'],
        },
      ],
      purpose: 'Understand user intent and decide how to answer or delegate.',
    }, {
      level: 1,
      subjectAgentId: 'root',
      purpose: 'Understand user intent and decide how to answer or delegate.',
    });
  }

  private createSubagentToMProfile(archetype: SubAgentArchetype, subjectAgentId: string, task: string, parentId = 'root'): ToMProfile {
    const analysis = this.tomPlanner.analyzeTask({
      task,
      parentId,
      parentProfile: this.ctx?.manager.getAgentById(parentId)?.getIdentity().tomProfile,
    });
    const plan = this.tomPlanner.completePlans(analysis, [{ archetype, task }], 1)[0];
    const profile = normalizeToMProfile(plan?.tomProfile, {
      level: this.defaultToMLevel(archetype),
      subjectAgentId: subjectAgentId || archetype,
      purpose: this.defaultToMPurpose(archetype, task),
    });
    profile.subjectAgentId = subjectAgentId || archetype;
    return profile;
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
    const communicationTrace = this.communicationManager?.recordTransition(transition);
    if (communicationTrace && this.ctx) {
      for (const actor of [communicationTrace.from, ...communicationTrace.to]) {
        const info = this.ctx.manager.getAgentById(actor.id)?.getInfo();
        if (!info) continue;
        actor.name = info.identity.name;
        actor.parentId = info.identity.parentId;
        actor.teamId = info.identity.teamId;
        actor.type = info.identity.role === 'subteam' ? 'team' : 'agent';
      }
      const participantIds = new Set([communicationTrace.from.id, ...communicationTrace.to.map(actor => actor.id)]);
      for (const info of this.ctx.manager.listAgentInfo()) {
        const agent = this.ctx.manager.getAgentById(info.identity.id);
        if (agent && participantIds.has(agent.id)) agent.receiveSystemTrace(communicationTrace);
      }
    }
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
        communicationProtocol: message.metadata?.communicationProtocol,
        communicationTraceId: communicationTrace?.id,
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

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function finiteRatio(value: unknown): number | undefined {
  const number = finiteNumber(value);
  return number === undefined ? undefined : Math.max(0, Math.min(1, number));
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const defined = values.filter((value): value is number => value !== undefined);
  return defined.length > 0 ? defined.reduce((sum, value) => sum + value, 0) / defined.length : undefined;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value)) return minimum;
  return Math.max(minimum, Math.min(maximum, Math.floor(value)));
}

function teamCompletionRatio(team: TeamRuntimeState): number {
  if (team.memberAgentIds.length === 0) return 0;
  const completed = Object.values(team.memberStatuses).filter(status => status === 'completed').length;
  return completed / team.memberAgentIds.length;
}

export const runtime = Runtime.getInstance();
export default Runtime;
