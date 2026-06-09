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
import type { AgentInfo, AgentUsage, ToMProfile } from '../agent/BaseAgent.js';
import { actionRegistry } from '../actions/index.js';
import { toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import { DelegateToSubagentSkill } from '../skills/delegation.js';
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

export interface SpawnAgentSpec {
  parentId: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  archetype: SubAgentArchetype;
  tomLevel: number;
  description: string;
  task?: string;
  tools?: string[];
  budgetTokens?: number;
  systemPrompt?: string;
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
  name?: string;
  customRole?: string;
  customStyle?: string;
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
}

export interface RootMediatedSpawnResult {
  correlationId: string;
  agent: AgentInfo;
  subagentResult: RunAgentResult;
  finalResponse: string;
  messages: RuntimeMessage[];
  creationUsage: AgentCreationUsage;
}

export interface AgentCreationUsage {
  mode: 'generated' | 'cache_hit';
  patternIds: string[];
  cacheHits: string[];
  definitionTokens: number;
  renderedPromptTokens: number;
  renderedPromptChars: number;
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
    const memory = new WorkspaceMemoryManager();
    await memory.initWorkspace(options.workspaceCwd ?? process.cwd(), options.sessionId ?? 'main');
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

    this.registerCoreSkills();

    // Register capabilities with agent
    const capabilities = this.registerCapabilities(agent);

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

  private registerCapabilities(agent: UnifiedAgent): RuntimeContext['capabilities'] {
    // Register actions
    const actions = actionRegistry.list();
    for (const action of actions) {
      agent.registerAction(action);
      logger.debug(`Registered action: ${action.name}`);
    }

    // Register tools
    const tools = toolRegistry.list();
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
    skillRegistry.register(new DelegateToSubagentSkill(() => this));
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

  async handleSpawnCommand(payload: SpawnCommandPayload): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const correlationId = this.createCorrelationId();
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
      from: 'cli',
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
    const agent = await this.spawnAgent({
      parentId,
      name: payload.name,
      customRole: payload.customRole,
      customStyle: payload.customStyle,
      archetype: payload.archetype,
      tomLevel: tomProfile.level,
      description: payload.task,
      task: payload.task,
      systemPrompt: undefined,
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

    const finalResponse = requireRootSynthesis
      ? await this.synthesizeSubagentResult(payload.task, agent, subagentResult, correlationId, resultMessage.id)
      : '[system] Subagent completed without root synthesis.';

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
    await ctx.memory.ensureAgentMemory(spec.archetype, {
      name: this.capitalize(spec.archetype),
      role: spec.customRole ?? spec.archetype,
      description: `Reusable ${spec.archetype} agent archetype memory.`,
    });
    const agentMemory = await ctx.memory.loadAgentMemory(spec.archetype);
    if (!spec.description.trim()) {
      throw new Error('Subagent description is required');
    }

    const parent = ctx.manager.getAgentById(spec.parentId);
    if (!parent) {
      throw new Error(`Parent agent "${spec.parentId}" not found`);
    }

    const sequence = ++this.agentSequence;
    const id = this.createAgentId(spec.archetype, sequence);
    const name = spec.name ?? `${this.capitalize(spec.archetype)}-${sequence}`;
    if (ctx.manager.getAgent(name)) {
      throw new Error(`Agent name "${name}" already exists`);
    }
    const parentIdentity = parent.getIdentity();
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
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
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
      toolRegistry.list().map(tool => tool.name).join(','),
      skillRegistry.list().map(skill => skill.name).join(','),
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

    this.registerCapabilities(agent);
    ctx.manager.addAgent(agent);
    await ctx.manager.attachAgentToSessions(agent);
    await ctx.memory.upsertAgentPattern({
      key: spec.archetype,
      name: this.capitalize(spec.archetype),
      archetype: spec.archetype,
      tomLevel: spec.tomLevel,
      description: spec.description,
      tools: toolRegistry.list().map(tool => tool.name),
      skills: skillRegistry.list().map(skill => skill.name),
    });

    const info = agent.getInfo();
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
    if (spec.budgetTokens !== undefined) {
      this.emit({ type: 'budget.allocated', agentId: id, data: { budgetTokens: spec.budgetTokens } });
    }

    return info;
  }

  async runAgent(
    agentId: string,
    task: string,
    options: { correlationId?: string; parentMessageId?: string; archetype?: SubAgentArchetype } = {}
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
    this.emit({ type: 'agent.run.started', agentId, data: { task, correlationId: options.correlationId } });
    this.emit({ type: 'agent.status.changed', agentId, data: { from, to: 'thinking' } });

    try {
      const grounding = await this.runGroundingCheck(agentId, task, options);
      this.emit({ type: 'agent.llm.called', agentId, data: { task } });
      await agent.step(this.buildGroundedTask(task, grounding));
      agent.setRuntimeState('done');

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

    const session = ctx.manager.getSession(ctx.sessionId);
    session?.messageQueue.clear('env');

    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('synthesizing');
    this.emit({ type: 'root.synthesis.started', agentId: 'root', data: { correlationId, subagentId: agent.identity.id } });
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'synthesizing' } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.synthesis', correlationId } });

    await ctx.agent.step(this.buildRootSynthesisPrompt(userTask, agent, subagentResult));

    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.toTokenUsage({
      llmCalls: usageAfter.llmCalls - usageBefore.llmCalls,
      promptTokens: usageAfter.promptTokens - usageBefore.promptTokens,
      completionTokens: usageAfter.completionTokens - usageBefore.completionTokens,
      totalTokens: usageAfter.totalTokens - usageBefore.totalTokens,
    });
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });

    const response = session ? await this.drainAgentOutput(session.messageQueue, ctx.agent.name) : ctx.agent.getInfo().lastResult ?? '';
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
    const toolCall = await this.enqueueMessage({
      kind: 'tool.call',
      sessionId: ctx.sessionId,
      from: agentId,
      to: 'tool.fs.list',
      correlationId: options.correlationId,
      parentMessageId: options.parentMessageId,
      payload: { path: process.cwd(), maxDepth: 2 },
      metadata: { agentId },
    });
    await this.processQueuedMessage(toolCall.id);

    try {
      const files = await this.listProjectFiles(process.cwd(), 2);
      const observedPaths = files.slice(0, 80);
      const toolResultSummary = observedPaths.join('\n');
      const record: ToolCallRecord = {
        toolName: 'fs.list',
        params: { path: process.cwd(), maxDepth: 2 },
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
      const warning = `Researcher produced a report without using project inspection tools: ${message}`;
      this.emit({ type: 'agent.grounding.warning', agentId, data: { warning, correlationId: options.correlationId } });
      return {
        toolCalls: [{
          toolName: 'fs.list',
          params: { path: process.cwd(), maxDepth: 2 },
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
