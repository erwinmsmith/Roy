// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import { config } from '../../config/index.js';
import { logger } from '../utils/logger.js';
import { configureLogging, shutdownLogging } from '../logging/index.js';
import { llmFactory, type LLMProvider } from '../llm/index.js';
import { AgentManager } from './AgentManager.js';
import { FSM } from '../executor/FSM.js';
import { signalBus } from '../executor/SignalBus.js';
import { UnifiedAgent } from '../agent/UnifiedAgent.js';
import type { AgentInfo, AgentUsage } from '../agent/BaseAgent.js';
import { actionRegistry } from '../actions/index.js';
import { toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import {
  InMemoryMessageQueue,
  MessageScheduler,
  type EnqueueMessageInput,
  type MessageQueue,
  type QueueState,
  type QueueTransition,
  type RuntimeMessage,
} from '../queue/index.js';
import { WorkspaceMemoryManager, type WorkspaceMemoryState, type RootMemoryContext } from '../memory/index.js';

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
}

export class Runtime {
  private static instance: Runtime | null = null;

  private ctx: RuntimeContext | null = null;
  private initialized = false;
  private events: RuntimeEvent[] = [];
  private perTurnUsage: TokenUsage[] = [];
  private agentSequence = 0;
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
    const queue = new InMemoryMessageQueue(transition => this.handleQueueTransition(transition));
    const scheduler = new MessageScheduler(queue);

    // Create unified agent
    const agentName = options.agentName ?? 'Roy';
    const agentGoal = options.agentGoal ?? 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.';

    const agent = new UnifiedAgent({
      name: agentName,
      goal: agentGoal,
      llm: llm ?? undefined,  // Convert null to undefined for agent
      fsm: options.fsmEnabled !== false ? fsm : undefined,
      id: 'root',
      role: 'root',
      generation: 0,
      tomLevel: 1,
      description: 'Root agent of the Roy autonomous agent system',
      mode: options.mode ?? 'hybrid',
    });

    logger.info(`Agent created: ${agentName} in ${options.mode ?? 'hybrid'} mode`);

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

  async getMemoryState(): Promise<WorkspaceMemoryState> {
    const ctx = this.getContext();
    return ctx.memory.getState();
  }

  async loadRootMemoryContext(): Promise<RootMemoryContext> {
    const ctx = this.getContext();
    return ctx.memory.loadRootContext();
  }

  async spawnAgent(spec: SpawnAgentSpec): Promise<AgentInfo> {
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

    const goal = [
      spec.systemPrompt,
      `You are ${name}, a ${spec.archetype} subagent spawned by Roy.`,
      `Your parent agent is ${parentIdentity.name}.`,
      `Your scope: ${spec.description}`,
      spec.task ? `Initial task: ${spec.task}` : undefined,
    ].filter(Boolean).join('\n');

    const agent = new UnifiedAgent({
      id,
      name,
      role: 'subagent',
      parentId: spec.parentId,
      generation,
      tomLevel: spec.tomLevel,
      description: spec.description,
      goal,
      llm: ctx.llm ?? undefined,
      fsm,
      mode: 'hybrid',
    });

    this.registerCapabilities(agent);
    ctx.manager.addAgent(agent);
    await ctx.manager.attachAgentToSessions(agent);

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
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: id, data: { from: 'none', to: info.state } });
    if (spec.budgetTokens !== undefined) {
      this.emit({ type: 'budget.allocated', agentId: id, data: { budgetTokens: spec.budgetTokens } });
    }

    return info;
  }

  async runAgent(agentId: string, task: string): Promise<RunAgentResult> {
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
    this.emit({ type: 'agent.run.started', agentId, data: { task } });
    this.emit({ type: 'agent.status.changed', agentId, data: { from, to: 'thinking' } });

    try {
      this.emit({ type: 'agent.llm.called', agentId, data: { task } });
      await agent.step(task);
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
      this.emit({ type: 'agent.run.completed', agentId, data: { task, totalTokens: usageDelta.totalTokens } });

      return {
        agent: agent.getInfo(),
        result: result || agent.getInfo().lastResult || '',
        usage: usageDelta,
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

  private buildAgentTree(agent: AgentInfo): AgentTreeNode {
    return {
      agent,
      children: this.getChildren(agent.identity.id).map(child => this.buildAgentTree(child)),
    };
  }

  private createAgentId(archetype: SubAgentArchetype, sequence: number): string {
    return `agent_${archetype}_${String(sequence).padStart(3, '0')}`;
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
