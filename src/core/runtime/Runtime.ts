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
  toolCalls: ToolCallRecord[];
  grounded: boolean;
  warnings: string[];
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
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
}

export interface RootMediatedSpawnResult {
  correlationId: string;
  agent: AgentInfo;
  subagentResult: RunAgentResult;
  finalResponse: string;
  messages: RuntimeMessage[];
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
      tomProfile: this.createRootToMProfile(),
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

  async handleSpawnCommand(payload: SpawnCommandPayload): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const correlationId = this.createCorrelationId();
    const parentId = payload.parentId ?? 'root';
    const requireRootSynthesis = payload.requireRootSynthesis ?? true;

    const command = await this.enqueueMessage({
      kind: 'user.command.spawn',
      sessionId: ctx.sessionId,
      from: 'cli',
      to: 'root',
      correlationId,
      payload,
      metadata: { agentId: 'root' },
    });
    await this.processQueuedMessage(command.id);
    await ctx.queue.ack(command.id);

    const tomProfile = this.createSubagentToMProfile(payload.archetype, '', payload.task);
    const agent = await this.spawnAgent({
      parentId,
      name: payload.name,
      archetype: payload.archetype,
      tomLevel: tomProfile.level,
      description: payload.task,
      task: payload.task,
      systemPrompt: undefined,
      tomProfile,
    });

    const taskMessage = await this.enqueueMessage({
      kind: 'agent.task',
      sessionId: ctx.sessionId,
      from: 'root',
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
      to: 'root',
      correlationId,
      parentMessageId: taskMessage.id,
      payload: subagentResult,
      metadata: {
        agentId: agent.identity.id,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(resultMessage.id);
    await ctx.queue.ack(resultMessage.id);
    this.emit({ type: 'agent.result.sent', agentId: agent.identity.id, data: { correlationId, to: 'root' } });

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
    await this.processQueuedMessage(finalMessage.id);
    await ctx.queue.ack(finalMessage.id);

    return {
      correlationId,
      agent,
      subagentResult,
      finalResponse,
      messages: await this.getMessages({ correlationId }),
    };
  }

  async spawnAgent(spec: SpawnAgentSpec & { tomProfile?: ToMProfile }): Promise<AgentInfo> {
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
      this.emit({
        type: 'agent.run.completed',
        agentId,
        data: {
          task,
          totalTokens: usageDelta.totalTokens,
          grounded: grounding.grounded,
          warnings: grounding.warnings,
        },
      });

      return {
        agent: agent.getInfo(),
        result: result || agent.getInfo().lastResult || '',
        usage: usageDelta,
        toolCalls: grounding.toolCalls,
        grounded: grounding.grounded,
        warnings: grounding.warnings,
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
- tool calls: ${subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
- warnings:
${warnings}

Produce the final response to the user as Roy, the root agent. Do not claim you personally inspected files unless the report is grounded. Mention limitations if the report is ungrounded.`;
  }

  private async runGroundingCheck(
    agentId: string,
    task: string,
    options: { correlationId?: string; parentMessageId?: string; archetype?: SubAgentArchetype }
  ): Promise<{ toolCalls: ToolCallRecord[]; grounded: boolean; warnings: string[]; context: string }> {
    const required = this.requiresProjectInspection(options.archetype, task);
    if (!required) {
      return { toolCalls: [], grounded: true, warnings: [], context: '' };
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
