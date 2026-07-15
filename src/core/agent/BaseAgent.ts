// Base Agent interface and implementation

import type { LLMProvider, LLMMessage, LLMCompletionResult, LLMStreamChunk } from '../llm/types.js';
import type { MessageQueue } from '../message/MessageQueue.js';
import type { FSM } from '../executor/FSM.js';
import type { Action } from '../actions/Action.js';
import type { Tool } from '../tools/types.js';
import { memoryRegistry } from '../memory/index.js';
import { buildPrompt } from '../prompts/builder.js';
import { conversationalTemplate } from '../prompts/templates/conversational.js';
import { logger } from '../utils/logger.js';
import { normalizeToMProfile, type ToMProfile } from '../tom/index.js';

export type { ToMProfile } from '../tom/index.js';

export type AgentState = 'idle' | 'thinking' | 'calling_tool' | 'synthesizing' | 'waiting' | 'done' | 'failed' | 'stopped';
export type AgentRole = 'root' | 'subagent' | 'subteam';

export interface AgentIdentity {
  id: string;
  name: string;
  role: AgentRole;
  parentId?: string;
  teamId?: string;
  generation: number;
  tomLevel: number;
  description?: string;
  tomProfile: ToMProfile;
}

export interface AgentUsage {
  llmCalls: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface AgentConfig {
  id?: string;
  name: string;
  goal?: string;
  example?: string;
  llm?: LLMProvider;
  fsm?: FSM;
  role?: AgentRole;
  parentId?: string;
  teamId?: string;
  generation?: number;
  tomLevel?: number;
  description?: string;
  tomProfile?: ToMProfile;
}

export interface AgentInfo {
  name: string;
  goal?: string;
  state: AgentState;
  identity: AgentIdentity;
  role: AgentRole;
  parentId?: string;
  usage: AgentUsage;
  memoryMessages: number;
  createdAt: number;
  updatedAt: number;
  lastTask?: string;
  lastResult?: string;
  error?: string;
}

/**
 * Base Agent class - all agents should extend this
 */
export abstract class BaseAgent {
  readonly id: string;
  readonly name: string;
  protected goal: string;
  protected example: string;
  protected llm?: LLMProvider;
  protected fsm?: FSM;
  protected role: AgentRole;
  protected parentId?: string;
  protected teamId?: string;
  protected generation: number;
  protected tomLevel: number;
  protected description?: string;
  protected tomProfile: ToMProfile;
  protected state: AgentState = 'idle';
  protected usage: AgentUsage = {
    llmCalls: 0,
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };
  protected messageQueue?: MessageQueue;
  protected shortTermMemory!: ReturnType<typeof memoryRegistry.getShortTerm>;
  protected longTermMemory!: ReturnType<typeof memoryRegistry.getLongTerm>;
  private activeSessionId?: string;
  protected actions: Map<string, Action> = new Map();
  protected tools: Map<string, Tool> = new Map();
  protected createdAt: number = Date.now();
  protected updatedAt: number = Date.now();
  protected lastTask?: string;
  protected lastResult?: string;
  protected error?: string;

  constructor(config: AgentConfig) {
    this.id = config.id ?? (config.role === 'root' || !config.role ? 'root' : config.name);
    this.name = config.name;
    this.goal = config.goal || 'I am an AI agent.';
    this.example = config.example || '';
    this.llm = config.llm;
    this.fsm = config.fsm;
    this.role = config.role ?? 'root';
    this.parentId = config.parentId;
    this.teamId = config.teamId;
    this.generation = config.generation ?? (this.role === 'root' ? 0 : 1);
    this.tomLevel = config.tomLevel ?? (this.role === 'root' ? 1 : 1);
    this.description = config.description;
    const defaultProfile = this.createDefaultToMProfile();
    this.tomProfile = normalizeToMProfile(config.tomProfile, defaultProfile);
    this.tomLevel = this.tomProfile.level;
    this.shortTermMemory = memoryRegistry.getShortTerm(this.name, '');
    this.longTermMemory = memoryRegistry.getLongTerm(this.name);
  }

  /**
   * Check if agent has LLM configured
   */
  hasLLM(): boolean {
    return this.llm !== undefined && this.llm !== null;
  }

  /**
   * Get agent info
   */
  getInfo(): AgentInfo {
    return {
      name: this.name,
      goal: this.goal,
      state: this.state,
      identity: this.getIdentity(),
      role: this.role,
      parentId: this.parentId,
      usage: this.getUsage(),
      memoryMessages: this.shortTermMemory.get(1000).length,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      lastTask: this.lastTask,
      lastResult: this.lastResult,
      error: this.error,
    };
  }

  /**
   * Get stable identity metadata for runtime registries and UI.
   */
  getIdentity(): AgentIdentity {
    return {
      id: this.id,
      name: this.name,
      role: this.role,
      parentId: this.parentId,
      teamId: this.teamId,
      generation: this.generation,
      tomLevel: this.tomLevel,
      description: this.description,
      tomProfile: this.tomProfile,
    };
  }

  private createDefaultToMProfile(): ToMProfile {
    if (this.role === 'root') {
      return {
        level: 1,
        subjectAgentId: this.id,
        beliefScope: ['user intent', 'runtime state', 'delegation constraints'],
        goalModel: ['Answer the user reliably and delegate only when cognitive gaps justify it.'],
        uncertainty: [],
        perspective: 'root coordinator',
        observesAgents: [],
        modelsAgents: [],
        capabilityScope: ['reasoning', 'delegation', 'synthesis'],
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
      };
    }

    return {
      level: 0,
      subjectAgentId: this.id,
      beliefScope: [],
      goalModel: ['Complete the assigned bounded task.'],
      uncertainty: [],
      perspective: 'bounded specialist',
      observesAgents: this.parentId ? [this.parentId] : [],
      modelsAgents: [],
      capabilityScope: [],
      cognitiveGaps: [],
      models: [],
      purpose: this.description ?? 'Complete the assigned task within its local scope.',
    };
  }

  /**
   * Get current state
   */
  getState(): AgentState {
    return this.state;
  }

  /**
   * Update runtime-observable state from orchestration layers.
   */
  setRuntimeState(state: AgentState): void {
    this.state = state;
    this.updatedAt = Date.now();
  }

  /**
   * Get accumulated LLM usage for this agent.
   */
  getUsage(): AgentUsage {
    return { ...this.usage };
  }

  /**
   * Track LLM token usage reported by providers.
   */
  protected recordUsage(result: LLMCompletionResult | LLMStreamChunk): void {
    if (!result.usage) return;

    this.usage.llmCalls += 1;
    this.usage.promptTokens += result.usage.promptTokens;
    this.usage.completionTokens += result.usage.completionTokens;
    this.usage.totalTokens += result.usage.totalTokens;
    this.updatedAt = Date.now();
  }

  /**
   * Record a completion executed directly by the runtime orchestration layer.
   */
  recordRuntimeCompletion(content: string, result: LLMCompletionResult | LLMStreamChunk): void {
    this.recordUsage(result);
    this.addToMemory('result', content);
  }

  /**
   * Set message queue for communication
   */
  setMessageQueue(queue: MessageQueue): void {
    this.messageQueue = queue;
  }

  /**
   * Register an action
   */
  registerAction(action: Action): void {
    this.actions.set(action.name, action);
  }

  /**
   * Get an action
   */
  getAction(name: string): Action | undefined {
    return this.actions.get(name);
  }

  /**
   * Register a tool
   */
  registerTool(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  /**
   * Get a tool
   */
  getTool(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  /**
   * Build profile prompt
   */
  protected buildProfile(): string {
    const examplePart = this.example ? `\nExamples:\n${this.example}` : '';
    return buildPrompt(conversationalTemplate.template, {
      agent_identity: this.buildIdentityPrompt(),
      agent_goal: this.goal,
      agent_example: examplePart,
    });
  }

  protected buildIdentityPrompt(): string {
    if (this.role === 'root') {
      return [
        `You are ${this.name}, the root agent of a Theory-of-Mind based autonomous agent system.`,
        'You are not DeepSeek, Claude, OpenAI, Anthropic, or any model provider.',
        `Identity: id=${this.id}, role=${this.role}, generation=${this.generation}, ToM level=${this.tomLevel}.`,
      ].join('\n');
    }

    return [
      `You are ${this.name}, a ${this.role} in the Roy autonomous agent runtime.`,
      'You are not DeepSeek, Claude, OpenAI, Anthropic, or any model provider.',
      'The model provider is only your inference backend.',
      `Your parent agent is ${this.parentId ?? 'none'}.`,
      `Identity: id=${this.id}, parent=${this.parentId ?? 'none'}, generation=${this.generation}, ToM level=${this.tomLevel}.`,
    ].join('\n');
  }

  /**
   * Register multiple capabilities at once
   */
  registerCapabilities(capabilities: { actions?: Action[]; tools?: Tool[] }): void {
    if (capabilities.actions) {
      for (const action of capabilities.actions) {
        this.actions.set(action.name, action);
        logger.debug(`${this.name}: registered action ${action.name}`);
      }
    }
    if (capabilities.tools) {
      for (const tool of capabilities.tools) {
        this.tools.set(tool.name, tool);
        logger.debug(`${this.name}: registered tool ${tool.name}`);
      }
    }
  }

  /**
   * Get all registered capability names
   */
  getCapabilities(): { actions: string[]; tools: string[] } {
    return {
      actions: Array.from(this.actions.keys()),
      tools: Array.from(this.tools.keys()),
    };
  }

  /**
   * Get recent messages from memory as LLMMessage[]
   */
  getRecentMessages(count: number = 50): LLMMessage[] {
    const entries = this.shortTermMemory.get(count);
    return entries.map(entry => ({
      role: entry.type === 'action' ? 'assistant' as const : 'user' as const,
      content: entry.content,
    }));
  }

  /**
   * Execute FSM transition if configured
   */
  protected async maybeTransition(): Promise<boolean> {
    if (!this.fsm) return false;
    if (this.fsm.isTerminal()) return false;

    // Check budget
    const ctx = this.fsm.getContext();
    if (ctx.budget !== null && ctx.cost >= ctx.budget) {
      await this.fsm.transition('S_final');
      return true;
    }

    // Try to transition
    return await this.fsm.trigger();
  }

  /**
   * Get FSM state info for display
   */
  getFSMInfo(): { state: string; trace: string[]; budget: number | null; cost: number } | null {
    if (!this.fsm) return null;
    const ctx = this.fsm.getContext();
    return {
      state: this.fsm.getStateName(),
      trace: ctx.trace,
      budget: ctx.budget,
      cost: ctx.cost,
    };
  }

  /**
   * Add to short-term memory (public for external access)
   */
  addToMemory(type: 'observation' | 'action' | 'result' | 'meta', content: string): void {
    this.shortTermMemory.add({ type, content });
    this.updatedAt = Date.now();

    if (type === 'observation') {
      this.lastTask = content;
    }
    if (type === 'action' || type === 'result') {
      this.lastResult = content;
      this.error = content.startsWith('Error:') || content.startsWith('Action error:') ? content : undefined;
    }
  }

  /**
   * Get conversation history
   */
  protected getHistory(): LLMMessage[] {
    const entries = this.shortTermMemory.get(50);
    return entries.map(entry => ({
      role: entry.type === 'action' ? 'assistant' : 'user',
      content: entry.content,
    }));
  }

  /**
   * Main step - process observation and generate response
   */
  abstract step(observation: string): Promise<void>;

  /**
   * Run the agent (main loop)
   */
  abstract run(): Promise<void>;

  /**
   * Initialize the agent
   */
  async initialize(sessionId: string): Promise<void> {
    this.activeSessionId = sessionId;
    this.shortTermMemory = memoryRegistry.getShortTerm(this.name, sessionId);
    this.state = 'idle';
    this.updatedAt = Date.now();
    logger.info(`Agent ${this.name} initialized for session ${sessionId}`);
  }

  /**
   * Cleanup the agent
   */
  async cleanup(sessionId?: string): Promise<void> {
    const targetSessionId = sessionId ?? this.activeSessionId;
    if (targetSessionId) memoryRegistry.clearSession(this.name, targetSessionId);
    if (!sessionId || this.activeSessionId === targetSessionId) {
      this.state = 'stopped';
      this.activeSessionId = undefined;
    }
    this.updatedAt = Date.now();
    logger.info(`Agent ${this.name} cleaned up${targetSessionId ? ` for session ${targetSessionId}` : ''}`);
  }
}

export default BaseAgent;
