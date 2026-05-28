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

export type AgentState = 'idle' | 'running' | 'waiting' | 'stopped';
export type AgentRole = 'root' | 'subagent' | 'subteam';

export interface AgentUsage {
  llmCalls: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface AgentConfig {
  name: string;
  goal?: string;
  example?: string;
  llm?: LLMProvider;
  fsm?: FSM;
  role?: AgentRole;
  parentId?: string;
}

export interface AgentInfo {
  name: string;
  goal?: string;
  state: AgentState;
  role: AgentRole;
  parentId?: string;
  usage: AgentUsage;
}

/**
 * Base Agent class - all agents should extend this
 */
export abstract class BaseAgent {
  readonly name: string;
  protected goal: string;
  protected example: string;
  protected llm?: LLMProvider;
  protected fsm?: FSM;
  protected role: AgentRole;
  protected parentId?: string;
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
  protected actions: Map<string, Action> = new Map();
  protected tools: Map<string, Tool> = new Map();

  constructor(config: AgentConfig) {
    this.name = config.name;
    this.goal = config.goal || 'I am an AI agent.';
    this.example = config.example || '';
    this.llm = config.llm;
    this.fsm = config.fsm;
    this.role = config.role ?? 'root';
    this.parentId = config.parentId;
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
      role: this.role,
      parentId: this.parentId,
      usage: this.getUsage(),
    };
  }

  /**
   * Get current state
   */
  getState(): AgentState {
    return this.state;
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
      agent_goal: this.goal,
      agent_example: examplePart,
    });
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
    this.shortTermMemory = memoryRegistry.getShortTerm(this.name, sessionId);
    this.state = 'idle';
    logger.info(`Agent ${this.name} initialized for session ${sessionId}`);
  }

  /**
   * Cleanup the agent
   */
  async cleanup(): Promise<void> {
    this.state = 'stopped';
    logger.info(`Agent ${this.name} cleaned up`);
  }
}

export default BaseAgent;
