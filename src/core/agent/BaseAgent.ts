// Base Agent interface and implementation

import type { LLMProvider, LLMMessage } from '../llm/types.js';
import type { MessageQueue, QueueMessage } from '../message/MessageQueue.js';
import type { FSM, FSMContext } from '../executor/FSM.js';
import type { Skill } from '../../skills/types.js';
import type { Tool } from '../../tools/types.js';
import { memoryRegistry } from '../../memory/index.js';
import { buildPrompt } from '../../prompts/builder.js';
import { conversationalTemplate } from '../../prompts/templates/conversational.js';
import { logger } from '../utils/logger.js';

export type AgentState = 'idle' | 'running' | 'waiting' | 'stopped';

export interface AgentConfig {
  name: string;
  goal?: string;
  example?: string;
  llm?: LLMProvider;
  fsm?: FSM;
}

export interface AgentInfo {
  name: string;
  goal?: string;
  state: AgentState;
}

/**
 * Base Agent class - all agents should extend this
 */
export abstract class BaseAgent {
  readonly name: string;
  protected goal: string;
  protected example: string;
  protected llm: LLMProvider;
  protected fsm?: FSM;
  protected state: AgentState = 'idle';
  protected messageQueue?: MessageQueue;
  protected shortTermMemory!: ReturnType<typeof memoryRegistry.getShortTerm>;
  protected longTermMemory!: ReturnType<typeof memoryRegistry.getLongTerm>;
  protected skills: Map<string, Skill> = new Map();
  protected tools: Map<string, Tool> = new Map();

  constructor(config: AgentConfig) {
    this.name = config.name;
    this.goal = config.goal || 'I am an AI agent.';
    this.example = config.example || '';
    this.llm = config.llm!;
    this.fsm = config.fsm;
    this.shortTermMemory = memoryRegistry.getShortTerm(this.name, '');
    this.longTermMemory = memoryRegistry.getLongTerm(this.name);
  }

  /**
   * Get agent info
   */
  getInfo(): AgentInfo {
    return {
      name: this.name,
      goal: this.goal,
      state: this.state,
    };
  }

  /**
   * Get current state
   */
  getState(): AgentState {
    return this.state;
  }

  /**
   * Set message queue for communication
   */
  setMessageQueue(queue: MessageQueue): void {
    this.messageQueue = queue;
  }

  /**
   * Register a skill
   */
  registerSkill(skill: Skill): void {
    this.skills.set(skill.name, skill);
  }

  /**
   * Get a skill
   */
  getSkill(name: string): Skill | undefined {
    return this.skills.get(name);
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
   * Add to short-term memory
   */
  protected addToMemory(type: 'observation' | 'action' | 'result' | 'meta', content: string): void {
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