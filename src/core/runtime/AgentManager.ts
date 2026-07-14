// Agent Manager - manages agents and sessions

import type { BaseAgent } from '../agent/BaseAgent.js';
import { MessageQueue } from '../message/MessageQueue.js';
import type { LLMProvider } from '../llm/types.js';
import { llmFactory } from '../llm/index.js';
import { logger } from '../utils/logger.js';

interface Session {
  id: string;
  agents: BaseAgent[];
  messageQueue: MessageQueue;
}

export class AgentManager {
  private agents: Map<string, BaseAgent> = new Map();
  private agentIds: Map<string, string> = new Map();
  private sessions: Map<string, Session> = new Map();
  private defaultLlm?: LLMProvider;
  private interactWithEnv: string | null = null;

  constructor() {
    // Get default LLM provider
    try {
      this.defaultLlm = llmFactory.getDefault();
    } catch {
      logger.warn('No LLM providers configured');
    }
  }

  /**
   * Add an agent to the manager
   */
  addAgent(agent: BaseAgent): void {
    if (this.agents.has(agent.name)) {
      logger.warn(`Agent ${agent.name} already exists, overwriting`);
    }
    this.agents.set(agent.name, agent);
    this.agentIds.set(agent.id, agent.name);
  }

  /**
   * Remove an agent
   */
  removeAgent(name: string): boolean {
    const agent = this.agents.get(name);
    if (!agent) return false;

    for (const session of this.sessions.values()) {
      session.messageQueue.removeReceiver(agent.name);
      session.agents = session.agents.filter(item => item.name !== agent.name);
    }

    this.agentIds.delete(agent.id);
    return this.agents.delete(name);
  }

  /**
   * Get an agent
   */
  getAgent(name: string): BaseAgent | undefined {
    return this.agents.get(name);
  }

  /**
   * Get an agent by stable identity id.
   */
  getAgentById(id: string): BaseAgent | undefined {
    const name = this.agentIds.get(id);
    return name ? this.agents.get(name) : undefined;
  }

  /**
   * Attach an agent to all existing sessions.
   */
  async attachAgentToSessions(agent: BaseAgent): Promise<void> {
    for (const session of this.sessions.values()) {
      session.messageQueue.addReceiver(agent.name);
      agent.setMessageQueue(session.messageQueue);
      await agent.initialize(session.id);
      if (!session.agents.some(item => item.name === agent.name)) {
        session.agents.push(agent);
      }
    }
  }

  /**
   * Set which agent interacts with environment
   */
  setInteractWithEnv(agentName: string): void {
    this.interactWithEnv = agentName;
  }

  /**
   * Create a new session
   */
  createSession(sessionId: string): Session {
    if (this.sessions.has(sessionId)) {
      return this.sessions.get(sessionId)!;
    }

    const receivers = ['env', ...this.agents.keys()];
    const messageQueue = new MessageQueue(receivers);

    const session: Session = {
      id: sessionId,
      agents: [],
      messageQueue,
    };

    // Initialize all agents with this session
    for (const [, agent] of this.agents) {
      agent.setMessageQueue(messageQueue);
      agent.initialize(sessionId).catch(err => {
        logger.error(`Failed to initialize agent ${agent.name}:`, err);
      });
      session.agents.push(agent);
    }

    this.sessions.set(sessionId, session);
    logger.info(`Session ${sessionId} created`);

    return session;
  }

  /**
   * Get a session
   */
  getSession(sessionId: string): Session | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Close a session
   */
  async closeSession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    // Cleanup agents
    for (const agent of session.agents) {
      await agent.cleanup();
    }

    // Cleanup message queue
    await session.messageQueue.cleanup();

    this.sessions.delete(sessionId);
    logger.info(`Session ${sessionId} closed`);
  }

  /**
   * Send message to environment
   */
  async sendToEnv(sessionId: string, message: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    if (!this.interactWithEnv) {
      throw new Error('No agent configured to interact with env');
    }

    await session.messageQueue.send('env', this.interactWithEnv, message);
  }

  /**
   * Stream response from agent
   */
  async *streamResponse(
    sessionId: string,
    message: string
  ): AsyncGenerator<string, void, unknown> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    if (!this.interactWithEnv) {
      throw new Error('No agent configured to interact with env');
    }

    // Send message to env
    await session.messageQueue.send('env', this.interactWithEnv, message);

    // Stream messages from agent
    const agent = this.agents.get(this.interactWithEnv);
    if (!agent) {
      throw new Error(`Agent ${this.interactWithEnv} not found`);
    }

    // Agent responses are emitted to the environment receiver through the session queue.
    let done = false;
    while (!done) {
      const msg = await session.messageQueue.receive('env');
      if (msg) {
        yield String(msg.content);
        done = msg.metadata?.done === true;
      }
      // Small delay to prevent tight loop
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }

  /**
   * List all sessions
   */
  listSessions(): string[] {
    return Array.from(this.sessions.keys());
  }

  /**
   * List all agents
   */
  listAgents(): string[] {
    return Array.from(this.agents.keys());
  }

  /**
   * List agent state and usage snapshots.
   */
  listAgentInfo(): ReturnType<BaseAgent['getInfo']>[] {
    return Array.from(this.agents.values()).map(agent => agent.getInfo());
  }

  /**
   * Cleanup manager
   */
  async cleanup(): Promise<void> {
    for (const sessionId of this.sessions.keys()) {
      await this.closeSession(sessionId);
    }
    this.agents.clear();
    this.agentIds.clear();
    logger.info('AgentManager cleaned up');
  }
}

export default AgentManager;
