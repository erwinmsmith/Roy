// Conversational Agent implementation

import type { LLMMessage } from '../llm/types.js';
import { BaseAgent } from './BaseAgent.js';
import type { AgentConfig } from './BaseAgent.js';
import { logger } from '../utils/logger.js';

export interface ConversationalAgentConfig extends AgentConfig {
  dbName?: string;
}

export class ConversationalAgent extends BaseAgent {
  private dbName: string;
  private sid: string = '';

  constructor(config: ConversationalAgentConfig) {
    super({
      name: config.name,
      goal: config.goal,
      example: config.example,
      llm: config.llm,
      fsm: config.fsm,
    });
    this.dbName = config.dbName || `${config.name}_db`;
  }

  /**
   * Process a single step
   */
  async step(observation: string): Promise<void> {
    this.state = 'running';
    this.addToMemory('observation', observation);

    const history = this.getHistory();
    const systemPrompt = this.buildProfile();

    const messages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: observation },
    ];

    try {
      let fullResponse = '';

      // Stream response
      for await (const chunk of this.llm.stream(messages)) {
        fullResponse += chunk.content;

        // Send to message queue if available
        if (this.messageQueue) {
          await this.messageQueue.send(
            this.name,
            'env',
            chunk.content,
            { stream: !chunk.done }
          );
        }
      }

      // Add response to memory
      this.addToMemory('result', fullResponse);

      // Add to FSM trace
      this.fsm?.addToTrace(`Agent ${this.name}: ${observation.substring(0, 50)}...`);

      logger.info(`Agent ${this.name} completed step`);
    } catch (error) {
      logger.error(`Agent ${this.name} step error:`, error);
      this.addToMemory('result', `Error: ${error}`);
    }

    this.state = 'idle';
  }

  /**
   * Run the agent main loop
   */
  async run(): Promise<void> {
    if (!this.messageQueue) {
      throw new Error('Message queue not set');
    }

    logger.info(`Agent ${this.name} started running`);

    while (true) {
      try {
        // Receive message from queue
        const message = await this.messageQueue.receive(this.name);

        if (message) {
          const observation = String(message.content);
          await this.step(observation);
        }

        // Small delay to prevent tight loop
        await new Promise(resolve => setTimeout(resolve, 10));
      } catch (error) {
        logger.error(`Agent ${this.name} run error:`, error);
        this.state = 'idle';
      }
    }

    logger.info(`Agent ${this.name} stopped`);
  }

  /**
   * Initialize with session ID
   */
  async initialize(sessionId: string): Promise<void> {
    this.sid = sessionId;
    await super.initialize(sessionId);
  }
}

export default ConversationalAgent;