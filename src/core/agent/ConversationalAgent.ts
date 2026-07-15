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
   * Process a single step with FSM integration
   */
  async step(observation: string): Promise<void> {
    if (this.fsm?.isTerminal()) {
      this.fsm.reset();
    }

    this.state = 'thinking';
    this.addToMemory('observation', observation);

    // Check if LLM is available
    if (!this.llm) {
      logger.warn(`Agent ${this.name} has no LLM configured`);
      this.addToMemory('result', 'Error: LLM not configured');
      this.state = 'idle';
      return;
    }

    // Update FSM context based on observation
    if (this.fsm) {
      // Calculate metrics
      this.fsm.setUncertainty(this.calculateUncertainty(observation));
      this.fsm.setConflict(this.analyzeConflict(observation));
      this.fsm.setEvidence(this.analyzeEvidence(observation));

      // Pre-step transition
      await this.maybeTransition();

      // Add current state to trace
      this.fsm.addToTrace(`[${this.fsm.getStateName()}] Input: ${observation.substring(0, 50)}...`);
    }

    const history = this.getHistory();
    const systemPrompt = this.buildProfile();

    const messages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: observation },
    ];
    const tokensBefore = this.usage.totalTokens;

    try {
      let fullResponse = '';

      // Stream response
      const estimatedInputTokens = Math.ceil(
        messages.map(message => `${message.role}:${message.content}`).join('\n').length / 4
      );
      for await (const chunk of this.llm.stream(messages, this.completionOptions({}, estimatedInputTokens))) {
        if (chunk.usage) {
          this.recordUsage(chunk);
        }

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
      this.fsm?.addToTrace(`[${this.fsm.getStateName()}] Output: ${fullResponse.substring(0, 50)}...`);

      // Track cost
      if (this.fsm) {
        this.fsm.addCost(this.usage.totalTokens - tokensBefore);

        // Post-step transition
        await this.maybeTransition();
      }

      logger.info(`Agent ${this.name} completed step`);
    } catch (error) {
      logger.error(`Agent ${this.name} step error:`, error);
      this.addToMemory('result', `Error: ${error}`);
    }

    this.state = 'idle';
  }

  /**
   * Calculate uncertainty metric for FSM (0-1)
   */
  private calculateUncertainty(text: string): number {
    // Simple heuristic: longer text = more uncertainty
    const len = text.length;
    return Math.min(1.0, len / 2000);
  }

  /**
   * Analyze conflict in text (0-1)
   */
  private analyzeConflict(text: string): number {
    const conflictIndicators = ['but', 'however', 'although', 'conflict', 'disagree', 'contradict', 'versus'];
    const lowerText = text.toLowerCase();
    let conflicts = 0;
    for (const indicator of conflictIndicators) {
      if (lowerText.includes(indicator)) conflicts++;
    }
    return Math.min(1.0, conflicts * 0.25);
  }

  /**
   * Analyze evidence in text (0-1)
   */
  private analyzeEvidence(text: string): number {
    const evidenceIndicators = ['because', 'therefore', 'evidence', 'proves', 'shows', 'since', 'thus'];
    const lowerText = text.toLowerCase();
    let evidence = 0;
    for (const indicator of evidenceIndicators) {
      if (lowerText.includes(indicator)) evidence++;
    }
    return Math.min(1.0, evidence * 0.2);
  }

  /**
   * Run the agent main loop
   */
  async run(): Promise<void> {
    if (!this.messageQueue) {
      throw new Error('Message queue not set');
    }

    logger.info(`Agent ${this.name} started running`);

    while (this.state !== 'stopped') {
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
