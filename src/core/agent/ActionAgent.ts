// Action Agent implementation

import type { LLMProvider, LLMMessage } from '../llm/types.js';
import type { MessageQueue } from '../message/MessageQueue.js';
import type { FSM } from '../executor/FSM.js';
import { BaseAgent, AgentConfig } from './BaseAgent.js';
import { buildPrompt } from '../../prompts/builder.js';
import { actionTemplate, actionCallTemplate } from '../../prompts/templates/action.js';
import { toolRegistry } from '../../tools/index.js';
import { logger } from '../utils/logger.js';

export interface ActionConfig {
  name: string;
  actions: string; // Description of available actions
  goal?: string;
  example?: string;
  llm?: LLMProvider;
  fsm?: FSM;
}

export interface ActionDecision {
  action: string;
  params: Record<string, unknown>;
}

export class ActionAgent extends BaseAgent {
  private actionsDescription: string;
  private route: Map<string, string[]> = new Map();

  constructor(config: ActionConfig) {
    super({
      name: config.name,
      goal: config.goal,
      example: config.example,
      llm: config.llm,
      fsm: config.fsm,
    });
    this.actionsDescription = config.actions;
  }

  /**
   * Set route mapping (action -> receivers)
   */
  setRoute(action: string, receivers: string[]): void {
    this.route.set(action, receivers);
  }

  /**
   * Decide which action to take
   */
  private async decide(observation: string): Promise<ActionDecision | null> {
    const systemPrompt = buildPrompt(actionTemplate.template, {
      agent_goal: this.goal,
      agent_actions: this.actionsDescription,
      agent_example: this.example,
    });

    const messages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Observation: ${observation}` },
    ];

    try {
      const response = await this.llm.completeJSON<{
        action: string;
        params: Record<string, unknown>;
      }>(messages);

      return {
        action: response.action,
        params: response.params || {},
      };
    } catch (error) {
      logger.error(`Agent ${this.name} decision error:`, error);
      return null;
    }
  }

  /**
   * Execute an action
   */
  private async executeAction(action: string, params: Record<string, unknown>): Promise<unknown> {
    const tool = toolRegistry.get(action);

    if (tool) {
      const result = await toolRegistry.execute(action, params);
      return result;
    }

    // Try registered tools
    const registeredTool = this.getTool(action);
    if (registeredTool) {
      return await registeredTool.execute(params);
    }

    throw new Error(`Action "${action}" not found`);
  }

  /**
   * Execute an action with streaming
   */
  private async *executeActionStream(
    action: string,
    params: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown> {
    const tool = toolRegistry.get(action);

    if (tool && 'executeStream' in tool && typeof tool.executeStream === 'function') {
      const streamingTool = tool as { executeStream: (params: Record<string, unknown>) => AsyncGenerator<string, void, unknown> };
      yield* streamingTool.executeStream(params);
    } else {
      const result = await this.executeAction(action, params);
      yield String(result);
    }
  }

  /**
   * Process a single step
   */
  async step(observation: string): Promise<void> {
    this.state = 'running';
    this.addToMemory('observation', observation);

    // Decide action
    const decision = await this.decide(observation);

    if (!decision) {
      this.state = 'idle';
      return;
    }

    const { action, params } = decision;
    const receivers = this.route.get(action) || ['env'];

    // Check if tool is streaming
    const tool = toolRegistry.get(action);
    const isStreaming = tool?.isStream || false;

    try {
      if (isStreaming) {
        let fullResult = '';
        for await (const chunk of this.executeActionStream(action, params)) {
          fullResult += chunk;

          if (this.messageQueue) {
            await this.messageQueue.send(this.name, receivers[0], chunk, { stream: true });
          }
        }

        this.addToMemory('action', `${action}: ${fullResult}`);
      } else {
        const result = await this.executeAction(action, params);

        if (this.messageQueue) {
          await this.messageQueue.send(this.name, receivers[0], result);
        }

        this.addToMemory('action', `${action}: ${JSON.stringify(result)}`);
      }

      // Add to FSM trace
      this.fsm?.addToTrace(`Agent ${this.name} executed ${action}`);
    } catch (error) {
      logger.error(`Agent ${this.name} action error:`, error);
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

    logger.info(`ActionAgent ${this.name} started running`);

    while (true) {
      try {
        const message = await this.messageQueue.receive(this.name);

        if (message) {
          await this.step(String(message.content));
        }

        await new Promise(resolve => setTimeout(resolve, 10));
      } catch (error) {
        logger.error(`ActionAgent ${this.name} run error:`, error);
      }
    }
  }
}

export default ActionAgent;