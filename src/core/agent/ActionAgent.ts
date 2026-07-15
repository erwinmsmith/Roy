// Action Agent - Agent that decides and executes actions using Planner

import type { LLMMessage } from '../llm/types.js';
import { BaseAgent, type AgentConfig } from './BaseAgent.js';
import type { Planner } from '../actions/Planner.js';
import { actionRegistry } from '../actions/index.js';
import { logger } from '../utils/logger.js';

export interface ActionAgentConfig extends Omit<AgentConfig, 'goal'> {
  actions: string; // Description of available actions
  goal?: string;
  planner?: Planner;
}

export interface ActionDecision {
  action: string;
  params: Record<string, unknown>;
}

/**
 * Action Agent - Uses Planner to decide and execute actions
 */
export class ActionAgent extends BaseAgent {
  private actionsDescription: string;
  private planner?: Planner;
  private route: Map<string, string[]> = new Map();

  constructor(config: ActionAgentConfig) {
    super({
      name: config.name,
      goal: config.goal,
      example: config.example,
      llm: config.llm,
      fsm: config.fsm,
    });
    this.actionsDescription = config.actions;
    this.planner = config.planner;
  }

  /**
   * Set route mapping (action -> receivers)
   */
  setRoute(action: string, receivers: string[]): void {
    this.route.set(action, receivers);
  }

  /**
   * Set planner
   */
  setPlanner(planner: Planner): void {
    this.planner = planner;
  }

  /**
   * Decide which action to take using LLM
   */
  private async decideWithLLM(observation: string): Promise<ActionDecision | null> {
    if (!this.llm) {
      logger.warn(`Agent ${this.name} has no LLM configured`);
      return null;
    }

    const messages: LLMMessage[] = [
      {
        role: 'system',
        content: `You are an action planner. Choose the best action based on the observation.

Available actions:
${this.actionsDescription}

Return a JSON object with:
- action: the name of the action
- params: the parameters for the action`,
      },
      { role: 'user', content: `Observation: ${observation}` },
    ];

    try {
      const response = await this.completeJSONWithAccounting<{
        action: string;
        params: Record<string, unknown>;
      }>(messages);

      return {
        action: response.action,
        params: response.params || {},
      };
    } catch (error) {
      logger.error(`Agent ${this.name} LLM decision error:`, error);
      return null;
    }
  }

  /**
   * Decide action using Planner
   */
  private async decideWithPlanner(observation: string): Promise<ActionDecision | null> {
    if (!this.planner) {
      return this.decideWithLLM(observation);
    }

    try {
      const plan = await this.planner.plan({
        agentInfo: {
          name: this.name,
          goal: this.goal,
          actions: this.actionsDescription,
        },
        observation,
        availableActions: actionRegistry.keys(),
      });

      if (plan) {
        return {
          action: plan.action,
          params: plan.params,
        };
      }
    } catch (error) {
      logger.error(`Agent ${this.name} planner error:`, error);
    }

    // Fallback to LLM
    return this.decideWithLLM(observation);
  }

  /**
   * Execute an action using actionRegistry
   */
  private async executeAction(
    action: string,
    params: Record<string, unknown>
  ): Promise<unknown> {
    // Try actionRegistry first
    if (actionRegistry.has(action)) {
      const result = await actionRegistry.execute(action, params);
      if (result.success) {
        return result.result;
      }
      throw new Error(result.error);
    }

    // Fallback to registered tools
    const tool = this.getTool(action);
    if (tool) {
      return await tool.execute(params);
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
    // Try actionRegistry first
    if (actionRegistry.has(action)) {
      yield* actionRegistry.executeStream(action, params);
      return;
    }

    // Fallback to registered tools
    const tool = this.getTool(action);
    if (tool) {
      if ('executeStream' in tool && typeof tool.executeStream === 'function') {
        const streamingTool = tool as {
          executeStream: (params: Record<string, unknown>) => AsyncGenerator<string, void, unknown>;
        };
        yield* streamingTool.executeStream(params);
        return;
      }

      const result = await tool.execute(params);
      yield String(result);
      return;
    }

    throw new Error(`Action "${action}" not found`);
  }

  /**
   * Process a single step
   */
  async step(observation: string): Promise<void> {
    this.state = 'thinking';
    this.addToMemory('observation', observation);

    // Decide action
    const decision = this.planner
      ? await this.decideWithPlanner(observation)
      : await this.decideWithLLM(observation);

    if (!decision) {
      this.state = 'idle';
      return;
    }

    const { action, params } = decision;
    const receivers = this.route.get(action) || ['env'];

    try {
      this.state = 'calling_tool';
      // Check if action is streaming
      const isStreaming =
        actionRegistry.has(action) ||
        (this.getTool(action)?.isStream ?? false);

      if (isStreaming) {
        let fullResult = '';
        for await (const chunk of this.executeActionStream(action, params)) {
          fullResult += chunk;

          if (this.messageQueue) {
            await this.messageQueue.send(this.name, receivers[0], chunk, {
              stream: true,
            });
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
      logger.info(`Agent ${this.name} completed action: ${action}`);
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

    while (this.state !== 'stopped') {
      try {
        const message = await this.messageQueue.receive(this.name);

        if (message) {
          await this.step(String(message.content));
        }

        await new Promise(resolve => setTimeout(resolve, 10));
      } catch (error) {
        logger.error(`ActionAgent ${this.name} run error:`, error);
        this.state = 'idle';
      }
    }

    logger.info(`ActionAgent ${this.name} stopped`);
  }
}

export default ActionAgent;
