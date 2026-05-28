// UnifiedAgent - Full-featured agent that integrates FSM, Actions, Tools, Prompts, Memory

import type { LLMProvider, LLMMessage } from '../llm/types.js';
import { BaseAgent, type AgentConfig } from './BaseAgent.js';
import { FSM, FSMState } from '../executor/FSM.js';
import type { Plan, Planner } from '../actions/Planner.js';
import { actionRegistry } from '../actions/index.js';
import { toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import { contextManager } from '../memory/context.js';
import { buildPrompt } from '../prompts/builder.js';
import {
  conversationalTemplate,
  fsmDiagnoseTemplate,
  fsmDecideTemplate,
  fsmDeriveTemplate,
  fsmVerifyTemplate,
  actionTemplate,
} from '../prompts/index.js';
import { logger } from '../utils/logger.js';

export type AgentMode = 'conversational' | 'action' | 'hybrid';

export interface UnifiedAgentConfig extends AgentConfig {
  mode?: AgentMode;
  planner?: Planner;
  useContextManager?: boolean;
}

export class UnifiedAgent extends BaseAgent {
  private mode: AgentMode;
  private planner?: Planner;
  private useContextManager: boolean;
  private sessionId: string = '';

  constructor(config: UnifiedAgentConfig) {
    super({
      name: config.name,
      goal: config.goal,
      example: config.example,
      llm: config.llm,
      fsm: config.fsm,
    });
    this.mode = config.mode ?? 'hybrid';
    this.planner = config.planner;
    this.useContextManager = config.useContextManager ?? true;
  }

  /**
   * Main step - orchestrates FSM, prompts, planning, and execution
   */
  async step(observation: string): Promise<void> {
    this.state = 'running';
    this.addToMemory('observation', observation);

    if (!this.llm) {
      const errorMsg = 'Error: LLM not configured';
      logger.warn(`Agent ${this.name} has no LLM configured`);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
      this.state = 'idle';
      return;
    }

    // === FSM Integration ===
    await this.updateFSMContext(observation);

    // Pre-step transition
    await this.maybeTransition();

    if (this.fsm?.isTerminal()) {
      logger.info(`Agent ${this.name} reached terminal FSM state`);
      this.addToMemory('meta', 'FSM reached terminal state');
      this.state = 'idle';
      return;
    }

    // === FSM State-Based Prompt Selection ===
    const systemPrompt = this.buildFSMPrompt(observation);
    const messages = this.buildMessages(systemPrompt, observation);

    // === Determine Action Mode ===
    const shouldAct = await this.shouldExecuteAction(observation);

    if (shouldAct && this.mode !== 'conversational') {
      await this.executeActionMode(observation, messages);
    } else {
      await this.executeConversationalMode(messages);
    }

    // Track cost
    if (this.fsm) {
      this.fsm.addCost(1);
      this.fsm.addToTrace(`[${this.fsm.getStateName()}] Output complete`);
      await this.maybeTransition();
    }

    this.state = 'idle';
  }

  /**
   * Update FSM context based on observation
   */
  private async updateFSMContext(observation: string): Promise<void> {
    if (!this.fsm) return;

    this.fsm.setUncertainty(this.calculateUncertainty(observation));
    this.fsm.setConflict(this.analyzeConflict(observation));
    this.fsm.setEvidence(this.analyzeEvidence(observation));

    if (this.useContextManager && this.sessionId) {
      const context = contextManager.get(this.name, this.sessionId);
      if (context) {
        const lines = context.content.split('\n').length;
        this.fsm.setUncertainty(Math.min(1, this.fsm.getContext().uncertainty + lines / 100));
      }
    }
  }

  /**
   * Build prompt based on FSM state
   */
  private buildFSMPrompt(currentObservation: string): string {
    const baseGoal = this.goal || 'You are a helpful assistant.';
    const examplePart = this.example ? `\nExamples:\n${this.example}` : '';

    if (!this.fsm) {
      return buildPrompt(conversationalTemplate.template, {
        agent_goal: baseGoal,
        agent_example: examplePart,
      });
    }

    const state = this.fsm.getState();
    const ctx = this.fsm.getContext();

    switch (state) {
      case 'S_solo':
        return buildPrompt(conversationalTemplate.template, {
          agent_goal: baseGoal,
          agent_example: examplePart,
        });

      case 'S_diagnose':
        return buildPrompt(fsmDiagnoseTemplate.template, {
          trace: ctx.trace.join('\n') || 'No trace yet',
          state,
        });

      case 'S_decide': {
        const candidates = this.formatCapabilitiesForPrompt();
        return buildPrompt(fsmDecideTemplate.template, {
          current_state: state,
          budget: String(ctx.budget - ctx.cost),
          candidates: candidates || 'No candidate actions available',
          uncertainty: String(ctx.uncertainty),
          conflict: String(ctx.conflict),
        });
      }

      case 'S_derive':
        return buildPrompt(fsmDeriveTemplate.template, {
          parent_unit: this.name,
          trace: ctx.trace.join('\n'),
          budget: String(ctx.budget - ctx.cost),
          state,
        });

      case 'S_execute': {
        return buildPrompt(actionTemplate.template, {
          agent_goal: baseGoal,
          agent_actions: this.formatCapabilitiesForPrompt(),
          agent_example: examplePart,
        });
      }

      case 'S_verify': {
        const history = this.getHistory();
        const lastMsg = history[history.length - 1];
        return buildPrompt(fsmVerifyTemplate.template, {
          question: currentObservation,
          answer: lastMsg?.content ?? '',
          trace: ctx.trace.join('\n'),
          state,
        });
      }

      case 'S_final':
        return buildPrompt(conversationalTemplate.template, {
          agent_goal: baseGoal,
          agent_example: 'Provide a clear, final answer summarizing all reasoning.',
        });

      case 'S_backtrack':
        return buildPrompt(fsmDeriveTemplate.template, {
          parent_unit: this.name,
          trace: ctx.trace.join('\n'),
          budget: String(ctx.budget - ctx.cost),
          state: 'S_backtrack',
        });

      default:
        return buildPrompt(conversationalTemplate.template, {
          agent_goal: baseGoal,
          agent_example: examplePart,
        });
    }
  }

  /**
   * Build messages for LLM
   */
  private buildMessages(systemPrompt: string, observation: string): LLMMessage[] {
    const history = this.getHistory();
    return [
      { role: 'system', content: systemPrompt },
      ...history,
      { role: 'user', content: observation },
    ];
  }

  /**
   * Decide whether to execute an action
   */
  private async shouldExecuteAction(observation: string): Promise<boolean> {
    const actionIndicators = [
      'do', 'run', 'execute', 'perform', 'search', 'find',
      'calculate', 'create', 'update', 'delete', 'fetch', 'get me',
    ];
    const lowerObs = observation.toLowerCase();

    const hasActionIndicator = actionIndicators.some(ind => lowerObs.includes(ind));
    const hasActions = actionRegistry.list().length > 0;
    const hasTools = toolRegistry.list().length > 0;
    const hasSkills = skillRegistry.list().length > 0;

    return (hasActionIndicator && (hasActions || hasTools || hasSkills)) || this.mode === 'action';
  }

  /**
   * Execute in action mode with planner
   */
  private async executeActionMode(
    observation: string,
    messages: LLMMessage[]
  ): Promise<void> {
    let plan: Plan | null = null;

    if (this.planner) {
      try {
        plan = await this.planner.plan({
          agentInfo: {
            name: this.name,
            goal: this.goal,
            actions: this.formatCapabilitiesForPrompt(),
          },
          observation,
          availableActions: this.getCapabilityNames(),
          history: messages.slice(0, -1),
        });
      } catch (error) {
        logger.error('Planner error:', error);
      }
    }

    if (!plan) {
      plan = await this.decideActionWithLLM(observation, messages);
    }

    if (!plan) {
      await this.executeConversationalMode(messages);
      return;
    }

    try {
      const result = await this.executeCapability(plan.action, plan.params);

      if (result.success) {
        const response = String(result.result ?? 'Action completed successfully');
        this.addToMemory('action', `${plan.action}: ${response}`);
        this.fsm?.addToTrace(`Action executed: ${plan.action}`);

        if (this.messageQueue) {
          await this.messageQueue.send(this.name, 'env', response, { done: true });
        } else {
          process.stdout.write(response);
        }
      } else {
        const errorMsg = `Action error: ${result.error}`;
        this.addToMemory('result', errorMsg);
        if (this.messageQueue) {
          await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
        } else {
          process.stdout.write(errorMsg);
        }
      }
    } catch (error) {
      const errorMsg = `Error: ${error}`;
      logger.error(`Action execution error:`, error);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
    }
  }

  /**
   * Decide action using LLM
   */
  private async decideActionWithLLM(
    observation: string,
    messages: LLMMessage[]
  ): Promise<Plan | null> {
    const prompt = `Based on the observation, choose the best action to execute.

Available capabilities:
${this.formatCapabilitiesForPrompt()}

Observation: ${observation}

Return a JSON object with:
- action: the name of the action, tool, or skill to execute (or "none" if no action needed)
- params: parameters for the action (or empty object)
- reasoning: why you chose this action`;

    const decisionMessages: LLMMessage[] = [
      ...messages.slice(0, 1),
      { role: 'user', content: prompt },
    ];

    try {
      const response = await this.llm!.completeJSON<{
        action: string;
        params: Record<string, unknown>;
        reasoning?: string;
      }>(decisionMessages);

      if (response.action === 'none' || !response.action) {
        return null;
      }

      return {
        action: response.action,
        params: response.params || {},
        reasoning: response.reasoning,
        confidence: 0.8,
      };
    } catch (error) {
      logger.error('LLM action decision error:', error);
      return null;
    }
  }

  /**
   * Execute an action, tool, or skill by name.
   */
  private async executeCapability(
    name: string,
    params: Record<string, unknown>
  ): Promise<{ success: boolean; result?: unknown; error?: string; metadata?: Record<string, unknown> }> {
    if (actionRegistry.has(name)) {
      return actionRegistry.execute(name, params);
    }

    if (toolRegistry.has(name)) {
      return toolRegistry.execute(name, params);
    }

    if (skillRegistry.has(name)) {
      const validation = skillRegistry.get(name)?.validate?.({ action: name, params });
      if (validation && !validation.valid) {
        return {
          success: false,
          error: `Validation failed: ${validation.errors?.join(', ')}`,
        };
      }

      return skillRegistry.execute(
        name,
        { action: name, params },
        {
          agentId: this.name,
          sessionId: this.sessionId,
          variables: {
            mode: this.mode,
            fsmState: this.fsm?.getStateName(),
          },
        }
      );
    }

    return {
      success: false,
      error: `Capability "${name}" not found`,
    };
  }

  /**
   * List all executable capability names.
   */
  private getCapabilityNames(): string[] {
    return [
      ...actionRegistry.keys(),
      ...toolRegistry.keys(),
      ...skillRegistry.keys(),
    ];
  }

  /**
   * Format registered capabilities for planner and LLM prompts.
   */
  private formatCapabilitiesForPrompt(): string {
    const lines: string[] = [];

    const actions = actionRegistry.formatActionsForPrompt();
    if (actions) {
      lines.push('Actions:', actions);
    }

    const tools = toolRegistry.list();
    if (tools.length > 0) {
      lines.push(
        'Tools:',
        ...tools.map(tool => `- ${tool.name}: ${tool.description || 'No description'}`)
      );
    }

    const skills = skillRegistry.listManifests();
    if (skills.length > 0) {
      lines.push(
        'Skills:',
        ...skills.map(skill => {
          const tags = skill.tags.length > 0 ? ` [${skill.tags.join(', ')}]` : '';
          return `- ${skill.name}: ${skill.description}${tags}`;
        })
      );
    }

    return lines.join('\n') || 'No executable capabilities registered';
  }

  /**
   * Execute in conversational mode
   */
  private async executeConversationalMode(messages: LLMMessage[]): Promise<void> {
    try {
      let fullResponse = '';

      for await (const chunk of this.llm!.stream(messages)) {
        fullResponse += chunk.content;

        if (this.messageQueue) {
          await this.messageQueue.send(
            this.name,
            'env',
            chunk.content,
            { stream: !chunk.done, done: chunk.done }
          );
        } else {
          process.stdout.write(chunk.content);
        }
      }

      this.addToMemory('result', fullResponse);

      if (this.fsm) {
        this.fsm.addToTrace(`Response: ${fullResponse.substring(0, 50)}...`);
      }
    } catch (error) {
      const errorMsg = `Error: ${error}`;
      logger.error(`Agent ${this.name} error:`, error);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
    }
  }

  /**
   * Calculate uncertainty metric (0-1)
   */
  private calculateUncertainty(text: string): number {
    const len = text.length;
    const questionCount = (text.match(/\?/g) || []).length;
    const complexWords = (text.match(/\b(however|although|therefore|because|perhaps|might|may|could)\b/gi) || []).length;
    return Math.min(1.0, (len / 2000) + (questionCount * 0.15) + (complexWords * 0.1));
  }

  /**
   * Analyze conflict in text (0-1)
   */
  private analyzeConflict(text: string): number {
    const conflictIndicators = ['but', 'however', 'although', 'conflict', 'disagree', 'contradict', 'versus', 'or', 'vs'];
    const lowerText = text.toLowerCase();
    let conflicts = 0;
    for (const indicator of conflictIndicators) {
      const regex = new RegExp(`\\b${indicator}\\b`, 'gi');
      const matches = lowerText.match(regex);
      if (matches) conflicts += matches.length;
    }
    return Math.min(1.0, conflicts * 0.15);
  }

  /**
   * Analyze evidence in text (0-1)
   */
  private analyzeEvidence(text: string): number {
    const evidenceIndicators = [
      'because', 'therefore', 'evidence', 'proves', 'shows',
      'since', 'thus', 'according to', 'data shows', 'research shows',
      'studies show', 'resulting in', 'leads to',
    ];
    const lowerText = text.toLowerCase();
    let evidence = 0;
    for (const indicator of evidenceIndicators) {
      if (lowerText.includes(indicator)) evidence++;
    }
    return Math.min(1.0, evidence * 0.15);
  }

  /**
   * Initialize with session ID
   */
  async initialize(sessionId: string): Promise<void> {
    this.sessionId = sessionId;
    await super.initialize(sessionId);

    if (this.useContextManager) {
      contextManager.upsert(this.name, sessionId, `# Session: ${sessionId}\n\n`);
    }
  }

  /**
   * Cleanup
   */
  async cleanup(): Promise<void> {
    if (this.sessionId && this.useContextManager) {
      contextManager.delete(this.name, this.sessionId);
    }
    await super.cleanup();
  }

  /**
   * Run the agent main loop
   */
  async run(): Promise<void> {
    if (!this.messageQueue) {
      throw new Error('Message queue not set');
    }

    logger.info(`Agent ${this.name} started running in ${this.mode} mode`);

    while (this.state !== 'stopped') {
      try {
        const message = await this.messageQueue.receive(this.name);

        if (message) {
          await this.step(String(message.content));
        }

        await new Promise(resolve => setTimeout(resolve, 10));
      } catch (error) {
        logger.error(`Agent ${this.name} run error:`, error);
        this.state = 'idle';
      }
    }

    logger.info(`Agent ${this.name} stopped`);
  }

  /**
   * Get context from ContextManager
   */
  getContext(): string {
    if (!this.sessionId || !this.useContextManager) {
      return '';
    }
    const doc = contextManager.get(this.name, this.sessionId);
    return doc?.content ?? '';
  }
}

export default UnifiedAgent;
