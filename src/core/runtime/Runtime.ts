// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import { config } from '../../config/index.js';
import { logger } from '../utils/logger.js';
import { configureLogging, shutdownLogging } from '../logging/index.js';
import { llmFactory, type LLMProvider } from '../llm/index.js';
import { AgentManager } from './AgentManager.js';
import { FSM } from '../executor/FSM.js';
import { signalBus } from '../executor/SignalBus.js';
import { UnifiedAgent } from '../agent/UnifiedAgent.js';
import { actionRegistry } from '../actions/index.js';
import { toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';

export interface RuntimeConfig {
  agentName?: string;
  agentGoal?: string;
  sessionId?: string;
  fsmEnabled?: boolean;
  budget?: number;
  mode?: 'conversational' | 'action' | 'hybrid';
}

export interface RuntimeContext {
  config: typeof config;
  llm: LLMProvider | null;
  fsm: FSM;
  signalBus: typeof signalBus;
  manager: AgentManager;
  agent: UnifiedAgent;
  sessionId: string;
  capabilities: {
    skills: number;
    actions: number;
    tools: number;
  };
}

export class Runtime {
  private static instance: Runtime | null = null;

  private ctx: RuntimeContext | null = null;
  private initialized = false;

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
    const llm = this.createLLMProvider();
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
      },
      onStateChange: (state) => {
        logger.debug(`FSM state: ${state}`);
        signalBus.emit('fsm:stateChange', { state });
      },
    });

    if (options.budget !== undefined) {
      fsm.setBudget(options.budget);
    }

    // Create AgentManager
    const manager = new AgentManager();

    // Create unified agent
    const agentName = options.agentName ?? 'Roy';
    const agentGoal = options.agentGoal ?? 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.';

    const agent = new UnifiedAgent({
      name: agentName,
      goal: agentGoal,
      llm: llm ?? undefined,  // Convert null to undefined for agent
      fsm: options.fsmEnabled !== false ? fsm : undefined,
      role: 'root',
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
      capabilities,
    };

    this.initialized = true;
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
}

export const runtime = Runtime.getInstance();
export default Runtime;
