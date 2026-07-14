// Planner interface - decision making for agents

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  complete(messages: LLMMessage[], options?: unknown): Promise<unknown>;
  stream(messages: LLMMessage[], options?: unknown): AsyncGenerator<{ content: string; done: boolean }, void, unknown>;
  completeJSON<T>(messages: LLMMessage[], options?: unknown): Promise<T>;
  isConfigured(): boolean;
}

export interface Plan {
  action: string;
  params: Record<string, unknown>;
  reasoning?: string;
  confidence?: number;
}

export interface PlanContext {
  agentInfo: AgentInfo;
  observation: string;
  availableActions: string[];
  history?: LLMMessage[];
}

export interface AgentInfo {
  name: string;
  goal?: string;
  actions?: string;
  example?: string;
}

export interface PlannerConfig {
  name: string;
  llm: LLMProvider;
  systemPrompt?: string;
  temperature?: number;
}

/**
 * Base interface for planners
 * A planner decides which action to take based on the current observation
 */
export interface Planner {
  readonly name: string;

  /**
   * Plan the next action based on observation
   */
  plan(context: PlanContext): Promise<Plan | null>;

  /**
   * Plan with streaming output
   */
  planStream(
    context: PlanContext
  ): AsyncGenerator<string, Plan | null, unknown>;
}

/**
 * Simple LLM-based planner
 */
export class LLMPlanner implements Planner {
  readonly name: string;
  private llm: LLMProvider;
  private systemPrompt: string;
  private temperature: number;

  constructor(config: PlannerConfig) {
    this.name = config.name;
    this.llm = config.llm;
    this.systemPrompt = config.systemPrompt || this.defaultSystemPrompt();
    this.temperature = config.temperature ?? 0.7;
  }

  private defaultSystemPrompt(): string {
    return `You are a planner that decides which action to take.
Given an observation and available actions, choose the best action.
Return a JSON object with:
- action: the name of the action to take
- params: the parameters for the action
- reasoning: why you chose this action
- confidence: how confident you are (0-1)`;
  }

  async plan(context: PlanContext): Promise<Plan | null> {
    const messages: LLMMessage[] = [
      { role: 'system', content: this.systemPrompt },
    ];

    if (context.history && context.history.length > 0) {
      messages.push(...context.history);
    }

    const availableActions = context.availableActions.join(', ') || 'none';
    const userContent = `Observation: ${context.observation}
Available actions: ${availableActions}
${context.agentInfo.goal ? `Agent goal: ${context.agentInfo.goal}` : ''}

Return a JSON object with the action and parameters.`;

    messages.push({ role: 'user', content: userContent });

    try {
      const response = await this.llm.completeJSON<Plan>(messages, {
        temperature: this.temperature,
      });
      return this.normalizePlan(response, context.availableActions);
    } catch (error) {
      console.error(`Planner ${this.name} error:`, error);
      return null;
    }
  }

  async *planStream(
    context: PlanContext
  ): AsyncGenerator<string, Plan | null, unknown> {
    const messages: LLMMessage[] = [
      { role: 'system', content: this.systemPrompt },
    ];

    if (context.history && context.history.length > 0) {
      messages.push(...context.history);
    }

    const availableActions = context.availableActions.join(', ') || 'none';
    const userContent = `Observation: ${context.observation}
Available actions: ${availableActions}
${context.agentInfo.goal ? `Agent goal: ${context.agentInfo.goal}` : ''}

Return a JSON object with the action and parameters.`;

    messages.push({ role: 'user', content: userContent });

    let fullResponse = '';
    let lastPlan: Plan | null = null;

    try {
      for await (const chunk of this.llm.stream(messages, {
        temperature: this.temperature,
      })) {
        fullResponse += chunk.content;
        yield chunk.content;

        if (chunk.done && fullResponse) {
          try {
            lastPlan = this.normalizePlan(this.parsePlan(fullResponse), context.availableActions);
          } catch {
            // Not valid JSON yet
          }
        }
      }
    } catch (error) {
      console.error(`Planner ${this.name} stream error:`, error);
    }

    return lastPlan;
  }

  private parsePlan(content: string): Plan {
    const fenced = content.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1] ?? content;
    return JSON.parse(fenced.trim()) as Plan;
  }

  private normalizePlan(plan: Plan, availableActions: string[]): Plan | null {
    if (!plan || typeof plan.action !== 'string' || !availableActions.includes(plan.action)) return null;
    const confidence = plan.confidence === undefined
      ? undefined
      : Math.max(0, Math.min(1, Number(plan.confidence)));
    return {
      action: plan.action,
      params: plan.params && typeof plan.params === 'object' ? plan.params : {},
      reasoning: typeof plan.reasoning === 'string' ? plan.reasoning : undefined,
      confidence: Number.isFinite(confidence) ? confidence : undefined,
    };
  }
}

/**
 * Rule-based planner for simple decision making
 */
export class RuleBasedPlanner implements Planner {
  readonly name: string;
  private rules: Array<{
    pattern: RegExp;
    action: string;
    paramGenerator?: (match: RegExpMatchArray) => Record<string, unknown>;
  }>;

  constructor(name = 'rule-based-planner') {
    this.name = name;
    this.rules = [];
  }

  /**
   * Add a rule
   */
  addRule(
    pattern: RegExp,
    action: string,
    paramGenerator?: (match: RegExpMatchArray) => Record<string, unknown>
  ): void {
    this.rules.push({ pattern, action, paramGenerator });
  }

  async plan(context: PlanContext): Promise<Plan | null> {
    for (const rule of this.rules) {
      const match = context.observation.match(rule.pattern);
      if (match) {
        return {
          action: rule.action,
          params: rule.paramGenerator ? rule.paramGenerator(match) : {},
          reasoning: `Matched pattern: ${rule.pattern}`,
          confidence: 1.0,
        };
      }
    }

    // No rule matched
    return null;
  }

  async *planStream(
    context: PlanContext
  ): AsyncGenerator<string, Plan | null, unknown> {
    const plan = await this.plan(context);
    if (plan) {
      yield JSON.stringify(plan);
    }
    return plan;
  }
}

/**
 * Composite planner that tries multiple planners
 */
export class CompositePlanner implements Planner {
  readonly name: string;
  private planners: Planner[];
  private fallbackEnabled: boolean;

  constructor(
    name: string,
    planners: Planner[],
    fallbackEnabled = true
  ) {
    this.name = name;
    this.planners = planners;
    this.fallbackEnabled = fallbackEnabled;
  }

  async plan(context: PlanContext): Promise<Plan | null> {
    for (const planner of this.activePlanners()) {
      try {
        const plan = await planner.plan(context);
        if (plan) {
          return plan;
        }
      } catch (error) {
        console.error(`Planner ${planner.name} failed:`, error);
      }
    }

    return null;
  }

  async *planStream(
    context: PlanContext
  ): AsyncGenerator<string, Plan | null, unknown> {
    for (const planner of this.activePlanners()) {
      try {
        const iterator = planner.planStream(context);
        let step = await iterator.next();
        while (!step.done) {
          yield step.value;
          step = await iterator.next();
        }
        if (step.value) return step.value;
      } catch (error) {
        console.error(`Planner ${planner.name} stream failed:`, error);
      }
    }

    return null;
  }

  private activePlanners(): Planner[] {
    return this.fallbackEnabled ? this.planners : this.planners.slice(0, 1);
  }
}

export default Planner;
