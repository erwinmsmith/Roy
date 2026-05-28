// Action base class - all agent actions should extend this

import type { LLMProvider, LLMMessage } from '../llm/types.js';

export interface ActionConfig {
  name: string;
  description?: string;
  parameters?: ActionParameter[];
  examples?: ActionExample[];
}

export interface ActionParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required?: boolean;
  default?: unknown;
  description?: string;
  enum?: string[];
}

export interface ActionExample {
  input: Record<string, unknown>;
  output: unknown;
  description?: string;
}

export interface ActionResult {
  success: boolean;
  result?: unknown;
  error?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Base class for all Actions
 * Actions are executable capabilities that agents can use
 */
export abstract class Action {
  readonly name: string;
  readonly description: string;
  readonly parameters: ActionParameter[];
  readonly examples: ActionExample[];
  readonly isStream: boolean = false;

  constructor(config: ActionConfig) {
    this.name = config.name;
    this.description = config.description || '';
    this.parameters = config.parameters || [];
    this.examples = config.examples || [];
  }

  /**
   * Validate parameters against the action's schema
   */
  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];

    for (const param of this.parameters) {
      const value = params[param.name];

      if (param.required && (value === undefined || value === null)) {
        errors.push(`Required parameter "${param.name}" is missing`);
        continue;
      }

      if (value !== undefined && value !== null) {
        const actualType = Array.isArray(value) ? 'array' : typeof value;
        if (actualType !== param.type) {
          errors.push(`Parameter "${param.name}" should be ${param.type}, got ${actualType}`);
        }

        if (param.enum && !param.enum.includes(String(value))) {
          errors.push(`Parameter "${param.name}" must be one of: ${param.enum.join(', ')}`);
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
    };
  }

  /**
   * Execute the action with given parameters
   */
  abstract execute(params: Record<string, unknown>): Promise<ActionResult>;

  /**
   * Execute with streaming (optional override)
   */
  async *executeStream(
    params: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown> {
    const result = await this.execute(params);
    if (result.success && result.result !== undefined) {
      yield String(result.result);
    } else if (result.error) {
      yield result.error;
    }
  }

  /**
   * Get action schema for LLM
   */
  getSchema(): ActionSchema {
    return {
      name: this.name,
      description: this.description,
      parameters: this.parameters.map(p => ({
        name: p.name,
        type: p.type,
        required: p.required ?? false,
        description: p.description,
        enum: p.enum,
      })),
      examples: this.examples,
    };
  }

  /**
   * Convert to JSON string for LLM prompt
   */
  toJSONString(): string {
    return JSON.stringify(this.getSchema(), null, 2);
  }
}

export interface ActionSchema {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    required: boolean;
    description?: string;
    enum?: string[];
  }>;
  examples?: ActionExample[];
}

/**
 * Streaming action mixin
 */
export abstract class StreamingAction extends Action {
  readonly isStream = true;

  abstract executeStream(
    params: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown>;
}

/**
 * Action execution context
 */
export interface ActionContext {
  agentName: string;
  sessionId: string;
  llm?: LLMProvider;
  metadata?: Record<string, unknown>;
}

/**
 * Factory function type for creating actions
 */
export type ActionFactory = () => Action;

export default Action;