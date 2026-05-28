// Action Registry - manages action registration and discovery

import type { Action, ActionConfig, ActionResult } from './Action.js';

class ActionRegistry {
  private actions: Map<string, Action> = new Map();
  private categories: Map<string, Set<string>> = new Map();

  /**
   * Register an action
   */
  register(action: Action, category?: string): void {
    if (this.actions.has(action.name)) {
      console.warn(`Action "${action.name}" already registered, overwriting`);
    }
    this.actions.set(action.name, action);

    if (category) {
      this.addToCategory(action.name, category);
    }
  }

  /**
   * Unregister an action
   */
  unregister(name: string): boolean {
    const action = this.actions.get(name);
    if (action) {
      for (const [, tools] of this.categories) {
        tools.delete(name);
      }
    }
    return this.actions.delete(name);
  }

  /**
   * Get an action by name
   */
  get(name: string): Action | undefined {
    return this.actions.get(name);
  }

  /**
   * Check if an action is registered
   */
  has(name: string): boolean {
    return this.actions.has(name);
  }

  /**
   * List all registered actions
   */
  list(): Action[] {
    return Array.from(this.actions.values());
  }

  /**
   * Get actions by category
   */
  getByCategory(category: string): Action[] {
    const names = this.categories.get(category);
    if (!names) return [];
    return Array.from(names)
      .map(name => this.actions.get(name))
      .filter((a): a is Action => a !== undefined);
  }

  /**
   * Get all categories
   */
  getCategories(): string[] {
    return Array.from(this.categories.keys());
  }

  /**
   * Add action to category
   */
  private addToCategory(actionName: string, category: string): void {
    if (!this.categories.has(category)) {
      this.categories.set(category, new Set());
    }
    this.categories.get(category)!.add(actionName);
  }

  /**
   * Execute an action by name
   */
  async execute(name: string, params: Record<string, unknown> = {}): Promise<ActionResult> {
    const action = this.get(name);
    if (!action) {
      return {
        success: false,
        error: `Action "${name}" not found`,
      };
    }

    try {
      const validation = action.validate(params);
      if (!validation.valid) {
        return {
          success: false,
          error: `Validation failed: ${validation.errors?.join(', ')}`,
        };
      }

      return await action.execute(params);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Execute a streaming action
   */
  async *executeStream(
    name: string,
    params: Record<string, unknown> = {}
  ): AsyncGenerator<string, void, unknown> {
    const action = this.get(name);
    if (!action) {
      throw new Error(`Action "${name}" not found`);
    }

    if (action.isStream) {
      yield* action.executeStream(params);
    } else {
      const result = await action.execute(params);
      if (result.success && result.result !== undefined) {
        yield String(result.result);
      } else if (result.error) {
        yield result.error;
      }
    }
  }

  /**
   * Get action metadata for discovery
   */
  getMetadata(name: string): { name: string; description: string; parameters: unknown[] } | undefined {
    const action = this.get(name);
    if (!action) return undefined;

    return {
      name: action.name,
      description: action.description,
      parameters: action.parameters,
    };
  }

  /**
   * List all action metadata
   */
  listMetadata(): Array<{ name: string; description: string; parameters: unknown[] }> {
    return this.list().map(action => this.getMetadata(action.name)!);
  }

  /**
   * Get action names
   */
  keys(): string[] {
    return Array.from(this.actions.keys());
  }

  /**
   * Clear all actions
   */
  clear(): void {
    this.actions.clear();
    this.categories.clear();
  }

  /**
   * Format actions as JSON for LLM prompts
   */
  formatActionsForPrompt(): string {
    const lines: string[] = [];
    for (const action of this.list()) {
      lines.push(`- ${action.name}: ${action.description}`);
      if (action.parameters.length > 0) {
        lines.push('  Parameters:');
        for (const param of action.parameters) {
          const required = param.required ? '(required)' : '(optional)';
          lines.push(`    - ${param.name} (${param.type}) ${required}: ${param.description || ''}`);
          if (param.enum) {
            lines.push(`      Options: ${param.enum.join(', ')}`);
          }
        }
      }
    }
    return lines.join('\n');
  }
}

// Singleton registry instance
export const actionRegistry = new ActionRegistry();

/**
 * Decorator to register an action class
 */
function actionDecorator(category?: string) {
  return function <T extends new () => Action>(ActionClass: T): T {
    const instance = new ActionClass();
    actionRegistry.register(instance, category);
    return ActionClass;
  };
}

export { actionDecorator as registerAction };

// Re-export Action class and types
export { Action } from './Action.js';
export type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
  ActionFactory,
} from './Action.js';

export default ActionRegistry;