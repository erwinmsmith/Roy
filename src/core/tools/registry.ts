// Tool registry implementation

import type { Tool, ToolConfig, ToolResult, ToolMetadata } from './types.js';
import { logger } from '../utils/logger.js';

class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  private categories: Map<string, Set<string>> = new Map();

  /**
   * Register a tool
   */
  register(tool: Tool, category?: string): void {
    if (this.tools.has(tool.name)) {
      logger.warn(`Tool "${tool.name}" already registered, overwriting`);
    }
    this.tools.set(tool.name, tool);

    if (category) {
      this.addToCategory(tool.name, category);
    }
  }

  /**
   * Unregister a tool
   */
  unregister(name: string): boolean {
    const tool = this.tools.get(name);
    if (tool) {
      // Remove from all categories
      for (const [, tools] of this.categories) {
        tools.delete(name);
      }
    }
    return this.tools.delete(name);
  }

  /**
   * Get a tool by name
   */
  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  /**
   * Check if a tool is registered
   */
  has(name: string): boolean {
    return this.tools.has(name);
  }

  /**
   * List all registered tools
   */
  list(): Tool[] {
    return Array.from(this.tools.values());
  }

  /**
   * Get tools by category
   */
  getByCategory(category: string): Tool[] {
    const names = this.categories.get(category);
    if (!names) return [];
    return Array.from(names)
      .map(name => this.tools.get(name))
      .filter((t): t is Tool => t !== undefined);
  }

  /**
   * Get all categories
   */
  getCategories(): string[] {
    return Array.from(this.categories.keys());
  }

  /**
   * Add tool to category
   */
  private addToCategory(toolName: string, category: string): void {
    if (!this.categories.has(category)) {
      this.categories.set(category, new Set());
    }
    this.categories.get(category)!.add(toolName);
  }

  /**
   * Execute a tool by name
   */
  async execute(name: string, params: Record<string, unknown> = {}): Promise<ToolResult> {
    const tool = this.get(name);
    if (!tool) {
      return {
        success: false,
        error: `Tool "${name}" not found`,
      };
    }

    try {
      // Validate parameters if tool has validate method
      if (tool.validate) {
        const validation = tool.validate(params);
        if (!validation.valid) {
          return {
            success: false,
            error: `Validation failed: ${validation.errors?.join(', ')}`,
          };
        }
      }

      const result = await tool.execute(params);
      return result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Execute a streaming tool
   */
  async *executeStream(name: string, params: Record<string, unknown> = {}): AsyncGenerator<string, void, unknown> {
    const tool = this.get(name);
    if (!tool) {
      throw new Error(`Tool "${name}" not found`);
    }

    if ('executeStream' in tool && typeof tool.executeStream === 'function') {
      const streamingTool = tool as { executeStream: (params: Record<string, unknown>) => AsyncGenerator<string, void, unknown> };
      yield* streamingTool.executeStream(params);
    } else {
      const result = await tool.execute(params);
      if (result.success && result.result !== undefined) {
        yield String(result.result);
      }
    }
  }

  /**
   * Get tool metadata for discovery
   */
  getMetadata(name: string): ToolMetadata | undefined {
    const tool = this.get(name);
    if (!tool) return undefined;

    const toolAny = tool as Tool & { parameters?: ToolMetadata['parameters'] };
    return {
      name: tool.name,
      description: tool.description,
      version: tool.version,
      parameters: toolAny.parameters,
    };
  }

  /**
   * List all tool metadata
   */
  listMetadata(): ToolMetadata[] {
    return this.list().map(tool => this.getMetadata(tool.name)!);
  }

  /**
   * Get tool names
   */
  keys(): string[] {
    return Array.from(this.tools.keys());
  }

  /**
   * Clear all tools
   */
  clear(): void {
    this.tools.clear();
    this.categories.clear();
  }
}

/**
 * Decorator to register a tool class
 */
function toolDecorator(category?: string) {
  return function <T extends new () => Tool>(toolClass: T): T {
    const instance = new toolClass();
    toolRegistry.register(instance, category);
    return toolClass;
  };
}

// Singleton registry instance
export const toolRegistry = new ToolRegistry();

// Export decorator for convenience
export { toolDecorator as registerTool };

// Re-export types
export type { Tool, ToolConfig, ToolResult, ToolMetadata } from './types.js';