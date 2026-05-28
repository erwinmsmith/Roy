// Tool types and interfaces

export interface ToolInput {
  action: string;
  parameters?: Record<string, unknown>;
}

export interface ToolResult {
  success: boolean;
  result?: unknown;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface ToolConfig {
  name: string;
  description?: string;
  version?: string;
  parameters?: Record<string, {
    type: 'string' | 'number' | 'boolean' | 'object' | 'array';
    required?: boolean;
    default?: unknown;
    description?: string;
  }>;
  timeout?: number;
}

/**
 * Base interface for all Tools
 * A Tool is an executable capability that can be registered and invoked
 */
export interface Tool {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;
  readonly isStream?: boolean;

  /**
   * Initialize the tool with configuration
   */
  initialize?(config: ToolConfig): Promise<void>;

  /**
   * Execute the tool with given parameters
   */
  execute(params: Record<string, unknown>): Promise<ToolResult>;

  /**
   * Validate input parameters
   */
  validate?(params: Record<string, unknown>): { valid: boolean; errors?: string[] };
}

/**
 * Tool that supports streaming output
 */
export interface StreamingTool extends Tool {
  isStream: true;

  executeStream(params: Record<string, unknown>): AsyncGenerator<string, void, unknown>;
}

/**
 * Tool metadata for discovery
 */
export interface ToolMetadata {
  name: string;
  description?: string;
  version?: string;
  categories?: string[];
  parameters?: ToolConfig['parameters'];
}