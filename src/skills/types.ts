// Skill types and interfaces

export interface SkillContext {
  agentId: string;
  sessionId: string;
  variables: Record<string, unknown>;
}

export interface SkillInput {
  action: string;
  params: Record<string, unknown>;
}

export interface SkillOutput {
  success: boolean;
  result?: unknown;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface SkillConfig {
  name: string;
  description?: string;
  version?: string;
  parameters?: Record<string, {
    type: 'string' | 'number' | 'boolean' | 'object';
    required?: boolean;
    default?: unknown;
    description?: string;
  }>;
  timeout?: number;
}

/**
 * Base interface for all Skills
 * A Skill is a composable unit of capability that can be registered and discovered
 */
export interface Skill {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;

  /**
   * Initialize the skill with configuration
   */
  initialize?(config: SkillConfig): Promise<void>;

  /**
   * Execute the skill with given input
   */
  execute(input: SkillInput, context: SkillContext): Promise<SkillOutput>;

  /**
   * Validate input parameters
   */
  validate?(input: SkillInput): { valid: boolean; errors?: string[] };
}

/**
 * Skill that supports streaming output
 */
export interface StreamingSkill extends Skill {
  executeStream(
    input: SkillInput,
    context: SkillContext
  ): AsyncGenerator<SkillOutput, void, unknown>;
}

/**
 * Skill that can be composed with other skills
 */
export interface ComposableSkill extends Skill {
  getDependencies(): string[];
  getOutputSchema(): Record<string, unknown>;
}