// Skill types and interfaces
// Skills are installable extensions that can be discovered, installed, and managed

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
  author?: string;
  tags?: string[];
  parameters?: Record<string, {
    type: 'string' | 'number' | 'boolean' | 'object' | 'array';
    required?: boolean;
    default?: unknown;
    description?: string;
  }>;
  timeout?: number;
  dependencies?: string[];
}

/**
 * Skill metadata for discovery and installation
 */
export interface SkillManifest {
  name: string;
  version: string;
  author?: string;
  description: string;
  tags: string[];
  source?: string;
  installedAt?: number;
}

/**
 * Base interface for all Skills
 * Skills are installable extensions that agents can use
 */
export interface Skill {
  readonly name: string;
  readonly description?: string;
  readonly version?: string;

  /**
   * Get skill manifest for installation tracking
   */
  getManifest(): SkillManifest;

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

  /**
   * Check if skill is ready to use
   */
  isReady?(): boolean;
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

/**
 * Remote skill that can be installed from a source
 */
export interface RemoteSkill extends Skill {
  getSource(): string;
  getChecksum?(): string;
}

/**
 * Skill installation result
 */
export interface SkillInstallationResult {
  success: boolean;
  skill?: Skill;
  error?: string;
  warning?: string;
}