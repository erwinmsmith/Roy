// Skill registry - manages skill registration, discovery, and installation

import type {
  Skill,
  SkillConfig,
  SkillInput,
  SkillContext,
  SkillOutput,
  SkillManifest,
  SkillInstallationResult,
} from './types.js';
import { logger } from '../utils/logger.js';

export type SkillInstaller = (source: string) => Promise<Skill>;

class SkillRegistry {
  private skills: Map<string, Skill> = new Map();
  private manifests: Map<string, SkillManifest> = new Map();
  private installers: Map<string, SkillInstaller> = new Map();

  /**
   * Register a skill
   */
  register(skill: Skill): void {
    if (this.skills.has(skill.name)) {
      logger.warn(`Skill "${skill.name}" already registered, overwriting`);
    }
    this.skills.set(skill.name, skill);

    const manifest = skill.getManifest();
    this.manifests.set(skill.name, manifest);
  }

  /**
   * Unregister a skill
   */
  unregister(name: string): boolean {
    this.manifests.delete(name);
    return this.skills.delete(name);
  }

  /**
   * Get a skill by name
   */
  get(name: string): Skill | undefined {
    return this.skills.get(name);
  }

  /**
   * Check if a skill is registered
   */
  has(name: string): boolean {
    return this.skills.has(name);
  }

  /**
   * List all registered skills
   */
  list(): Skill[] {
    return Array.from(this.skills.values());
  }

  /**
   * Find skills by predicate
   */
  find(predicate: (skill: Skill) => boolean): Skill[] {
    return this.list().filter(predicate);
  }

  /**
   * Find skills by tag
   */
  findByTag(tag: string): Skill[] {
    return this.list().filter(skill => {
      const manifest = this.manifests.get(skill.name);
      return manifest?.tags.includes(tag);
    });
  }

  /**
   * Get skill manifest
   */
  getManifest(name: string): SkillManifest | undefined {
    return this.manifests.get(name);
  }

  /**
   * List all manifests
   */
  listManifests(): SkillManifest[] {
    return Array.from(this.manifests.values());
  }

  /**
   * Register an installer for a source type
   */
  registerInstaller(sourceType: string, installer: SkillInstaller): void {
    this.installers.set(sourceType, installer);
  }

  /**
   * Install a skill from source
   */
  async install(source: string): Promise<SkillInstallationResult> {
    try {
      const sourceType = this.parseSourceType(source);
      const installer = this.installers.get(sourceType);

      if (!installer) {
        return {
          success: false,
          error: `No installer registered for source type: ${sourceType}`,
        };
      }

      const skill = await installer(source);
      this.register(skill);

      return {
        success: true,
        skill,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Execute a skill by name
   */
  async execute(
    name: string,
    input: SkillInput,
    context: SkillContext
  ): Promise<SkillOutput> {
    const skill = this.get(name);
    if (!skill) {
      return {
        success: false,
        error: `Skill "${name}" not found`,
      };
    }

    try {
      return await skill.execute(input, context);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get skill names
   */
  keys(): string[] {
    return Array.from(this.skills.keys());
  }

  /**
   * Get skill count
   */
  size(): number {
    return this.skills.size;
  }

  /**
   * Clear all skills
   */
  clear(): void {
    this.skills.clear();
    this.manifests.clear();
  }

  /**
   * Parse source type from source string
   */
  private parseSourceType(source: string): string {
    if (source.startsWith('npm:') || source.startsWith('npm://')) {
      return 'npm';
    }
    if (source.startsWith('github:') || source.includes('github.com')) {
      return 'github';
    }
    if (source.startsWith('file:') || source.startsWith('./') || source.startsWith('/')) {
      return 'file';
    }
    if (source.startsWith('http://') || source.startsWith('https://')) {
      return 'url';
    }
    return 'unknown';
  }
}

/**
 * Decorator to register a skill class
 */
function skillDecorator(skillClass: new () => Skill): Skill {
  const instance = new skillClass();
  skillRegistry.register(instance);
  return instance;
}

// Singleton registry instance
export const skillRegistry = new SkillRegistry();

// Export decorator for convenience
export { skillDecorator as registerSkill };
export { DelegateToSubagentSkill } from './delegation.js';

// Re-export types
export type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput } from './types.js';
