// Skill registry - manages skill registration and discovery

import type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput } from './types.js';

class SkillRegistry {
  private skills: Map<string, Skill> = new Map();

  /**
   * Register a skill
   */
  register(skill: Skill): void {
    if (this.skills.has(skill.name)) {
      console.warn(`Skill "${skill.name}" already registered, overwriting`);
    }
    this.skills.set(skill.name, skill);
  }

  /**
   * Unregister a skill
   */
  unregister(name: string): boolean {
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
   * Clear all skills
   */
  clear(): void {
    this.skills.clear();
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

// Re-export types
export type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput } from './types.js';