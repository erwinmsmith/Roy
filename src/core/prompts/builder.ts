// Prompt template types and builder

export interface PromptTemplate {
  name: string;
  template: string;
  description?: string;
}

export interface PromptVariables {
  [key: string]: string | undefined;
}

export class PromptBuilder {
  /**
   * Build a prompt from a template with variable substitution
   * Variables are denoted as {variable_name} in the template
   */
  static build(template: string, variables: PromptVariables): string {
    let result = template;

    for (const [key, value] of Object.entries(variables)) {
      const placeholder = `{${key}}`;
      if (value !== undefined) {
        result = result.split(placeholder).join(value);
      }
    }

    return result;
  }

  /**
   * Build a prompt from a template object
   */
  static fromTemplate(promptTemplate: PromptTemplate, variables: PromptVariables): string {
    return this.build(promptTemplate.template, variables);
  }

  /**
   * Validate that all required variables are provided
   */
  static validate(template: string, variables: PromptVariables): { valid: boolean; missing: string[] } {
    const matches = template.match(/\{(\w+)\}/g) || [];
    const required = matches.map(m => m.slice(1, -1));
    const provided = Object.keys(variables);
    const missing = required.filter(r => !provided.includes(r));

    return {
      valid: missing.length === 0,
      missing,
    };
  }
}

export function buildPrompt(template: string, variables: PromptVariables): string {
  return PromptBuilder.build(template, variables);
}