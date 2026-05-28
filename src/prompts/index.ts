// Prompt registry - exports all templates and registry functionality

export { buildPrompt, PromptBuilder } from './builder.js';
export type { PromptTemplate, PromptVariables } from './builder.js';

export { conversationalTemplate, CONVERSATIONAL_AGENT_TEMPLATE } from './templates/conversational.js';
export { actionTemplate, actionCallTemplate, ACTION_AGENT_TEMPLATE, ACTION_TEMPLATE } from './templates/action.js';
export { g1Template, g1FinalAnswerTemplate, G1_SYSTEM_PROMPT, G1_FINAL_ANSWER_PROMPT } from './templates/g1.js';
export {
  fsmDiagnoseTemplate,
  fsmDecideTemplate,
  fsmDeriveTemplate,
  fsmVerifyTemplate,
  FSM_STATES,
  FSM_DIAGNOSE_PROMPT,
  FSM_DECIDE_PROMPT,
  FSM_DERIVE_PROMPT,
  FSM_VERIFY_PROMPT,
} from './templates/fsm.js';
export type { FSMStateName } from './templates/fsm.js';

import type { PromptTemplate } from './builder.js';
import { conversationalTemplate } from './templates/conversational.js';
import { actionTemplate, actionCallTemplate } from './templates/action.js';
import { g1Template, g1FinalAnswerTemplate } from './templates/g1.js';
import { fsmDiagnoseTemplate, fsmDecideTemplate, fsmDeriveTemplate, fsmVerifyTemplate } from './templates/fsm.js';

const allTemplates: PromptTemplate[] = [
  conversationalTemplate,
  actionTemplate,
  actionCallTemplate,
  g1Template,
  g1FinalAnswerTemplate,
  fsmDiagnoseTemplate,
  fsmDecideTemplate,
  fsmDeriveTemplate,
  fsmVerifyTemplate,
];

class PromptRegistry {
  private templates: Map<string, PromptTemplate> = new Map();

  constructor() {
    for (const tmpl of allTemplates) {
      this.register(tmpl);
    }
  }

  register(template: PromptTemplate): void {
    this.templates.set(template.name, template);
  }

  get(name: string): PromptTemplate | undefined {
    return this.templates.get(name);
  }

  list(): PromptTemplate[] {
    return Array.from(this.templates.values());
  }
}

export const promptRegistry = new PromptRegistry();