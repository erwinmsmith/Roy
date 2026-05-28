// Capability System - unified abstraction for tools, actions, and skills
// This module re-exports from specialized modules for convenience.

// Re-export from actions module
export {
  actionRegistry,
  registerAction,
} from '../actions/index.js';
export type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
} from '../actions/index.js';

// Re-export Action class
export { Action } from '../actions/Action.js';
export { StreamingAction } from '../actions/Action.js';
export type { ActionFactory } from '../actions/Action.js';

// Re-export Planner
export {
  Planner,
  LLMPlanner,
  RuleBasedPlanner,
  CompositePlanner,
} from '../actions/Planner.js';
export type {
  Plan,
  PlanContext,
  AgentInfo,
  PlannerConfig,
} from '../actions/Planner.js';

// Re-export Tool types
export type { Tool, ToolConfig, ToolResult, ToolMetadata } from '../tools/types.js';
export { toolRegistry, registerTool } from '../tools/index.js';

// Re-export Skill types
export type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput } from '../skills/types.js';
export { skillRegistry, registerSkill } from '../skills/index.js';

// Re-export ActionRegistry (default export)
export { default as ActionRegistry } from '../actions/index.js';