// Capability System - unified abstraction for tools, actions, and skills
// This module provides a unified interface that combines:
// - Tool (legacy, alias for Capability)
// - Action (core abstraction with validation)
// - Skill (Capability with execution context)

// Re-export Action as the primary abstraction
export {
  Action,
  StreamingAction,
} from '../actions/Action.js';
export type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
  ActionFactory,
} from '../actions/Action.js';

// Re-export ActionRegistry
export { actionRegistry, registerAction } from '../actions/index.js';
export { ActionRegistry } from '../actions/index.js';

// Legacy Tool types (alias to Action types for backward compatibility)
export type { Tool as Tool, ToolConfig as ToolConfig, ToolResult as ToolResult } from '../tools/types.js';
export type { ToolMetadata } from '../tools/types.js';

// Legacy Skill types (Action with context)
export type { SkillConfig, SkillInput, SkillContext, SkillOutput, Skill } from '../skills/types.js';

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

export { default as Action } from '../actions/Action.js';