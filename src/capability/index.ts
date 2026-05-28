// Capability System - unified abstraction for tools, actions, and skills
// This module provides a unified interface that combines:
// - Tool (legacy, alias for Capability)
// - Action (core abstraction with validation)
// - Skill (Capability with execution context)

// Re-export Action as the primary abstraction
export {
  Action,
  StreamingAction,
} from './Action.js';
export type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
  ActionFactory,
} from './Action.js';

// Re-export ActionRegistry
export { actionRegistry, registerAction } from './index.js';
export { ActionRegistry } from './index.js';

// Legacy Tool types (alias to Action types for backward compatibility)
import type { Action, ActionConfig, ActionResult, ActionContext } from './Action.js';

export type Tool = Action;
export type ToolConfig = ActionConfig;
export type ToolResult = ActionResult;
export type ToolContext = ActionContext;

// Legacy Skill types (Action with context)
import type { Skill as SkillInterface } from '../skills/types.js';

// Capability interface - the unified abstraction
export interface Capability extends Action {
  // Additional metadata
  category?: string;
  tags?: string[];
}

// Re-export Planner
export {
  Planner,
  LLMPlanner,
  RuleBasedPlanner,
  CompositePlanner,
} from './Planner.js';
export type {
  Plan,
  PlanContext,
  AgentInfo,
  PlannerConfig,
} from './Planner.js';

export default Action;