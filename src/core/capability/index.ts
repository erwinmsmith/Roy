// Capability System - unified abstraction for tools, actions, and skills
// This module re-exports from specialized modules for convenience.

import {
  actionRegistry,
  registerAction,
} from '../actions/index.js';
import type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
} from '../actions/index.js';
import { Action } from '../actions/Action.js';
import { StreamingAction } from '../actions/Action.js';
import type { ActionFactory } from '../actions/Action.js';
import {
  Planner,
  LLMPlanner,
  RuleBasedPlanner,
  CompositePlanner,
} from '../actions/Planner.js';
import type {
  Plan,
  PlanContext,
  AgentInfo,
  PlannerConfig,
} from '../actions/Planner.js';
import type { Tool, ToolConfig, ToolResult, ToolMetadata } from '../tools/types.js';
import { toolRegistry, registerTool } from '../tools/index.js';
import type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput } from '../skills/types.js';
import { skillRegistry, registerSkill } from '../skills/index.js';

// Re-export from actions module
export { actionRegistry, registerAction };
export type {
  ActionConfig,
  ActionParameter,
  ActionExample,
  ActionResult,
  ActionSchema,
  ActionContext,
};

// Re-export Action class
export { Action, StreamingAction };
export type { ActionFactory };

// Re-export Planner
export { Planner, LLMPlanner, RuleBasedPlanner, CompositePlanner };
export type { Plan, PlanContext, AgentInfo, PlannerConfig };

// Re-export Tool types
export type { Tool, ToolConfig, ToolResult, ToolMetadata };
export { toolRegistry, registerTool };

// Re-export Skill types
export type { Skill, SkillConfig, SkillInput, SkillContext, SkillOutput };
export { skillRegistry, registerSkill };

// Re-export ActionRegistry (default export)
export { actionRegistry as ActionRegistry };
