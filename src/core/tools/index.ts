// Tools module exports

export { toolRegistry, registerTool } from './registry.js';
export { FsListTool, FsReadTool, FsReplaceTool, FsSearchTool, FsWriteTool } from './fsTools.js';
export type {
  FsListResult,
  FsReadResult,
  FsReplaceResult,
  FsSearchMatch,
  FsSearchResult,
  FsWriteResult,
} from './fsTools.js';
export { ShellExecTool } from './shellExec.js';
export type { ShellExecConfig, ShellExecMode, ShellExecResult } from './shellExec.js';
export { WebSearchTool, WebFetchTool, defaultWebToolConfig } from './webTools.js';
export { registerCoreTools } from './builtin.js';
export { AgentToolPlanner } from './planner.js';
export { AgentToolExecutionLoop } from './executionLoop.js';
export {
  completedWorkspaceReadCoversPlan,
  findParallelSourceMutation,
  isSuccessfulWorkspaceMutationCall,
  isSuccessfulWorkspaceVerificationCall,
  isWorkspaceVerificationCall,
  taskRequestsWorkspaceMutation,
  workspaceToolIntentFingerprint,
} from './executionIntent.js';
export { ToolApprovalManager } from './approval.js';
export type { Tool, ToolConfig, ToolResult, ToolMetadata } from './types.js';
export type { ObservedToolCall, PlannedToolCall, ToolPlanBinding, ToolPlanningInput } from './planner.js';
export type { WebToolConfig, WebSearchProviderName, WebSearchResult, WebSearchResultItem, WebFetchResult } from './webTools.js';
export type { ToolLoopCallRecord, ToolLoopContinuationContext, ToolLoopRound, ToolLoopStopReason, ToolLoopSummary, ToolExecutionLoopOptions } from './executionLoop.js';
export type { ExecutionIntentCall, ParallelSourceMutation } from './executionIntent.js';
export type { ToolApprovalDecision, ToolApprovalPolicy, ToolApprovalRequest, ToolApprovalResult } from './approval.js';
