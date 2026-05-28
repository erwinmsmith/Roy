// Core shared types for Roy

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface AgentInfo {
  name: string;
  goal?: string;
  example?: string;
  state: AgentState;
}

export type AgentState = 'idle' | 'running' | 'waiting' | 'stopped';

export interface StreamChunk {
  content: string;
  done: boolean;
}

export interface ToolResult {
  success: boolean;
  result?: unknown;
  error?: string;
}

export interface SkillResult {
  success: boolean;
  output?: unknown;
  error?: string;
}