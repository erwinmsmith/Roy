import type { AgentRole } from '../agent/BaseAgent.js';
import type { MultiPartyTrace } from '../communication/index.js';

export interface ContextWindowPolicy {
  sessionWindowTurns: number;
  maxContextTokens: number;
  includeToolResults: 'none' | 'summary';
  includeSubagentReports: 'none' | 'summary' | 'full';
  includePrivateMemory: boolean;
  includePublicMemory: boolean;
}

export interface ContextMemoryScope {
  public: boolean;
  private: boolean;
  parentContext: boolean;
  sessionWindowTurns: number;
}

export interface ContextWindowRequest {
  sessionId: string;
  agentId: string;
  agentKey: string;
  role: AgentRole;
  task: string;
  parentContext?: string;
  memoryScope: ContextMemoryScope;
  communicationContext?: string;
  systemTraces?: MultiPartyTrace[];
}

export interface ContextTokenBreakdown {
  publicContext: number;
  privateMemory: number;
  sessionWindow: number;
  parentContext: number;
  task: number;
  communicationContext: number;
  multiPartyTraces: number;
  total: number;
}

export interface ContextWindow {
  publicContext: string;
  privateMemory: string;
  sessionContext: string;
  parentContext: string;
  task: string;
  communicationContext: string;
  multiPartyTraceContext: string;
  tokenUsage: ContextTokenBreakdown;
  sources: {
    public: string[];
    private: string[];
    session: string;
    parent: string;
    communication: string;
    traces: string;
  };
}
