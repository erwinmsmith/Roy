// Memory module exports and registry

import { ShortTermMemory } from './shortTerm.js';
import { LongTermMemory } from './longTerm.js';
import { contextManager } from './context.js';
import type { ContextManager } from './context.js';

/**
 * Memory registry - manages multiple memory instances per agent/session
 */
class MemoryRegistry {
  private shortTermMemories: Map<string, ShortTermMemory> = new Map();
  private longTermMemories: Map<string, LongTermMemory> = new Map();

  /**
   * Get or create short-term memory for agent/session
   */
  getShortTerm(agentId: string, sessionId: string): ShortTermMemory {
    const key = this.makeKey(agentId, sessionId);
    if (!this.shortTermMemories.has(key)) {
      this.shortTermMemories.set(key, new ShortTermMemory(key));
    }
    return this.shortTermMemories.get(key)!;
  }

  /**
   * Get or create long-term memory for agent
   */
  getLongTerm(agentId: string): LongTermMemory {
    if (!this.longTermMemories.has(agentId)) {
      this.longTermMemories.set(agentId, new LongTermMemory(agentId));
    }
    return this.longTermMemories.get(agentId)!;
  }

  /**
   * Get context manager
   */
  getContextManager(): ContextManager {
    return contextManager;
  }

  /**
   * Clear all memories for session
   */
  clearSession(agentId: string, sessionId: string): void {
    const key = this.makeKey(agentId, sessionId);
    const shortTerm = this.shortTermMemories.get(key);
    if (shortTerm) {
      shortTerm.clear();
    }
    contextManager.delete(agentId, sessionId);
  }

  /**
   * Clear all memories for agent
   */
  clearAgent(agentId: string): void {
    // Clear all session memories for this agent
    for (const [key, memory] of this.shortTermMemories) {
      if (key.startsWith(`${agentId}:`)) {
        memory.clear();
        this.shortTermMemories.delete(key);
      }
    }

    // Clear long-term memory
    const longTerm = this.longTermMemories.get(agentId);
    if (longTerm) {
      longTerm.clear();
      this.longTermMemories.delete(agentId);
    }

    // Clear all contexts for this agent
    const contexts = contextManager.getByAgent(agentId);
    for (const ctx of contexts) {
      contextManager.delete(agentId, ctx.sessionId);
    }
  }

  /**
   * Clear all memories
   */
  clearAll(): void {
    this.shortTermMemories.clear();
    this.longTermMemories.clear();
    contextManager.clear();
  }

  /**
   * Get memory statistics
   */
  getStats(agentId: string, sessionId: string): {
    shortTerm: { count: number; size: number };
    longTerm: { count: number; size: number };
    context: { versions: number };
  } {
    const shortTerm = this.getShortTerm(agentId, sessionId);
    const longTerm = this.getLongTerm(agentId);
    const contexts = contextManager.getByAgent(agentId);

    return {
      shortTerm: shortTerm.stats(),
      longTerm: longTerm.stats(),
      context: { versions: contexts.length },
    };
  }

  private makeKey(agentId: string, sessionId: string): string {
    return `${agentId}:${sessionId}`;
  }
}

export const memoryRegistry = new MemoryRegistry();

export { ShortTermMemory } from './shortTerm.js';
export { LongTermMemory, InMemoryCache } from './longTerm.js';
export { contextManager, ContextManager } from './context.js';
export { WorkspaceMemoryManager } from './workspace.js';
export type { Memory, MemoryEntry, ContextDoc, CacheEntry, MemoryCache } from './types.js';
export type {
  AgentMemoryBundle,
  AgentPatternInput,
  DelegationPatternInput,
  ConversationEntry,
  ConversationSessionState,
  MemoryAutoState,
  MemoryDocState,
  MemoryMode,
  MemoryProposalSummary,
  MemorySignals,
  MemoryUpdateProposal,
  MemoryUpdateRecord,
  PatternCacheState,
  RootMemoryContext,
  WorkspaceRuntimeConfig,
  WorkspaceMemoryState,
} from './workspace.js';
