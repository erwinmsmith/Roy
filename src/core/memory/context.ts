// Context management - MD document mode

import { v4 as uuidv4 } from 'uuid';
import type { ContextDoc, MemoryEntry } from './types.js';

export class ContextManager {
  private contexts: Map<string, ContextDoc> = new Map();
  private agentContexts: Map<string, Map<string, ContextDoc>> = new Map();

  /**
   * Create or update a context document
   */
  upsert(agentId: string, sessionId: string, content: string): ContextDoc {
    const key = this.makeKey(agentId, sessionId);
    const existing = this.contexts.get(key);

    const doc: ContextDoc = {
      id: existing?.id || uuidv4(),
      agentId,
      sessionId,
      content,
      version: existing ? existing.version + 1 : 1,
      createdAt: existing?.createdAt || Date.now(),
      updatedAt: Date.now(),
    };

    this.contexts.set(key, doc);

    // Also index by agent
    if (!this.agentContexts.has(agentId)) {
      this.agentContexts.set(agentId, new Map());
    }
    this.agentContexts.get(agentId)!.set(sessionId, doc);

    return doc;
  }

  /**
   * Get context by agent and session
   */
  get(agentId: string, sessionId: string): ContextDoc | undefined {
    return this.contexts.get(this.makeKey(agentId, sessionId));
  }

  /**
   * Get all contexts for an agent
   */
  getByAgent(agentId: string): ContextDoc[] {
    const agentMap = this.agentContexts.get(agentId);
    return agentMap ? Array.from(agentMap.values()) : [];
  }

  /**
   * Append content to existing context
   */
  append(agentId: string, sessionId: string, newContent: string): ContextDoc {
    const existing = this.get(agentId, sessionId);
    const content = existing
      ? `${existing.content}\n\n---\n\n${newContent}`
      : newContent;
    return this.upsert(agentId, sessionId, content);
  }

  /**
   * Delete context
   */
  delete(agentId: string, sessionId: string): boolean {
    const key = this.makeKey(agentId, sessionId);
    const doc = this.contexts.get(key);

    if (doc) {
      this.contexts.delete(key);
      const agentMap = this.agentContexts.get(agentId);
      if (agentMap) {
        agentMap.delete(sessionId);
      }
      return true;
    }
    return false;
  }

  /**
   * Convert context to memory entries for agent consumption
   */
  toMemoryEntries(doc: ContextDoc): MemoryEntry[] {
    const lines = doc.content.split('\n');
    const entries: MemoryEntry[] = [];

    for (const line of lines) {
      if (line.trim()) {
        entries.push({
          id: uuidv4(),
          type: 'observation',
          content: line.trim(),
          timestamp: doc.updatedAt,
        });
      }
    }

    return entries;
  }

  /**
   * Export context as markdown
   */
  exportMarkdown(agentId: string, sessionId: string): string {
    const doc = this.get(agentId, sessionId);
    if (!doc) return '';

    const header = `# Context: ${agentId} / ${sessionId}\n\n`;
    const meta = `> Created: ${new Date(doc.createdAt).toISOString()}\n> Version: ${doc.version}\n> Updated: ${new Date(doc.updatedAt).toISOString()}\n\n---\n\n`;
    return header + meta + doc.content;
  }

  /**
   * List all context keys
   */
  keys(): string[] {
    return Array.from(this.contexts.keys());
  }

  /**
   * Clear all contexts
   */
  clear(): void {
    this.contexts.clear();
    this.agentContexts.clear();
  }

  private makeKey(agentId: string, sessionId: string): string {
    return `${agentId}:${sessionId}`;
  }
}

export const contextManager = new ContextManager();