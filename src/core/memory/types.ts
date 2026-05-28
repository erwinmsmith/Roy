// Memory types and interfaces

export interface MemoryEntry {
  id: string;
  type: 'observation' | 'action' | 'result' | 'meta';
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface MemoryBlock {
  entries: MemoryEntry[];
  createdAt: number;
  updatedAt: number;
}

export interface ContextDoc {
  id: string;
  agentId: string;
  sessionId: string;
  content: string; // MD formatted content
  version: number;
  createdAt: number;
  updatedAt: number;
}

export interface CacheEntry {
  key: string;
  value: unknown;
  ttl: number; // time to live in ms
  createdAt: number;
  accessCount: number;
  lastAccessedAt: number;
}

/**
 * Base interface for all memory types
 */
export interface Memory {
  readonly name: string;
  readonly type: 'short-term' | 'long-term' | 'context';

  /**
   * Add an entry to memory
   */
  add(entry: Omit<MemoryEntry, 'id' | 'timestamp'>): string;

  /**
   * Get entries from memory
   */
  get(limit?: number): MemoryEntry[];

  /**
   * Search entries
   */
  search(query: string): MemoryEntry[];

  /**
   * Clear memory
   */
  clear(): void;

  /**
   * Get memory stats
   */
  stats(): { count: number; size: number };
}

/**
 * Cache interface for long-term memory optimization
 */
export interface MemoryCache {
  get(key: string): unknown | undefined;
  set(key: string, value: unknown, ttl?: number): void;
  delete(key: string): boolean;
  clear(): void;
  size(): number;
}

/**
 * ToM-related memory types
 */
export interface BeliefState {
  agentId: string;
  beliefs: Record<string, unknown>;
  uncertainty: Record<string, number>;
  timestamp: number;
}

export interface EvidenceRecord {
  claim: string;
  supporting: string[];
  contradicting: string[];
  source: string;
  confidence: number;
}