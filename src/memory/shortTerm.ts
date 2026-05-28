// Short-term memory implementation

import { v4 as uuidv4 } from 'uuid';
import type { Memory, MemoryEntry } from './types.js';

export class ShortTermMemory implements Memory {
  readonly name: string;
  readonly type = 'short-term' as const;
  private entries: MemoryEntry[] = [];
  private maxSize: number;

  constructor(name: string, maxSize = 100) {
    this.name = name;
    this.maxSize = maxSize;
  }

  add(entry: Omit<MemoryEntry, 'id' | 'timestamp'>): string {
    const id = uuidv4();
    const fullEntry: MemoryEntry = {
      ...entry,
      id,
      timestamp: Date.now(),
    };

    this.entries.push(fullEntry);

    // Enforce max size by removing oldest entries
    while (this.entries.length > this.maxSize) {
      this.entries.shift();
    }

    return id;
  }

  get(limit?: number): MemoryEntry[] {
    if (limit !== undefined) {
      return this.entries.slice(-limit);
    }
    return [...this.entries];
  }

  search(query: string): MemoryEntry[] {
    const lowerQuery = query.toLowerCase();
    return this.entries.filter(entry =>
      entry.content.toLowerCase().includes(lowerQuery)
    );
  }

  clear(): void {
    this.entries = [];
  }

  stats(): { count: number; size: number } {
    return {
      count: this.entries.length,
      size: JSON.stringify(this.entries).length,
    };
  }

  /**
   * Get entries by type
   */
  getByType(type: MemoryEntry['type']): MemoryEntry[] {
    return this.entries.filter(entry => entry.type === type);
  }

  /**
   * Get recent entries
   */
  getRecent(count: number): MemoryEntry[] {
    return this.entries.slice(-count);
  }

  /**
   * Clear entries older than timestamp
   */
  clearOlderThan(timestamp: number): number {
    const before = this.entries.length;
    this.entries = this.entries.filter(entry => entry.timestamp >= timestamp);
    return before - this.entries.length;
  }
}