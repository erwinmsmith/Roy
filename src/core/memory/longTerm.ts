// Long-term memory with cache support

import { v4 as uuidv4 } from 'uuid';
import type { Memory, MemoryEntry, CacheEntry, MemoryCache } from './types.js';

export class InMemoryCache implements MemoryCache {
  private cache: Map<string, CacheEntry> = new Map();
  private defaultTTL: number;

  constructor(defaultTTL = 3600000) { // 1 hour default
    this.defaultTTL = defaultTTL;
  }

  get(key: string): unknown | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // Check TTL
    if (Date.now() > entry.createdAt + entry.ttl) {
      this.cache.delete(key);
      return undefined;
    }

    // Update access stats
    entry.accessCount++;
    entry.lastAccessedAt = Date.now();
    return entry.value;
  }

  set(key: string, value: unknown, ttl = this.defaultTTL): void {
    this.cache.set(key, {
      key,
      value,
      ttl,
      createdAt: Date.now(),
      accessCount: 0,
      lastAccessedAt: Date.now(),
    });
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    this.cleanup(); // Clean expired entries first
    return this.cache.size;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache) {
      if (now > entry.createdAt + entry.ttl) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Get cache statistics
   */
  stats(): { size: number; totalAccesses: number; hitRate: number } {
    let totalAccesses = 0;
    for (const entry of this.cache.values()) {
      totalAccesses += entry.accessCount;
    }
    return {
      size: this.cache.size,
      totalAccesses,
      hitRate: totalAccesses > 0 ? 1 - (this.cache.size / totalAccesses) : 0,
    };
  }
}

export class LongTermMemory implements Memory {
  readonly name: string;
  readonly type = 'long-term' as const;
  private entries: MemoryEntry[] = [];
  private cache: InMemoryCache;
  private maxSize: number;

  constructor(name: string, maxSize = 10000, cacheTTL = 3600000) {
    this.name = name;
    this.maxSize = maxSize;
    this.cache = new InMemoryCache(cacheTTL);
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
    this.cache.clear();
  }

  stats(): { count: number; size: number } {
    return {
      count: this.entries.length,
      size: JSON.stringify(this.entries).length,
    };
  }

  /**
   * Cache operations for optimization
   */
  cacheGet(key: string): unknown | undefined {
    return this.cache.get(key);
  }

  cacheSet(key: string, value: unknown, ttl?: number): void {
    this.cache.set(key, value, ttl);
  }

  cacheDelete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Search with caching
   */
  cachedSearch(key: string, query: string, ttl = 60000): MemoryEntry[] | undefined {
    // Check cache first
    const cached = this.cache.get(key) as MemoryEntry[] | undefined;
    if (cached !== undefined) {
      return cached;
    }

    // Perform search and cache result
    const results = this.search(query);
    this.cache.set(key, results, ttl);
    return results;
  }

  /**
   * Bottleneck-to-Subteam cache operations
   */
  cacheBottleneckMapping(bottleneckType: string, subteamSpec: unknown, ttl = 1800000): void {
    this.cache.set(`bottleneck:${bottleneckType}`, subteamSpec, ttl);
  }

  getBottleneckMapping(bottleneckType: string): unknown | undefined {
    return this.cache.get(`bottleneck:${bottleneckType}`);
  }

  /**
   * Team-Generation Direction cache
   */
  cacheTeamDirection(taskType: string, direction: unknown, ttl = 1800000): void {
    this.cache.set(`team_direction:${taskType}`, direction, ttl);
  }

  getTeamDirection(taskType: string): unknown | undefined {
    return this.cache.get(`team_direction:${taskType}`);
  }

  /**
   * Agent/ToM Inference cache
   */
  cacheToMInference(traceKey: string, inference: unknown, ttl = 300000): void {
    this.cache.set(`tom:${traceKey}`, inference, ttl);
  }

  getToMInference(traceKey: string): unknown | undefined {
    return this.cache.get(`tom:${traceKey}`);
  }

  /**
   * Get cache stats
   */
  getCacheStats() {
    return this.cache.stats();
  }
}