export interface TaskEmbeddingProvider {
  embed(text: string): number[];
  similarity(left: string, right: string): number;
}

export class HashTaskEmbeddingProvider implements TaskEmbeddingProvider {
  constructor(private readonly dimensions = 96) {}

  embed(text: string): number[] {
    const vector = Array.from({ length: this.dimensions }, () => 0);
    for (const token of this.tokenize(text)) {
      const hash = this.hash(token);
      const index = Math.abs(hash) % this.dimensions;
      vector[index] += hash % 2 === 0 ? 1 : -1;
    }
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return norm === 0 ? vector : vector.map(value => value / norm);
  }

  similarity(left: string, right: string): number {
    const a = this.embed(left);
    const b = this.embed(right);
    return Math.max(0, Math.min(1, a.reduce((sum, value, index) => sum + value * b[index], 0)));
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}._/-]+/gu, ' ')
      .split(/\s+/)
      .filter(Boolean);
  }

  private hash(value: string): number {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash | 0;
  }
}
