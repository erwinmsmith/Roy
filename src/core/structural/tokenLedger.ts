export interface StructuralTokenLedgerState {
  limit: number;
  used: number;
  remaining: number;
  exhausted: boolean;
}

export class StructuralTokenLedger {
  private used = 0;

  constructor(readonly limit: number) {
    if (!Number.isSafeInteger(limit) || limit <= 0) throw new Error('Token ledger limit must be a positive integer');
  }

  reserve(tokens: number): void {
    this.assertTokens(tokens);
    if (this.used + tokens > this.limit) {
      throw new Error(`Structural token budget exceeded: requested ${tokens}, remaining ${this.limit - this.used}`);
    }
    this.used += tokens;
  }

  refund(tokens: number): void {
    this.assertTokens(tokens);
    this.used = Math.max(0, this.used - tokens);
  }

  state(): StructuralTokenLedgerState {
    return {
      limit: this.limit,
      used: this.used,
      remaining: this.limit - this.used,
      exhausted: this.used >= this.limit,
    };
  }

  restore(state: Pick<StructuralTokenLedgerState, 'limit' | 'used'>): void {
    if (state.limit !== this.limit) throw new Error('Token ledger limit mismatch');
    if (!Number.isSafeInteger(state.used) || state.used < 0 || state.used > this.limit) {
      throw new Error('Invalid token ledger state');
    }
    this.used = state.used;
  }

  private assertTokens(tokens: number): void {
    if (!Number.isSafeInteger(tokens) || tokens < 0) throw new Error('Token count must be a non-negative integer');
  }
}
