import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { RuntimeSessionPool } from '../src/server/RuntimeSessionPool.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class SessionTestLLM implements LLMProvider {
  readonly name = 'session-test';
  readonly defaultModel = 'session-test-model';

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'ok', usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } };
  }

  async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'ok', done: true, usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } };
  }

  async completeJSON<T>(): Promise<T> {
    return { action: 'solve_directly', reason: 'Bounded request.' } as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('RuntimeSessionPool', () => {
  it('isolates runtime state by explicit server session ID and reuses the same session runtime', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-server-sessions-'));
    const defaultRuntime = new Runtime();
    const defaultContext = await defaultRuntime.initialize({
      sessionId: 'server-main',
      workspaceCwd: cwd,
      llmProvider: new SessionTestLLM(),
    });
    const pool = new RuntimeSessionPool({
      defaultSessionId: 'server-main',
      defaultRuntime,
      defaultContext,
      workspaceCwd: cwd,
      maxSessions: 2,
      idleTimeoutMs: 1000,
    });

    const alpha = await pool.get('client-alpha');
    const alphaAgain = await pool.get('client-alpha');
    const beta = await pool.get('client-beta');
    expect(alphaAgain).toBe(alpha);
    expect(beta).not.toBe(alpha);
    expect(alpha.getState().sessionId).toBe('client-alpha');
    expect(beta.getState().sessionId).toBe('client-beta');
    expect(pool.list().map(session => session.sessionId)).toEqual(['server-main', 'client-alpha', 'client-beta']);

    alpha.setBudget(1200);
    expect(alpha.getBudgetState().limitTokens).toBe(1200);
    expect(beta.getBudgetState().mode).toBe('unlimited');
    expect(defaultRuntime.getBudgetState().mode).toBe('unlimited');

    await expect(pool.get('client-gamma')).rejects.toThrow('Runtime session limit exceeded');
    expect(() => pool.normalizeSessionId('../invalid')).toThrow('Session ID must be 1-100 characters');
    expect(await pool.close('client-alpha')).toBe(true);
    const replacement = await pool.get('client-alpha');
    expect(replacement).not.toBe(alpha);
    expect(replacement.getBudgetState().mode).toBe('unlimited');

    const expired = await pool.sweepIdle(Date.now() + 1001);
    expect(expired.sort()).toEqual(['client-alpha', 'client-beta']);
    expect(pool.list().map(session => session.sessionId)).toEqual(['server-main']);
    const afterSweep = await pool.get('client-beta');
    expect(afterSweep).not.toBe(beta);

    await pool.shutdown();
    await defaultRuntime.shutdown();
  });
});
