import { describe, expect, it } from 'vitest';
import { UnifiedAgent } from '../src/core/agent/UnifiedAgent.js';
import { AgentManager } from '../src/core/runtime/AgentManager.js';
import { memoryRegistry } from '../src/core/memory/index.js';
import type {
  LLMCompletionOptions,
  LLMCompletionResult,
  LLMMessage,
  LLMProvider,
  LLMStreamChunk,
} from '../src/core/llm/types.js';

class ConversationLLM implements LLMProvider {
  readonly name = 'conversation-test';
  readonly defaultModel = 'test';

  async complete(): Promise<LLMCompletionResult> {
    return { content: 'Roy response' };
  }

  async *stream(
    _messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'Roy ', done: false };
    yield { content: 'response', done: true };
  }

  async completeJSON<T>(): Promise<T> {
    return {} as T;
  }

  isConfigured(): boolean {
    return true;
  }
}

describe('AgentManager sessions', () => {
  it('initializes an interactive agent and drains its streamed environment response', async () => {
    const manager = new AgentManager();
    const agent = new UnifiedAgent({
      id: 'root',
      name: 'Roy',
      role: 'root',
      mode: 'conversational',
      llm: new ConversationLLM(),
    });
    manager.addAgent(agent);
    manager.setInteractWithEnv('Roy');
    manager.createSession('manager-test');

    const chunks: string[] = [];
    for await (const chunk of manager.streamResponse('manager-test', 'hello')) chunks.push(chunk);

    expect(chunks).toEqual(['Roy ', 'response']);
    expect(agent.getInfo()).toMatchObject({ state: 'idle', memoryMessages: 2 });
    await manager.closeSession('manager-test');
    expect(agent.getInfo().memoryMessages).toBe(0);
    expect(manager.listSessions()).toEqual([]);
  });

  it('rejects agents that do not expose interactive steps', async () => {
    const manager = new AgentManager();
    const agent = new UnifiedAgent({
      name: 'Roy',
      mode: 'conversational',
      llm: new ConversationLLM(),
    });
    manager.addAgent(agent);
    manager.setInteractWithEnv('missing');
    manager.createSession('invalid-manager-test');

    await expect(manager.sendToEnv('invalid-manager-test', 'hello')).rejects.toThrow(
      'Agent missing not found'
    );
    await manager.cleanup();
  });

  it('keeps conversation memory isolated across manager sessions', async () => {
    const manager = new AgentManager();
    const agent = new UnifiedAgent({
      name: 'Roy-Isolated',
      mode: 'conversational',
      llm: new ConversationLLM(),
    });
    manager.addAgent(agent);
    manager.setInteractWithEnv('Roy-Isolated');
    manager.createSession('session-a');
    manager.createSession('session-b');

    for await (const _chunk of manager.streamResponse('session-a', 'from a')) {
      // Drain the response.
    }
    for await (const _chunk of manager.streamResponse('session-b', 'from b')) {
      // Drain the response.
    }

    expect(memoryRegistry.getStats('Roy-Isolated', 'session-a').shortTerm.count).toBe(2);
    expect(memoryRegistry.getStats('Roy-Isolated', 'session-b').shortTerm.count).toBe(2);

    await manager.closeSession('session-a');
    expect(memoryRegistry.getStats('Roy-Isolated', 'session-a').shortTerm.count).toBe(0);
    expect(memoryRegistry.getStats('Roy-Isolated', 'session-b').shortTerm.count).toBe(2);
    await manager.cleanup();
  });
});
