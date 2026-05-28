import { describe, it, expect, beforeEach } from 'vitest';
import { actionRegistry } from '../src/core/actions/index.js';
import { Action, type ActionResult } from '../src/core/actions/Action.js';

// Create test actions
class EchoAction extends Action {
  async execute(params: Record<string, unknown>): Promise<ActionResult> {
    return {
      success: true,
      result: params.message || 'echo',
    };
  }
}

class AddAction extends Action {
  async execute(params: Record<string, unknown>): Promise<ActionResult> {
    const a = params.a as number;
    const b = params.b as number;
    return {
      success: true,
      result: a + b,
    };
  }
}

describe('ActionRegistry', () => {
  beforeEach(() => {
    actionRegistry.clear();
  });

  describe('register()', () => {
    it('should register an action', () => {
      const action = new EchoAction({ name: 'echo' });
      actionRegistry.register(action);

      expect(actionRegistry.has('echo')).toBe(true);
      expect(actionRegistry.get('echo')).toBe(action);
    });

    it('should register with category', () => {
      const action = new EchoAction({ name: 'cat-echo' });
      actionRegistry.register(action, 'test-category');

      const byCategory = actionRegistry.getByCategory('test-category');
      expect(byCategory).toHaveLength(1);
      expect(byCategory[0].name).toBe('cat-echo');
    });

    it('should warn when overwriting', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      actionRegistry.register(new EchoAction({ name: 'dup' }));
      actionRegistry.register(new EchoAction({ name: 'dup' }));

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringMatching(/^\[WARN\]/),
        'Action "dup" already registered, overwriting'
      );
      warnSpy.mockRestore();
    });
  });

  describe('unregister()', () => {
    it('should unregister an action', () => {
      actionRegistry.register(new EchoAction({ name: 'remove-me' }));
      expect(actionRegistry.has('remove-me')).toBe(true);

      const result = actionRegistry.unregister('remove-me');
      expect(result).toBe(true);
      expect(actionRegistry.has('remove-me')).toBe(false);
    });

    it('should return false for non-existent action', () => {
      const result = actionRegistry.unregister('non-existent');
      expect(result).toBe(false);
    });
  });

  describe('list()', () => {
    it('should list all registered actions', () => {
      actionRegistry.register(new EchoAction({ name: 'action1' }));
      actionRegistry.register(new AddAction({ name: 'action2' }));

      const actions = actionRegistry.list();
      expect(actions).toHaveLength(2);
      expect(actions.map(a => a.name)).toContain('action1');
      expect(actions.map(a => a.name)).toContain('action2');
    });
  });

  describe('execute()', () => {
    it('should execute a registered action', async () => {
      actionRegistry.register(new EchoAction({ name: 'execute-test' }));

      const result = await actionRegistry.execute('execute-test', { message: 'hello' });
      expect(result.success).toBe(true);
      expect(result.result).toBe('hello');
    });

    it('should return error for non-existent action', async () => {
      const result = await actionRegistry.execute('non-existent', {});
      expect(result.success).toBe(false);
      expect(result.error).toContain('not found');
    });

    it('should validate parameters before execution', async () => {
      const action = new AddAction({
        name: 'add-validation',
        parameters: [
          { name: 'a', type: 'number', required: true },
          { name: 'b', type: 'number', required: true },
        ],
      });
      actionRegistry.register(action);

      const result = await actionRegistry.execute('add-validation', { a: 5 }); // missing b
      expect(result.success).toBe(false);
      expect(result.error).toContain('Validation failed');
    });
  });

  describe('executeStream()', () => {
    it('should stream results from streaming action', async () => {
      // Create a streaming action
      class StreamAction extends Action {
        readonly isStream = true;

        async *executeStream(): AsyncGenerator<string> {
          yield 'chunk1';
          yield 'chunk2';
          yield 'chunk3';
        }

        async execute(): Promise<ActionResult> {
          return { success: true, result: 'streaming' };
        }
      }

      actionRegistry.register(new StreamAction({ name: 'stream-test' }));

      const chunks: string[] = [];
      for await (const chunk of actionRegistry.executeStream('stream-test', {})) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['chunk1', 'chunk2', 'chunk3']);
    });

    it('should handle non-streaming action in stream mode', async () => {
      actionRegistry.register(new EchoAction({ name: 'non-stream-echo' }));

      const chunks: string[] = [];
      for await (const chunk of actionRegistry.executeStream('non-stream-echo', { message: 'test' })) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toBe('test');
    });
  });

  describe('formatActionsForPrompt()', () => {
    it('should format all actions for LLM prompt', () => {
      actionRegistry.register(new EchoAction({
        name: 'echo-action',
        description: 'Echoes back the message',
        parameters: [
          { name: 'message', type: 'string', required: true, description: 'Message to echo' },
        ],
      }));

      const formatted = actionRegistry.formatActionsForPrompt();
      expect(formatted).toContain('- echo-action: Echoes back the message');
      expect(formatted).toContain('message (string) (required)');
    });
  });

  describe('keys()', () => {
    it('should return action names', () => {
      actionRegistry.register(new EchoAction({ name: 'key1' }));
      actionRegistry.register(new EchoAction({ name: 'key2' }));

      const keys = actionRegistry.keys();
      expect(keys).toContain('key1');
      expect(keys).toContain('key2');
    });
  });

  describe('getMetadata()', () => {
    it('should return action metadata', () => {
      actionRegistry.register(new EchoAction({
        name: 'meta-test',
        description: 'Metadata test',
        parameters: [
          { name: 'msg', type: 'string' },
        ],
      }));

      const metadata = actionRegistry.getMetadata('meta-test');
      expect(metadata).toBeDefined();
      expect(metadata?.name).toBe('meta-test');
      expect(metadata?.description).toBe('Metadata test');
      expect(metadata?.parameters).toHaveLength(1);
    });

    it('should return undefined for non-existent action', () => {
      const metadata = actionRegistry.getMetadata('non-existent');
      expect(metadata).toBeUndefined();
    });
  });
});
