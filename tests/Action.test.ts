import { describe, it, expect, beforeEach } from 'vitest';
import { Action, type ActionConfig, type ActionResult } from '../src/actions/Action.js';

// Test action implementation
class TestAction extends Action {
  async execute(params: Record<string, unknown>): Promise<ActionResult> {
    return {
      success: true,
      result: `TestAction executed with: ${params.message}`,
    };
  }
}

// Test streaming action
class TestStreamingAction extends Action {
  readonly isStream = true;

  async *executeStream(params: Record<string, unknown>): AsyncGenerator<string> {
    const messages = params.messages as string[] || ['hello', 'world'];
    for (const msg of messages) {
      yield msg;
    }
  }

  async execute(params: Record<string, unknown>): Promise<ActionResult> {
    return {
      success: true,
      result: 'streaming action',
    };
  }
}

describe('Action', () => {
  describe('constructor', () => {
    it('should create action with required fields', () => {
      const action = new TestAction({ name: 'test-action' });
      expect(action.name).toBe('test-action');
      expect(action.description).toBe('');
      expect(action.parameters).toEqual([]);
      expect(action.examples).toEqual([]);
    });

    it('should create action with all fields', () => {
      const config: ActionConfig = {
        name: 'full-action',
        description: 'A test action',
        parameters: [
          { name: 'param1', type: 'string', required: true },
        ],
        examples: [
          { input: { param1: 'value' }, output: 'result' },
        ],
      };

      const action = new TestAction(config);
      expect(action.name).toBe('full-action');
      expect(action.description).toBe('A test action');
      expect(action.parameters).toHaveLength(1);
      expect(action.examples).toHaveLength(1);
    });
  });

  describe('validate()', () => {
    it('should pass validation when params are correct', () => {
      const action = new TestAction({
        name: 'validate-test',
        parameters: [
          { name: 'message', type: 'string', required: true },
          { name: 'count', type: 'number', required: false },
        ],
      });

      const result = action.validate({ message: 'hello', count: 5 });
      expect(result.valid).toBe(true);
      expect(result.errors).toBeUndefined();
    });

    it('should fail when required param is missing', () => {
      const action = new TestAction({
        name: 'required-test',
        parameters: [
          { name: 'required-param', type: 'string', required: true },
        ],
      });

      const result = action.validate({});
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Required parameter "required-param" is missing');
    });

    it('should fail when param type is wrong', () => {
      const action = new TestAction({
        name: 'type-test',
        parameters: [
          { name: 'count', type: 'number', required: true },
        ],
      });

      const result = action.validate({ count: 'not a number' });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Parameter "count" should be number, got string');
    });

    it('should fail when param value not in enum', () => {
      const action = new TestAction({
        name: 'enum-test',
        parameters: [
          { name: 'size', type: 'string', required: true, enum: ['small', 'medium', 'large'] },
        ],
      });

      const result = action.validate({ size: 'huge' });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Parameter "size" must be one of: small, medium, large');
    });

    it('should allow null/undefined for non-required params', () => {
      const action = new TestAction({
        name: 'optional-test',
        parameters: [
          { name: 'optional', type: 'string', required: false },
        ],
      });

      const result = action.validate({});
      expect(result.valid).toBe(true);
    });
  });

  describe('execute()', () => {
    it('should execute action and return result', async () => {
      const action = new TestAction({ name: 'execute-test' });
      const result = await action.execute({ message: 'test' });

      expect(result.success).toBe(true);
      expect(result.result).toContain('test');
    });
  });

  describe('executeStream()', () => {
    it('should yield streaming results', async () => {
      const action = new TestStreamingAction({ name: 'stream-test' });
      const chunks: string[] = [];

      for await (const chunk of action.executeStream({ messages: ['a', 'b', 'c'] })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['a', 'b', 'c']);
    });
  });

  describe('getSchema()', () => {
    it('should return action schema', () => {
      const action = new TestAction({
        name: 'schema-test',
        description: 'Test description',
        parameters: [
          { name: 'param1', type: 'string', required: true, description: 'A parameter' },
        ],
        examples: [
          { input: { param1: 'value' }, output: 'result' },
        ],
      });

      const schema = action.getSchema();

      expect(schema.name).toBe('schema-test');
      expect(schema.description).toBe('Test description');
      expect(schema.parameters).toHaveLength(1);
      expect(schema.parameters[0]).toEqual({
        name: 'param1',
        type: 'string',
        required: true,
        description: 'A parameter',
        enum: undefined,
      });
      expect(schema.examples).toHaveLength(1);
    });
  });

  describe('toJSONString()', () => {
    it('should return JSON string representation', () => {
      const action = new TestAction({
        name: 'json-test',
        description: 'Test',
        parameters: [
          { name: 'count', type: 'number', required: true },
        ],
      });

      const json = action.toJSONString();
      const parsed = JSON.parse(json);

      expect(parsed.name).toBe('json-test');
      expect(parsed.parameters).toHaveLength(1);
    });
  });

  describe('isStream', () => {
    it('should default to false', () => {
      const action = new TestAction({ name: 'non-stream' });
      expect(action.isStream).toBe(false);
    });

    it('should be true for streaming action', () => {
      const action = new TestStreamingAction({ name: 'stream' });
      expect(action.isStream).toBe(true);
    });
  });
});