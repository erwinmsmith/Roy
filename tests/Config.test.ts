import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  EnvSchema,
  ServerConfigSchema,
  LLMConfigSchema,
  LoggerConfigSchema,
  AppConfigSchema,
} from '../src/config/index.js';

describe('Config Schemas', () => {
  describe('EnvSchema', () => {
    it('should parse valid environment config', () => {
      const result = EnvSchema.safeParse({
        ANTHROPIC_API_KEY: 'test-key',
        DEFAULT_MODEL: 'claude-sonnet-4-20250514',
        PORT: '3000',
        LOG_LEVEL: 'info',
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.ANTHROPIC_API_KEY).toBe('test-key');
        expect(result.data.DEFAULT_MODEL).toBe('claude-sonnet-4-20250514');
        expect(result.data.PORT).toBe('3000');
      }
    });

    it('should use defaults for missing values', () => {
      const result = EnvSchema.safeParse({});

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.DEFAULT_MODEL).toBe('claude-sonnet-4-20250514');
        expect(result.data.PORT).toBe('3000');
        expect(result.data.LOG_LEVEL).toBe('info');
      }
    });

    it('should reject invalid log level', () => {
      const result = EnvSchema.safeParse({
        LOG_LEVEL: 'invalid',
      });

      expect(result.success).toBe(false);
    });
  });

  describe('ServerConfigSchema', () => {
    it('should parse valid server config', () => {
      const result = ServerConfigSchema.safeParse({
        port: 8080,
        host: 'localhost',
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.port).toBe(8080);
        expect(result.data.host).toBe('localhost');
      }
    });

    it('should use defaults', () => {
      const result = ServerConfigSchema.safeParse({});

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.port).toBe(3000);
        expect(result.data.host).toBe('0.0.0.0');
      }
    });

    it('should validate port is a number', () => {
      const result = ServerConfigSchema.safeParse({
        port: 'not-a-number',
      });

      expect(result.success).toBe(false);
    });
  });

  describe('LLMConfigSchema', () => {
    it('should parse valid LLM config', () => {
      const result = LLMConfigSchema.safeParse({
        provider: 'anthropic',
        model: 'claude-sonnet-4-20250514',
        apiKey: 'test-key',
        temperature: 0.8,
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.provider).toBe('anthropic');
        expect(result.data.model).toBe('claude-sonnet-4-20250514');
        expect(result.data.temperature).toBe(0.8);
      }
    });

    it('should validate temperature range', () => {
      const lowResult = LLMConfigSchema.safeParse({
        temperature: -1,
      });
      expect(lowResult.success).toBe(false);

      const highResult = LLMConfigSchema.safeParse({
        temperature: 3,
      });
      expect(highResult.success).toBe(false);
    });

    it('should validate provider enum', () => {
      const result = LLMConfigSchema.safeParse({
        provider: 'invalid',
      });

      expect(result.success).toBe(false);
    });
  });

  describe('LoggerConfigSchema', () => {
    it('should parse valid logger config', () => {
      const result = LoggerConfigSchema.safeParse({
        level: 'debug',
        transports: ['console', 'file'],
        batchSize: 200,
        flushInterval: 5.0,
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.level).toBe('debug');
        expect(result.data.transports).toContain('console');
        expect(result.data.transports).toContain('file');
        expect(result.data.batchSize).toBe(200);
      }
    });

    it('should use defaults', () => {
      const result = LoggerConfigSchema.safeParse({});

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.level).toBe('info');
        expect(result.data.transports).toEqual(['console']);
        expect(result.data.batchSize).toBe(100);
      }
    });

    it('should validate transport types', () => {
      const result = LoggerConfigSchema.safeParse({
        transports: ['invalid-transport'],
      });

      expect(result.success).toBe(false);
    });
  });

  describe('AppConfigSchema', () => {
    it('should parse full app config', () => {
      const result = AppConfigSchema.safeParse({
        server: {
          port: 4000,
          host: '0.0.0.0',
        },
        llm: {
          provider: 'openai',
          model: 'gpt-4',
        },
        logger: {
          level: 'debug',
          transports: ['console'],
        },
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.server?.port).toBe(4000);
        expect(result.data.llm?.provider).toBe('openai');
        expect(result.data.logger?.level).toBe('debug');
      }
    });

    it('should allow partial config', () => {
      const result = AppConfigSchema.safeParse({
        server: {
          port: 5000,
        },
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.server?.port).toBe(5000);
        expect(result.data.llm).toBeUndefined();
      }
    });

    it('should allow empty config', () => {
      const result = AppConfigSchema.safeParse({});
      expect(result.success).toBe(true);
    });
  });
});