// Configuration Layer with Zod validation and YAML support

import { z } from 'zod';
import yaml from 'js-yaml';
import { readFileSync, existsSync, statSync } from 'fs';
import { dirname, join } from 'path';

// Environment variable schema
export const EnvSchema = z.object({
  // LLM Providers
  ANTHROPIC_API_KEY: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_BASE_URL: z.string().optional(),

  // Default Model
  DEFAULT_MODEL: z.string().default('claude-sonnet-4-20250514'),

  // Server
  PORT: z.string().default('3000'),

  // Logging
  LOG_LEVEL: z.enum(['debug', 'info', 'warning', 'error']).default('info'),
  LOG_FILE: z.string().optional(),
  LOG_BATCH_SIZE: z.string().transform(v => parseInt(v, 10)).default('100'),
  LOG_FLUSH_INTERVAL: z.string().transform(v => parseFloat(v)).default('2.0'),

  // Path settings
  CONFIG_PATH: z.string().optional(),
  SECRETS_PATH: z.string().optional(),
});

// Type from schema
export type EnvConfig = z.infer<typeof EnvSchema>;

// Server config schema
export const ServerConfigSchema = z.object({
  port: z.number().default(3000),
  host: z.string().default('0.0.0.0'),
  cors: z.object({
    origin: z.string().default('*'),
    methods: z.array(z.string()).default(['GET', 'POST']),
  }).optional(),
});

// LLM config schema
export const LLMConfigSchema = z.object({
  provider: z.enum(['anthropic', 'openai']).default('anthropic'),
  model: z.string().default('claude-sonnet-4-20250514'),
  apiKey: z.string().optional(),
  baseUrl: z.string().optional(),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().default(4096),
  timeout: z.number().default(60000),
});

// Logger config schema
export const LoggerConfigSchema = z.object({
  level: z.enum(['debug', 'info', 'warning', 'error']).default('info'),
  transports: z.array(z.enum(['none', 'console', 'file', 'http'])).default(['console']),
  path: z.string().default('roy.jsonl'),
  batchSize: z.number().default(100),
  flushInterval: z.number().default(2.0),
  maxQueueSize: z.number().default(2048),
  httpEndpoint: z.string().optional(),
  httpHeaders: z.record(z.string()).optional(),
  httpTimeout: z.number().default(5.0),
  progressDisplay: z.boolean().default(false),
});

// Logging event types
export const EventTypeSchema = z.enum([
  'debug', 'info', 'warning', 'error', 'progress'
]);

export type EventType = z.infer<typeof EventTypeSchema>;

// Full app config schema
export const AppConfigSchema = z.object({
  server: ServerConfigSchema.optional(),
  llm: LLMConfigSchema.optional(),
  logger: LoggerConfigSchema.optional(),
  env: EnvSchema.optional(),
});

export type AppConfig = z.infer<typeof AppConfigSchema>;

/**
 * Configuration loader
 * Supports: env vars, YAML files, defaults
 */
export class ConfigLoader {
  private config: AppConfig;
  private configPath: string | null = null;
  private secretsPath: string | null = null;

  constructor() {
    this.config = this.loadConfig();
  }

  /**
   * Load configuration from multiple sources with priority:
   * 1. Environment variables
   * 2. YAML config file
   * 3. YAML secrets file
   * 4. Defaults
   */
  private loadConfig(): AppConfig {
    const defaults: AppConfig = {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      llm: {
        provider: 'anthropic',
        model: process.env.DEFAULT_MODEL || 'claude-sonnet-4-20250514',
        apiKey: process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY,
        baseUrl: process.env.OPENAI_BASE_URL,
        temperature: 0.7,
        maxTokens: 4096,
        timeout: 60000,
      },
      logger: {
        level: (process.env.LOG_LEVEL as 'debug' | 'info' | 'warning' | 'error') || 'info',
        transports: ['console'],
        path: 'roy.jsonl',
        batchSize: 100,
        flushInterval: 2.0,
        maxQueueSize: 2048,
        progressDisplay: false,
        httpTimeout: 5.0,
      },
    };

    // Find and load config files
    const configPath = this.findConfigFile();
    if (configPath) {
      this.configPath = configPath;
      const yamlConfig = this.loadYamlFile(configPath);
      this.mergeConfig(defaults, yamlConfig);

      // Try to load secrets file
      const secretsPath = this.findSecretsFile(dirname(configPath));
      if (secretsPath) {
        this.secretsPath = secretsPath;
        const secrets = this.loadYamlFile(secretsPath);
        this.mergeConfig(defaults, secrets);
      }
    }

    // Load and validate from environment
    const envConfig = this.loadFromEnv();
    this.mergeConfig(defaults, envConfig);

    return defaults;
  }

  /**
   * Find config file in current or parent directories
   */
  private findConfigFile(): string | null {
    const candidates = ['roy.config.yaml', 'roy.config.yml'];
    return this.findFile(candidates);
  }

  /**
   * Find secrets file
   */
  private findSecretsFile(dir: string): string | null {
    const candidates = ['roy.secrets.yaml', 'roy.secrets.yml'];
    const searchDir = dir || process.cwd();

    for (const candidate of candidates) {
      const path = join(searchDir, candidate);
      if (existsSync(path)) {
        return path;
      }
    }

    return null;
  }

  /**
   * Search for file in current and parent directories
   */
  private findFile(candidates: string[]): string | null {
    let currentDir = process.cwd();

    while (currentDir !== dirname(currentDir)) {
      for (const candidate of candidates) {
        const path = join(currentDir, candidate);
        if (existsSync(path) && statSync(path).isFile()) {
          return path;
        }
      }
      currentDir = dirname(currentDir);
    }

    return null;
  }

  /**
   * Load YAML file
   */
  private loadYamlFile(filePath: string): Partial<AppConfig> {
    try {
      const content = readFileSync(filePath, 'utf-8');
      const parsed = yaml.load(content);
      return this.validateYaml(parsed);
    } catch (error) {
      console.warn(`Failed to load YAML file ${filePath}:`, error);
      return {};
    }
  }

  /**
   * Validate YAML content against schema
   */
  private validateYaml(data: unknown): Partial<AppConfig> {
    const result = AppConfigSchema.safeParse(data);
    if (result.success) {
      return result.data;
    }

    // Try partial validation for nested objects
    const partial: Partial<AppConfig> = {};

    if (data && typeof data === 'object') {
      const d = data as Record<string, unknown>;

      if (d.server) {
        const serverResult = ServerConfigSchema.safeParse(d.server);
        if (serverResult.success) {
          partial.server = serverResult.data;
        }
      }

      if (d.llm) {
        const llmResult = LLMConfigSchema.safeParse(d.llm);
        if (llmResult.success) {
          partial.llm = llmResult.data;
        }
      }

      if (d.logger) {
        const loggerResult = LoggerConfigSchema.safeParse(d.logger);
        if (loggerResult.success) {
          partial.logger = loggerResult.data;
        }
      }
    }

    return partial;
  }

  /**
   * Load config from environment variables
   */
  private loadFromEnv(): Partial<AppConfig> {
    const config: Partial<AppConfig> = {};

    // Server from env
    if (process.env.PORT) {
      config.server = {
        port: parseInt(process.env.PORT, 10),
        host: '0.0.0.0',
      };
    }

    // Logger from env
    if (process.env.LOG_LEVEL) {
      config.logger = {
        level: process.env.LOG_LEVEL as 'debug' | 'info' | 'warning' | 'error',
        transports: ['console'],
        path: 'roy.jsonl',
        batchSize: 100,
        flushInterval: 2.0,
        maxQueueSize: 2048,
        progressDisplay: false,
        httpTimeout: 5.0,
      };
    }

    return config;
  }

  /**
   * Deep merge two configs
   */
  private mergeConfig(base: AppConfig, update: Partial<AppConfig>): void {
    if (!update) return;

    if (update.server) {
      base.server = { ...base.server, ...update.server };
    }
    if (update.llm) {
      base.llm = { ...base.llm, ...update.llm };
    }
    if (update.logger) {
      base.logger = { ...base.logger, ...update.logger };
    }
  }

  /**
   * Get the full config
   */
  getConfig(): AppConfig {
    return { ...this.config };
  }

  /**
   * Get server config
   */
  getServerConfig() {
    return this.config.server;
  }

  /**
   * Get LLM config
   */
  getLLMConfig() {
    return this.config.llm;
  }

  /**
   * Get logger config
   */
  getLoggerConfig() {
    return this.config.logger;
  }

  /**
   * Get config file paths
   */
  getConfigPaths(): { config: string | null; secrets: string | null } {
    return {
      config: this.configPath,
      secrets: this.secretsPath,
    };
  }

  /**
   * Reload configuration from files
   */
  reload(): AppConfig {
    this.config = this.loadConfig();
    return this.config;
  }
}

// Singleton instance
export const configLoader = new ConfigLoader();

// Global config - initialized with defaults
export const config: AppConfig = configLoader.getConfig();

// Convenience exports
export function getConfig(): AppConfig {
  return configLoader.getConfig();
}

export function getServerConfig() {
  return configLoader.getServerConfig();
}

export function getLLMConfig() {
  return configLoader.getLLMConfig();
}

export function getLoggerConfig() {
  return configLoader.getLoggerConfig();
}

export default configLoader;
