// Bootstrap - Thin wrapper using Runtime for Roy Agent System
// Entry points (CLI/Server) call bootstrap() which delegates to Runtime

import { runtime, type RuntimeContext, type RuntimeConfig } from './core/runtime/Runtime.js';

// Re-export types for backward compatibility
export type BootstrapOptions = RuntimeConfig;
export type BootstrapContext = RuntimeContext;

/**
 * Bootstrap the Roy agent system
 * Delegates to Runtime for initialization
 */
export async function bootstrap(options: RuntimeConfig = {}): Promise<RuntimeContext> {
  return await runtime.initialize(options);
}

/**
 * Cleanup resources from bootstrap
 */
export async function cleanup(_context?: RuntimeContext): Promise<void> {
  await runtime.shutdown();
}

// Default export for convenience
export default { bootstrap, cleanup };