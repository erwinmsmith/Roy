import { toolRegistry } from './registry.js';
import { ShellExecTool } from './shellExec.js';

export function registerCoreTools(): void {
  if (!toolRegistry.has('shell.exec')) {
    toolRegistry.register(new ShellExecTool(), 'system');
  }
}
