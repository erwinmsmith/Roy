import { toolRegistry } from './registry.js';
import { FsListTool, FsReadTool } from './fsTools.js';
import { ShellExecTool } from './shellExec.js';
import { WebFetchTool, WebSearchTool, type WebToolConfig } from './webTools.js';

export function registerCoreTools(options: { web?: Partial<WebToolConfig> } = {}): void {
  if (!toolRegistry.has('fs.list')) {
    toolRegistry.register(new FsListTool(), 'filesystem');
  }
  if (!toolRegistry.has('fs.read')) {
    toolRegistry.register(new FsReadTool(), 'filesystem');
  }
  if (!toolRegistry.has('shell.exec')) {
    toolRegistry.register(new ShellExecTool(), 'system');
  }
  if (options.web?.enabled !== false && !toolRegistry.has('web.search')) {
    toolRegistry.register(new WebSearchTool(options.web), 'web');
  }
  if (options.web?.enabled !== false && !toolRegistry.has('web.fetch')) {
    toolRegistry.register(new WebFetchTool(options.web), 'web');
  }
}
