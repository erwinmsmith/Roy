import { toolRegistry } from './registry.js';
import {
  FsListTool,
  FsReadTool,
  FsReplaceTool,
  FsSearchTool,
  FsSynthesizeTool,
  FsWriteTool,
} from './fsTools.js';
import { ShellExecTool, type ShellExecConfig } from './shellExec.js';
import { WebFetchTool, WebSearchTool, type WebToolConfig } from './webTools.js';

export function registerCoreTools(options: {
  web?: Partial<WebToolConfig>;
  shell?: Partial<ShellExecConfig>;
  workspaceRoot?: string;
} = {}): void {
  if (!toolRegistry.has('fs.list')) {
    toolRegistry.register(new FsListTool(options.workspaceRoot), 'filesystem');
  }
  if (!toolRegistry.has('fs.read')) {
    toolRegistry.register(new FsReadTool(options.workspaceRoot), 'filesystem');
  }
  if (!toolRegistry.has('fs.search')) {
    toolRegistry.register(new FsSearchTool(options.workspaceRoot), 'filesystem');
  }
  if (!toolRegistry.has('fs.replace')) {
    toolRegistry.register(new FsReplaceTool(options.workspaceRoot), 'filesystem');
  }
  if (!toolRegistry.has('fs.write')) {
    toolRegistry.register(new FsWriteTool(options.workspaceRoot), 'filesystem');
  }
  if (!toolRegistry.has('fs.synthesize')) {
    toolRegistry.register(new FsSynthesizeTool(), 'filesystem');
  }
  if (!toolRegistry.has('shell.exec')) {
    toolRegistry.register(new ShellExecTool({
      ...options.shell,
      workspaceRoot: options.workspaceRoot ?? options.shell?.workspaceRoot,
    }), 'system');
  }
  if (options.web?.enabled !== false && !toolRegistry.has('web.search')) {
    toolRegistry.register(new WebSearchTool(options.web), 'web');
  }
  if (options.web?.enabled !== false && !toolRegistry.has('web.fetch')) {
    toolRegistry.register(new WebFetchTool(options.web), 'web');
  }
}
