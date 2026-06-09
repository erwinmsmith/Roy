import { toolRegistry } from './registry.js';
import { FsListTool, FsReadTool } from './fsTools.js';
import { ShellExecTool } from './shellExec.js';

export function registerCoreTools(): void {
  if (!toolRegistry.has('fs.list')) {
    toolRegistry.register(new FsListTool(), 'filesystem');
  }
  if (!toolRegistry.has('fs.read')) {
    toolRegistry.register(new FsReadTool(), 'filesystem');
  }
  if (!toolRegistry.has('shell.exec')) {
    toolRegistry.register(new ShellExecTool(), 'system');
  }
}
