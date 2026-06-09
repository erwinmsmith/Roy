import { describe, expect, it } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime from '../src/core/runtime/Runtime.js';
import { ShellExecTool, registerCoreTools, toolRegistry } from '../src/core/tools/index.js';
import type { ShellExecResult } from '../src/core/tools/shellExec.js';

describe('shell.exec tool', () => {
  it('executes allowlisted read-only commands', async () => {
    const tool = new ShellExecTool();
    const result = await tool.execute({ command: 'pwd' });

    expect(result.success).toBe(true);
    const output = result.result as ShellExecResult;
    expect(output.executable).toBe('pwd');
    expect(output.stdout.trim()).toBe(process.cwd());
    expect(output.exitCode).toBe(0);
  });

  it('rejects commands outside the allowlist', async () => {
    const tool = new ShellExecTool();

    const gitPush = await tool.execute({ command: 'git push' });
    expect(gitPush.success).toBe(false);
    expect(gitPush.error).toContain('not allowlisted');

    const npmInstall = await tool.execute({ command: 'npm install' });
    expect(npmInstall.success).toBe(false);
    expect(npmInstall.error).toContain('not allowlisted');
  });

  it('returns validation-style errors for malformed command strings', async () => {
    const tool = new ShellExecTool();
    const result = await tool.execute({ command: 'ls "src' });

    expect(result.success).toBe(false);
    expect(result.error).toContain('Unclosed quote');
  });

  it('keeps cwd and path arguments inside the workspace', async () => {
    const tool = new ShellExecTool();

    const badCwd = await tool.execute({ command: 'pwd', cwd: '../' });
    expect(badCwd.success).toBe(false);
    expect(badCwd.error).toContain('cwd must stay inside');

    const badPath = await tool.execute({ command: 'cat ../package.json' });
    expect(badPath.success).toBe(false);
    expect(badPath.error).toContain('path arguments');
  });

  it('registers as a core tool available to Runtime agents', async () => {
    toolRegistry.clear();
    registerCoreTools();
    expect(toolRegistry.has('shell.exec')).toBe(true);

    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-shell-tool-runtime-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'shell-tool-runtime-test',
      workspaceCwd,
      fsmEnabled: false,
      llmProvider: undefined,
    });

    const tools = runtime.getContext().agent.getCapabilities().tools;
    expect(tools).toContain('shell.exec');

    await runtime.shutdown();
  });
});
