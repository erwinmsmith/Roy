import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
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

  it('supports explicitly configured unrestricted execution inside an isolated workspace', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-unrestricted-shell-'));
    const tool = new ShellExecTool({
      mode: 'unrestricted',
      workspaceRoot: workspaceCwd,
      shell: '/bin/sh',
    });
    const result = await tool.execute({
      command: "printf 'benchmark-ready' > artifact.txt && printf 'done'",
    });

    expect(result.success).toBe(true);
    const output = result.result as ShellExecResult;
    expect(output.mode).toBe('unrestricted');
    expect(output.stdout).toBe('done');
    expect(await readFile(path.join(workspaceCwd, 'artifact.txt'), 'utf8')).toBe('benchmark-ready');
  });

  it('lets verbose commands finish while returning a bounded causal failure tail', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-verbose-shell-'));
    const tool = new ShellExecTool({
      mode: 'unrestricted',
      workspaceRoot: workspaceCwd,
      shell: '/bin/sh',
    });
    const result = await tool.execute({
      command: `python3 -c "import sys; sys.stderr.write('x' * 100000 + 'FAILURE_AT_END\\\\n'); sys.exit(2)"`,
      maxOutputBytes: 1024,
    });

    expect(result.success).toBe(false);
    expect(result.error).not.toContain('maxBuffer');
    const output = result.result as ShellExecResult;
    expect(output.exitCode).toBe(2);
    expect(output.stderr).toContain('FAILURE_AT_END');
    expect(output.stderr).toContain('leading bytes');
    expect(Buffer.byteLength(output.stderr)).toBeLessThan(1200);
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
