import { execFile } from 'node:child_process';
import path from 'node:path';
import { promisify } from 'node:util';
import type { Tool, ToolResult } from './types.js';

const execFileAsync = promisify(execFile);

interface CommandPolicy {
  executable: string;
  allowedArgs?: string[][];
  allowAnyArgs?: boolean;
  readOnlyPaths?: boolean;
}

export interface ShellExecResult {
  command: string;
  executable: string;
  args: string[];
  cwd: string;
  stdout: string;
  stderr: string;
  exitCode: number;
  timedOut: boolean;
}

const COMMAND_POLICIES: CommandPolicy[] = [
  { executable: 'pwd', allowedArgs: [[]] },
  { executable: 'ls', allowAnyArgs: true, readOnlyPaths: true },
  { executable: 'cat', allowAnyArgs: true, readOnlyPaths: true },
  { executable: 'rg', allowAnyArgs: true, readOnlyPaths: true },
  { executable: 'git', allowedArgs: [['status'], ['diff'], ['branch'], ['log'], ['show']] },
  { executable: 'npm', allowedArgs: [['test'], ['run', 'build'], ['run', 'check'], ['run', 'lint']] },
  { executable: 'node', allowedArgs: [['--version'], ['-v']] },
  { executable: 'npm', allowedArgs: [['--version'], ['-v']] },
];

const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_OUTPUT_BYTES = 40_000;

export class ShellExecTool implements Tool {
  readonly name = 'shell.exec';
  readonly description = 'Execute a safe allowlisted command in the current project workspace.';
  readonly version = '0.1.0';
  readonly parameters = {
    command: {
      type: 'string' as const,
      required: true,
      description: 'Command line to execute. It must match the shell.exec allowlist.',
    },
    cwd: {
      type: 'string' as const,
      required: false,
      description: 'Working directory. Defaults to the process cwd and must stay inside the workspace.',
    },
    timeoutMs: {
      type: 'number' as const,
      required: false,
      description: 'Timeout in milliseconds. Max 60000.',
    },
    maxOutputBytes: {
      type: 'number' as const,
      required: false,
      description: 'Maximum stdout/stderr bytes returned.',
    },
  };

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.command !== 'string' || params.command.trim().length === 0) {
      errors.push('command must be a non-empty string');
    }
    if (params.cwd !== undefined && typeof params.cwd !== 'string') {
      errors.push('cwd must be a string when provided');
    }
    if (params.timeoutMs !== undefined && (typeof params.timeoutMs !== 'number' || !Number.isFinite(params.timeoutMs) || params.timeoutMs <= 0)) {
      errors.push('timeoutMs must be a positive number when provided');
    }
    if (params.maxOutputBytes !== undefined && (typeof params.maxOutputBytes !== 'number' || !Number.isFinite(params.maxOutputBytes) || params.maxOutputBytes <= 0)) {
      errors.push('maxOutputBytes must be a positive number when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const command = String(params.command).trim();
    let parsed: string[];
    try {
      parsed = this.parseCommand(command);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
    if (parsed.length === 0) {
      return { success: false, error: 'Command is empty' };
    }

    const [executable, ...args] = parsed;
    const policy = this.findPolicy(executable, args);
    if (!policy) {
      return {
        success: false,
        error: `Command is not allowlisted: ${this.redactCommand(command)}`,
        metadata: {
          allowed: this.formatAllowlist(),
        },
      };
    }

    const workspaceRoot = path.resolve(process.cwd());
    const cwd = this.resolveCwd(typeof params.cwd === 'string' ? params.cwd : undefined, workspaceRoot);
    if (!cwd) {
      return { success: false, error: 'cwd must stay inside the current workspace' };
    }

    if (policy.readOnlyPaths && !this.argsStayInsideWorkspace(args, workspaceRoot, cwd)) {
      return { success: false, error: 'path arguments must be relative paths inside the current workspace' };
    }

    const timeoutMs = Math.min(Number(params.timeoutMs ?? DEFAULT_TIMEOUT_MS), MAX_TIMEOUT_MS);
    const maxOutputBytes = Number(params.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES);

    try {
      const output = await execFileAsync(executable, args, {
        cwd,
        timeout: timeoutMs,
        maxBuffer: Math.max(maxOutputBytes * 2, 1024),
        windowsHide: true,
      });
      return {
        success: true,
        result: this.buildResult(command, executable, args, cwd, output.stdout, output.stderr, 0, false, maxOutputBytes),
        metadata: {
          allowlistPolicy: this.policyLabel(policy),
          timeoutMs,
        },
      };
    } catch (error) {
      const err = error as NodeJS.ErrnoException & {
        stdout?: string | Buffer;
        stderr?: string | Buffer;
        code?: number | string;
        killed?: boolean;
        signal?: string;
      };
      const exitCode = typeof err.code === 'number' ? err.code : 1;
      const timedOut = err.killed === true && err.signal === 'SIGTERM';
      const result = this.buildResult(
        command,
        executable,
        args,
        cwd,
        this.outputToString(err.stdout),
        this.outputToString(err.stderr),
        exitCode,
        timedOut,
        maxOutputBytes
      );
      return {
        success: false,
        result,
        error: timedOut ? `Command timed out after ${timeoutMs}ms` : err.message,
        metadata: {
          allowlistPolicy: this.policyLabel(policy),
          timeoutMs,
        },
      };
    }
  }

  private buildResult(
    command: string,
    executable: string,
    args: string[],
    cwd: string,
    stdout: string | Buffer | undefined,
    stderr: string | Buffer | undefined,
    exitCode: number,
    timedOut: boolean,
    maxOutputBytes: number
  ): ShellExecResult {
    return {
      command,
      executable,
      args,
      cwd,
      stdout: this.truncate(this.outputToString(stdout), maxOutputBytes),
      stderr: this.truncate(this.outputToString(stderr), maxOutputBytes),
      exitCode,
      timedOut,
    };
  }

  private parseCommand(command: string): string[] {
    const tokens: string[] = [];
    let current = '';
    let quote: '"' | "'" | null = null;

    for (let index = 0; index < command.length; index += 1) {
      const char = command[index];
      if (quote) {
        if (char === quote) {
          quote = null;
        } else {
          current += char;
        }
        continue;
      }
      if (char === '"' || char === "'") {
        quote = char;
        continue;
      }
      if (/\s/.test(char)) {
        if (current.length > 0) {
          tokens.push(current);
          current = '';
        }
        continue;
      }
      current += char;
    }
    if (quote) {
      throw new Error('Unclosed quote in command');
    }
    if (current.length > 0) {
      tokens.push(current);
    }
    return tokens;
  }

  private findPolicy(executable: string, args: string[]): CommandPolicy | undefined {
    return COMMAND_POLICIES.find(policy => {
      if (policy.executable !== executable) return false;
      if (policy.allowAnyArgs) return true;
      return policy.allowedArgs?.some(prefix => this.matchesPrefix(args, prefix)) ?? false;
    });
  }

  private matchesPrefix(args: string[], prefix: string[]): boolean {
    if (args.length < prefix.length) return false;
    return prefix.every((item, index) => args[index] === item);
  }

  private resolveCwd(input: string | undefined, workspaceRoot: string): string | null {
    const cwd = input ? path.resolve(workspaceRoot, input) : workspaceRoot;
    return this.isInside(workspaceRoot, cwd) ? cwd : null;
  }

  private argsStayInsideWorkspace(args: string[], workspaceRoot: string, cwd: string): boolean {
    return args.every(arg => {
      if (arg.startsWith('-')) return true;
      if (arg.includes('\0')) return false;
      if (/[;&|`$<>]/.test(arg)) return false;
      if (path.isAbsolute(arg)) return this.isInside(workspaceRoot, path.resolve(arg));
      if (arg === '.' || arg === './') return true;
      if (arg.startsWith('..')) return false;
      const resolved = path.resolve(cwd, arg);
      return this.isInside(workspaceRoot, resolved);
    });
  }

  private isInside(root: string, candidate: string): boolean {
    const relative = path.relative(root, candidate);
    return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
  }

  private outputToString(value: string | Buffer | undefined): string {
    if (value === undefined) return '';
    return Buffer.isBuffer(value) ? value.toString('utf8') : String(value);
  }

  private truncate(value: string, maxBytes: number): string {
    const buffer = Buffer.from(value);
    if (buffer.byteLength <= maxBytes) return value;
    return buffer.subarray(0, maxBytes).toString('utf8') + '\n[truncated]';
  }

  private redactCommand(command: string): string {
    return command.length > 160 ? `${command.slice(0, 160)}...` : command;
  }

  private policyLabel(policy: CommandPolicy): string {
    if (policy.allowAnyArgs) return `${policy.executable} *`;
    return `${policy.executable} ${policy.allowedArgs?.map(args => args.join(' ')).join(' | ')}`;
  }

  private formatAllowlist(): string[] {
    return COMMAND_POLICIES.map(policy => this.policyLabel(policy));
  }
}
