import { appendFile, mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { Tool, ToolResult } from './types.js';

export interface FsListResult {
  root: string;
  maxDepth: number;
  entries: string[];
}

export interface FsReadResult {
  path: string;
  content: string;
  bytes: number;
  truncated: boolean;
}

export interface FsWriteResult {
  path: string;
  bytes: number;
  mode: 'overwrite' | 'append';
}

const DEFAULT_MAX_DEPTH = 2;
const DEFAULT_MAX_BYTES = 80_000;
const IGNORED_DIRS = new Set(['.git', 'node_modules', 'dist', '.roy', '.cache', 'coverage']);

export class FsListTool implements Tool {
  readonly name = 'fs.list';
  readonly description = 'List files and directories inside the current workspace.';
  readonly version = '0.1.0';
  readonly parameters = {
    path: { type: 'string' as const, required: false, description: 'Relative path inside the workspace.' },
    maxDepth: { type: 'number' as const, required: false, description: 'Maximum directory depth.' },
  };

  constructor(private readonly workspaceRoot = process.cwd()) {}

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (params.path !== undefined && typeof params.path !== 'string') {
      errors.push('path must be a string when provided');
    }
    if (params.maxDepth !== undefined && (typeof params.maxDepth !== 'number' || !Number.isFinite(params.maxDepth) || params.maxDepth < 0)) {
      errors.push('maxDepth must be a non-negative number when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const workspaceRoot = path.resolve(this.workspaceRoot);
    const target = resolveWorkspacePath(typeof params.path === 'string' ? params.path : '.', workspaceRoot);
    if (!target) {
      return { success: false, error: 'path must stay inside the current workspace' };
    }

    const maxDepth = Math.min(Number(params.maxDepth ?? DEFAULT_MAX_DEPTH), 5);
    try {
      const entries = await listFiles(target, target, maxDepth);
      return {
        success: true,
        result: {
          root: path.relative(workspaceRoot, target) || '.',
          maxDepth,
          entries,
        } satisfies FsListResult,
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

export class FsReadTool implements Tool {
  readonly name = 'fs.read';
  readonly description = 'Read a text file inside the current workspace.';
  readonly version = '0.1.0';
  readonly parameters = {
    path: { type: 'string' as const, required: true, description: 'Relative file path inside the workspace.' },
    maxBytes: { type: 'number' as const, required: false, description: 'Maximum bytes to return.' },
  };

  constructor(private readonly workspaceRoot = process.cwd()) {}

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.path !== 'string' || params.path.trim().length === 0) {
      errors.push('path must be a non-empty string');
    }
    if (params.maxBytes !== undefined && (typeof params.maxBytes !== 'number' || !Number.isFinite(params.maxBytes) || params.maxBytes <= 0)) {
      errors.push('maxBytes must be a positive number when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const workspaceRoot = path.resolve(this.workspaceRoot);
    const target = resolveWorkspacePath(String(params.path), workspaceRoot);
    if (!target) {
      return { success: false, error: 'path must stay inside the current workspace' };
    }

    try {
      const fileStat = await stat(target);
      if (!fileStat.isFile()) {
        return { success: false, error: 'path must point to a file' };
      }
      const maxBytes = Math.min(Number(params.maxBytes ?? DEFAULT_MAX_BYTES), DEFAULT_MAX_BYTES);
      const content = await readFile(target, 'utf8');
      const truncated = Buffer.byteLength(content, 'utf8') > maxBytes;
      return {
        success: true,
        result: {
          path: path.relative(workspaceRoot, target),
          content: truncated ? Buffer.from(content).subarray(0, maxBytes).toString('utf8') : content,
          bytes: Math.min(Buffer.byteLength(content, 'utf8'), maxBytes),
          truncated,
        } satisfies FsReadResult,
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

export class FsWriteTool implements Tool {
  readonly name = 'fs.write';
  readonly description = 'Write or append UTF-8 text to a file inside the configured workspace.';
  readonly version = '0.1.0';
  readonly parameters = {
    path: { type: 'string' as const, required: true, description: 'Relative file path inside the workspace.' },
    content: { type: 'string' as const, required: true, description: 'UTF-8 text to write.' },
    mode: { type: 'string' as const, required: false, description: 'overwrite (default) or append.' },
    createDirectories: { type: 'boolean' as const, required: false, description: 'Create missing parent directories. Defaults to true.' },
  };

  constructor(private readonly workspaceRoot = process.cwd()) {}

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.path !== 'string' || params.path.trim().length === 0) {
      errors.push('path must be a non-empty string');
    }
    if (typeof params.content !== 'string') {
      errors.push('content must be a string');
    }
    if (params.mode !== undefined && params.mode !== 'overwrite' && params.mode !== 'append') {
      errors.push('mode must be overwrite or append');
    }
    if (params.createDirectories !== undefined && typeof params.createDirectories !== 'boolean') {
      errors.push('createDirectories must be a boolean when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const workspaceRoot = path.resolve(this.workspaceRoot);
    const target = resolveWorkspacePath(String(params.path), workspaceRoot);
    if (!target || target === workspaceRoot) {
      return { success: false, error: 'path must point to a file inside the configured workspace' };
    }

    const content = String(params.content);
    const mode = params.mode === 'append' ? 'append' : 'overwrite';
    try {
      if (params.createDirectories !== false) {
        await mkdir(path.dirname(target), { recursive: true });
      }
      if (mode === 'append') {
        await appendFile(target, content, 'utf8');
      } else {
        await writeFile(target, content, 'utf8');
      }
      return {
        success: true,
        result: {
          path: path.relative(workspaceRoot, target),
          bytes: Buffer.byteLength(content, 'utf8'),
          mode,
        } satisfies FsWriteResult,
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

function resolveWorkspacePath(input: string, workspaceRoot: string): string | null {
  if (input.includes('\0')) return null;
  const target = path.resolve(workspaceRoot, input);
  const relative = path.relative(workspaceRoot, target);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative)) ? target : null;
}

async function listFiles(root: string, current: string, maxDepth: number, depth = 0): Promise<string[]> {
  if (depth > maxDepth) return [];
  const entries = await readdir(current, { withFileTypes: true });
  const results: string[] = [];
  for (const entry of entries) {
    if (IGNORED_DIRS.has(entry.name)) continue;
    const fullPath = path.join(current, entry.name);
    const relative = path.relative(root, fullPath);
    results.push(entry.isDirectory() ? `${relative}/` : relative);
    if (entry.isDirectory()) {
      results.push(...await listFiles(root, fullPath, maxDepth, depth + 1));
    }
  }
  return results.sort();
}
