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
  startLine: number;
  endLine: number;
  totalLines: number;
}

export interface FsWriteResult {
  path: string;
  bytes: number;
  mode: 'overwrite' | 'append';
}

export interface FsSearchMatch {
  path: string;
  line: number;
  column: number;
  preview: string;
}

export interface FsSearchResult {
  root: string;
  query: string;
  filesSearched: number;
  matches: FsSearchMatch[];
  truncated: boolean;
}

export interface FsReplaceResult {
  path: string;
  replacements: number;
  bytes: number;
  startLine?: number;
  endLine?: number;
}

const DEFAULT_MAX_DEPTH = 2;
const DEFAULT_MAX_BYTES = 80_000;
const DEFAULT_SEARCH_RESULTS = 100;
const DEFAULT_SEARCH_FILE_BYTES = 200_000;
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
  readonly description = 'Read a text file inside the current workspace. Use inclusive 1-based startLine/endLine ranges around reported errors or symbols instead of rereading a large file from the beginning.';
  readonly version = '0.1.0';
  readonly parameters = {
    path: { type: 'string' as const, required: true, description: 'Relative file path inside the workspace.' },
    startLine: { type: 'number' as const, required: false, description: 'Optional inclusive 1-based first line to return.' },
    endLine: { type: 'number' as const, required: false, description: 'Optional inclusive 1-based last line to return.' },
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
    for (const field of ['startLine', 'endLine'] as const) {
      if (params[field] !== undefined
        && (typeof params[field] !== 'number'
          || !Number.isInteger(params[field])
          || Number(params[field]) <= 0)) {
        errors.push(`${field} must be a positive integer when provided`);
      }
    }
    if (typeof params.startLine === 'number'
      && typeof params.endLine === 'number'
      && params.endLine < params.startLine) {
      errors.push('endLine must be greater than or equal to startLine');
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
      const lines = content.length === 0 ? [] : content.split(/\r?\n/);
      const totalLines = lines.length;
      const hasLineRange = params.startLine !== undefined || params.endLine !== undefined;
      const requestedStartLine = Number(params.startLine ?? 1);
      const requestedEndLine = Number(params.endLine ?? Math.max(totalLines, 1));
      if (totalLines > 0 && requestedStartLine > totalLines) {
        return {
          success: false,
          error: `startLine ${requestedStartLine} exceeds the file's ${totalLines} lines`,
        };
      }
      const startLine = totalLines === 0 ? 1 : requestedStartLine;
      const endLine = totalLines === 0
        ? 0
        : Math.min(requestedEndLine, totalLines);
      const selectedContent = !hasLineRange
        ? content
        : totalLines === 0
          ? ''
          : lines.slice(startLine - 1, endLine).join('\n');
      const selectedBytes = Buffer.byteLength(selectedContent, 'utf8');
      const truncated = selectedBytes > maxBytes;
      const returnedContent = truncated
        ? Buffer.from(selectedContent).subarray(0, maxBytes).toString('utf8')
        : selectedContent;
      return {
        success: true,
        result: {
          path: path.relative(workspaceRoot, target),
          content: returnedContent,
          bytes: Buffer.byteLength(returnedContent, 'utf8'),
          truncated,
          startLine,
          endLine,
          totalLines,
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

export class FsSearchTool implements Tool {
  readonly name = 'fs.search';
  readonly description = 'Search text files inside the configured workspace without relying on external grep or ripgrep binaries.';
  readonly version = '0.1.0';
  readonly parameters = {
    query: { type: 'string' as const, required: true, description: 'Literal text or regular expression to search for.' },
    path: { type: 'string' as const, required: false, description: 'Relative directory or file path inside the workspace.' },
    filePattern: { type: 'string' as const, required: false, description: 'Optional filename glob such as *.py or *.toml.' },
    regex: { type: 'boolean' as const, required: false, description: 'Interpret query as a regular expression. Defaults to false.' },
    caseSensitive: { type: 'boolean' as const, required: false, description: 'Use case-sensitive matching. Defaults to false.' },
    maxResults: { type: 'number' as const, required: false, description: 'Maximum returned matches. Defaults to 100 and is capped at 500.' },
    maxFileBytes: { type: 'number' as const, required: false, description: 'Maximum bytes read from one file. Defaults to 200000 and is capped at 1000000.' },
  };

  constructor(private readonly workspaceRoot = process.cwd()) {}

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.query !== 'string' || params.query.length === 0) {
      errors.push('query must be a non-empty string');
    }
    if (params.path !== undefined && typeof params.path !== 'string') {
      errors.push('path must be a string when provided');
    }
    if (params.filePattern !== undefined && typeof params.filePattern !== 'string') {
      errors.push('filePattern must be a string when provided');
    }
    if (params.regex !== undefined && typeof params.regex !== 'boolean') {
      errors.push('regex must be a boolean when provided');
    }
    if (params.caseSensitive !== undefined && typeof params.caseSensitive !== 'boolean') {
      errors.push('caseSensitive must be a boolean when provided');
    }
    for (const field of ['maxResults', 'maxFileBytes'] as const) {
      if (params[field] !== undefined
        && (typeof params[field] !== 'number' || !Number.isFinite(params[field]) || params[field] <= 0)) {
        errors.push(`${field} must be a positive number when provided`);
      }
    }
    if (params.regex === true && typeof params.query === 'string') {
      try {
        void new RegExp(params.query);
      } catch (error) {
        errors.push(`query is not a valid regular expression: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const validation = this.validate(params);
    if (!validation.valid) {
      return { success: false, error: `Validation failed: ${validation.errors?.join(', ')}` };
    }
    const workspaceRoot = path.resolve(this.workspaceRoot);
    const target = resolveWorkspacePath(typeof params.path === 'string' ? params.path : '.', workspaceRoot);
    if (!target) {
      return { success: false, error: 'path must stay inside the current workspace' };
    }
    const maxResults = Math.min(Math.floor(Number(params.maxResults ?? DEFAULT_SEARCH_RESULTS)), 500);
    const maxFileBytes = Math.min(Math.floor(Number(params.maxFileBytes ?? DEFAULT_SEARCH_FILE_BYTES)), 1_000_000);
    const caseSensitive = params.caseSensitive === true;
    const flags = caseSensitive ? 'g' : 'gi';
    const expression = params.regex === true
      ? new RegExp(String(params.query), flags)
      : new RegExp(escapeRegExp(String(params.query)), flags);
    const filenameExpression = typeof params.filePattern === 'string' && params.filePattern.trim()
      ? globToRegExp(params.filePattern.trim())
      : undefined;
    try {
      const files = await collectSearchFiles(target, filenameExpression);
      const matches: FsSearchMatch[] = [];
      let filesSearched = 0;
      let truncated = false;
      for (const file of files) {
        if (matches.length >= maxResults) {
          truncated = true;
          break;
        }
        const fileStat = await stat(file);
        if (!fileStat.isFile() || fileStat.size > maxFileBytes) continue;
        const content = await readFile(file, 'utf8');
        if (content.includes('\0')) continue;
        filesSearched += 1;
        const lines = content.split(/\r?\n/);
        for (let index = 0; index < lines.length; index += 1) {
          expression.lastIndex = 0;
          const match = expression.exec(lines[index]!);
          if (!match) continue;
          matches.push({
            path: path.relative(workspaceRoot, file),
            line: index + 1,
            column: match.index + 1,
            preview: lines[index]!.slice(0, 500),
          });
          if (matches.length >= maxResults) {
            truncated = index < lines.length - 1 || file !== files.at(-1);
            break;
          }
        }
      }
      return {
        success: true,
        result: {
          root: path.relative(workspaceRoot, target) || '.',
          query: String(params.query),
          filesSearched,
          matches,
          truncated,
        } satisfies FsSearchResult,
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

export class FsReplaceTool implements Tool {
  readonly name = 'fs.replace';
  readonly description = 'Replace exact text or an inclusive line range in one workspace file, avoiding fragile shell quoting or full-file rewrites.';
  readonly version = '0.1.0';
  readonly parameters = {
    path: { type: 'string' as const, required: true, description: 'Relative file path inside the workspace.' },
    oldText: { type: 'string' as const, required: false, description: 'Exact existing text to replace. Omit when using startLine/endLine.' },
    startLine: { type: 'number' as const, required: false, description: 'One-based first line to replace when exact oldText is fragile.' },
    endLine: { type: 'number' as const, required: false, description: 'One-based inclusive final line to replace.' },
    newText: { type: 'string' as const, required: true, description: 'Replacement text.' },
    replaceAll: { type: 'boolean' as const, required: false, description: 'Replace every occurrence. Defaults to false.' },
    expectedReplacements: { type: 'number' as const, required: false, description: 'Optional exact occurrence count required before writing.' },
  };

  constructor(private readonly workspaceRoot = process.cwd()) {}

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.path !== 'string' || params.path.trim().length === 0) {
      errors.push('path must be a non-empty string');
    }
    if (typeof params.newText !== 'string') {
      errors.push('newText must be a string');
    }
    const hasOldText = typeof params.oldText === 'string' && params.oldText.length > 0;
    const hasStartLine = params.startLine !== undefined;
    const hasEndLine = params.endLine !== undefined;
    if (!hasOldText && !hasStartLine && !hasEndLine) {
      errors.push('provide either non-empty oldText or startLine/endLine');
    }
    if (hasOldText && (hasStartLine || hasEndLine)) {
      errors.push('oldText and startLine/endLine are mutually exclusive');
    }
    if (hasStartLine !== hasEndLine) {
      errors.push('startLine and endLine must be provided together');
    }
    if (hasStartLine && (
      typeof params.startLine !== 'number'
      || !Number.isInteger(params.startLine)
      || params.startLine <= 0
    )) {
      errors.push('startLine must be a positive integer');
    }
    if (hasEndLine && (
      typeof params.endLine !== 'number'
      || !Number.isInteger(params.endLine)
      || params.endLine <= 0
    )) {
      errors.push('endLine must be a positive integer');
    }
    if (hasStartLine && hasEndLine && Number(params.endLine) < Number(params.startLine)) {
      errors.push('endLine must be greater than or equal to startLine');
    }
    if (params.replaceAll !== undefined && typeof params.replaceAll !== 'boolean') {
      errors.push('replaceAll must be a boolean when provided');
    }
    if (params.expectedReplacements !== undefined
      && (typeof params.expectedReplacements !== 'number'
        || !Number.isInteger(params.expectedReplacements)
        || params.expectedReplacements <= 0)) {
      errors.push('expectedReplacements must be a positive integer when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const validation = this.validate(params);
    if (!validation.valid) {
      return { success: false, error: `Validation failed: ${validation.errors?.join(', ')}` };
    }
    const workspaceRoot = path.resolve(this.workspaceRoot);
    const target = resolveWorkspacePath(String(params.path), workspaceRoot);
    if (!target || target === workspaceRoot) {
      return { success: false, error: 'path must point to a file inside the configured workspace' };
    }
    try {
      const fileStat = await stat(target);
      if (!fileStat.isFile()) return { success: false, error: 'path must point to a file' };
      const content = await readFile(target, 'utf8');
      if (params.startLine !== undefined && params.endLine !== undefined) {
        const startLine = Number(params.startLine);
        const endLine = Number(params.endLine);
        const lineStarts = [0];
        for (let index = 0; index < content.length; index += 1) {
          if (content[index] === '\n') lineStarts.push(index + 1);
        }
        const totalLines = content.length === 0 ? 0 : lineStarts.length;
        if (startLine > totalLines || endLine > totalLines) {
          return {
            success: false,
            error: `line range ${startLine}-${endLine} exceeds the file's ${totalLines} lines`,
          };
        }
        const startOffset = lineStarts[startLine - 1]!;
        const endOffset = endLine < lineStarts.length
          ? lineStarts[endLine]!
          : content.length;
        const newline = content.includes('\r\n') ? '\r\n' : '\n';
        let replacement = String(params.newText).replace(/\r?\n/g, newline);
        const suffix = content.slice(endOffset);
        if (suffix && replacement && !replacement.endsWith(newline)) {
          replacement += newline;
        }
        const updated = content.slice(0, startOffset) + replacement + suffix;
        await writeFile(target, updated, 'utf8');
        return {
          success: true,
          result: {
            path: path.relative(workspaceRoot, target),
            replacements: 1,
            bytes: Buffer.byteLength(updated, 'utf8'),
            startLine,
            endLine,
          } satisfies FsReplaceResult,
        };
      }
      const oldText = String(params.oldText);
      const occurrences = countOccurrences(content, oldText);
      const expected = typeof params.expectedReplacements === 'number'
        ? params.expectedReplacements
        : undefined;
      if (occurrences === 0) {
        return { success: false, error: 'oldText was not found; inspect the current file before retrying' };
      }
      if (expected !== undefined && occurrences !== expected) {
        return {
          success: false,
          error: `expected ${expected} occurrence(s) but found ${occurrences}; no write was performed`,
        };
      }
      if (params.replaceAll !== true && occurrences > 1 && expected === undefined) {
        return {
          success: false,
          error: `oldText occurs ${occurrences} times; set expectedReplacements or replaceAll to make the edit unambiguous`,
        };
      }
      const replacements = params.replaceAll === true ? occurrences : 1;
      const updated = params.replaceAll === true
        ? content.replaceAll(oldText, String(params.newText))
        : content.replace(oldText, String(params.newText));
      await writeFile(target, updated, 'utf8');
      return {
        success: true,
        result: {
          path: path.relative(workspaceRoot, target),
          replacements,
          bytes: Buffer.byteLength(updated, 'utf8'),
        } satisfies FsReplaceResult,
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

async function collectSearchFiles(target: string, filenameExpression?: RegExp): Promise<string[]> {
  const targetStat = await stat(target);
  if (targetStat.isFile()) {
    return !filenameExpression || filenameExpression.test(path.basename(target)) ? [target] : [];
  }
  const files: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory() && IGNORED_DIRS.has(entry.name)) continue;
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(fullPath);
      } else if (entry.isFile() && (!filenameExpression || filenameExpression.test(entry.name))) {
        files.push(fullPath);
      }
    }
  };
  await visit(target);
  return files.sort();
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function globToRegExp(glob: string): RegExp {
  const pattern = glob
    .split('*')
    .map(part => part.split('?').map(escapeRegExp).join('.'))
    .join('.*');
  return new RegExp(`^${pattern}$`);
}

function countOccurrences(content: string, needle: string): number {
  let count = 0;
  let cursor = 0;
  while (cursor <= content.length - needle.length) {
    const index = content.indexOf(needle, cursor);
    if (index < 0) break;
    count += 1;
    cursor = index + needle.length;
  }
  return count;
}
