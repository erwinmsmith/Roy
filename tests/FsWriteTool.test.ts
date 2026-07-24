import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FsReplaceTool, FsSearchTool, FsWriteTool } from '../src/core/tools/index.js';

describe('fs.write tool', () => {
  it('writes and appends text inside the configured workspace', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-fs-write-'));
    const tool = new FsWriteTool(workspace);

    expect((await tool.execute({
      path: 'nested/result.txt',
      content: 'first',
    })).success).toBe(true);
    expect((await tool.execute({
      path: 'nested/result.txt',
      content: '-second',
      mode: 'append',
    })).success).toBe(true);
    expect(await readFile(path.join(workspace, 'nested/result.txt'), 'utf8')).toBe('first-second');
  });

  it('rejects paths outside the configured workspace', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-fs-write-boundary-'));
    const tool = new FsWriteTool(workspace);
    const result = await tool.execute({ path: '../escape.txt', content: 'no' });

    expect(result.success).toBe(false);
    expect(result.error).toContain('inside the configured workspace');
  });
});

describe('workspace search and replace tools', () => {
  it('finds stale declarations across manifests without relying on shell utilities', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-fs-search-'));
    await writeFile(path.join(workspace, 'requirements.txt'), 'langchain==0.1.0\n', 'utf8');
    await writeFile(path.join(workspace, 'pyproject.toml'), 'langchain = ">=0.1.0"\n', 'utf8');
    const result = await new FsSearchTool(workspace).execute({
      query: 'langchain',
      glob: '*.{txt,toml}',
    });

    expect(result.success).toBe(true);
    expect(result.result?.matches.map(match => match.path)).toEqual([
      'pyproject.toml',
      'requirements.txt',
    ]);
  });

  it('replaces an exact occurrence and refuses ambiguous edits', async () => {
    const workspace = await mkdtemp(path.join(tmpdir(), 'roy-fs-replace-'));
    const target = path.join(workspace, 'config.txt');
    await writeFile(target, 'old\nold\n', 'utf8');
    const tool = new FsReplaceTool(workspace);

    const ambiguous = await tool.execute({
      path: 'config.txt',
      oldText: 'old',
      newText: 'new',
    });
    expect(ambiguous.success).toBe(false);
    expect(await readFile(target, 'utf8')).toBe('old\nold\n');

    const replaced = await tool.execute({
      path: 'config.txt',
      oldText: 'old',
      newText: 'new',
      expectedReplacements: 2,
      replaceAll: true,
    });
    expect(replaced.success).toBe(true);
    expect(replaced.result?.replacements).toBe(2);
    expect(await readFile(target, 'utf8')).toBe('new\nnew\n');
  });
});
