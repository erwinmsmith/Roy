import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FsWriteTool } from '../src/core/tools/index.js';

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
