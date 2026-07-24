import { describe, expect, it } from 'vitest';
import { AgentToolPlanner } from '../src/core/tools/planner.js';

describe('AgentToolPlanner', () => {
  it('reads package.json when a package export inspection needs manifest evidence', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect this package exports and identify one concrete architecture risk.',
      workspacePath: '/workspace',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans.map(plan => plan.toolName)).toEqual(['fs.list', 'fs.read']);
    expect(plans[1].params).toEqual({ path: 'package.json' });
    expect(plans.every(plan => plan.groundingRequired)).toBe(true);
  });

  it('does not guess an npm verification command without observed project metadata', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Verify the claims against tests and failure cases.',
      workspacePath: '/workspace',
      archetype: 'tester',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    });

    expect(plans).toEqual([]);
  });

  it('honors an explicitly requested npm verification command', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Run npm test and report the real exit status.',
      workspacePath: '/workspace',
      archetype: 'tester',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'shell.exec', params: { command: 'npm test' }, groundingRequired: true }),
    ]);
  });

  it('executes a task-declared required CLI command before optional file reads', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Implement the workspace pipeline.',
        '## Required Command',
        '```bash',
        'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        '```',
        'Inspect configs/public_audit.yml when repairing failures.',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'coder',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list' }),
      expect.objectContaining({
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'configs/public_audit.yml' },
      }),
    ]);
  });

  it('turns a verifier traceback into a bounded source read', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: { command: 'pytest -q' },
        success: false,
        result: {
          cwd: '/app',
          stdout: '',
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/src/dq_audit/audit.py", line 612',
            '    batch = batch_def.get_batch()',
            'IndentationError: unexpected indent',
          ].join('\n'),
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/dq_audit/audit.py',
          startLine: 587,
          endLine: 637,
        },
      }),
    ]);
  });

  it('uses shell error text to inspect an imported workspace module', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
        success: false,
        error: [
          'Traceback (most recent call last):',
          '  File "/app/src/dq_audit/cli.py", line 6, in <module>',
          '    from .audit import run_audit',
          "ImportError: cannot import name 'run_audit' from 'dq_audit.audit' (/app/src/dq_audit/audit.py)",
        ].join('\n'),
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'src/dq_audit/audit.py' },
      }),
    ]);
  });

  it('reads the package manifest for an architecture critic', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Identify architectural coupling risks using filesystem evidence.',
      workspacePath: '/workspace',
      archetype: 'critic',
      bindings: [{ name: 'fs.read', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'package.json' }, groundingRequired: true }),
    ]);
  });

  it('plans concrete source reads for runtime API export inspection', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect the exported runtime API surface and identify mismatches.',
      workspacePath: '/workspace',
      archetype: 'custom',
      bindings: [{ name: 'fs.read', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/index.ts' } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/core/runtime/index.ts' } }),
    ]);
  });

  it('merges an explicit manifest target with inferred runtime API source targets', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Read package.json and inspect exported runtime APIs for a consistency mismatch.',
      workspacePath: '/workspace',
      archetype: 'custom',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans.map(plan => plan.params.path)).toEqual([
      'package.json',
      'src/index.ts',
      'src/core/runtime/index.ts',
    ]);
    expect(plans.every(plan => plan.toolName === 'fs.read')).toBe(true);
  });

  it('uses an allowlisted manifest command when a cached critic only exposes shell execution', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Identify architectural coupling risks using filesystem evidence.',
      workspacePath: '/workspace',
      archetype: 'critic',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'shell.exec', params: { command: 'cat package.json' }, groundingRequired: true }),
    ]);
  });

  it('prioritizes an explicitly requested directory and multiple source files', () => {
    const planner = new AgentToolPlanner();
    const plans = planner.plan({
      task: 'Read src/core/runtime/index.ts and src/server/RuntimeSessionPool.ts. Also read src/core/delegation/index.ts and list tests/ directory.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list', params: { path: 'tests', maxDepth: 3 } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/core/runtime/index.ts' } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/server/RuntimeSessionPool.ts' } }),
    ]);
  });

  it('keeps the broad workspace listing fallback for tasks without explicit paths', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect this repository structure using filesystem evidence.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [{ name: 'fs.list', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list', params: { path: '.', maxDepth: 4 } }),
    ]);
  });

  it('searches the web for current externally verifiable evidence', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Search the web for the latest official Node.js fetch documentation and cite sources.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'web.search', groundingRequired: true }),
    ]);
    expect(plans[0].params.query).toBe('Node.js fetch documentation and cite sources.');
  });

  it('removes imperative web-source scaffolding from the search query', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Use public web sources to compare Node.js AbortSignal.timeout and MDN AbortSignal.timeout. Open at least two sources.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [{ name: 'web.search', enabled: true }],
    });

    expect(plans[0].params.query).toBe('Node.js AbortSignal.timeout and MDN AbortSignal.timeout');
  });

  it('fetches an explicitly supplied public URL instead of searching again', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Read https://nodejs.org/api/globals.html and summarize the fetch section.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/api/globals.html' },
      }),
    ]);
  });

  it('ignores incidental URLs embedded in terminal feedback for workspace tasks', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Repair the current workspace package and rerun its tests.',
        'Latest command output:',
        'WARNING: use a virtual environment: https://pip.pypa.io/warnings/venv',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'coder',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
      }),
    ]);
  });

  it('prioritizes official API documentation links over unofficial downloads', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find official Node.js documentation for the global fetch API and AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [
        {
          toolName: 'web.fetch',
          params: { url: 'https://nodejs.org/' },
          success: true,
          result: {
            links: [
              { text: 'Unofficial builds', url: 'https://unofficial-builds.nodejs.org/download/' },
              { text: 'Docs', url: 'https://nodejs.org/docs/latest/api/' },
              { text: 'Global objects', url: 'https://nodejs.org/api/globals.html' },
            ],
          },
        },
      ],
    });

    expect(plans.map(plan => plan.params.url)).toEqual([
      'https://nodejs.org/docs/latest/api/',
      'https://nodejs.org/api/globals.html',
    ]);
  });

  it('treats a task-relevant fragment as focused evidence and hands control back to the agent planner', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find official Node.js documentation for the global fetch API and AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay' },
        success: true,
        result: {
          finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay',
          title: 'Global objects - Static method: AbortSignal.timeout(delay)',
          text: 'AbortSignal.timeout(delay) returns a signal that aborts after delay milliseconds.',
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('does not follow another fragment from an already opened document', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Open and compare at least two official Node.js sources about AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/docs/latest/api/globals.html' },
        success: true,
        result: {
          finalUrl: 'https://nodejs.org/docs/latest/api/globals.html',
          title: 'Node.js globals introduction',
          text: 'Node.js global API overview.',
          links: [{
            text: 'AbortSignal.timeout',
            url: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay',
          }],
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('does not create false relevance matches across unrelated token boundaries', () => {
    const planner = new AgentToolPlanner();
    const score = planner.webEvidenceScore(
      'Find the Node.js global fetch API and AbortSignal.timeout documentation.',
      {
        toolName: 'web.fetch',
        params: { url: 'https://earth.google.com/web/' },
        success: true,
        result: {
          finalUrl: 'https://earth.google.com/web/',
          title: 'Google Earth',
          text: 'Aw snap. See system requirements for more information.',
        },
      }
    );

    expect(score).toBe(0);
  });

  it('does not fetch unrelated low-relevance search results', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find the Node.js global fetch API and AbortSignal.timeout documentation.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.search',
        params: { query: 'AbortSignal.timeout documentation' },
        success: true,
        result: {
          results: [
            { title: 'Google Earth', url: 'https://earth.google.com/web/', snippet: 'Explore the world.' },
            { title: 'Node.js Global objects', url: 'https://nodejs.org/docs/latest/api/globals.html#fetch', snippet: 'Global fetch API.' },
          ],
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' } }),
    ]);
  });

  it('requires core API entities before following otherwise keyword-rich search results', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Use public web sources to compare Node.js AbortSignal.timeout and MDN AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.search',
        params: { query: 'compare Node.js AbortSignal.timeout MDN' },
        success: true,
        result: {
          results: [
            {
              title: 'USE company information',
              url: 'https://www.use-ebisu.co.jp/',
              snippet: 'Public web information, current versions, availability, and comparison sources.',
            },
          ],
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('stops web replanning after enough distinct relevant documents were opened', () => {
    const planner = new AgentToolPlanner();
    expect(planner.hasSufficientWebEvidence(
      'Open and compare at least two relevant public sources about Node.js fetch and AbortSignal.timeout.',
      [
        {
          toolName: 'web.fetch', params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' }, success: true,
          result: { finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#fetch', title: 'Node.js fetch API', text: 'Global fetch API.' },
        },
        {
          toolName: 'web.fetch', params: { url: 'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static' }, success: true,
          result: { finalUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static', title: 'AbortSignal timeout API', text: 'AbortSignal.timeout documentation.' },
        },
      ]
    )).toBe(true);
  });

  it('treats "both URLs" as a two-document evidence requirement', () => {
    const planner = new AgentToolPlanner();
    const oneDocument = [{
      toolName: 'web.fetch',
      params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' },
      success: true,
      result: {
        finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#fetch',
        title: 'Node.js fetch API',
        text: 'AbortSignal.timeout and global fetch API documentation.',
      },
    }];

    expect(planner.hasSufficientWebEvidence(
      'Open both URLs and compare Node.js AbortSignal.timeout with MDN.',
      oneDocument
    )).toBe(false);
  });
});
