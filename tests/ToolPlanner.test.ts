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

  it('runs the test suite when a tester is assigned behavioral verification', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Verify the claims against tests and failure cases.',
      workspacePath: '/workspace',
      archetype: 'tester',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'shell.exec', params: { command: 'npm test' }, groundingRequired: true }),
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
      expect.objectContaining({ toolName: 'fs.list', params: { path: '.', maxDepth: 2 } }),
    ]);
  });
});
