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
});
