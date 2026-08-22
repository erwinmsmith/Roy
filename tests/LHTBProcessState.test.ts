import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { GlobalEpistemicStateRecorder, LHTBAutonomousController,
  RoyLHTBSession } from '../src/core/structural/index.js';
import { LLMJSONParseError } from '../src/core/llm/providers/openai.js';

function baseInput(sequence: number, previousFingerprint?: string) {
  return {
    trajectoryId: 'trajectory', taskId: 'task', sequence,
    requirements: [], claims: [], assumptions: [], evidence: [], externalObservations: [],
    semanticRelations: [], blindSpots: [], dependencies: [], nodes: [], dagEdges: [],
    activeSubtree: [], runtimeEvents: [],
    usage: { inputTokens: 0, outputTokens: 0, wallTimeMs: 0 },
    environmentRevision: 'lhtb-pinned', previousFingerprint,
  };
}

describe('LHTB process state', () => {
  it('round-trips an append-only fingerprint chain', () => {
    const recorder = new GlobalEpistemicStateRecorder();
    const first = recorder.append(baseInput(0));
    recorder.append(baseInput(1, first.fingerprint));
    const restored = new GlobalEpistemicStateRecorder();
    restored.restore(recorder.snapshot());
    expect(restored.latest()?.sequence).toBe(1);
    expect(() => recorder.append(baseInput(3))).toThrow(/Expected process state/);
  });

  it('aligns terminal side effects with M_t and restores the session', () => {
    const session = new RoyLHTBSession('trajectory', 'task', 'solve it', 'commit');
    session.requestTerminal({ id: 'command-1', command: 'pwd', cwd: '/workspace',
      timeoutMs: 1000, nodeId: 'root' });
    session.acceptTerminalResult({ requestId: 'command-1', exitCode: 0,
      stdout: '/workspace\n', stderr: '', durationMs: 5, fileChanges: ['answer.txt'] });
    const snapshot = session.snapshot();
    expect(snapshot.processStates).toHaveLength(3);
    expect(snapshot.processStates.at(-1)?.runtimeEvents.at(-1)?.exitCode).toBe(0);
    const restored = RoyLHTBSession.restore(snapshot);
    expect(restored.snapshot().runtime.fingerprint).toBe(snapshot.runtime.fingerprint);
  });

  it('makes direct a true single root agent in the same runtime', () => {
    const direct = new RoyLHTBSession('direct', 'task', 'solve it', 'commit',
      'single_agent_direct');
    expect(direct.snapshot().runtime.requirements[0]?.id).toBe('root-task-requirement');
    expect(() => direct.applyOrganizationAction({ kind: 'CONNECT', actorNodeId: 'root',
      connection: { from: 'root', to: 'root', required: false } })).toThrow(/single_agent_direct/);
  });

  it('reopens the same session after an official verifier rejection', () => {
    const session = new RoyLHTBSession('resume', 'task', 'solve it', 'commit');
    session.applyOrganizationAction({ kind: 'STOP', actorNodeId: 'root', finalOutput: 'draft' });
    expect(session.snapshot().runtime.stopped).toBe(true);
    session.resumeAfterVerifierRejection('Verifier rejected phase one');
    expect(session.snapshot().runtime.stopped).toBe(false);
    expect(session.snapshot().processStates.at(-1)?.runtimeEvents.at(-1)?.kind).toBe('verifier');
    expect(session.snapshot().processStates.at(-1)?.runtimeEvents.at(-1)?.attributes?.feedback)
      .toBe('Verifier rejected phase one');
  });

  it('integrates mock DeepSeek semantics before the next organization decision', async () => {
    const session = new RoyLHTBSession('mock', 'task', 'solve it', 'commit',
      'roy_runtime_heuristic');
    session.requestTerminal({ id: 'one', command: 'pwd', timeoutMs: 1000, nodeId: 'root' });
    session.acceptTerminalResult({ requestId: 'one', exitCode: 0, stdout: '/workspace',
      stderr: '', durationMs: 1 });
    let semanticCalls = 0;
    const semantic = {
      async processEvent(event: { id: string }) {
        semanticCalls += 1;
        return { event_id: event.id, requirements: [], claims: [{ id: `claim-${event.id}`,
          statement: 'The command completed', status: 'supported', originNodeId: 'root' }],
        assumptions: [], evidence: [], external_observations: [], blind_spots: [],
        relations: [] };
      },
      close() {},
    };
    const provider = {
      isConfigured: () => true,
      async completeJSONWithUsage() {
        return { value: { preferred_candidate_id: 'stop', candidates: [{
          id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'finish',
          schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root',
            finalOutput: 'done' },
        }] }, completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
      },
    };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    const result = await controller.advance(session, 1);
    expect(result.status).toBe('completed');
    expect(semanticCalls).toBe(1);
    expect(session.snapshot().processStates.at(-1)?.claims.length).toBe(1);
    expect(session.snapshot().processedSemanticEventIds).toHaveLength(2);
    controller.close();
  });

  it('accepts a terminal acquisition candidate whose command is stored outside action', async () => {
    const session = new RoyLHTBSession('acquire', 'task', 'inspect the workspace', 'commit',
      'roy_runtime_heuristic');
    const semantic = {
      async processEvent(event: { id: string }) {
        return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
          external_observations: [], blind_spots: [], relations: [] };
      },
      close() {},
    };
    const provider = {
      isConfigured: () => true,
      async completeJSONWithUsage() {
        return { value: { preferred_candidate_id: 'inspect', candidates: [{
          id: 'inspect', kind: 'ACQUIRE', actorNodeId: 'root',
          description: 'Inspect files', schedulerComplexity: 1,
          command: 'find . -maxdepth 2 -type f', timeoutMs: 120_000,
          action: { kind: 'ACQUIRE', actorNodeId: 'root' },
        }] }, completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
      },
    };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    const result = await controller.advance(session, 1);
    expect(result.status).toBe('terminal_request');
    if (result.status === 'terminal_request') {
      expect(result.request.command).toBe('find . -maxdepth 2 -type f');
      expect(result.request.nodeId).toBe('root');
    }
    controller.close();
  });

  it('runs EXECUTE candidates through the same audited terminal boundary', async () => {
    const session = new RoyLHTBSession('execute', 'task', 'change the workspace', 'commit',
      'roy_runtime_heuristic');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'implement', candidates: [{
        id: 'implement', kind: 'EXECUTE', actorNodeId: 'root', description: 'Implement',
        schedulerComplexity: 2, command: 'python implement.py',
        action: { kind: 'EXECUTE', actorNodeId: 'root' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    const result = await controller.advance(session, 1);
    expect(result.status).toBe('terminal_request');
    if (result.status === 'terminal_request') {
      expect(result.request.organizationActionKind).toBe('EXECUTE');
      expect(result.request.command).toBe('python implement.py');
    }
    controller.close();
  });

  it('accepts a strict open child specification for autonomous derivation', async () => {
    const session = new RoyLHTBSession('derive', 'task', 'implement and verify', 'commit',
      'roy_runtime_heuristic');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const childSpecification = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'implement and verify', triggeringGapId: 'root-task-requirement',
      localObjective: 'Implement the bounded code change and report verification evidence.',
      refinement: { parentScope: 'implement and verify', childScope: 'implement code change',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'Repository implementation details',
        executableEndCondition: 'Tests pass and evidence is recorded',
        duplicatedByExistingNode: false },
      requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'Modify and test files' },
      expectedOutput: { requiredInformation: 'Implementation and test evidence',
        outputType: 'epistemic_report' },
      terminationCondition: 'Return after tests pass or a concrete blocker is established',
    };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'derive-worker', candidates: [{
        id: 'derive-worker', kind: 'DERIVE', actorNodeId: 'root',
        description: 'Delegate the bounded implementation', schedulerComplexity: 2,
        action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    const result = await controller.advance(session, 1);
    expect(result.status).toBe('continue');
    expect(session.snapshot().runtime.nodes.map(value => value.id)).toEqual(['root', 'worker']);
    expect(session.snapshot().runtime.derivationEdges).toEqual([{ parentId: 'root', childId: 'worker' }]);
    controller.close();
  });

  it('audits and repairs a malformed proposer response without losing token usage', async () => {
    const auditRoot = await mkdtemp(path.join(tmpdir(), 'roy-lhtb-proposal-'));
    const session = new RoyLHTBSession('repair', 'task', 'finish', 'commit',
      'roy_runtime_heuristic');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const requests: Array<Array<{ role: string; content: string }>> = [];
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      role: string; content: string;
    }>) {
      requests.push(messages);
      calls += 1;
      if (calls === 1) throw new LLMJSONParseError('malformed', {
        content: '{bad', model: 'mock', usage: { promptTokens: 2, completionTokens: 3,
          totalTokens: 5 },
      });
      return { value: { preferred_candidate_id: 'stop', candidates: [{
        id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'finish',
        schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root', finalOutput: 'done' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 5, completionTokens: 7, totalTokens: 12,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('completed');
      expect(calls).toBe(2);
      expect(requests[1]).toHaveLength(2);
      expect(requests[1]?.some(message => message.content.includes('{bad'))).toBe(false);
      expect(session.snapshot().processStates.at(-1)?.usage).toMatchObject({
        inputTokens: 7, outputTokens: 10,
      });
      expect((await readFile(path.join(auditRoot, 'proposal-failures.jsonl'), 'utf8')).trim())
        .not.toBe('');
    } finally {
      controller.close();
      await rm(auditRoot, { recursive: true, force: true });
    }
  });

  it('projects cumulative terminal text for the proposer without mutating M_t', async () => {
    const session = new RoyLHTBSession('projection', 'task', 'finish', 'commit',
      'roy_runtime_heuristic');
    const longOutput = `head-${'x'.repeat(40_000)}-tail`;
    session.requestTerminal({ id: 'large', command: 'python inspect.py', timeoutMs: 1000,
      nodeId: 'root', organizationActionKind: 'ACQUIRE' });
    session.acceptTerminalResult({ requestId: 'large', exitCode: 0, stdout: longOutput,
      stderr: '', durationMs: 1, fileChanges: [] });
    session.applyOrganizationAction({ kind: 'ACQUIRE', actorNodeId: 'root', observation: {
      id: 'large-observation', sourceType: 'environment', queryOrAction: 'python inspect.py',
      observation: 'large output retained in the process event', provenance: 'test', supports: [],
    } });
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let proposerInput = '';
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      role: string; content: string;
    }>) {
      proposerInput = messages.at(-1)?.content ?? '';
      return { value: { preferred_candidate_id: 'stop', candidates: [{
        id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'finish',
        schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root', finalOutput: 'done' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    try {
      await controller.advance(session, 1);
      expect(proposerInput).toContain('proposer projection omitted');
      expect(proposerInput.length).toBeLessThan(30_000);
      const rawEvent = session.snapshot().processStates
        .flatMap(state => state.runtimeEvents)
        .find(event => event.id === 'result-large');
      expect(rawEvent?.output).toBe(longOutput);
    } finally {
      controller.close();
    }
  });

  it('rejects an unchanged failed command and requests a legal replacement', async () => {
    const auditRoot = await mkdtemp(path.join(tmpdir(), 'roy-lhtb-command-repair-'));
    const session = new RoyLHTBSession('command-repair', 'task', 'finish', 'commit',
      'roy_runtime_heuristic');
    session.requestTerminal({ id: 'failed', command: 'python missing.py', timeoutMs: 1000,
      nodeId: 'root', organizationActionKind: 'EXECUTE' });
    session.acceptTerminalResult({ requestId: 'failed', exitCode: 2, stdout: '',
      stderr: 'missing', durationMs: 1, fileChanges: [] });
    session.applyOrganizationAction({ kind: 'EXECUTE', actorNodeId: 'root' });
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      calls += 1;
      const candidate = calls === 1 ? {
        id: 'repeat', kind: 'EXECUTE', actorNodeId: 'root', description: 'Repeat failure',
        schedulerComplexity: 1, command: 'python missing.py',
        action: { kind: 'EXECUTE', actorNodeId: 'root' },
      } : {
        id: 'repair', kind: 'EXECUTE', actorNodeId: 'root', description: 'Inspect and repair',
        schedulerComplexity: 1, command: 'ls -la && python repair.py',
        action: { kind: 'EXECUTE', actorNodeId: 'root' },
      };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: JSON.stringify({ candidates: [candidate] }), model: 'mock',
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('terminal_request');
      if (result.status === 'terminal_request') {
        expect(result.request.command).toBe('ls -la && python repair.py');
      }
      expect(calls).toBe(2);
      const validations = (await readFile(
        path.join(auditRoot, 'candidate-validation.jsonl'), 'utf8'
      )).trim().split('\n').map(line => JSON.parse(line));
      expect(validations[0].dispositions[0].reasons)
        .toContain('repeats_unchanged_failed_command');
    } finally {
      controller.close();
      await rm(auditRoot, { recursive: true, force: true });
    }
  });
});
