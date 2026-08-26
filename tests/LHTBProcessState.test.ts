import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { GlobalEpistemicStateRecorder, LHTBAutonomousController,
  resolveTopologySamplingProfile, RoyLHTBSession, topologySamplingCandidateLogitBias,
  topologySamplingProfile } from '../src/core/structural/index.js';
import { LLMJSONParseError } from '../src/core/llm/providers/openai.js';

function baseInput(sequence: number, previousFingerprint?: string) {
  return {
    trajectoryId: 'trajectory', taskId: 'task', sequence,
    requirements: [], claims: [], assumptions: [], evidence: [], externalObservations: [],
    semanticRelations: [], blindSpots: [], dependencies: [], nodes: [], dagEdges: [],
    activeSubtree: [], runtimeEvents: [], runtimeEventRange: {
      start: 0, endExclusive: 0, total: 0,
    },
    usage: { inputTokens: 0, outputTokens: 0, wallTimeMs: 0 },
    environmentRevision: 'lhtb-pinned', previousFingerprint,
  };
}

describe('LHTB process state', () => {
  it('covers compact through connected topology profiles without changing utility', () => {
    expect([0, 1, 2, 3].map(topologySamplingProfile).map(value => value.id))
      .toEqual(['compact', 'branching', 'recursive', 'connected']);
    expect(topologySamplingProfile(3).preferredNodeRange).toEqual([6, 8]);
    expect(topologySamplingProfile(2).preferredMinimumDepth).toBe(2);
    expect(resolveTopologySamplingProfile(0, 'recursive').preferredNodeRange).toEqual([5, 7]);
    expect(() => resolveTopologySamplingProfile(0, 'invalid')).toThrow(/Invalid/);
  });

  it('turns a recursive profile into a real root to sub to subsub derivation', () => {
    const session = new RoyLHTBSession('recursive', 'task', 'implement and verify', 'commit',
      'learned_information_realization', 'same', 2);
    session.applySemanticUpdate({ event_id: 'task-instruction', requirements: [{
      id: 'root-a', description: 'implement one bounded component', requiredInformation: 'code',
      likelyMechanism: 'conversion',
    }, {
      id: 'root-b', description: 'verify another bounded component', requiredInformation: 'test',
      likelyMechanism: 'conversion',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    const childSpecification = {
      id: 'spec-child', nodeId: 'child', parentId: 'root', depth: 1,
      parentGoal: 'implement and verify', triggeringGapId: 'root-a',
      localObjective: 'Implement one bounded component.', refinement: {
        parentScope: 'implement and verify', childScope: 'implement one component',
        triggeringRequirementId: 'root-a', narrowerThanParent: true,
        newInformationNeeded: 'repository details', executableEndCondition: 'component test passes',
        duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect and implement' },
      expectedOutput: { requiredInformation: 'tested component', outputType: 'epistemic_report' as const },
      terminationCondition: 'return tested component',
    };
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root', childSpecification });
    const shallow = session.snapshot();
    expect(topologySamplingCandidateLogitBias(shallow, {
      kind: 'EXECUTE', actorNodeId: 'child', action: { kind: 'EXECUTE', actorNodeId: 'child' },
    })).toBe(4);
    expect(topologySamplingCandidateLogitBias(shallow, {
      kind: 'DERIVE', actorNodeId: 'root', action: { kind: 'DERIVE', actorNodeId: 'root',
        childSpecification: { ...childSpecification, id: 'spec-root-b', nodeId: 'root-b-worker',
          triggeringGapId: 'root-b', refinement: { ...childSpecification.refinement,
            triggeringRequirementId: 'root-b' } } },
    })).toBe(-8);

    session.requestTerminal({ id: 'inspect-child', command: 'pytest -q', timeoutMs: 1000,
      nodeId: 'child', organizationActionKind: 'EXECUTE' });
    session.acceptTerminalResult({ requestId: 'inspect-child', exitCode: 1, stdout: '',
      stderr: 'a concrete child-local failure', durationMs: 1 });
    session.applySemanticUpdate({ event_id: 'result-inspect-child', requirements: [{
      id: 'child-gap', description: 'repair the child-local failure', requiredInformation: 'fix',
      likelyMechanism: 'conversion',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    const grandchildSpecification = {
      ...childSpecification, id: 'spec-grandchild', nodeId: 'grandchild', parentId: 'child',
      depth: 2, parentGoal: childSpecification.localObjective, triggeringGapId: 'child-gap',
      localObjective: 'Repair and verify the isolated child-local failure.',
      refinement: { ...childSpecification.refinement,
        parentScope: childSpecification.localObjective, childScope: 'repair isolated failure',
        triggeringRequirementId: 'child-gap' },
    };
    expect(topologySamplingCandidateLogitBias(session.snapshot(), {
      kind: 'DERIVE', actorNodeId: 'child', action: { kind: 'DERIVE', actorNodeId: 'child',
        childSpecification: grandchildSpecification },
    })).toBe(4);
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'child',
      childSpecification: grandchildSpecification });
    expect(session.snapshot().runtime.derivationEdges).toEqual([
      { parentId: 'root', childId: 'child' },
      { parentId: 'child', childId: 'grandchild' },
    ]);
    expect(Math.max(...session.snapshot().runtime.nodes.map(node => node.depth))).toBe(2);
    const capped = session.snapshot();
    while (capped.runtime.nodes.length < 7) {
      const index = capped.runtime.nodes.length;
      capped.runtime.nodes.push({ ...capped.runtime.nodes[0]!, id: `extra-${index}`,
        parentId: 'root', depth: 1, localObjective: `extra objective ${index}` });
    }
    expect(topologySamplingCandidateLogitBias(capped, {
      kind: 'DERIVE', actorNodeId: 'grandchild', action: { kind: 'DERIVE',
        actorNodeId: 'grandchild', childSpecification: { ...grandchildSpecification,
          id: 'too-wide', nodeId: 'too-wide', parentId: 'grandchild', depth: 3 } },
    })).toBe(-8);
  });

  it('finalizes a pending terminal request at the rollout deadline', () => {
    const session = new RoyLHTBSession('deadline', 'task', 'solve it', 'commit');
    session.requestTerminal({ id: 'long-command', command: 'run-long-task', timeoutMs: 60_000,
      nodeId: 'root', organizationActionKind: 'EXECUTE' });
    const before = session.snapshot().processStates.length;
    session.finalizeAtRolloutDeadline();
    const snapshot = session.snapshot();
    expect(snapshot.pendingTerminalRequest).toBeUndefined();
    expect(snapshot.processStates).toHaveLength(before + 1);
    expect(snapshot.processStates.at(-1)?.runtimeEvents.slice(-2).map(event => event.kind))
      .toEqual(['failure', 'verifier']);
    expect(snapshot.processStates.at(-1)?.runtimeEvents.at(-1)?.attributes?.next)
      .toBe('official_final_verifier');
  });
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

  it('stores one complete event ledger with bounded per-state projections', () => {
    const session = new RoyLHTBSession('bounded-ledger', 'task', 'solve it', 'commit');
    for (let index = 0; index < 120; index += 1) {
      session.requestTerminal({ id: `command-${index}`, command: `inspect-${index}`,
        timeoutMs: 1000, nodeId: 'root' });
      session.acceptTerminalResult({ requestId: `command-${index}`, exitCode: 0,
        stdout: `${index}:${'x'.repeat(20_000)}`, stderr: '', durationMs: 1 });
    }
    const snapshot = session.snapshot();
    expect(snapshot.runtimeEvents).toHaveLength(241);
    expect(Math.max(...snapshot.processStates.map(state => state.runtimeEvents.length))).toBe(24);
    expect(snapshot.processStates.at(-1)?.runtimeEventRange).toEqual({
      start: 217, endExclusive: 241, total: 241,
    });
    expect(JSON.stringify(snapshot).length).toBeLessThan(15_000_000);
    const searchSnapshot = RoyLHTBSession.compactForSearch(snapshot);
    expect(searchSnapshot.processStates).toHaveLength(1);
    expect(searchSnapshot.runtimeEvents).toHaveLength(24);
    expect(JSON.stringify(searchSnapshot).length).toBeLessThan(1_000_000);
    const restored = RoyLHTBSession.restore(snapshot).snapshot();
    expect(restored.runtimeEvents).toEqual(snapshot.runtimeEvents);
    expect(restored.processStates.at(-1)?.fingerprint)
      .toBe(snapshot.processStates.at(-1)?.fingerprint);
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
    expect(semanticCalls).toBe(2);
    expect(session.snapshot().processStates.at(-1)?.claims.length).toBe(2);
    expect(session.snapshot().processedSemanticEventIds).toHaveLength(3);
    controller.close();
  });

  it('replaces the aggregate root gap with explicit task requirements', () => {
    const session = new RoyLHTBSession('task-decomposition', 'task',
      'produce an audit and a reconciliation report', 'commit');
    session.applySemanticUpdate({ event_id: 'task-instruction', requirements: [{
      id: 'audit-output', description: 'produce the requested audit',
      requiredInformation: 'verified audit artifact', likelyMechanism: 'conversion',
    }, {
      id: 'reconciliation-output', description: 'produce the reconciliation report',
      requiredInformation: 'verified reconciliation artifact', likelyMechanism: 'conversion',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    const requirements = session.snapshot().runtime.requirements;
    expect(requirements.find(value => value.id === 'root-task-requirement')?.status)
      .toBe('rejected');
    expect(requirements.filter(value => value.status === 'open').map(value => value.id))
      .toEqual(['audit-output', 'reconciliation-output']);
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

  it('searches the spawned-agent library and reuses a compatible node before deriving', async () => {
    const session = new RoyLHTBSession('reuse', 'task', 'implement and verify', 'commit',
      'roy_runtime_heuristic');
    session.applySemanticUpdate({ event_id: 'task-instruction', requirements: [{
      id: 'first-gap', description: 'inspect the first validation failure',
      requiredInformation: 'first diagnosis', likelyMechanism: 'acquisition',
    }, {
      id: 'second-gap', description: 'inspect a second validation failure',
      requiredInformation: 'second diagnosis', likelyMechanism: 'acquisition',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    const childSpecification = {
      id: 'spec-validator', nodeId: 'validator', parentId: 'root', depth: 1,
      parentGoal: 'implement and verify', triggeringGapId: 'first-gap',
      localObjective: 'Inspect one validation failure and return a concrete diagnosis.',
      refinement: { parentScope: 'implement and verify', childScope: 'inspect one failure',
        triggeringRequirementId: 'first-gap', narrowerThanParent: true,
        newInformationNeeded: 'validation output', executableEndCondition: 'diagnosis is recorded',
        duplicatedByExistingNode: false },
      requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'run validation' },
      expectedOutput: { requiredInformation: 'diagnosis', outputType: 'epistemic_report' as const },
      terminationCondition: 'return the diagnosis',
    };
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root', childSpecification });
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      calls += 1;
      const candidate = calls === 1 ? {
        id: 'duplicate-validator', kind: 'DERIVE', actorNodeId: 'root',
        description: 'Create another validator', schedulerComplexity: 2,
        action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification: {
          ...childSpecification, id: 'spec-validator-copy', nodeId: 'validator-copy',
          triggeringGapId: 'second-gap',
          refinement: { ...childSpecification.refinement,
            triggeringRequirementId: 'second-gap' },
        } },
      } : {
        id: 'reuse-validator', kind: 'CONNECT', actorNodeId: 'root',
        description: 'Assign the second diagnosis to the existing validator',
        schedulerComplexity: 1,
        action: { kind: 'CONNECT', actorNodeId: 'root', requirementId: 'second-gap',
          connection: { from: 'root', to: 'validator', required: true },
          reuseReview: { searchedNodeIds: ['validator'], decision: 'reuse_existing',
            reusableNodeId: 'validator',
            reason: 'the existing validator already has the required diagnostic capability' } },
      };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    const result = await controller.advance(session, 1);
    expect(result.status).toBe('continue');
    expect(calls).toBe(2);
    const snapshot = session.snapshot();
    expect(snapshot.runtime.nodes.map(value => value.id)).toEqual(['root', 'validator']);
    expect(snapshot.runtime.requirements.find(value => value.id === 'second-gap'))
      .toMatchObject({ status: 'assigned', assignedNodeId: 'validator' });
    expect(snapshot.runtime.communicationEdges).toContainEqual({
      from: 'root', to: 'validator', required: true, active: true,
    });
    controller.close();
  });

  it('assigns a semantic requirement to the child that produced its terminal event', () => {
    const session = new RoyLHTBSession('semantic-child', 'task', 'implement and verify', 'commit',
      'roy_runtime_heuristic');
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root',
      childSpecification: {
        id: 'spec-child', nodeId: 'child', parentId: 'root', depth: 1,
        parentGoal: 'implement and verify', triggeringGapId: 'root-task-requirement',
        localObjective: 'verify one bounded behavior', refinement: {
          parentScope: 'implement and verify', childScope: 'verify one bounded behavior',
          triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
          newInformationNeeded: 'test output', executableEndCondition: 'test result is recorded',
          duplicatedByExistingNode: false,
        }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
        externalAccess: { allowed: true, tools: ['terminal'], purpose: 'run the test' },
        expectedOutput: { requiredInformation: 'test result', outputType: 'epistemic_report' },
        terminationCondition: 'return the test result',
      } });
    session.requestTerminal({ id: 'child-command', command: 'pytest -q', timeoutMs: 1000,
      nodeId: 'child', organizationActionKind: 'EXECUTE' });
    session.acceptTerminalResult({ requestId: 'child-command', exitCode: 1, stdout: '',
      stderr: 'one assertion failed', durationMs: 1, fileChanges: [] });
    session.applySemanticUpdate({ event_id: 'result-child-command', requirements: [{
      id: 'child-gap', description: 'diagnose the failing assertion',
      requiredInformation: 'the root cause', likelyMechanism: 'conversion',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    expect(session.snapshot().runtime.requirements.find(value => value.id === 'child-gap')
      ?.parentNodeId).toBe('child');
  });

  it('classifies exhausted inactive-actor proposals as environment-invalid', async () => {
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '2';
    const session = new RoyLHTBSession('inactive', 'task', 'finish', 'commit',
      'roy_runtime_heuristic');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      calls += 1;
      return { value: { preferred_candidate_id: 'stale', candidates: [{
        id: 'stale', kind: 'EXECUTE', actorNodeId: 'returned-worker',
        description: 'stale actor', schedulerComplexity: 1, command: 'true',
        action: { kind: 'EXECUTE', actorNodeId: 'returned-worker' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    try {
      await expect(controller.advance(session, 1)).rejects
        .toThrow('environment_invalid:inactive_actor');
      expect(calls).toBe(2);
      expect(session.snapshot().runtime.stopped).toBe(false);
    } finally {
      controller.close();
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
    }
  });

  it('keeps a legal continuation when topology coverage remains incomplete after repair', async () => {
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '2';
    process.env.ROY_LHTB_TOPOLOGY_PROFILE = 'connected';
    const session = new RoyLHTBSession('shallow-interface', 'task', 'solve it', 'commit',
      'learned_information_realization');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      calls += 1;
      return { value: { preferred_candidate_id: 'inspect', candidates: [{
        id: 'inspect', kind: 'ACQUIRE', actorNodeId: 'root', description: 'inspect',
        schedulerComplexity: 1, command: 'pwd', action: { kind: 'ACQUIRE', actorNodeId: 'root' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const learnedPolicy = { async select(_state: Record<string, unknown>, candidates: Array<{
      id: string; actorNodeId: string; kind: string;
    }>) {
      const candidate = candidates[0];
      return { candidate, record: { stateFingerprint: 'state', activeNodeId: candidate.actorNodeId,
        candidateId: candidate.id, maskedOldLogProbability: 0, envelopeId: 'connected' } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('terminal_request');
      if (result.status === 'terminal_request') {
        expect(result.request.command).toBe('pwd');
        expect(result.request.nodeId).toBe('root');
      }
      expect(calls).toBe(2);
      expect(session.snapshot().runtime.stopped).toBe(false);
    } finally {
      controller.close();
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
      if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
      else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
    }
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

  it('classifies repeated malformed proposal JSON as an audited sampling failure', async () => {
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '2';
    const auditRoot = await mkdtemp(path.join(tmpdir(), 'roy-lhtb-malformed-dead-end-'));
    const session = new RoyLHTBSession('malformed-dead-end', 'task', 'finish', 'commit',
      'roy_runtime_heuristic');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      calls += 1;
      throw new LLMJSONParseError('truncated proposal', {
        content: '{"candidates":[{"command":"unfinished', model: 'mock',
        usage: { promptTokens: 2, completionTokens: 3, totalTokens: 5 },
      });
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot });
    try {
      await expect(controller.advance(session, 1))
        .rejects.toThrow('sampling_invalid:proposal_json_malformed');
      expect(calls).toBe(2);
      expect(session.snapshot().runtime.stopped).toBe(false);
      expect(session.snapshot().runtime.finalOutput).toBeUndefined();
      // The task semantic extraction is a legitimate state transition; the
      // rejected proposal itself must not fabricate an organization transition.
      expect(session.snapshot().processStates).toHaveLength(2);
      expect(session.snapshot().policyRecords).toHaveLength(0);
      const failures = (await readFile(path.join(auditRoot, 'proposal-failures.jsonl'), 'utf8'))
        .trim().split('\n').map(line => JSON.parse(line) as {
          usage: { promptTokens: number; completionTokens: number };
        });
      expect(failures).toHaveLength(2);
      expect(failures.reduce((total, failure) => total + failure.usage.promptTokens, 0)).toBe(4);
      expect(failures.reduce((total, failure) => total + failure.usage.completionTokens, 0)).toBe(6);
    } finally {
      controller.close();
      await rm(auditRoot, { recursive: true, force: true });
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
    }
  });

  it('keeps STOP legal when it is the only usable early-exploration candidate', async () => {
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '1';
    process.env.ROY_LHTB_TOPOLOGY_PROFILE = 'connected';
    const session = new RoyLHTBSession('stop-only', 'task', 'finish', 'commit',
      'learned_information_realization');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const stop = { id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'finish',
      schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root', finalOutput: 'done' } };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'stop', candidates: [stop] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    let observedState: Record<string, unknown> | undefined;
    const learnedPolicy = { async select(policyState: Record<string, unknown>) {
      observedState = policyState;
      return { candidate: stop, record: { stateFingerprint: 'state', activeNodeId: 'root',
        candidateId: 'stop', maskedOldLogProbability: 0, envelopeId: 'connected' } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('completed');
      expect(observedState?.exploration_stop_masked).toBe(true);
      expect(observedState?.active_node_legal).toEqual([true]);
      expect((observedState?.candidates as Array<Record<string, unknown>>)[0]?.legal).toBe(true);
    } finally {
      controller.close();
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
      if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
      else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
    }
  });

  it('drops terminal candidates whose actor returned inside an MCTS branch', async () => {
    const priorMCTS = process.env.ROY_LHTB_MCTS_ENABLED;
    const priorSimulations = process.env.ROY_LHTB_MCTS_SIMULATIONS;
    const priorDepth = process.env.ROY_LHTB_MCTS_MAX_DEPTH;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_MCTS_ENABLED = 'true';
    process.env.ROY_LHTB_MCTS_SIMULATIONS = '4';
    process.env.ROY_LHTB_MCTS_MAX_DEPTH = '2';
    process.env.ROY_LHTB_TOPOLOGY_PROFILE = 'compact';
    const session = new RoyLHTBSession('mcts-returned-actor', 'task', 'finish', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'finish', triggeringGapId: 'root-task-requirement',
      localObjective: 'Inspect and report one bounded result.', refinement: {
        parentScope: 'finish', childScope: 'inspect one result',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'inspection result', executableEndCondition: 'report is returned',
        duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect' },
      expectedOutput: { requiredInformation: 'inspection result',
        outputType: 'epistemic_report' as const }, terminationCondition: 'return the report',
    };
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root', childSpecification });
    const report = { id: 'worker-report', nodeId: 'worker', parentId: 'root', depth: 1,
      localObjective: childSpecification.localObjective,
      triggeringGapId: 'root-task-requirement', conclusion: 'inspection completed',
      reasoningSummary: 'bounded result recorded', claims: [], evidence: [],
      externalObservations: [], assumptions: [], uncertainty: { confidence: 0.8,
        uncertainAbout: [], confidenceBasis: 'inspection' }, conflicts: [],
      coverage: { resolved: ['root-task-requirement'], unresolved: [], notExamined: [] },
      blindSpots: [], residualRequirements: [], proposedChildren: [], resolvedParentGap: true,
      informationToPropagate: [],
    };
    const candidates = [{ id: 'return-worker', kind: 'RETURN', actorNodeId: 'worker',
      description: 'return the completed report', schedulerComplexity: 0,
      action: { kind: 'RETURN', actorNodeId: 'worker', report } },
    { id: 'execute-worker', kind: 'EXECUTE', actorNodeId: 'worker',
      description: 'run one more check', schedulerComplexity: 1, command: 'true',
      action: { kind: 'EXECUTE', actorNodeId: 'worker' } }];
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'return-worker', candidates },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    let analyses = 0;
    const learnedPolicy = { async analyze(policyState: Record<string, unknown>) {
      analyses += 1;
      expect((policyState.active_node_legal as boolean[]).some(Boolean)).toBe(true);
      const values = policyState.candidates as Array<Record<string, unknown>>;
      return { targetValue: 0.5, targetRevision: 0,
        candidatePriors: Object.fromEntries(values.map(value => [String(value.id), 1])),
        actionPriors: {}, actorPaths: [] };
    }, async targetValue() { return { targetValue: 0.5, targetRevision: 0 }; }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 7);
      expect(['continue', 'terminal_request']).toContain(result.status);
      expect(analyses).toBeGreaterThan(0);
    } finally {
      controller.close();
      if (priorMCTS === undefined) delete process.env.ROY_LHTB_MCTS_ENABLED;
      else process.env.ROY_LHTB_MCTS_ENABLED = priorMCTS;
      if (priorSimulations === undefined) delete process.env.ROY_LHTB_MCTS_SIMULATIONS;
      else process.env.ROY_LHTB_MCTS_SIMULATIONS = priorSimulations;
      if (priorDepth === undefined) delete process.env.ROY_LHTB_MCTS_MAX_DEPTH;
      else process.env.ROY_LHTB_MCTS_MAX_DEPTH = priorDepth;
      if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
      else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
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
      const snapshot = session.snapshot();
      const rawEvent = snapshot.runtimeEvents?.find(event => event.id === 'result-large');
      expect(rawEvent?.output).toBe(longOutput);
      const projectedEvent = snapshot.processStates
        .flatMap(state => state.runtimeEvents)
        .find(event => event.id === 'result-large');
      expect(projectedEvent?.output?.length).toBeLessThan(longOutput.length);
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
    const requests: Array<Array<{ role: string; content: string }>> = [];
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      role: string; content: string;
    }>) {
      requests.push(messages);
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
      expect(requests[1]?.at(-1)?.content).toContain('python missing.py');
      expect(requests[1]?.at(-1)?.content).toContain('Do not reproduce an unchanged rejected candidate');
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
