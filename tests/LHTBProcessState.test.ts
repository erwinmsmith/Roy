import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { GlobalEpistemicStateRecorder, LHTBAutonomousController,
  compactProposalRepairRequest, repairProposalJSONStructure, resolveTopologySamplingProfile,
  RoyLHTBSession, scheduledOrganizationContextNode,
  topologySamplingCandidateLogitBias,
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
  it('mechanically repairs extra, missing, and trailing JSON delimiters', () => {
    const extra = '{"preferred_candidate_id":"a","candidates":[{"id":"a",'
      + '"action":{"kind":"STOP"}}},{"id":"b","action":{"kind":"STOP"}}}]}';
    expect(repairProposalJSONStructure(extra)?.candidates).toHaveLength(2);
    const missing = '{"preferred_candidate_id":"a","candidates":[{"id":"a",'
      + '"action":{"kind":"STOP"},{"id":"b","action":{"kind":"STOP"}}';
    expect(repairProposalJSONStructure(missing)?.candidates).toHaveLength(2);
    expect(repairProposalJSONStructure('{"candidates":[{"id":"unfinished'))
      .toBeUndefined();
  });

  it('names every externally executable child in candidate repair requests', () => {
    const payload = JSON.parse(compactProposalRepairRequest({ organization: {
      rootId: 'root', activeNodes: [{ id: 'worker' }],
    } }, ['missing_external_child_progress_candidate:worker',
      'missing_child_return_candidate:worker'], []));
    expect(payload.legalInterface.requiredExternalChildProgressNodeIds).toEqual(['worker']);
    expect(payload.legalInterface.requiredChildReturnNodeIds).toEqual(['worker']);
    expect(payload.legalInterface.returnCandidateSchema.action.report.nodeId)
      .toBe('<same requiredChildReturnNodeId>');
    expect(payload.legalInterface.returnCandidateSchema.action.report)
      .toHaveProperty('informationToPropagate');
    expect(payload.instruction.join(' ')).toContain('ACQUIRE or EXECUTE');
    expect(payload.instruction.join(' ')).toContain('evidence-grounded report');
  });

  it('covers single-agent through connected topology profiles without changing utility', () => {
    expect([0, 1, 2, 3, 4].map(topologySamplingProfile).map(value => value.id))
      .toEqual(['single', 'compact', 'branching', 'recursive', 'connected']);
    expect(topologySamplingProfile(0).preferredNodeRange).toEqual([1, 1]);
    expect(topologySamplingProfile(4).preferredNodeRange).toEqual([6, 8]);
    expect(topologySamplingProfile(3).preferredMinimumDepth).toBe(2);
    expect(resolveTopologySamplingProfile(0, 'recursive').preferredNodeRange).toEqual([5, 7]);
    expect(() => resolveTopologySamplingProfile(0, 'invalid')).toThrow(/Invalid/);
  });

  it('leaves formal topology unconstrained when no diagnostic profile is set', () => {
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    const session = new RoyLHTBSession('unconstrained', 'task', 'inspect and solve', 'commit',
      'learned_information_realization', 'same', 0);
    expect(topologySamplingCandidateLogitBias(session.snapshot(), {
      kind: 'DERIVE', actorNodeId: 'root', action: { kind: 'DERIVE', actorNodeId: 'root' },
    })).toBe(0);
    if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
  });

  it('masks topology-changing candidates in the learned single-agent profile', async () => {
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_TOPOLOGY_PROFILE = 'single';
    const session = new RoyLHTBSession('single-profile', 'task', 'inspect and solve', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'inspect and solve', triggeringGapId: 'root-task-requirement',
      realizationMode: 'acquire_external' as const,
      localObjective: 'Inspect one part.', refinement: {
        parentScope: 'inspect and solve', childScope: 'inspect one part',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'one result', executableEndCondition: 'result recorded',
        duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect' },
      expectedOutput: { requiredInformation: 'one result',
        outputType: 'epistemic_report' as const }, terminationCondition: 'return result',
    };
    const candidates = [{ id: 'derive', kind: 'DERIVE', actorNodeId: 'root',
      description: 'derive a worker', schedulerComplexity: 1,
      action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification } },
    { id: 'inspect', kind: 'ACQUIRE', actorNodeId: 'root', description: 'inspect locally',
      schedulerComplexity: 1, command: 'pwd',
      action: { kind: 'ACQUIRE', actorNodeId: 'root' } }];
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'derive', candidates },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    let policyCandidates: Array<Record<string, unknown>> = [];
    const learnedPolicy = { async select(policyState: Record<string, unknown>, values: Array<
      Record<string, unknown>>) {
      policyCandidates = values;
      return { candidate: values[0], record: { stateFingerprint: 'state',
        contextNodeId: 'root', candidateId: String(values[0]?.id),
        maskedOldLogProbability: 0, envelopeId: 'single' } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('terminal_request');
      expect(policyCandidates.map(value => value.kind)).toEqual(['CONTINUE', 'FINISH']);
      expect(session.snapshot().runtime.nodes).toHaveLength(1);
      expect(session.snapshot().runtime.derivationEdges).toHaveLength(0);
    } finally {
      controller.close();
      if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
      else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
    }
  });

  it('turns a recursive profile into a real root to sub to subsub derivation', () => {
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_TOPOLOGY_PROFILE = 'recursive';
    const session = new RoyLHTBSession('recursive', 'task', 'implement and verify', 'commit',
      'learned_information_realization', 'same', 3);
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
      realizationMode: 'acquire_external' as const,
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
    expect(scheduledOrganizationContextNode(shallow)).toBe('child');
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
    expect(scheduledOrganizationContextNode(session.snapshot())).toBe('grandchild');
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
    if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
  });

  it('keeps full semantic audit data once while bounding every value-state projection', () => {
    const session = new RoyLHTBSession('bounded-state', 'task', 'solve', 'commit');
    for (let index = 0; index < 320; index += 1) {
      session.applySemanticUpdate({ event_id: `semantic-${index}`, requirements: [], claims: [{
        id: `claim-${index}`, statement: `claim ${index}`, status: 'tentative', originNodeId: 'root',
      }], assumptions: [], evidence: [{ id: `evidence-${index}`, content: `evidence ${index}`,
        supports: [`claim-${index}`], contradicts: [], provenance: 'mock' }],
      external_observations: [], blind_spots: [`blind-${index}`], relations: [{
        left_id: `evidence-${index}`, right_id: `claim-${index}`, label: 'entail',
        probabilities: { entail: 1, contradict: 0, unknown: 0 },
        provenance: { model: 'mock', model_revision: 'fixed', cache_key: `key-${index}` },
      }] });
    }
    const snapshot = session.snapshot();
    const latest = snapshot.processStates.at(-1)!;
    expect(snapshot.semanticOverlay.claims).toHaveLength(320);
    expect(snapshot.semanticOverlay.evidence).toHaveLength(320);
    expect(snapshot.semanticOverlay.relations).toHaveLength(320);
    expect(latest.claims).toHaveLength(96);
    expect(latest.evidence).toHaveLength(96);
    expect(latest.semanticRelations.length).toBeLessThanOrEqual(256);
    expect(latest.blindSpots).toHaveLength(96);
    expect(snapshot.processStates).toHaveLength(321);
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

  it('masks STOP after verifier rejection until a new external result arrives', () => {
    const session = new RoyLHTBSession('retry-mask', 'task', 'solve it', 'commit');
    session.applyOrganizationAction({ kind: 'STOP', actorNodeId: 'root', finalOutput: 'draft' });
    session.resumeAfterVerifierRejection('not solved');
    const execute = { id: 'repair', kind: 'EXECUTE', actorNodeId: 'root',
      description: 'apply a repair', schedulerComplexity: 1, command: 'true',
      action: { kind: 'EXECUTE', actorNodeId: 'root' } };
    const stop = { id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'submit',
      schedulerComplexity: 1,
      action: { kind: 'STOP', actorNodeId: 'root', finalOutput: 'done' } };
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider: { isConfigured: () => true },
      semantic, auditRoot: false, learnedPolicy: { close() {} } as never });
    const policyState = (controller as unknown as { policyState(
      snapshot: ReturnType<RoyLHTBSession['snapshot']>, candidates: unknown[]):
      Record<string, unknown> }).policyState.bind(controller);
    try {
      const before = policyState(session.snapshot(), [execute, stop]);
      expect(before.verifier_retry_stop_masked).toBe(true);
      expect((before.candidates as Array<{ id: string; legal: boolean }>)
        .find(candidate => candidate.id === 'controller:FINISH')?.legal).toBe(false);
      session.requestTerminal({ id: 'repair-command', command: 'true', timeoutMs: 1000,
        nodeId: 'root', organizationActionKind: 'EXECUTE' });
      session.acceptTerminalResult({ requestId: 'repair-command', exitCode: 0, stdout: '',
        stderr: '', durationMs: 1 });
      const after = policyState(session.snapshot(), [execute, stop]);
      expect(after.verifier_retry_stop_masked).toBe(false);
      expect((after.candidates as Array<{ id: string; legal: boolean }>)
        .find(candidate => candidate.id === 'controller:FINISH')?.legal).toBe(true);
    } finally {
      controller.close();
    }
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

  it('records a semantic extractor outage without killing the rollout', async () => {
    const session = new RoyLHTBSession('semantic-failure', 'task', 'solve it', 'commit');
    let semanticCalls = 0;
    const semantic = {
      async processEvent() {
        semanticCalls += 1;
        throw new Error('The user aborted a request.');
      },
      close() {},
    };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      throw new Error('provider should not be called');
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    try {
      await controller.prepareDecisionBoundary(session);
      await controller.prepareDecisionBoundary(session);
      const snapshot = session.snapshot();
      expect(semanticCalls).toBe(1);
      expect(snapshot.processedSemanticEventIds).toContain('task-instruction');
      expect(snapshot.processStates.at(-1)?.blindSpots).toContain(
        'Semantic projection unavailable for runtime event task-instruction; raw event retained.',
      );
      expect(snapshot.runtimeEvents).toContainEqual(expect.objectContaining({
        kind: 'failure',
        attributes: expect.objectContaining({
          failureType: 'semantic_projection',
          sourceEventId: 'task-instruction',
          reason: 'The user aborted a request.',
        }),
      }));
    } finally {
      controller.close();
    }
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
      realizationMode: 'acquire_external' as const,
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
      realizationMode: 'acquire_external' as const,
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
    expect(scheduledOrganizationContextNode(session.snapshot())).toBe('validator');
    session.requestTerminal({ id: 'validator-inspection', command: 'true', timeoutMs: 1000,
      nodeId: 'validator', organizationActionKind: 'ACQUIRE' });
    session.acceptTerminalResult({ requestId: 'validator-inspection', exitCode: 0,
      stdout: 'first diagnosis', stderr: '', durationMs: 1 });
    expect(scheduledOrganizationContextNode(session.snapshot())).toBe('root');
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
        realizationMode: 'acquire_external' as const,
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
      return { candidate, record: { stateFingerprint: 'state', contextNodeId: candidate.actorNodeId,
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
      expect(calls).toBe(1);
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

  it('does not let Worker payload preference define learned Controller support', async () => {
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '1';
    delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    const session = new RoyLHTBSession('stop-only', 'task', 'finish', 'commit',
      'learned_information_realization');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const stop = { id: 'stop', kind: 'STOP', actorNodeId: 'root', description: 'finish',
      schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root', finalOutput: 'done' } };
    const rootReturn = { id: 'root-return', kind: 'RETURN', actorNodeId: 'root',
      description: 'incorrect root return', schedulerComplexity: 0,
      action: { kind: 'RETURN', actorNodeId: 'root', report: {} } };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'stop', candidates: [rootReturn, stop] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    let observedState: Record<string, unknown> | undefined;
    const learnedPolicy = { async select(policyState: Record<string, unknown>, values: Array<{
      id: string; kind: string;
    }>) {
      observedState = policyState;
      const candidate = values.find(value => value.kind === 'FINISH')!;
      return { candidate, record: { stateFingerprint: 'state', contextNodeId: 'root',
        candidateId: candidate.id, maskedOldLogProbability: 0, envelopeId: 'connected' } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('completed');
      expect(observedState?.exploration_stop_masked).toBe(false);
      expect(observedState?.context_node_id).toBe('root');
      expect((observedState?.candidates as Array<Record<string, unknown>>)
        .map(candidate => candidate.kind)).toEqual([
          'CONTINUE', 'DERIVE_INFO', 'DERIVE_ORG', 'FINISH',
        ]);
      expect((observedState?.candidates as Array<Record<string, unknown>>)
        .find(candidate => candidate.kind === 'FINISH')?.legal).toBe(true);
    } finally {
      controller.close();
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
      if (priorProfile === undefined) delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
      else process.env.ROY_LHTB_TOPOLOGY_PROFILE = priorProfile;
    }
  });

  it('invokes the shared direct actor with a child node local context', async () => {
    const priorMCTS = process.env.ROY_LHTB_MCTS_ENABLED;
    const priorSimulations = process.env.ROY_LHTB_MCTS_SIMULATIONS;
    const priorDepth = process.env.ROY_LHTB_MCTS_MAX_DEPTH;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_MCTS_ENABLED = 'true';
    process.env.ROY_LHTB_MCTS_SIMULATIONS = '4';
    process.env.ROY_LHTB_MCTS_MAX_DEPTH = '2';
    delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    const session = new RoyLHTBSession('mcts-returned-actor', 'task', 'finish', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'finish', triggeringGapId: 'root-task-requirement',
      realizationMode: 'acquire_external' as const,
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
    let observedState: Record<string, unknown> | undefined;
    const learnedPolicy = { async select(policyState: Record<string, unknown>, values: Array<{
      id: string; kind: string;
    }>) {
      observedState = policyState;
      expect(values.map(value => value.kind)).toEqual([
        'CONTINUE', 'DERIVE_INFO', 'DERIVE_ORG', 'PRUNE',
      ]);
      const candidate = values.find(value => value.kind === 'CONTINUE')!;
      return { candidate, record: { stateFingerprint: String(policyState.state_fingerprint),
        contextNodeId: 'worker', candidateId: candidate.id, maskedOldLogProbability: 0,
        envelopeId: 'lhtb-open', behaviorPolicy: 'actor', policyState } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 7);
      expect(result.status).toBe('terminal_request');
      expect(observedState?.context_node_id).toBe('worker');
      expect((observedState?.context_node as Record<string, unknown>).local_objective)
        .toBe(childSpecification.localObjective);
      expect((observedState?.context_node as Record<string, unknown>).ancestry)
        .toEqual([expect.objectContaining({ id: 'root', depth: 0 })]);
      expect((observedState?.context_node as Record<string, unknown>).requirements)
        .toEqual([expect.objectContaining({
          id: 'root-task-requirement', status: 'assigned', assigned_node_id: 'worker',
        })]);
      expect((observedState?.topology_search as Record<string, unknown>).mode)
        .toBe('actor_direct_on_policy');
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

  it('keeps a legal single-node STOP in direct actor support', async () => {
    const priorMCTS = process.env.ROY_LHTB_MCTS_ENABLED;
    const priorSimulations = process.env.ROY_LHTB_MCTS_SIMULATIONS;
    const priorDepth = process.env.ROY_LHTB_MCTS_MAX_DEPTH;
    const priorProfile = process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    process.env.ROY_LHTB_MCTS_ENABLED = 'true';
    process.env.ROY_LHTB_MCTS_SIMULATIONS = '4';
    process.env.ROY_LHTB_MCTS_MAX_DEPTH = '2';
    delete process.env.ROY_LHTB_TOPOLOGY_PROFILE;
    const session = new RoyLHTBSession('mcts-stop-support', 'task', 'finish', 'commit',
      'learned_information_realization');
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: { preferred_candidate_id: 'inspect', candidates: [{
        id: 'inspect', kind: 'ACQUIRE', actorNodeId: 'root', description: 'inspect',
        schedulerComplexity: 1, command: 'pwd', action: { kind: 'ACQUIRE', actorNodeId: 'root' },
      }] }, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const learnedPolicy = { async select(policyState: Record<string, unknown>, candidates: Array<{
      id: string;
    }>) {
      const values = policyState.candidates as Array<Record<string, unknown>>;
      expect((policyState.topology_search as Record<string, unknown>).mode)
        .toBe('actor_direct_on_policy');
      expect(values.map(value => value.id)).toContain('controller:FINISH');
      const candidate = candidates.find(value => value.id === 'controller:FINISH')!;
      return { candidate, record: { stateFingerprint: String(policyState.state_fingerprint),
        contextNodeId: 'root', candidateId: candidate.id, maskedOldLogProbability: 0,
        envelopeId: 'lhtb-open', behaviorPolicy: 'actor', policyState } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('completed');
      expect(session.snapshot().policyRecords.at(-1)?.candidateId)
        .toBe('controller:FINISH');
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

  it('executes one direct actor sample without hypothetical search expansions',
    async () => {
      const prior = {
        enabled: process.env.ROY_LHTB_MCTS_ENABLED,
        simulations: process.env.ROY_LHTB_MCTS_SIMULATIONS,
        depth: process.env.ROY_LHTB_MCTS_MAX_DEPTH,
        expansions: process.env.ROY_LHTB_MCTS_AGENT_EXPANSIONS,
        attempts: process.env.ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS,
      };
      process.env.ROY_LHTB_MCTS_ENABLED = 'true';
      process.env.ROY_LHTB_MCTS_SIMULATIONS = '8';
      process.env.ROY_LHTB_MCTS_MAX_DEPTH = '2';
      process.env.ROY_LHTB_MCTS_AGENT_EXPANSIONS = '2';
      process.env.ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS = '1';
      const session = new RoyLHTBSession('dynamic-mcts', 'task', 'finish', 'commit',
        'learned_information_realization');
      const childSpecification = {
        id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
        parentGoal: 'finish', triggeringGapId: 'root-task-requirement',
        realizationMode: 'acquire_external' as const,
        localObjective: 'Inspect the environment and return verified evidence.', refinement: {
          parentScope: 'finish', childScope: 'inspect the environment',
          triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
          newInformationNeeded: 'environment evidence',
          executableEndCondition: 'one terminal inspection completes',
          duplicatedByExistingNode: false,
        }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
        externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect' },
        expectedOutput: { requiredInformation: 'verified evidence',
          outputType: 'epistemic_report' as const },
        terminationCondition: 'return after inspection',
      };
      let calls = 0;
      const observedContexts: string[] = [];
      const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
        role: string; content: string;
      }>) {
        calls += 1;
        const request = JSON.parse(messages.at(-1)?.content ?? '{}') as Record<string, unknown>;
        const organization = request.organization as Record<string, unknown> | undefined;
        const active = organization?.activeNodes as Array<Record<string, unknown>> | undefined;
        const context = String((request.mctsSearchExpansion as Record<string, unknown> | undefined)
          ?.contextNodeId ?? active?.[0]?.id ?? 'root');
        observedContexts.push(context);
        const candidate = request.mctsSearchExpansion ? {
          id: 'child-execute', kind: 'EXECUTE', actorNodeId: 'worker',
          description: 'inspect from the child context', schedulerComplexity: 1,
          command: 'true', action: { kind: 'EXECUTE', actorNodeId: 'worker' },
        } : {
          id: 'derive-worker', kind: 'DERIVE', actorNodeId: 'root',
          description: 'derive an inspecting child', schedulerComplexity: 1,
          action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification },
        };
        return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
          completion: { content: '{}', model: 'mock-deepseek', usage: {
            promptTokens: 3, completionTokens: 2, totalTokens: 5,
          } } };
      } };
      const semantic = { async processEvent(event: { id: string }) {
        return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
          external_observations: [], blind_spots: [], relations: [] };
      }, close() {} };
      const learnedPolicy = { async select(policyState: Record<string, unknown>, candidates: Array<{
        id: string;
      }>) {
        const candidate = candidates.find(value => value.id === 'controller:DERIVE_INFO')!;
        return { candidate, record: { stateFingerprint: String(policyState.state_fingerprint),
          contextNodeId: 'root', candidateId: candidate.id, maskedOldLogProbability: 0,
          envelopeId: 'lhtb-open', behaviorPolicy: 'actor', policyState } };
      }, close() {} };
      const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
        learnedPolicy: learnedPolicy as never });
      try {
        const result = await controller.advance(session, 17);
        expect(result.status).toBe('continue');
        const record = session.snapshot().policyRecords.at(-1)!;
        expect(calls).toBe(1);
        expect(observedContexts).toEqual(['root']);
        expect(record.behaviorPolicy).toBe('actor');
        expect(record.mctsSearchSamples).toBeUndefined();
        expect(session.snapshot().runtime.nodes.map(node => node.id)).toContain('worker');
      } finally {
        controller.close();
        for (const [name, value] of Object.entries({
          ROY_LHTB_MCTS_ENABLED: prior.enabled,
          ROY_LHTB_MCTS_SIMULATIONS: prior.simulations,
          ROY_LHTB_MCTS_MAX_DEPTH: prior.depth,
          ROY_LHTB_MCTS_AGENT_EXPANSIONS: prior.expansions,
          ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS: prior.attempts,
        })) {
          if (value === undefined) delete process.env[name];
          else process.env[name] = value;
        }
      }
    });

  it('requires externally executable children to produce a local terminal result before return', async () => {
    const session = new RoyLHTBSession('external-child-return', 'task', 'finish', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'finish', triggeringGapId: 'root-task-requirement',
      realizationMode: 'acquire_external' as const,
      localObjective: 'Inspect and verify the workspace.', refinement: {
        parentScope: 'finish', childScope: 'inspect and verify the workspace',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'verified workspace state',
        executableEndCondition: 'a terminal check has completed', duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'run the check' },
      expectedOutput: { requiredInformation: 'verified result',
        outputType: 'epistemic_report' as const }, terminationCondition: 'return verified result',
    };
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root', childSpecification });
    const report = { id: 'worker-report', nodeId: 'worker', parentId: 'root', depth: 1,
      localObjective: childSpecification.localObjective,
      triggeringGapId: 'root-task-requirement', conclusion: 'verification complete',
      reasoningSummary: 'the workspace was checked', claims: [], evidence: [],
      externalObservations: [], assumptions: [], uncertainty: { confidence: 0.8,
        uncertainAbout: [], confidenceBasis: 'terminal check' }, conflicts: [],
      coverage: { resolved: [], unresolved: ['root-task-requirement'], notExamined: [] },
      blindSpots: [], residualRequirements: [], proposedChildren: [], resolvedParentGap: false,
      informationToPropagate: [],
    };
    const response = { preferred_candidate_id: 'return-worker', candidates: [{
      id: 'return-worker', kind: 'RETURN', actorNodeId: 'worker',
      description: 'return verified result', schedulerComplexity: 1,
      action: { kind: 'RETURN', actorNodeId: 'worker', report },
    }] };
    const blockedResponse = { ...response, candidates: [...response.candidates, {
      id: 'stop-root', kind: 'STOP', actorNodeId: 'root', description: 'submit partial work',
      schedulerComplexity: 0, action: { kind: 'STOP', actorNodeId: 'root',
        finalOutput: 'partial' },
    }] };
    const semantic = { async processEvent() { return { event_id: 'none', requirements: [],
      claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
      relations: [] }; }, close() {} };
    const provider = { isConfigured: () => true, async completeJSONWithUsage() {
      return { value: blockedResponse, completion: { content: '{}', model: 'mock', usage: {
        promptTokens: 1, completionTokens: 1, totalTokens: 2,
      } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    type Harness = {
      validateCandidates(value: unknown, current: RoyLHTBSession): {
        candidates: Array<{ id: string }>;
        dispositions: Array<{ reasons: string[] }>;
      };
      structuralCandidateDeficits(snapshot: ReturnType<RoyLHTBSession['snapshot']>,
        candidates: Array<{ kind: string; actorNodeId: string }>): string[];
      legalStructuralControllerCandidates(snapshot: ReturnType<RoyLHTBSession['snapshot']>):
        Array<{ kind: string }>;
    };
    const harness = controller as unknown as Harness;
    const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
    try {
      const rejected = harness.validateCandidates(response, session);
      expect(rejected.candidates).toHaveLength(0);
      expect(rejected.dispositions[0].reasons)
        .toContain('external_child_return_without_local_terminal_result');
      expect(harness.structuralCandidateDeficits(session.snapshot(), []))
        .toContain('missing_external_child_progress_candidate:worker');
      expect(harness.legalStructuralControllerCandidates(session.snapshot())
        .map(value => value.kind)).toEqual([
          'CONTINUE', 'DERIVE_INFO', 'DERIVE_ORG', 'PRUNE',
        ]);
      session.requestTerminal({ id: 'worker-check', command: 'true', timeoutMs: 1000,
        nodeId: 'worker', organizationActionKind: 'EXECUTE' });
      session.acceptTerminalResult({ requestId: 'worker-check', exitCode: 0, stdout: '',
        stderr: '', durationMs: 1 });
      const accepted = harness.validateCandidates(response, session);
      expect(accepted.candidates.map(value => value.id)).toContain('return-worker');
      const providerShaped = structuredClone(response) as unknown as { candidates: Array<{
        report?: typeof report; action: { report?: typeof report } }> };
      providerShaped.candidates[0].report = report;
      delete providerShaped.candidates[0].action.report;
      const normalizedProviderShape = harness.validateCandidates(providerShaped, session);
      expect(normalizedProviderShape.candidates.map(value => value.id))
        .toContain('return-worker');
      const providerClaimShape = structuredClone(response) as unknown as { candidates: Array<{
        action: { report: typeof report & { claims: Array<Record<string, unknown>> } } }> };
      providerClaimShape.candidates[0].action.report.claims = [{
        id: 'provider-claim', statement: 'one verified finding', type: 'finding',
        confidence: 0.8, provenance: 'worker',
      }];
      const normalizedClaim = harness.validateCandidates(providerClaimShape, session);
      expect((normalizedClaim.candidates[0] as unknown as { action: { report: {
        claims: Array<Record<string, unknown>> } } }).action.report.claims[0])
        .toMatchObject({ status: 'tentative', originNodeId: 'worker' });
      const malformedResidual = structuredClone(response);
      malformedResidual.candidates[0].action.report.residualRequirements = [{} as never];
      const normalized = harness.validateCandidates(malformedResidual, session);
      expect(normalized.candidates).toHaveLength(1);
      expect((normalized.candidates[0] as unknown as { action: { report: {
        residualRequirements: unknown[] } } }).action.report.residualRequirements).toEqual([]);
      const providerObservationShape = structuredClone(response) as unknown as { candidates: Array<{
        action: { report: typeof report & { externalObservations: Array<Record<string, unknown>> } } }> };
      providerObservationShape.candidates[0].action.report.externalObservations = [{
        id: 'terminal-observation', source: 'terminal', observation: 'manifest inspected',
        timestamp: '2026-01-01T00:00:00Z',
      }];
      const normalizedObservation = harness.validateCandidates(providerObservationShape, session);
      expect(normalizedObservation.candidates).toHaveLength(1);
      const normalizedObservationAction = normalizedObservation.candidates[0] as unknown as {
        action: { report: { externalObservations: Array<Record<string, unknown>> } } };
      expect(normalizedObservationAction.action.report.externalObservations[0]).toMatchObject({
        id: 'terminal-observation', sourceType: 'tool',
        queryOrAction: childSpecification.localObjective,
        observation: 'manifest inspected', provenance: 'terminal:worker-report', supports: [],
      });
      expect(harness.structuralCandidateDeficits(session.snapshot(), []))
        .not.toContain('missing_external_child_progress_candidate:worker');
      expect(harness.legalStructuralControllerCandidates(session.snapshot())
        .map(value => value.kind)).toContain('RETURN');
      expect(harness.structuralCandidateDeficits(session.snapshot(), []))
        .toContain('missing_child_return_candidate:worker');
      expect(harness.structuralCandidateDeficits(session.snapshot(), accepted.candidates))
        .not.toContain('missing_child_return_candidate:worker');
      session.applyOrganizationAction(normalizedObservationAction.action as never);
      expect(session.snapshot().processStates.at(-1)?.externalObservations)
        .toEqual(expect.arrayContaining([expect.objectContaining({
          id: 'terminal-observation', sourceType: 'tool', observation: 'manifest inspected',
        })]));
    } finally {
      controller.close();
      if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
    }
  });

  it('executes DERIVE_INFO through one child acquisition macro boundary', async () => {
    const session = new RoyLHTBSession('info-macro', 'task', 'inspect and solve', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-info', nodeId: 'info-child', parentId: 'root', depth: 1,
      parentGoal: 'inspect and solve', triggeringGapId: 'root-task-requirement',
      realizationMode: 'acquire_external' as const,
      localObjective: 'Inspect one missing environmental fact.', refinement: {
        parentScope: 'inspect and solve', childScope: 'inspect one fact',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'one environmental fact',
        executableEndCondition: 'one terminal observation is recorded',
        duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect' },
      expectedOutput: { requiredInformation: 'one verified fact',
        outputType: 'epistemic_report' as const },
      terminationCondition: 'record the observation',
      reuseReview: { searchedNodeIds: [], decision: 'spawn_distinct' as const,
        reason: 'no spawned agent exists' },
    };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      content: string;
    }>) {
      calls += 1;
      const request = JSON.parse(messages.at(-1)?.content ?? '{}') as Record<string, unknown>;
      const selected = String(request.selectedControllerAction);
      const candidate = selected === 'CONTINUE' ? {
        id: 'info-acquire', kind: 'ACQUIRE', actorNodeId: 'info-child',
        description: 'inspect one fact', schedulerComplexity: 1, command: 'printf fact',
        action: { kind: 'ACQUIRE', actorNodeId: 'info-child' },
      } : {
        id: 'derive-info', kind: 'DERIVE', actorNodeId: 'root',
        description: 'derive an information child', schedulerComplexity: 1,
        action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification },
      };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const learnedPolicy = { async select(policyState: Record<string, unknown>, candidates: Array<{
      id: string; kind: string;
    }>) {
      const candidate = candidates.find(value => value.kind === 'DERIVE_INFO')!;
      return { candidate, record: { stateFingerprint: String(policyState.state_fingerprint),
        contextNodeId: 'root', candidateId: candidate.id, maskedOldLogProbability: 0,
        envelopeId: 'lhtb-open', behaviorPolicy: 'actor', policyState } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advanceMacro(session, 19);
      expect(result.status).toBe('terminal_request');
      if (result.status !== 'terminal_request') throw new Error('expected terminal request');
      expect(result.request.nodeId).toBe('info-child');
      expect(result.controllerActionKind).toBe('DERIVE_INFO');
      expect(calls).toBe(2);
      expect(session.snapshot().policyRecords).toHaveLength(1);
    } finally {
      controller.close();
    }
  });

  it('executes DERIVE_ORG through a child report integration macro boundary', async () => {
    const session = new RoyLHTBSession('org-macro', 'task', 'synthesize and solve', 'commit',
      'learned_information_realization');
    const childSpecification = {
      id: 'spec-org', nodeId: 'org-child', parentId: 'root', depth: 1,
      parentGoal: 'synthesize and solve', triggeringGapId: 'root-task-requirement',
      realizationMode: 'organize_knowledge' as const,
      localObjective: 'Synthesize the represented evidence.', refinement: {
        parentScope: 'synthesize and solve', childScope: 'synthesize represented evidence',
        triggeringRequirementId: 'root-task-requirement', narrowerThanParent: true,
        newInformationNeeded: 'none; organize represented evidence',
        executableEndCondition: 'one report is integrated', duplicatedByExistingNode: false,
      }, requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: false, tools: [] },
      expectedOutput: { requiredInformation: 'organized conclusion',
        outputType: 'epistemic_report' as const },
      terminationCondition: 'return one synthesis report',
      reuseReview: { searchedNodeIds: [], decision: 'spawn_distinct' as const,
        reason: 'no spawned agent exists' },
    };
    const report = { id: 'org-report', nodeId: 'org-child', parentId: 'root', depth: 1,
      localObjective: childSpecification.localObjective,
      triggeringGapId: 'root-task-requirement', conclusion: 'organized conclusion',
      reasoningSummary: 'represented evidence was synthesized', claims: [], evidence: [],
      externalObservations: [], assumptions: [], uncertainty: { confidence: 0.7,
        uncertainAbout: [], confidenceBasis: 'represented evidence' }, conflicts: [],
      coverage: { resolved: ['root-task-requirement'], unresolved: [], notExamined: [] },
      blindSpots: [], residualRequirements: [], proposedChildren: [], resolvedParentGap: true,
      informationToPropagate: ['organized conclusion'],
    };
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      content: string;
    }>) {
      calls += 1;
      const request = JSON.parse(messages.at(-1)?.content ?? '{}') as Record<string, unknown>;
      const selected = String(request.selectedControllerAction);
      const candidate = selected === 'RETURN' ? {
        id: 'org-return', kind: 'RETURN', actorNodeId: 'org-child',
        description: 'integrate the synthesis', schedulerComplexity: 0,
        action: { kind: 'RETURN', actorNodeId: 'org-child', report },
      } : {
        id: 'derive-org', kind: 'DERIVE', actorNodeId: 'root',
        description: 'derive an organization child', schedulerComplexity: 1,
        action: { kind: 'DERIVE', actorNodeId: 'root', childSpecification },
      };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const learnedPolicy = { async select(policyState: Record<string, unknown>, candidates: Array<{
      id: string; kind: string;
    }>) {
      const candidate = candidates.find(value => value.kind === 'DERIVE_ORG')!;
      return { candidate, record: { stateFingerprint: String(policyState.state_fingerprint),
        contextNodeId: 'root', candidateId: candidate.id, maskedOldLogProbability: 0,
        envelopeId: 'lhtb-open', behaviorPolicy: 'actor', policyState } };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false,
      learnedPolicy: learnedPolicy as never });
    try {
      const result = await controller.advanceMacro(session, 23);
      expect(result.status).toBe('continue');
      expect(result.controllerActionKind).toBe('DERIVE_ORG');
      expect(calls).toBe(2);
      expect(session.snapshot().policyRecords).toHaveLength(1);
      expect(session.snapshot().runtime.nodes.find(node => node.id === 'org-child')?.status)
        .toBe('returned');
      expect(session.snapshot().runtime.reports.map(value => value.id)).toContain('org-report');
      const integrated = session.snapshot().runtime.reports.find(value => value.id === 'org-report');
      expect(integrated?.claims).toEqual([expect.objectContaining({
        statement: 'organized conclusion', status: 'tentative', originNodeId: 'org-child',
      })]);
      expect(session.snapshot().processStates.at(-1)?.claims)
        .toEqual(expect.arrayContaining([expect.objectContaining({
          statement: 'organized conclusion', originNodeId: 'org-child',
        })]));
    } finally {
      controller.close();
    }
  });

  it('uses frozen finalize-now for bounded root-local readout without actor decisions', async () => {
    const session = new RoyLHTBSession('finalize-a0', 'task', 'finish the artifact', 'commit',
      'learned_information_realization');
    let calls = 0;
    const provider = { isConfigured: () => true, async completeJSONWithUsage(messages: Array<{
      content: string;
    }>) {
      calls += 1;
      const request = JSON.parse(messages.at(-1)?.content ?? '{}') as Record<string, unknown>;
      const instruction = String(request.selectedControllerInstruction ?? '');
      if (calls === 1) {
        expect(instruction).toContain(
          'requires exactly one of these Runtime payload kinds: ACQUIRE, EXECUTE');
        const acquire = { id: 'inspect-before-convert', kind: 'ACQUIRE', actorNodeId: 'root',
          description: 'inspect before converting', schedulerComplexity: 1,
          command: 'pwd', action: { kind: 'ACQUIRE', actorNodeId: 'root' } };
        return { value: { preferred_candidate_id: acquire.id, candidates: [acquire] },
          completion: { content: '{}', model: 'mock', usage: {
            promptTokens: 1, completionTokens: 1, totalTokens: 2,
          } } };
      }
      const candidate = { id: 'finalize-execute', kind: 'EXECUTE', actorNodeId: 'root',
        description: 'materialize the represented solution', schedulerComplexity: 1,
        command: 'printf done > result.txt',
        action: { kind: 'EXECUTE', actorNodeId: 'root' } };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: '{}', model: 'mock', usage: {
          promptTokens: 1, completionTokens: 1, totalTokens: 2,
        } } };
    } };
    const semantic = { async processEvent(event: { id: string }) {
      return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
        external_observations: [], blind_spots: [], relations: [] };
    }, close() {} };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    try {
      const inspection = await controller.advanceFinalizeNow(session);
      expect(inspection.status).toBe('terminal_request');
      if (inspection.status !== 'terminal_request') throw new Error('expected terminal request');
      expect(inspection.request.organizationActionKind).toBe('ACQUIRE');
      session.acceptTerminalResult({ requestId: inspection.request.id, exitCode: 0,
        stdout: '/app', stderr: '', durationMs: 1, fileChanges: [] });
      session.applyOrganizationAction({ kind: 'ACQUIRE', actorNodeId: 'root', observation: {
        id: 'a0-inspection', sourceType: 'environment', queryOrAction: 'pwd',
        observation: '/app', provenance: 'test', supports: [],
      } });
      const conversion = await controller.advanceFinalizeNow(session);
      expect(conversion.status).toBe('terminal_request');
      if (conversion.status !== 'terminal_request') throw new Error('expected terminal request');
      expect(conversion.request.nodeId).toBe('root');
      expect(conversion.request.organizationActionKind).toBe('EXECUTE');
      expect(conversion.request.command).toContain('result.txt');
      expect(calls).toBe(2);
      expect(session.snapshot().policyRecords).toHaveLength(0);
    } finally {
      controller.close();
    }
  });

  it('submits the current artifact when frozen A0 cannot materialize another legal command',
    async () => {
      const priorAttempts = process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
      process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = '1';
      const session = new RoyLHTBSession('finalize-fallback', 'task', 'finish', 'commit',
        'learned_information_realization');
      const provider = { isConfigured: () => true, async completeJSONWithUsage() {
        const candidate = { id: 'derive', kind: 'DERIVE', actorNodeId: 'root',
          description: 'invalid structural readout', schedulerComplexity: 1,
          action: { kind: 'DERIVE', actorNodeId: 'root' } };
        return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
          completion: { content: '{}', model: 'mock', usage: {
            promptTokens: 1, completionTokens: 1, totalTokens: 2,
          } } };
      } };
      const semantic = { async processEvent(event: { id: string }) {
        return { event_id: event.id, requirements: [], claims: [], assumptions: [], evidence: [],
          external_observations: [], blind_spots: [], relations: [] };
      }, close() {} };
      const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
      try {
        const result = await controller.advanceFinalizeNow(session);
        expect(result.status).toBe('completed');
        expect(session.snapshot().policyRecords).toHaveLength(0);
      } finally {
        controller.close();
        if (priorAttempts === undefined) delete process.env.ROY_LHTB_PROPOSAL_ATTEMPTS;
        else process.env.ROY_LHTB_PROPOSAL_ATTEMPTS = priorAttempts;
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

  it('repairs a DERIVE proposal with the authoritative gap-owner interface', async () => {
    const session = new RoyLHTBSession('gap-owner-repair', 'task',
      'inspect two independent failures', 'commit', 'roy_runtime_heuristic');
    session.applySemanticUpdate({ event_id: 'task-instruction', requirements: [{
      id: 'child-gap', description: 'inspect the first failure',
      requiredInformation: 'first diagnosis', likelyMechanism: 'acquisition',
    }, {
      id: 'root-gap', description: 'inspect the independent second failure',
      requiredInformation: 'second diagnosis', likelyMechanism: 'acquisition',
    }], claims: [], assumptions: [], evidence: [], external_observations: [], blind_spots: [],
    relations: [] });
    const firstChild = {
      id: 'spec-worker', nodeId: 'worker', parentId: 'root', depth: 1,
      parentGoal: 'inspect two independent failures', triggeringGapId: 'child-gap',
      realizationMode: 'acquire_external' as const,
      localObjective: 'Inspect only the first failure and report its diagnosis.',
      refinement: { parentScope: 'inspect two failures', childScope: 'inspect first failure',
        triggeringRequirementId: 'child-gap', narrowerThanParent: true,
        newInformationNeeded: 'first failure output',
        executableEndCondition: 'first diagnosis is recorded', duplicatedByExistingNode: false },
      requiredClaims: [], requiredEvidence: [], relevantReportIds: [],
      externalAccess: { allowed: true, tools: ['terminal'], purpose: 'inspect first failure' },
      expectedOutput: { requiredInformation: 'first diagnosis',
        outputType: 'epistemic_report' as const },
      terminationCondition: 'return the first diagnosis',
    };
    session.applyOrganizationAction({ kind: 'DERIVE', actorNodeId: 'root',
      childSpecification: firstChild });
    session.requestTerminal({ id: 'worker-inspection', command: 'true', timeoutMs: 1000,
      nodeId: 'worker', organizationActionKind: 'ACQUIRE' });
    session.acceptTerminalResult({ requestId: 'worker-inspection', exitCode: 0,
      stdout: 'first diagnosis', stderr: '', durationMs: 1 });
    expect(scheduledOrganizationContextNode(session.snapshot())).toBe('root');
    const wrongOwnerChild = {
      ...firstChild, id: 'spec-wrong-owner', nodeId: 'wrong-owner', parentId: 'worker', depth: 2,
      parentGoal: firstChild.localObjective, triggeringGapId: 'root-gap',
      localObjective: 'Inspect the independent second failure.',
      refinement: { ...firstChild.refinement, parentScope: firstChild.localObjective,
        childScope: 'inspect second failure', triggeringRequirementId: 'root-gap',
        newInformationNeeded: 'second failure output' },
      expectedOutput: { requiredInformation: 'second diagnosis',
        outputType: 'epistemic_report' as const },
      reuseReview: { searchedNodeIds: ['worker'], decision: 'spawn_distinct',
        reason: 'the proposed worker is incorrectly assumed to own the second gap' },
    };
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
        id: 'wrong-owner', kind: 'DERIVE', actorNodeId: 'worker',
        description: 'Incorrectly assign a root gap to the child', schedulerComplexity: 2,
        action: { kind: 'DERIVE', actorNodeId: 'worker',
          childSpecification: wrongOwnerChild },
      } : {
        id: 'inspect-root-gap', kind: 'ACQUIRE', actorNodeId: 'root',
        description: 'Inspect the second failure at its owning root', schedulerComplexity: 1,
        command: 'printf second-failure', action: { kind: 'ACQUIRE', actorNodeId: 'root' },
      };
      return { value: { preferred_candidate_id: candidate.id, candidates: [candidate] },
        completion: { content: JSON.stringify({ candidates: [candidate] }), model: 'mock',
          usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } } };
    } };
    const controller = new LHTBAutonomousController({ provider, semantic, auditRoot: false });
    try {
      const result = await controller.advance(session, 1);
      expect(result.status).toBe('terminal_request');
      expect(calls).toBe(2);
      const repair = JSON.parse(requests[1]?.at(-1)?.content ?? '{}') as {
        repairProtocol: string;
        instruction: string[];
        legalInterface: { realOpenGaps: Array<{ id: string; parentNodeId: string }> };
        rejectionReasons: string[];
      };
      expect(repair.repairProtocol).toBe('legal-candidate-interface-v3');
      expect(repair.legalInterface.realOpenGaps).toContainEqual({
        id: 'root-gap', parentNodeId: 'root', description: 'inspect the independent second failure',
        requiredInformation: 'second diagnosis',
      });
      expect(repair.instruction.join(' ')).toContain('actorNodeId');
      expect(repair.rejectionReasons.some(reason =>
        reason.startsWith('derive_requirement_owner_mismatch:root-gap:expected=root'))).toBe(true);
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
