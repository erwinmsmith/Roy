import { describe, expect, it } from 'vitest';
import { GlobalEpistemicStateRecorder, LHTBAutonomousController,
  RoyLHTBSession } from '../src/core/structural/index.js';

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
    expect(() => direct.applyOrganizationAction({ kind: 'CONNECT', actorNodeId: 'root',
      connection: { from: 'root', to: 'root', required: false } })).toThrow(/single_agent_direct/);
  });

  it('reopens the same session after an official verifier rejection', () => {
    const session = new RoyLHTBSession('resume', 'task', 'solve it', 'commit');
    session.applyOrganizationAction({ kind: 'STOP', actorNodeId: 'root', finalOutput: 'draft' });
    expect(session.snapshot().runtime.stopped).toBe(true);
    session.resumeAfterVerifierRejection();
    expect(session.snapshot().runtime.stopped).toBe(false);
    expect(session.snapshot().processStates.at(-1)?.runtimeEvents.at(-1)?.kind).toBe('verifier');
  });

  it('integrates mock DeepSeek semantics before the next organization decision', async () => {
    const session = new RoyLHTBSession('mock', 'task', 'solve it', 'commit',
      'roy_runtime_heuristic');
    session.requestTerminal({ id: 'one', command: 'pwd', timeoutMs: 1000, nodeId: 'root' });
    session.acceptTerminalResult({ requestId: 'one', exitCode: 0, stdout: '/workspace',
      stderr: '', durationMs: 1 });
    const semantic = {
      async processEvent(event: { id: string }) {
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
    expect(session.snapshot().processStates.at(-1)?.claims.length).toBe(2);
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
});
