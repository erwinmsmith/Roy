#!/usr/bin/env node

import { createInterface } from 'node:readline';
import type { OrganizationAction } from '../core/structural/informationRealizationTypes.js';
import { RoyLHTBSession, type LHTBSessionSnapshot, type TerminalRequest,
  type TerminalResult } from '../core/structural/lhtbSession.js';
import { LHTBAutonomousController } from '../core/structural/lhtbController.js';

interface Request { id: string | number; method: string; params?: Record<string, unknown> }
let session: RoyLHTBSession | undefined;
let controller: LHTBAutonomousController | undefined;
let organizationSeed = 20260820;

async function dispatch(request: Request): Promise<unknown> {
  const params = request.params ?? {};
  if (request.method === 'run' || request.method === 'initialize') {
    session = new RoyLHTBSession(String(params.trajectoryId), String(params.taskId),
      String(params.instruction), String(params.environmentRevision ?? 'lhtb-pinned'),
      (params.organizationMode ?? 'learned_information_realization') as LHTBSessionSnapshot['organizationMode'],
      String(params.initialSnapshotFingerprint ?? ''),
      Number(params.organizationSeed ?? 20260820));
    organizationSeed = session.organizationSeed;
    if (request.method === 'initialize') {
      return { status: 'ready', snapshot: session.snapshot() };
    }
    controller ??= new LHTBAutonomousController();
    return controller.advance(session, organizationSeed++);
  }
  if (request.method === 'restore') {
    session = RoyLHTBSession.restore(params.snapshot as unknown as LHTBSessionSnapshot);
    organizationSeed = session.organizationSeed;
    return { status: 'restored', snapshot: session.snapshot() };
  }
  if (!session) throw new Error('run or restore must initialize the LHTB session');
  if (request.method === 'organization_action') {
    const state = session.applyOrganizationAction(params.action as unknown as OrganizationAction);
    return { status: 'ready', state, snapshot: session.snapshot() };
  }
  if (request.method === 'terminal_request') {
    const state = session.requestTerminal(params.request as unknown as TerminalRequest);
    return { status: 'terminal_request', request: params.request, state,
      snapshot: session.snapshot() };
  }
  if (request.method === 'resume') {
    const before = session.snapshot().pendingTerminalRequest;
    const result = params.result as unknown as TerminalResult;
    session.acceptTerminalResult(result);
    if (!before) throw new Error('resume has no pending terminal request');
    if (before.organizationActionKind === 'EXECUTE') {
      session.applyOrganizationAction({ kind: 'EXECUTE', actorNodeId: before.nodeId });
    } else {
      session.applyOrganizationAction({ kind: 'ACQUIRE', actorNodeId: before.nodeId,
        observation: { id: `observation-${before.id}`, sourceType: 'environment',
          queryOrAction: before.command,
          observation: `${result.stdout}${result.stderr}`,
          provenance: `harbor:${before.id}:exit-${result.exitCode}`, supports: [] } });
    }
    controller ??= new LHTBAutonomousController();
    return controller.advance(session, organizationSeed++);
  }
  if (request.method === 'resume_boundary') {
    const before = session.snapshot().pendingTerminalRequest;
    const result = params.result as unknown as TerminalResult;
    session.acceptTerminalResult(result);
    if (!before) throw new Error('resume_boundary has no pending terminal request');
    if (before.organizationActionKind === 'EXECUTE') {
      session.applyOrganizationAction({ kind: 'EXECUTE', actorNodeId: before.nodeId });
    } else {
      session.applyOrganizationAction({ kind: 'ACQUIRE', actorNodeId: before.nodeId,
        observation: { id: `observation-${before.id}`, sourceType: 'environment',
          queryOrAction: before.command,
          observation: `${result.stdout}${result.stderr}`,
          provenance: `harbor:${before.id}:exit-${result.exitCode}`, supports: [] } });
    }
    return { status: 'ready', snapshot: session.snapshot() };
  }
  if (request.method === 'advance' || request.method === 'advance_one') {
    controller ??= new LHTBAutonomousController();
    return controller.advance(session, organizationSeed++);
  }
  if (request.method === 'verifier_rejection') {
    session.resumeAfterVerifierRejection(String(params.feedback ?? 'Verifier rejected completion'));
    controller ??= new LHTBAutonomousController();
    return controller.advance(session, organizationSeed++);
  }
  if (request.method === 'rollout_deadline') {
    session.finalizeAtRolloutDeadline(String(params.reason ?? 'training_rollout_deadline'));
    return { status: 'completed', terminationReason: 'rollout_deadline',
      snapshot: session.snapshot() };
  }
  if (request.method === 'snapshot') return session.snapshot();
  if (request.method === 'shutdown') {
    controller?.close();
    return { status: 'shutdown' };
  }
  throw new Error(`Unknown Roy LHTB JSON-RPC method ${request.method}`);
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  let request: Request | undefined;
  try {
    request = JSON.parse(line) as Request;
    const result = await dispatch(request);
    process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id: request.id, result })}\n`);
    if (request.method === 'shutdown') break;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id: request?.id ?? null,
      error: { code: -32000, message: error instanceof Error ? error.message : String(error) } })}\n`);
  }
}
