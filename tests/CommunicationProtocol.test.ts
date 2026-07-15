import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { BaseAgent } from '../src/core/agent/BaseAgent.js';
import {
  AgentCommunicationManager,
  StructuredCommunicationProtocol,
  type AgentCommunicationProtocol,
  type CommunicationProtocolInput,
  type MultiPartyTrace,
} from '../src/core/communication/index.js';
import type { RuntimeMessage } from '../src/core/queue/index.js';
import { Runtime } from '../src/core/runtime/Runtime.js';

class TraceAgent extends BaseAgent {
  async step(_observation: string): Promise<void> {}
  async run(): Promise<void> {}
}

const message = (protocol = 'tom'): RuntimeMessage => ({
  id: 'msg_communication_1',
  kind: 'agent.task',
  sessionId: 'communication-session',
  correlationId: 'communication-correlation',
  from: 'root',
  to: 'agent_researcher_001',
  priority: 'normal',
  status: 'processing',
  createdAt: 1,
  updatedAt: 2,
  payload: { task: 'Inspect the project structure' },
  metadata: { communicationProtocol: protocol },
});

describe('agent communication protocols', () => {
  const runtimes: Runtime[] = [];

  afterEach(async () => {
    await Promise.all(runtimes.splice(0).map(runtime => runtime.shutdown()));
  });

  it('renders a simple structured message without requiring ToM interpretation', () => {
    const protocol = new StructuredCommunicationProtocol();
    const rendered = protocol.render({
      message: message('structured'),
      recipient: { actor: { id: 'agent_researcher_001', type: 'agent' } },
      participants: [],
      traces: [],
      task: 'Inspect the project structure',
    });

    expect(rendered.protocolId).toBe('structured');
    expect(rendered.rendered).toContain('kind: agent.task');
    expect(rendered.rendered).toContain('from: root');
    expect(rendered.rendered).not.toContain('<recipient_model>');
  });

  it('uses ToM by default and derives traces from message transitions', () => {
    const manager = new AgentCommunicationManager();
    const trace = manager.recordTransition({ type: 'message.processing', message: message() });

    expect(manager.getDefaultProtocolId()).toBe('tom');
    expect(trace).toMatchObject({
      protocolId: 'tom',
      kind: 'agent.task',
      phase: 'processing',
      correlationId: 'communication-correlation',
    });
    expect(manager.getState().tracesByProtocol.tom).toBe(1);
  });

  it('supports custom protocol registration and selection', () => {
    const extension: AgentCommunicationProtocol = {
      id: 'compact-json',
      version: '1.0',
      description: 'Test extension',
      render(input: CommunicationProtocolInput) {
        return {
          protocolId: this.id,
          protocolVersion: this.version,
          messageId: input.message.id,
          correlationId: input.message.correlationId,
          rendered: JSON.stringify({ kind: input.message.kind, traceCount: input.traces.length }),
          traces: input.traces,
        };
      },
    };
    const manager = new AgentCommunicationManager();
    manager.registerProtocol(extension);
    manager.setDefaultProtocol('compact-json');

    expect(manager.getState().defaultProtocol).toBe('compact-json');
    expect(manager.registry.get('compact-json')).toBe(extension);
  });

  it('gives every BaseAgent a bounded multi-party trace receiver interface', () => {
    const agent = new TraceAgent({ name: 'TraceAgent', role: 'subagent' });
    const trace: MultiPartyTrace = {
      id: 'trace_external_1',
      sessionId: 'communication-session',
      timestamp: Date.now(),
      protocolId: 'structured',
      kind: 'system.observation',
      phase: 'system',
      from: { id: 'runtime', type: 'runtime' },
      to: [{ id: agent.id, type: 'agent' }],
      correlationId: 'external-correlation',
      visibility: 'participants',
      content: 'A public test observation.',
    };

    agent.receiveSystemTrace(trace);
    expect(agent.getSystemTraces({ correlationId: 'external-correlation' })).toHaveLength(1);
    expect(agent.getIdentity().communicationProtocol).toBe('tom');
    agent.setCommunicationProtocol('structured');
    expect(agent.getIdentity().communicationProtocol).toBe('structured');
  });

  it('switches a live runtime from ToM to structured communication', async () => {
    const cwd = await mkdtemp(path.join(tmpdir(), 'roy-communication-runtime-'));
    const runtime = new Runtime();
    runtimes.push(runtime);
    await runtime.initialize({ workspaceCwd: cwd, sessionId: 'communication-live' });

    await runtime.handleUserTurn('Who are you?');
    expect(runtime.getContext().agent.getCommunicationContext()?.protocolId).toBe('tom');
    expect(runtime.getCommunicationTraces({ agentId: 'root' }).length).toBeGreaterThan(0);

    runtime.setDefaultCommunicationProtocol('structured');
    await runtime.handleUserTurn('What is 2 + 2?');
    expect(runtime.getContext().agent.getCommunicationContext()?.protocolId).toBe('structured');
    expect(runtime.getCommunicationState().registeredProtocols.map(item => item.id)).toEqual(
      expect.arrayContaining(['tom', 'structured'])
    );
  });
});
