import type { AgentInfo } from '../agent/BaseAgent.js';
import type { QueueTransition, RuntimeMessage } from '../queue/types.js';
import { StructuredCommunicationProtocol, ToMCommunicationProtocol } from './protocols.js';
import type {
  AgentCommunicationContext,
  AgentCommunicationProtocol,
  CommunicationParticipant,
  CommunicationRuntimeConfig,
  CommunicationState,
  MultiPartyTrace,
  TraceActorRef,
  TraceActorType,
} from './types.js';

const DEFAULT_CONFIG: CommunicationRuntimeConfig = {
  defaultProtocol: 'tom',
  allowMessageOverride: true,
  traceWindowSize: 200,
  includeCompletedMessages: true,
};

export class CommunicationProtocolRegistry {
  private readonly protocols = new Map<string, AgentCommunicationProtocol>();

  register(protocol: AgentCommunicationProtocol): void {
    if (!protocol.id.trim()) throw new Error('Communication protocol id must not be empty');
    this.protocols.set(protocol.id, protocol);
  }

  unregister(id: string): boolean {
    return this.protocols.delete(id);
  }

  get(id: string): AgentCommunicationProtocol | undefined {
    return this.protocols.get(id);
  }

  list(): AgentCommunicationProtocol[] {
    return [...this.protocols.values()];
  }
}

export class MultiPartyTraceStore {
  private traces: MultiPartyTrace[] = [];

  constructor(private readonly maxEntries = 200) {}

  append(trace: MultiPartyTrace): void {
    this.traces.push(trace);
    if (this.traces.length > this.maxEntries) this.traces = this.traces.slice(-this.maxEntries);
  }

  list(options: {
    sessionId?: string;
    correlationId?: string;
    actorId?: string;
    limit?: number;
    visibleTo?: TraceActorRef;
  } = {}): MultiPartyTrace[] {
    const filtered = this.traces.filter(trace => {
      if (options.sessionId && trace.sessionId !== options.sessionId) return false;
      if (options.correlationId && trace.correlationId !== options.correlationId) return false;
      if (options.actorId && trace.from.id !== options.actorId && !trace.to.some(actor => actor.id === options.actorId)) return false;
      if (options.visibleTo && !this.isVisibleTo(trace, options.visibleTo)) return false;
      return true;
    });
    return filtered.slice(-Math.max(1, options.limit ?? this.maxEntries)).map(trace => ({
      ...trace,
      from: { ...trace.from },
      to: trace.to.map(actor => ({ ...actor })),
      metadata: trace.metadata ? { ...trace.metadata } : undefined,
    }));
  }

  clear(): void {
    this.traces = [];
  }

  size(): number {
    return this.traces.length;
  }

  private isVisibleTo(trace: MultiPartyTrace, actor: TraceActorRef): boolean {
    if (trace.visibility === 'public') return true;
    if (trace.from.id === actor.id || trace.to.some(recipient => recipient.id === actor.id)) return true;
    if (trace.visibility === 'team') {
      const teamIds = new Set([trace.from.teamId, ...trace.to.map(recipient => recipient.teamId)].filter(Boolean));
      return Boolean(actor.teamId && teamIds.has(actor.teamId));
    }
    if (trace.visibility === 'parent_chain') {
      return trace.from.parentId === actor.id
        || trace.to.some(recipient => recipient.parentId === actor.id)
        || actor.parentId === trace.from.id
        || trace.to.some(recipient => recipient.id === actor.parentId);
    }
    return false;
  }
}

export class AgentCommunicationManager {
  readonly registry = new CommunicationProtocolRegistry();
  readonly traces: MultiPartyTraceStore;
  private config: CommunicationRuntimeConfig;

  constructor(
    config: Partial<CommunicationRuntimeConfig> = {},
    extensions: AgentCommunicationProtocol[] = []
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.traces = new MultiPartyTraceStore(this.config.traceWindowSize);
    this.registry.register(new StructuredCommunicationProtocol());
    this.registry.register(new ToMCommunicationProtocol());
    for (const extension of extensions) this.registry.register(extension);
    this.assertProtocol(this.config.defaultProtocol);
  }

  registerProtocol(protocol: AgentCommunicationProtocol): void {
    this.registry.register(protocol);
  }

  setDefaultProtocol(id: string): void {
    this.assertProtocol(id);
    this.config.defaultProtocol = id;
  }

  getDefaultProtocolId(): string {
    return this.config.defaultProtocol;
  }

  resolveProtocol(message?: RuntimeMessage, agentProtocol?: string): AgentCommunicationProtocol {
    const requested = this.config.allowMessageOverride
      ? message?.metadata?.communicationProtocol ?? agentProtocol
      : agentProtocol;
    const id = requested ?? this.config.defaultProtocol;
    return this.registry.get(id) ?? this.assertProtocol(this.config.defaultProtocol);
  }

  recordTransition(transition: QueueTransition): MultiPartyTrace | undefined {
    if (!this.config.includeCompletedMessages && transition.type === 'message.completed') return undefined;
    const message = transition.message;
    const phase = transition.type.replace('message.', '') as MultiPartyTrace['phase'];
    if (!['enqueued', 'processing', 'completed', 'failed', 'cancelled'].includes(phase)) return undefined;
    const protocolId = this.resolveProtocol(message).id;
    const trace: MultiPartyTrace = {
      id: `${message.id}:${phase}:${message.updatedAt}`,
      sessionId: message.sessionId,
      timestamp: message.updatedAt,
      protocolId,
      kind: message.kind,
      phase,
      from: this.actorRef(message.from, message.metadata),
      to: [this.actorRef(message.to, message.metadata)],
      content: this.summarizePayload(message.payload),
      turnId: message.turnId,
      traceId: message.traceId,
      correlationId: message.correlationId,
      parentTraceId: message.parentMessageId,
      visibility: message.metadata?.traceVisibility ?? 'participants',
      metadata: {
        messageId: message.id,
        status: message.status,
        nodeId: message.metadata?.nodeId,
        teamId: message.metadata?.teamId,
        error: transition.error,
        reason: transition.reason,
      },
    };
    this.traces.append(trace);
    return trace;
  }

  buildContext(input: {
    message: RuntimeMessage;
    recipient: AgentInfo;
    participants: AgentInfo[];
    task?: string;
    protocolId?: string;
  }): AgentCommunicationContext {
    const protocol = input.protocolId
      ? this.registry.get(input.protocolId) ?? this.resolveProtocol(input.message, input.recipient.identity.communicationProtocol)
      : this.resolveProtocol(input.message, input.recipient.identity.communicationProtocol);
    const participants = input.participants.map(agent => this.participant(agent));
    const traces = this.traces.list({
      sessionId: input.message.sessionId,
      correlationId: input.message.correlationId,
      limit: this.config.traceWindowSize,
      visibleTo: this.participant(input.recipient).actor,
    });
    return protocol.render({
      message: input.message,
      recipient: this.participant(input.recipient),
      participants,
      traces,
      task: input.task,
    });
  }

  getState(): CommunicationState {
    const traces = this.traces.list();
    return {
      defaultProtocol: this.config.defaultProtocol,
      registeredProtocols: this.registry.list().map(protocol => ({
        id: protocol.id,
        version: protocol.version,
        description: protocol.description,
      })),
      traces: traces.length,
      tracesByProtocol: traces.reduce<Record<string, number>>((counts, trace) => {
        counts[trace.protocolId] = (counts[trace.protocolId] ?? 0) + 1;
        return counts;
      }, {}),
    };
  }

  private participant(agent: AgentInfo): CommunicationParticipant {
    return {
      actor: {
        id: agent.identity.id,
        type: agent.identity.role === 'subteam' ? 'team' : 'agent',
        name: agent.identity.name,
        parentId: agent.identity.parentId,
        teamId: agent.identity.teamId,
      },
      tomProfile: agent.identity.tomProfile,
    };
  }

  private actorRef(id: string, metadata?: RuntimeMessage['metadata']): TraceActorRef {
    return {
      id,
      type: this.actorType(id),
      teamId: metadata?.teamId,
    };
  }

  private actorType(id: string): TraceActorType {
    if (id === 'user') return 'user';
    if (id === 'runtime') return 'runtime';
    if (id === 'cli' || id === 'server') return 'adapter';
    if (id.startsWith('team_')) return 'team';
    if (id.startsWith('tool.') || id.startsWith('fs.') || id.startsWith('shell.')) return 'tool';
    return 'agent';
  }

  private summarizePayload(payload: unknown): string | undefined {
    if (payload === undefined || payload === null) return undefined;
    const value = typeof payload === 'string' ? payload : JSON.stringify(payload);
    return value.length <= 500 ? value : `${value.slice(0, 476)}...[trace truncated]`;
  }

  private assertProtocol(id: string): AgentCommunicationProtocol {
    const protocol = this.registry.get(id);
    if (!protocol) throw new Error(`Communication protocol "${id}" is not registered`);
    return protocol;
  }
}

export { DEFAULT_CONFIG as DEFAULT_COMMUNICATION_CONFIG };
