import type { RuntimeMessage } from '../queue/types.js';
import type { ToMProfile } from '../tom/types.js';

export type CommunicationProtocolId = string;
export type TraceActorType = 'user' | 'agent' | 'team' | 'runtime' | 'tool' | 'adapter';
export type TraceVisibility = 'participants' | 'parent_chain' | 'team' | 'public';

export interface TraceActorRef {
  id: string;
  type: TraceActorType;
  name?: string;
  parentId?: string;
  teamId?: string;
}

/**
 * Provider-neutral trace record supplied by the runtime to communication-aware actors.
 * It stores observable messages and state transitions, never hidden chain-of-thought.
 */
export interface MultiPartyTrace {
  id: string;
  sessionId: string;
  timestamp: number;
  protocolId: CommunicationProtocolId;
  kind: string;
  phase: 'enqueued' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'system';
  from: TraceActorRef;
  to: TraceActorRef[];
  content?: string;
  turnId?: string;
  traceId?: string;
  correlationId?: string;
  parentTraceId?: string;
  visibility: TraceVisibility;
  metadata?: Record<string, unknown>;
}

export interface CommunicationParticipant {
  actor: TraceActorRef;
  tomProfile?: ToMProfile;
}

export interface CommunicationProtocolInput {
  message: RuntimeMessage;
  recipient: CommunicationParticipant;
  participants: CommunicationParticipant[];
  traces: MultiPartyTrace[];
  task?: string;
}

export interface AgentCommunicationContext {
  protocolId: CommunicationProtocolId;
  protocolVersion: string;
  messageId: string;
  correlationId?: string;
  rendered: string;
  traces: MultiPartyTrace[];
  metadata?: Record<string, unknown>;
}

/** A protocol controls how runtime messages and observable traces enter an agent context. */
export interface AgentCommunicationProtocol {
  readonly id: CommunicationProtocolId;
  readonly version: string;
  readonly description: string;
  render(input: CommunicationProtocolInput): AgentCommunicationContext;
}

export interface CommunicationRuntimeConfig {
  defaultProtocol: CommunicationProtocolId;
  allowMessageOverride: boolean;
  traceWindowSize: number;
  includeCompletedMessages: boolean;
}

export interface CommunicationState {
  defaultProtocol: CommunicationProtocolId;
  registeredProtocols: Array<{
    id: CommunicationProtocolId;
    version: string;
    description: string;
  }>;
  traces: number;
  tracesByProtocol: Record<string, number>;
}

export interface AgentTraceReceiver {
  receiveSystemTrace(trace: MultiPartyTrace): void;
  receiveSystemTraces(traces: MultiPartyTrace[]): void;
  getSystemTraces(options?: { correlationId?: string; limit?: number }): MultiPartyTrace[];
  receiveCommunicationContext(context: AgentCommunicationContext): void;
  getCommunicationContext(): AgentCommunicationContext | undefined;
}
