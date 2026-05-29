export type MessageKind =
  | 'user.input'
  | 'user.command.spawn'
  | 'agent.task'
  | 'agent.result'
  | 'agent.error'
  | 'agent.control'
  | 'root.synthesis'
  | 'root.final_response'
  | 'team.task'
  | 'team.result'
  | 'tool.call'
  | 'tool.result'
  | 'budget.request'
  | 'budget.grant'
  | 'memory.load'
  | 'memory.update'
  | 'evo.propose'
  | 'evo.evaluate'
  | 'evo.select';

export type MessagePriority = 'low' | 'normal' | 'high' | 'critical';
export type MessageStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';

export interface RuntimeMessage<TPayload = unknown> {
  id: string;
  kind: MessageKind;
  sessionId: string;
  turnId?: string;
  traceId?: string;
  from: string;
  to: string;
  parentMessageId?: string;
  correlationId?: string;
  priority: MessagePriority;
  status: MessageStatus;
  createdAt: number;
  updatedAt: number;
  availableAt?: number;
  expiresAt?: number;
  payload: TPayload;
  error?: string;
  metadata?: {
    agentId?: string;
    teamId?: string;
    tomLevel?: number;
    budgetTokens?: number;
    retryCount?: number;
    maxRetries?: number;
    tags?: string[];
  };
}

export interface EnqueueMessageInput<TPayload = unknown> {
  kind: MessageKind;
  sessionId: string;
  from: string;
  to: string;
  payload: TPayload;
  turnId?: string;
  traceId?: string;
  parentMessageId?: string;
  correlationId?: string;
  priority?: MessagePriority;
  availableAt?: number;
  expiresAt?: number;
  metadata?: RuntimeMessage<TPayload>['metadata'];
}

export interface DequeueOptions {
  to?: string;
  kind?: MessageKind[];
  readyOnly?: boolean;
}

export interface MessageFilter {
  status?: MessageStatus;
  kind?: MessageKind;
  to?: string;
  from?: string;
  limit?: number;
}

export interface QueueStats {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  cancelled: number;
  total: number;
}

export interface QueueState {
  stats: QueueStats;
  recent: RuntimeMessage[];
}

export type QueueTransitionType =
  | 'message.enqueued'
  | 'message.processing'
  | 'message.completed'
  | 'message.failed'
  | 'message.cancelled'
  | 'message.expired';

export interface QueueTransition {
  type: QueueTransitionType;
  message: RuntimeMessage;
  error?: string;
  reason?: string;
}

export interface MessageQueue {
  enqueue<TPayload>(message: EnqueueMessageInput<TPayload>): Promise<RuntimeMessage<TPayload>>;
  dequeue(options?: DequeueOptions): Promise<RuntimeMessage | undefined>;
  ack(messageId: string): Promise<void>;
  fail(messageId: string, error: Error): Promise<void>;
  cancel(messageId: string, reason?: string): Promise<void>;
  getMessage(messageId: string): Promise<RuntimeMessage | undefined>;
  listMessages(filter?: MessageFilter): Promise<RuntimeMessage[]>;
  getStats(): Promise<QueueStats>;
}
