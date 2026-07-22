import { createHash } from 'node:crypto';
import type { RuntimeMessage } from '../queue/index.js';
import type {
  RootExecutionActivity,
  RootExecutionActivityKind,
  RootExecutionCheckpoint,
  RootExecutionStep,
  RootExecutionTreeState,
} from './executionTree.js';

export interface ExecutionObservableEvent {
  type: string;
  timestamp: number;
  agentId?: string;
  correlationId?: string;
  data?: Record<string, unknown>;
}

export interface ProjectRootStepActivitiesInput {
  tree: RootExecutionTreeState;
  step: RootExecutionStep;
  messages: RuntimeMessage[];
  events: ExecutionObservableEvent[];
  now?: number;
}

export interface BuildRootCheckpointInput {
  tree: RootExecutionTreeState;
  step: RootExecutionStep;
  resultSummary?: string;
  activities: RootExecutionActivity[];
  actorIds?: string[];
  teamIds?: string[];
}

export class RootExecutionActivityProjector {
  project(input: ProjectRootStepActivitiesInput): RootExecutionActivity[] {
    const { tree, step } = input;
    const now = input.now ?? Date.now();
    const activities: RootExecutionActivity[] = [];
    const push = (activity: Omit<RootExecutionActivity, 'id'>): string => {
      const id = `${step.id}.activity_${String(activities.length + 1).padStart(3, '0')}`;
      activities.push({ ...activity, id });
      return id;
    };

    if (step.index === 1) {
      push({
        kind: 'conversation',
        status: 'completed',
        label: 'User input received',
        actorId: tree.rootAgentId,
        summary: tree.task,
        startedAt: tree.createdAt,
        completedAt: step.startedAt,
      });
    }
    push({
      kind: 'thinking',
      status: 'completed',
      label: `Root step decision: ${step.decision.action}`,
      actorId: tree.rootAgentId,
      summary: step.decision.reason,
      startedAt: step.startedAt,
      completedAt: now,
      data: { decision: step.decision },
    });

    const activityIdByMessage = new Map<string, string>();
    for (const message of input.messages
      .filter(item => item.kind !== 'root.step.plan' && item.kind !== 'root.step.result')
      .filter(item => step.index === 1
        ? item.createdAt <= now
        : item.createdAt >= step.startedAt && item.createdAt <= now)) {
      const kind = this.kindForMessage(message.kind);
      if (!kind) continue;
      const id = push({
        kind,
        status: message.status === 'failed' || message.status === 'cancelled'
          ? 'failed'
          : message.status === 'processing' || message.status === 'pending' ? 'running' : 'completed',
        label: `${message.kind}: ${message.from} -> ${message.to}`,
        actorId: message.metadata?.agentId ?? message.from,
        parentActivityId: message.parentMessageId
          ? activityIdByMessage.get(message.parentMessageId)
          : undefined,
        messageId: message.id,
        summary: this.summarizeMessage(message),
        startedAt: message.createdAt,
        completedAt: ['completed', 'failed', 'cancelled'].includes(message.status) ? message.updatedAt : undefined,
        data: {
          kind: message.kind,
          from: message.from,
          to: message.to,
          nodeId: message.metadata?.nodeId,
          teamId: message.metadata?.teamId,
        },
      });
      activityIdByMessage.set(message.id, id);
    }

    for (const event of input.events
      .filter(item => item.correlationId === tree.correlationId || item.data?.correlationId === tree.correlationId)
      .filter(item => item.timestamp >= step.startedAt && item.timestamp <= now)
      .filter(item => !item.type.startsWith('message.') && !item.type.startsWith('root.step.'))) {
      const kind = this.kindForEvent(event.type);
      if (!kind) continue;
      const failed = event.type.includes('failed') || event.type.includes('error') || event.type.includes('rejected');
      push({
        kind,
        status: failed ? 'failed' : 'completed',
        label: event.type,
        actorId: event.agentId,
        eventType: event.type,
        summary: this.summarizeEvent(event),
        tokenUsage: finiteNumber(event.data?.totalTokens),
        startedAt: event.timestamp,
        completedAt: event.timestamp,
        data: this.compactData(event.data),
      });
    }
    return activities.sort((left, right) => left.startedAt - right.startedAt || left.id.localeCompare(right.id));
  }

  checkpoint(input: BuildRootCheckpointInput): RootExecutionCheckpoint {
    const successfulToolResults = input.activities.filter(activity => this.isSuccessfulToolResult(activity));
    const messageToolResults = successfulToolResults.filter(activity => activity.data?.kind === 'tool.result');
    const evidence = (messageToolResults.length > 0 ? messageToolResults : successfulToolResults)
      .map(activity => activity.summary ?? activity.label)
      .filter((value, index, values) => values.indexOf(value) === index)
      .slice(0, 20);
    const completed = [...new Set([
      ...input.activities
        .filter(activity => ['agent', 'team', 'synthesis'].includes(activity.kind) && activity.status === 'completed')
        .map(activity => activity.label),
      ...(input.resultSummary ? [input.resultSummary.replace(/\s+/g, ' ').slice(0, 500)] : []),
    ])].slice(0, 30);
    const pending = input.step.decision.action === 'finalize'
      ? []
      : input.step.decision.action === 'ask_clarification'
        ? ['Await user clarification before resuming the task loop.']
        : ['Root must reassess accumulated state and decide whether to delegate, use tools, or finalize.'];
    const stateFingerprint = createHash('sha256').update(JSON.stringify({
      actorIds: input.actorIds ?? input.step.actorIds,
      teamIds: input.teamIds ?? input.step.teamIds,
      completed,
      evidence,
      resultSummary: input.resultSummary,
    })).digest('hex');
    return {
      objective: input.tree.task,
      completed,
      pending,
      evidence,
      decisionBasis: input.step.decision.reason,
      stateFingerprint,
      createdAt: Date.now(),
    };
  }

  private isSuccessfulToolResult(activity: RootExecutionActivity): boolean {
    if (activity.kind !== 'tool' || activity.status !== 'completed' || activity.data?.success === false) return false;
    return activity.eventType === 'tool.result' || activity.data?.kind === 'tool.result';
  }

  private kindForMessage(kind: RuntimeMessage['kind']): RootExecutionActivityKind | undefined {
    if (kind === 'user.input' || kind === 'root.final_response') return 'conversation';
    if (kind === 'tool.call' || kind === 'tool.result') return 'tool';
    if (kind === 'agent.task' || kind === 'agent.result' || kind === 'agent.synthesis') return 'agent';
    if (kind === 'team.task' || kind === 'team.result') return 'team';
    if (kind === 'root.synthesis') return 'synthesis';
    if (kind.startsWith('agent.create') || kind.startsWith('team.create') || kind.startsWith('evo.')) return 'delegation';
    return undefined;
  }

  private kindForEvent(type: string): RootExecutionActivityKind | undefined {
    if (type === 'context.loaded' || type === 'team.context.loaded' || type.startsWith('memory.load')) return 'context';
    if (type === 'agent.llm.called' || type === 'delegation.decision' || type.startsWith('tom.')) return 'thinking';
    if (type.startsWith('tool.') || type.includes('.tool_loop')) return 'tool';
    if (type.startsWith('delegation.') || type.startsWith('spawn.') || type.startsWith('agent.create')) return 'delegation';
    if (type.startsWith('team.')) return 'team';
    if (type.startsWith('root.synthesis')) return 'synthesis';
    if (type.startsWith('agent.')) return 'agent';
    if (type.startsWith('fsm.') || type.startsWith('turn.fsm.')) return 'control';
    return undefined;
  }

  private summarizeMessage(message: RuntimeMessage): string {
    if (message.kind === 'user.input') {
      const value = (message.payload as { input?: unknown })?.input;
      return typeof value === 'string' ? value.slice(0, 500) : 'User input';
    }
    if (message.kind === 'tool.call') {
      const payload = message.payload as { toolName?: unknown; params?: unknown };
      return `${String(payload.toolName ?? message.to)} ${JSON.stringify(payload.params ?? {}).slice(0, 500)}`;
    }
    if (message.kind === 'tool.result') {
      const payload = message.payload as {
        success?: unknown;
        error?: unknown;
        result?: {
          root?: unknown;
          entries?: unknown;
          path?: unknown;
          content?: unknown;
          command?: unknown;
          stdout?: unknown;
          query?: unknown;
          provider?: unknown;
          results?: Array<{ title?: unknown; url?: unknown }>;
          finalUrl?: unknown;
          title?: unknown;
          text?: unknown;
        };
      };
      if (payload.success === false) return `Tool failed: ${String(payload.error ?? 'unknown error')}`;
      if (Array.isArray(payload.result?.entries)) {
        const entries = payload.result.entries.filter((item): item is string => typeof item === 'string');
        return `Observed ${String(payload.result.root ?? '.')}: ${entries.slice(0, 30).join(', ')}`.slice(0, 1000);
      }
      if (typeof payload.result?.path === 'string') {
        return `Read ${payload.result.path}: ${String(payload.result.content ?? '').replace(/\s+/g, ' ').slice(0, 700)}`;
      }
      if (typeof payload.result?.command === 'string') {
        return `Executed ${payload.result.command}: ${String(payload.result.stdout ?? '').replace(/\s+/g, ' ').slice(0, 700)}`;
      }
      if (Array.isArray(payload.result?.results)) {
        return `Searched ${String(payload.result.query ?? 'the web')} via ${String(payload.result.provider ?? 'provider')}: ${payload.result.results
          .slice(0, 8)
          .map(item => `${String(item.title ?? 'Untitled')} (${String(item.url ?? '')})`)
          .join(', ')}`.slice(0, 1000);
      }
      if (typeof payload.result?.finalUrl === 'string') {
        return `Fetched ${payload.result.finalUrl}: ${String(payload.result.title ?? '').trim()} ${String(payload.result.text ?? '').replace(/\s+/g, ' ').slice(0, 700)}`;
      }
      return `${message.from} returned a successful tool result to ${message.to}`;
    }
    return `${message.from} sent ${message.kind} to ${message.to}`;
  }

  private summarizeEvent(event: ExecutionObservableEvent): string | undefined {
    if (typeof event.data?.reason === 'string') return event.data.reason.slice(0, 500);
    if (typeof event.data?.purpose === 'string') return event.data.purpose.slice(0, 500);
    if (typeof event.data?.toolName === 'string') return `${event.data.toolName}${event.data.success === false ? ' failed' : ''}`;
    if (typeof event.data?.round === 'number' || typeof event.data?.rounds === 'number') {
      return `round=${String(event.data.round ?? event.data.rounds)}, calls=${String(event.data.calls ?? event.data.totalCalls ?? 0)}, stop=${String(event.data.stopReason ?? 'continue')}`;
    }
    return undefined;
  }

  private compactData(data?: Record<string, unknown>): Record<string, unknown> | undefined {
    if (!data) return undefined;
    const keys = [
      'reason', 'purpose', 'from', 'to', 'state', 'toolName', 'params', 'success', 'error',
      'parentId', 'teamId', 'archetype', 'name', 'totalTokens', 'action', 'count', 'nodeId',
      'round', 'rounds', 'calls', 'totalCalls', 'successfulCalls', 'failedCalls', 'stopReason', 'durationMs',
    ];
    const compact = Object.fromEntries(keys.filter(key => data[key] !== undefined).map(key => [key, data[key]]));
    return Object.keys(compact).length > 0 ? compact : undefined;
  }
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
