import type { AgentRole } from '../agent/BaseAgent.js';
import type { MultiPartyTrace } from '../communication/index.js';
import type { ConversationEntry, RootMemoryContext, WorkspaceMemoryManager } from '../memory/index.js';
import type {
  ContextWindow,
  ContextWindowPolicy,
  ContextWindowRequest,
} from './types.js';

const DEFAULT_POLICY: ContextWindowPolicy = {
  sessionWindowTurns: 10,
  maxContextTokens: 4000,
  includeToolResults: 'summary',
  includeSubagentReports: 'summary',
  includePrivateMemory: true,
  includePublicMemory: true,
};

export class ContextWindowManager {
  constructor(
    private readonly memory: WorkspaceMemoryManager,
    private readonly policy: ContextWindowPolicy = DEFAULT_POLICY
  ) {}

  async build(request: ContextWindowRequest): Promise<ContextWindow> {
    const effectiveTurns = Math.max(
      0,
      Math.min(request.memoryScope.sessionWindowTurns, this.policy.sessionWindowTurns)
    );
    const [rootContext, privateBundle, conversation] = await Promise.all([
      request.memoryScope.public && this.policy.includePublicMemory
        ? this.memory.loadRootContext()
        : Promise.resolve(undefined),
      request.memoryScope.private && this.policy.includePrivateMemory
        ? this.memory.loadAgentMemory(request.agentKey)
        : Promise.resolve(undefined),
      effectiveTurns > 0
        ? this.memory.readConversation(request.sessionId, Math.max(40, effectiveTurns * 12))
        : Promise.resolve([]),
    ]);

    const publicContext = rootContext ? this.formatPublicContext(rootContext) : '';
    const privateMemory = privateBundle
      ? [privateBundle.memory.trim(), privateBundle.context.trim()].filter(Boolean).join('\n\n')
      : '';
    const sessionContext = this.compactConversation(
      conversation,
      request.role,
      request.agentId,
      effectiveTurns
    );
    const parentContext = request.memoryScope.parentContext ? request.parentContext?.trim() ?? '' : '';
    const communicationContext = request.communicationContext?.trim() ?? '';
    const multiPartyTraceContext = this.compactSystemTraces(request.systemTraces ?? []);

    const bounded = this.fitToBudget({
      publicContext,
      privateMemory,
      sessionContext,
      parentContext,
      task: request.task,
      communicationContext,
      multiPartyTraceContext,
    });

    return {
      ...bounded,
      tokenUsage: {
        publicContext: this.estimateTokens(bounded.publicContext),
        privateMemory: this.estimateTokens(bounded.privateMemory),
        sessionWindow: this.estimateTokens(bounded.sessionContext),
        parentContext: this.estimateTokens(bounded.parentContext),
        task: this.estimateTokens(bounded.task),
        communicationContext: this.estimateTokens(bounded.communicationContext),
        multiPartyTraces: this.estimateTokens(bounded.multiPartyTraceContext),
        total: this.estimateTokens(Object.values(bounded).join('\n')),
      },
      sources: {
        public: rootContext
          ? ['.roy/public/project.md', '.roy/public/context.md', '.roy/public/constraints.md', '.roy/public/decisions.md', '.roy/public/glossary.md', '.roy/public/user.md']
          : [],
        private: privateBundle
          ? [`.roy/agents/${privateBundle.key}/memory.md`, `.roy/agents/${privateBundle.key}/context.md`]
          : [],
        session: `.roy/sessions/${request.sessionId}.jsonl (last ${effectiveTurns} turns, compacted)`,
        parent: parentContext ? `approved context from ${request.agentId === 'root' ? 'runtime' : 'parent agent'}` : 'none',
        communication: communicationContext ? 'runtime communication protocol context' : 'none',
        traces: request.systemTraces?.length ? `${request.systemTraces.length} observable system trace(s)` : 'none',
      },
    };
  }

  private compactConversation(
    entries: ConversationEntry[],
    role: AgentRole,
    agentId: string,
    maxTurns: number
  ): string {
    if (maxTurns <= 0 || entries.length === 0) return '';

    const visible = entries.filter(entry => this.isVisible(entry, role, agentId));
    const turns = new Map<string, ConversationEntry[]>();
    for (const entry of visible) {
      const key = entry.turnId ?? entry.correlationId ?? `entry:${entry.id}`;
      const list = turns.get(key) ?? [];
      list.push(entry);
      turns.set(key, list);
    }

    const selected = [...turns.entries()].slice(-maxTurns);
    if (selected.length === 0) return '';
    const lines = ['<recent_conversation>'];
    selected.forEach(([turnId, records], index) => {
      lines.push(`Turn ${index + 1} (${turnId}):`);
      for (const record of records) {
        const content = this.compactEntry(record);
        if (content) lines.push(`- ${record.speaker} [${record.role}]: ${content}`);
      }
    });
    lines.push('</recent_conversation>');
    return lines.join('\n');
  }

  private isVisible(entry: ConversationEntry, role: AgentRole, agentId: string): boolean {
    if (entry.role === 'user' || entry.role === 'assistant' || entry.role === 'system') return true;
    if (entry.role === 'tool') return this.policy.includeToolResults === 'summary';
    if (entry.role !== 'agent' || this.policy.includeSubagentReports === 'none') return false;
    if (role === 'root') return true;
    const metadata = entry.metadata ?? {};
    return metadata.agentId === agentId || metadata.parentId === agentId;
  }

  private compactEntry(entry: ConversationEntry): string {
    if (entry.role === 'tool') {
      const metadata = entry.metadata ?? {};
      const toolName = String(metadata.toolName ?? metadata.kind ?? 'tool');
      return `${toolName}: ${this.truncate(entry.content.replace(/\s+/g, ' '), 320)}`;
    }
    const maxLength = entry.role === 'agent' && this.policy.includeSubagentReports === 'full' ? 1400 : 700;
    return this.truncate(entry.content.replace(/\s+/g, ' ').trim(), maxLength);
  }

  private fitToBudget(parts: {
    publicContext: string;
    privateMemory: string;
    sessionContext: string;
    parentContext: string;
    task: string;
    communicationContext: string;
    multiPartyTraceContext: string;
  }): typeof parts {
    const maxChars = Math.max(800, this.policy.maxContextTokens * 4);
    const taskReserve = Math.min(parts.task.length, Math.floor(maxChars * 0.25));
    let remaining = maxChars - taskReserve;
    const allocate = (value: string, share: number): string => {
      const limit = Math.max(0, Math.min(remaining, Math.floor(maxChars * share)));
      const selected = this.truncate(value, limit);
      remaining -= selected.length;
      return selected;
    };

    return {
      publicContext: allocate(parts.publicContext, 0.24),
      privateMemory: allocate(parts.privateMemory, 0.15),
      communicationContext: allocate(parts.communicationContext, 0.16),
      multiPartyTraceContext: allocate(parts.multiPartyTraceContext, 0.16),
      sessionContext: allocate(parts.sessionContext, 0.18),
      parentContext: allocate(parts.parentContext, 0.11),
      task: this.truncate(parts.task, taskReserve || maxChars),
    };
  }

  private formatPublicContext(context: RootMemoryContext): string {
    return [
      '<project_memory>', context.projectMemory.trim(), '</project_memory>',
      '<constraints>', context.constraints.trim(), '</constraints>',
      '<decisions>', context.decisions.trim(), '</decisions>',
      '<glossary>', context.glossary.trim(), '</glossary>',
      `<agent_patterns>${JSON.stringify(context.agentPatterns)}</agent_patterns>`,
      `<team_patterns>${JSON.stringify(context.teamPatterns)}</team_patterns>`,
      `<delegation_patterns>${JSON.stringify(context.delegationPatterns)}</delegation_patterns>`,
    ].join('\n');
  }

  private truncate(value: string, maxChars: number): string {
    if (maxChars <= 0 || !value) return '';
    if (value.length <= maxChars) return value;
    return `${value.slice(0, Math.max(0, maxChars - 24)).trimEnd()}\n...[context truncated]`;
  }

  private estimateTokens(value: string): number {
    if (!value.trim()) return 0;
    return Math.max(1, Math.ceil(value.length / 4));
  }

  private compactSystemTraces(traces: MultiPartyTrace[]): string {
    if (traces.length === 0) return '';
    return [
      '<multi_party_traces>',
      ...traces.slice(-50).map(trace => {
        const recipients = trace.to.map(actor => actor.id).join(', ') || 'none';
        const content = trace.content ? `: ${this.truncate(trace.content.replace(/\s+/g, ' '), 320)}` : '';
        return `- [${trace.phase}] ${trace.from.id} -> ${recipients} ${trace.kind}${content}`;
      }),
      '</multi_party_traces>',
    ].join('\n');
  }
}

export { DEFAULT_POLICY as DEFAULT_CONTEXT_WINDOW_POLICY };
