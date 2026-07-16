import type {
  ActorKind,
  ActorLifecycleAction,
  ActorLifecycleDecision,
  ActorLifecycleOrigin,
  ActorLifecycleOutcome,
  ActorLifecyclePolicy,
  ActorLifecycleRecord,
  ActorLifecycleRegistration,
} from './types.js';

export class ActorLifecycleRegistry {
  private readonly records = new Map<string, ActorLifecycleRecord>();
  private sequence = 0;

  register(input: ActorLifecycleRegistration): ActorLifecycleRecord {
    const record: ActorLifecycleRecord = {
      ...input,
      policy: { ...input.policy },
      status: 'active',
      updatedAt: input.createdAt,
    };
    this.records.set(input.actorId, record);
    return this.clone(record);
  }

  get(actorId: string): ActorLifecycleRecord | undefined {
    const record = this.records.get(actorId);
    return record ? this.clone(record) : undefined;
  }

  list(filter: { actorKind?: ActorKind; status?: ActorLifecycleRecord['status'] } = {}): ActorLifecycleRecord[] {
    return [...this.records.values()]
      .filter(record => !filter.actorKind || record.actorKind === filter.actorKind)
      .filter(record => !filter.status || record.status === filter.status)
      .map(record => this.clone(record));
  }

  decide(
    actorId: string,
    outcome: ActorLifecycleOutcome,
    options: { action?: ActorLifecycleAction; correlationId?: string; reason?: string } = {}
  ): ActorLifecycleDecision {
    const record = this.require(actorId);
    const action = options.action ?? this.resolveAction(record.origin, record.policy, outcome);
    const now = Date.now();
    const decision: ActorLifecycleDecision = {
      id: `lifecycle_${now}_${String(++this.sequence).padStart(4, '0')}`,
      actorId,
      actorKind: record.actorKind,
      action,
      outcome,
      origin: record.origin,
      reason: options.reason ?? this.defaultReason(record.origin, record.policy, outcome, action),
      cascade: record.policy.cascade,
      correlationId: options.correlationId,
      decidedAt: now,
    };
    record.lastDecision = decision;
    record.updatedAt = now;
    return { ...decision };
  }

  markApplied(actorId: string, decision: ActorLifecycleDecision, snapshotPath?: string): ActorLifecycleRecord {
    const record = this.require(actorId);
    const now = Date.now();
    record.status = decision.action === 'release'
      ? 'released'
      : decision.action === 'persist'
        ? 'persisted'
        : 'retained';
    record.lastDecision = { ...decision, appliedAt: now, snapshotPath };
    record.updatedAt = now;
    return this.clone(record);
  }

  markRestored(actorId: string): ActorLifecycleRecord {
    return this.markActive(actorId);
  }

  markActive(actorId: string): ActorLifecycleRecord {
    const record = this.require(actorId);
    record.status = 'active';
    record.updatedAt = Date.now();
    return this.clone(record);
  }

  remove(actorId: string): boolean {
    return this.records.delete(actorId);
  }

  clear(): void {
    this.records.clear();
    this.sequence = 0;
  }

  private resolveAction(
    origin: ActorLifecycleOrigin,
    policy: ActorLifecyclePolicy,
    outcome: ActorLifecycleOutcome
  ): ActorLifecycleAction {
    if (outcome === 'failure' && policy.retainOnFailure) return 'retain_session';
    if (policy.mode !== 'adaptive') return policy.mode;
    if (origin === 'automatic_delegation' || origin === 'evolution') return 'release';
    return 'retain_session';
  }

  private defaultReason(
    origin: ActorLifecycleOrigin,
    policy: ActorLifecyclePolicy,
    outcome: ActorLifecycleOutcome,
    action: ActorLifecycleAction
  ): string {
    if (outcome === 'failure' && policy.retainOnFailure) return 'Retained after failure for inspection and recovery.';
    if (policy.mode !== 'adaptive') return `Applied explicit ${policy.mode} lifecycle policy.`;
    if (action === 'release') return `Released completed ${origin.replaceAll('_', ' ')} runtime instance.`;
    return 'Retained runtime instance for the current session.';
  }

  private require(actorId: string): ActorLifecycleRecord {
    const record = this.records.get(actorId);
    if (!record) throw new Error(`Lifecycle actor "${actorId}" not found`);
    return record;
  }

  private clone(record: ActorLifecycleRecord): ActorLifecycleRecord {
    return {
      ...record,
      policy: { ...record.policy },
      lastDecision: record.lastDecision ? { ...record.lastDecision } : undefined,
    };
  }
}
