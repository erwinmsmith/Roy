export type ToolApprovalDecision = 'auto' | 'ask' | 'deny';
export type ToolPermission = 'read_only' | 'write' | 'execute';

export interface ToolApprovalPolicy {
  readOnly: ToolApprovalDecision;
  write: ToolApprovalDecision;
  execute: ToolApprovalDecision;
  overrides: Record<string, ToolApprovalDecision>;
}

export interface ToolApprovalRequest {
  id: string;
  agentId: string;
  toolName: string;
  permission: ToolPermission;
  params: Record<string, unknown>;
  reason?: string;
  status: 'pending' | 'approved' | 'denied';
  createdAt: number;
  updatedAt: number;
}

export interface ToolApprovalResult {
  decision: 'approved' | 'denied' | 'pending';
  request: ToolApprovalRequest;
}

export class ToolApprovalManager {
  private requests = new Map<string, ToolApprovalRequest>();
  private sequence = 0;

  constructor(private policy: ToolApprovalPolicy) {}

  configure(policy: ToolApprovalPolicy): void {
    this.policy = policy;
  }

  authorize(input: Omit<ToolApprovalRequest, 'id' | 'status' | 'createdAt' | 'updatedAt'>): ToolApprovalResult {
    const now = Date.now();
    const configured = this.policy.overrides[input.toolName] ?? this.forPermission(input.permission);
    const status = configured === 'auto' ? 'approved' : configured === 'deny' ? 'denied' : 'pending';
    const request: ToolApprovalRequest = {
      ...input,
      id: `tool_approval_${now}_${String(++this.sequence).padStart(4, '0')}`,
      status,
      createdAt: now,
      updatedAt: now,
    };
    this.requests.set(request.id, request);
    return { decision: status, request: { ...request, params: { ...request.params } } };
  }

  resolve(id: string, decision: 'approved' | 'denied'): ToolApprovalRequest | undefined {
    const request = this.requests.get(id);
    if (!request || request.status !== 'pending') return undefined;
    request.status = decision;
    request.updatedAt = Date.now();
    return { ...request, params: { ...request.params } };
  }

  list(status?: ToolApprovalRequest['status']): ToolApprovalRequest[] {
    return [...this.requests.values()]
      .filter(request => !status || request.status === status)
      .map(request => ({ ...request, params: { ...request.params } }));
  }

  get(id: string): ToolApprovalRequest | undefined {
    const request = this.requests.get(id);
    return request ? { ...request, params: { ...request.params } } : undefined;
  }

  private forPermission(permission: ToolPermission): ToolApprovalDecision {
    if (permission === 'read_only') return this.policy.readOnly;
    if (permission === 'write') return this.policy.write;
    return this.policy.execute;
  }
}
