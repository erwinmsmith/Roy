// Runtime - Lifecycle management and orchestration for Roy Agent System

import 'dotenv/config';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { config } from '../../config/index.js';
import { logger } from '../utils/logger.js';
import { configureLogging, shutdownLogging } from '../logging/index.js';
import {
  llmFactory,
  tokenUsageRegistry,
  type LLMCompletionOptions,
  type LLMCompletionResult,
  type LLMMessage,
  type LLMProvider,
  type ModelTokenUsage,
} from '../llm/index.js';
import { AgentManager } from './AgentManager.js';
import {
  RootExecutionTreeRegistry,
  type CompleteRootExecutionStepInput,
  type RootExecutionActivity,
  type RootExecutionCheckpoint,
  type RootExecutionNodeSnapshot,
  type RootExecutionStep,
  type RootExecutionStepDecision,
  type RootExecutionTreeState,
} from './executionTree.js';
import { RootExecutionActivityProjector } from './executionActivity.js';
import {
  compactExecutionKnowledgeForPrompt,
  type ExecutionCachedActor,
  type ExecutionCachedToolCall,
  type ExecutionCacheSnapshot,
  type ExecutionFeedbackRecord,
  type ExecutionKnowledgeCacheState,
} from './executionCache.js';
import { RootTaskLoopController } from './taskLoop.js';
import { FSM } from '../executor/FSM.js';
import { signalBus } from '../executor/SignalBus.js';
import { UnifiedAgent } from '../agent/UnifiedAgent.js';
import type { AgentInfo, AgentUsage, BaseAgent, ToMProfile } from '../agent/BaseAgent.js';
import { actionRegistry } from '../actions/index.js';
import {
  AgentToolExecutionLoop,
  completedWorkspaceReadCoversPlan,
  effectiveWorkspaceMutationCallIndices,
  FsListTool,
  FsReadTool,
  FsReplaceTool,
  FsSearchTool,
  FsSynthesizeTool,
  FsWriteTool,
  findParallelSourceMutation,
  hasEffectiveWorkspaceMutationCall,
  isSuccessfulWorkspaceMutationCall,
  isSuccessfulWorkspaceVerificationCall,
  isUnavailableWorkspaceVerificationCall,
  isWorkspaceVerificationCall,
  plannedWorkspaceMutationPath,
  ShellExecTool,
  taskRequestsWorkspaceMutation,
  workspaceCandidateRollbackFromCall,
  workspaceTargetNeedsFreshNoGainEvidence,
  workspaceToolIntentFingerprint,
  WebFetchTool,
  WebSearchTool,
  registerCoreTools,
  toolRegistry,
  type Tool,
  type ToolLoopSummary,
} from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import { DelegateToSubagentSkill } from '../skills/delegation.js';
import type {
  AgentComputeNodeDefinition,
  AgentComputeNodeExecution,
  AgentComputeNodeRequest,
  AgentCreationInvocation,
  AgentNodeCreationMode,
} from '../skills/agentCreation.js';
import { UseToolWhenNeededSkill } from '../skills/toolUse.js';
import {
  DefaultDelegationCandidatePlanner,
  HashTaskEmbeddingProvider,
  type DelegationCandidateInput,
  type DelegationCandidateSelection,
  type LLMDelegationScorerInvocation,
} from '../delegation/index.js';
import {
  CompositeEvolutionEvaluator,
  EvolutionLifecycleEngine,
  TeamFirstGenomePlanner,
  WeightedTopKSelectionPolicy,
  defaultMutationOperators,
  validateTeamGenome,
  type EvolutionAblations,
  type EvolutionCandidate,
  type EvolutionEvaluationDimensions,
  type EvolutionEvaluationResult,
  type EvolutionExecutionArtifact,
  type EvolutionLifecycleResult,
  type EvolutionJudge,
  type EvolutionPattern,
  type EvolutionProfile,
  type EvolutionRunOptions,
  type EvolutionRunResult,
  type EvolutionSeedAgent,
  type GenomeToMProfile,
} from '../evolution/index.js';
import {
  normalizeToMProfile,
  ToMDelegationPlanner,
  type ToMAnalysisSignals,
  type ToMDelegationEngine,
  type ToMTaskAnalysis,
} from '../tom/index.js';
import {
  AgentCommunicationManager,
  type AgentCommunicationProtocol,
  type CommunicationState,
  type MultiPartyTrace,
} from '../communication/index.js';
import { ContextWindowManager, type ContextWindow } from '../context/index.js';
import {
  BudgetMarket,
  WeightedReasoningInvestmentModel,
  type BudgetAllocation,
  type BudgetMarketState,
  type BudgetOutcome,
  type BudgetPriority,
  type BudgetRebalanceResult,
  type BudgetRequest,
  type ReasoningInvestmentModel,
} from '../budget/index.js';
import {
  executeTeamItems,
  normalizeTeamExecutionPolicy,
  TeamRegistry,
  type TeamExecutionOutcome,
  type TeamExecutionPolicy,
  type TeamFSMState,
  type TeamRuntimeState,
} from '../team/index.js';
import { ToolApprovalManager, type ToolApprovalRequest } from '../tools/approval.js';
import type { ToolResult } from '../tools/types.js';
import { AgentToolPlanner, type PlannedToolCall } from '../tools/planner.js';
import {
  InMemoryMessageQueue,
  MessageScheduler,
  type EnqueueMessageInput,
  type MessageQueue,
  type QueueState,
  type QueueTransition,
  type RuntimeMessage,
} from '../queue/index.js';
import {
  WorkspaceMemoryManager,
  type ConversationEntry,
  type ConversationSessionState,
  type MemoryAutoState,
  type MemoryMode,
  type MemoryProposalSummary,
  type MemorySignals,
  type MemoryUpdateProposal,
  type MemoryUpdateRecord,
  type WorkspaceMemoryState,
  type WorkspaceRuntimeConfig,
  type RootMemoryContext,
} from '../memory/index.js';
import {
  ActorLifecycleRegistry,
  type ActorKind,
  type ActorLifecycleAction,
  type ActorLifecycleMode,
  type ActorLifecycleOrigin,
  type ActorLifecycleOutcome,
  type ActorLifecyclePolicy,
  type ActorLifecycleRecord,
  type PersistedActorSnapshot,
} from '../lifecycle/index.js';

export interface RuntimeConfig {
  agentName?: string;
  agentGoal?: string;
  sessionId?: string;
  fsmEnabled?: boolean;
  budget?: number;
  /** Optional external wall-clock deadline for this invocation. */
  wallClockLimitMs?: number;
  mode?: 'conversational' | 'action' | 'hybrid';
  llmProvider?: LLMProvider;
  workspaceCwd?: string;
  communicationProtocols?: AgentCommunicationProtocol[];
  tomPlanner?: ToMDelegationEngine;
  reasoningInvestmentModel?: ReasoningInvestmentModel;
}

export interface RuntimeContext {
  config: typeof config;
  llm: LLMProvider | null;
  fsm: FSM;
  signalBus: typeof signalBus;
  manager: AgentManager;
  agent: UnifiedAgent;
  sessionId: string;
  queue: MessageQueue;
  scheduler: MessageScheduler;
  memory: WorkspaceMemoryManager;
  communication: AgentCommunicationManager;
  capabilities: {
    skills: number;
    actions: number;
    tools: number;
  };
}

export interface TokenUsage extends AgentUsage {
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number | null;
  cachedInputTokens: number | null;
  cacheCreationInputTokens: number | null;
  estimatedCostUsd?: number;
}

export interface BudgetState {
  mode: 'unlimited' | 'limited';
  limitTokens?: number;
  usedTokens: number;
  remainingTokens?: number;
  perAgent: Record<string, TokenUsage>;
  perTeam: Record<string, TokenUsage>;
  perTurn: TokenUsage[];
}

interface DelegationScorerBudgetContext {
  parentId: string;
  correlationId?: string;
  usageBefore: AgentUsage;
  allocation?: BudgetAllocation;
  ownsAllocation: boolean;
}

export interface RuntimeEvent {
  type: string;
  timestamp: number;
  agentId?: string;
  sessionId?: string;
  correlationId?: string;
  nodeId?: string;
  data?: Record<string, unknown>;
}

export interface RuntimeState {
  sessionId: string;
  rootAgentId: string;
  rootAgent: AgentInfo;
  agents: AgentInfo[];
  teams: TeamRuntimeState[];
  events: RuntimeEvent[];
  budget: BudgetState;
}

export interface ToMRuntimeState {
  analyses: ToMTaskAnalysis[];
  agents: Array<{
    agentId: string;
    name: string;
    parentId?: string;
    teamId?: string;
    profile: ToMProfile;
  }>;
  teams: Array<{
    teamId: string;
    name: string;
    parentAgentId: string;
    profile: ToMProfile;
  }>;
}

export type SubAgentArchetype =
  | 'researcher'
  | 'critic'
  | 'planner'
  | 'coder'
  | 'summarizer'
  | 'tester'
  | 'custom';

export interface ToolBinding {
  name: string;
  enabled: boolean;
  permission: 'read_only' | 'write' | 'execute';
  constraints?: {
    allowedPaths?: string[];
    blockedPaths?: string[];
    allowlistedCommands?: string[];
    maxCalls?: number;
  };
}

export interface SkillBinding {
  name: string;
  enabled: boolean;
  description: string;
  constraints?: {
    maxCalls?: number;
    requiresApproval?: boolean;
  };
}

export interface AgentSpawnPolicy {
  canSpawn: boolean;
  maxChildren: number;
  maxDepth: number;
  maxTotalAgentsPerTurn: number;
  allowCustomAgents: boolean;
  budgetAware: boolean;
  allowedStates: string[];
}

export interface AgentMemoryScope {
  public: boolean;
  private: boolean;
  parentContext: boolean;
  sessionWindowTurns: number;
}

export interface SpawnAgentSpec {
  parentId: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  archetype: SubAgentArchetype;
  tomLevel: number;
  description: string;
  task?: string;
  tools?: string[] | ToolBinding[];
  skills?: string[] | SkillBinding[];
  memoryScope?: AgentMemoryScope;
  spawnPolicy?: Partial<AgentSpawnPolicy>;
  budgetTokens?: number;
  systemPrompt?: string;
  outputContract?: AgentComputeNodeRequest['outputContract'];
  correlationId?: string;
  teamId?: string;
  nodeDefinition?: AgentComputeNodeDefinition;
  cognitiveGapIds?: string[];
  existenceReason?: string;
  communicationProtocol?: string;
  tomProfile?: ToMProfile;
  lifecycle?: Partial<ActorLifecyclePolicy>;
  lifecycleOrigin?: ActorLifecycleOrigin;
  instanceId?: string;
}

export interface AgentTreeNode {
  agent: AgentInfo;
  children: AgentTreeNode[];
}

export interface TeamMemberSpec {
  archetype: SubAgentArchetype;
  task: string;
  name?: string;
  role?: string;
  style?: string;
  tools?: string[];
  skills?: string[];
  budgetTokens?: number;
  tomLevel?: number;
  tomProfile?: ToMProfile;
  cognitiveGapIds?: string[];
  existenceReason?: string;
  memoryScope?: AgentMemoryScope;
  lead?: boolean;
  communicationProtocol?: string;
  systemPrompt?: string;
}

export interface SpawnTeamSpec {
  parentAgentId?: string;
  name: string;
  description: string;
  tomLevel?: number;
  tomProfile?: ToMProfile;
  tomAnalysis?: ToMTaskAnalysis;
  leadAgentId?: string;
  task?: string;
  synthesisPolicy?: string;
  members?: TeamMemberSpec[];
  correlationId?: string;
  executionPolicy?: Partial<TeamExecutionPolicy>;
  lifecycle?: Partial<ActorLifecyclePolicy>;
  lifecycleOrigin?: ActorLifecycleOrigin;
  instanceId?: string;
}

export interface TeamTreeNode {
  team: TeamRuntimeState;
  members: AgentTreeNode[];
}

export type RuntimeActorNode = RuntimeAgentActorNode | RuntimeTeamActorNode;

export interface RuntimeAgentActorNode {
  type: 'agent';
  agent: AgentInfo;
  children: RuntimeActorNode[];
}

export interface RuntimeTeamActorNode {
  type: 'team';
  team: TeamRuntimeState;
  children: RuntimeActorNode[];
}

export interface RuntimeActorTree {
  root: AgentInfo;
  teams: TeamTreeNode[];
  hierarchy: RuntimeAgentActorNode;
}

export interface TeamRunResult {
  team: TeamRuntimeState;
  result: string;
  members: RunAgentResult[];
  memberExecutions: RootMediatedSpawnResult[];
  memberOutcomes: TeamMemberRunOutcome[];
  correlationId: string;
  messages: RuntimeMessage[];
  usage: TokenUsage;
}

export interface TeamMemberRunOutcome {
  key: string;
  agentId?: string;
  status: TeamExecutionOutcome<unknown>['status'];
  error?: string;
}

export interface RunAgentResult {
  agent: AgentInfo;
  result: string;
  usage: TokenUsage;
  toolCalls: ToolCallRecord[];
  evidence: RunEvidence;
  grounded: boolean;
  warnings: string[];
  toolLoop?: ToolLoopSummary;
}

export interface RunEvidence {
  toolGrounded: boolean;
  outputGrounded: boolean;
  observedPaths: string[];
  observedUrls?: string[];
  relevantObservedUrls?: string[];
  discoveredUrls?: string[];
  toolResultSummary?: string;
}

export interface ToolCallRecord {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  success: boolean;
  error?: string;
  reason?: string;
  round?: number;
  startedAt?: number;
  completedAt?: number;
}

interface GroundingRunResult {
  toolCalls: ToolCallRecord[];
  grounded: boolean;
  warnings: string[];
  context: string;
  evidence: RunEvidence;
  toolLoop: ToolLoopSummary;
  acceptanceAudit?: WorkspaceAcceptanceAudit;
}

interface WorkspaceAcceptanceAuditItem {
  id: string;
  status: 'verified' | 'failed' | 'blocked' | 'unverified';
  evidence: string;
}

interface WorkspaceAcceptanceAudit {
  required: boolean;
  performed: boolean;
  passed: boolean;
  items: WorkspaceAcceptanceAuditItem[];
  missingItemIds: string[];
  toolEvidenceSufficient: boolean;
  reason: string;
}

interface WorkspaceExecutionClosureStatus {
  mutationApplied: boolean;
  verificationAttemptedAfterMutation: boolean;
  verificationPassed: boolean;
  acceptanceAuditRequired: boolean;
  acceptanceAuditPerformed: boolean;
  acceptanceAuditPassed: boolean;
  acceptanceItems: number;
  acceptanceItemsVerified: number;
  acceptanceItemsFailed: number;
  closed: boolean;
  lastMutationCallIndex: number;
  lastVerificationCallIndex: number;
  failedVerificationCallsAfterMutation: number;
  unresolvedVerificationFailures: number;
}

interface ExecutionResumeState {
  sourceCorrelationId: string;
  anchorPathId: string;
  knowledge: ExecutionKnowledgeCacheState;
  actionableFeedback: number;
  openPaths: number;
}

export interface SpawnCommandPayload {
  archetype: SubAgentArchetype;
  task: string;
  parentId?: string;
  correlationId?: string;
  source?: string;
  name?: string;
  customRole?: string;
  customStyle?: string;
  tools?: string[];
  skills?: string[];
  tomLevel?: number;
  budgetTokens?: number;
  memoryScope?: AgentMemoryScope;
  spawnPolicy?: Partial<AgentSpawnPolicy>;
  tomProfile?: ToMProfile;
  reuseMode?: NonNullable<AgentComputeNodeRequest['reuse']>['mode'];
  outputContract?: AgentComputeNodeRequest['outputContract'];
  requireRootSynthesis?: boolean;
  showSubagentOutput?: boolean;
  disableRecursiveDelegation?: boolean;
  teamId?: string;
  cognitiveGapIds?: string[];
  existenceReason?: string;
  communicationProtocol?: string;
  systemPrompt?: string;
  lifecycle?: Partial<ActorLifecyclePolicy>;
  lifecycleOrigin?: ActorLifecycleOrigin;
}

export type DelegationDecision =
  | {
      action: 'solve_directly';
      reason: string;
    }
  | {
      action: 'ask_clarification';
      reason: string;
      question: string;
    }
  | {
      action: 'spawn_subagents';
      reason: string;
      agents: DelegationAgentPlan[];
      coordination?: 'independent' | 'team';
      team?: DelegationTeamPlan;
      continuationPolicy?: 'reassess' | 'finalize_after_round';
    };

export interface DelegationAgentPlan {
  archetype: SubAgentArchetype;
  name?: string;
  role?: string;
  style?: string;
  description?: string;
  task: string;
  tools?: string[];
  skills?: string[];
  tomLevel?: number;
  budgetTokens?: number;
  tomProfile?: ToMProfile;
  cognitiveGapIds?: string[];
  existenceReason?: string;
  memoryScope?: AgentMemoryScope;
  systemPrompt?: string;
}

export interface DelegationTeamPlan {
  name: string;
  description: string;
  task?: string;
  synthesisPolicy?: string;
  tomLevel?: number;
  executionPolicy?: Partial<TeamExecutionPolicy>;
  memberDelegationPolicy?: 'allow' | 'deny';
}

type RootContinuationDecision =
  | {
      action: 'finalize';
      reason: string;
    }
  | {
      action: 'ask_clarification';
      reason: string;
      question: string;
    }
  | {
      action: 'delegate_more';
      reason: string;
      agents: DelegationAgentPlan[];
      coordination?: 'independent' | 'team';
      team?: DelegationTeamPlan;
      continuationPolicy?: 'reassess' | 'finalize_after_round';
    };

interface RootDelegationRoundResult {
  subagents: RootMediatedSpawnResult[];
  teams: TeamRunResult[];
  evolution?: EvolutionRunResult;
}

interface RootResponseAcceptanceAudit {
  complete: boolean;
  unmetRequirements: string[];
  reason: string;
  obligations: Array<{
    requirement: string;
    satisfied: boolean;
    candidateEvidence: string;
    factualConcern?: string;
  }>;
}

interface RootResponseAcceptanceReference {
  requirement: string;
  acceptedAnswers: string[];
  confidence: number;
  evidenceGrounded: boolean;
}

export interface RootMediatedSpawnResult {
  correlationId: string;
  node: AgentComputeNodeDefinition;
  agent: AgentInfo;
  subagentResult: RunAgentResult;
  finalResponse: string;
  messages: RuntimeMessage[];
  creationUsage: AgentCreationUsage;
}

export interface RootTurnResult {
  correlationId: string;
  decision: DelegationDecision;
  finalResponse: string;
  subagents: RootMediatedSpawnResult[];
  teams: TeamRunResult[];
  evolution?: EvolutionRunResult;
  evolutions: EvolutionRunResult[];
  executionTree: RootExecutionTreeState;
  messages: RuntimeMessage[];
  usage: {
    root: TokenUsage;
    subagents: Record<string, TokenUsage>;
    teamSynthesis: Record<string, TokenUsage>;
    total: TokenUsage;
  };
}

export interface RootTurnRecoveryOptions {
  maxAttempts?: number;
  retryInitialDelayMs?: number;
  retryMaxDelayMs?: number;
}

export interface RootTurnRecoveryResult {
  result: RootTurnResult;
  attempts: number;
  recovered: boolean;
  correlationIds: string[];
}

export interface MultiTurnExperimentInput {
  turns: string[];
  stopOnError?: boolean;
}

export interface MultiTurnExperimentTurn {
  index: number;
  input: string;
  status: 'completed' | 'failed';
  result?: RootTurnResult;
  error?: string;
  eventTypes: string[];
  agentIds: string[];
  teamIds: string[];
  budget: BudgetState;
}

export interface MultiTurnExperimentResult {
  sessionId: string;
  startedAt: number;
  completedAt: number;
  turns: MultiTurnExperimentTurn[];
  completedTurns: number;
  failedTurns: number;
  totalUsage: TokenUsage;
}

export interface RunEvolutionInput {
  task: string;
  parentId?: string;
  correlationId?: string;
  seedAgents?: EvolutionSeedAgent[];
  profile?: EvolutionProfile;
  options?: Partial<Omit<EvolutionRunOptions, 'ablations'>> & {
    ablations?: Partial<EvolutionAblations>;
  };
}

export interface EvolutionBenchmarkResult {
  task: string;
  profiles: EvolutionProfile[];
  runs: EvolutionRunResult[];
  comparison: Array<{
    profile: EvolutionProfile;
    success: boolean;
    score: number;
    totalTokens: number;
    thinkingTokens: number | null;
    wallClockMs: number;
    agentsSpawned: number;
    teamsSpawned: number;
  }>;
}

export interface AgentCreationUsage {
  mode: AgentNodeCreationMode;
  nodeId?: string;
  definitionFingerprint?: string;
  patternIds: string[];
  cacheHits: string[];
  definitionTokens: number;
  renderedPromptTokens: number;
  renderedPromptChars: number;
}

export interface AgentBindingState {
  tools: ToolBinding[];
  skills: SkillBinding[];
  memoryScope: AgentMemoryScope;
  spawnPolicy: AgentSpawnPolicy;
}

export interface AgentPolicyView extends AgentBindingState {
  agentId: string;
  parentId?: string;
  depth: number;
  currentChildren: number;
  allowedChildren: number;
}

export interface AgentArchetypeProfile {
  archetype: SubAgentArchetype;
  tools: ToolBinding[];
  skills: SkillBinding[];
  spawnPolicy: AgentSpawnPolicy;
}

interface VerifierScorecard {
  reward: number;
  groups: Record<string, number>;
  failureFrontiers?: Record<string, number>;
}

interface WorkspaceMutationCheckpoint {
  path: string;
  previousContent: string;
  baseline?: VerifierScorecard;
  candidateFingerprint: string;
  createdAt: number;
}

interface PersistentVerifierWorkspaceCheckpoint {
  version: 1;
  sessionId: string;
  scorecard: VerifierScorecard;
  files: Array<{
    path: string;
    content: string;
    sha256: string;
  }>;
  correlationId: string;
  verifiedAt: number;
}

interface SharedReadOnlyToolCacheEntry {
  call: ToolCallRecord;
  sourceAgentId: string;
  correlationId?: string;
  cachedAt: number;
}

export class Runtime {
  private static instance: Runtime | null = null;

  private ctx: RuntimeContext | null = null;
  private initialized = false;
  private events: RuntimeEvent[] = [];
  private perTurnUsage: TokenUsage[] = [];
  private agentSequence = 0;
  private delegationSequence = 0;
  private queue: MessageQueue | null = null;
  private scheduler: MessageScheduler | null = null;
  private memory: WorkspaceMemoryManager | null = null;
  private workspaceRoot = process.cwd();
  private agentBindings = new Map<string, AgentBindingState>();
  private workspaceRuntimeConfig: WorkspaceRuntimeConfig | null = null;
  private contextWindowManager: ContextWindowManager | null = null;
  private agentFsms = new Map<string, FSM>();
  private budgetMarket: BudgetMarket | null = null;
  private agentBudgetAllocations = new Map<string, string>();
  private agentBudgetLimits = new Map<string, number>();
  private toolApprovalManager: ToolApprovalManager | null = null;
  private toolCallCounts = new Map<string, number>();
  private readonly failedPathObservations = new Map<string, Map<string, {
    toolName: string;
    path: string;
    error: string;
    actorId: string;
    observedAt: number;
  }>>();
  private readonly changedPathsByCorrelation = new Map<string, Set<string>>();
  private readonly mutationCheckpointsByCorrelation =
    new Map<string, Map<string, WorkspaceMutationCheckpoint>>();
  private readonly resumedExecutionByCorrelation = new Map<string, ExecutionResumeState>();
  private runtimeToolOverrides = new Map<string, Tool>();
  private readonly teams = new TeamRegistry();
  private teamMemberPlans = new Map<string, TeamMemberSpec[]>();
  private teamToolEvidenceCache = new Map<string, ToolCallRecord[]>();
  private teamToolEvidenceCorrelations = new Map<string, string>();
  private readonly sharedReadOnlyToolResultCache =
    new Map<string, SharedReadOnlyToolCacheEntry>();
  private readonly sharedReadOnlyToolRequests =
    new Map<string, Promise<ToolResult>>();
  private teamSpawnReservations = new Map<string, {
    parentId: string;
    correlationId: string;
    baseChildren: number;
    baseTurnAgents: number;
    plannedMembers: number;
    consumedMembers: number;
    allowedChildren: number;
  }>();
  private readonly toolPlanner = new AgentToolPlanner();
  private candidatePlanner: DefaultDelegationCandidatePlanner | null = null;
  private turnAgentCounts = new Map<string, number>();
  private tomPlanner: ToMDelegationEngine = new ToMDelegationPlanner();
  private reasoningInvestmentModel: ReasoningInvestmentModel = new WeightedReasoningInvestmentModel();
  private readonly tomAnalyses = new Map<string, ToMTaskAnalysis>();
  private communicationManager: AgentCommunicationManager | null = null;
  private evolutionSequence = 0;
  private readonly evolutionRuns: EvolutionRunResult[] = [];
  private readonly archivedAgentUsage = new Map<string, TokenUsage>();
  private readonly archivedTeamUsage = new Map<string, TokenUsage>();
  private readonly archivedTeamSynthesisUsage = new Map<string, TokenUsage>();
  private readonly archivedAgentInfo = new Map<string, AgentInfo>();
  private readonly archivedTeamStates = new Map<string, TeamRuntimeState>();
  private readonly lifecycle = new ActorLifecycleRegistry();
  private readonly agentRestoreSpecs = new Map<string, SpawnAgentSpec>();
  private readonly teamRestoreSpecs = new Map<string, SpawnTeamSpec>();
  private readonly evolutionBudgetBypassCorrelations = new Set<string>();
  private readonly executionTrees = new RootExecutionTreeRegistry();
  private readonly executionActivityProjector = new RootExecutionActivityProjector();

  static getInstance(): Runtime {
    if (!Runtime.instance) {
      Runtime.instance = new Runtime();
    }
    return Runtime.instance;
  }

  async initialize(options: RuntimeConfig = {}): Promise<RuntimeContext> {
    if (this.initialized && this.ctx) {
      return this.ctx;
    }

    const startTime = Date.now();

    // Initialize logger
    const logLevel = config.logger?.level ?? 'info';
    logger.setLevel(logLevel as 'debug' | 'info' | 'warn' | 'error');
    await configureLogging();
    logger.info('Runtime initializing...');

    // Create LLM provider
    const llm = options.llmProvider ?? this.createLLMProvider();
    if (llm) {
      logger.info(`LLM provider: ${llm.name}, model: ${llm.defaultModel}`);
    } else {
      logger.warn('No LLM provider configured - agent will have limited functionality');
    }

    // Create FSM
    const fsm = new FSM({
      initialState: 'S_solo',
      strict: true,
      signalBus,
      onTransition: (from, to) => {
        logger.debug(`FSM transition: ${from} -> ${to}`);
        signalBus.emit('fsm:transition', { from, to });
        this.emit({ type: 'fsm.transition', agentId: 'root', data: { from, to } });
      },
      onStateChange: (state) => {
        logger.debug(`FSM state: ${state}`);
        signalBus.emit('fsm:stateChange', { state });
        this.emit({ type: 'fsm.state.changed', agentId: 'root', data: { state } });
      },
      onInvalidTransition: (from, to) => {
        this.emit({ type: 'fsm.invalid_transition', agentId: 'root', data: { from, to } });
      },
    });

    if (options.budget !== undefined) {
      fsm.setBudget(options.budget);
    }

    // Create AgentManager
    const manager = new AgentManager();
    const memory = new WorkspaceMemoryManager();
    await memory.initWorkspace(options.workspaceCwd ?? process.cwd(), options.sessionId ?? 'main');
    this.workspaceRuntimeConfig = await memory.getWorkspaceConfig();
    this.applyExternalWallClockLimit(options.wallClockLimitMs);
    const workspaceRoot = options.workspaceCwd ?? process.cwd();
    this.workspaceRoot = path.resolve(workspaceRoot);
    registerCoreTools({
      web: this.workspaceRuntimeConfig.tools.web,
      shell: this.workspaceRuntimeConfig.tools.shell,
      workspaceRoot,
    });
    this.runtimeToolOverrides.clear();
    this.runtimeToolOverrides.set('fs.list', new FsListTool(workspaceRoot));
    this.runtimeToolOverrides.set('fs.read', new FsReadTool(workspaceRoot));
    this.runtimeToolOverrides.set('fs.search', new FsSearchTool(workspaceRoot));
    this.runtimeToolOverrides.set('fs.replace', new FsReplaceTool(workspaceRoot));
    this.runtimeToolOverrides.set('fs.write', new FsWriteTool(workspaceRoot));
    this.runtimeToolOverrides.set('fs.synthesize', new FsSynthesizeTool());
    this.runtimeToolOverrides.set('shell.exec', new ShellExecTool({
      ...this.workspaceRuntimeConfig.tools.shell,
      workspaceRoot,
    }));
    if (this.workspaceRuntimeConfig.tools.web.enabled) {
      const webConfig = this.workspaceRuntimeConfig.tools.web;
      this.runtimeToolOverrides.set('web.search', new WebSearchTool(webConfig));
      this.runtimeToolOverrides.set('web.fetch', new WebFetchTool(webConfig));
    }
    this.registerCoreSkills();
    this.tomPlanner = options.tomPlanner ?? new ToMDelegationPlanner();
    this.reasoningInvestmentModel = options.reasoningInvestmentModel ?? new WeightedReasoningInvestmentModel();
    const communication = new AgentCommunicationManager(
      this.workspaceRuntimeConfig.communication,
      options.communicationProtocols
    );
    this.communicationManager = communication;
    this.candidatePlanner = new DefaultDelegationCandidatePlanner({
      llm,
      enabledScorers: this.workspaceRuntimeConfig.delegation.candidateScoring.enabledScorers,
      minimumScore: this.workspaceRuntimeConfig.delegation.candidateScoring.minimumScore,
      minimumToMCoverage: this.workspaceRuntimeConfig.tom.minimumCoverage,
      investmentModel: this.reasoningInvestmentModel,
      llmHooks: {
        before: (input, messages, completionOptions) => this.beforeDelegationScorerCall(input, messages, completionOptions),
        after: (completion, input, hookContext) => this.afterDelegationScorerCall(completion, input, hookContext),
        failed: (_error, _input, hookContext) => this.releaseDelegationScorerBudget(hookContext),
      },
    });
    this.budgetMarket = new BudgetMarket(() => this.ctx ? this.getAccountedRuntimeUsedTokens() : 0, {
      mode: this.workspaceRuntimeConfig.budgetMarket.mode,
      minimumGrantTokens: this.workspaceRuntimeConfig.budgetMarket.minimumGrantTokens,
      accountingDimension: this.workspaceRuntimeConfig.budgetMarket.accountingDimension,
      priorityWeights: this.workspaceRuntimeConfig.budgetMarket.priorityWeights,
      investmentModel: this.reasoningInvestmentModel,
    });
    this.budgetMarket.configure(options.budget ?? null);
    this.toolApprovalManager = new ToolApprovalManager(this.workspaceRuntimeConfig.tools.approval);
    const contextWindowManager = new ContextWindowManager(memory, this.workspaceRuntimeConfig.context);
    this.contextWindowManager = contextWindowManager;
    const rootMemory = await memory.loadAgentMemory('roy');
    const rootWindow = await contextWindowManager.build({
      sessionId: options.sessionId ?? 'main',
      agentId: 'root',
      agentKey: 'roy',
      role: 'root',
      task: 'Operate as the root agent for the current Roy runtime session.',
      memoryScope: this.getDefaultMemoryScope('root'),
    });
    const queue = new InMemoryMessageQueue(transition => this.handleQueueTransition(transition));
    const scheduler = new MessageScheduler(queue);

    // Create unified agent
    const agentName = options.agentName ?? 'Roy';
    const agentGoal = options.agentGoal ?? this.buildAgentPromptFromMemory({
      name: agentName,
      role: 'root',
      parentName: 'none',
      task: 'Operate as the root agent for the current Roy runtime session.',
      description: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      bundle: rootMemory,
      publicContext: [rootWindow.publicContext, rootWindow.sessionContext].filter(Boolean).join('\n\n'),
      tomProfile: this.createRootToMProfile(),
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
    });

    const agent = new UnifiedAgent({
      name: agentName,
      goal: agentGoal,
      llm: llm ?? undefined,  // Convert null to undefined for agent
      fsm: options.fsmEnabled !== false ? fsm : undefined,
      id: 'root',
      role: 'root',
      generation: 0,
      tomLevel: 1,
      tomProfile: this.createRootToMProfile(),
      communicationProtocol: communication.getDefaultProtocolId(),
      description: 'Root agent of the Roy autonomous agent system',
      mode: options.mode ?? 'hybrid',
    });

    logger.info(`Agent created: ${agentName} in ${options.mode ?? 'hybrid'} mode`);

    // Register capabilities with agent
    const capabilities = this.registerCapabilities(agent);
    this.agentBindings.set('root', {
      tools: this.getRootToolBindings(),
      skills: this.getRootSkillBindings(),
      memoryScope: this.getDefaultMemoryScope('root'),
      spawnPolicy: this.getDefaultSpawnPolicy('root'),
    });

    // Add agent to manager
    manager.addAgent(agent);
    manager.setInteractWithEnv(agentName);

    // Create main session
    const sessionId = options.sessionId ?? 'main';
    manager.createSession(sessionId);

    const elapsed = Date.now() - startTime;
    logger.info(`Runtime initialized in ${elapsed}ms`);

    this.ctx = {
      config,
      llm,
      fsm,
      signalBus,
      manager,
      agent,
      sessionId,
      queue,
      scheduler,
      memory,
      communication,
      capabilities,
    };
    this.queue = queue;
    this.scheduler = scheduler;
    this.memory = memory;
    this.agentFsms.set('root', fsm);

    this.initialized = true;
    this.emit({ type: 'runtime.initialized', agentId: 'root', data: { sessionId, provider: llm?.name ?? null } });
    return this.ctx;
  }

  private applyExternalWallClockLimit(limitMs: number | undefined): void {
    if (limitMs === undefined) return;
    if (!Number.isSafeInteger(limitMs) || limitMs < 1_000) {
      throw new Error('Runtime wall-clock limit must be an integer of at least 1000ms');
    }
    if (!this.workspaceRuntimeConfig) return;
    const rootSteps = this.workspaceRuntimeConfig.delegation.rootSteps;
    const configuredWallClockMs = rootSteps.maxWallClockMs;
    const appliedWallClockMs = Math.min(configuredWallClockMs, limitMs);
    rootSteps.maxWallClockMs = appliedWallClockMs;
    rootSteps.finalizationReserveMs = Math.min(
      rootSteps.finalizationReserveMs,
      Math.max(1_000, Math.floor(appliedWallClockMs * 0.15))
    );
    rootSteps.executionReserveMs = Math.min(
      rootSteps.executionReserveMs,
      Math.max(1_000, Math.floor(appliedWallClockMs * 0.4))
    );
    this.workspaceRuntimeConfig.tools.executionLoop.maxWallClockMs = Math.min(
      this.workspaceRuntimeConfig.tools.executionLoop.maxWallClockMs,
      Math.max(1_000, appliedWallClockMs - rootSteps.finalizationReserveMs)
    );
    this.emit({
      type: 'runtime.wall_clock_limit.applied',
      agentId: 'root',
      data: {
        requestedWallClockMs: limitMs,
        configuredWallClockMs,
        appliedWallClockMs,
        executionReserveMs: rootSteps.executionReserveMs,
        finalizationReserveMs: rootSteps.finalizationReserveMs,
        toolLoopWallClockMs: this.workspaceRuntimeConfig.tools.executionLoop.maxWallClockMs,
      },
    });
  }

  getContext(): RuntimeContext {
    if (!this.ctx) {
      throw new Error('Runtime not initialized. Call initialize() first.');
    }
    return this.ctx;
  }

  createSession(sessionId: string): void {
    if (!this.ctx) {
      throw new Error('Runtime not initialized');
    }
    this.ctx.manager.createSession(sessionId);
  }

  async shutdown(): Promise<void> {
    if (!this.ctx) return;

    logger.info('Runtime shutting down...');

    for (const sessionId of this.ctx.manager.listSessions()) {
      await this.ctx.manager.closeSession(sessionId);
    }

    await shutdownLogging();

    this.ctx = null;
    this.queue = null;
    this.scheduler = null;
    this.memory = null;
    this.workspaceRuntimeConfig = null;
    this.contextWindowManager = null;
    this.communicationManager = null;
    this.candidatePlanner = null;
    this.agentBindings.clear();
    this.agentFsms.clear();
    this.agentBudgetAllocations.clear();
    this.agentBudgetLimits.clear();
    this.budgetMarket = null;
    this.toolApprovalManager = null;
    this.runtimeToolOverrides.clear();
    this.toolCallCounts.clear();
    this.teams.clear();
    this.teamMemberPlans.clear();
    this.teamToolEvidenceCache.clear();
    this.teamToolEvidenceCorrelations.clear();
    this.sharedReadOnlyToolResultCache.clear();
    this.sharedReadOnlyToolRequests.clear();
    this.teamSpawnReservations.clear();
    this.turnAgentCounts.clear();
    this.tomAnalyses.clear();
    this.evolutionRuns.length = 0;
    this.archivedAgentUsage.clear();
    this.archivedTeamUsage.clear();
    this.archivedTeamSynthesisUsage.clear();
    this.archivedAgentInfo.clear();
    this.archivedTeamStates.clear();
    this.lifecycle.clear();
    this.agentRestoreSpecs.clear();
    this.teamRestoreSpecs.clear();
    this.evolutionBudgetBypassCorrelations.clear();
    this.executionTrees.clear();
    this.evolutionSequence = 0;
    this.initialized = false;
    logger.info('Runtime shutdown complete');
  }

  private createLLMProvider(): LLMProvider | null {
    const llmConfig = config.llm;

    try {
      let provider: LLMProvider;

      if (llmConfig?.provider === 'anthropic') {
        provider = llmFactory.get('anthropic')!;
      } else if (llmConfig?.provider === 'openai') {
        provider = llmFactory.get('openai')!;
      } else if (llmConfig?.provider === 'deepseek') {
        provider = llmFactory.get('deepseek')!;
      } else {
        provider = llmFactory.getDefault();
      }

      if (provider?.isConfigured()) {
        return provider;
      }

      logger.warn('LLM provider not configured');
      return null;
    } catch (error) {
      logger.error('Failed to create LLM provider:', error);
      return null;
    }
  }

  private registerCapabilities(agent: UnifiedAgent, toolNames?: string[]): RuntimeContext['capabilities'] {
    // Register actions
    const actions = actionRegistry.list();
    for (const action of actions) {
      agent.registerAction(action);
      logger.debug(`Registered action: ${action.name}`);
    }

    // Register tools
    const allowedToolNames = toolNames ? new Set(toolNames) : undefined;
    const tools = toolRegistry.list().filter(tool => !allowedToolNames || allowedToolNames.has(tool.name));
    for (const tool of tools) {
      agent.registerTool(tool);
      logger.debug(`Registered tool: ${tool.name}`);
    }

    // Skills are executed via skillRegistry
    const skills = skillRegistry.list();
    logger.debug(`Available ${skills.length} skills`);

    return {
      skills: skills.length,
      actions: actions.length,
      tools: tools.length,
    };
  }

  private registerCoreSkills(): void {
    skillRegistry.unregister('use_tool_when_needed');
    skillRegistry.registerSystem(new DelegateToSubagentSkill(this));
    skillRegistry.register(new UseToolWhenNeededSkill(
      (agentId, toolName, params, reason) => this.executeToolForAgent(agentId, toolName, params, { reason })
    ));
  }

  getAgentArchetypeProfiles(): AgentArchetypeProfile[] {
    const archetypes: SubAgentArchetype[] = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'];
    return archetypes.map(archetype => ({
      archetype,
      tools: this.getDefaultToolBindings(archetype),
      skills: this.getDefaultSkillBindings(archetype),
      spawnPolicy: this.getDefaultSpawnPolicy('subagent', archetype),
    }));
  }

  getAgentPolicy(agentId: string): AgentPolicyView | undefined {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) return undefined;
    const identity = agent.getIdentity();
    const bindings = this.agentBindings.get(agentId) ?? {
      tools: [],
      skills: [],
      memoryScope: this.getDefaultMemoryScope(identity.role),
      spawnPolicy: this.getDefaultSpawnPolicy(identity.role === 'root' ? 'root' : 'subagent'),
    };
    const depth = this.getAgentDepth(agentId);
    return {
      ...bindings,
      agentId,
      parentId: identity.parentId,
      depth,
      currentChildren: this.getChildren(agentId).length,
      allowedChildren: this.computeAllowedChildren(bindings.spawnPolicy),
    };
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  emit(event: Omit<RuntimeEvent, 'timestamp'>): RuntimeEvent {
    const runtimeEvent: RuntimeEvent = {
      ...event,
      timestamp: Date.now(),
    };
    this.events.push(runtimeEvent);
    if (this.events.length > 5000) {
      this.events = this.events.slice(-5000);
    }
    void this.memory?.writeTrace(runtimeEvent).catch(error => {
      logger.warn(`Failed to write runtime trace: ${error instanceof Error ? error.message : String(error)}`);
    });
    return runtimeEvent;
  }

  recordTurnUsage(usage: AgentUsage): void {
    this.perTurnUsage.push(this.toTokenUsage(usage));
  }

  getState(): RuntimeState {
    const ctx = this.getContext();
    const agents = ctx.manager.listAgentInfo();
    const rootAgent = ctx.agent.getInfo();
    return {
      sessionId: ctx.sessionId,
      rootAgentId: rootAgent.identity.id,
      rootAgent,
      agents,
      teams: this.teams.list(),
      events: this.getEvents(),
      budget: this.getBudgetState(),
    };
  }

  getEvents(): RuntimeEvent[] {
    return [...this.events];
  }

  getRootExecutionTree(correlationId?: string): RootExecutionTreeState | undefined {
    return correlationId ? this.executionTrees.get(correlationId) : this.executionTrees.latest();
  }

  async loadRootExecutionTree(correlationId?: string): Promise<RootExecutionTreeState | undefined> {
    const live = this.getRootExecutionTree(correlationId);
    if (live) return live;
    const memory = this.getContext().memory;
    const persisted = correlationId
      ? await memory.readExecutionTree(correlationId)
      : await memory.readLatestExecutionTree();
    return persisted ? this.executionTrees.restore(persisted) : undefined;
  }

  listRootExecutionTrees(): RootExecutionTreeState[] {
    return this.executionTrees.list();
  }

  async listPersistedRootExecutionTrees(sessionId?: string): Promise<Awaited<ReturnType<WorkspaceMemoryManager['listExecutionTrees']>>> {
    return this.getContext().memory.listExecutionTrees(sessionId);
  }

  getToMState(correlationId?: string): ToMRuntimeState {
    const ctx = this.getContext();
    const analyses = correlationId
      ? [this.tomAnalyses.get(correlationId)].filter((item): item is ToMTaskAnalysis => Boolean(item))
      : [...this.tomAnalyses.values()];
    return {
      analyses: analyses.map(analysis => ({
        ...analysis,
        parentBeliefs: [...analysis.parentBeliefs],
        parentGoals: [...analysis.parentGoals],
        parentUncertainties: [...analysis.parentUncertainties],
        gaps: analysis.gaps.map(gap => ({
          ...gap,
          beliefScope: [...gap.beliefScope],
          uncertainty: [...gap.uncertainty],
          requiredCapabilities: [...gap.requiredCapabilities],
          modelsTargets: [...gap.modelsTargets],
        })),
      })),
      agents: [...new Map([
        ...this.archivedAgentInfo.entries(),
        ...ctx.manager.listAgentInfo().map(agent => [agent.identity.id, agent] as const),
      ]).values()].map(agent => ({
        agentId: agent.identity.id,
        name: agent.identity.name,
        parentId: agent.identity.parentId,
        teamId: agent.identity.teamId,
        profile: normalizeToMProfile(agent.identity.tomProfile, agent.identity.tomProfile),
      })),
      teams: [...new Map([
        ...this.archivedTeamStates.entries(),
        ...this.teams.list().map(team => [team.identity.id, team] as const),
      ]).values()].map(team => ({
        teamId: team.identity.id,
        name: team.identity.name,
        parentAgentId: team.identity.parentAgentId,
        profile: normalizeToMProfile(team.identity.tomProfile, team.identity.tomProfile),
      })),
    };
  }

  getBudgetState(): BudgetState {
    const ctx = this.getContext();
    const fsmCtx = ctx.fsm.getContext();
    const agents = ctx.manager.listAgentInfo();
    const perAgent: Record<string, TokenUsage> = {};
    const perTeam: Record<string, TokenUsage> = {};
    let usedTokens = 0;

    for (const agent of agents) {
      const usage = this.toTokenUsage(agent.usage);
      perAgent[agent.identity.id] = usage;
      usedTokens += usage.totalTokens;
    }
    for (const [agentId, usage] of this.archivedAgentUsage) {
      perAgent[agentId] = perAgent[agentId]
        ? this.sumUsage([perAgent[agentId], usage])
        : { ...usage };
      usedTokens += usage.totalTokens;
    }
    for (const team of this.teams.list()) {
      perTeam[team.identity.id] = { ...team.tokenUsage };
      usedTokens += team.synthesisUsage.totalTokens;
    }
    for (const [teamId, usage] of this.archivedTeamUsage) {
      perTeam[teamId] = perTeam[teamId]
        ? this.sumUsage([perTeam[teamId], usage])
        : { ...usage };
      usedTokens += this.archivedTeamSynthesisUsage.get(teamId)?.totalTokens ?? 0;
    }

    return {
      mode: fsmCtx.budget === null ? 'unlimited' : 'limited',
      limitTokens: fsmCtx.budget ?? undefined,
      usedTokens,
      remainingTokens: fsmCtx.budget === null ? undefined : Math.max(0, fsmCtx.budget - usedTokens),
      perAgent,
      perTeam,
      perTurn: [...this.perTurnUsage],
    };
  }

  setBudget(limitTokens: number | null): BudgetState {
    const ctx = this.getContext();
    if (limitTokens === null) {
      ctx.fsm.clearBudget();
      this.emit({ type: 'budget.updated', data: { mode: 'unlimited' } });
    } else {
      ctx.fsm.setBudget(limitTokens);
      this.emit({ type: 'budget.updated', data: { mode: 'limited', limitTokens } });
    }
    this.budgetMarket?.configure(limitTokens);
    return this.getBudgetState();
  }

  getBudgetMarketState(): BudgetMarketState {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    return this.budgetMarket.getState();
  }

  private getAccountedRuntimeUsedTokens(): number {
    const budget = this.getBudgetState();
    const dimension = this.workspaceRuntimeConfig?.budgetMarket.accountingDimension ?? 'total_tokens';
    // Team tokenUsage includes member usage, which is already present in perAgent.
    // Only team synthesis is an additional model call at this aggregation level.
    const usage = [
      ...Object.values(budget.perAgent),
      ...this.teams.list().map(team => team.synthesisUsage),
      ...this.archivedTeamSynthesisUsage.values(),
    ];
    return usage.reduce((sum, item) => {
      if (dimension === 'output_tokens') return sum + item.outputTokens;
      if (dimension === 'thinking_tokens') {
        return sum + (item.thinkingAccountingTokens ?? item.thinkingTokens ?? item.totalTokens);
      }
      return sum + item.totalTokens;
    }, 0);
  }

  rebalanceBudgetMarket(): BudgetRebalanceResult {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const result = this.budgetMarket.rebalance();
    this.emit({
      type: 'budget.rebalanced',
      data: {
        changed: result.changed.map(item => item.id),
        releasedTokens: result.releasedTokens,
        reservedTokens: result.reservedTokens,
      },
    });
    return result;
  }

  getBudgetAllocation(allocationId: string): BudgetAllocation | undefined {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    return this.budgetMarket.getAllocation(allocationId);
  }

  allocateBudget(request: BudgetRequest): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    this.emit({
      type: 'budget.requested',
      agentId: request.requesterId,
      correlationId: request.correlationId,
      data: { ...request },
    });
    const allocation = this.budgetMarket.request(request);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.allocated' : 'budget.denied',
      agentId: request.requesterId,
      correlationId: request.correlationId,
      data: {
        allocationId: allocation.id,
        requestedTokens: request.requestedTokens,
        allocatedTokens: allocation.allocatedTokens,
        policy: allocation.policy,
        score: allocation.score,
        rationale: allocation.rationale,
      },
    });
    return allocation;
  }

  allocateBudgets(requests: BudgetRequest[]): BudgetAllocation[] {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    for (const request of requests) {
      this.emit({ type: 'budget.requested', agentId: request.requesterId, correlationId: request.correlationId, data: { ...request } });
    }
    const allocations = this.budgetMarket.requestMany(requests);
    for (const allocation of allocations) {
      this.emit({
        type: allocation.status === 'granted' ? 'budget.allocated' : 'budget.denied',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: {
          allocationId: allocation.id,
          requestedTokens: allocation.request.requestedTokens,
          allocatedTokens: allocation.allocatedTokens,
          policy: allocation.policy,
          score: allocation.score,
          rationale: allocation.rationale,
        },
      });
    }
    return allocations;
  }

  consumeBudget(allocationId: string, usage: number | ModelTokenUsage): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.consume(allocationId, usage);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.consumed',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        consumedTokens: allocation.consumedTokens,
        utilization: allocation.utilization,
      },
    });
    if (allocation.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens },
      });
    }
    return allocation;
  }

  settleBudget(allocationId: string, usage: number | ModelTokenUsage): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.settle(allocationId, usage);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.settled',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        allocatedTokens: allocation.allocatedTokens,
        consumedTokens: allocation.consumedTokens,
        utilization: allocation.utilization,
        efficiency: allocation.efficiency,
        status: allocation.status,
      },
    });
    if (allocation.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId: allocation.request.requesterId,
        correlationId: allocation.request.correlationId,
        data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens },
      });
    }
    return allocation;
  }

  recordBudgetOutcome(allocationId: string, outcome: BudgetOutcome): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.recordOutcome(allocationId, outcome);
    if (!allocation) throw new Error(`Budget allocation "${allocationId}" not found or cannot accept an outcome`);
    this.emit({
      type: 'budget.outcome.recorded',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: {
        allocationId,
        success: allocation.outcome?.success,
        realizedUtility: allocation.outcome?.realizedUtility,
        efficiency: allocation.efficiency,
      },
    });
    return allocation;
  }

  releaseBudget(allocationId: string, reason = 'released_by_controller'): BudgetAllocation {
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const allocation = this.budgetMarket.release(allocationId, reason);
    if (!allocation) throw new Error(`Active budget allocation "${allocationId}" not found`);
    this.emit({
      type: 'budget.released',
      agentId: allocation.request.requesterId,
      correlationId: allocation.request.correlationId,
      data: { allocationId, reason },
    });
    return allocation;
  }

  getTeams(): TeamRuntimeState[] {
    return this.teams.list();
  }

  getTeam(teamId: string): TeamRuntimeState | undefined {
    return this.teams.get(teamId);
  }

  getTeamState(teamId: string): TeamRuntimeState | undefined {
    return this.getTeam(teamId);
  }

  getTeamTree(teamId: string): TeamTreeNode | undefined {
    const team = this.teams.get(teamId);
    if (!team) return undefined;
    const ctx = this.getContext();
    return {
      team,
      members: team.memberAgentIds
        .map(agentId => ctx.manager.getAgentById(agentId)?.getInfo())
        .filter((agent): agent is AgentInfo => Boolean(agent))
        .map(agent => this.buildAgentTree(agent)),
    };
  }

  getTeamActorTree(): RuntimeActorTree {
    return {
      root: this.getContext().agent.getInfo(),
      teams: this.teams.list()
        .map(team => this.getTeamTree(team.identity.id))
        .filter((team): team is TeamTreeNode => Boolean(team)),
      hierarchy: this.buildRuntimeAgentActorTree(this.getContext().agent.getInfo(), new Set()),
    };
  }

  async spawnTeam(spec: SpawnTeamSpec): Promise<TeamRuntimeState> {
    const ctx = this.getContext();
    const parentAgentId = spec.parentAgentId ?? 'root';
    const correlationId = spec.correlationId ?? this.createCorrelationId();
    const request = await this.enqueueMessage({
      kind: 'team.create.request',
      sessionId: ctx.sessionId,
      from: parentAgentId,
      to: 'runtime',
      correlationId,
      payload: { ...spec, parentAgentId },
      metadata: { agentId: parentAgentId, tomLevel: spec.tomLevel ?? 2 },
    });
    await this.processQueuedMessage(request.id);

    let createdTeamId: string | undefined;
    try {
      if (this.workspaceRuntimeConfig?.teams.enabled === false) throw new Error('Subteams are disabled by workspace policy');
      const parent = ctx.manager.getAgentById(parentAgentId);
      if (!parent) throw new Error(`Parent agent "${parentAgentId}" not found`);
      if (!spec.name?.trim()) throw new Error('Team name is required');
      if (!spec.description?.trim()) throw new Error('Team description is required');
      if (spec.task !== undefined && (typeof spec.task !== 'string' || !spec.task.trim())) {
        throw new Error('Team task must be a non-empty string when provided');
      }
      if (spec.tomLevel !== undefined
        && (!Number.isInteger(spec.tomLevel) || spec.tomLevel < 0 || spec.tomLevel > 3)) {
        throw new Error('Team tomLevel must be an integer from 0 to 3');
      }
      const teamKey = this.safeAgentKey(spec.name);
      const cachedTeamPattern = (await ctx.memory.getCachePatterns('teams'))
        .find(item => item.id === `team_pattern_${teamKey}_v1` || item.key === teamKey);
      const cachedMembers = Array.isArray(cachedTeamPattern?.members)
        ? cachedTeamPattern.members as TeamMemberSpec[]
        : [];
      const cachedExecutionPolicy = cachedTeamPattern?.executionPolicy
        && typeof cachedTeamPattern.executionPolicy === 'object'
        ? cachedTeamPattern.executionPolicy as Partial<TeamExecutionPolicy>
        : {};
      const synthesisPolicy = spec.synthesisPolicy
        ?? (typeof cachedTeamPattern?.synthesisPolicy === 'string' ? cachedTeamPattern.synthesisPolicy : undefined);
      const configuredTeamPolicy = this.workspaceRuntimeConfig?.teams;
      const executionPolicy = normalizeTeamExecutionPolicy({
        mode: configuredTeamPolicy?.executionMode,
        failureMode: configuredTeamPolicy?.failureMode,
        maxConcurrency: configuredTeamPolicy?.maxConcurrency,
        minimumSuccessfulMembers: configuredTeamPolicy?.minimumSuccessfulMembers,
        ...cachedExecutionPolicy,
        ...spec.executionPolicy,
      });
      const requestedMembers = spec.members ?? cachedMembers;
      for (const member of requestedMembers) {
        if (!member || typeof member !== 'object') throw new Error('Every planned team member must be an object');
        if (!this.isValidArchetype(member.archetype)) throw new Error(`Unsupported team member archetype "${member.archetype}"`);
        if (typeof member.task !== 'string' || !member.task.trim()) throw new Error('Every planned team member requires a task');
        if (member.name !== undefined && (typeof member.name !== 'string' || !member.name.trim())) {
          throw new Error('Team member name must be a non-empty string when provided');
        }
        if (member.tools !== undefined
          && (!Array.isArray(member.tools) || member.tools.some(tool => typeof tool !== 'string' || !tool.trim()))) {
          throw new Error('Team member tools must be an array of non-empty strings');
        }
        if (member.skills !== undefined
          && (!Array.isArray(member.skills) || member.skills.some(skill => typeof skill !== 'string' || !skill.trim()))) {
          throw new Error('Team member skills must be an array of non-empty strings');
        }
        if (member.budgetTokens !== undefined
          && (!Number.isFinite(member.budgetTokens) || member.budgetTokens <= 0)) {
          throw new Error('Team member budgetTokens must be a positive number when provided');
        }
        if (member.tomLevel !== undefined
          && (!Number.isInteger(member.tomLevel) || member.tomLevel < 0 || member.tomLevel > 3)) {
          throw new Error('Team member tomLevel must be an integer from 0 to 3');
        }
      }
      const members = requestedMembers.map(member => ({
        ...member,
        tools: member.tools ?? this.getDefaultToolBindings(member.archetype).map(binding => binding.name),
        skills: member.skills ?? this.getDefaultSkillBindings(member.archetype).map(binding => binding.name),
        tomLevel: member.tomLevel ?? this.createSubagentToMProfile(
          member.archetype,
          '',
          member.task,
          parentAgentId
        ).level,
        tomProfile: member.tomProfile ?? this.createSubagentToMProfile(
          member.archetype,
          member.name ?? member.archetype,
          member.task,
          parentAgentId
        ),
      }));
      const maxMembers = this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5;
      if (members.length > maxMembers) {
        throw new Error(`Team member limit exceeded: requested ${members.length}, maximum ${maxMembers}`);
      }
      if (members.length > 0 && executionPolicy.minimumSuccessfulMembers > members.length) {
        throw new Error(
          `Team minimumSuccessfulMembers ${executionPolicy.minimumSuccessfulMembers} exceeds planned members ${members.length}`
        );
      }
      const parentPolicy = this.getAgentPolicy(parentAgentId);
      if (!parentPolicy?.spawnPolicy.canSpawn) {
        throw new Error(`Agent "${parentAgentId}" is not authorized to create subteams`);
      }
      const parentFsmState = parentAgentId === 'root'
        ? ctx.fsm.getState()
        : this.agentFsms.get(parentAgentId)?.getState();
      if (!parentFsmState || !parentPolicy.spawnPolicy.allowedStates.includes(parentFsmState)) {
        throw new Error(`Agent "${parentAgentId}" cannot create a subteam in FSM state "${parentFsmState ?? 'unknown'}"`);
      }
      const reservedChildren = this.getOutstandingTeamReservations(parentAgentId);
      const reservedTurnAgents = this.getOutstandingTeamReservations(undefined, correlationId);
      const reservableChildren = Math.max(
        0,
        Math.min(parentPolicy.allowedChildren, parentPolicy.spawnPolicy.maxChildren)
          - parentPolicy.currentChildren
          - reservedChildren
      );
      const reservableTurnAgents = Math.max(
        0,
        this.getRemainingTotalAgentsForTurn(parentAgentId, correlationId) - reservedTurnAgents
      );
      const reservableMembers = Math.min(reservableChildren, reservableTurnAgents);
      if (members.length > reservableMembers) {
        throw new Error(
          `Team capacity exceeded: planned ${members.length} members, but only ${reservableMembers} `
          + `reserved child slots are available for parent "${parentAgentId}" in this turn`
        );
      }
      const teamTomProfile = spec.tomProfile ?? this.tomPlanner.createTeamProfile({
        teamId: 'pending-team',
        parentId: parentAgentId,
        task: spec.task ?? spec.description,
        members,
      });
      const team = this.teams.create({
        id: spec.instanceId,
        name: spec.name,
        parentAgentId,
        description: spec.description,
        generation: parent.getIdentity().generation + 1,
        tomLevel: teamTomProfile.level,
        tomProfile: teamTomProfile,
        leadAgentId: spec.leadAgentId,
        task: spec.task,
        synthesisPolicy,
        correlationId,
        executionPolicy: { ...executionPolicy },
      });
      createdTeamId = team.identity.id;
      if (members.length > 0) {
        this.teamSpawnReservations.set(team.identity.id, {
          parentId: parentAgentId,
          correlationId,
          baseChildren: parentPolicy.currentChildren,
          baseTurnAgents: this.getTurnAgentCount(correlationId),
          plannedMembers: members.length,
          consumedMembers: 0,
          allowedChildren: parentPolicy.currentChildren + members.length,
        });
        this.emit({
          type: 'team.capacity.reserved',
          agentId: team.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: team.identity.id,
            parentAgentId,
            plannedMembers: members.length,
            baseChildren: parentPolicy.currentChildren,
            allowedChildren: parentPolicy.currentChildren + members.length,
            remainingUnreservedChildren: reservableMembers - members.length,
          },
        });
      }
      this.teamMemberPlans.set(team.identity.id, members.map(member => ({ ...member })));
      const lifecycleOrigin = spec.lifecycleOrigin ?? 'manual';
      const lifecyclePolicy = this.resolveLifecyclePolicy(
        lifecycleOrigin,
        spec.lifecycle ?? this.inheritParentLifecyclePolicy(parentAgentId, lifecycleOrigin)
      );
      this.lifecycle.register({
        actorId: team.identity.id,
        actorKind: 'team',
        origin: lifecycleOrigin,
        parentId: parentAgentId,
        policy: lifecyclePolicy,
        createdAt: team.createdAt,
      });
      this.teamRestoreSpecs.set(team.identity.id, {
        parentAgentId,
        name: spec.name,
        description: spec.description,
        tomLevel: team.identity.tomLevel,
        tomProfile: team.identity.tomProfile,
        tomAnalysis: spec.tomAnalysis,
        leadAgentId: spec.leadAgentId,
        task: spec.task,
        synthesisPolicy,
        members: members.map(member => ({ ...member })),
        correlationId: spec.correlationId,
        executionPolicy: { ...executionPolicy },
        lifecycle: { ...lifecyclePolicy },
        lifecycleOrigin,
        instanceId: team.identity.id,
      });
      await ctx.memory.ensureTeamMemory(teamKey, { name: spec.name, purpose: spec.description });
      if (cachedTeamPattern) {
        this.emit({
          type: 'cache.hit',
          agentId: parentAgentId,
          sessionId: ctx.sessionId,
          correlationId,
          data: { cacheType: 'team-pattern', patternId: cachedTeamPattern.id, teamKey },
        });
      }
      const pattern = await ctx.memory.upsertTeamPattern({
        key: teamKey,
        name: spec.name,
        purpose: spec.description,
        parentId: parentAgentId,
        memberArchetypes: members.map(member => member.archetype),
        tomLevel: team.identity.tomLevel,
        tomProfile: team.identity.tomProfile,
        tomAnalysis: spec.tomAnalysis,
        leadArchetype: members.find(member => member.lead)?.archetype,
        members: members.map(member => ({
          archetype: member.archetype,
          name: member.name,
          role: member.role,
          task: member.task,
          tools: member.tools,
          skills: member.skills,
          tomLevel: member.tomLevel,
          tomProfile: member.tomProfile,
          cognitiveGapIds: member.cognitiveGapIds,
          existenceReason: member.existenceReason,
          systemPrompt: member.systemPrompt,
          lead: member.lead ?? false,
        })),
        executionPolicy: { ...executionPolicy },
        synthesisPolicy,
      });
      await ctx.memory.writeTeamTopology(teamKey, {
        type: 'subteam',
        teamId: team.identity.id,
        parentAgentId,
        leadAgentId: spec.leadAgentId,
        members: [],
        plannedMembers: members,
        tomLevel: team.identity.tomLevel,
        tomProfile: team.identity.tomProfile,
        tomAnalysis: spec.tomAnalysis,
        executionPolicy,
        synthesisPolicy,
        updatedAt: new Date().toISOString(),
      });
      const approved = await this.enqueueMessage({
        kind: 'team.create.approved',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: parentAgentId,
        correlationId,
        parentMessageId: request.id,
        payload: team,
        metadata: { agentId: parentAgentId, teamId: team.identity.id, tomLevel: team.identity.tomLevel },
      });
      await this.processQueuedMessage(approved.id);
      await ctx.queue.ack(request.id);
      await ctx.queue.ack(approved.id);
      this.emit({
        type: 'team.created',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          name: spec.name,
          description: spec.description,
          tomLevel: team.identity.tomLevel,
          tomProfile: team.identity.tomProfile,
          plannedMembers: members.length,
          patternId: pattern.id,
          parentAgentId,
          executionPolicy,
          cognitiveGapIds: team.identity.tomProfile.cognitiveGaps,
          perspective: team.identity.tomProfile.perspective,
        },
      });
      this.emit({
        type: 'tom.team.profile.created',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          parentAgentId,
          profile: team.identity.tomProfile,
          analysisId: spec.tomAnalysis?.id,
        },
      });
      return this.teams.get(team.identity.id)!;
    } catch (error) {
      const failure = error instanceof Error ? error : new Error(String(error));
      const current = await ctx.queue.getMessage(request.id);
      if (current?.status === 'processing' || current?.status === 'pending') {
        await ctx.queue.fail(request.id, failure);
      }
      if (createdTeamId) {
        try {
          await this.transitionTeamFsm(createdTeamId, 'S_team_failed', { error: failure.message });
        } catch {
          // The original creation failure is the actionable error.
        }
        this.lifecycle.remove(createdTeamId);
        this.teamRestoreSpecs.delete(createdTeamId);
        this.releaseTeamSpawnReservation(createdTeamId, 'team_creation_failed');
      }
      const rejected = await this.enqueueMessage({
        kind: 'team.create.rejected',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: parentAgentId,
        correlationId,
        parentMessageId: request.id,
        payload: { parentAgentId, teamId: createdTeamId, reason: failure.message },
        metadata: { agentId: createdTeamId ?? parentAgentId, teamId: createdTeamId },
      });
      await this.processQueuedMessage(rejected.id);
      await ctx.queue.ack(rejected.id);
      this.emit({
        type: createdTeamId ? 'team.create.failed' : 'team.create.rejected',
        agentId: createdTeamId ?? parentAgentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: createdTeamId, parentAgentId, error: failure.message },
      });
      throw failure;
    }
  }

  async createSubteam(input: {
    parentId: string;
    name: string;
    purpose: string;
    memberArchetypes: SubAgentArchetype[];
    correlationId?: string;
  }): Promise<TeamRuntimeState> {
    return this.spawnTeam({
      parentAgentId: input.parentId,
      name: input.name,
      description: input.purpose,
      members: input.memberArchetypes.map(archetype => ({ archetype, task: input.purpose })),
      correlationId: input.correlationId,
    });
  }

  async spawnAgentIntoTeam(teamId: string, spec: TeamMemberSpec): Promise<TeamRuntimeState> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    if (team.status === 'done' || team.status === 'failed') {
      throw new Error(`Team "${teamId}" cannot add members while status is ${team.status}`);
    }
    const maxMembers = this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5;
    const plans = this.teamMemberPlans.get(teamId) ?? [];
    if (team.memberAgentIds.length + plans.length >= maxMembers) {
      throw new Error(`Team "${teamId}" already has the maximum ${maxMembers} members`);
    }
    if (!this.isValidArchetype(spec.archetype)) throw new Error(`Unsupported team member archetype "${spec.archetype}"`);
    if (!spec.task.trim()) throw new Error('Team member task is required');
    const normalizedSpec: TeamMemberSpec = {
      ...spec,
      tools: spec.tools ?? this.getToolBindingsForTask(spec.archetype, spec.task).map(binding => binding.name),
      skills: spec.skills ?? this.getDefaultSkillBindings(spec.archetype).map(binding => binding.name),
      tomLevel: spec.tomLevel ?? this.createSubagentToMProfile(
        spec.archetype,
        '',
        spec.task,
        team.identity.parentAgentId
      ).level,
      tomProfile: spec.tomProfile ?? this.createSubagentToMProfile(
        spec.archetype,
        spec.name ?? spec.archetype,
        spec.task,
        team.identity.parentAgentId
      ),
    };
    const nextPlans = [...plans, normalizedSpec];
    this.teamMemberPlans.set(teamId, nextPlans);
    const restoreSpec = this.teamRestoreSpecs.get(teamId);
    if (restoreSpec) this.teamRestoreSpecs.set(teamId, { ...restoreSpec, members: nextPlans.map(member => ({ ...member })) });
    await this.getContext().memory.updateTeamPatternMembers(this.safeAgentKey(team.identity.name), {
      memberArchetypes: nextPlans.map(member => member.archetype),
      leadArchetype: nextPlans.find(member => member.lead)?.archetype,
      tomLevel: team.identity.tomLevel,
      members: nextPlans.map(member => ({
        archetype: member.archetype,
        name: member.name,
        role: member.role,
        task: member.task,
        tools: member.tools,
        skills: member.skills,
        tomLevel: member.tomLevel,
        tomProfile: member.tomProfile,
        cognitiveGapIds: member.cognitiveGapIds,
        existenceReason: member.existenceReason,
        lead: member.lead ?? false,
      })),
    });
    await this.persistTeamTopology(team);
    this.emit({
      type: 'team.member.planned',
      agentId: team.identity.id,
      sessionId: this.getContext().sessionId,
      correlationId: team.correlationId,
      data: { teamId, archetype: spec.archetype, name: spec.name, task: spec.task, lead: spec.lead ?? false },
    });
    return this.teams.get(teamId)!;
  }

  private async executeTeamMember(
    teamId: string,
    spec: TeamMemberSpec,
    recursiveDelegation: boolean
  ): Promise<RootMediatedSpawnResult> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    if (team.fsmState !== 'S_member_execute') {
      throw new Error(`Team "${teamId}" cannot execute a member in FSM state "${team.fsmState}"`);
    }
    const effectiveTask = this.buildTeamMemberTaskWithStepCache(team, spec.task);
    if (effectiveTask !== spec.task) {
      this.emit({
        type: 'team.member.step_cache.injected',
        agentId: teamId,
        sessionId: this.getContext().sessionId,
        correlationId: team.correlationId,
        data: {
          teamId,
          memberName: spec.name,
          completedMembers: Object.keys(team.memberResults).length,
          failedMembers: Object.keys(team.memberErrors).length,
        },
      });
    }
    const membersBefore = new Set(team.memberAgentIds);
    const memberSkills = Array.from(new Set([
      ...(spec.skills ?? this.getDefaultSkillBindings(spec.archetype).map(binding => binding.name)),
      ...(recursiveDelegation ? ['delegate_to_subagent'] : []),
    ])).filter(skill => recursiveDelegation || skill !== 'delegate_to_subagent');
    let execution: Awaited<ReturnType<Runtime['createAgentComputeNode']>>;
    try {
      execution = await this.createAgentComputeNode({
        parentId: team.identity.parentAgentId,
        archetype: spec.archetype,
        task: effectiveTask,
        intentTask: spec.task,
        name: spec.name,
        role: spec.role,
        style: spec.style,
        tools: spec.tools,
        skills: memberSkills,
        budgetTokens: spec.budgetTokens,
        memoryScope: spec.memoryScope,
        tomProfile: spec.tomProfile ?? (spec.tomLevel === undefined
          ? undefined
          : {
            ...this.createSubagentToMProfile(spec.archetype, '', spec.task, team.identity.parentAgentId),
            level: spec.tomLevel as ToMProfile['level'],
          }),
        tomProfileMode: 'runtime_assignment',
        cognitiveGapIds: spec.cognitiveGapIds,
        existenceReason: spec.existenceReason,
        systemPrompt: spec.systemPrompt,
        execution: {
          requireParentSynthesis: false,
          showSubagentOutput: false,
          disableRecursiveDelegation: !recursiveDelegation,
          teamId,
        },
        lifecycleOrigin: 'team_member',
      }, {
        agentId: team.identity.parentAgentId,
        sessionId: this.getContext().sessionId,
        source: teamId,
      }, team.correlationId);
    } catch (error) {
      const createdAgentId = this.teams.get(teamId)?.memberAgentIds.find(agentId => !membersBefore.has(agentId));
      const failure = error instanceof Error ? error : new Error(String(error));
      if (createdAgentId) Object.assign(failure, { teamMemberAgentId: createdAgentId });
      throw failure;
    }
    const result = execution.delegation;
    const updated = this.teams.addMember(teamId, result.agent.identity.id, spec.task, spec.lead);
    this.teams.recordMemberResult(
      teamId,
      result.agent.identity.id,
      spec.task,
      result.subagentResult.result,
      result.subagentResult.usage
    );
    await this.persistTeamTopology(updated);
    this.emit({
      type: 'team.member.completed',
      agentId: result.agent.identity.id,
      sessionId: this.getContext().sessionId,
      correlationId: result.correlationId,
      data: {
        teamId,
        parentAgentId: team.identity.parentAgentId,
        task: spec.task,
        totalTokens: result.subagentResult.usage.totalTokens,
      },
    });
    return result;
  }

  private buildTeamMemberTaskWithStepCache(team: TeamRuntimeState, task: string): string {
    const boundedTask = this.compactDelegatedTask(task);
    const completed = Object.entries(team.memberResults);
    const failed = Object.entries(team.memberErrors);
    if (completed.length === 0 && failed.length === 0) return boundedTask;
    return [
      boundedTask,
      '<team_step_cache>',
      JSON.stringify({
        teamId: team.identity.id,
        taskReference: 'Use the immutable assigned task above; this cache contains deltas only.',
        completedMembers: completed.slice(-8).map(([agentId, result]) => ({
          agentId,
          taskSummary: this.compactTeamMemberTaskSummary(
            team.memberTasks[agentId] ?? '',
            team.task
          ),
          result: result.slice(0, 4000),
        })),
        failedMembers: failed.slice(-8).map(([memberKey, error]) => ({
          memberKey,
          error: error.slice(0, 2000),
        })),
      }, null, 2),
      '</team_step_cache>',
      [
        'Team-step attention:',
        '- Treat grounded prior member evidence as shared state, but independently verify it when your role requires verification.',
        '- Consume failure output and do not repeat an equivalent failed path without a changed hypothesis.',
        '- Add new authoritative paths, mutations, verification evidence, and unresolved feedback for team synthesis.',
        '- Escalate a newly exposed gap in your result; runtime coordination handles follow-up structure only when explicitly assigned.',
      ].join('\n'),
    ].join('\n\n');
  }

  private compactTeamMemberTaskSummary(memberTask: string, teamTask?: string): string {
    let summary = memberTask.trim();
    if (teamTask?.trim()) {
      summary = summary.replace(teamTask.trim(), '[immutable team task]');
    }
    const originalTaskIndex = summary.search(/\bOriginal task\s*:/i);
    if (originalTaskIndex >= 0) {
      summary = `${summary.slice(0, originalTaskIndex)}Original task: [immutable team task]`;
    }
    return summary.replace(/\s+/g, ' ').slice(0, 500);
  }

  private enrichRepairPlansWithTeamStepEvidence(
    plans: PlannedToolCall[],
    task: string
  ): PlannedToolCall[] {
    const cacheMatch = /<team_step_cache>\s*([\s\S]*?)\s*<\/team_step_cache>/i.exec(
      task
    );
    if (!cacheMatch) return plans;
    let latestResult: string;
    try {
      const cache = JSON.parse(cacheMatch[1]!) as {
        completedMembers?: Array<{ result?: unknown }>;
      };
      const completed = Array.isArray(cache.completedMembers)
        ? cache.completedMembers
        : [];
      latestResult = String(completed.at(-1)?.result ?? '').trim();
    } catch {
      return plans;
    }
    if (!latestResult) return plans;
    return plans.map(plan => {
      if (plan.toolName !== 'fs.synthesize') return plan;
      const instructions = String(plan.params.instructions ?? '').trim();
      if (!instructions) return plan;
      const targetPath = this.normalizeToolWorkspacePath(String(
        plan.params.path ?? ''
      ));
      const causalEvidence = this.compactTeamStepEvidenceForRepair(
        latestResult,
        targetPath,
        instructions,
        1_200
      );
      if (!causalEvidence || instructions.includes(causalEvidence)) return plan;
      const preface = 'Grounded diagnosis from the immediately preceding sequential team member:';
      const suffix = 'Use this diagnosis only where it agrees with the authoritative current source and executable verifier evidence.';
      const structuralChars = preface.length + suffix.length + 6;
      const contentBudget = Math.max(1, 2_400 - structuralChars);
      const baseBudget = Math.min(
        instructions.length,
        Math.max(900, Math.floor(contentBudget * 0.5))
      );
      const resultBudget = Math.max(1, contentBudget - baseBudget);
      const compact = (
        value: string,
        maxChars: number,
        marker: string,
        headRatio: number
      ): string => {
        if (value.length <= maxChars) return value;
        const available = Math.max(2, maxChars - marker.length - 2);
        const head = Math.floor(available * headRatio);
        return `${value.slice(0, head)}\n${marker}\n${value.slice(-(available - head))}`;
      };
      const causalInstructions = compact(
        instructions,
        baseBudget,
        '[repair contract compacted]',
        0.8
      );
      const causalResult = compact(
        causalEvidence,
        resultBudget,
        '[semantically selected team evidence compacted]',
        0.78
      );
      return {
        ...plan,
        params: {
          ...plan.params,
          instructions: [
            causalInstructions,
            preface,
            causalResult,
            suffix,
          ].join('\n\n'),
        },
      };
    });
  }

  private compactTeamStepEvidenceForRepair(
    report: string,
    targetPath: string,
    instructions: string,
    maxChars: number
  ): string {
    const targetBase = targetPath.slice(targetPath.lastIndexOf('/') + 1)
      .toLowerCase();
    const targetStem = targetBase.replace(/\.[^.]+$/, '');
    const ignored = new Set([
      'apply', 'smallest', 'coherent', 'interface', 'preserving', 'change',
      'resolves', 'newest', 'external', 'verifier', 'feedback',
      'authoritative', 'current', 'source', 'already', 'observed', 'patch',
      'base', 'preserve', 'unrelated', 'working', 'declarations', 'alter',
      'benchmark', 'files', 'reported', 'failure', 'repair', 'implementation',
    ]);
    const terms = new Set<string>();
    if (targetBase) terms.add(targetBase);
    if (targetStem.length >= 3) terms.add(targetStem);
    for (const match of instructions.matchAll(
      /\b[A-Za-z_][A-Za-z0-9_.-]{3,}\b/g
    )) {
      const value = match[0]!.toLowerCase();
      if (!ignored.has(value)) terms.add(value);
      if (terms.size >= 32) break;
    }
    const chunks = report
      .replace(/\r\n/g, '\n')
      .split('\n')
      .flatMap(line => {
        if (line.length <= 480) return [line];
        const parts: string[] = [];
        for (let offset = 0; offset < line.length; offset += 420) {
          parts.push(line.slice(offset, offset + 480));
        }
        return parts;
      })
      .map((text, index) => {
        const lower = text.toLowerCase();
        let score = /\b(?:error|failed|failure|assert|traceback|mismatch|expected|actual)\b/i.test(
          text
        ) ? 2 : 0;
        for (const term of terms) {
          if (lower.includes(term)) score += term === targetBase ? 12 : 4;
        }
        return { text: text.trim(), index, score };
      })
      .filter(item => item.text);
    const selected = chunks
      .filter(item => item.score > 0)
      .sort((left, right) => right.score - left.score || left.index - right.index)
      .slice(0, 8)
      .sort((left, right) => left.index - right.index);
    const evidence = (selected.length > 0 ? selected : chunks.slice(0, 2))
      .map(item => item.text)
      .join('\n');
    if (evidence.length <= maxChars) return evidence;
    return `${evidence.slice(0, Math.max(1, maxChars - 42))}\n[semantic team evidence compacted]`;
  }

  private compactDelegatedTask(task: string, maxChars = 12_000): string {
    if (task.length <= maxChars) return task;
    const marker = '\n\n[runtime_compacted_delegated_task_middle]\n\n';
    const available = Math.max(2_000, maxChars - marker.length);
    const headChars = Math.floor(available * 0.42);
    return `${task.slice(0, headChars)}${marker}${task.slice(-(available - headChars))}`;
  }

  private compactAgentPromptDescription(description: string, task?: string): string {
    const normalizedDescription = description.trim().replace(/\s+/g, ' ');
    const normalizedTask = task?.trim().replace(/\s+/g, ' ');
    if (normalizedTask && (
      normalizedDescription === normalizedTask
      || normalizedDescription.includes(normalizedTask)
    )) {
      return 'Execute the immutable assigned task in the current agent role and return grounded incremental evidence.';
    }
    if (normalizedDescription.length <= 800) return description.trim();
    return `${normalizedDescription.slice(0, 760)} … [description compacted; full assignment is in the task slot]`;
  }

  private agentTaskObservationReference(agentId: string, task: string): string {
    const assignedTask = this.agentRestoreSpecs.get(agentId)?.task?.trim();
    if (!assignedTask || assignedTask !== task.trim()) return task;
    return [
      '[runtime_current_assignment]',
      'Execute the immutable current task defined in the agent system prompt.',
      'Use the runtime evidence and warnings below as the only incremental state for this step.',
    ].join('\n');
  }

  private isolatedAgentTaskObservation(task: string): string {
    return [
      '[runtime_current_assignment]',
      'This completion is isolated from the earlier agent conversation, so the immutable task is repeated below.',
      this.compactDelegatedTask(task, 12_000),
    ].join('\n\n');
  }

  private failedAgentRunProducedWorkspaceMutation(error: unknown): boolean {
    const inspected = new Set<unknown>();
    let current: unknown = error;
    while (current && typeof current === 'object' && !inspected.has(current)) {
      inspected.add(current);
      const candidate = current as {
        runtimeToolCalls?: unknown;
        cause?: unknown;
      };
      if (Array.isArray(candidate.runtimeToolCalls)
        && candidate.runtimeToolCalls.some(call =>
          call
          && typeof call === 'object'
          && isSuccessfulWorkspaceMutationCall(
            call as Parameters<typeof isSuccessfulWorkspaceMutationCall>[0]
          )
        )) {
        return true;
      }
      current = candidate.cause;
    }
    return false;
  }

  async runTeam(
    teamId: string,
    task: string,
    options: { correlationId?: string; memberRecursiveDelegation?: boolean } = {}
  ): Promise<TeamRunResult> {
    const ctx = this.getContext();
    const initial = this.teams.get(teamId);
    if (!initial) throw new Error(`Team "${teamId}" not found`);
    if (!task.trim()) throw new Error('Team task is required');
    const usageBefore = { ...initial.tokenUsage };
    // A team definition can run repeatedly, but every execution is a distinct trace.
    const correlationId = options.correlationId ?? this.createCorrelationId();
    const reservation = this.teamSpawnReservations.get(teamId);
    if (reservation) {
      reservation.correlationId = correlationId;
      reservation.baseTurnAgents = this.getTurnAgentCount(correlationId);
    }
    this.activateActorLifecycle(teamId, correlationId);
    this.teams.setTask(teamId, task, correlationId);
    await this.transitionTeamFsm(teamId, 'S_team_plan', { task });
    this.emit({
      type: 'team.execution.policy.applied',
      agentId: teamId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId, ...initial.executionPolicy },
    });

    const taskMessage = await this.enqueueMessage({
      kind: 'team.task',
      sessionId: ctx.sessionId,
      from: initial.identity.parentAgentId,
      to: teamId,
      correlationId,
      payload: { teamId, task },
      metadata: { agentId: initial.identity.parentAgentId, teamId, tomLevel: initial.identity.tomLevel },
    });
    await this.processQueuedMessage(taskMessage.id);

    type TeamWorkValue = {
      agentId: string;
      result: RunAgentResult;
      execution?: RootMediatedSpawnResult;
    };
    let executionOutcomes: TeamExecutionOutcome<TeamWorkValue>[];
    try {
      const plans = this.teamMemberPlans.get(teamId) ?? [];
      if (plans.length > 0) {
        await this.transitionTeamFsm(teamId, 'S_member_spawn', { count: plans.length });
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: plans.length });
        let diagnosticPhaseFailed = false;
        let implementationPhaseFailed = false;
        let implementationPhaseProducedMutation = false;
        executionOutcomes = await executeTeamItems(plans.map((plan, index) => {
          const key = `planned:${index + 1}:${plan.archetype}`;
          const phase = this.longHorizonMemberPhase(plan);
          return {
            key,
            execute: async (): Promise<TeamWorkValue> => {
              if (phase > 0 && diagnosticPhaseFailed) {
                this.emit({
                  type: 'team.member.dependency_blocked',
                  agentId: teamId,
                  sessionId: ctx.sessionId,
                  correlationId,
                  data: {
                    teamId,
                    memberKey: key,
                    archetype: plan.archetype,
                    reason: 'The required verifier-diagnostic phase failed before producing focused executable evidence.',
                  },
                });
                throw new Error('dependent_repair_blocked_by_failed_diagnostic');
              }
              if (phase === 2
                && implementationPhaseFailed
                && !implementationPhaseProducedMutation) {
                this.emit({
                  type: 'team.member.dependency_blocked',
                  agentId: teamId,
                  sessionId: ctx.sessionId,
                  correlationId,
                  data: {
                    teamId,
                    memberKey: key,
                    archetype: plan.archetype,
                    reason: 'The implementation phase failed before producing mutation-and-verification closure.',
                  },
                });
                throw new Error('dependent_verification_blocked_by_failed_implementation');
              }
              if (phase === 2
                && implementationPhaseFailed
                && implementationPhaseProducedMutation) {
                this.emit({
                  type: 'team.member.partial_handoff.accepted',
                  agentId: teamId,
                  sessionId: ctx.sessionId,
                  correlationId,
                  data: {
                    teamId,
                    memberKey: key,
                    archetype: plan.archetype,
                    reason: 'The implementation member failed after producing an effective workspace mutation; continue with independent verification so the mutation receives authoritative feedback.',
                  },
                });
              }
              this.teams.markMemberRunning(teamId, key);
              this.emit({
                type: 'team.member.started',
                agentId: teamId,
                sessionId: ctx.sessionId,
                correlationId,
                data: { teamId, memberKey: key, archetype: plan.archetype, task: plan.task },
              });
              try {
                const execution = await this.executeTeamMember(
                  teamId,
                  plan,
                  options.memberRecursiveDelegation !== false
                );
                this.teams.clearMemberTracking(teamId, key);
                return {
                  agentId: execution.agent.identity.id,
                  result: execution.subagentResult,
                  execution,
                };
              } catch (error) {
                if (phase === 0 && plan.role === 'verifier-guided diagnostic probe') {
                  diagnosticPhaseFailed = true;
                }
                if (phase === 1) {
                  implementationPhaseFailed = true;
                  implementationPhaseProducedMutation =
                    implementationPhaseProducedMutation
                    || this.failedAgentRunProducedWorkspaceMutation(error);
                }
                this.teams.recordMemberFailure(
                  teamId,
                  key,
                  error instanceof Error ? error.message : String(error)
                );
                throw error;
              }
            },
          };
        }), initial.executionPolicy);
        this.teamMemberPlans.set(teamId, []);
      } else {
        const team = this.teams.get(teamId)!;
        if (team.memberAgentIds.length === 0) throw new Error(`Team "${teamId}" has no members or member plans`);
        await this.transitionTeamFsm(teamId, 'S_member_execute', { count: team.memberAgentIds.length });
        executionOutcomes = await executeTeamItems(team.memberAgentIds.map(agentId => {
          const assignedTask = team.memberTasks[agentId] ?? task;
          return {
            key: agentId,
            execute: async (): Promise<TeamWorkValue> => {
              const currentTeam = this.teams.get(teamId)!;
              const memberTask = this.buildTeamMemberTaskWithStepCache(currentTeam, assignedTask);
              this.teams.markMemberRunning(teamId, agentId);
              this.emit({
                type: 'team.member.started',
                agentId,
                sessionId: ctx.sessionId,
                correlationId,
                data: { teamId, memberKey: agentId, task: memberTask },
              });
              try {
                const result = await this.runAgent(agentId, memberTask, {
                  correlationId,
                  disableRecursiveDelegation: options.memberRecursiveDelegation === false,
                });
                this.teams.recordMemberResult(teamId, agentId, memberTask, result.result, result.usage);
                this.emit({
                  type: 'team.member.completed',
                  agentId,
                  sessionId: ctx.sessionId,
                  correlationId,
                  data: { teamId, task: memberTask, totalTokens: result.usage.totalTokens },
                });
                return { agentId, result };
              } catch (error) {
                this.teams.recordMemberFailure(
                  teamId,
                  agentId,
                  error instanceof Error ? error.message : String(error)
                );
                throw error;
              }
            },
          };
        }), initial.executionPolicy);
      }

      for (const outcome of executionOutcomes) {
        if (outcome.status === 'completed') continue;
        if (outcome.status === 'failed') {
          const failedAgentId = outcome.cause instanceof Error
            && typeof (outcome.cause as Error & { teamMemberAgentId?: unknown }).teamMemberAgentId === 'string'
            ? (outcome.cause as Error & { teamMemberAgentId: string }).teamMemberAgentId
            : undefined;
          const memberKey = failedAgentId ?? outcome.key;
          if (memberKey !== outcome.key) this.teams.clearMemberTracking(teamId, outcome.key);
          const failedUsage = outcome.cause instanceof Error
            ? (outcome.cause as Error & { runtimeUsage?: TokenUsage }).runtimeUsage
            : undefined;
          this.teams.recordMemberFailure(
            teamId,
            memberKey,
            outcome.error ?? 'unknown member execution failure',
            failedUsage
          );
          this.emit({
            type: 'team.member.failed',
            agentId: failedAgentId ?? outcome.value?.agentId ?? teamId,
            sessionId: ctx.sessionId,
            correlationId,
            data: { teamId, memberKey, error: outcome.error },
          });
        } else {
          this.teams.markMemberSkipped(teamId, outcome.key);
          this.emit({
            type: 'team.member.skipped',
            agentId: teamId,
            sessionId: ctx.sessionId,
            correlationId,
            data: { teamId, memberKey: outcome.key, reason: 'fail_fast' },
          });
        }
      }
      const completedOutcomes = executionOutcomes.filter(
        (outcome): outcome is TeamExecutionOutcome<TeamWorkValue> & { status: 'completed'; value: TeamWorkValue } =>
          outcome.status === 'completed' && outcome.value !== undefined
      );
      const failedOutcomes = executionOutcomes.filter(outcome => outcome.status === 'failed');
      if (initial.executionPolicy.failureMode === 'fail_fast' && failedOutcomes.length > 0) {
        throw new Error(`Team member execution failed: ${failedOutcomes[0].error ?? failedOutcomes[0].key}`);
      }
      if (completedOutcomes.length < initial.executionPolicy.minimumSuccessfulMembers) {
        throw new Error(
          `Team completed ${completedOutcomes.length} members, below minimum ${initial.executionPolicy.minimumSuccessfulMembers}`
        );
      }
      const members = completedOutcomes.map(outcome => outcome.value.result);
      const memberExecutions = completedOutcomes
        .map(outcome => outcome.value.execution)
        .filter((execution): execution is RootMediatedSpawnResult => execution !== undefined);

      await this.transitionTeamFsm(teamId, 'S_member_aggregate', {
        completed: members.length,
        failed: failedOutcomes.length,
      });
      await this.transitionTeamFsm(teamId, 'S_team_synthesize', {
        completed: members.length,
        failed: failedOutcomes.length,
      });
      const synthesis = await this.completeAsTeam(
        this.teams.get(teamId)!,
        task,
        members,
        failedOutcomes,
        correlationId
      );
      this.teams.recordSynthesis(teamId, synthesis.content, synthesis.usage);

      const resultMessage = await this.enqueueMessage({
        kind: 'team.result',
        sessionId: ctx.sessionId,
        from: teamId,
        to: initial.identity.parentAgentId,
        correlationId,
        parentMessageId: taskMessage.id,
        payload: {
          teamId,
          task,
          result: synthesis.content,
          memberAgentIds: this.teams.get(teamId)!.memberAgentIds,
        },
        metadata: { agentId: initial.identity.parentAgentId, teamId, tomLevel: initial.identity.tomLevel },
      });
      await this.processQueuedMessage(resultMessage.id);
      await ctx.queue.ack(taskMessage.id);
      await ctx.queue.ack(resultMessage.id);
      await this.transitionTeamFsm(teamId, 'S_team_done', { totalTokens: this.teams.get(teamId)!.tokenUsage.totalTokens });
      const completedTeam = this.teams.get(teamId)!;
      const runUsage = this.subtractTokenUsage(completedTeam.tokenUsage, usageBefore);
      await this.persistTeamRunArtifacts({
        team: completedTeam,
        task,
        result: synthesis.content,
        correlationId,
        usage: runUsage,
        success: true,
      });
      this.emit({
        type: 'team.completed',
        agentId: teamId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId,
          result: synthesis.content,
          memberAgentIds: this.teams.get(teamId)!.memberAgentIds,
          totalTokens: runUsage.totalTokens,
          cumulativeTokens: this.teams.get(teamId)!.tokenUsage.totalTokens,
          parentAgentId: initial.identity.parentAgentId,
          failedMembers: failedOutcomes.length,
          partial: failedOutcomes.length > 0,
        },
      });
      const teamRunResult: TeamRunResult = {
        team: this.teams.get(teamId)!,
        result: synthesis.content,
        members,
        memberExecutions,
        memberOutcomes: executionOutcomes.map(outcome => ({
          key: outcome.key,
          agentId: outcome.value?.agentId ?? (
            outcome.cause instanceof Error
            && typeof (outcome.cause as Error & { teamMemberAgentId?: unknown }).teamMemberAgentId === 'string'
              ? (outcome.cause as Error & { teamMemberAgentId: string }).teamMemberAgentId
              : undefined
          ),
          status: outcome.status,
          error: outcome.error,
        })),
        correlationId,
        messages: await this.getMessages({ correlationId }),
        usage: runUsage,
      };
      await this.finalizeActorLifecycle(teamId, 'success', correlationId);
      this.releaseTeamSpawnReservation(teamId, 'team_run_completed');
      return teamRunResult;
    } catch (error) {
      const current = await ctx.queue.getMessage(taskMessage.id);
      if (current?.status === 'pending' || current?.status === 'processing') {
        await ctx.queue.fail(taskMessage.id, error instanceof Error ? error : new Error(String(error)));
      }
      await this.transitionTeamFsm(teamId, 'S_team_failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      const failedTeam = this.teams.get(teamId)!;
      await this.persistTeamRunArtifacts({
        team: failedTeam,
        task,
        correlationId,
        usage: this.subtractTokenUsage(failedTeam.tokenUsage, usageBefore),
        success: false,
      });
      this.emit({
        type: 'team.failed',
        agentId: teamId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId, parentAgentId: initial.identity.parentAgentId, error: error instanceof Error ? error.message : String(error) },
      });
      await this.finalizeActorLifecycle(teamId, 'failure', correlationId);
      this.releaseTeamSpawnReservation(teamId, 'team_run_failed');
      throw error;
    }
  }

  getToolApprovals(status?: ToolApprovalRequest['status']): ToolApprovalRequest[] {
    if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
    return this.toolApprovalManager.list(status);
  }

  async executeToolForAgent(
    agentId: string,
    toolName: string,
    params: Record<string, unknown>,
    options: {
      reason?: string;
      approvalId?: string;
      correlationId?: string;
      nodeId?: string;
      synthesisTask?: string;
      groundingCalls?: ToolCallRecord[];
    } = {}
  ): Promise<ToolResult> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) return { success: false, error: `Agent "${agentId}" not found` };
    const binding = (this.agentBindings.get(agentId)?.tools ?? []).find(item => item.name === toolName && item.enabled);
    if (!binding) return { success: false, error: `Tool "${toolName}" is not authorized for agent "${agentId}"` };
    const runtimeTool = this.runtimeToolOverrides.get(toolName);
    if (!runtimeTool && !toolRegistry.has(toolName)) return { success: false, error: `Tool "${toolName}" not found` };
    const immutableMutationError = this.immutableRuntimeEvidenceMutationError(toolName, params);
    if (immutableMutationError) {
      this.emit({
        type: 'tool.policy.rejected',
        agentId,
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          toolName,
          params,
          reason: immutableMutationError,
        },
      });
      return {
        success: false,
        error: immutableMutationError,
        metadata: { immutableRuntimeEvidence: true },
      };
    }
    const cachedPathRejection = await this.getCachedInvalidPathRejection(
      agentId,
      toolName,
      params,
      options
    );
    if (cachedPathRejection) return cachedPathRejection;

    const callKey = `${agentId}:${toolName}`;
    const calls = this.toolCallCounts.get(callKey) ?? 0;
    const maxCalls = binding.constraints?.maxCalls;
    if (maxCalls !== undefined && calls >= maxCalls) {
      return { success: false, error: `Tool call limit reached for ${toolName}`, metadata: { maxCalls } };
    }

    let approved = false;
    if (options.approvalId) {
      const prior = this.toolApprovalManager?.get(options.approvalId);
      approved = prior?.status === 'approved' && prior.agentId === agentId && prior.toolName === toolName;
    }
    if (!approved) {
      if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
      const authorization = this.toolApprovalManager.authorize({
        agentId,
        toolName,
        permission: binding.permission,
        params,
        reason: options.reason,
      });
      const approvalMessage = await this.enqueueMessage({
        kind: 'tool.approval.request',
        sessionId: ctx.sessionId,
        from: agentId,
        to: 'runtime.approval',
        correlationId: options.correlationId,
        payload: authorization.request,
        metadata: { agentId, nodeId: options.nodeId },
      });
      await this.processQueuedMessage(approvalMessage.id);
      await ctx.queue.ack(approvalMessage.id);
      this.emit({
        type: 'tool.approval.requested',
        agentId,
        data: {
          approvalId: authorization.request.id,
          toolName,
          permission: binding.permission,
          decision: authorization.decision,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
        },
      });
      if (authorization.decision !== 'pending') {
        const resolvedMessage = await this.enqueueMessage({
          kind: 'tool.approval.resolved',
          sessionId: ctx.sessionId,
          from: 'runtime.approval',
          to: agentId,
          correlationId: options.correlationId,
          parentMessageId: approvalMessage.id,
          payload: authorization.request,
          metadata: { agentId, nodeId: options.nodeId },
        });
        await this.processQueuedMessage(resolvedMessage.id);
        await ctx.queue.ack(resolvedMessage.id);
      }
      if (authorization.decision === 'pending') {
        return {
          success: false,
          error: `Tool approval required for ${toolName}`,
          metadata: { pendingApproval: true, approvalId: authorization.request.id },
        };
      }
      if (authorization.decision === 'denied') {
        return { success: false, error: `Tool policy denied ${toolName}`, metadata: { approvalId: authorization.request.id } };
      }
    }

    const toolCall = await this.enqueueMessage({
      kind: 'tool.call',
      sessionId: ctx.sessionId,
      from: agentId,
      to: `tool.${toolName}`,
      correlationId: options.correlationId,
      payload: { toolName, params, reason: options.reason },
      metadata: { agentId, nodeId: options.nodeId },
    });
    await this.processQueuedMessage(toolCall.id);
    this.emit({
      type: 'tool.call',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: { toolName, params, correlationId: options.correlationId },
    });
    const mutationCheckpoint = await this.captureWorkspaceMutationCheckpoint(
      toolName,
      params,
      options.groundingCalls ?? [],
      options.correlationId ?? ctx.sessionId
    );
    let result = toolName === 'fs.synthesize'
      ? await this.executeSynthesizedFileForAgent(
        agent,
        params,
        options.synthesisTask ?? this.agentRestoreSpecs.get(agentId)?.task ?? '',
        options.groundingCalls ?? [],
        options.correlationId ?? ctx.sessionId
      )
      : runtimeTool
        ? await this.executeRuntimeTool(runtimeTool, params)
        : await toolRegistry.execute(toolName, params);
    if (result.success && mutationCheckpoint) {
      const transactionId = options.correlationId ?? ctx.sessionId;
      const checkpoints = this.mutationCheckpointsByCorrelation.get(transactionId)
        ?? new Map<string, WorkspaceMutationCheckpoint>();
      const existing = checkpoints.get(mutationCheckpoint.path);
      if (!existing) {
        checkpoints.set(mutationCheckpoint.path, mutationCheckpoint);
        this.mutationCheckpointsByCorrelation.set(transactionId, checkpoints);
      } else {
        this.emit({
          type: 'workspace.mutation.checkpoint.retained',
          agentId,
          correlationId: transactionId,
          data: {
            path: existing.path,
            reason: 'preserve_accepted_snapshot_across_unverified_mutation_chain',
            checkpointAgeMs: Date.now() - existing.createdAt,
          },
        });
      }
    }
    if (toolName === 'shell.exec'
      && isWorkspaceVerificationCall({
        toolName,
        params,
        success: result.success,
      })
      && this.shouldAttachVerifierDiagnostics(String(params.command ?? ''))) {
      result = await this.attachVerifierDiagnostics(
        result,
        agentId,
        options.correlationId
      );
      result = await this.rollbackVerifierRegression(
        result,
        agentId,
        options.correlationId ?? ctx.sessionId
      );
      const persistentRecovery = (
        result.result as {
          persistentCheckpointRecovery?: {
            restored?: unknown;
            expectedReward?: unknown;
          };
        } | undefined
      )?.persistentCheckpointRecovery;
      if (persistentRecovery?.restored === true) {
        const recoveryResult = result;
        let reverified = runtimeTool
          ? await this.executeRuntimeTool(runtimeTool, params)
          : await toolRegistry.execute(toolName, params);
        reverified = await this.attachVerifierDiagnostics(
          reverified,
          agentId,
          options.correlationId
        );
        const reverifiedScorecard = this.verifierScorecardFromToolResult(
          reverified.result
        );
        const expectedReward = Number(persistentRecovery.expectedReward);
        const restoreVerified = Boolean(
          reverified.success
          && reverifiedScorecard
          && Number.isFinite(expectedReward)
          && reverifiedScorecard.reward + 1e-12 >= expectedReward
        );
        const recoveryMetadata = {
          ...persistentRecovery,
          reverified: restoreVerified,
          reverifiedReward: reverifiedScorecard?.reward,
        };
        this.emit({
          type: restoreVerified
            ? 'workspace.verifier_checkpoint.reverified'
            : 'workspace.verifier_checkpoint.reverification_failed',
          agentId,
          correlationId: options.correlationId,
          data: recoveryMetadata,
        });
        result = {
          ...reverified,
          ...(!restoreVerified
            ? {
              success: false,
              error: [
                reverified.error,
                `Persistent verifier checkpoint restoration did not reproduce reward ${expectedReward}.`,
              ].filter(Boolean).join('\n'),
            }
            : {}),
          result: {
            ...(
              reverified.result && typeof reverified.result === 'object'
                ? reverified.result as Record<string, unknown>
                : {}
            ),
            persistentCheckpointRecovery: recoveryMetadata,
            candidateRollback: (
              recoveryResult.result as { candidateRollback?: unknown } | undefined
            )?.candidateRollback,
            regressionRollback: (
              recoveryResult.result as { regressionRollback?: unknown } | undefined
            )?.regressionRollback,
          },
        };
      }
    }
    this.recordToolPathOutcome(agentId, toolName, params, result, options.correlationId);
    this.toolCallCounts.set(callKey, calls + 1);
    const resultMessage = await this.enqueueMessage({
      kind: 'tool.result',
      sessionId: ctx.sessionId,
      from: `tool.${toolName}`,
      to: agentId,
      correlationId: options.correlationId,
      parentMessageId: toolCall.id,
      payload: result,
      metadata: { agentId, nodeId: options.nodeId },
    });
    await this.processQueuedMessage(resultMessage.id);
    if (result.success) await ctx.queue.ack(toolCall.id);
    else await ctx.queue.fail(toolCall.id, new Error(result.error ?? 'tool_failed'));
    await ctx.queue.ack(resultMessage.id);
    this.emit({
      type: result.success ? 'tool.result' : 'tool.error',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: {
        toolName,
        correlationId: options.correlationId,
        success: result.success,
        error: result.error,
        result: this.compactToolTraceResult(toolName, result.result),
      },
    });
    if (!result.success && /\b(?:timed?\s*out|timeout|deadline exceeded)\b/i.test(result.error ?? '')) {
      this.emit({
        type: 'tool.timeout',
        agentId,
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          toolName,
          correlationId: options.correlationId,
          error: result.error,
        },
      });
    }
    return result;
  }

  private compactToolTraceResult(toolName: string, result: unknown): unknown {
    if (!result || typeof result !== 'object') return undefined;
    if (toolName !== 'shell.exec') return undefined;
    const shell = result as {
      command?: unknown;
      cwd?: unknown;
      stdout?: unknown;
      stderr?: unknown;
      exitCode?: unknown;
      timedOut?: unknown;
      verifierDiagnostics?: unknown;
      candidateRetention?: unknown;
      candidateRollback?: unknown;
      regressionRollback?: unknown;
    };
    const compactTail = (value: unknown, maxChars = 8_000): string | undefined => {
      if (typeof value !== 'string' || value.length === 0) return undefined;
      return value.length <= maxChars
        ? value
        : `[runtime_trace_compacted_${value.length - maxChars}_leading_chars]\n${value.slice(-maxChars)}`;
    };
    return {
      command: typeof shell.command === 'string' ? shell.command.slice(0, 2_000) : undefined,
      cwd: typeof shell.cwd === 'string' ? shell.cwd : undefined,
      exitCode: typeof shell.exitCode === 'number' ? shell.exitCode : undefined,
      timedOut: shell.timedOut === true,
      stdout: compactTail(shell.stdout),
      stderr: compactTail(shell.stderr),
      verifierDiagnostics: shell.verifierDiagnostics,
      candidateRetention: shell.candidateRetention,
      candidateRollback: shell.candidateRollback,
      regressionRollback: shell.regressionRollback,
    };
  }

  private async attachVerifierDiagnostics(
    toolResult: ToolResult,
    agentId: string,
    correlationId?: string
  ): Promise<ToolResult> {
    if (!toolResult.result || typeof toolResult.result !== 'object') return toolResult;
    const verifierRoots = [
      '/logs/verifier',
      path.join(this.workspaceRoot, 'logs', 'verifier'),
      path.join(this.workspaceRoot, '.roy', 'verifier'),
    ];
    const candidateFiles = [
      '/logs/verifier/scorecard.json',
      '/logs/verifier/grade.log',
      '/logs/verifier/reward.txt',
      path.join(this.workspaceRoot, 'logs', 'verifier', 'scorecard.json'),
      path.join(this.workspaceRoot, 'logs', 'verifier', 'grade.log'),
      path.join(this.workspaceRoot, 'logs', 'verifier', 'reward.txt'),
      path.join(this.workspaceRoot, '.roy', 'verifier', 'scorecard.json'),
      path.join(this.workspaceRoot, '.roy', 'verifier', 'reward.txt'),
    ];
    const knownCandidates = new Set(candidateFiles);
    const discovered: string[] = [];
    for (const root of verifierRoots) {
      try {
        const entries = await readdir(root, { withFileTypes: true });
        for (const entry of entries) {
          if (!entry.isFile()
            || !/(?:scorecard|details|summary|report|grade|reward|failure|error)[^/]*\.(?:json|log|txt)$/i.test(
              entry.name
            )) {
            continue;
          }
          const candidate = path.join(root, entry.name);
          if (!knownCandidates.has(candidate)) discovered.push(candidate);
        }
      } catch {
        // Verifier roots are optional and may live outside the workspace.
      }
    }
    candidateFiles.push(...discovered
      .sort((left, right) => {
        const priority = (candidate: string): number =>
          /scorecard/i.test(candidate) ? 5
          : /details/i.test(candidate) ? 4
          : /(?:failure|error)/i.test(candidate) ? 3
          : /(?:summary|report)/i.test(candidate) ? 2
          : 1;
        return priority(right) - priority(left) || left.localeCompare(right);
      })
      .slice(0, 8));
    const diagnostics: Array<{ path: string; content: string }> = [];
    for (const candidate of candidateFiles) {
      if (diagnostics.some(item => item.path === candidate)) continue;
      try {
        const metadata = await stat(candidate);
        if (!metadata.isFile() || metadata.size > 256_000) continue;
        const content = await readFile(candidate, 'utf8');
        if (!content.trim()) continue;
        diagnostics.push({
          path: candidate,
          content: content.length <= 12_000
            ? content
            : `${content.slice(0, 7_000)}\n[${content.length - 12_000} middle chars compacted]\n${content.slice(-5_000)}`,
        });
      } catch {
        // Verifier artifacts are optional and environment-dependent.
      }
    }
    if (diagnostics.length === 0) return toolResult;
    this.emit({
      type: 'tool.verifier_diagnostics.attached',
      agentId,
      correlationId,
      data: {
        files: diagnostics.map(item => item.path),
        totalChars: diagnostics.reduce((sum, item) => sum + item.content.length, 0),
      },
    });
    return {
      ...toolResult,
      result: {
        ...(toolResult.result as Record<string, unknown>),
        verifierDiagnostics: diagnostics,
      },
    };
  }

  private shouldAttachVerifierDiagnostics(command: string): boolean {
    return /(?:^|[\s/])\.roy\/official-verifier\/|\bgrade\.py\b|\b(?:pytest|vitest|jest|mocha)\b/i.test(
      command
    );
  }

  private async captureWorkspaceMutationCheckpoint(
    toolName: string,
    params: Record<string, unknown>,
    groundingCalls: ToolCallRecord[],
    correlationId?: string
  ): Promise<WorkspaceMutationCheckpoint | undefined> {
    if (toolName !== 'fs.write'
      && toolName !== 'fs.replace'
      && toolName !== 'fs.synthesize') {
      return undefined;
    }
    if (typeof params.path !== 'string' || !params.path.trim()) return undefined;
    const normalizedPath = this.normalizeCachedPath(params.path);
    const absolutePath = path.resolve(this.workspaceRoot, normalizedPath);
    const relativePath = path.relative(this.workspaceRoot, absolutePath);
    if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) return undefined;
    try {
      const previousContent = await readFile(absolutePath, 'utf8');
      return {
        path: normalizedPath,
        previousContent,
        baseline: this.latestVerifierScorecardFromCalls(groundingCalls)
          ?? (correlationId
            ? this.latestVerifierScorecardFromCalls(
                this.restoredToolCallsFromResume(correlationId)
              )
            : undefined),
        candidateFingerprint: this.fingerprint({
          toolName,
          path: normalizedPath,
          params,
          previousContentHash: createHash('sha256')
            .update(previousContent)
            .digest('hex'),
        }),
        createdAt: Date.now(),
      };
    } catch {
      // New-file mutations are not rolled back automatically because removal is
      // a materially different operation. Existing source files are protected.
      return undefined;
    }
  }

  private latestVerifierScorecardFromCalls(
    calls: ToolCallRecord[]
  ): VerifierScorecard | undefined {
    for (const call of [...calls].reverse()) {
      const scorecard = this.verifierScorecardFromToolResult(call.result);
      if (scorecard) return scorecard;
    }
    return undefined;
  }

  private verifierScorecardFromToolResult(result: unknown): VerifierScorecard | undefined {
    if (!result || typeof result !== 'object') return undefined;
    const resultRecord = result as {
      verifierDiagnostics?: unknown;
      candidateRollback?: unknown;
      regressionRollback?: unknown;
    };
    const diagnostics = resultRecord.verifierDiagnostics;
    if (!Array.isArray(diagnostics)) return undefined;
    let scalarReward: number | undefined;
    let structuredGroups: Record<string, number> | undefined;
    const failureFrontiers: Record<string, number> = {};
    for (const item of diagnostics) {
      if (!item || typeof item !== 'object') continue;
      const diagnostic = item as { path?: unknown; content?: unknown };
      if (typeof diagnostic.content !== 'string') continue;
      const diagnosticPath = String(diagnostic.path ?? '').replaceAll('\\', '/');
      const scalar = Number(diagnostic.content.trim());
      if (/(?:^|\/)reward\.txt$/i.test(diagnosticPath)
        && Number.isFinite(scalar)) {
        scalarReward = scalar;
      }
      try {
        const parsed = JSON.parse(diagnostic.content) as {
          reward?: unknown;
          groups?: unknown;
          gates?: unknown;
        } | number;
        if (typeof parsed === 'number') {
          if (Number.isFinite(parsed)
            && /(?:^|\/)reward\.txt$/i.test(diagnosticPath)) {
            scalarReward = parsed;
          }
          continue;
        }
        Object.assign(
          failureFrontiers,
          this.verifierFailureFrontierCounts(parsed)
        );
        if ('gates' in parsed
          && parsed.gates
          && typeof parsed.gates === 'object'
          && !Array.isArray(parsed.gates)) {
          const gates = Object.fromEntries(
            Object.entries(parsed.gates)
              .map(([group, value]): [string, number] | undefined =>
                typeof value === 'boolean'
                  ? [group, value ? 1 : 0]
                  : typeof value === 'number' && Number.isFinite(value)
                    ? [group, value]
                    : undefined
              )
              .filter((entry): entry is [string, number] => Boolean(entry))
          );
          if (Object.keys(gates).length > 0) structuredGroups = gates;
        }
        if (typeof parsed.reward !== 'number'
          || !('groups' in parsed)
          || !parsed.groups
          || typeof parsed.groups !== 'object'
          || Array.isArray(parsed.groups)) {
          continue;
        }
        const groups = Object.fromEntries(
          Object.entries(parsed.groups)
            .filter((entry): entry is [string, number] =>
              typeof entry[1] === 'number' && Number.isFinite(entry[1])
            )
        );
        const rollback = workspaceCandidateRollbackFromCall({
          toolName: 'shell.exec',
          params: {},
          result,
          success: true,
        });
        if (!rollback || typeof rollback.baselineReward !== 'number') {
          return {
            reward: parsed.reward,
            groups,
            ...(Object.keys(failureFrontiers).length > 0
              ? { failureFrontiers }
              : {}),
          };
        }
        const baselineGroups = {
          ...(rollback.baselineGroups ?? groups),
        };
        for (const group of [
          ...(rollback.regressedGroups ?? []),
          ...(rollback.improvedGroups ?? []),
        ]) {
          if (typeof group.before === 'number') baselineGroups[group.group] = group.before;
        }
        return {
          reward: rollback.baselineReward,
          groups: baselineGroups,
          ...(rollback.baselineFailureFrontiers
            ? { failureFrontiers: rollback.baselineFailureFrontiers }
            : Object.keys(failureFrontiers).length > 0
              ? { failureFrontiers }
              : {}),
        };
      } catch {
        // grade.log and other diagnostic files may not be scorecards.
      }
    }
    if (structuredGroups && scalarReward !== undefined) {
      return {
        reward: scalarReward,
        groups: structuredGroups,
        ...(Object.keys(failureFrontiers).length > 0
          ? { failureFrontiers }
          : {}),
      };
    }
    return scalarReward === undefined
      ? undefined
      : { reward: scalarReward, groups: { __reward__: scalarReward } };
  }

  private verifierFailureFrontierCounts(
    value: unknown
  ): Record<string, number> {
    const counts: Record<string, number> = {};
    const visit = (
      current: unknown,
      pathParts: string[],
      insideFailureFrontier: boolean,
      depth: number
    ): void => {
      if (depth > 6 || !current || typeof current !== 'object') return;
      if (Array.isArray(current)) {
        if (insideFailureFrontier && pathParts.length > 0) {
          counts[pathParts.join('.')] = current.length;
        }
        return;
      }
      for (const [key, child] of Object.entries(current)) {
        if (['gates', 'groups', 'reward'].includes(key.toLowerCase())) continue;
        const failureKey = /\b(?:violations?|failures?|errors?|failed|missing|mismatches?|issues?)\b/i.test(
          key.replaceAll('_', ' ')
        );
        visit(
          child,
          [...pathParts, key],
          insideFailureFrontier || failureKey,
          depth + 1
        );
      }
    };
    visit(value, [], false, 0);
    return counts;
  }

  private async rollbackVerifierRegression(
    toolResult: ToolResult,
    agentId: string,
    correlationId: string
  ): Promise<ToolResult> {
    const checkpointMap = this.mutationCheckpointsByCorrelation.get(correlationId);
    const checkpoints = [...(checkpointMap?.values() ?? [])]
      .sort((left, right) => left.createdAt - right.createdAt);
    const checkpoint = checkpoints.find(item => item.baseline)
      ?? checkpoints[0];
    const current = this.verifierScorecardFromToolResult(toolResult.result);
    if (!current) return toolResult;
    if (!checkpoint) {
      return await this.restorePersistentVerifierWorkspaceCheckpoint(
        toolResult,
        current,
        agentId,
        correlationId
      );
    }
    if (!checkpoint.baseline) {
      this.mutationCheckpointsByCorrelation.delete(correlationId);
      await this.persistAcceptedVerifierWorkspaceCheckpoint(
        current,
        checkpoints,
        agentId,
        correlationId
      );
      return toolResult;
    }
    this.mutationCheckpointsByCorrelation.delete(correlationId);
    const baseline = checkpoint.baseline;
    if (current.reward > baseline.reward + 1e-12) {
      await this.persistAcceptedVerifierWorkspaceCheckpoint(
        current,
        checkpoints,
        agentId,
        correlationId
      );
      return toolResult;
    }
    const regressedGroups = Object.keys(baseline.groups)
      .filter(group => (current.groups[group] ?? 0) + 1e-12 < baseline.groups[group]!)
      .map(group => ({
        group,
        before: baseline.groups[group],
        after: current.groups[group] ?? 0,
      }));
    const improvedGroups = Object.keys(current.groups)
      .filter(group => (current.groups[group] ?? 0) > (baseline.groups[group] ?? 0) + 1e-12)
      .map(group => ({
        group,
        before: baseline.groups[group] ?? 0,
        after: current.groups[group],
      }));
    const baselineFailureFrontiers = baseline.failureFrontiers ?? {};
    const candidateFailureFrontiers = current.failureFrontiers ?? {};
    const improvedFailureFrontiers = Object.keys(baselineFailureFrontiers)
      .filter(frontier =>
        (candidateFailureFrontiers[frontier] ?? 0)
          < baselineFailureFrontiers[frontier]!
      )
      .map(frontier => ({
        frontier,
        before: baselineFailureFrontiers[frontier],
        after: candidateFailureFrontiers[frontier] ?? 0,
      }));
    const regressedFailureFrontiers = Object.keys(baselineFailureFrontiers)
      .filter(frontier =>
        frontier in candidateFailureFrontiers
        && candidateFailureFrontiers[frontier]!
          > baselineFailureFrontiers[frontier]!
      )
      .map(frontier => ({
        frontier,
        before: baselineFailureFrontiers[frontier],
        after: candidateFailureFrontiers[frontier],
      }));
    const hardGateComposition = baseline.reward <= 1e-12
      && current.reward <= 1e-12
      && regressedGroups.length === 0;
    const failureFrontierReduced = current.reward + 1e-12 >= baseline.reward
      && regressedGroups.length === 0
      && regressedFailureFrontiers.length === 0
      && improvedFailureFrontiers.length > 0;
    if (hardGateComposition || failureFrontierReduced) {
      await this.persistAcceptedVerifierWorkspaceCheckpoint(
        current,
        checkpoints,
        agentId,
        correlationId
      );
      const retainedPaths = checkpoints.map(item => item.path);
      const candidateRetention = {
        retained: true,
        path: checkpoint.path,
        retainedPaths,
        reason: failureFrontierReduced
          ? 'failure_frontier_reduced'
          : 'hard_gate_composition',
        candidateFingerprint: checkpoint.candidateFingerprint,
        baselineReward: baseline.reward,
        candidateReward: current.reward,
        baselineGroups: baseline.groups,
        candidateGroups: current.groups,
        regressedGroups,
        improvedGroups,
        baselineFailureFrontiers,
        candidateFailureFrontiers,
        regressedFailureFrontiers,
        improvedFailureFrontiers,
      };
      this.emit({
        type: 'workspace.mutation.candidate_retained',
        agentId,
        correlationId,
        data: candidateRetention,
      });
      return {
        ...toolResult,
        result: {
          ...(toolResult.result as Record<string, unknown>),
          candidateRetention,
        },
      };
    }
    const reason = current.reward + 1e-12 < baseline.reward
      ? 'reward_regression'
      : 'no_objective_gain';
    const restoreFailures: Array<{ path: string; error: string }> = [];
    for (const item of [...checkpoints].reverse()) {
      const restored = await new FsWriteTool(this.workspaceRoot).execute({
        path: item.path,
        content: item.previousContent,
        mode: 'overwrite',
        createDirectories: false,
      });
      if (!restored.success) {
        restoreFailures.push({
          path: item.path,
          error: restored.error ?? 'unknown restore failure',
        });
      }
    }
    if (restoreFailures.length > 0) {
      return {
        ...toolResult,
        error: [
          toolResult.error,
          `Verifier candidate changed reward from ${baseline.reward} to ${current.reward} without objective improvement, and rollback failed: ${JSON.stringify(restoreFailures)}`,
        ].filter(Boolean).join('\n'),
      };
    }
    const changed = this.changedPathsByCorrelation.get(correlationId) ?? new Set<string>();
    for (const item of checkpoints) changed.add(item.path);
    this.changedPathsByCorrelation.set(correlationId, changed);
    const checkpointAgeMs = Date.now() - Math.min(
      ...checkpoints.map(item => item.createdAt)
    );
    const restoredPaths = checkpoints.map(item => item.path);
    this.emit({
      type: 'workspace.mutation.candidate_rolled_back',
      agentId,
      correlationId,
      data: {
        path: checkpoint.path,
        reason,
        candidateFingerprint: checkpoint.candidateFingerprint,
        restoredPaths,
        baselineReward: baseline.reward,
        candidateReward: current.reward,
        regressedGroups,
        improvedGroups,
        baselineFailureFrontiers,
        candidateFailureFrontiers,
        regressedFailureFrontiers,
        improvedFailureFrontiers,
        checkpointAgeMs,
      },
    });
    if (reason === 'reward_regression') {
      this.emit({
        type: 'workspace.mutation.regression_rolled_back',
        agentId,
        correlationId,
        data: {
          path: checkpoint.path,
          restoredPaths,
          baselineReward: baseline.reward,
          regressedReward: current.reward,
          regressedGroups,
          checkpointAgeMs,
        },
      });
    }
    const candidateRollback = {
      restored: true,
      path: checkpoint.path,
      restoredPaths,
      reason,
      candidateFingerprint: checkpoint.candidateFingerprint,
      baselineReward: baseline.reward,
      candidateReward: current.reward,
      baselineGroups: baseline.groups,
      candidateGroups: current.groups,
      regressedGroups,
      improvedGroups,
      baselineFailureFrontiers,
      candidateFailureFrontiers,
      regressedFailureFrontiers,
      improvedFailureFrontiers,
    };
    return {
      ...toolResult,
      result: {
        ...(toolResult.result as Record<string, unknown>),
        candidateRollback,
        ...(reason === 'reward_regression'
          ? {
            regressionRollback: {
              ...candidateRollback,
              regressedReward: current.reward,
            },
          }
          : {}),
      },
    };
  }

  private persistentVerifierWorkspaceCheckpointPath(): string {
    return path.join(
      this.workspaceRoot,
      '.roy',
      'cache',
      'accepted-workspace-checkpoint.json'
    );
  }

  private async readPersistentVerifierWorkspaceCheckpoint():
  Promise<PersistentVerifierWorkspaceCheckpoint | undefined> {
    try {
      const parsed = JSON.parse(await readFile(
        this.persistentVerifierWorkspaceCheckpointPath(),
        'utf8'
      )) as PersistentVerifierWorkspaceCheckpoint;
      if (parsed.version !== 1
        || parsed.sessionId !== this.getContext().sessionId
        || !parsed.scorecard
        || typeof parsed.scorecard.reward !== 'number'
        || !Array.isArray(parsed.files)
        || parsed.files.length === 0) {
        return undefined;
      }
      return parsed;
    } catch {
      return undefined;
    }
  }

  private async persistAcceptedVerifierWorkspaceCheckpoint(
    scorecard: VerifierScorecard,
    checkpoints: WorkspaceMutationCheckpoint[],
    agentId: string,
    correlationId: string
  ): Promise<void> {
    const existing = await this.readPersistentVerifierWorkspaceCheckpoint();
    if (existing && existing.scorecard.reward > scorecard.reward + 1e-12) return;
    const filesByPath = new Map(
      existing?.files.map(file => [file.path, file]) ?? []
    );
    for (const checkpoint of checkpoints) {
      const normalizedPath = this.normalizeCachedPath(checkpoint.path);
      if (!normalizedPath
        || normalizedPath === '.'
        || normalizedPath.startsWith('.roy/official-verifier/')) {
        continue;
      }
      const absolutePath = path.resolve(this.workspaceRoot, normalizedPath);
      const relativePath = path.relative(this.workspaceRoot, absolutePath);
      if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) continue;
      try {
        const content = await readFile(absolutePath, 'utf8');
        filesByPath.set(normalizedPath, {
          path: normalizedPath,
          content,
          sha256: createHash('sha256').update(content).digest('hex'),
        });
      } catch {
        // A checkpoint remains useful even when one optional generated file is
        // no longer present. Existing accepted files are retained above.
      }
    }
    if (filesByPath.size === 0) return;
    const persisted: PersistentVerifierWorkspaceCheckpoint = {
      version: 1,
      sessionId: this.getContext().sessionId,
      scorecard,
      files: [...filesByPath.values()].sort((left, right) =>
        left.path.localeCompare(right.path)
      ),
      correlationId,
      verifiedAt: Date.now(),
    };
    const destination = this.persistentVerifierWorkspaceCheckpointPath();
    await mkdir(path.dirname(destination), { recursive: true });
    const temporary = `${destination}.${process.pid}.${randomUUID()}.tmp`;
    await writeFile(temporary, `${JSON.stringify(persisted, null, 2)}\n`, 'utf8');
    await rename(temporary, destination);
    this.emit({
      type: 'workspace.verifier_checkpoint.persisted',
      agentId,
      correlationId,
      data: {
        reward: scorecard.reward,
        groups: scorecard.groups,
        paths: persisted.files.map(file => file.path),
        checkpointPath: path.relative(this.workspaceRoot, destination),
      },
    });
  }

  private async restorePersistentVerifierWorkspaceCheckpoint(
    toolResult: ToolResult,
    current: VerifierScorecard,
    agentId: string,
    correlationId: string
  ): Promise<ToolResult> {
    const accepted = await this.readPersistentVerifierWorkspaceCheckpoint();
    if (!accepted || current.reward + 1e-12 >= accepted.scorecard.reward) {
      return toolResult;
    }
    const failures: Array<{ path: string; error: string }> = [];
    const restoredPaths: string[] = [];
    for (const file of accepted.files) {
      const normalizedPath = this.normalizeCachedPath(file.path);
      const expectedHash = createHash('sha256').update(file.content).digest('hex');
      if (expectedHash !== file.sha256
        || normalizedPath === '.'
        || normalizedPath.startsWith('.roy/official-verifier/')) {
        failures.push({ path: file.path, error: 'checkpoint integrity validation failed' });
        continue;
      }
      const restored = await new FsWriteTool(this.workspaceRoot).execute({
        path: normalizedPath,
        content: file.content,
        mode: 'overwrite',
        createDirectories: true,
      });
      if (!restored.success) {
        failures.push({
          path: normalizedPath,
          error: restored.error ?? 'unknown restore failure',
        });
        continue;
      }
      restoredPaths.push(normalizedPath);
    }
    if (failures.length > 0 || restoredPaths.length === 0) {
      return {
        ...toolResult,
        success: false,
        error: [
          toolResult.error,
          `Persistent accepted workspace restoration failed: ${JSON.stringify(failures)}`,
        ].filter(Boolean).join('\n'),
      };
    }
    const candidateRollback = {
      restored: true,
      path: restoredPaths[0],
      restoredPaths,
      reason: 'persisted_accepted_workspace_regression',
      candidateFingerprint: this.fingerprint({
        current,
        acceptedAt: accepted.verifiedAt,
      }),
      baselineReward: accepted.scorecard.reward,
      candidateReward: current.reward,
      baselineGroups: accepted.scorecard.groups,
      candidateGroups: current.groups,
      regressedGroups: Object.keys(accepted.scorecard.groups)
        .filter(group =>
          (current.groups[group] ?? 0) + 1e-12 < accepted.scorecard.groups[group]!
        )
        .map(group => ({
          group,
          before: accepted.scorecard.groups[group],
          after: current.groups[group] ?? 0,
        })),
      improvedGroups: [],
    };
    const persistentCheckpointRecovery = {
      restored: true,
      restoredPaths,
      expectedReward: accepted.scorecard.reward,
      observedReward: current.reward,
      checkpointVerifiedAt: accepted.verifiedAt,
      checkpointCorrelationId: accepted.correlationId,
    };
    this.emit({
      type: 'workspace.verifier_checkpoint.restored',
      agentId,
      correlationId,
      data: persistentCheckpointRecovery,
    });
    return {
      ...toolResult,
      result: {
        ...(toolResult.result as Record<string, unknown>),
        persistentCheckpointRecovery,
        candidateRollback,
        regressionRollback: {
          ...candidateRollback,
          regressedReward: current.reward,
        },
      },
    };
  }

  private immutableRuntimeEvidenceMutationError(
    toolName: string,
    params: Record<string, unknown>
  ): string | undefined {
    const immutableRoot = '.roy/official-verifier';
    if (toolName === 'fs.write'
      || toolName === 'fs.replace'
      || toolName === 'fs.synthesize') {
      const target = this.normalizeCachedPath(String(params.path ?? ''));
      if (target === immutableRoot || target.startsWith(`${immutableRoot}/`)) {
        return `${immutableRoot} is immutable runtime evidence; repair the implementation under test instead of modifying the verifier mirror`;
      }
      return undefined;
    }
    if (toolName !== 'shell.exec') return undefined;
    const command = String(params.command ?? '');
    if (!command.includes(immutableRoot)) return undefined;
    const explicitTargets = this.extractShellMutationPaths(command)
      .map(target => this.normalizeCachedPath(target));
    const directInPlaceMutation =
      /\b(?:sed\s+-i|perl\s+-pi|chmod|chown|truncate|touch|rm|rmdir)\b[^;&|\n]*\.roy\/official-verifier(?:\/|\b)/i.test(command);
    if (explicitTargets.some(target =>
      target === immutableRoot || target.startsWith(`${immutableRoot}/`)
    ) || directInPlaceMutation) {
      return `${immutableRoot} is immutable runtime evidence; shell mutations that target the verifier mirror are not allowed`;
    }
    return undefined;
  }

  private async getCachedInvalidPathRejection(
    agentId: string,
    toolName: string,
    params: Record<string, unknown>,
    options: { reason?: string; correlationId?: string; nodeId?: string }
  ): Promise<ToolResult | undefined> {
    if ((toolName !== 'fs.read' && toolName !== 'fs.list' && toolName !== 'fs.search')
      || typeof params.path !== 'string') {
      return undefined;
    }
    const requestedPath = this.normalizeCachedPath(params.path);
    const correlationKey = options.correlationId ?? this.getContext().sessionId;
    const retryReason = options.reason ?? '';
    if (/\b(?:changed hypothesis|path created|created by|after mutation|retry after|state changed)\b/i.test(retryReason)
      || /(?:假设已改变|路径已创建|写入后重试|状态已改变)/.test(retryReason)) {
      return undefined;
    }
    const changedPaths = this.changedPathsByCorrelation.get(correlationKey) ?? new Set<string>();
    if ([...changedPaths].some(item =>
      item === requestedPath || requestedPath.startsWith(`${item}/`) || item.startsWith(`${requestedPath}/`)
    )) {
      return undefined;
    }
    // Persisted path knowledge is a scheduling hint, not a substitute for the
    // current filesystem. In particular, a valid file read with an obsolete
    // line range must never poison the entire path across later repair turns.
    const absolutePath = path.resolve(this.workspaceRoot, requestedPath);
    const relativePath = path.relative(this.workspaceRoot, absolutePath);
    if (!relativePath.startsWith('..') && !path.isAbsolute(relativePath)) {
      try {
        await stat(absolutePath);
        return undefined;
      } catch {
        // The path is still absent or inaccessible, so cached rejection remains useful.
      }
    }
    const liveFailure = this.failedPathObservations.get(correlationKey)?.get(requestedPath);
    const knowledge = await this.getContext().memory.readExecutionKnowledge(undefined, 24);
    const lastInvalidAt = knowledge.paths
      .filter(item => item.invalidPaths.includes(requestedPath))
      .reduce((latest, item) => Math.max(latest, item.updatedAt), 0);
    const lastObservedAt = knowledge.paths
      .filter(item => item.observedPaths.includes(requestedPath))
      .reduce((latest, item) => Math.max(latest, item.updatedAt), 0);
    if (!liveFailure && (lastInvalidAt === 0 || lastObservedAt > lastInvalidAt)) return undefined;

    const basename = path.posix.basename(requestedPath);
    const alternatives = Array.from(new Set(
      knowledge.paths.flatMap(item => item.observedPaths)
        .filter(item =>
          item !== requestedPath
          && (path.posix.basename(item) === basename || item.includes(`/${basename}/`))
        )
    )).slice(0, 8);
    const source = liveFailure ? 'current execution path' : 'persisted execution knowledge';
    const error = [
      `Cached invalid path rejected: ${requestedPath} already failed in ${source}.`,
      alternatives.length > 0
        ? `Use an authoritative observed alternative: ${alternatives.join(', ')}.`
        : 'Inspect an observed parent directory before selecting a replacement path.',
      'Retry this exact path only after a mutation or with a reason that states the changed hypothesis.',
    ].join(' ');
    this.emit({
      type: 'tool.path.cache_rejected',
      agentId,
      sessionId: this.getContext().sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: {
        toolName,
        path: requestedPath,
        source,
        alternatives,
        previousError: liveFailure?.error,
      },
    });
    return {
      success: false,
      error,
      metadata: {
        cacheRejected: true,
        path: requestedPath,
        alternatives,
      },
    };
  }

  private recordToolPathOutcome(
    agentId: string,
    toolName: string,
    params: Record<string, unknown>,
    result: ToolResult,
    correlationId?: string
  ): void {
    const correlationKey = correlationId ?? this.getContext().sessionId;
    if (toolName === 'shell.exec'
      && result.success
      && typeof params.command === 'string'
      && isSuccessfulWorkspaceMutationCall({
        toolName,
        params,
        success: true,
      })) {
      const changed = this.changedPathsByCorrelation.get(correlationKey) ?? new Set<string>();
      for (const changedPath of this.extractShellMutationPaths(params.command)) {
        changed.add(changedPath);
        this.failedPathObservations.get(correlationKey)?.delete(changedPath);
      }
      this.changedPathsByCorrelation.set(correlationKey, changed);
      return;
    }
    if (typeof params.path !== 'string') return;
    const observedPath = this.normalizeCachedPath(params.path);
    if (result.success) {
      this.failedPathObservations.get(correlationKey)?.delete(observedPath);
      if (toolName === 'fs.write'
        || toolName === 'fs.replace'
        || toolName === 'fs.synthesize') {
        const changed = this.changedPathsByCorrelation.get(correlationKey) ?? new Set<string>();
        changed.add(observedPath);
        this.changedPathsByCorrelation.set(correlationKey, changed);
      }
      return;
    }
    if (toolName !== 'fs.read' && toolName !== 'fs.list' && toolName !== 'fs.search') return;
    if (!this.isAuthoritativeInvalidPathFailure(result.error)) return;
    const failures = this.failedPathObservations.get(correlationKey) ?? new Map();
    failures.set(observedPath, {
      toolName,
      path: observedPath,
      error: result.error ?? 'unknown path failure',
      actorId: agentId,
      observedAt: Date.now(),
    });
    this.failedPathObservations.set(correlationKey, failures);
  }

  private isAuthoritativeInvalidPathFailure(error: string | undefined): boolean {
    if (!error) return false;
    return /\bENOENT\b|no such file or directory|cannot find the (?:file|path)|path must point to a file/i.test(error);
  }

  private extractShellMutationPaths(command: string): string[] {
    const tokens = command.match(/"[^"]*"|'[^']*'|[^\s]+/g) ?? [];
    const changed = new Set<string>();
    const mutationCommands = new Set(['mkdir', 'touch', 'rm', 'rmdir', 'mv', 'cp', 'install', 'truncate']);
    const controls = new Set(['&&', '||', '|', ';']);
    for (let index = 0; index < tokens.length; index += 1) {
      const commandToken = tokens[index].replace(/^.*\//, '');
      if (!mutationCommands.has(commandToken)) continue;
      for (let cursor = index + 1; cursor < tokens.length; cursor += 1) {
        const raw = tokens[cursor];
        const unquoted = raw.replace(/^['"]|['"]$/g, '').replace(/[;,]+$/, '');
        if (controls.has(raw) || /^(?:&&|\|\||[|;])$/.test(unquoted)) break;
        if (!unquoted || unquoted.startsWith('-')) continue;
        if (mutationCommands.has(unquoted.replace(/^.*\//, ''))) break;
        changed.add(this.normalizeCachedPath(unquoted));
      }
    }
    for (const match of command.matchAll(/>{1,2}\s*(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))/g)) {
      const target = match[1] ?? match[2] ?? match[3];
      if (target) changed.add(this.normalizeCachedPath(target));
    }
    return [...changed];
  }

  async resolveToolApproval(id: string, decision: 'approved' | 'denied'): Promise<ToolApprovalRequest | undefined> {
    if (!this.toolApprovalManager) throw new Error('Tool approval manager is not initialized');
    const request = this.toolApprovalManager.resolve(id, decision);
    if (request) {
      const ctx = this.getContext();
      const message = await this.enqueueMessage({
        kind: 'tool.approval.resolved',
        sessionId: ctx.sessionId,
        from: 'runtime.approval',
        to: request.agentId,
        payload: request,
        metadata: { agentId: request.agentId },
      });
      await this.processQueuedMessage(message.id);
      await ctx.queue.ack(message.id);
      this.emit({
        type: 'tool.approval.resolved',
        agentId: request.agentId,
        data: { approvalId: id, decision, toolName: request.toolName },
      });
    }
    return request;
  }

  private async requestAgentBudget(input: {
    parentId: string;
    requesterId?: string;
    archetype: SubAgentArchetype;
    correlationId?: string;
    nodeId?: string;
    requestedTokens?: number;
    minimumTokens?: number;
    priority?: BudgetPriority;
    expectedUtility?: number;
    purpose: string;
  }): Promise<BudgetAllocation | undefined> {
    if (input.correlationId && this.evolutionBudgetBypassCorrelations.has(input.correlationId)) {
      this.emit({
        type: 'budget.bypassed',
        agentId: input.requesterId ?? input.parentId,
        correlationId: input.correlationId,
        data: { reason: 'evolution_ablation_without_budget_market', requestedTokens: input.requestedTokens, purpose: input.purpose },
      });
      return undefined;
    }
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = input.requestedTokens ?? this.estimateAgentBudget(input.archetype, input.purpose);
    const requesterId = input.requesterId ?? `${input.parentId}:${input.archetype}`;
    const analysis = input.correlationId ? this.tomAnalyses.get(input.correlationId) : undefined;
    const tools = this.getToolBindingsForTask(input.archetype, input.purpose).filter(binding => binding.enabled).length;
    const cachedPattern = await this.getContext().memory.findAgentPattern(input.archetype);
    const evaluation = cachedPattern?.evaluation && typeof cachedPattern.evaluation === 'object'
      ? cachedPattern.evaluation as Record<string, unknown>
      : undefined;
    const historicalUtility = averageDefined([
      finiteRatio(evaluation?.successRate),
      finiteRatio(evaluation?.groundingRate),
    ]) ?? 0.5;
    const evaluationRuns = finiteNumber(evaluation?.runs) ?? 0;
    const cacheConfidence = cachedPattern
      ? Math.min(0.95, 0.5 + evaluationRuns * 0.08)
      : 0;
    const requestMessage = await this.enqueueMessage({
      kind: 'budget.request',
      sessionId: this.getContext().sessionId,
      from: input.parentId,
      to: 'budget.market',
      correlationId: input.correlationId,
      payload: { ...input, requestedTokens },
      metadata: { agentId: input.parentId, budgetTokens: requestedTokens, nodeId: input.nodeId },
    });
    await this.processQueuedMessage(requestMessage.id);
    this.emit({
      type: 'budget.requested',
      agentId: input.parentId,
      correlationId: input.correlationId,
      nodeId: input.nodeId,
      data: { correlationId: input.correlationId, archetype: input.archetype, requestedTokens, purpose: input.purpose },
    });
    const allocation = this.budgetMarket.request({
      requesterId,
      parentId: input.parentId,
      actorType: 'agent',
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens: input.minimumTokens ?? this.workspaceRuntimeConfig?.budgetMarket.minimumGrantTokens ?? 256,
      priority: input.priority ?? this.workspaceRuntimeConfig?.budgetMarket.defaultPriority ?? 'medium',
      expectedUtility: input.expectedUtility ?? this.defaultBudgetUtility(input.archetype),
      purpose: input.purpose,
      resourceEstimate: {
        tokens: requestedTokens,
        contextTokens: Math.min(4000, Math.round(requestedTokens * 0.35)),
        toolCalls: tools,
      },
      metadata: {
        investmentKind: input.purpose,
        evidenceGain: analysis?.gaps.some(gap => gap.kind === 'evidence') ? 0.82 : 0.25,
        uncertaintyReduction: analysis ? Math.min(1, analysis.gaps.length / 4) : 0.4,
        conflictResolution: analysis?.signals.conflictLevel ?? 0,
        verificationGain: input.archetype === 'tester' ? 0.9 : 0,
        historicalUtility,
        cacheConfidence,
        executionRisk: analysis ? Math.min(1, (1 - analysis.confidence) + analysis.signals.failedTraceCount * 0.08) : 0.25,
        confidence: analysis?.confidence ?? 0.62,
        investmentHistoryKey: typeof cachedPattern?.id === 'string'
          ? `${cachedPattern.id}:${input.purpose}`
          : `${input.archetype}:${input.purpose}`,
      },
    });
    if (this.workspaceRuntimeConfig?.budgetMarket.rebalanceOnRequest && allocation.status === 'denied') {
      this.rebalanceBudgetMarket();
    }
    const responseMessage = await this.enqueueMessage({
      kind: allocation.status === 'granted' ? 'budget.grant' : 'budget.denied',
      sessionId: this.getContext().sessionId,
      from: 'budget.market',
      to: input.parentId,
      correlationId: input.correlationId,
      parentMessageId: requestMessage.id,
      payload: allocation,
      metadata: { agentId: input.parentId, budgetTokens: allocation.grantedTokens, nodeId: input.nodeId },
    });
    await this.processQueuedMessage(responseMessage.id);
    await this.getContext().queue.ack(requestMessage.id);
    await this.getContext().queue.ack(responseMessage.id);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.granted' : 'budget.denied',
      agentId: input.parentId,
      correlationId: input.correlationId,
      nodeId: input.nodeId,
      data: {
        correlationId: input.correlationId,
        allocationId: allocation.id,
        requestedTokens,
        grantedTokens: allocation.grantedTokens,
        allocatedTokens: allocation.allocatedTokens,
        policy: allocation.policy,
        score: allocation.score,
        reason: allocation.reason,
      },
    });
    if (allocation.status === 'granted') {
      this.emit({
        type: 'budget.allocated',
        agentId: requesterId,
        correlationId: input.correlationId,
        nodeId: input.nodeId,
        data: {
          allocationId: allocation.id,
          requestedTokens,
          allocatedTokens: allocation.allocatedTokens,
          policy: allocation.policy,
          rationale: allocation.rationale,
        },
      });
    }
    return allocation;
  }

  private budgetAccountingDimension(): 'total_tokens' | 'output_tokens' | 'thinking_tokens' {
    return this.workspaceRuntimeConfig?.budgetMarket.accountingDimension ?? 'total_tokens';
  }

  private hasUnlimitedBudgetSupply(): boolean {
    return this.getBudgetState().mode === 'unlimited';
  }

  private accountedUsageTokens(actual: number | TokenUsage): number {
    if (typeof actual === 'number') return Math.max(0, Math.floor(actual));
    if (this.budgetAccountingDimension() === 'output_tokens') {
      return Math.max(0, Math.floor(actual.outputTokens ?? actual.completionTokens));
    }
    if (this.budgetAccountingDimension() === 'thinking_tokens') {
      return Math.max(0, Math.floor(
        actual.thinkingAccountingTokens
          ?? actual.thinkingTokens
          ?? actual.totalTokens
      ));
    }
    return Math.max(0, Math.floor(actual.totalTokens));
  }

  private ensureUnlimitedAllocationCoverage(
    allocationId: string,
    requiredConsumedTokens: number,
    agentId: string,
    purpose: string,
    correlationId?: string
  ): BudgetAllocation | undefined {
    if (!this.budgetMarket) return undefined;
    const allocation = this.budgetMarket.getAllocation(allocationId);
    if (!allocation || !this.hasUnlimitedBudgetSupply()) return allocation;
    const missingTokens = Math.max(
      0,
      Math.floor(requiredConsumedTokens) - allocation.allocatedTokens
    );
    if (missingTokens === 0) return allocation;
    const updated = this.budgetMarket.augment(allocationId, missingTokens, 1);
    const addedTokens = Math.max(
      0,
      (updated?.allocatedTokens ?? allocation.allocatedTokens)
        - allocation.allocatedTokens
    );
    if (addedTokens > 0) {
      this.emit({
        type: 'budget.rebalanced',
        agentId,
        correlationId,
        data: {
          allocationId,
          purpose,
          previousAllocatedTokens: allocation.allocatedTokens,
          addedTokens,
          allocatedTokens: updated?.allocatedTokens,
          requiredConsumedTokens,
          reason: 'unlimited_active_actor_lease_renewal',
        },
      });
    }
    return updated ?? allocation;
  }

  private budgetRequestTokens(inputTokens: number, completionTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens'
      ? inputTokens + completionTokens
      : completionTokens;
  }

  private budgetMinimumTokens(inputTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens' ? inputTokens + 1 : 1;
  }

  private completionCapacity(allocatedTokens: number, inputTokens: number): number {
    return this.budgetAccountingDimension() === 'total_tokens'
      ? Math.max(0, allocatedTokens - inputTokens)
      : Math.max(0, allocatedTokens);
  }

  private reasoningAwareCompletionTokenBudget(visibleOutputTokens: number): number {
    const llm = this.getContext().llm;
    const family = `${llm?.name ?? ''}/${llm?.defaultModel ?? ''}`.toLowerCase();
    const usesSharedReasoningBudget = family.includes('deepseek')
      || family.includes('reasoner')
      || /\/(?:o1|o3|o4)(?:-|$)/.test(family);
    return usesSharedReasoningBudget
      ? visibleOutputTokens + Math.max(3072, visibleOutputTokens * 2)
      : visibleOutputTokens;
  }

  private rootVisibleCompletionTokenBudget(purpose: string): number {
    if (purpose.includes('synthesis')) return 2_048;
    if (purpose.includes('repair')) return 1_536;
    return 512;
  }

  private isTruncatedFinishReason(finishReason?: string): boolean {
    return Boolean(
      finishReason
      && ['length', 'max_tokens'].includes(finishReason.toLowerCase())
    );
  }

  private mergeCompletionContinuation(existing: string, continuation: string): string {
    if (!existing) return continuation;
    if (!continuation) return existing;
    const maximumOverlap = Math.min(existing.length, continuation.length, 4_000);
    for (let overlap = maximumOverlap; overlap >= 16; overlap -= 1) {
      if (existing.endsWith(continuation.slice(0, overlap))) {
        return existing + continuation.slice(overlap);
      }
    }
    if (existing.endsWith(continuation)) return existing;
    return existing + continuation;
  }

  private compactRootContinuationContext(prompt: string): string {
    const maximumCharacters = 12_000;
    if (prompt.length <= maximumCharacters) return prompt;
    const userTask = prompt.match(/<user_task>([\s\S]*?)<\/user_task>/i)?.[0];
    if (userTask && userTask.length <= 8_000) {
      return [
        userTask,
        '[The original request also supplied delegated evidence. Continue the existing draft without inventing facts beyond it.]',
        prompt.slice(-4_000),
      ].join('\n\n');
    }
    return [
      prompt.slice(0, 8_000),
      '...[continuation context compacted]...',
      prompt.slice(-4_000),
    ].join('\n');
  }

  private async requestTeamSynthesisBudget(input: {
    team: TeamRuntimeState;
    correlationId: string;
    promptTokens: number;
    completionTokens: number;
  }): Promise<BudgetAllocation | undefined> {
    if (this.workspaceRuntimeConfig?.budgetMarket.enabled === false) return undefined;
    if (!this.budgetMarket) throw new Error('Budget market is not initialized');
    const requestedTokens = this.budgetRequestTokens(input.promptTokens, input.completionTokens);
    const minimumTokens = this.budgetMinimumTokens(input.promptTokens);
    const requestMessage = await this.enqueueMessage({
      kind: 'budget.request',
      sessionId: this.getContext().sessionId,
      from: input.team.identity.id,
      to: 'budget.market',
      correlationId: input.correlationId,
      payload: {
        requesterId: input.team.identity.id,
        parentId: input.team.identity.parentAgentId,
        requestedTokens,
        minimumTokens,
        purpose: 'team_synthesis',
      },
      metadata: {
        agentId: input.team.identity.id,
        teamId: input.team.identity.id,
        budgetTokens: requestedTokens,
      },
    });
    await this.processQueuedMessage(requestMessage.id);
    this.emit({
      type: 'budget.requested',
      agentId: input.team.identity.id,
      correlationId: input.correlationId,
      data: {
        teamId: input.team.identity.id,
        requestedTokens,
        minimumTokens,
        purpose: 'team_synthesis',
      },
    });
    const allocation = this.budgetMarket.request({
      requesterId: input.team.identity.id,
      parentId: input.team.identity.parentAgentId,
      actorType: 'team',
      correlationId: input.correlationId,
      requestedTokens,
      minimumTokens,
      priority: 'high',
      expectedUtility: 0.85,
      purpose: 'team_synthesis',
      resourceEstimate: {
        tokens: requestedTokens,
        inputTokens: input.promptTokens,
        outputTokens: input.completionTokens,
        contextTokens: input.promptTokens,
      },
      metadata: {
        investmentKind: 'team_synthesis',
        parentUtility: teamCompletionRatio(input.team),
        conflictResolution: input.team.identity.tomProfile.level >= 2 ? 0.9 : 0.55,
        uncertaintyReduction: Math.min(1, input.team.memberAgentIds.length / 4),
        executionRisk: Object.values(input.team.memberStatuses).some(status => status === 'failed') ? 0.65 : 0.2,
        confidence: Object.values(input.team.memberStatuses).some(status => status === 'failed') ? 0.55 : 0.8,
      },
    });
    const responseMessage = await this.enqueueMessage({
      kind: allocation.status === 'granted' ? 'budget.grant' : 'budget.denied',
      sessionId: this.getContext().sessionId,
      from: 'budget.market',
      to: input.team.identity.id,
      correlationId: input.correlationId,
      parentMessageId: requestMessage.id,
      payload: allocation,
      metadata: {
        agentId: input.team.identity.id,
        teamId: input.team.identity.id,
        budgetTokens: allocation.grantedTokens,
      },
    });
    await this.processQueuedMessage(responseMessage.id);
    await this.getContext().queue.ack(requestMessage.id);
    await this.getContext().queue.ack(responseMessage.id);
    this.emit({
      type: allocation.status === 'granted' ? 'budget.granted' : 'budget.denied',
      agentId: input.team.identity.id,
      correlationId: input.correlationId,
      data: {
        teamId: input.team.identity.id,
        allocationId: allocation.id,
        requestedTokens,
        grantedTokens: allocation.grantedTokens,
        reason: allocation.reason,
      },
    });
    return allocation;
  }

  private settleTeamSynthesisBudget(
    teamId: string,
    allocation: BudgetAllocation | undefined,
    actual: number | TokenUsage,
    correlationId: string
  ): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const modelTotalTokens = typeof actual === 'number' ? actual : actual.totalTokens;
    const settled = this.budgetMarket.settle(allocation.id, actual);
    if (!settled) return;
    const latestTeam = this.teams.get(teamId);
    const completedMembers = latestTeam
      ? Object.values(latestTeam.memberStatuses).filter(status => status === 'completed').length
      : 0;
    this.recordBudgetOutcome(allocation.id, {
      success: true,
      conflictResolution: latestTeam?.memberAgentIds.length
        ? completedMembers / latestTeam.memberAgentIds.length
        : 0.5,
      quality: latestTeam ? teamCompletionRatio(latestTeam) : undefined,
      metadata: { teamId, phase: 'synthesis' },
    });
    this.emit({
      type: 'budget.settled',
      agentId: teamId,
      correlationId,
      data: {
        teamId,
        allocationId: allocation.id,
        grantedTokens: allocation.grantedTokens,
        actualTokens: settled.consumedTokens,
        modelTotalTokens,
      },
    });
    this.emit({
      type: 'budget.consumed',
      agentId: teamId,
      correlationId,
      data: {
        teamId,
        allocationId: allocation.id,
        consumedTokens: settled.consumedTokens,
        modelTotalTokens,
        inputTokens: typeof actual === 'number' ? undefined : actual.inputTokens,
        outputTokens: typeof actual === 'number' ? undefined : actual.outputTokens,
        thinkingTokens: typeof actual === 'number' ? undefined : actual.thinkingTokens,
        utilization: settled.utilization,
      },
    });
    if (settled.status === 'exceeded') {
      this.emit({
        type: 'budget.overrun',
        agentId: teamId,
        correlationId,
        data: {
          teamId,
          allocationId: allocation.id,
          grantedTokens: allocation.grantedTokens,
          actualTokens: settled.consumedTokens,
          modelTotalTokens,
        },
      });
      this.emit({
        type: 'budget.exceeded',
        agentId: teamId,
        correlationId,
        data: { allocationId: allocation.id, allocatedTokens: allocation.allocatedTokens, consumedTokens: settled.consumedTokens, modelTotalTokens },
      });
    }
  }

  private releaseTeamSynthesisBudget(
    teamId: string,
    allocation: BudgetAllocation | undefined,
    correlationId: string,
    reason: string
  ): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    const released = this.budgetMarket.release(allocation.id, reason);
    if (released) {
      this.recordBudgetOutcome(allocation.id, {
        success: false,
        error: reason,
        metadata: { teamId, phase: 'synthesis' },
      });
      this.emit({
        type: 'budget.released',
        agentId: teamId,
        correlationId,
        data: { teamId, allocationId: allocation.id, reason },
      });
    }
  }

  private settleAgentBudget(
    agentId: string,
    actual: number | TokenUsage,
    outcome: BudgetOutcome = { success: true }
  ): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    this.ensureUnlimitedAllocationCoverage(
      allocationId,
      this.accountedUsageTokens(actual),
      agentId,
      'agent_run_settlement'
    );
    const allocation = this.budgetMarket.settle(allocationId, actual);
    this.agentBudgetAllocations.delete(agentId);
    this.getContext().manager.getAgentById(agentId)?.setCompletionTokenLimit(undefined);
    if (allocation) {
      this.recordBudgetOutcome(allocationId, {
        ...outcome,
        metadata: { ...outcome.metadata, agentId, phase: 'agent_run' },
      });
      const modelTotalTokens = typeof actual === 'number' ? actual : actual.totalTokens;
      this.emit({
        type: 'budget.settled',
        agentId,
        data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens: allocation.consumedTokens, modelTotalTokens },
      });
      this.emit({
        type: 'budget.consumed',
        agentId,
        data: {
          allocationId,
          consumedTokens: allocation.consumedTokens,
          modelTotalTokens,
          inputTokens: typeof actual === 'number' ? undefined : actual.inputTokens,
          outputTokens: typeof actual === 'number' ? undefined : actual.outputTokens,
          thinkingTokens: typeof actual === 'number' ? undefined : actual.thinkingTokens,
          utilization: allocation.utilization,
        },
      });
      if (allocation.status === 'exceeded') {
        this.emit({
          type: 'budget.overrun',
          agentId,
          data: { allocationId, grantedTokens: allocation.grantedTokens, actualTokens: allocation.consumedTokens, modelTotalTokens },
        });
        this.emit({
          type: 'budget.exceeded',
          agentId,
          data: { allocationId, allocatedTokens: allocation.allocatedTokens, consumedTokens: allocation.consumedTokens, modelTotalTokens },
        });
      }
    }
  }

  private settleDirectBudget(agentId: string, allocation: BudgetAllocation | undefined, usage: TokenUsage, correlationId?: string): void {
    if (!allocation || allocation.status !== 'granted' || !this.budgetMarket) return;
    this.ensureUnlimitedAllocationCoverage(
      allocation.id,
      this.accountedUsageTokens(usage),
      agentId,
      'direct_reasoning_settlement',
      correlationId
    );
    const settled = this.budgetMarket.settle(allocation.id, usage);
    if (!settled) return;
    this.recordBudgetOutcome(allocation.id, {
      success: true,
      metadata: { agentId, phase: 'direct_reasoning' },
    });
    this.emit({
      type: 'budget.consumed',
      agentId,
      correlationId,
      data: {
        allocationId: allocation.id,
        consumedTokens: settled.consumedTokens,
        modelTotalTokens: usage.totalTokens,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        cachedInputTokens: usage.cachedInputTokens,
        utilization: settled.utilization,
      },
    });
    this.emit({
      type: 'budget.settled',
      agentId,
      correlationId,
      data: {
        allocationId: allocation.id,
        allocatedTokens: allocation.allocatedTokens,
        actualTokens: settled.consumedTokens,
        modelTotalTokens: usage.totalTokens,
      },
    });
    if (settled.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId,
        correlationId,
        data: { allocationId: allocation.id, allocatedTokens: allocation.allocatedTokens, consumedTokens: usage.totalTokens },
      });
      this.emit({
        type: 'budget.overrun',
        agentId,
        correlationId,
        data: {
          allocationId: allocation.id,
          grantedTokens: allocation.grantedTokens,
          actualTokens: settled.consumedTokens,
          modelTotalTokens: usage.totalTokens,
        },
      });
    }
  }

  private consumeActiveAgentBudget(
    agentId: string,
    usage: TokenUsage,
    correlationId?: string,
    purpose?: string
  ): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    let before = this.budgetMarket.getAllocation(allocationId);
    if (before) {
      before = this.ensureUnlimitedAllocationCoverage(
        allocationId,
        before.consumedTokens + this.accountedUsageTokens(usage),
        agentId,
        purpose ?? 'active_agent_completion',
        correlationId
      );
    }
    const consumed = this.budgetMarket.consume(allocationId, usage);
    if (!consumed) return;
    const consumedDelta = Math.max(0, consumed.consumedTokens - (before?.consumedTokens ?? 0));
    this.emit({
      type: 'budget.consumed',
      agentId,
      correlationId,
      data: {
        allocationId,
        purpose,
        consumedTokens: consumedDelta,
        cumulativeConsumedTokens: consumed.consumedTokens,
        modelTotalTokens: usage.totalTokens,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        utilization: consumed.utilization,
      },
    });
    if (consumed.status === 'exceeded') {
      this.emit({
        type: 'budget.exceeded',
        agentId,
        correlationId,
        data: {
          allocationId,
          purpose,
          allocatedTokens: consumed.allocatedTokens,
          consumedTokens: consumed.consumedTokens,
          modelTotalTokens: usage.totalTokens,
        },
      });
    }
  }

  private releaseAgentBudget(agentId: string, reason: string): void {
    const allocationId = this.agentBudgetAllocations.get(agentId);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.release(allocationId, reason);
    this.agentBudgetAllocations.delete(agentId);
    this.getContext().manager.getAgentById(agentId)?.setCompletionTokenLimit(undefined);
    if (allocation) {
      this.recordBudgetOutcome(allocationId, { success: false, error: reason, metadata: { agentId } });
      this.emit({ type: 'budget.released', agentId, data: { allocationId, reason } });
    }
  }

  private estimateAgentBudget(archetype: SubAgentArchetype, purpose = ''): number {
    const configured = this.workspaceRuntimeConfig?.budgetMarket.defaultRequestsByArchetype[archetype] ?? ({
      researcher: 2200,
      critic: 1600,
      planner: 1400,
      coder: 2600,
      summarizer: 1000,
      tester: 1800,
      custom: 1800,
    }[archetype]);
    if (!this.taskNeedsWebAccess(purpose)) return configured;
    const requiredSources = Math.max(1, this.requiredWebFetchCount(purpose));
    const webResearchFloor = 8000 + requiredSources * 4000;
    return Math.max(configured, webResearchFloor);
  }

  private defaultBudgetUtility(archetype: SubAgentArchetype): number {
    return ({ researcher: 0.82, critic: 0.72, planner: 0.76, coder: 0.88, summarizer: 0.65, tester: 0.8, custom: 0.7 })[archetype];
  }

  private getRootToolBindings(): ToolBinding[] {
    const webEnabled = this.workspaceRuntimeConfig?.tools.web.enabled !== false;
    return toolRegistry.list()
      .filter(tool => webEnabled || !tool.name.startsWith('web.'))
      .map(tool => this.createToolBinding(tool.name));
  }

  private async executeRuntimeTool(tool: Tool, params: Record<string, unknown>): Promise<ToolResult> {
    try {
      const validation = tool.validate?.(params);
      if (validation && !validation.valid) {
        return { success: false, error: `Validation failed: ${validation.errors?.join(', ')}` };
      }
      return await tool.execute(params);
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  /**
   * Execute one source-changing hypothesis at a time.
   *
   * Deterministic causal repair plans are assembled before optional model
   * planning. Running two mutations from that combined frontier makes the
   * second call operate on an unverified state, duplicates synthesis context,
   * and can overwrite a successful focused patch. Read-only inspections and
   * verification calls remain in the frontier.
   */
  private keepSingleWorkspaceMutationHypothesis(
    plans: PlannedToolCall[],
    agentId?: string,
    options?: { correlationId?: string; nodeId?: string }
  ): PlannedToolCall[] {
    let mutationSeen = false;
    const skipped: PlannedToolCall[] = [];
    const selected = plans.filter(plan => {
      const isMutation = isSuccessfulWorkspaceMutationCall({
        toolName: plan.toolName,
        params: plan.params,
        success: true,
      });
      if (!isMutation) return true;
      if (!mutationSeen) {
        mutationSeen = true;
        return true;
      }
      skipped.push(plan);
      return false;
    });
    if (skipped.length > 0 && agentId) {
      this.emit({
        type: 'tool.plan.mutation_hypothesis.deferred',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options?.correlationId,
        nodeId: options?.nodeId,
        data: {
          reason: 'verify_current_workspace_hypothesis_before_another_mutation',
          deferred: skipped.map(plan => ({
            toolName: plan.toolName,
            params: plan.params,
          })),
        },
      });
    }
    return selected;
  }

  private deferRepeatedNoGainMutationTargets(
    plans: PlannedToolCall[],
    calls: ToolCallRecord[],
    agentId?: string,
    options?: { correlationId?: string; nodeId?: string }
  ): PlannedToolCall[] {
    const deferred: PlannedToolCall[] = [];
    const selected = plans.filter(plan => {
      const mutationPath = plannedWorkspaceMutationPath(plan);
      if (!mutationPath
        || !workspaceTargetNeedsFreshNoGainEvidence(calls, mutationPath)) {
        return true;
      }
      deferred.push(plan);
      return false;
    });
    if (deferred.length > 0 && agentId) {
      this.emit({
        type: 'tool.plan.no_gain_target.deferred',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options?.correlationId,
        nodeId: options?.nodeId,
        data: {
          reason: 'repeated_retained_candidate_requires_fresh_causal_evidence',
          deferred: deferred.map(plan => ({
            toolName: plan.toolName,
            path: plannedWorkspaceMutationPath(plan),
          })),
        },
      });
    }
    return selected;
  }

  private async executeSynthesizedFileForAgent(
    agent: BaseAgent,
    params: Record<string, unknown>,
    task: string,
    groundingCalls: ToolCallRecord[],
    correlationId: string
  ): Promise<ToolResult> {
    const tool = new FsSynthesizeTool();
    const validation = tool.validate(params);
    if (!validation.valid) {
      return { success: false, error: `Validation failed: ${validation.errors?.join(', ')}` };
    }
    const requestedPath = String(params.path);
    const normalizedPath = this.normalizeToolWorkspacePath(requestedPath);
    const directlyObserved = groundingCalls.some(call =>
      this.normalizeToolWorkspacePath(String(call.params.path ?? '')) === normalizedPath
      && (
        call.toolName === 'fs.read'
        || call.toolName === 'fs.write'
        || call.toolName === 'fs.replace'
        || call.toolName === 'fs.synthesize'
      )
    );
    if (!directlyObserved) {
      return {
        success: false,
        error: `fs.synthesize requires a prior direct observation of ${normalizedPath}; inspect the authoritative target with fs.read first`,
      };
    }

    const currentRead = await new FsReadTool(this.workspaceRoot).execute({
      path: requestedPath,
      maxBytes: 80_000,
    });
    const existing = currentRead.success
      ? currentRead.result as {
        content?: unknown;
        truncated?: unknown;
        path?: unknown;
      }
      : undefined;
    if (existing?.truncated === true) {
      return {
        success: false,
        error: `fs.synthesize refuses to replace ${normalizedPath} because its current snapshot exceeds 80000 bytes`,
      };
    }
    const currentContent = typeof existing?.content === 'string'
      ? existing.content
      : undefined;
    if (!currentRead.success
      && !/\b(?:enoent|no such file|cannot find)\b/i.test(currentRead.error ?? '')) {
      return {
        success: false,
        error: `Unable to read the current target snapshot before synthesis: ${currentRead.error ?? 'unknown error'}`,
      };
    }

    const assignedTask = this.compactFileSynthesisAssignment(
      this.agentRestoreSpecs.get(agent.id)?.task?.trim() || task.trim()
    );
    const synthesisStrategy = params.strategy === 'patch' && currentContent !== undefined
      ? 'patch'
      : 'complete';
    const focusedInstructions = String(params.instructions).trim();
    const evidence = this.compactFileSynthesisEvidence(
      groundingCalls,
      normalizedPath,
      synthesisStrategy === 'patch'
        ? focusedInstructions.includes('VERIFIER_PROBE_') ? 6_000 : 10_000
        : 30_000,
      focusedInstructions
    );
    const prompt = [
      synthesisStrategy === 'patch'
        ? '[runtime_workspace_file_patch_synthesis]'
        : '[runtime_workspace_file_synthesis]',
      synthesisStrategy === 'patch'
        ? `Generate a minimal unified diff that repairs workspace file: ${normalizedPath}`
        : `Generate the exact complete UTF-8 contents for workspace file: ${normalizedPath}`,
      '',
      'Immutable assignment:',
      assignedTask || 'Implement the current grounded workspace task.',
      '',
      'Focused implementation instructions:',
      focusedInstructions,
      '',
      'Current target snapshot:',
      currentContent === undefined
        ? '[target does not exist]'
        : currentContent,
      '',
      'Grounded supporting evidence:',
      evidence || '[no additional supporting evidence]',
      '',
      'Output contract:',
      synthesisStrategy === 'patch'
        ? `- Return only a valid unified diff for ${normalizedPath}, with ---/+++ headers and @@ hunks.`
        : '- Return only the complete file contents.',
      synthesisStrategy === 'patch'
        ? '- Use exact unchanged context from the current snapshot. Keep the patch minimal and do not replace the whole file.'
        : '- Do not use Markdown fences, prose, a diff, or tool-call JSON.',
      synthesisStrategy === 'patch'
        ? '- Change only lines directly required by the focused verifier evidence. Do not emit no-op hunks, duplicate existing lines, or alter unrelated constants, algorithms, formatting, identifiers, or output fields.'
        : '',
      '- Do not use Markdown fences, prose, or tool-call JSON.',
      '- Preserve working behavior not contradicted by the assignment or verifier evidence.',
      '- Implement the actual task; do not leave TODOs, placeholders, pseudocode, or NotImplemented stubs.',
      '- Use only paths, schemas, APIs, and requirements supported by the grounded evidence.',
    ].join('\n');
    this.emit({
      type: 'tool.synthesis.started',
      agentId: agent.id,
      sessionId: this.getContext().sessionId,
      correlationId,
      data: {
        path: normalizedPath,
        evidenceCalls: groundingCalls.length,
        promptChars: prompt.length,
        operation: currentContent === undefined ? 'create' : 'replace',
        strategy: synthesisStrategy,
      },
    });
    let generatedPayload = await this.completeAsAgent(
      agent,
      prompt,
      'workspace.file_synthesis',
      correlationId,
      {
        isolatedContext: true,
        temperature: 0,
        maxOutputTokens: 32_000,
      }
    );
    generatedPayload = this.stripSingleMarkdownCodeFence(generatedPayload);
    if (synthesisStrategy === 'patch') {
      generatedPayload = this.extractUnifiedDiffPayload(generatedPayload);
    }
    let materialized = synthesisStrategy === 'patch'
      ? this.applyUnifiedPatchToContent(currentContent!, generatedPayload)
      : { content: generatedPayload };
    let validationError = materialized.error
      ?? await this.synthesizedContentRejectionReason(
        normalizedPath,
        materialized.content ?? '',
        currentContent
      );
    if (validationError && synthesisStrategy === 'patch') {
      this.emit({
        type: 'tool.synthesis.retrying',
        agentId: agent.id,
        sessionId: this.getContext().sessionId,
        correlationId,
        data: {
          path: normalizedPath,
          reason: validationError,
          attempt: 1,
          maxAttempts: 2,
          workspacePreserved: true,
          strategy: synthesisStrategy,
        },
      });
      const recoveryPrompt = [
        '[runtime_workspace_file_patch_recovery]',
        `Generate a new minimal unified diff for ${normalizedPath}.`,
        `The prior patch was rejected before any workspace mutation: ${validationError}`,
        'Correct the rejected anchors against the exact current source excerpts below.',
        'Discard unrelated and no-op hunks from the prior attempt. Do not duplicate a line already present in the snapshot.',
        'Change only the smallest source region directly supported by the focused verifier evidence.',
        '',
        'Focused implementation instructions:',
        focusedInstructions,
        '',
        'Exact current target excerpts around the rejected patch anchors:',
        this.focusedPatchRecoverySnapshot(currentContent!, generatedPayload),
        '',
        'Grounded supporting evidence:',
        this.compactFileSynthesisEvidence(
          groundingCalls,
          normalizedPath,
          6_000,
          focusedInstructions
        ) || '[no additional supporting evidence]',
        '',
        'Rejected patch (failed path; do not repeat it):',
        generatedPayload.length <= 4_000
          ? generatedPayload
          : `${generatedPayload.slice(0, 2_000)}\n[runtime_compacted_rejected_patch]\n${generatedPayload.slice(-2_000)}`,
        '',
        `Return only a valid unified diff for ${normalizedPath}, with exact ---/+++ headers and @@ hunks.`,
      ].join('\n');
      generatedPayload = await this.completeAsAgent(
        agent,
        recoveryPrompt,
        'workspace.file_patch_recovery',
        correlationId,
        {
          isolatedContext: true,
          temperature: 0,
          maxOutputTokens: 12_000,
        }
      );
      generatedPayload = this.stripSingleMarkdownCodeFence(generatedPayload);
      generatedPayload = this.extractUnifiedDiffPayload(generatedPayload);
      materialized = this.applyUnifiedPatchToContent(currentContent!, generatedPayload);
      validationError = materialized.error
        ?? await this.synthesizedContentRejectionReason(
          normalizedPath,
          materialized.content ?? '',
          currentContent
        );
    }
    if (validationError && synthesisStrategy === 'complete') {
      this.emit({
        type: 'tool.synthesis.retrying',
        agentId: agent.id,
        sessionId: this.getContext().sessionId,
        correlationId,
        data: {
          path: normalizedPath,
          reason: validationError,
          attempt: 1,
          maxAttempts: 2,
          workspacePreserved: true,
        },
      });
      const recoveryPrompt = [
        '[runtime_workspace_file_synthesis_recovery]',
        `Generate the exact complete UTF-8 contents for ${normalizedPath}.`,
        `The prior isolated generation was rejected before any workspace mutation: ${validationError}`,
        'Do not request tools. Tool-protocol markup is data corruption in this channel.',
        'All available verifier, input-schema, current-file, and assignment evidence is already attached below.',
        '',
        'Immutable assignment:',
        assignedTask || 'Implement the current grounded workspace task.',
        '',
        'Focused implementation instructions:',
        String(params.instructions).trim(),
        '',
        'Current target snapshot:',
        currentContent === undefined ? '[target does not exist]' : currentContent,
        '',
        'Grounded supporting evidence:',
        evidence || '[no additional supporting evidence]',
        '',
        'Rejected output tail (do not repeat it):',
        generatedPayload.slice(-2_000),
        '',
        'Return source bytes only: no prose, Markdown, diff, JSON, DSML, XML, or tool call.',
      ].join('\n');
      generatedPayload = await this.completeAsAgent(
        agent,
        recoveryPrompt,
        'workspace.file_synthesis_recovery',
        correlationId,
        {
          isolatedContext: true,
          temperature: 0,
          maxOutputTokens: 32_000,
        }
      );
      generatedPayload = this.stripSingleMarkdownCodeFence(generatedPayload);
      materialized = { content: generatedPayload };
      validationError = materialized.error
        ?? await this.synthesizedContentRejectionReason(
          normalizedPath,
          materialized.content ?? '',
          currentContent
        );
    }
    if (validationError) {
      this.emit({
        type: 'tool.synthesis.rejected',
        agentId: agent.id,
        sessionId: this.getContext().sessionId,
        correlationId,
        data: {
          path: normalizedPath,
          reason: validationError,
          existingBytes: currentContent === undefined
            ? 0
            : Buffer.byteLength(currentContent, 'utf8'),
          generatedBytes: Buffer.byteLength(generatedPayload, 'utf8'),
          workspacePreserved: true,
          strategy: synthesisStrategy,
          generatedPreview: generatedPayload.length <= 6_000
            ? generatedPayload
            : `${generatedPayload.slice(0, 3_000)}\n[runtime_rejected_payload_compacted]\n${generatedPayload.slice(-3_000)}`,
        },
      });
      return {
        success: false,
        error: `File synthesis for ${normalizedPath} was rejected before mutation: ${validationError}`,
        result: {
          synthesisRejected: true,
          workspacePreserved: true,
          path: normalizedPath,
          strategy: synthesisStrategy,
          reason: validationError,
          generatedPreview: generatedPayload.length <= 6_000
            ? generatedPayload
            : `${generatedPayload.slice(0, 3_000)}\n[runtime_rejected_payload_compacted]\n${generatedPayload.slice(-3_000)}`,
        },
        metadata: {
          synthesisRejected: true,
          workspacePreserved: true,
          path: normalizedPath,
        },
      };
    }
    const generated = materialized.content!;

    const mutation = currentContent === undefined
      ? await new FsWriteTool(this.workspaceRoot).execute({
        path: requestedPath,
        content: generated,
        mode: 'overwrite',
        createDirectories: true,
      })
      : currentContent.length === 0
        ? await new FsWriteTool(this.workspaceRoot).execute({
          path: requestedPath,
          content: generated,
          mode: 'overwrite',
          createDirectories: false,
        })
        : await new FsReplaceTool(this.workspaceRoot).execute({
          path: requestedPath,
          oldText: currentContent,
          newText: generated,
          expectedReplacements: 1,
        });
    if (!mutation.success) return mutation;
    const result = mutation.result as { path?: unknown; bytes?: unknown } | undefined;
    const operation = currentContent === undefined ? 'create' : 'replace';
    this.emit({
      type: 'tool.synthesis.completed',
      agentId: agent.id,
      sessionId: this.getContext().sessionId,
      correlationId,
      data: {
        path: String(result?.path ?? normalizedPath),
        bytes: Number(result?.bytes ?? Buffer.byteLength(generated, 'utf8')),
        operation,
        strategy: synthesisStrategy,
      },
    });
    return {
      success: true,
      result: {
        path: String(result?.path ?? normalizedPath),
        bytes: Number(result?.bytes ?? Buffer.byteLength(generated, 'utf8')),
        operation,
        synthesized: true,
        strategy: synthesisStrategy,
      },
    };
  }

  private async synthesizedContentRejectionReason(
    filePath: string,
    generated: string,
    currentContent: string | undefined
  ): Promise<string | undefined> {
    if (!generated.trim()) return 'generation returned empty content';
    if (generated === currentContent) return 'generation produced no workspace change';
    if (/^\s*(?:here(?:'s| is)|the (?:complete )?file|implementation:)/i.test(generated)
      && !this.looksLikeSourceFileContent(filePath, generated)) {
      return 'generation returned narrative text instead of file contents';
    }
    return this.validateSynthesizedFileContent(filePath, generated);
  }

  private async validateSynthesizedFileContent(
    filePath: string,
    content: string
  ): Promise<string | undefined> {
    if (content.includes(String.fromCharCode(0))) {
      return 'generated content contains a NUL byte';
    }
    if (/(?:<\|?\|?DSML\|?\|?|<tool_calls?>|<invoke\s+name=|<parameter\s+name=|assistant\s+to=|tool_calls?\s*>)/i.test(content)
      || /[<｜]\s*[｜|]{0,2}DSML[｜|]{0,2}\s*[>｜]/i.test(content)) {
      return 'generated content contains model tool-protocol markup instead of source code';
    }
    if (/^\s*```/m.test(content)) {
      return 'generated content contains an embedded Markdown code fence';
    }
    const extension = path.extname(filePath).toLowerCase();
    if (extension === '.json') {
      try {
        JSON.parse(content);
      } catch (error) {
        return `generated JSON is invalid: ${error instanceof Error ? error.message : String(error)}`;
      }
      return undefined;
    }
    const syntaxCheck = extension === '.py'
      ? {
        executable: this.pythonExecutable(),
        args: ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'],
        label: 'Python',
      }
      : extension === '.js' || extension === '.mjs' || extension === '.cjs'
        ? {
          executable: process.execPath,
          args: ['--input-type=module', '--check', '-'],
          label: 'JavaScript',
        }
        : extension === '.sh'
          ? {
            executable: '/bin/sh',
            args: ['-n'],
            label: 'shell',
          }
          : undefined;
    if (!syntaxCheck) return undefined;
    const result = await new Promise<{ code: number | null; stderr: string; spawnError?: string }>(resolve => {
      const child = spawn(syntaxCheck.executable, syntaxCheck.args, {
        cwd: this.workspaceRoot,
        stdio: ['pipe', 'ignore', 'pipe'],
      });
      let stderr = '';
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', chunk => {
        if (stderr.length < 8_000) stderr += String(chunk);
      });
      child.on('error', error => resolve({
        code: null,
        stderr,
        spawnError: error.message,
      }));
      child.on('close', code => resolve({ code, stderr }));
      child.stdin.end(content);
    });
    if (result.spawnError) {
      return `${syntaxCheck.label} syntax validator could not start: ${result.spawnError}`;
    }
    if (result.code !== 0) {
      return `${syntaxCheck.label} syntax validation failed: ${result.stderr.trim().slice(-4_000) || `exit ${result.code}`}`;
    }
    return undefined;
  }

  private compactFileSynthesisEvidence(
    calls: ToolCallRecord[],
    targetPath: string,
    maxChars: number,
    focusText = ''
  ): string {
    const candidates: Array<{
      section: string;
      score: number;
      index: number;
    }> = [];
    const seen = new Set<string>();
    const targetBase = targetPath.slice(targetPath.lastIndexOf('/') + 1)
      .toLowerCase();
    const targetStem = targetBase.replace(/\.[^.]+$/, '');
    const focusLower = focusText.toLowerCase();
    const ignoredTerms = new Set([
      'apply', 'smallest', 'coherent', 'interface', 'preserving', 'change',
      'newest', 'external', 'verifier', 'feedback', 'authoritative', 'current',
      'source', 'already', 'observed', 'patch', 'preserve', 'unrelated',
      'working', 'benchmark', 'files', 'grounded', 'diagnosis', 'preceding',
      'sequential', 'member', 'runtime', 'compacted',
    ]);
    const semanticTerms = Array.from(new Set(
      [...focusText.matchAll(/\b[A-Za-z_][A-Za-z0-9_.-]{3,}\b/g)]
        .map(match => match[0]!.toLowerCase())
        .filter(term => !ignoredTerms.has(term))
    )).slice(0, 40);
    const semanticHits = (value: string): number => {
      const lower = value.toLowerCase();
      return semanticTerms.reduce(
        (count, term) => count + (lower.includes(term) ? 1 : 0),
        0
      );
    };
    const compactRelevantFile = (
      content: string,
      perFileLimit: number
    ): string => {
      if (content.length <= perFileLimit) return content;
      const lines = content.replace(/\r\n/g, '\n').split('\n');
      const matching = lines
        .map((line, index) => ({
          index,
          hits: semanticHits(line)
            + (targetStem && line.toLowerCase().includes(targetStem) ? 3 : 0)
            + (/(?:assert|raise|expected|actual|fail(?:ed|ure)?)/i.test(
              line
            ) ? 2 : 0),
        }))
        .filter(item => item.hits > 0)
        .sort((left, right) => right.hits - left.hits)
        .slice(0, 8)
        .map(item => item.index)
        .sort((left, right) => left - right);
      if (matching.length === 0) {
        const head = Math.floor(perFileLimit * 0.6);
        return `${content.slice(0, head)}\n[runtime_compacted_file_middle]\n${content.slice(-(perFileLimit - head))}`;
      }
      const selected = new Set<number>();
      for (const center of matching) {
        for (
          let index = Math.max(0, center - 6);
          index <= Math.min(lines.length - 1, center + 6);
          index += 1
        ) {
          selected.add(index);
        }
      }
      let excerpt = [...selected]
        .sort((left, right) => left - right)
        .map(index => `${index + 1}: ${lines[index]}`)
        .join('\n');
      if (excerpt.length > perFileLimit) {
        excerpt = `${excerpt.slice(0, perFileLimit - 38)}\n[relevant file excerpts compacted]`;
      }
      return excerpt;
    };
    for (const [reverseIndex, call] of [...calls].reverse().entries()) {
      const callPath = this.normalizeToolWorkspacePath(String(call.params.path ?? ''));
      const key = call.toolName === 'shell.exec'
        ? `shell:${String(call.params.command ?? '')}`
        : `${call.toolName}:${callPath}`;
      if (seen.has(key)) continue;
      seen.add(key);
      let section = '';
      let score = Math.max(0, 80 - reverseIndex);
      if (call.toolName === 'fs.read' && call.success && callPath !== targetPath) {
        const read = call.result as { content?: unknown; truncated?: unknown } | undefined;
        const content = typeof read?.content === 'string' ? read.content : '';
        if (content) {
          const perFileLimit = callPath.startsWith('.roy/official-verifier/')
            ? 8_000
            : 4_000;
          const compact = compactRelevantFile(content, perFileLimit);
          section = `File ${callPath}${read?.truncated ? ' (observed prefix)' : ''}:\n${compact}`;
          const pathLower = callPath.toLowerCase();
          const hits = semanticHits(`${callPath}\n${content}`);
          score += hits * 90;
          if (callPath.startsWith('.roy/official-verifier/')) score += 1_000;
          if (targetStem && pathLower.includes(targetStem)) score += 500;
          if (focusLower.includes(pathLower)
            || focusLower.includes(pathLower.slice(pathLower.lastIndexOf('/') + 1))) {
            score += 650;
          }
        }
      } else if (call.toolName === 'fs.list' && call.success) {
        const entries = (call.result as { entries?: unknown } | undefined)?.entries;
        if (Array.isArray(entries)) {
          section = `Workspace listing:\n${entries.filter(item => typeof item === 'string').slice(0, 160).join('\n')}`;
          score += 20 + semanticHits(section) * 40;
        }
      } else if (call.toolName === 'fs.search' && call.success) {
        const result = call.result as { matches?: unknown } | undefined;
        if (Array.isArray(result?.matches)) {
          section = `Workspace search evidence:\n${JSON.stringify(result.matches.slice(0, 60))}`;
          score += 300 + semanticHits(section) * 80;
        }
      } else if (call.toolName === 'shell.exec') {
        const shell = call.result as {
          command?: unknown;
          stdout?: unknown;
          stderr?: unknown;
          exitCode?: unknown;
          verifierDiagnostics?: unknown;
          candidateRetention?: unknown;
          candidateRollback?: unknown;
          regressionRollback?: unknown;
        } | undefined;
        const output = [
          String(shell?.stdout ?? ''),
          String(shell?.stderr ?? ''),
          shell?.verifierDiagnostics
            ? `Verifier diagnostics:\n${JSON.stringify(shell.verifierDiagnostics)}`
            : '',
          shell?.candidateRetention
            ? `Retained verifier candidate for coordinated hard-gate composition:\n${JSON.stringify(shell.candidateRetention)}`
            : '',
          shell?.candidateRollback
            ? `Rejected verifier candidate:\n${JSON.stringify(shell.candidateRollback)}`
            : '',
          shell?.regressionRollback
            ? `Regression rollback:\n${JSON.stringify(shell.regressionRollback)}`
            : '',
          String(call.error ?? ''),
        ].filter(Boolean).join('\n');
        const verifierProbeDuplicatedByFocus =
          focusText.includes('VERIFIER_PROBE_EVIDENCE_VERSION')
          && (
            String(shell?.command ?? call.params.command ?? '').includes('ROY_VERIFIER_PROBE=1')
            || output.includes('VERIFIER_PROBE_EVIDENCE_VERSION')
          );
        if (verifierProbeDuplicatedByFocus) continue;
        if (output) {
          const command = this.compactShellCommandForEvidence(
            String(shell?.command ?? call.params.command ?? '')
          );
          section = [
            `Command ${command} (exit ${String(shell?.exitCode ?? 'unknown')}):`,
            output.length <= 8_000
              ? output
              : `${output.slice(0, 3_000)}\n[runtime_compacted_command_middle]\n${output.slice(-5_000)}`,
          ].join('\n');
          const exitCode = Number(shell?.exitCode);
          score += exitCode !== 0 || !call.success ? 900 : 180;
          if (/\b(?:pytest|test|verify|check|lint|build)\b/i.test(command)) {
            score += 350;
          }
          score += semanticHits(`${command}\n${output}`) * 100;
        }
      }
      if (!section) continue;
      candidates.push({ section, score, index: calls.length - reverseIndex - 1 });
    }
    const ranked = candidates
      .sort((left, right) => right.score - left.score || right.index - left.index)
      .slice(0, 8)
      .sort((left, right) => left.index - right.index);
    const sections: string[] = [];
    let remaining = maxChars;
    for (const candidate of ranked) {
      if (remaining <= 0) break;
      const bounded = candidate.section.slice(0, remaining);
      sections.push(bounded);
      remaining -= bounded.length + 2;
    }
    return sections.join('\n\n');
  }

  private compactFileSynthesisAssignment(task: string): string {
    const withoutDerivedCaches = task
      .replace(/<team_step_cache>[\s\S]*?<\/team_step_cache>/gi, '')
      .replace(/<system_communication_context\b[\s\S]*?<\/system_communication_context>/gi, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    return this.compactDelegatedTask(withoutDerivedCaches, 4_000);
  }

  private focusedPatchRecoverySnapshot(
    currentContent: string,
    rejectedPatch: string,
    maxChars = 12_000
  ): string {
    if (currentContent.length <= maxChars) return currentContent;
    const sourceLines = currentContent.replace(/\r\n/g, '\n').split('\n');
    const patchLines = rejectedPatch.replace(/\r\n/g, '\n').split('\n');
    const occurrenceCounts = new Map<string, number>();
    for (const line of sourceLines) {
      occurrenceCounts.set(line, (occurrenceCounts.get(line) ?? 0) + 1);
    }
    const anchors = patchLines
      .filter(line => (line.startsWith(' ') || line.startsWith('-'))
        && line.slice(1).trim().length >= 4)
      .map(line => line.slice(1))
      .filter((line, index, lines) => lines.indexOf(line) === index)
      .sort((left, right) =>
        (occurrenceCounts.get(left) ?? 0) - (occurrenceCounts.get(right) ?? 0)
        || right.length - left.length
      );
    const centers: number[] = [];
    for (const anchor of anchors) {
      const index = sourceLines.indexOf(anchor);
      if (index >= 0 && !centers.some(center => Math.abs(center - index) <= 20)) {
        centers.push(index);
      }
      if (centers.length >= 4) break;
    }
    if (centers.length === 0) {
      for (const line of patchLines) {
        const match = /^@@ -(\d+)/.exec(line);
        const declaredIndex = match ? Number(match[1]) - 1 : -1;
        if (declaredIndex >= 0 && declaredIndex < sourceLines.length) {
          centers.push(declaredIndex);
        }
      }
    }
    if (centers.length === 0) {
      const head = Math.floor(maxChars * 0.5);
      return [
        currentContent.slice(0, head),
        '[current source middle omitted: rejected patch contained no resolvable anchor]',
        currentContent.slice(-(maxChars - head)),
      ].join('\n');
    }
    const ranges = centers
      .map(center => ({
        start: Math.max(0, center - 45),
        end: Math.min(sourceLines.length, center + 46),
      }))
      .sort((left, right) => left.start - right.start)
      .reduce<Array<{ start: number; end: number }>>((merged, range) => {
        const previous = merged.at(-1);
        if (previous && range.start <= previous.end + 8) {
          previous.end = Math.max(previous.end, range.end);
        } else {
          merged.push({ ...range });
        }
        return merged;
      }, []);
    const excerpts: string[] = [];
    let remaining = maxChars;
    for (const range of ranges) {
      const header = `[exact current source excerpt lines ${range.start + 1}-${range.end}]`;
      const body = sourceLines.slice(range.start, range.end).join('\n');
      const section = `${header}\n${body}\n[end exact current source excerpt]`;
      if (section.length <= remaining) {
        excerpts.push(section);
        remaining -= section.length + 2;
        continue;
      }
      if (excerpts.length === 0) {
        excerpts.push(`${section.slice(0, Math.max(1, remaining - 36))}\n[excerpt compacted at character budget]`);
      }
      break;
    }
    return excerpts.join('\n\n');
  }

  private compactRecoveryAssignment(task: string): string {
    const immutableHead = task
      .replace(
        /\n(?:---\s*\n)?##\s+VERIFICATION FAILED[\s\S]*$/i,
        ''
      )
      .replace(/<official_verifier_feedback>[\s\S]*?<\/official_verifier_feedback>/gi, '')
      .replace(/<team_step_cache>[\s\S]*?<\/team_step_cache>/gi, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    return this.compactDelegatedTask(immutableHead || task.trim(), 5_000);
  }

  private stripSingleMarkdownCodeFence(content: string): string {
    const trimmed = content.trim();
    const fenced = /^```[A-Za-z0-9_+.-]*\s*\n([\s\S]*?)\n```\s*$/.exec(trimmed);
    return fenced ? fenced[1]! : content;
  }

  private extractUnifiedDiffPayload(content: string): string {
    const lines = content.replace(/\r\n/g, '\n').trim().split('\n');
    const headerIndex = lines.findIndex((line, index) =>
      line.startsWith('--- ')
      && lines.slice(index + 1, index + 4).some(candidate => candidate.startsWith('+++ '))
    );
    if (headerIndex < 0) return content;
    const payload = lines.slice(headerIndex);
    const closingFenceIndex = payload.findIndex((line, index) =>
      index > 0 && /^```(?:[A-Za-z0-9_+.-]*)?\s*$/.test(line.trim())
    );
    return (closingFenceIndex < 0
      ? payload
      : payload.slice(0, closingFenceIndex)
    ).join('\n');
  }

  private applyUnifiedPatchToContent(
    currentContent: string,
    patch: string
  ): { content?: string; error?: string } {
    const patchLines = patch.replace(/\r\n/g, '\n').trim().split('\n');
    const firstHunk = patchLines.findIndex(line => line.startsWith('@@ '));
    if (firstHunk < 0) {
      return { error: 'focused patch generation returned no unified-diff hunks' };
    }
    const preamble = patchLines.slice(0, firstHunk).filter(Boolean);
    if (!preamble.some(line => line.startsWith('--- '))
      || !preamble.some(line => line.startsWith('+++ '))) {
      return { error: 'focused patch generation omitted the ---/+++ unified-diff headers' };
    }

    const hadTrailingNewline = currentContent.endsWith('\n');
    const currentLines = currentContent.replace(/\r\n/g, '\n').split('\n');
    if (hadTrailingNewline) currentLines.pop();
    const hunks: Array<{
      oldStart: number;
      declaredIndex: number;
      sourceIndex: number;
      sourcePattern: string[];
      lines: string[];
    }> = [];
    let patchIndex = firstHunk;
    while (patchIndex < patchLines.length) {
      const header = patchLines[patchIndex]!;
      if (!header.trim()) {
        patchIndex += 1;
        continue;
      }
      const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(header);
      if (!match) {
        return { error: `focused patch contains invalid hunk header: ${header.slice(0, 200)}` };
      }
      const oldStart = Number(match[1]);
      const hunkEnd = (() => {
        for (let index = patchIndex + 1; index < patchLines.length; index += 1) {
          if (patchLines[index]!.startsWith('@@ ')) return index;
        }
        return patchLines.length;
      })();
      const hunkLines = patchLines
        .slice(patchIndex + 1, hunkEnd)
        .filter(line => line !== '\\ No newline at end of file')
        // A blank context line in a unified diff should carry an invisible
        // leading space. Models frequently omit only that prefix. Normalize
        // the unambiguous case and still validate it against the source below.
        .map(line => line === '' ? ' ' : line);
      const sourcePatternFor = (lines: string[]): string[] => lines
        .filter(line => line[0] === ' ' || line[0] === '-')
        .map(line => line.slice(1));
      let effectiveHunkLines = hunkLines;
      let sourcePattern = sourcePatternFor(effectiveHunkLines);
      const declaredIndex = Math.max(0, oldStart - 1);
      const matchesAt = (pattern: string[], index: number): boolean =>
        index >= 0
        && index + pattern.length <= currentLines.length
        && pattern.every((line, offset) => currentLines[index + offset] === line);
      const matchingIndices = (pattern: string[]): number[] => {
        const candidates: number[] = [];
        for (
          let index = 0;
          index + pattern.length <= currentLines.length;
          index += 1
        ) {
          if (matchesAt(pattern, index)) candidates.push(index);
        }
        return candidates;
      };
      let hunkSourceIndex = declaredIndex;
      if (!matchesAt(sourcePattern, hunkSourceIndex) && sourcePattern.length > 0) {
        let candidates = matchingIndices(sourcePattern);
        // Match the behavior of a conservative patch fuzz factor: a model may
        // carry stale context lines at a hunk edge even though its deletion
        // anchor is exact. Only context lines may be discarded; wider fuzz
        // additionally requires one unique exact deletion anchor.
        if (candidates.length === 0) {
          const leadingContextLines = hunkLines.findIndex(line => line[0] !== ' ');
          const trailingContextLines = [...hunkLines]
            .reverse()
            .findIndex(line => line[0] !== ' ');
          const maximumEdgeTrim = Math.max(0, leadingContextLines)
            + Math.max(0, trailingContextLines);
          outer: for (let totalTrim = 1; totalTrim <= maximumEdgeTrim; totalTrim += 1) {
            for (let leadingTrim = 0; leadingTrim <= totalTrim; leadingTrim += 1) {
              const trailingTrim = totalTrim - leadingTrim;
              if (leadingTrim > Math.max(0, leadingContextLines)
                || trailingTrim > Math.max(0, trailingContextLines)) {
                continue;
              }
              if (!hunkLines.slice(0, leadingTrim).every(line => line[0] === ' ')) {
                continue;
              }
              if (trailingTrim > 0
                && !hunkLines.slice(-trailingTrim).every(line => line[0] === ' ')) {
                continue;
              }
              const end = trailingTrim > 0
                ? hunkLines.length - trailingTrim
                : hunkLines.length;
              const fuzzyLines = hunkLines.slice(leadingTrim, end);
              const fuzzyPattern = sourcePatternFor(fuzzyLines);
              if (fuzzyPattern.length === 0) continue;
              const fuzzyCandidates = matchingIndices(fuzzyPattern);
              if (fuzzyCandidates.length === 0) continue;
              // Wider fuzz is allowed only when an exact deletion anchor
              // remains and resolves uniquely in the immutable snapshot.
              if (totalTrim > 4
                && (!fuzzyLines.some(line => line[0] === '-')
                  || fuzzyCandidates.length !== 1)) {
                continue;
              }
              effectiveHunkLines = fuzzyLines;
              sourcePattern = fuzzyPattern;
              candidates = fuzzyCandidates;
              break outer;
            }
          }
        }
        if (candidates.length === 0) {
          const changeBlocks: string[][] = [];
          let activeBlock: string[] = [];
          for (const line of hunkLines) {
            if (line[0] === '+' || line[0] === '-') {
              activeBlock.push(line);
            } else if (activeBlock.length > 0) {
              changeBlocks.push(activeBlock);
              activeBlock = [];
            }
          }
          if (activeBlock.length > 0) changeBlocks.push(activeBlock);
          const resolvedBlocks = changeBlocks.map(lines => {
            const deletionPattern = lines
              .filter(line => line[0] === '-')
              .map(line => line.slice(1));
            const indices = deletionPattern.length > 0
              ? matchingIndices(deletionPattern)
              : [];
            return {
              lines,
              deletionPattern,
              indices,
            };
          });
          if (resolvedBlocks.length > 0
            && resolvedBlocks.every(block => block.indices.length === 1)) {
            for (const block of resolvedBlocks) {
              hunks.push({
                oldStart,
                declaredIndex,
                sourceIndex: block.indices[0]!,
                sourcePattern: block.deletionPattern,
                lines: block.lines,
              });
            }
            patchIndex = hunkEnd;
            continue;
          }
        }
        if (candidates.length === 0) {
          return {
            error: `focused patch source anchor does not match the current snapshot near line ${oldStart}`,
          };
        }
        candidates.sort((left, right) =>
          Math.abs(left - declaredIndex) - Math.abs(right - declaredIndex)
          || left - right
        );
        hunkSourceIndex = candidates[0]!;
      }
      if (hunkSourceIndex > currentLines.length) {
        return { error: `focused patch hunk starts outside the current snapshot at line ${oldStart}` };
      }
      hunks.push({
        oldStart,
        declaredIndex,
        sourceIndex: hunkSourceIndex,
        sourcePattern,
        lines: effectiveHunkLines,
      });
      patchIndex = hunkEnd;
    }

    // Models occasionally preserve exact hunk context while emitting stale line
    // numbers or putting hunks in declared rather than actual source order.
    // Resolve every hunk against the immutable snapshot first, then apply the
    // non-overlapping exact anchors in real source order.
    hunks.sort((left, right) =>
      left.sourceIndex - right.sourceIndex
      || left.declaredIndex - right.declaredIndex
    );
    const output: string[] = [];
    let sourceIndex = 0;
    let changedLines = 0;
    for (const hunk of hunks) {
      if (hunk.sourceIndex < sourceIndex) {
        return {
          error: `focused patch contains overlapping source anchors near line ${hunk.oldStart}`,
        };
      }
      output.push(...currentLines.slice(sourceIndex, hunk.sourceIndex));
      sourceIndex = hunk.sourceIndex;
      for (const line of hunk.lines) {
        const prefix = line[0];
        const text = line.slice(1);
        if (prefix === ' ') {
          if (currentLines[sourceIndex] !== text) {
            return {
              error: `focused patch context does not match current snapshot at line ${sourceIndex + 1}`,
            };
          }
          output.push(text);
          sourceIndex += 1;
        } else if (prefix === '-') {
          if (currentLines[sourceIndex] !== text) {
            return {
              error: `focused patch deletion does not match current snapshot at line ${sourceIndex + 1}`,
            };
          }
          sourceIndex += 1;
          changedLines += 1;
        } else if (prefix === '+') {
          output.push(text);
          changedLines += 1;
        } else {
          return { error: `focused patch contains invalid hunk content: ${line.slice(0, 200)}` };
        }
      }
    }
    if (changedLines === 0) return { error: 'focused patch contains no source changes' };
    output.push(...currentLines.slice(sourceIndex));
    return {
      content: `${output.join('\n')}${hadTrailingNewline ? '\n' : ''}`,
    };
  }

  private looksLikeSourceFileContent(filePath: string, content: string): boolean {
    const extension = path.extname(filePath).toLowerCase();
    if (extension === '.py') {
      return /(?:^|\n)\s*(?:from |import |def |class |if __name__|[A-Za-z_]\w*\s*=)/m.test(content);
    }
    if (['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'].includes(extension)) {
      return /(?:^|\n)\s*(?:import |export |const |let |var |function |class |interface |type )/m.test(content);
    }
    return true;
  }

  private getRootSkillBindings(): SkillBinding[] {
    return skillRegistry.list().map(skill => this.createSkillBinding(skill.name));
  }

  private getDefaultToolBindings(archetype: SubAgentArchetype): ToolBinding[] {
    const namesByArchetype: Record<SubAgentArchetype, string[]> = {
      researcher: ['fs.list', 'fs.read', 'fs.search'],
      critic: ['fs.read', 'fs.search'],
      planner: [],
      coder: ['fs.list', 'fs.read', 'fs.search', 'fs.replace', 'fs.write', 'fs.synthesize', 'shell.exec'],
      summarizer: [],
      tester: ['fs.list', 'fs.read', 'fs.search', 'shell.exec'],
      custom: [],
    };
    const configured = this.workspaceRuntimeConfig?.agents.defaultToolsByArchetype[archetype];
    return (configured ?? namesByArchetype[archetype]).map(name => this.createToolBinding(name));
  }

  private getToolBindingsForTask(archetype: SubAgentArchetype, task: string): ToolBinding[] {
    const bindings = this.getDefaultToolBindings(archetype);
    if (!this.taskNeedsWebAccess(task) || this.workspaceRuntimeConfig?.tools.web.enabled === false) return bindings;
    const names = new Set(bindings.map(binding => binding.name));
    for (const name of ['web.search', 'web.fetch']) {
      if (!names.has(name) && toolRegistry.has(name)) bindings.push(this.createToolBinding(name));
    }
    return bindings;
  }

  private getAutomaticallyApprovedToolBindings(
    archetype: SubAgentArchetype,
    task?: string,
    requestedTools: string[] = []
  ): ToolBinding[] {
    const approval = this.workspaceRuntimeConfig?.tools.approval;
    const bindings = task ? this.getToolBindingsForTask(archetype, task) : this.getDefaultToolBindings(archetype);
    const names = new Set(bindings.map(binding => binding.name));
    for (const name of requestedTools) {
      if (name.startsWith('web.') && (!task || !this.taskNeedsWebAccess(task))) {
        continue;
      }
      if (!names.has(name) && toolRegistry.has(name)) {
        bindings.push(this.createToolBinding(name));
        names.add(name);
      }
    }
    return bindings.filter(binding => {
      if (!approval) return binding.permission === 'read_only';
      const decision = approval.overrides[binding.name]
        ?? (binding.permission === 'read_only'
          ? approval.readOnly
          : binding.permission === 'write' ? approval.write : approval.execute);
      return decision === 'auto';
    });
  }

  private taskNeedsWebAccess(task: string): boolean {
    // Long-horizon workspace continuations append verifier output, install logs,
    // warnings, and cached execution evidence to the original task. URLs inside
    // that evidence (for example a pytest warning link) are observations, not a
    // request to abandon the local repair path and browse the web. Route tools
    // from the authored task section only; downstream agents still receive the
    // complete feedback as execution context.
    const runtimeOriginalTask = task.match(
      /(?:^|\n)Original task:\s*\n([\s\S]*?)(?=\n(?:<acceptance_checklist>|Execution attempt \d+|Delegated findings and proposals:|Prior tool evidence:|Execution contract:|Recovery contract:|Audit contract:|<delegated_execution_state>|<execution_resume_ledger>))/
    )?.[1];
    const authoredTask = this.stripPassiveReferenceSections((runtimeOriginalTask ?? task)
      .split(/\n(?:---\s*\n)?##\s+VERIFICATION FAILED\b/i, 1)[0]
      .split(/\n<official_verifier_feedback>/i, 1)[0]
      .split(/\n(?:latest command output|terminal output|command output)\s*:/i, 1)[0]);
    const lower = authoredTask.toLowerCase();
    return /\b(?:open|read|fetch|visit|consult|browse|summari[sz]e|analy[sz]e|review|research|compare|verify)\b[\s\S]{0,240}https?:\/\//i.test(authoredTask)
      || /\b(?:web|internet|online|website|browse|news|up-to-date|citations?|official documentation|public documentation)\b/.test(lower)
      || /\bsearch\b[\s\S]{0,100}\b(?:external|official|public)?\s*(?:sources?|documentation|websites?|internet|web)\b/.test(lower)
      || /\b(?:external|official|public)?\s*(?:sources?|documentation|websites?|internet|web)\b[\s\S]{0,100}\bsearch\b/.test(lower)
      || /\blatest\b[\s\S]*\b(?:documentation|release|version|news|announcement|api)\b/.test(lower)
      || /\b(?:research|compare|verify)\b[\s\S]*\b(?:external|official|independent)\s+sources?\b/.test(lower);
  }

  private taskRequiresIndependentWebEvidence(
    task: string,
    archetype?: SubAgentArchetype,
    descriptor = ''
  ): boolean {
    if (!this.taskNeedsWebAccess(task)) return false;
    if (task.includes('[runtime_independent_web_evidence]')) return true;
    const intent = `${descriptor}\n${task}`.toLowerCase();
    return archetype === 'critic'
      || archetype === 'tester'
      || /\b(?:fact[- ]?check(?:er|ing)?|factuality|reviewer|critic|verifier|validator|cross[- ]?check|corroborat|independent(?:ly)?\s+(?:check|verify|research)|precise canonical|supporting sources?|source[- ]grounded|scope[- ]sensitive)\b/.test(
        intent
      );
  }

  private stripPassiveReferenceSections(task: string): string {
    return task.replace(
      /(?:^|\n)(?:#{1,6}\s*)?(?:(?:official\s*\/\s*public|official|public)\s+)?references?\s*:?\s*\n[\s\S]*$/i,
      ''
    ).trimEnd();
  }

  private getDefaultSkillBindings(archetype: SubAgentArchetype): SkillBinding[] {
    const namesByArchetype: Record<SubAgentArchetype, string[]> = {
      researcher: ['use_tool_when_needed', 'delegate_to_subagent'],
      critic: ['use_tool_when_needed', 'delegate_to_subagent'],
      planner: ['delegate_to_subagent'],
      coder: ['use_tool_when_needed', 'delegate_to_subagent'],
      summarizer: [],
      tester: ['use_tool_when_needed', 'delegate_to_subagent'],
      custom: [],
    };
    const configured = this.workspaceRuntimeConfig?.agents.defaultSkillsByArchetype[archetype];
    return (configured ?? namesByArchetype[archetype]).map(name => this.createSkillBinding(name));
  }

  private createToolBinding(name: string): ToolBinding {
    const permission: ToolBinding['permission'] = name === 'shell.exec'
      ? 'execute'
      : name === 'fs.write' || name === 'fs.replace' || name === 'fs.synthesize'
        ? 'write'
        : 'read_only';
    return {
      name,
      enabled: true,
      permission,
      constraints: name === 'shell.exec'
        ? {
            allowlistedCommands: this.workspaceRuntimeConfig?.tools.shell.mode === 'unrestricted'
              ? ['*']
              : ['npm', 'node', 'git', 'pwd', 'ls', 'cat', 'rg'],
            maxCalls: this.workspaceRuntimeConfig?.tools.shell.maxCallsPerAgent ?? 5,
          }
        : { allowedPaths: [this.workspaceRoot] },
    };
  }

  private createSkillBinding(name: string): SkillBinding {
    const skill = skillRegistry.get(name);
    return {
      name,
      enabled: true,
      description: skill?.description ?? name,
      constraints: name === 'delegate_to_subagent'
        ? { maxCalls: 5, requiresApproval: false }
        : undefined,
    };
  }

  private normalizeToolBindings(input: SpawnAgentSpec['tools'] | SpawnCommandPayload['tools'], archetype: SubAgentArchetype): ToolBinding[] {
    const raw = input ?? this.getDefaultToolBindings(archetype);
    return raw.map(item => typeof item === 'string' ? this.createToolBinding(item) : item);
  }

  private normalizeSkillBindings(input: SpawnAgentSpec['skills'] | SpawnCommandPayload['skills'], archetype: SubAgentArchetype): SkillBinding[] {
    const raw = input ?? this.getDefaultSkillBindings(archetype);
    return raw.map(item => typeof item === 'string' ? this.createSkillBinding(item) : item);
  }

  private getDefaultSpawnPolicy(role: 'root' | 'subagent' | string, archetype?: SubAgentArchetype): AgentSpawnPolicy {
    const isRoot = role === 'root';
    const delegation = this.workspaceRuntimeConfig?.delegation;
    const archetypeSkills = archetype ? this.getDefaultSkillBindings(archetype).map(binding => binding.name) : [];
    const canSpawn = delegation?.enabled !== false && (isRoot || archetypeSkills.includes('delegate_to_subagent'));
    return {
      canSpawn,
      maxChildren: delegation?.maxChildrenPerParent ?? 5,
      maxDepth: delegation?.maxDepth ?? 3,
      maxTotalAgentsPerTurn: delegation?.maxTotalAgentsPerTurn ?? 10,
      allowCustomAgents: delegation?.allowCustomAgents ?? true,
      budgetAware: delegation?.budgetAware ?? true,
      allowedStates: isRoot
        ? ['S_solo', 'S_delegate_planning', 'S_spawn_subagents']
        : ['S_planning', 'S_delegating'],
    };
  }

  private getDefaultMemoryScope(role: string): AgentMemoryScope {
    return {
      public: true,
      private: true,
      parentContext: role !== 'root',
      sessionWindowTurns: role === 'root' ? 10 : 5,
    };
  }

  private mergeSpawnPolicy(base: AgentSpawnPolicy, override?: Partial<AgentSpawnPolicy>): AgentSpawnPolicy {
    return {
      ...base,
      ...override,
      allowedStates: override?.allowedStates ?? base.allowedStates,
    };
  }

  private computeAllowedChildren(policy: AgentSpawnPolicy): number {
    if (!policy.canSpawn) return 0;
    const budget = this.getBudgetState();
    if (!policy.budgetAware || budget.mode === 'unlimited') return policy.maxChildren;
    const remaining = budget.remainingTokens ?? 0;
    if (remaining < 1000) return 0;
    if (remaining < 3000) return Math.min(policy.maxChildren, 1);
    if (remaining < 8000) return Math.min(policy.maxChildren, 2);
    return policy.maxChildren;
  }

  private getAgentDepth(agentId: string): number {
    const ctx = this.getContext();
    let depth = 0;
    let current = ctx.manager.getAgentById(agentId)?.getIdentity().parentId;
    while (current) {
      depth += 1;
      current = ctx.manager.getAgentById(current)?.getIdentity().parentId;
    }
    return depth;
  }

  private getTurnAgentCount(correlationId?: string): number {
    return correlationId ? this.turnAgentCounts.get(correlationId) ?? 0 : 0;
  }

  private getMaxTotalAgentsPerTurn(parentId: string): number {
    return this.getAgentPolicy(parentId)?.spawnPolicy.maxTotalAgentsPerTurn
      ?? this.workspaceRuntimeConfig?.delegation.maxTotalAgentsPerTurn
      ?? 10;
  }

  private getRemainingTotalAgentsForTurn(parentId: string, correlationId?: string): number {
    return Math.max(0, this.getMaxTotalAgentsPerTurn(parentId) - this.getTurnAgentCount(correlationId));
  }

  private getOutstandingTeamReservations(
    parentId: string | undefined,
    correlationId?: string,
    excludingTeamId?: string
  ): number {
    let total = 0;
    for (const [teamId, reservation] of this.teamSpawnReservations) {
      if (teamId === excludingTeamId
        || (parentId !== undefined && reservation.parentId !== parentId)
        || (correlationId && reservation.correlationId !== correlationId)) {
        continue;
      }
      total += Math.max(0, reservation.plannedMembers - reservation.consumedMembers);
    }
    return total;
  }

  private releaseTeamSpawnReservation(teamId: string, reason: string): void {
    const reservation = this.teamSpawnReservations.get(teamId);
    if (!reservation) return;
    this.teamSpawnReservations.delete(teamId);
    this.emit({
      type: 'team.capacity.released',
      agentId: teamId,
      sessionId: this.ctx?.sessionId,
      correlationId: reservation.correlationId,
      data: {
        teamId,
        parentAgentId: reservation.parentId,
        plannedMembers: reservation.plannedMembers,
        consumedMembers: reservation.consumedMembers,
        reason,
      },
    });
  }

  private recordTurnAgentCreated(correlationId?: string): void {
    if (!correlationId) return;
    this.turnAgentCounts.set(correlationId, this.getTurnAgentCount(correlationId) + 1);
  }

  private validateSpawnPolicy(input: {
    parentId: string;
    archetype: SubAgentArchetype;
    tools: ToolBinding[];
    skills: SkillBinding[];
    correlationId?: string;
    teamId?: string;
  }): {
    allowed: boolean;
    reason?: string;
    currentChildren: number;
    allowedChildren: number;
    depth: number;
  } {
    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(input.parentId);
    if (!parent) {
      return { allowed: false, reason: 'parent_not_found', currentChildren: 0, allowedChildren: 0, depth: 0 };
    }

    const parentInfo = parent.getInfo();
    const parentBindings = this.agentBindings.get(input.parentId) ?? {
      tools: [],
      skills: [],
      memoryScope: this.getDefaultMemoryScope(parentInfo.role),
      spawnPolicy: this.getDefaultSpawnPolicy(parentInfo.role === 'root' ? 'root' : 'subagent'),
    };
    const currentChildren = this.getChildren(input.parentId).length;
    const dynamicAllowedChildren = this.computeAllowedChildren(parentBindings.spawnPolicy);
    const reservation = input.teamId ? this.teamSpawnReservations.get(input.teamId) : undefined;
    const matchingReservation = reservation
      && reservation.parentId === input.parentId
      ? reservation
      : undefined;
    const outstandingChildReservations = this.getOutstandingTeamReservations(
      input.parentId,
      undefined,
      input.teamId
    );
    const allowedChildren = matchingReservation
      ? Math.max(dynamicAllowedChildren, matchingReservation.allowedChildren)
      : Math.max(0, dynamicAllowedChildren - outstandingChildReservations);
    const depth = this.getAgentDepth(input.parentId);
    const nextDepth = depth + 1;

    if (parentInfo.state === 'failed' || parentInfo.state === 'stopped') {
      return { allowed: false, reason: 'invalid_fsm_state', currentChildren, allowedChildren, depth };
    }

    if (!parentBindings.spawnPolicy.canSpawn) {
      return { allowed: false, reason: 'spawn_disabled_for_parent', currentChildren, allowedChildren, depth };
    }
    const parentFsmState = input.parentId === 'root'
      ? ctx.fsm.getState()
      : this.agentFsms.get(input.parentId)?.getState();
    if (!parentFsmState || !parentBindings.spawnPolicy.allowedStates.includes(parentFsmState)) {
      return { allowed: false, reason: 'invalid_fsm_state', currentChildren, allowedChildren, depth };
    }
    if (currentChildren >= allowedChildren) {
      return { allowed: false, reason: 'max_children_exceeded', currentChildren, allowedChildren, depth };
    }
    const outstandingTurnReservations = this.getOutstandingTeamReservations(
      undefined,
      input.correlationId,
      input.teamId
    );
    const turnCapacityAvailable = matchingReservation
      ? matchingReservation.consumedMembers < matchingReservation.plannedMembers
      : this.getRemainingTotalAgentsForTurn(input.parentId, input.correlationId) - outstandingTurnReservations > 0;
    if (!turnCapacityAvailable) {
      return { allowed: false, reason: 'max_total_agents_per_turn_exceeded', currentChildren, allowedChildren, depth };
    }
    if (nextDepth > parentBindings.spawnPolicy.maxDepth) {
      return { allowed: false, reason: 'max_depth_exceeded', currentChildren, allowedChildren, depth };
    }
    if (input.archetype === 'custom' && !parentBindings.spawnPolicy.allowCustomAgents) {
      return { allowed: false, reason: 'custom_agents_not_allowed', currentChildren, allowedChildren, depth };
    }
    for (const binding of input.tools) {
      if (!toolRegistry.has(binding.name)) {
        return { allowed: false, reason: `tool_not_registered:${binding.name}`, currentChildren, allowedChildren, depth };
      }
    }
    for (const binding of input.skills) {
      if (!skillRegistry.has(binding.name)) {
        return { allowed: false, reason: `skill_not_registered:${binding.name}`, currentChildren, allowedChildren, depth };
      }
    }

    return { allowed: true, currentChildren, allowedChildren, depth };
  }

  private safeAgentKey(value: string): string {
    const key = value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return key || 'custom';
  }

  private deriveTeamName(plans: DelegationAgentPlan[]): string {
    const archetypes = Array.from(new Set(plans.map(plan => plan.archetype)));
    const has = (archetype: SubAgentArchetype) => archetypes.includes(archetype);
    if (has('coder') && has('tester')) return 'ImplementationTeam';
    if (has('researcher') && has('critic') && has('summarizer')) return 'AnalysisTeam';
    if (has('researcher') && has('critic')) return 'ReviewTeam';
    if (has('researcher') && has('planner')) return 'DiscoveryTeam';
    const label = archetypes
      .map(archetype => archetype.charAt(0).toUpperCase() + archetype.slice(1))
      .join('');
    return `${label || 'Delegation'}Team`;
  }

  private inferAgentArchetype(agent: AgentInfo): SubAgentArchetype {
    const id = agent.identity.id.toLowerCase();
    const name = agent.identity.name.toLowerCase();
    const text = `${id} ${name}`;
    const archetypes: SubAgentArchetype[] = ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester'];
    return archetypes.find(archetype => text.includes(archetype)) ?? 'custom';
  }

  async enqueueMessage<TPayload>(message: EnqueueMessageInput<TPayload>): Promise<RuntimeMessage<TPayload>> {
    const ctx = this.getContext();
    const recipientProtocol = ctx.manager.getAgentById(message.to)?.getIdentity().communicationProtocol;
    return ctx.queue.enqueue({
      ...message,
      metadata: {
        ...message.metadata,
        communicationProtocol:
          message.metadata?.communicationProtocol ?? recipientProtocol ?? ctx.communication.getDefaultProtocolId(),
      },
    });
  }

  registerCommunicationProtocol(protocol: AgentCommunicationProtocol): void {
    this.requireCommunicationManager().registerProtocol(protocol);
    this.emit({
      type: 'communication.protocol.registered',
      agentId: 'runtime',
      data: { protocolId: protocol.id, version: protocol.version },
    });
  }

  setDefaultCommunicationProtocol(protocolId: string): void {
    const ctx = this.getContext();
    const previous = ctx.communication.getDefaultProtocolId();
    ctx.communication.setDefaultProtocol(protocolId);
    for (const info of ctx.manager.listAgentInfo()) {
      if (info.identity.communicationProtocol !== previous) continue;
      ctx.manager.getAgentById(info.identity.id)?.setCommunicationProtocol(protocolId);
    }
    this.emit({
      type: 'communication.protocol.default.changed',
      agentId: 'runtime',
      data: { protocolId },
    });
  }

  getCommunicationState(): CommunicationState {
    return this.requireCommunicationManager().getState();
  }

  getCommunicationTraces(options: {
    correlationId?: string;
    agentId?: string;
    limit?: number;
  } = {}): MultiPartyTrace[] {
    const ctx = this.getContext();
    return ctx.communication.traces.list({
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      actorId: options.agentId,
      limit: options.limit,
    });
  }

  injectSystemTrace(agentId: string | 'broadcast', trace: MultiPartyTrace): void {
    const ctx = this.getContext();
    if (!ctx.communication.registry.get(trace.protocolId)) {
      throw new Error(`Communication protocol "${trace.protocolId}" is not registered`);
    }
    if (trace.sessionId !== ctx.sessionId) {
      throw new Error(`Trace session "${trace.sessionId}" does not match active session "${ctx.sessionId}"`);
    }
    ctx.communication.traces.append(trace);
    const targets = agentId === 'broadcast'
      ? ctx.manager.listAgentInfo().map(info => ctx.manager.getAgentById(info.identity.id)!).filter(Boolean)
      : [ctx.manager.getAgentById(agentId)].filter((agent): agent is BaseAgent => Boolean(agent));
    if (targets.length === 0) throw new Error(`Agent "${agentId}" not found`);
    for (const agent of targets) agent.receiveSystemTrace(trace);
    this.emit({
      type: 'communication.trace.injected',
      agentId: agentId === 'broadcast' ? 'runtime' : agentId,
      sessionId: trace.sessionId,
      correlationId: trace.correlationId,
      data: { traceId: trace.id, kind: trace.kind, protocolId: trace.protocolId },
    });
  }

  async getQueueState(limit = 20): Promise<QueueState> {
    const ctx = this.getContext();
    const [stats, recent] = await Promise.all([
      ctx.queue.getStats(),
      ctx.queue.listMessages({ limit }),
    ]);

    return { stats, recent };
  }

  async getMessages(filter: { correlationId?: string; limit?: number } = {}): Promise<RuntimeMessage[]> {
    const ctx = this.getContext();
    const messages = await ctx.queue.listMessages({ limit: filter.limit });
    return filter.correlationId
      ? messages.filter(message => message.correlationId === filter.correlationId)
      : messages;
  }

  async getMemoryState(): Promise<WorkspaceMemoryState> {
    const ctx = this.getContext();
    return ctx.memory.getState();
  }

  async loadRootMemoryContext(): Promise<RootMemoryContext> {
    const ctx = this.getContext();
    return ctx.memory.loadRootContext();
  }

  async listTraces(): Promise<Array<{ name: string; path: string; size: number; updatedAt: number }>> {
    const ctx = this.getContext();
    return ctx.memory.listTraces();
  }

  async readTrace(name = 'latest', limit = 50): Promise<RuntimeEvent[]> {
    const ctx = this.getContext();
    return ctx.memory.readTrace(name, limit);
  }

  async readPublicMemoryDoc(name: string): Promise<string> {
    const ctx = this.getContext();
    return ctx.memory.readPublicDoc(name);
  }

  async readAgentMemoryDoc(agentKey: string, doc = 'memory'): Promise<string> {
    const ctx = this.getContext();
    return ctx.memory.readAgentDoc(agentKey, doc);
  }

  async readTeamMemoryDoc(teamKey: string, doc = 'memory'): Promise<string> {
    return this.getContext().memory.readTeamDoc(teamKey, doc);
  }

  async getMemoryMode(): Promise<MemoryMode> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryMode();
  }

  async setMemoryMode(mode: MemoryMode): Promise<MemoryMode> {
    const ctx = this.getContext();
    const next = await ctx.memory.setMemoryMode(mode);
    this.emit({ type: 'memory.mode.changed', data: { mode: next } });
    return next;
  }

  async listMemoryProposals(): Promise<MemoryUpdateProposal[]> {
    const ctx = this.getContext();
    return ctx.memory.listMemoryProposals();
  }

  async getMemoryProposal(id: string): Promise<MemoryUpdateProposal | undefined> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryProposal(id);
  }

  async proposeMemoryUpdates(source = 'manual'): Promise<MemoryUpdateProposal[]> {
    const ctx = this.getContext();
    this.emit({ type: 'memory.update.propose.started', agentId: 'root', data: { source } });
    const signals = await ctx.memory.collectMemorySignals();
    this.emit({
      type: 'memory.signals.collected',
      agentId: 'root',
      data: {
        source,
        sessionId: signals.source.sessionId,
        agentResults: signals.counts.agentResults,
        rootFinalResponses: signals.counts.rootFinalResponses,
        toolCalls: signals.toolCalls.length,
        outputGrounded: signals.agents.filter(agent => agent.outputGrounded).length,
        candidateSignals: signals.candidateSignals,
      },
    });
    const proposals = await ctx.memory.proposeMemoryUpdates();
    for (const proposal of proposals) {
      this.emit({
        type: 'memory.proposal.created',
        agentId: proposal.target.type === 'agent' ? proposal.target.key : 'root',
        data: {
          source,
          proposalId: proposal.id,
          target: proposal.target.path,
          section: proposal.target.section,
          risk: proposal.risk,
          confidence: proposal.confidence,
        },
      });
    }
    if (proposals.length === 0) {
      this.emit({
        type: 'memory.update.skipped',
        agentId: 'root',
        data: {
          reason: signals.candidateSignals.length === 0 ? 'no_signals' : 'no_new_proposals',
          source,
          signalCounts: signals.counts,
          candidateSignals: signals.candidateSignals,
        },
      });
    }
    const records = await ctx.memory.listAllMemoryProposalRecords();
    const summary: MemoryProposalSummary = {
      createdThisRun: proposals.length,
      skippedDuplicates: proposals.length === 0 ? signals.candidateSignals.length : 0,
      updatedPendingProposals: 0,
      pendingProposals: records.filter(record => record.status === 'pending').length,
      alreadyCommitted: records.filter(record => record.status === 'committed').length,
    };
    await ctx.memory.recordAutoPropose(source, summary, proposals.length === 0 ? 'no_new_proposals' : undefined);
    this.emit({
      type: 'memory.update.propose.completed',
      agentId: 'root',
      data: {
        source,
        created: proposals.length,
        updated: summary.updatedPendingProposals,
        skippedDuplicates: summary.skippedDuplicates,
        pending: summary.pendingProposals,
        committed: summary.alreadyCommitted,
      },
    });
    return proposals;
  }

  async summarizeMemoryUpdates(source = 'manual'): Promise<MemoryProposalSummary> {
    const ctx = this.getContext();
    this.emit({ type: 'memory.update.propose.started', agentId: 'root', data: { source } });
    const summary = await ctx.memory.summarizeMemoryUpdates();
    await ctx.memory.recordAutoPropose(source, summary, summary.createdThisRun === 0 ? 'no_new_proposals' : undefined);
    this.emit({
      type: summary.createdThisRun > 0 ? 'memory.update.propose.completed' : 'memory.update.skipped',
      agentId: 'root',
      data: {
        createdThisRun: summary.createdThisRun,
        skippedDuplicates: summary.skippedDuplicates,
        updatedPendingProposals: summary.updatedPendingProposals,
        pendingProposals: summary.pendingProposals,
        alreadyCommitted: summary.alreadyCommitted,
        source,
        reason: summary.createdThisRun === 0 ? 'no_new_proposals' : undefined,
      },
    });
    return summary;
  }

  async collectMemorySignals(): Promise<MemorySignals> {
    const ctx = this.getContext();
    return ctx.memory.collectMemorySignals();
  }

  async getMemoryAutoState(): Promise<MemoryAutoState> {
    const ctx = this.getContext();
    return ctx.memory.getMemoryAutoState();
  }

  async getCachePatterns(kind: 'agents' | 'delegations' | 'teams'): Promise<Array<Record<string, unknown>>> {
    const ctx = this.getContext();
    return ctx.memory.getCachePatterns(kind);
  }

  async getExecutionKnowledge(task?: string, limit = 24): Promise<ExecutionKnowledgeCacheState> {
    return this.getContext().memory.readExecutionKnowledge(task, limit);
  }

  async getEvolutionHistory(limit = 50): Promise<Array<Record<string, unknown>>> {
    return this.getContext().memory.readEvolutionHistory(limit);
  }

  async getEvolutionPatterns(): Promise<EvolutionPattern[]> {
    return this.getContext().memory.getEvolutionPatterns();
  }

  getEvolutionRuns(limit = 20): EvolutionRunResult[] {
    return this.evolutionRuns.slice(-Math.max(1, limit)).map(run => structuredClone(run));
  }

  getEvolutionConfig(): WorkspaceRuntimeConfig['evolution'] {
    if (!this.workspaceRuntimeConfig) throw new Error('Runtime workspace config is not initialized');
    return structuredClone(this.workspaceRuntimeConfig.evolution);
  }

  async updateEvolutionConfig(
    patch: Partial<Omit<WorkspaceRuntimeConfig['evolution'], 'ablations'>> & {
      ablations?: Partial<WorkspaceRuntimeConfig['evolution']['ablations']>;
    }
  ): Promise<WorkspaceRuntimeConfig['evolution']> {
    this.validateEvolutionConfigPatch(patch);
    const next = await this.getContext().memory.updateEvolutionConfig(patch);
    if (!this.workspaceRuntimeConfig) throw new Error('Runtime workspace config is not initialized');
    this.workspaceRuntimeConfig.evolution = next;
    this.emit({ type: 'evo.config.updated', agentId: 'root', data: next as unknown as Record<string, unknown> });
    return structuredClone(next);
  }

  async runEvolution(input: RunEvolutionInput): Promise<EvolutionRunResult> {
    const ctx = this.getContext();
    if (!input.task?.trim()) throw new Error('Evolution task is required');
    if (this.workspaceRuntimeConfig?.evolution.enabled === false) throw new Error('Evolution is disabled by workspace policy');
    const parentId = input.parentId ?? 'root';
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) throw new Error(`Evolution parent agent "${parentId}" not found`);
    const correlationId = input.correlationId ?? this.createCorrelationId();
    const runId = `evo_run_${Date.now()}_${(++this.evolutionSequence).toString(36)}`;
    const options = this.resolveEvolutionRunOptions(input.profile, input.options);
    if (options.profile === 'solo' || options.ablations.withoutSubagents) {
      return this.runSoloEvolutionBaseline(runId, correlationId, input.task, options);
    }

    const startedAt = Date.now();
    if (options.ablations.withoutBudgetMarket) this.evolutionBudgetBypassCorrelations.add(correlationId);
    const initialBudgetMarket = this.getBudgetMarketState();
    const patterns = options.ablations.withoutPatternMemory
      ? []
      : await this.findRelevantEvolutionPatterns(input.task, options.patternSimilarityThreshold);
    for (const pattern of patterns) {
      this.emit({
        type: 'cache.hit',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { cacheType: 'evolution-pattern', patternId: pattern.id, similarityThreshold: options.patternSimilarityThreshold },
      });
    }
    const seedAgents = input.seedAgents?.length
      ? input.seedAgents.map(seed => this.normalizeEvolutionSeed(seed))
      : await this.createEvolutionSeeds(input.task, parentId, correlationId, options);
    const policy = this.getAgentPolicy(parentId);
    const availableAgentSlots = Math.max(0, Math.min(
      policy?.allowedChildren ?? 0,
      this.getRemainingTotalAgentsForTurn(parentId, correlationId)
    ));
    if (availableAgentSlots <= 0) throw new Error('Evolution cannot execute because no agent slots remain for the parent or turn');
    const proposalInput = {
      runId,
      task: input.task,
      parentId,
      agents: seedAgents,
      patterns,
      availableTokens: options.ablations.withoutBudgetMarket ? undefined : this.getBudgetState().remainingTokens,
      availableAgentSlots,
      options,
    };

    const instantiated = new Map<string, { kind: 'agent' | 'team'; actorId?: string; error?: string }>();
    const details = new Map<string, { agent?: RunAgentResult; team?: TeamRunResult }>();
    let evaluationUsage = this.sumUsage([]);
    const judge = options.useLlmJudge && ctx.llm
      ? this.createEvolutionJudge(parentId, correlationId, (usage: TokenUsage) => {
        evaluationUsage = this.sumUsage([evaluationUsage, usage]);
      })
      : undefined;
    const engine = new EvolutionLifecycleEngine(
      new TeamFirstGenomePlanner(),
      new CompositeEvolutionEvaluator(judge),
      new WeightedTopKSelectionPolicy(),
      defaultMutationOperators(),
      {
        onTransition: async (from, to, data) => {
          this.emit({ type: 'evo.fsm.transition', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, from, to, ...data } });
          await this.recordEvolutionLifecycleMessage(to, parentId, correlationId, { runId, from, to, ...data });
        },
        onCandidateRejected: async (candidate, reason) => {
          const patternIds = candidate.source === 'cache_hit' ? candidate.lineage.parentPatternIds : [];
          if (patternIds.length > 0) await ctx.memory.deprecateEvolutionPatterns(patternIds);
          this.emit({
            type: 'evo.candidate.rejected',
            agentId: parentId,
            sessionId: ctx.sessionId,
            correlationId,
            data: {
              runId,
              candidateId: candidate.id,
              genomeId: candidate.genome.id,
              source: candidate.source,
              reason,
              deprecatedPatternIds: patternIds,
            },
          });
        },
        instantiate: async candidate => {
          try {
            const actor = await this.instantiateEvolutionCandidate(candidate, parentId, correlationId, options);
            instantiated.set(candidate.id, actor);
            this.emit({
              type: 'evo.candidate.spawned', agentId: actor.actorId ?? parentId, sessionId: ctx.sessionId, correlationId,
              data: {
                runId, candidateId: candidate.id, genomeId: candidate.genome.id,
                actorKind: actor.kind, actorId: actor.actorId, memberCount: candidate.genome.members.length,
                degeneratedToAgent: candidate.genome.members.length === 1, source: candidate.source,
              },
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            instantiated.set(candidate.id, { kind: candidate.genome.members.length === 1 ? 'agent' : 'team', error: message });
            this.emit({ type: 'evo.candidate.instantiate.failed', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, candidateId: candidate.id, error: message } });
          }
        },
        execute: async candidate => {
          const actor = instantiated.get(candidate.id);
          if (!actor || actor.error || !actor.actorId) {
            return this.failedEvolutionArtifact(candidate, actor?.actorId ?? 'not-created', actor?.error ?? 'candidate_not_instantiated');
          }
          const artifact = await this.executeEvolutionCandidate(candidate, actor, correlationId, details);
          this.emit({
            type: artifact.success ? 'evo.candidate.executed' : 'evo.candidate.failed',
            agentId: artifact.actorId,
            sessionId: ctx.sessionId,
            correlationId,
            data: {
              runId, candidateId: candidate.id, actorKind: artifact.actorKind,
              totalTokens: artifact.usage.totalTokens, wallClockMs: artifact.wallClockMs,
              toolSuccessRate: artifact.toolCalls > 0 ? artifact.successfulToolCalls / artifact.toolCalls : null,
              failedActors: artifact.failedActors,
            },
          });
          await this.archiveEvolutionCandidateActors(artifact, runId, candidate.id, correlationId);
          return artifact;
        },
        integrate: async (selected, evaluation, execution) => {
          if (options.ablations.withoutPatternMemory) return undefined;
          return this.integrateEvolutionPattern(selected, evaluation, execution);
        },
      }
    );

    try {
      const lifecycle = await engine.run(proposalInput);
      for (const evaluation of lifecycle.evaluations) {
        this.emit({
          type: 'evo.candidate.evaluated', agentId: parentId, sessionId: ctx.sessionId, correlationId,
          data: { runId, candidateId: evaluation.candidateId, score: evaluation.score, dimensions: evaluation.dimensions, evaluator: evaluation.evaluator },
        });
      }
      if (lifecycle.selected) {
        this.emit({
          type: 'evo.candidate.selected', agentId: parentId, sessionId: ctx.sessionId, correlationId,
          data: {
            runId, candidateId: lifecycle.selected.id, genomeId: lifecycle.selected.genome.id,
            score: lifecycle.selectedEvaluation?.score, source: lifecycle.selected.source,
            lineage: lifecycle.selected.lineage,
          },
        });
      }
      const completedAt = Date.now();
      const executionsUsage = this.sumUsage(lifecycle.executions.map(execution => this.evolutionUsageToTokenUsage(execution.usage)));
      const totalUsage = this.sumUsage([executionsUsage, evaluationUsage]);
      const finalBudgetMarket = this.getBudgetMarketState();
      const runAllocations = finalBudgetMarket.allocations.filter(allocation => allocation.request.correlationId === correlationId);
      const metrics = this.buildEvolutionMetrics({
        lifecycle,
        usage: totalUsage,
        startedAt,
        completedAt,
        cacheHits: patterns.length,
        budgetRequested: runAllocations.reduce((sum, allocation) => sum + allocation.request.requestedTokens, 0),
        budgetAllocated: runAllocations.reduce((sum, allocation) => sum + allocation.allocatedTokens, 0),
      });
      const run: EvolutionRunResult = {
        id: runId,
        correlationId,
        task: input.task,
        profile: options.profile,
        state: lifecycle.state,
        candidates: lifecycle.candidates,
        executions: lifecycle.executions,
        evaluations: lifecycle.evaluations,
        selected: lifecycle.selected,
        selectedExecution: lifecycle.selectedExecution,
        selectedEvaluation: lifecycle.selectedEvaluation,
        integratedPatternId: lifecycle.integratedPatternId,
        metrics,
        ablations: options.ablations,
        startedAt,
        completedAt,
      };
      this.evolutionRuns.push(run);
      if (this.evolutionRuns.length > 100) this.evolutionRuns.splice(0, this.evolutionRuns.length - 100);
      await ctx.memory.recordEvolutionRun({ ...run, initialBudgetMarket, finalBudgetMarket });
      this.emit({
        type: 'evo.run.completed', agentId: parentId, sessionId: ctx.sessionId, correlationId,
        data: { runId, profile: options.profile, selected: run.selected?.id, score: run.selectedEvaluation?.score, metrics },
      });
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
      return structuredClone(run);
    } catch (error) {
      const completedAt = Date.now();
      const failed: EvolutionRunResult = {
        id: runId, correlationId, task: input.task, profile: options.profile, state: 'S_evo_failed',
        candidates: [], executions: [], evaluations: [], selected: undefined,
        selectedExecution: undefined, selectedEvaluation: undefined,
        metrics: {
          taskSuccess: false, answerQuality: 0, toolSuccessRate: 0, agentsSpawned: 0, teamsSpawned: 0,
          totalTokens: evaluationUsage.totalTokens, thinkingTokens: evaluationUsage.thinkingTokens,
          wallClockMs: completedAt - startedAt, budgetRequested: 0, budgetAllocated: 0,
          failureRecoveryCount: 0, candidateCount: 0, executedCandidateCount: 0,
          cacheHits: patterns.length, mutationsApplied: 0,
        },
        ablations: options.ablations, startedAt, completedAt,
        error: error instanceof Error ? error.message : String(error),
      };
      this.evolutionRuns.push(failed);
      await ctx.memory.recordEvolutionRun(failed as unknown as Record<string, unknown>);
      this.emit({ type: 'evo.run.failed', agentId: parentId, sessionId: ctx.sessionId, correlationId, data: { runId, error: failed.error } });
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
      throw error;
    }
  }

  async runEvolutionBenchmark(
    task: string,
    profiles: EvolutionProfile[] = ['solo', 'fixed_subagents', 'tom_subteam', 'budget_market', 'evo_team']
  ): Promise<EvolutionBenchmarkResult> {
    const runs: EvolutionRunResult[] = [];
    for (const profile of profiles) {
      runs.push(await this.runEvolution({ task, profile }));
    }
    return {
      task,
      profiles,
      runs,
      comparison: runs.map(run => ({
        profile: run.profile,
        success: run.metrics.taskSuccess,
        score: run.selectedEvaluation?.score ?? run.metrics.answerQuality,
        totalTokens: run.metrics.totalTokens,
        thinkingTokens: run.metrics.thinkingTokens,
        wallClockMs: run.metrics.wallClockMs,
        agentsSpawned: run.metrics.agentsSpawned,
        teamsSpawned: run.metrics.teamsSpawned,
      })),
    };
  }

  async acceptMemoryProposal(id: string): Promise<MemoryUpdateRecord | undefined> {
    const ctx = this.getContext();
    const record = await ctx.memory.acceptMemoryProposal(id);
    this.emit({
      type: record ? 'memory.update.committed' : 'memory.update.skipped',
      agentId: 'root',
      data: { proposalId: id, target: record?.targetPath },
    });
    return record;
  }

  async rejectMemoryProposal(id: string): Promise<boolean> {
    const ctx = this.getContext();
    const rejected = await ctx.memory.rejectMemoryProposal(id);
    this.emit({
      type: rejected ? 'memory.update.rejected' : 'memory.update.skipped',
      agentId: 'root',
      data: { proposalId: id },
    });
    return rejected;
  }

  async listMemoryUpdates(): Promise<MemoryUpdateRecord[]> {
    const ctx = this.getContext();
    return ctx.memory.listMemoryUpdates();
  }

  async recordConversation(entry: Omit<ConversationEntry, 'id' | 'timestamp' | 'sessionId'> & { sessionId?: string }): Promise<ConversationEntry> {
    const ctx = this.getContext();
    return ctx.memory.appendConversation({
      ...entry,
      sessionId: entry.sessionId ?? ctx.sessionId,
    });
  }

  async getConversation(sessionId?: string, limit = 50): Promise<ConversationEntry[]> {
    const ctx = this.getContext();
    return ctx.memory.readConversation(sessionId ?? ctx.sessionId, limit);
  }

  async listConversationSessions(): Promise<ConversationSessionState[]> {
    const ctx = this.getContext();
    return ctx.memory.listConversationSessions();
  }

  async importConversation(filePath: string, sessionId?: string): Promise<{ imported: number; path: string }> {
    const ctx = this.getContext();
    const result = await ctx.memory.importConversation(filePath, sessionId ?? ctx.sessionId);
    this.emit({
      type: 'conversation.imported',
      data: {
        imported: result.imported,
        path: result.path,
        sessionId: sessionId ?? ctx.sessionId,
      },
    });
    return result;
  }

  async handleUserTurn(userInput: string): Promise<RootTurnResult> {
    const correlationId = this.createCorrelationId();
    try {
      return await this.executeUserTurn(userInput, correlationId);
    } catch (error) {
      await this.recoverFailedRootTurn(correlationId, error);
      throw error;
    } finally {
      this.failedPathObservations.delete(correlationId);
      this.changedPathsByCorrelation.delete(correlationId);
      this.resumedExecutionByCorrelation.delete(correlationId);
    }
  }

  /**
   * Run a non-interactive turn through transient provider failures without
   * discarding the persisted workspace, execution cache, tree, or trace.
   *
   * Stream-level retries handle brief transport faults first. This boundary is
   * deliberately separate from handleUserTurn so interactive callers keep
   * explicit failure semantics, while batch runners can resume the same task
   * from Roy's persisted execution ledger after a fully exhausted stream.
   */
  async handleUserTurnWithRecovery(
    userInput: string,
    options: RootTurnRecoveryOptions = {}
  ): Promise<RootTurnRecoveryResult> {
    const retryConfig = this.workspaceRuntimeConfig?.llm;
    const maxAttempts = Math.max(
      1,
      Math.floor(options.maxAttempts ?? retryConfig?.turnMaxAttempts ?? 3)
    );
    const initialDelayMs = Math.max(
      0,
      Math.floor(options.retryInitialDelayMs ?? retryConfig?.retryInitialDelayMs ?? 250)
    );
    const maxDelayMs = Math.max(
      initialDelayMs,
      Math.floor(options.retryMaxDelayMs ?? retryConfig?.retryMaxDelayMs ?? 2_000)
    );
    const correlationIds: string[] = [];

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const eventStart = this.events.length;
      try {
        const result = await this.handleUserTurn(userInput);
        correlationIds.push(result.correlationId);
        if (attempt > 1) {
          this.emit({
            type: 'runtime.transient_turn.recovered',
            agentId: 'root',
            correlationId: result.correlationId,
            data: {
              attempt,
              maxAttempts,
              failedCorrelationIds: correlationIds.slice(0, -1),
            },
          });
        }
        return {
          result,
          attempts: attempt,
          recovered: attempt > 1,
          correlationIds,
        };
      } catch (error) {
        const failedCorrelationId = this.events
          .slice(eventStart)
          .reverse()
          .find(event => event.type === 'root.turn.failed')
          ?.correlationId;
        if (failedCorrelationId && !correlationIds.includes(failedCorrelationId)) {
          correlationIds.push(failedCorrelationId);
        }
        const retryable = this.isRetryableLLMStreamError(error);
        const willRetry = retryable && attempt < maxAttempts;
        this.emit({
          type: willRetry
            ? 'runtime.transient_turn.retrying'
            : 'runtime.transient_turn.failed',
          agentId: 'root',
          correlationId: failedCorrelationId,
          data: {
            attempt,
            maxAttempts,
            retryable,
            error: error instanceof Error ? error.message : String(error),
            persistedState: true,
          },
        });
        if (!willRetry) throw error;
        const delayMs = Math.min(maxDelayMs, initialDelayMs * (2 ** (attempt - 1)));
        if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    throw new Error('Transient turn recovery exhausted without a result');
  }

  private async executeUserTurn(userInput: string, correlationId: string): Promise<RootTurnResult> {
    const ctx = this.getContext();
    const rootUsageBefore = ctx.agent.getUsage();
    const perTurnUsageStartIndex = this.perTurnUsage.length;
    const rootStepConfig = this.workspaceRuntimeConfig?.delegation.rootSteps;
    const externalExecutionFeedback = this.containsExternalExecutionFeedback(userInput);
    const requiresWorkspaceMutation = this.taskRequiresRootWorkspaceMutation(userInput);
    const reservedFinalSteps = requiresWorkspaceMutation ? 2 : 1;
    const maxSteps = Math.max(
      reservedFinalSteps + 1,
      rootStepConfig?.maxStepsPerTurn ?? 12
    );
    const loopController = new RootTaskLoopController({
      maxIterations: maxSteps,
      maxWallClockMs: Math.max(1, rootStepConfig?.maxWallClockMs ?? 15 * 60_000),
      maxStalledIterations: Math.max(1, rootStepConfig?.maxStalledIterations ?? 2),
      reserveFinalSteps: reservedFinalSteps,
    });
    this.executionTrees.begin({
      correlationId,
      sessionId: ctx.sessionId,
      task: userInput,
      rootAgentId: 'root',
      rootAgentName: ctx.agent.name,
      maxSteps,
      maxWallClockMs: rootStepConfig?.maxWallClockMs,
      maxStalledIterations: rootStepConfig?.maxStalledIterations,
    });
    await this.persistRootExecutionTree(correlationId);

    const inputMessage = await this.enqueueMessage({
      kind: 'user.input',
      sessionId: ctx.sessionId,
      from: 'cli',
      to: 'root',
      correlationId,
      payload: { input: userInput },
      metadata: { agentId: 'root' },
    });
    await this.processQueuedMessage(inputMessage.id);
    await ctx.queue.ack(inputMessage.id);

    await this.recordConversation({
      role: 'user',
      speaker: 'user',
      content: userInput,
      correlationId,
      metadata: { kind: 'user.input' },
    });

    await this.transitionRootTurnState('S_input_received', { correlationId });
    await this.transitionRootTurnState('S_assess_task', { correlationId });
    const requiresLongHorizon = rootStepConfig?.enabled !== false && this.requiresLongHorizonLoop(userInput);
    this.emit({
      type: 'root.task.execution.classified',
      agentId: 'root',
      correlationId,
      data: {
        requiresWorkspaceMutation,
        requiresLongHorizon,
        externalExecutionFeedback,
      },
    });
    const cachedExecution = requiresLongHorizon && requiresWorkspaceMutation
      ? await ctx.memory.readExecutionKnowledge(
        userInput,
        rootStepConfig?.maxFeedbackItemsInPrompt ?? 24
      )
      : undefined;
    const resumeState = cachedExecution
      ? this.findExecutionResumeState(userInput, cachedExecution)
      : undefined;
    if (resumeState) {
      this.resumedExecutionByCorrelation.set(correlationId, resumeState);
      this.emit({
        type: 'root.task_loop.resumed',
        agentId: 'root',
        correlationId,
        data: {
          sourceCorrelationId: resumeState.sourceCorrelationId,
          anchorPathId: resumeState.anchorPathId,
          steps: resumeState.knowledge.steps.length,
          paths: resumeState.knowledge.paths.length,
          openPaths: resumeState.openPaths,
          actionableFeedback: resumeState.actionableFeedback,
        },
      });
    }
    const resumeRecoveryDecision = resumeState
      ? this.buildVerifierGuidedResumeRecoveryDecision(userInput, resumeState)
      : undefined;
    let decision: DelegationDecision = resumeState
      ? resumeRecoveryDecision ?? {
        action: 'solve_directly',
        reason: 'Resume the persisted execution ledger and close its unresolved workspace paths without rebuilding the initial team.',
      }
      : await this.decideDelegation(userInput, correlationId);
    if (resumeRecoveryDecision) {
      this.emit({
        type: 'root.task_loop.resume_strategy_changed',
        agentId: 'root',
        correlationId,
        data: {
          sourceCorrelationId: resumeState!.sourceCorrelationId,
          from: 'root_direct_repair',
          to: 'verifier_guided_recovery_team',
          agents: resumeRecoveryDecision.agents.map(agent => agent.name),
          reason: resumeRecoveryDecision.reason,
        },
      });
    }
    let requiredLongHorizonDecision: DelegationDecision | undefined;
    if (decision.action === 'solve_directly'
      && requiresLongHorizon
      && !resumeState) {
      decision = this.buildLongHorizonTeamDecision(userInput, requiresWorkspaceMutation);
      requiredLongHorizonDecision = decision;
      this.emit({
        type: 'root.task_loop.promoted',
        agentId: 'root',
        correlationId,
        data: { reason: 'long_horizon_task_detected' },
      });
    } else if (decision.action === 'spawn_subagents' && requiresLongHorizon) {
      decision = this.ensureLongHorizonTeamDecision(decision, userInput, requiresWorkspaceMutation, correlationId);
      requiredLongHorizonDecision = decision;
    }
    const executableDecision = this.overrideExecutableTaskClarification(
      decision,
      userInput,
      requiresLongHorizon,
      requiresWorkspaceMutation,
      correlationId
    );
    if (executableDecision !== decision) {
      decision = executableDecision;
      requiredLongHorizonDecision = decision;
    }
    if (!resumeState) {
      decision = await this.selectDelegationCandidate(
        'root',
        userInput,
        decision,
        correlationId,
        'root',
        requiresLongHorizon || (decision.action === 'spawn_subagents'
          && decision.coordination === 'team'
          && Boolean(decision.team))
      );
    }
    if (requiresLongHorizon && decision.action === 'spawn_subagents') {
      decision = this.ensureLongHorizonTeamDecision(decision, userInput, requiresWorkspaceMutation, correlationId);
    }
    if (requiredLongHorizonDecision?.action === 'spawn_subagents' && decision.action !== 'spawn_subagents') {
      const rootPolicy = this.getAgentPolicy('root');
      const hasCapacity = Boolean(rootPolicy && rootPolicy.allowedChildren > rootPolicy.currentChildren)
        && this.getRemainingTotalAgentsForTurn('root', correlationId) > 0;
      if (hasCapacity) {
        this.emit({
          type: 'delegation.candidate.overridden',
          agentId: 'root',
          correlationId,
          data: {
            reason: 'explicit_long_horizon_loop_requires_initial_checkpoint',
            rejectedAction: decision.action,
          },
        });
        decision = requiredLongHorizonDecision;
      }
    }
    const decisionMetadata = await this.buildDelegationDecisionMetadata(decision);
    this.emit({
      type: 'delegation.decision',
      agentId: 'root',
      data: {
        correlationId,
        action: decision.action,
        reason: decision.reason,
        agents: decision.action === 'spawn_subagents' ? decision.agents : [],
        coordination: decision.action === 'spawn_subagents' ? decision.coordination : undefined,
        team: decision.action === 'spawn_subagents' ? decision.team : undefined,
        continuationPolicy: decision.action === 'spawn_subagents' ? decision.continuationPolicy : undefined,
        ...decisionMetadata,
      },
    });

    let finalResponse: string;
    const subagents: RootMediatedSpawnResult[] = [];
    const teamResults: TeamRunResult[] = [];
    const evolutions: EvolutionRunResult[] = [];
    let evolution: EvolutionRunResult | undefined;
    let loopStopReason: RootExecutionTreeState['loop']['stopReason'] = 'completed';
    let requiredExecutionClosure: WorkspaceExecutionClosureStatus | undefined;

    if (decision.action === 'ask_clarification') {
      const step = await this.startRootExecutionStep(correlationId, {
        action: 'ask_clarification',
        reason: decision.reason,
        agentCount: 0,
      });
      await this.transitionRootTurnState('S_solo_reasoning', { correlationId, reason: decision.reason });
      this.emit({
        type: 'delegation.skipped',
        agentId: 'root',
        data: {
          correlationId,
          action: decision.action,
          reason: decision.reason,
        },
      });
      finalResponse = decision.question;
      await this.completeRootExecutionStep(correlationId, step, { resultSummary: finalResponse });
    } else if (decision.action === 'solve_directly') {
      const step = await this.startRootExecutionStep(correlationId, {
        action: 'solve_directly',
        reason: decision.reason,
        agentCount: 0,
      });
      await this.transitionRootTurnState('S_solo_reasoning', { correlationId, reason: decision.reason });
      this.emit({
        type: 'delegation.skipped',
        agentId: 'root',
        data: {
          correlationId,
          action: decision.action,
          reason: decision.reason,
        },
      });
      if (requiresWorkspaceMutation) {
        this.emit({
          type: 'root.execution.required.started',
          agentId: 'root',
          correlationId,
          data: { stepId: step.id, source: 'solve_directly' },
        });
        const rootExecution = await this.runRequiredRootExecution(
          userInput,
          [],
          [],
          correlationId
        );
        requiredExecutionClosure = this.analyzeWorkspaceExecutionClosure(
          rootExecution.toolCalls,
          rootExecution.acceptanceAudit,
          this.taskRequiresAcceptanceAudit(userInput)
        );
        this.emit({
          type: requiredExecutionClosure.closed
            ? 'root.execution.required.completed'
            : 'root.execution.required.unmet',
          agentId: 'root',
          correlationId,
          data: {
            stepId: step.id,
            source: 'solve_directly',
            ...requiredExecutionClosure,
            verificationRan: requiredExecutionClosure.verificationPassed,
            toolCalls: rootExecution.toolCalls.map(call => ({
              toolName: call.toolName,
              success: call.success,
            })),
          },
        });
        finalResponse = await this.synthesizeDelegatedResults(
          userInput,
          [],
          correlationId,
          [],
          rootExecution
        );
      } else {
        finalResponse = await this.runRootSoloReasoning(userInput, correlationId);
      }
      await this.completeRootExecutionStep(correlationId, step, { resultSummary: finalResponse });
    } else {
      let roundDecision = decision;
      let delegationRounds = 0;
      let previousStepId: string | undefined;
      let clarification: string | undefined;
      let rootExecution: GroundingRunResult | undefined;
      let diagnosticPrerequisiteFailed = false;

      while (roundDecision.action === 'spawn_subagents') {
        delegationRounds += 1;
        const plans = roundDecision.agents.slice(0, 3);
        const step = await this.startRootExecutionStep(correlationId, {
          action: 'delegate',
          reason: roundDecision.reason,
          agentCount: plans.length,
        }, previousStepId ? [previousStepId] : []);

        await this.transitionRootTurnState('S_delegate_planning', {
          correlationId,
          stepId: step.id,
          count: plans.length,
        });
        this.emitDelegationPlan(correlationId, plans, roundDecision.reason, decisionMetadata, step.id);
        if (roundDecision.team && plans.length > 1) {
          this.emit({
            type: 'delegation.team.designed',
            agentId: 'root',
            correlationId,
            data: {
              stepId: step.id,
              coordination: roundDecision.coordination,
              team: roundDecision.team,
              members: plans.map(plan => ({
                archetype: plan.archetype,
                name: plan.name,
                role: plan.role,
                task: plan.task,
                tools: plan.tools,
                skills: plan.skills,
              })),
            },
          });
        }
        await this.transitionRootTurnState('S_spawn_subagents', {
          correlationId,
          stepId: step.id,
          count: plans.length,
        });

        let round: RootDelegationRoundResult;
        try {
          round = await this.executeRootDelegationRound(userInput, roundDecision, correlationId);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          const hasCompletedPriorWork = subagents.length > 0 || teamResults.length > 0;
          const failedFocusedDiagnostic = this.getEvents().some(event =>
            event.correlationId === correlationId
            && event.type === 'agent.verifier_diagnostic.missing'
          );
          if (failedFocusedDiagnostic) diagnosticPrerequisiteFailed = true;
          const canRecoverWithRootExecution = requiresWorkspaceMutation
            && !failedFocusedDiagnostic;
          const canRecoverWithRootReasoning = !requiresWorkspaceMutation;
          const canRecover = failedFocusedDiagnostic
            || hasCompletedPriorWork
            || canRecoverWithRootExecution
            || canRecoverWithRootReasoning;
          await this.failRootExecutionStep(correlationId, step, message, !canRecover);
          await this.persistRootExecutionTree(correlationId);
          this.emit({
            type: 'root.step.failed',
            agentId: 'root',
            correlationId,
            data: { stepId: step.id, error: message },
          });
          if (canRecover) {
            this.emit({
              type: 'root.step.recovered',
              agentId: 'root',
              correlationId,
              data: {
                stepId: step.id,
                error: message,
                recovery: failedFocusedDiagnostic
                  ? 'checkpoint_failed_diagnostic_without_mutation'
                  : hasCompletedPriorWork
                  ? 'synthesize_completed_prior_steps'
                  : canRecoverWithRootExecution
                    ? 'root_execution_after_failed_delegation'
                    : 'root_reasoning_after_failed_delegation',
                completedSubagents: subagents.length,
                completedTeams: teamResults.length,
              },
            });
            await this.transitionRootTurnState('S_wait_subagents', {
              correlationId,
              stepId: step.id,
              recovered: true,
            });
            break;
          }
          throw error;
        }
        subagents.push(...round.subagents);
        teamResults.push(...round.teams);
        if (round.evolution) {
          evolution = round.evolution;
          evolutions.push(round.evolution);
        }

        await this.transitionRootTurnState('S_wait_subagents', {
          correlationId,
          stepId: step.id,
          completed: round.evolution?.metrics.agentsSpawned ?? round.subagents.length,
        });
        const completedStep = await this.completeRootExecutionStep(correlationId, step, {
          actorIds: this.collectDelegationRoundActorIds(round, step.startedAt),
          teamIds: this.collectDelegationRoundTeamIds(round, step.startedAt),
          nodes: this.buildRootExecutionNodes(correlationId, step.index),
          resultSummary: this.summarizeDelegationRound(round),
        });
        previousStepId = completedStep.id;

        const tree = this.executionTrees.get(correlationId)!;
        const maxRounds = Math.max(1, rootStepConfig?.maxDelegationRounds ?? 8);
        const loopGuard = loopController.evaluate(tree);
        const roundMutationApplied = this.delegationRoundHasWorkspaceMutation(round);
        const exploratoryDelegationLimit = Math.max(
          1,
          rootStepConfig?.maxExploratoryDelegationRounds ?? 2
        );
        const configuredExecutionReserveMs = Math.max(
          0,
          rootStepConfig?.executionReserveMs ?? 2 * 60_000
        );
        const executionReserveMs = Math.min(
          configuredExecutionReserveMs,
          Math.floor(tree.loop.maxWallClockMs * 0.4)
        );
        const timeReserveHandoffRequired = requiresWorkspaceMutation
          && loopGuard.elapsedMs >= Math.max(
            0,
            tree.loop.maxWallClockMs - executionReserveMs
          );
        const verifierGuidedRecoveryRound = round.teams.some(result =>
          result.team.identity.name.startsWith('VerifierGuidedRecoveryTeam')
        );
        const executionHandoffRequired = timeReserveHandoffRequired
          || verifierGuidedRecoveryRound
          || this.shouldHandoffToRootExecution({
          requiresWorkspaceMutation,
          requiresLongHorizon,
          roundMutationApplied,
          delegationRounds,
          maxRounds,
          exploratoryDelegationLimit,
          });
        const canReassess = rootStepConfig?.enabled !== false
          && rootStepConfig?.reassessAfterDelegation !== false
          && roundDecision.continuationPolicy !== 'finalize_after_round'
          && !round.evolution
          && delegationRounds < maxRounds
          && loopGuard.continue
          && !executionHandoffRequired;

        await this.transitionRootTurnState('S_assess_task', {
          correlationId,
          stepId: step.id,
          delegationRounds,
        });
        if (!canReassess) {
          if (executionHandoffRequired) {
            this.emit({
              type: 'root.execution.handoff.required',
              agentId: 'root',
              correlationId,
              data: {
                stepId: step.id,
                delegationRounds,
                reason: timeReserveHandoffRequired
                  ? 'execution_time_reserve_reached'
                  : verifierGuidedRecoveryRound
                    ? 'verifier_guided_recovery_round_completed'
                  : roundMutationApplied
                    ? 'delegated_workspace_mutation_observed'
                    : 'delegation_round_cap_without_mutation',
                elapsedMs: loopGuard.elapsedMs,
                executionReserveMs,
                rootWallClockMs: tree.loop.maxWallClockMs,
              },
            });
          } else {
            if (!loopGuard.continue && loopGuard.reason !== 'continue') loopStopReason = loopGuard.reason;
            else if (delegationRounds >= maxRounds) loopStopReason = 'max_iterations';
            this.emit({
              type: 'root.step.limit_reached',
              agentId: 'root',
              correlationId,
              data: {
                stepId: step.id,
                delegationRounds,
                maxRounds,
                maxSteps: tree.maxSteps,
                reason: loopGuard.continue ? 'max_delegation_rounds' : loopGuard.reason,
                remainingSteps: loopGuard.remainingSteps,
                elapsedMs: loopGuard.elapsedMs,
                stalledIterations: tree.loop.stalledIterations,
              },
            });
          }
          break;
        }

        let continuation = await this.decideRootContinuation(
          userInput,
          correlationId,
          tree.steps,
          subagents,
          teamResults
        );
        continuation = this.ensureLongHorizonRecoveryContinuation(
          continuation,
          userInput,
          completedStep,
          delegationRounds,
          maxRounds,
          requiresLongHorizon,
          requiresWorkspaceMutation,
          correlationId
        );
        this.emit({
          type: 'root.step.decision',
          agentId: 'root',
          correlationId,
          data: { stepId: step.id, nextAction: continuation.action, reason: continuation.reason },
        });
        if (continuation.action === 'delegate_more') {
          let next: DelegationDecision = {
            action: 'spawn_subagents',
            reason: continuation.reason,
            agents: continuation.agents,
            coordination: continuation.coordination,
            team: continuation.team,
            continuationPolicy: continuation.continuationPolicy,
          };
          next = await this.selectDelegationCandidate('root', userInput, this.applyBudgetConstraints(next), correlationId, 'root', true);
          if (next.action === 'spawn_subagents') {
            if (requiresLongHorizon
              && (requiresWorkspaceMutation || next.coordination === 'team' || Boolean(next.team))) {
              next = this.ensureLongHorizonTeamDecision(
                next,
                userInput,
                requiresWorkspaceMutation,
                correlationId
              );
            }
            roundDecision = next;
            continue;
          }
        } else if (continuation.action === 'ask_clarification') {
          clarification = continuation.question;
        }
        break;
      }

      if (!clarification && requiresWorkspaceMutation && !diagnosticPrerequisiteFailed) {
        const executionStep = await this.startRootExecutionStep(correlationId, {
          action: 'solve_directly',
          reason: 'Delegated analysis is complete; the root must apply and verify the requested workspace changes.',
          agentCount: 0,
        }, previousStepId ? [previousStepId] : []);
        this.emit({
          type: 'root.execution.required.started',
          agentId: 'root',
          correlationId,
          data: { stepId: executionStep.id },
        });
        rootExecution = await this.runRequiredRootExecution(
          userInput,
          subagents,
          teamResults,
          correlationId
        );
        requiredExecutionClosure = this.analyzeWorkspaceExecutionClosure(
          rootExecution.toolCalls,
          rootExecution.acceptanceAudit,
          this.taskRequiresAcceptanceAudit(userInput)
        );
        const completedExecutionStep = await this.completeRootExecutionStep(correlationId, executionStep, {
          resultSummary: this.summarizeRootExecutionClosure(rootExecution),
        });
        previousStepId = completedExecutionStep.id;
        this.emit({
          type: requiredExecutionClosure.closed
            ? 'root.execution.required.completed'
            : 'root.execution.required.unmet',
          agentId: 'root',
          correlationId,
          data: {
            stepId: executionStep.id,
            ...requiredExecutionClosure,
            verificationRan: requiredExecutionClosure.verificationPassed,
            toolCalls: rootExecution.toolCalls.map(call => ({
              toolName: call.toolName,
              success: call.success,
            })),
          },
        });
      } else if (!clarification && requiresWorkspaceMutation) {
        rootExecution = this.collectResumedToolGrounding(correlationId);
        requiredExecutionClosure = this.analyzeWorkspaceExecutionClosure(
          rootExecution?.toolCalls ?? []
        );
        this.emit({
          type: 'root.execution.required.blocked',
          agentId: 'root',
          correlationId,
          data: {
            reason: 'focused_verifier_diagnostic_missing',
            nextAction: 'Retry the focused diagnostic checkpoint before any further workspace mutation.',
            ...requiredExecutionClosure,
          },
        });
      }

      const finalStep = await this.startRootExecutionStep(correlationId, {
        action: clarification ? 'ask_clarification' : 'finalize',
        reason: clarification
          ? 'More user input is required after delegated inspection.'
          : requiredExecutionClosure && !requiredExecutionClosure.closed
            ? 'Execution closure remains unmet; report the concrete mutation and verification state without claiming completion.'
            : 'Roy has sufficient accumulated state to produce the final result.',
        agentCount: 0,
      }, previousStepId ? [previousStepId] : []);
      if (clarification) {
        loopStopReason = 'clarification';
        await this.transitionRootTurnState('S_solo_reasoning', { correlationId, stepId: finalStep.id });
        finalResponse = clarification;
      } else {
        await this.transitionRootTurnState('S_synthesize', {
          correlationId,
          stepId: finalStep.id,
          completed: evolution?.metrics.agentsSpawned ?? subagents.length,
        });
        finalResponse = evolution && evolutions.length === 1 && subagents.length === 0
          ? await this.synthesizeEvolutionResult(userInput, evolution, correlationId, rootExecution)
          : await this.synthesizeDelegatedResults(
            userInput,
            subagents,
            correlationId,
            teamResults,
            rootExecution
          );
      }
      await this.completeRootExecutionStep(correlationId, finalStep, {
        nodes: this.buildRootExecutionNodes(correlationId, finalStep.index),
        resultSummary: finalResponse,
      });
      this.emit({
        type: 'delegation.completed',
        agentId: 'root',
        data: {
          correlationId,
          subagentIds: subagents.map(result => result.agent.identity.id),
          totalSubagents: evolution?.metrics.agentsSpawned ?? subagents.length,
          evolutionRunId: evolution?.id,
          selectedGenomeId: evolution?.selected?.genome.id,
        },
      });
    }

    if (!requiresWorkspaceMutation && decision.action !== 'ask_clarification') {
      finalResponse = await this.enforceRootResponseAcceptance(
        finalResponse,
        userInput,
        correlationId,
        [
          ...teamResults.map(result => result.result),
          ...subagents.map(result => result.subagentResult.result),
        ],
        [
          ...teamResults.flatMap(result =>
            result.members.map(member => member.evidence.toolResultSummary ?? '')
          ),
          ...subagents.map(result => result.subagentResult.evidence.toolResultSummary ?? ''),
        ]
      );
    }
    finalResponse = await this.enforceExplicitRootOutputContract(
      finalResponse,
      userInput,
      correlationId
    );
    if (requiredExecutionClosure && !requiredExecutionClosure.closed) {
      loopStopReason = 'closure_unmet';
      finalResponse = [
        '[runtime_execution_closure_unmet]',
        'The requested workspace task is not complete: Runtime could not establish a successful verification after the latest mutation.',
        `Closure evidence: ${JSON.stringify(requiredExecutionClosure)}`,
        finalResponse,
      ].join('\n\n');
      this.emit({
        type: 'root.execution.closure.unmet',
        agentId: 'root',
        correlationId,
        data: { ...requiredExecutionClosure },
      });
    }
    await this.transitionRootTurnState('S_respond', { correlationId });
    const finalMessage = await this.enqueueMessage({
      kind: 'root.final_response',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'cli',
      correlationId,
      payload: { content: finalResponse },
      metadata: { agentId: 'root' },
    });
    await this.recordConversation({
      role: 'assistant',
      speaker: 'Roy',
      content: finalResponse,
      correlationId,
      metadata: {
        kind: subagents.length > 0 || evolution ? 'root.delegated_final_response' : 'root.chat_response',
        decision: decision.action,
        subagentIds: subagents.map(result => result.agent.identity.id),
        grounded: evolution
          ? Boolean(evolution.selectedExecution && evolution.selectedExecution.groundedResults > 0)
          : subagents.length === 0 ? undefined : subagents.every(result => result.subagentResult.grounded),
        evolutionRunId: evolution?.id,
      },
    });
    await this.processQueuedMessage(finalMessage.id);
    await ctx.queue.ack(finalMessage.id);
    await this.transitionRootTurnState('S_turn_done', { correlationId });
    await this.transitionRootTurnState('S_solo', { correlationId });
    const executionTree = this.executionTrees.finish(correlationId, loopStopReason);
    await this.persistRootExecutionTree(correlationId);
    this.emit({
      type: 'root.execution_tree.completed',
      agentId: 'root',
      correlationId,
      data: { steps: executionTree.steps.length, nodes: executionTree.nodes.length },
    });
    await this.proposeMemoryUpdates('turn.completed');

    const rootUsageAfter = ctx.agent.getUsage();
    const rootUsage = this.usageDifference(rootUsageBefore, rootUsageAfter);
    const subagentUsage: Record<string, TokenUsage> = {};
    for (const item of subagents) {
      subagentUsage[item.agent.identity.id] = item.subagentResult.usage;
    }
    for (const teamResult of teamResults) {
      for (const [agentId, usage] of Object.entries(teamResult.team.memberUsage)) {
        subagentUsage[agentId] = { ...usage };
      }
    }
    const teamSynthesisUsage = Object.fromEntries(
      teamResults.map(result => [result.team.identity.id, { ...result.team.synthesisUsage }])
    );
    const reportedUsage = this.sumUsage([
      rootUsage,
      ...Object.values(subagentUsage),
      ...Object.values(teamSynthesisUsage),
      ...(evolution ? [this.sumUsage(evolution.executions.map(item =>
        this.evolutionUsageToTokenUsage(item.usage)
      ))] : []),
    ]);
    const recordedUsage = this.sumUsage(
      this.perTurnUsage.slice(perTurnUsageStartIndex)
    );

    return {
      correlationId,
      decision,
      finalResponse,
      subagents,
      teams: teamResults,
      evolution,
      evolutions,
      executionTree,
      messages: await this.getMessages({ correlationId }),
      usage: {
        root: rootUsage,
        subagents: subagentUsage,
        teamSynthesis: teamSynthesisUsage,
        total: recordedUsage.totalTokens > reportedUsage.totalTokens
          ? recordedUsage
          : reportedUsage,
      },
    };
  }

  async runMultiTurnExperiment(input: MultiTurnExperimentInput): Promise<MultiTurnExperimentResult> {
    const turns = input.turns.map(turn => turn.trim()).filter(Boolean);
    if (turns.length === 0) throw new Error('Multi-turn experiment requires at least one non-empty turn');
    const ctx = this.getContext();
    const startedAt = Date.now();
    const results: MultiTurnExperimentTurn[] = [];
    this.emit({
      type: 'experiment.multi_turn.started',
      agentId: 'root',
      sessionId: ctx.sessionId,
      data: { turnCount: turns.length },
    });

    for (const [index, turn] of turns.entries()) {
      const turnStartedAt = Date.now();
      this.emit({
        type: 'experiment.turn.started',
        agentId: 'root',
        sessionId: ctx.sessionId,
        data: { index: index + 1, input: turn },
      });
      try {
        const result = await this.handleUserTurn(turn);
        const turnEvents = this.getEvents().filter(event =>
          event.timestamp >= turnStartedAt
          && (event.correlationId === result.correlationId || event.data?.correlationId === result.correlationId)
        );
        const agentIds = [...new Set(turnEvents
          .filter(event => event.type === 'agent.spawned' && event.agentId)
          .map(event => event.agentId!))];
        const teamIds = [...new Set(turnEvents
          .filter(event => event.type === 'team.created' && event.agentId)
          .map(event => event.agentId!))];
        results.push({
          index: index + 1,
          input: turn,
          status: 'completed',
          result,
          eventTypes: turnEvents.map(event => event.type),
          agentIds,
          teamIds,
          budget: this.getBudgetState(),
        });
        this.emit({
          type: 'experiment.turn.completed',
          agentId: 'root',
          sessionId: ctx.sessionId,
          correlationId: result.correlationId,
          data: {
            index: index + 1,
            decision: result.decision.action,
            agentsCreated: agentIds.length,
            teamsCreated: teamIds.length,
            totalTokens: result.usage.total.totalTokens,
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        results.push({
          index: index + 1,
          input: turn,
          status: 'failed',
          error: message,
          eventTypes: this.getEvents().filter(event => event.timestamp >= turnStartedAt).map(event => event.type),
          agentIds: [],
          teamIds: [],
          budget: this.getBudgetState(),
        });
        this.emit({
          type: 'experiment.turn.failed',
          agentId: 'root',
          sessionId: ctx.sessionId,
          data: { index: index + 1, error: message },
        });
        if (input.stopOnError !== false) break;
      }
    }

    const completedAt = Date.now();
    const completedTurns = results.filter(turn => turn.status === 'completed').length;
    const failedTurns = results.length - completedTurns;
    const totalUsage = this.sumUsage(results.flatMap(turn => turn.result ? [turn.result.usage.total] : []));
    this.emit({
      type: 'experiment.multi_turn.completed',
      agentId: 'root',
      sessionId: ctx.sessionId,
      data: { completedTurns, failedTurns, totalTokens: totalUsage.totalTokens, wallClockMs: completedAt - startedAt },
    });
    return {
      sessionId: ctx.sessionId,
      startedAt,
      completedAt,
      turns: results,
      completedTurns,
      failedTurns,
      totalUsage,
    };
  }

  async handleSpawnCommand(payload: SpawnCommandPayload): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const execution = await this.createAgentComputeNode({
      parentId: payload.parentId,
      archetype: payload.archetype,
      task: payload.task,
      name: payload.name,
      role: payload.customRole,
      style: payload.customStyle,
      tools: payload.tools,
      skills: payload.skills,
      budgetTokens: payload.budgetTokens,
      memoryScope: payload.memoryScope,
      spawnPolicy: payload.spawnPolicy,
      tomProfile: payload.tomProfile ?? (payload.tomLevel === undefined
        ? undefined
        : { ...this.createSubagentToMProfile(payload.archetype, '', payload.task, payload.parentId ?? 'root'), level: payload.tomLevel as ToMProfile['level'] }),
      tomProfileMode: payload.source && !['cli', 'server'].includes(payload.source)
        ? 'runtime_assignment'
        : 'definition_override',
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
      communicationProtocol: payload.communicationProtocol,
      reuse: { mode: payload.reuseMode ?? 'prefer_cache' },
      outputContract: payload.outputContract,
      lifecycle: payload.lifecycle,
      lifecycleOrigin: payload.lifecycleOrigin,
      execution: {
        requireParentSynthesis: payload.requireRootSynthesis ?? true,
        showSubagentOutput: payload.showSubagentOutput ?? false,
        disableRecursiveDelegation: payload.disableRecursiveDelegation ?? false,
        teamId: payload.teamId,
      },
    }, {
      agentId: payload.parentId ?? 'root',
      sessionId: ctx.sessionId,
      source: payload.source ?? 'cli',
    }, payload.correlationId);
    return execution.delegation;
  }

  async createAgentComputeNode(
    request: AgentComputeNodeRequest,
    invocation: AgentCreationInvocation,
    requestedCorrelationId?: string
  ): Promise<AgentComputeNodeExecution> {
    const ctx = this.getContext();
    if (invocation.sessionId !== ctx.sessionId) {
      throw new Error(`Agent creation session mismatch: expected "${ctx.sessionId}", received "${invocation.sessionId}"`);
    }
    const parentId = request.parentId ?? invocation.agentId;
    if (parentId !== invocation.agentId) {
      throw new Error(`Agent creation parent mismatch: "${invocation.agentId}" cannot create a child for "${parentId}"`);
    }
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) throw new Error(`Parent agent "${parentId}" not found`);
    const parentBindings = this.agentBindings.get(parentId);
    const delegationBinding = parentBindings?.skills.find(binding => binding.name === 'delegate_to_subagent' && binding.enabled);
    if (!delegationBinding) {
      throw new Error(`Agent "${parentId}" is not authorized to use delegate_to_subagent`);
    }

    const correlationId = requestedCorrelationId ?? this.createCorrelationId();
    const eventStart = this.events.length;
    let node: AgentComputeNodeDefinition;
    try {
      node = await this.resolveAgentComputeNode(request, invocation, correlationId);
    } catch (error) {
      this.emit({
        type: 'agent.node.resolve.failed',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          sessionId: ctx.sessionId,
          correlationId,
          archetype: request.archetype,
          error: error instanceof Error ? error.message : String(error),
        },
      });
      throw error;
    }
    const rootUsageBefore = ctx.agent.getUsage();
    this.emitNodeEvent('agent.node.execution.started', node, parentId, {
      archetype: node.identity.archetype,
      creationMode: node.reuse.creationMode,
    });

    try {
      const delegation = await this.executeAgentComputeNode(node);
      const rootUsageAfter = ctx.agent.getUsage();
      const rootUsage = this.usageDifference(rootUsageBefore, rootUsageAfter);
      const subagentUsage = delegation.subagentResult.usage;
      const totalUsage = this.sumUsage([rootUsage, subagentUsage]);
      this.emitNodeEvent('agent.node.execution.completed', node, delegation.agent.identity.id, {
        totalTokens: totalUsage.totalTokens,
        grounded: delegation.subagentResult.grounded,
      });
      return {
        node,
        delegation,
        tokenUsage: { root: rootUsage, subagent: subagentUsage, total: totalUsage },
        events: this.getEvents().slice(eventStart).filter(event => this.eventCorrelationId(event) === correlationId),
      };
    } catch (error) {
      this.emitNodeEvent('agent.node.execution.failed', node, parentId, {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  private async resolveAgentComputeNode(
    request: AgentComputeNodeRequest,
    invocation: AgentCreationInvocation,
    correlationId: string
  ): Promise<AgentComputeNodeDefinition> {
    const ctx = this.getContext();
    if (!this.isValidArchetype(request.archetype)) {
      throw new Error(`Unsupported subagent archetype "${request.archetype}"`);
    }
    if (!request.task.trim()) throw new Error('Agent compute node task is required');
    this.validateAgentNodeRequest(request);

    const parentId = request.parentId ?? invocation.agentId;
    const reuseMode = request.reuse?.mode ?? 'prefer_cache';
    const intentTask = request.intentTask?.trim() || request.task;
    const patternKey = request.archetype === 'custom' && request.name
      ? `custom_${this.safeAgentKey(request.name)}`
      : request.archetype;
    this.emit({
      type: 'agent.node.resolve.started',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { archetype: request.archetype, reuseMode, source: invocation.source },
    });

    const canReadCache = reuseMode !== 'fresh';
    const canonicalAgentPattern = await ctx.memory.findAgentPattern(patternKey);
    const cachedAgentPattern = canReadCache
      ? request.reuse?.agentPatternId
        ? await ctx.memory.findAgentPatternById(request.reuse.agentPatternId)
        : canonicalAgentPattern
      : undefined;
    const cachedDelegationPattern = canReadCache
      ? await ctx.memory.findDelegationPattern(request.archetype, intentTask)
      : undefined;
    if (reuseMode === 'require_cache' && !cachedAgentPattern) {
      throw new Error(`Agent creation requires a cached pattern for archetype "${request.archetype}"`);
    }
    if (request.reuse?.agentPatternId && cachedAgentPattern?.id !== request.reuse.agentPatternId) {
      throw new Error(`Requested agent pattern "${request.reuse.agentPatternId}" was not found`);
    }
    if (cachedAgentPattern && cachedAgentPattern.archetype !== request.archetype) {
      throw new Error(`Agent pattern "${String(cachedAgentPattern.id)}" does not match archetype "${request.archetype}"`);
    }
    if (request.reuse?.delegationPatternId && cachedDelegationPattern?.id !== request.reuse.delegationPatternId) {
      throw new Error(`Requested delegation pattern "${request.reuse.delegationPatternId}" was not found`);
    }

    const cachedTools = this.stringArray(cachedAgentPattern?.tools);
    const cachedSkills = this.stringArray(cachedAgentPattern?.skills);
    const defaultToolBindings = this.getToolBindingsForTask(
      request.archetype,
      intentTask
    );
    const defaultTools = defaultToolBindings.map(item => item.name);
    const defaultReadOnlyTools = defaultToolBindings
      .filter(item => item.permission === 'read_only')
      .map(item => item.name);
    const baseTools = cachedTools.length > 0
      ? Array.from(new Set([...defaultReadOnlyTools, ...cachedTools]))
      : defaultTools;
    const baseSkills = cachedSkills.length > 0 ? cachedSkills : this.getDefaultSkillBindings(request.archetype).map(item => item.name);
    const tools = request.tools ?? baseTools;
    const skills = request.skills ?? baseSkills;
    const cacheCapabilityRepair = cachedTools.length > 0 && !this.sameStringSet(cachedTools, baseTools);
    const toolsOverrideDefinition = request.tools !== undefined && !this.sameStringSet(request.tools, baseTools);
    const skillsOverrideDefinition = request.skills !== undefined && !this.sameStringSet(request.skills, baseSkills);
    this.validateDelegatedCapabilities(parentId, tools, skills);

    const cachedMemoryScope = this.agentMemoryScope(cachedAgentPattern?.memoryScope);
    const cachedSpawnPolicy = this.partialSpawnPolicy(cachedAgentPattern?.spawnPolicy);
    const memoryScope = this.constrainMemoryScope(
      request.memoryScope ?? cachedMemoryScope ?? this.getDefaultMemoryScope('subagent')
    );
    const cachedCommunicationProtocol = typeof cachedAgentPattern?.communicationProtocol === 'string'
      ? cachedAgentPattern.communicationProtocol
      : undefined;
    const communicationProtocol = request.communicationProtocol
      ?? cachedCommunicationProtocol
      ?? ctx.communication.getDefaultProtocolId();
    if (!ctx.communication.registry.get(communicationProtocol)) {
      throw new Error(`Communication protocol "${communicationProtocol}" is not registered`);
    }
    const requestedSpawnPolicy = this.mergeSpawnPolicy(
      this.getDefaultSpawnPolicy('subagent', request.archetype),
      { ...cachedSpawnPolicy, ...request.spawnPolicy }
    );
    if (request.spawnPolicy?.canSpawn === undefined
      && skills.includes('delegate_to_subagent')) {
      requestedSpawnPolicy.canSpawn = true;
    }
    const spawnPolicy = this.constrainChildSpawnPolicy(parentId, requestedSpawnPolicy, skills);
    const agentPatternId = typeof cachedAgentPattern?.id === 'string' ? cachedAgentPattern.id : undefined;
    const delegationPatternId = typeof cachedDelegationPattern?.id === 'string' ? cachedDelegationPattern.id : undefined;
    const cacheHits = [agentPatternId, delegationPatternId].filter((item): item is string => Boolean(item));
    const hasDefinitionOverrides = (request.archetype === 'custom' && request.name !== undefined)
      || request.role !== undefined
      || request.style !== undefined
      || request.description !== undefined
      || request.systemPrompt !== undefined
      || cacheCapabilityRepair
      || toolsOverrideDefinition
      || skillsOverrideDefinition
      || request.memoryScope !== undefined
      || request.spawnPolicy !== undefined
      || (request.tomProfile !== undefined && request.tomProfileMode !== 'runtime_assignment')
      || (request.communicationProtocol !== undefined && request.communicationProtocol !== cachedCommunicationProtocol)
      || request.outputContract !== undefined;
    const creationMode: AgentNodeCreationMode = cachedAgentPattern && (reuseMode === 'mutate_cache' || hasDefinitionOverrides)
      ? 'mutated_from_cache'
      : cachedAgentPattern
        ? 'cache_hit'
        : request.archetype === 'custom'
          ? 'custom'
          : 'generated';
    const outputContract = request.outputContract ?? {
      format: 'markdown',
      groundingRequired: this.taskRequiresGrounding(request.archetype, request.task),
    };
    const description = request.description ?? `Reusable ${request.archetype} agent compute node.`;
    const definitionSeed = {
      archetype: request.archetype,
      name: request.archetype === 'custom' ? request.name : undefined,
      role: request.role ?? request.archetype,
      style: request.style,
      description,
      systemPrompt: request.systemPrompt,
      tools,
      skills,
      memoryScope,
      spawnPolicy,
      tomProfile: request.tomProfileMode === 'definition_override' ? request.tomProfile : undefined,
      communicationProtocol,
      outputContract,
    };
    const definitionFingerprint = this.fingerprint(definitionSeed);
    const invocationFingerprint = this.fingerprint({
      definitionFingerprint,
      parentId,
      task: request.task,
      sessionId: ctx.sessionId,
      correlationId,
    });
    const canonicalPatternId = `agent_pattern_${this.safeAgentKey(patternKey)}_v1`;
    const targetPatternId = ((hasDefinitionOverrides && canonicalAgentPattern)
      || creationMode === 'mutated_from_cache')
      ? `agent_pattern_${this.safeAgentKey(patternKey)}_${definitionFingerprint.slice(0, 12)}_v1`
      : canonicalPatternId;
    const nodeId = `node_${correlationId}_${definitionFingerprint.slice(0, 10)}`;
    const node: AgentComputeNodeDefinition = {
      nodeId,
      sessionId: ctx.sessionId,
      correlationId,
      parentId,
      depth: this.getAgentDepth(parentId) + 1,
      definitionFingerprint,
      invocationFingerprint,
      identity: {
        archetype: request.archetype,
        name: request.name,
        role: request.role ?? request.archetype,
        style: request.style,
        description,
        systemPrompt: request.systemPrompt,
        tomProfile: request.tomProfile,
        tomProfileMode: request.tomProfileMode ?? 'definition_override',
        cognitiveGapIds: [...(request.cognitiveGapIds ?? [])],
        existenceReason: request.existenceReason,
      },
      assignment: {
        task: request.task,
        ...(request.intentTask ? { intentTask: request.intentTask } : {}),
        outputContract,
      },
      capabilities: { tools: [...tools], skills: [...skills] },
      context: { memoryScope, communicationProtocol },
      resources: { budgetTokens: request.budgetTokens },
      governance: {
        spawnPolicy,
        lifecycle: request.lifecycle,
        lifecycleOrigin: request.lifecycleOrigin,
      },
      execution: {
        requireParentSynthesis: request.execution?.requireParentSynthesis ?? true,
        showSubagentOutput: request.execution?.showSubagentOutput ?? false,
        disableRecursiveDelegation: request.execution?.disableRecursiveDelegation ?? false,
        teamId: request.execution?.teamId,
      },
      reuse: {
        mode: reuseMode,
        creationMode,
        definitionOverrides: hasDefinitionOverrides,
        cacheHits,
        agentPatternId,
        basePatternId: targetPatternId === canonicalPatternId
          ? undefined
          : typeof canonicalAgentPattern?.id === 'string' ? canonicalAgentPattern.id : undefined,
        delegationPatternId,
        targetPatternId,
      },
      source: invocation.source,
    };

    this.emitNodeEvent('agent.node.cache.evaluated', node, parentId, {
      reuseMode,
      cacheHits,
      creationMode,
      hasDefinitionOverrides,
    });
    for (const patternId of cacheHits) {
      this.emitNodeEvent('cache.hit', node, parentId, {
        cacheType: patternId.startsWith('agent_pattern_') ? 'agent-pattern' : 'delegation-pattern',
        patternId,
        archetype: request.archetype,
      });
    }
    this.emitNodeEvent('agent.node.resolved', node, parentId, {
      definitionFingerprint,
      invocationFingerprint,
      tools,
      skills,
      depth: node.depth,
    });
    return node;
  }

  private async executeAgentComputeNode(node: AgentComputeNodeDefinition): Promise<RootMediatedSpawnResult> {
    const ctx = this.getContext();
    const payload: SpawnCommandPayload = {
      parentId: node.parentId,
      archetype: node.identity.archetype,
      task: node.assignment.task,
      name: node.identity.name,
      customRole: node.identity.role,
      customStyle: node.identity.style,
      tools: node.capabilities.tools,
      skills: node.capabilities.skills,
      budgetTokens: node.resources.budgetTokens,
      memoryScope: node.context.memoryScope,
      spawnPolicy: node.governance.spawnPolicy,
      tomProfile: node.identity.tomProfile,
      cognitiveGapIds: node.identity.cognitiveGapIds,
      existenceReason: node.identity.existenceReason,
      systemPrompt: node.identity.systemPrompt,
      communicationProtocol: node.context.communicationProtocol,
      outputContract: node.assignment.outputContract,
      correlationId: node.correlationId,
      source: node.source,
      requireRootSynthesis: node.execution.requireParentSynthesis,
      showSubagentOutput: node.execution.showSubagentOutput,
      disableRecursiveDelegation: node.execution.disableRecursiveDelegation,
      teamId: node.execution.teamId,
      lifecycle: node.governance.lifecycle,
      lifecycleOrigin: node.governance.lifecycleOrigin,
    };
    const correlationId = node.correlationId;
    const parentId = node.parentId;
    const requireRootSynthesis = node.execution.requireParentSynthesis;
    const cacheHits = node.reuse.cacheHits;

    const externalCommand = payload.source === 'cli' || payload.source === 'server';
    const command = await this.enqueueMessage({
      kind: externalCommand ? 'user.command.spawn' : 'agent.control',
      sessionId: ctx.sessionId,
      from: externalCommand ? payload.source ?? 'cli' : parentId,
      to: externalCommand ? parentId : 'runtime',
      correlationId,
      payload: externalCommand ? payload : { action: 'delegate_to_subagent', node },
      metadata: { agentId: parentId, nodeId: node.nodeId },
    });
    if (externalCommand) {
      await this.recordConversation({
        role: 'user',
        speaker: payload.source ?? 'cli',
        content: `/spawn ${payload.archetype} "${payload.task}"`,
        correlationId,
        metadata: { command: 'spawn', archetype: payload.archetype },
      });
    }
    await this.processQueuedMessage(command.id);
    await ctx.queue.ack(command.id);

    if (parentId !== 'root') {
      await this.prepareParentForDelegation(parentId, correlationId, payload.task);
    }

    const tomProfile = payload.tomProfile
      ? normalizeToMProfile(payload.tomProfile, payload.tomProfile)
      : this.createSubagentToMProfile(payload.archetype, '', payload.task, parentId);
    if (payload.tomLevel !== undefined && [0, 1, 2, 3].includes(payload.tomLevel)) {
      tomProfile.level = payload.tomLevel as ToMProfile['level'];
    }
    const lifecycleOrigin = payload.lifecycleOrigin ?? this.lifecycleOriginForSource(payload.source, payload.teamId);
    const lifecyclePolicy = payload.lifecycle
      ?? this.inheritParentLifecyclePolicy(parentId, lifecycleOrigin);
    const agent = await this.spawnAgent({
      parentId,
      name: payload.name,
      customRole: payload.customRole,
      customStyle: payload.customStyle,
      archetype: payload.archetype,
      tomLevel: tomProfile.level,
      description: payload.task,
      task: payload.task,
      tools: payload.tools,
      skills: payload.skills,
      memoryScope: node.context.memoryScope,
      spawnPolicy: node.governance.spawnPolicy,
      budgetTokens: payload.budgetTokens,
      systemPrompt: payload.systemPrompt,
      outputContract: node.assignment.outputContract,
      correlationId,
      tomProfile,
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
      cacheHits,
      nodeDefinition: node,
      teamId: payload.teamId,
      lifecycle: lifecyclePolicy,
      lifecycleOrigin,
    });
    if (payload.teamId) {
      const team = this.teams.addMember(payload.teamId, agent.identity.id);
      try {
        await this.persistTeamTopology(team);
      } catch (error) {
        this.emit({
          type: 'team.persistence.failed',
          agentId: team.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: team.identity.id,
            parentAgentId: team.identity.parentAgentId,
            operation: 'persist_member_addition',
            error: error instanceof Error ? error.message : String(error),
          },
        });
      }
      this.emit({
        type: 'team.member.added',
        agentId: agent.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: payload.teamId, parentId, correlationId },
      });
    }
    const delegationPattern = await ctx.memory.upsertDelegationPattern({
      archetype: payload.archetype,
      task: node.assignment.intentTask ?? payload.task,
      parentId,
      agentPatternId: node.reuse.targetPatternId,
      tomProfile: agent.identity.tomProfile,
      cognitiveGapIds: payload.cognitiveGapIds,
      existenceReason: payload.existenceReason,
    });
    this.emit({
      type: 'memory.pattern.updated',
      agentId: parentId,
      data: {
        cacheType: 'delegation-pattern',
        patternId: delegationPattern.id,
        path: '.roy/cache/delegation-patterns.json',
      },
    });

    const taskSender = payload.teamId ?? parentId;
    const resultRecipient = payload.teamId ?? parentId;
    const taskMessage = await this.enqueueMessage({
      kind: 'agent.task',
      sessionId: ctx.sessionId,
      from: taskSender,
      to: agent.identity.id,
      correlationId,
      parentMessageId: command.id,
      payload: {
        task: payload.task,
        archetype: payload.archetype,
      },
      metadata: {
        agentId: agent.identity.id,
        nodeId: node.nodeId,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(taskMessage.id);

    let subagentResult: RunAgentResult;
    try {
      let sharedTeamToolCalls = payload.teamId
        ? this.teamToolEvidenceCache.get(payload.teamId) ?? []
        : [];
      if (payload.teamId && sharedTeamToolCalls.length === 0) {
        const sourceTeamIds = [...this.teamToolEvidenceCache.keys()].filter(teamId =>
          teamId !== payload.teamId
          && this.teamToolEvidenceCorrelations.get(teamId) === correlationId
          && (this.teamToolEvidenceCache.get(teamId)?.length ?? 0) > 0
        );
        const seenCalls = new Set<ToolCallRecord>();
        sharedTeamToolCalls = sourceTeamIds
          .flatMap(teamId => this.teamToolEvidenceCache.get(teamId) ?? [])
          .filter(call => {
            if (seenCalls.has(call)) return false;
            seenCalls.add(call);
            return true;
          })
          .sort((left, right) =>
            (left.completedAt ?? left.startedAt ?? 0)
            - (right.completedAt ?? right.startedAt ?? 0)
          );
        sharedTeamToolCalls = this.boundExecutionToolFrontier(
          sharedTeamToolCalls,
          80,
          720_000
        );
        if (sharedTeamToolCalls.length > 0) {
          this.teamToolEvidenceCache.set(payload.teamId, sharedTeamToolCalls);
          this.teamToolEvidenceCorrelations.set(payload.teamId, correlationId);
          this.emit({
            type: 'team.tool_evidence.seeded_from_correlation',
            agentId: agent.identity.id,
            sessionId: ctx.sessionId,
            correlationId,
            data: {
              teamId: payload.teamId,
              sourceTeamIds,
              cachedCalls: sharedTeamToolCalls.length,
              successfulCalls: sharedTeamToolCalls.filter(call => call.success).length,
            },
          });
        }
      }
      if (payload.teamId && sharedTeamToolCalls.length === 0) {
        sharedTeamToolCalls = this.restoredToolCallsFromResume(correlationId, 48);
        if (sharedTeamToolCalls.length > 0) {
          this.teamToolEvidenceCache.set(payload.teamId, sharedTeamToolCalls);
          this.teamToolEvidenceCorrelations.set(payload.teamId, correlationId);
          this.emit({
            type: 'team.tool_evidence.seeded_from_resume',
            agentId: agent.identity.id,
            sessionId: ctx.sessionId,
            correlationId,
            data: {
              teamId: payload.teamId,
              cachedCalls: sharedTeamToolCalls.length,
              successfulCalls: sharedTeamToolCalls.filter(call => call.success).length,
            },
          });
        }
      }
      if (payload.teamId && sharedTeamToolCalls.length > 0) {
        this.emit({
          type: 'team.tool_evidence.reused',
          agentId: agent.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: payload.teamId,
            cachedCalls: sharedTeamToolCalls.length,
            successfulCalls: sharedTeamToolCalls.filter(call => call.success).length,
          },
        });
      }
      subagentResult = await this.runAgent(agent.identity.id, payload.task, {
        correlationId,
        parentMessageId: taskMessage.id,
        archetype: payload.archetype,
        disableRecursiveDelegation: payload.disableRecursiveDelegation,
        nodeId: node.nodeId,
        patternId: node.reuse.targetPatternId,
        priorToolCalls: sharedTeamToolCalls,
        intentTask: node.assignment.intentTask,
      });
      if (payload.teamId && subagentResult.toolCalls.length > 0) {
        const cachedCalls = this.boundExecutionToolFrontier(
          [
            ...sharedTeamToolCalls,
            ...subagentResult.toolCalls,
          ],
          80,
          720_000
        );
        this.teamToolEvidenceCache.set(payload.teamId, cachedCalls);
        this.teamToolEvidenceCorrelations.set(payload.teamId, correlationId);
        this.emit({
          type: 'team.tool_evidence.cached',
          agentId: agent.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: payload.teamId,
            addedCalls: subagentResult.toolCalls.length,
            cachedCalls: cachedCalls.length,
          },
        });
      }
      await ctx.queue.ack(taskMessage.id);
    } catch (error) {
      const failedToolCalls = error instanceof Error
        && Array.isArray((error as Error & { runtimeToolCalls?: unknown }).runtimeToolCalls)
        ? (error as Error & { runtimeToolCalls: ToolCallRecord[] }).runtimeToolCalls
        : [];
      if (payload.teamId && failedToolCalls.length > 0) {
        const cachedCalls = this.boundExecutionToolFrontier(
          [
            ...(this.teamToolEvidenceCache.get(payload.teamId) ?? []),
            ...failedToolCalls,
          ],
          80,
          720_000
        );
        this.teamToolEvidenceCache.set(payload.teamId, cachedCalls);
        this.teamToolEvidenceCorrelations.set(payload.teamId, correlationId);
        this.emit({
          type: 'team.tool_evidence.cached_from_failure',
          agentId: agent.identity.id,
          sessionId: ctx.sessionId,
          correlationId,
          data: {
            teamId: payload.teamId,
            addedCalls: failedToolCalls.length,
            cachedCalls: cachedCalls.length,
          },
        });
      }
      const currentTask = await ctx.queue.getMessage(taskMessage.id);
      if (currentTask?.status === 'pending' || currentTask?.status === 'processing') {
        await ctx.queue.fail(taskMessage.id, error instanceof Error ? error : new Error(String(error)));
      }
      this.emitNodeEvent('agent.task.failed', node, agent.identity.id, {
        task: payload.task,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }

    const resultMessage = await this.enqueueMessage({
      kind: 'agent.result',
      sessionId: ctx.sessionId,
      from: agent.identity.id,
      to: resultRecipient,
      correlationId,
      parentMessageId: taskMessage.id,
      payload: subagentResult,
      metadata: {
        agentId: agent.identity.id,
        nodeId: node.nodeId,
        tomLevel: agent.identity.tomProfile.level,
      },
    });
    await this.recordConversation({
      role: 'agent',
      speaker: agent.identity.name,
      content: subagentResult.result,
      correlationId,
      metadata: {
        kind: 'agent.result',
        agentId: agent.identity.id,
        archetype: payload.archetype,
        parentId,
        teamId: payload.teamId,
        grounded: subagentResult.grounded,
        warnings: subagentResult.warnings,
        toolCalls: subagentResult.toolCalls.map(call => call.toolName),
        toolLoop: subagentResult.toolLoop,
        evidence: subagentResult.evidence,
        nodeId: node.nodeId,
        definitionFingerprint: node.definitionFingerprint,
        invocationFingerprint: node.invocationFingerprint,
        creationMode: node.reuse.creationMode,
      },
    });
    await this.processQueuedMessage(resultMessage.id);
    await ctx.queue.ack(resultMessage.id);
    this.emit({ type: 'agent.result.sent', agentId: agent.identity.id, data: { correlationId, to: resultRecipient, teamId: payload.teamId } });

    let finalResponse = '';
    if (requireRootSynthesis) {
      const parentSynthesis = await this.synthesizeChildResult(parentId, payload.task, agent, subagentResult, correlationId, resultMessage.id);
      if (parentId === 'root') {
        finalResponse = parentSynthesis;
      } else {
        const parentAgent = ctx.manager.getAgentById(parentId);
        if (!parentAgent) {
          throw new Error(`Parent agent "${parentId}" not found for root synthesis`);
        }
        const grandParentId = parentAgent.getIdentity().parentId ?? 'root';
        const parentResultMessage = await this.enqueueMessage({
          kind: 'agent.result',
          sessionId: ctx.sessionId,
          from: parentId,
          to: grandParentId,
          correlationId,
          parentMessageId: resultMessage.id,
          payload: {
            task: payload.task,
            result: parentSynthesis,
            childId: agent.identity.id,
          },
          metadata: {
            agentId: parentId,
            tomLevel: parentAgent.getIdentity().tomProfile.level,
          },
        });
        await this.recordConversation({
          role: 'agent',
          speaker: parentAgent.name,
          content: parentSynthesis,
          correlationId,
          metadata: {
            kind: 'agent.parent_synthesis',
            agentId: parentId,
            childId: agent.identity.id,
            parentId: grandParentId,
            grounded: subagentResult.grounded,
            evidence: subagentResult.evidence,
          },
        });
        await this.processQueuedMessage(parentResultMessage.id);
        await ctx.queue.ack(parentResultMessage.id);
        finalResponse = await this.synthesizeSubagentResult(
          payload.task,
          parentAgent.getInfo(),
          this.createSyntheticRunResult(parentAgent.getInfo(), parentSynthesis, subagentResult),
          correlationId,
          parentResultMessage.id
        );
      }
    }

    if (requireRootSynthesis) {
      const finalMessage = await this.enqueueMessage({
        kind: 'root.final_response',
        sessionId: ctx.sessionId,
        from: 'root',
        to: 'cli',
        correlationId,
        payload: { content: finalResponse },
        metadata: { agentId: 'root' },
      });
      await this.recordConversation({
        role: 'assistant',
        speaker: 'Roy',
        content: finalResponse,
        correlationId,
        metadata: {
          kind: 'root.final_response',
          subagentId: agent.identity.id,
          grounded: subagentResult.grounded,
          nodeId: node.nodeId,
          definitionFingerprint: node.definitionFingerprint,
        },
      });
      await this.processQueuedMessage(finalMessage.id);
      await ctx.queue.ack(finalMessage.id);
      await this.proposeMemoryUpdates('turn.completed');
    }

    return {
      correlationId,
      node,
      agent,
      subagentResult,
      finalResponse,
      messages: await this.getMessages({ correlationId }),
      creationUsage: this.measureAgentCreationUsage(agent.identity.id, node),
    };
  }

  private completeDelegatedTaskReference(
    task: string | undefined,
    parentId: string,
    correlationId: string
  ): {
    task: string;
    source: 'execution_tree' | 'parent_assignment';
    originalCharacters: number;
  } | undefined {
    if (!task?.trim() || !this.delegatedTaskHasUnresolvedReference(task)) {
      return undefined;
    }
    const treeTask = this.executionTrees.get(correlationId)?.task;
    const parentTask = this.agentRestoreSpecs.get(parentId)?.task;
    const source = treeTask?.trim()
      ? { task: treeTask, kind: 'execution_tree' as const }
      : parentTask?.trim()
        ? { task: parentTask, kind: 'parent_assignment' as const }
        : undefined;
    if (!source
      || source.task.trim() === task.trim()
      || source.task.length <= task.length
      || this.countIndependentTaskObligations(source.task) < 2) {
      return undefined;
    }
    const boundedSource = source.task.slice(0, 12_000);
    return {
      task: [
        task.trim(),
        '',
        '[runtime_referenced_parent_assignment]',
        'The delegated task referred to omitted objects. Use the following immutable parent assignment only to resolve those references:',
        boundedSource,
        '[/runtime_referenced_parent_assignment]',
      ].join('\n'),
      source: source.kind,
      originalCharacters: task.length,
    };
  }

  private delegatedTaskHasUnresolvedReference(task: string): boolean {
    const referencedObjects = task.match(
      /\b(?:these|those|above|following|all|each\s+of\s+the|the)\s+(?:(?:[a-z]+|\d+)[ -]?){0,4}(questions?|requirements?|criteria|items?|files?|tasks?|answers?)\b/i
    )?.[1]?.toLowerCase();
    if (!referencedObjects) return false;
    const numberedObjects = [...task.matchAll(/(?:^|\n)\s*\d{1,2}[.)]\s+\S/g)].length;
    if (numberedObjects >= 2) return false;
    if (referencedObjects.startsWith('question')
      || referencedObjects.startsWith('answer')) {
      return (task.match(/\?/g) ?? []).length < 2;
    }
    if (referencedObjects.startsWith('file')) {
      const paths = task.match(
        /(?:\.{0,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+\.[A-Za-z0-9]{1,8}/g
      ) ?? [];
      return paths.length < 2;
    }
    return true;
  }

  async spawnAgent(spec: SpawnAgentSpec & { tomProfile?: ToMProfile; cacheHits?: string[] }): Promise<AgentInfo> {
    const ctx = this.getContext();
    if (!this.isValidArchetype(spec.archetype)) {
      throw new Error(`Unsupported subagent archetype "${spec.archetype}"`);
    }
    if (!spec.description.trim()) {
      throw new Error('Subagent description is required');
    }

    const parent = ctx.manager.getAgentById(spec.parentId);
    if (!parent) {
      throw new Error(`Parent agent "${spec.parentId}" not found`);
    }

    const parentIdentity = parent.getIdentity();
    const creationCorrelationId = spec.correlationId ?? this.createCorrelationId();
    const delegatedTaskCompletion = this.completeDelegatedTaskReference(
      spec.task,
      spec.parentId,
      creationCorrelationId
    );
    if (delegatedTaskCompletion) {
      spec = {
        ...spec,
        task: delegatedTaskCompletion.task,
      };
      this.emit({
        type: 'agent.assignment.reference.completed',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          source: delegatedTaskCompletion.source,
          originalCharacters: delegatedTaskCompletion.originalCharacters,
          completedCharacters: delegatedTaskCompletion.task.length,
        },
      });
    }
    if (spec.parentId !== 'root') {
      await this.prepareParentForDelegation(spec.parentId, creationCorrelationId, spec.task ?? spec.description);
    }
    const toolBindings = this.normalizeToolBindings(
      spec.tools ?? this.getToolBindingsForTask(spec.archetype, spec.task ?? spec.description),
      spec.archetype
    )
      .filter(binding => binding.enabled);
    const skillBindings = this.normalizeSkillBindings(spec.skills, spec.archetype)
      .filter(binding => binding.enabled);
    const memoryScope = spec.memoryScope ?? this.getDefaultMemoryScope('subagent');
    const spawnPolicy = this.mergeSpawnPolicy(this.getDefaultSpawnPolicy('subagent', spec.archetype), spec.spawnPolicy);
    const createRequestMessage = await this.enqueueMessage({
      kind: 'agent.create.request',
      sessionId: ctx.sessionId,
      from: spec.parentId,
      to: 'runtime',
      correlationId: creationCorrelationId,
      payload: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name: spec.name,
        task: spec.task,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        invocationFingerprint: spec.nodeDefinition?.invocationFingerprint,
        creationMode: spec.nodeDefinition?.reuse.creationMode,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
      metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
    });
    await this.processQueuedMessage(createRequestMessage.id);

    this.emit({
      type: 'agent.create.requested',
      agentId: spec.parentId,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name: spec.name,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        creationMode: spec.nodeDefinition?.reuse.creationMode,
        tools: toolBindings.map(binding => binding.name),
        skills: skillBindings.map(binding => binding.name),
      },
    });
    const policyResult = this.validateSpawnPolicy({
      parentId: spec.parentId,
      archetype: spec.archetype,
      tools: toolBindings,
      skills: skillBindings,
      correlationId: creationCorrelationId,
      teamId: spec.teamId,
    });
    this.emit({
      type: 'spawn.policy.checked',
      agentId: spec.parentId,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        allowed: policyResult.allowed,
        reason: policyResult.reason,
        currentChildren: policyResult.currentChildren,
        allowedChildren: policyResult.allowedChildren,
        turnAgentsCreated: this.getTurnAgentCount(creationCorrelationId),
        remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(spec.parentId, creationCorrelationId),
        maxTotalAgentsPerTurn: this.getMaxTotalAgentsPerTurn(spec.parentId),
        depth: policyResult.depth,
      },
    });
    if (!policyResult.allowed) {
      this.emit({
        type: 'spawn.policy.rejected',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'agent.create.rejected',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      this.emit({
        type: 'delegation.rejected',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
      });
      const rejectedMessage = await this.enqueueMessage({
        kind: 'agent.create.rejected',
        sessionId: ctx.sessionId,
        from: 'runtime',
        to: spec.parentId,
        correlationId: creationCorrelationId,
        parentMessageId: createRequestMessage.id,
        payload: {
          parentId: spec.parentId,
          archetype: spec.archetype,
          reason: policyResult.reason,
        },
        metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
      });
      await this.processQueuedMessage(rejectedMessage.id);
      await ctx.queue.ack(rejectedMessage.id);
      await ctx.queue.fail(createRequestMessage.id, new Error(policyResult.reason ?? 'spawn_rejected'));
      throw new Error(`Spawn rejected: ${policyResult.reason}`);
    }

    const budgetAllocation = await this.requestAgentBudget({
      parentId: spec.parentId,
      archetype: spec.archetype,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      requestedTokens: spec.budgetTokens,
      purpose: spec.task ?? spec.description,
    });
    if (budgetAllocation?.status === 'denied') {
      await ctx.queue.fail(createRequestMessage.id, new Error('budget_request_denied'));
      this.emit({
        type: 'agent.create.rejected',
        agentId: spec.parentId,
        data: { parentId: spec.parentId, archetype: spec.archetype, reason: 'budget_request_denied' },
      });
      throw new Error('Spawn rejected: budget_request_denied');
    }

    const restoredSequence = Number(spec.instanceId?.match(/_(\d+)$/)?.[1] ?? 0);
    const sequence = spec.instanceId
      ? Math.max(restoredSequence, this.agentSequence + 1)
      : this.agentSequence + 1;
    this.agentSequence = Math.max(this.agentSequence, sequence);
    const id = spec.instanceId ?? this.createAgentId(spec.archetype, sequence);
    if (ctx.manager.getAgentById(id)) throw new Error(`Agent "${id}" already exists`);
    if (budgetAllocation?.status === 'granted') {
      this.budgetMarket?.assignRequester(budgetAllocation.id, id, 'agent');
    }
    const requestedName = spec.name ?? `${this.capitalize(spec.archetype)}-${sequence}`;
    const name = ctx.manager.getAgent(requestedName)
      ? this.createUniqueAgentName(spec.archetype, requestedName, sequence)
      : requestedName;
    if (ctx.manager.getAgent(name)) {
      if (budgetAllocation?.status === 'granted') {
        this.agentBudgetAllocations.set(id, budgetAllocation.id);
        this.releaseAgentBudget(id, 'agent_creation_failed');
      }
      throw new Error(`Agent name "${name}" already exists`);
    }
    if (budgetAllocation?.status === 'granted') {
      this.agentBudgetAllocations.set(id, budgetAllocation.id);
    }
    if (spec.budgetTokens !== undefined) {
      this.agentBudgetLimits.set(id, Math.max(0, Math.floor(spec.budgetTokens)));
    }

    try {
      const agentMemoryKey =
        spec.archetype === 'custom' && spec.name ? this.safeAgentKey(spec.name) : spec.archetype;
      await ctx.memory.ensureAgentMemory(agentMemoryKey, {
        name: spec.name ?? this.capitalize(spec.archetype),
        role: spec.customRole ?? spec.archetype,
        description: `Reusable ${spec.archetype} agent archetype memory.`,
      });
      const agentMemory = await ctx.memory.loadAgentMemory(agentMemoryKey);

      const generation = parentIdentity.generation + 1;

      const fsm = new FSM({
      initialState: 'S_created',
      strict: true,
      signalBus,
      onTransition: (from, to) => {
        logger.debug(`FSM transition for ${id}: ${from} -> ${to}`);
        this.emit({ type: 'fsm.transition', agentId: id, data: { from, to } });
      },
      onStateChange: (state) => {
        logger.debug(`FSM state for ${id}: ${state}`);
        this.emit({ type: 'fsm.state.changed', agentId: id, data: { state } });
      },
      onInvalidTransition: (from, to) => {
        this.emit({ type: 'fsm.invalid_transition', agentId: id, data: { from, to } });
      },
      });

      const cacheHits = spec.cacheHits ?? [];
      const creationMode =
        spec.nodeDefinition?.reuse.creationMode ??
        (cacheHits.length > 0 ? 'cache_hit' : spec.archetype === 'custom' ? 'custom' : 'generated');
      const resolvedTomProfile = spec.tomProfile
        ? normalizeToMProfile({ ...spec.tomProfile, subjectAgentId: id }, {
          level: spec.tomProfile.level,
          subjectAgentId: id,
          purpose: spec.tomProfile.purpose,
        })
        : this.createSubagentToMProfile(spec.archetype, id, spec.task ?? '', spec.parentId);
    const promptDescription = this.compactAgentPromptDescription(spec.description, spec.task);
    const contextWindow = await this.requireContextWindowManager().build({
      sessionId: ctx.sessionId,
      agentId: id,
      agentKey: agentMemoryKey,
      role: 'subagent',
      task: spec.task ?? '',
      parentContext: [
        `Parent agent ${parentIdentity.name} (${parentIdentity.id}) delegated the immutable task in the task slot.`,
        spec.existenceReason
          ? `Delegation purpose: ${spec.existenceReason.replace(/\s+/g, ' ').slice(0, 600)}`
          : '',
      ].filter(Boolean).join('\n'),
      memoryScope,
    });
    const boundedAgentMemory = {
      ...agentMemory,
      memory: contextWindow.privateMemory,
      context: '',
    };
    let goal = this.buildAgentPromptFromMemory({
      name,
      role: spec.customRole ?? spec.archetype,
      parentName: parentIdentity.name,
      task: contextWindow.task,
      description: [
          promptDescription,
          spec.customRole ? `Custom role: ${spec.customRole}` : undefined,
          spec.customStyle ? `Custom style: ${spec.customStyle}` : undefined,
          spec.outputContract
            ? `Output contract: ${JSON.stringify(spec.outputContract)}`
            : undefined,
        ]
          .filter(Boolean)
          .join('\n'),
        systemPrompt: spec.systemPrompt,
        bundle: boundedAgentMemory,
        publicContext: [
          contextWindow.publicContext,
          contextWindow.sessionContext,
          contextWindow.multiPartyTraceContext,
          cacheHits.length > 0 ? this.formatCachedPublicContext(cacheHits) : '',
        ]
          .filter(Boolean)
          .join('\n\n'),
        tomProfile: resolvedTomProfile,
        communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
        communicationContext: contextWindow.communicationContext,
        availableSkills: skillBindings.map((binding) => binding.name),
        availableTools: toolBindings.map((binding) => binding.name),
        parentContext: contextWindow.parentContext,
      });
      const originalRenderedPromptTokens = this.estimateTextTokens(goal);
      if (budgetAllocation?.status === 'granted'
        && !this.hasUnlimitedBudgetSupply()
        && this.budgetAccountingDimension() === 'total_tokens') {
        const outputReserve = Math.min(512, Math.max(128, Math.floor(budgetAllocation.allocatedTokens * 0.2)));
        const executionOverhead = Math.min(512, Math.max(128, Math.floor(budgetAllocation.allocatedTokens * 0.15)));
        const promptBudget = Math.max(64, budgetAllocation.allocatedTokens - outputReserve - executionOverhead);
        if (originalRenderedPromptTokens > promptBudget) {
          goal = this.truncateTextToTokenBudget(goal, promptBudget);
          this.emit({
            type: 'budget.context.truncated',
            agentId: id,
            sessionId: ctx.sessionId,
            correlationId: creationCorrelationId,
            nodeId: spec.nodeDefinition?.nodeId,
            data: {
              purpose: 'agent.system_prompt',
              contextType: 'rendered_agent_prompt',
              originalTokens: originalRenderedPromptTokens,
              allowedTokens: promptBudget,
              outputReserve,
              executionOverhead,
            },
          });
        }
      }
      const renderedPromptTokens = this.estimateTextTokens(goal);
    const definitionText = [
      name,
      spec.archetype,
      spec.customRole,
        spec.customStyle,
        spec.description,
        spec.tomProfile ? JSON.stringify(spec.tomProfile) : '',
        toolBindings.map((binding) => binding.name).join(','),
        skillBindings.map((binding) => binding.name).join(','),
      ]
        .filter(Boolean)
        .join('\n');
      const definitionTokens =
        creationMode === 'cache_hit' ? 0 : this.estimateTextTokens(definitionText);

      const agent = new UnifiedAgent({
        id,
      name,
      role: 'subagent',
      parentId: spec.parentId,
      teamId: spec.teamId,
      generation,
      tomLevel: resolvedTomProfile.level,
      tomProfile: resolvedTomProfile,
      communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
      description: spec.description,
      goal,
        llm: ctx.llm ?? undefined,
        fsm,
        mode: 'hybrid',
        allowedTools: toolBindings.map(binding => binding.name),
        allowedSkills: skillBindings.map(binding => binding.name),
      });

      this.registerCapabilities(
        agent,
        toolBindings.map((binding) => binding.name)
      );
      this.agentBindings.set(id, {
        tools: toolBindings,
        skills: skillBindings,
      memoryScope,
        spawnPolicy,
      });
      this.agentFsms.set(id, fsm);
      ctx.manager.addAgent(agent);
      await ctx.manager.attachAgentToSessions(agent);
      const lifecycleOrigin = spec.lifecycleOrigin ?? 'manual';
      const lifecyclePolicy = this.resolveLifecyclePolicy(lifecycleOrigin, spec.lifecycle);
      this.lifecycle.register({
        actorId: id,
        actorKind: 'agent',
        origin: lifecycleOrigin,
        parentId: spec.parentId,
        policy: lifecyclePolicy,
        createdAt: agent.getInfo().createdAt,
      });
      this.agentRestoreSpecs.set(id, {
        parentId: spec.parentId,
        name,
        customRole: spec.customRole,
        customStyle: spec.customStyle,
        archetype: spec.archetype,
        tomLevel: resolvedTomProfile.level,
        description: spec.description,
        task: spec.task,
        tools: toolBindings.map(binding => ({ ...binding })),
        skills: skillBindings.map(binding => ({ ...binding })),
        memoryScope: { ...memoryScope },
        spawnPolicy: { ...spawnPolicy },
        budgetTokens: spec.budgetTokens,
        systemPrompt: spec.systemPrompt,
        outputContract: spec.outputContract,
        teamId: spec.teamId,
        cognitiveGapIds: [...(spec.cognitiveGapIds ?? [])],
        existenceReason: spec.existenceReason,
        communicationProtocol: spec.communicationProtocol,
        tomProfile: resolvedTomProfile,
        lifecycle: { ...lifecyclePolicy },
        lifecycleOrigin,
        instanceId: id,
      });
      await ctx.memory.upsertAgentPattern({
        key: agentMemoryKey,
        patternId: spec.nodeDefinition?.reuse.targetPatternId
          ?? (spec.archetype === 'custom'
            ? `agent_pattern_custom_${this.safeAgentKey(agentMemoryKey)}_v1`
            : undefined),
        basePatternId: spec.nodeDefinition?.reuse.basePatternId,
        status: spec.nodeDefinition?.reuse.basePatternId ? 'candidate' : undefined,
        name: spec.name ?? this.capitalize(spec.archetype),
        archetype: spec.archetype,
        tomLevel: resolvedTomProfile.level,
        tomProfile: resolvedTomProfile,
        cognitiveGapIds: spec.cognitiveGapIds ?? resolvedTomProfile.cognitiveGaps,
        existenceReason: spec.existenceReason,
        description: spec.description,
        tools: toolBindings.map((binding) => binding.name),
        skills: skillBindings.map((binding) => binding.name),
        spawnPolicy,
        memoryScope,
        outputContract: spec.outputContract,
        communicationProtocol: spec.communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
      definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
      creationMode,
    });

    const info = agent.getInfo();
    if (spec.teamId) {
      const reservation = this.teamSpawnReservations.get(spec.teamId);
      if (reservation) {
        reservation.consumedMembers = Math.min(
          reservation.plannedMembers,
          reservation.consumedMembers + 1
        );
      }
    }
    this.recordTurnAgentCreated(creationCorrelationId);
    this.emit({
      type: 'tom.profile.assigned',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        level: resolvedTomProfile.level,
        perspective: resolvedTomProfile.perspective,
        beliefScope: resolvedTomProfile.beliefScope,
        goalModel: resolvedTomProfile.goalModel,
        uncertainty: resolvedTomProfile.uncertainty,
        observesAgents: resolvedTomProfile.observesAgents,
        modelsAgents: resolvedTomProfile.modelsAgents,
        cognitiveGapIds: spec.cognitiveGapIds ?? resolvedTomProfile.cognitiveGaps,
        existenceReason: spec.existenceReason,
      },
    });
    this.emit({
      type: 'context.loaded',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        sources: contextWindow.sources,
        tokenUsage: contextWindow.tokenUsage,
        sessionWindowTurns: memoryScope.sessionWindowTurns,
        },
      });
      this.emit({
        type:
          creationMode === 'cache_hit'
            ? 'agent.definition.loaded_from_cache'
            : 'agent.definition.generated',
        agentId: id,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        archetype: spec.archetype,
        cacheHits,
        creationMode,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
      },
    });
    this.emit({
      type: 'agent.create.approved',
      agentId: spec.parentId,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        childId: id,
        archetype: spec.archetype,
          creationMode,
          nodeId: spec.nodeDefinition?.nodeId,
          definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
          skills: skillBindings.map((binding) => binding.name),
          tools: toolBindings.map((binding) => binding.name),
          maxChildrenForParent: policyResult.allowedChildren,
        },
      });
    const approvedMessage = await this.enqueueMessage({
      kind: 'agent.create.approved',
      sessionId: ctx.sessionId,
      from: 'runtime',
      to: spec.parentId,
      correlationId: creationCorrelationId,
      parentMessageId: createRequestMessage.id,
      payload: {
        parentId: spec.parentId,
        childId: id,
        archetype: spec.archetype,
        name,
        nodeId: spec.nodeDefinition?.nodeId,
          definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
          invocationFingerprint: spec.nodeDefinition?.invocationFingerprint,
          creationMode,
          tools: toolBindings.map((binding) => binding.name),
          skills: skillBindings.map((binding) => binding.name),
        },
        metadata: { agentId: spec.parentId, nodeId: spec.nodeDefinition?.nodeId },
      });
      await this.processQueuedMessage(approvedMessage.id);
      await ctx.queue.ack(approvedMessage.id);
    await ctx.queue.ack(createRequestMessage.id);
    this.emit({
      type: 'agent.instance.created',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        archetype: spec.archetype,
        name,
        memoryKey: agentMemoryKey,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        },
      });
      for (const binding of toolBindings) {
        this.emit({
          type: 'agent.tool.bound',
          agentId: id,
          data: { tool: binding.name, permission: binding.permission },
        });
      }
      for (const binding of skillBindings) {
        this.emit({ type: 'agent.skill.bound', agentId: id, data: { skill: binding.name } });
    }
    this.emit({
      type: 'agent.spawned',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        parentId: spec.parentId,
        name,
        archetype: spec.archetype,
        tomLevel: resolvedTomProfile.level,
        description: spec.description,
        mode: creationMode,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
      },
    });
    this.emit({
      type: 'agent.creation.measured',
      agentId: id,
      sessionId: ctx.sessionId,
      correlationId: creationCorrelationId,
      nodeId: spec.nodeDefinition?.nodeId,
      data: {
        mode: creationMode,
        nodeId: spec.nodeDefinition?.nodeId,
        definitionFingerprint: spec.nodeDefinition?.definitionFingerprint,
        definitionTokens,
        renderedPromptTokens,
        renderedPromptChars: goal.length,
        cacheHits,
        },
      });
      await ctx.memory.updateCacheUsageMetrics(cacheHits, {
        definitionTokensSaved:
          creationMode === 'cache_hit' ? this.estimateTextTokens(definitionText) : 0,
        renderedPromptTokens,
      });
      this.emit({
      type: 'memory.pattern.updated',
      agentId: id,
      data: {
        cacheType: 'agent-pattern',
        archetype: spec.archetype,
          path: `.roy/cache/agent-patterns.json`,
        },
      });
      this.emit({
        type: 'agent.status.changed',
        agentId: id,
        data: { from: 'none', to: info.state },
      });
      await this.transitionAgentFsm(id, 'S_ready', { runtimeState: info.state });
      if (budgetAllocation?.status === 'granted') {
        this.emit({
        type: 'budget.allocated',
        agentId: id,
        data: { allocationId: budgetAllocation.id, budgetTokens: budgetAllocation.grantedTokens },
      });
      }

      return info;
    } catch (error) {
      ctx.manager.removeAgent(name);
      this.agentBindings.delete(id);
      this.agentFsms.delete(id);
      this.agentRestoreSpecs.delete(id);
      this.agentBudgetLimits.delete(id);
      this.releaseAgentBudget(id, 'agent_creation_failed');
      const currentRequest = await ctx.queue.getMessage(createRequestMessage.id);
      if (currentRequest?.status === 'pending' || currentRequest?.status === 'processing') {
        await ctx.queue.fail(
          createRequestMessage.id,
          error instanceof Error ? error : new Error(String(error))
        );
      }
      this.emit({
        type: 'agent.create.failed',
        agentId: spec.parentId,
        sessionId: ctx.sessionId,
        correlationId: creationCorrelationId,
        nodeId: spec.nodeDefinition?.nodeId,
        data: {
          parentId: spec.parentId,
          childId: id,
          archetype: spec.archetype,
          error: error instanceof Error ? error.message : String(error),
        },
      });
      throw error;
    }
  }

  async runAgent(
    agentId: string,
    task: string,
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      disableRecursiveDelegation?: boolean;
      nodeId?: string;
      patternId?: string;
      priorToolCalls?: ToolCallRecord[];
      intentTask?: string;
    } = {}
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent) {
      throw new Error(`Agent "${agentId}" not found`);
    }
    const actorArchetype = options.archetype ?? this.inferAgentArchetype(agent.getInfo());
    const assignedActorTask = this.agentRestoreSpecs.get(agentId)?.task?.trim();
    const immutableActorTask = options.intentTask?.trim()
      || (task.includes('<team_step_cache>')
        ? assignedActorTask || task.trim()
        : task.trim());

    this.activateActorLifecycle(agentId, options.correlationId);

    if (!this.agentBudgetAllocations.has(agentId)) {
      const allocation = await this.requestAgentBudget({
        parentId: agent.getIdentity().parentId ?? 'root',
        archetype: actorArchetype,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        requestedTokens: this.agentBudgetLimits.get(agentId),
        purpose: task,
      });
      if (allocation?.status === 'denied') {
        this.emit({
          type: 'agent.run.rejected',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: { task, reason: 'budget_request_denied' },
        });
        throw new Error('Agent run rejected: budget_request_denied');
      }
      if (allocation?.status === 'granted') {
        this.agentBudgetAllocations.set(agentId, allocation.id);
      }
    }

    const activeAllocationId = this.agentBudgetAllocations.get(agentId);
    const activeAllocation = activeAllocationId ? this.budgetMarket?.getAllocation(activeAllocationId) : undefined;
    agent.setCompletionTokenLimit(
      this.hasUnlimitedBudgetSupply()
        ? undefined
        : activeAllocation?.allocatedTokens,
      this.budgetAccountingDimension()
    );
    const actorWorkspaceExecutionRequired = this.taskRequiresWorkspaceMutation(
      immutableActorTask
    )
      && (this.agentBindings.get(agentId)?.tools ?? []).some(binding =>
        binding.enabled && (
          binding.name === 'fs.write'
          || binding.name === 'fs.replace'
          || binding.name === 'fs.synthesize'
        )
      );
    const actorVerificationEvidenceRequired = actorArchetype === 'tester'
      && (this.agentBindings.get(agentId)?.tools ?? []).some(binding =>
        binding.enabled && binding.name === 'shell.exec'
      );
    const actorDiagnosticProbeRequired =
      /\[runtime_verifier_diagnostic_probe\]/.test(immutableActorTask);
    const priorIndependentVerification = actorVerificationEvidenceRequired
      && !actorDiagnosticProbeRequired
      && !actorWorkspaceExecutionRequired
      ? this.selectAuthoritativePriorVerification(
        options.priorToolCalls ?? [],
        immutableActorTask
      )
      : undefined;

    const session = ctx.manager.getSession(ctx.sessionId);
    if (session) {
      session.messageQueue.clear('env');
    }

    const usageBefore = agent.getUsage();
    const from = agent.getState();
    const actorFsm = this.requireAgentFsm(agentId);
    if (actorFsm.getState() === 'S_done' || actorFsm.getState() === 'S_failed') {
      await this.transitionAgentFsm(agentId, 'S_ready', { reason: 'new_task' });
    }
    agent.setRuntimeState('thinking');
    await this.transitionAgentFsm(agentId, 'S_task_received', { task, correlationId: options.correlationId });
    this.emit({
      type: 'agent.run.started',
      agentId,
      sessionId: ctx.sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: { task, correlationId: options.correlationId },
    });
    this.emit({ type: 'agent.status.changed', agentId, data: { from, to: 'thinking' } });

    let activeGrounding: GroundingRunResult | undefined;
    try {
      await this.transitionAgentFsm(agentId, 'S_context_loading', { task, correlationId: options.correlationId });
      const delegationAssessment = this.shouldAssessAgentDelegation(agent.getInfo(), task);
      const recursiveDelegation = options.disableRecursiveDelegation
        ? { action: 'solve_directly', reason: 'Recursive delegation disabled for this run.' } satisfies DelegationDecision
        : !delegationAssessment.assess
          ? {
            action: 'solve_directly',
            reason: delegationAssessment.reason,
          } satisfies DelegationDecision
          : await this.decideAgentDelegation(
            agent.getInfo(),
            task,
            options.correlationId ?? this.createCorrelationId()
          );
      if (!options.disableRecursiveDelegation && !delegationAssessment.assess) {
        this.emit({
          type: 'agent.delegation.assessment.skipped',
          agentId,
          correlationId: options.correlationId,
          data: {
            reason: delegationAssessment.reason,
            teamId: agent.getIdentity().teamId,
          },
        });
      }
      this.emit({
        type: 'delegation.decision',
        agentId,
        data: {
          correlationId: options.correlationId,
          scope: 'agent',
          action: recursiveDelegation.action,
          reason: recursiveDelegation.reason,
          agents: recursiveDelegation.action === 'spawn_subagents' ? recursiveDelegation.agents : [],
        },
      });
      if (recursiveDelegation.action === 'spawn_subagents' && recursiveDelegation.agents.length > 0) {
        return await this.runAgentDelegatedChildren(agentId, task, recursiveDelegation, usageBefore, options);
      }
      if (recursiveDelegation.action !== 'spawn_subagents') {
        this.emit({
          type: 'delegation.skipped',
          agentId,
          data: {
            correlationId: options.correlationId,
            action: recursiveDelegation.action,
            reason: recursiveDelegation.reason,
          },
        });
      }
      let grounding = await this.runGroundingCheck(agentId, task, {
        ...options,
        intentTask: immutableActorTask,
        ...(priorIndependentVerification
          ? {
            initialPlans: [{
              toolName: 'shell.exec',
              params: { ...priorIndependentVerification.params },
              reason: 'Rerun the current authoritative verifier independently against the post-mutation workspace.',
              groundingRequired: true,
            }],
            skipInitialModelPlanning: true,
          }
          : {}),
        onBeforeExecution: async initialPlans => {
          await this.transitionAgentFsm(agentId, 'S_tool_calling', {
            toolCalls: initialPlans.map(call => call.toolName),
            correlationId: options.correlationId,
          });
        },
      });
      activeGrounding = grounding;
      if (actorVerificationEvidenceRequired
        && !actorDiagnosticProbeRequired
        && !grounding.toolCalls.some(call => isWorkspaceVerificationCall(call))) {
        const priorVerification = [...(options.priorToolCalls ?? [])].reverse()
          .find(call => isWorkspaceVerificationCall(call));
        if (priorVerification) {
          const verification = await this.runGroundingCheck(agentId, task, {
            ...options,
            intentTask: immutableActorTask,
            priorToolCalls: grounding.toolCalls,
            initialPlans: [{
              toolName: 'shell.exec',
              params: { ...priorVerification.params },
              reason: 'Run a fresh independent verification for the tester role instead of reusing another actor\'s cached result.',
              groundingRequired: true,
            }],
            skipInitialModelPlanning: true,
          });
          grounding = this.combineGroundingRuns([grounding, verification]);
          activeGrounding = grounding;
        }
      }
      if (actorVerificationEvidenceRequired && !actorDiagnosticProbeRequired) {
        const verificationCalls = grounding.toolCalls.filter(call =>
          isWorkspaceVerificationCall(call)
        );
        this.emit({
          type: verificationCalls.length > 0
            ? 'agent.verification.evidence.completed'
            : 'agent.verification.evidence.missing',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            attempts: verificationCalls.length,
            successful: verificationCalls.filter(call =>
              isSuccessfulWorkspaceVerificationCall(call)
            ).length,
          },
        });
        if (verificationCalls.length === 0) {
          throw new Error(
            'Tester execution produced no fresh verification command evidence'
          );
        }
      }
      if (actorDiagnosticProbeRequired) {
        const diagnosticCalls = grounding.toolCalls.filter(call =>
          this.isFocusedVerifierDiagnosticCall(call)
        );
        this.emit({
          type: diagnosticCalls.length > 0
            ? 'agent.verifier_diagnostic.completed'
            : 'agent.verifier_diagnostic.missing',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            focusedReproductions: diagnosticCalls.length,
            commands: diagnosticCalls.map(call =>
              String(call.params.command ?? '').slice(0, 500)
            ),
          },
        });
        if (diagnosticCalls.length === 0) {
          throw new Error(
            'Verifier diagnostic probe produced no fresh focused fixture reproduction'
          );
        }
      }
      if (actorWorkspaceExecutionRequired) {
        let closure = this.analyzeWorkspaceExecutionClosure(grounding.toolCalls);
        let continuation = 0;
        while (!closure.closed) {
          continuation += 1;
          const next = await this.runGroundingCheck(
            agentId,
            this.buildAgentExecutionClosureTask(task, grounding, closure, continuation),
            {
              ...options,
              intentTask: immutableActorTask,
              priorToolCalls: [
                ...(options.priorToolCalls ?? []),
                ...grounding.toolCalls,
              ],
            }
          );
          this.emit({
            type: 'agent.execution.closure.attempted',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              continuation,
              toolCalls: next.toolCalls.length,
              successfulToolCalls: next.toolCalls.filter(call => call.success).length,
              previousClosure: closure,
            },
          });
          if (!this.agentExecutionContinuationAdvanced(next)) {
            this.emit({
              type: 'agent.execution.no_progress',
              agentId,
              sessionId: ctx.sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: {
                continuation,
                closure,
                reason: 'The execution planner produced no new successful evidence, mutation, or verifier feedback.',
              },
            });
            throw new Error(
              'Workspace execution remained open: no grounded mutation-and-verification progress was produced'
            );
          }
          grounding = this.combineGroundingRuns([grounding, next]);
          activeGrounding = grounding;
          closure = this.analyzeWorkspaceExecutionClosure(grounding.toolCalls);
        }
        this.emit({
          type: 'agent.execution.closure.completed',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: { continuations: continuation, ...closure },
        });
      }
      await this.transitionAgentFsm(agentId, 'S_reasoning', { task, correlationId: options.correlationId });
      let result = '';
      const freshVerifierDiagnostic = actorDiagnosticProbeRequired
        && grounding.toolCalls.some(call =>
          this.isFocusedVerifierDiagnosticCall(call)
        );
      const freshIndependentVerification = priorIndependentVerification
        && grounding.toolCalls.some(call =>
          isSuccessfulWorkspaceVerificationCall(call)
        );
      if (freshVerifierDiagnostic) {
        result = this.buildDeterministicVerifierDiagnosticResult(grounding);
        this.emit({
          type: 'agent.verifier_diagnostic.summary.deterministic',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            focusedReproductions: grounding.toolCalls.filter(call =>
              this.isFocusedVerifierDiagnosticCall(call)
            ).length,
            reason: 'executable_probe_evidence_flows_directly_to_repair_synthesis',
            resultChars: result.length,
          },
        });
      } else if (freshIndependentVerification) {
        result = [
          '[runtime_independent_verification_closure]',
          'The runtime reran the authoritative post-mutation verifier for this tester role.',
          (grounding.evidence.toolResultSummary ?? grounding.context).slice(-8_000),
        ].join('\n\n');
        this.emit({
          type: 'agent.verification.summary.deterministic',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            successfulVerificationCalls: grounding.toolCalls.filter(call =>
              isSuccessfulWorkspaceVerificationCall(call)
            ).length,
            reason: 'authoritative_verifier_result_requires_no_additional_inference',
          },
        });
      } else {
        this.emit({ type: 'agent.llm.called', agentId, data: { task } });
        const communicationContext = agent.getCommunicationContext();
        const rawObservation = [
          this.buildGroundedTask(
            this.agentTaskObservationReference(agentId, task),
            grounding
          ),
          communicationContext
            ? `<system_communication_context protocol="${communicationContext.protocolId}">\n${communicationContext.rendered}\n</system_communication_context>`
            : '',
        ].filter(Boolean).join('\n\n');
        const observation = this.constrainAgentObservation(
          agent,
          rawObservation,
          options.correlationId,
          'agent.task_execution'
        );
        await agent.step(observation);
        const stepError = agent.getInfo().error;
        if (stepError) {
          if (/^Action error:\s*Validation failed:/i.test(stepError)) {
            this.emit({
              type: 'agent.output.action_validation.recovery.started',
              agentId,
              sessionId: ctx.sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: {
                task,
                error: stepError.slice(0, 1_000),
                recovery: 'complete_bounded_task_without_invalid_action',
              },
            });
            result = '';
          } else {
            throw new Error(stepError.replace(/^Error:\s*/, ''));
          }
        } else {
          result = session
            ? await this.drainAgentOutput(session.messageQueue, agent.name)
            : agent.getInfo().lastResult ?? '';
        }
        if (!result.trim()) {
          const recoveryPrompt = [
            `Complete this bounded assigned task directly:\n${this.isolatedAgentTaskObservation(task)}`,
            `Available runtime grounding:\n${(grounding.evidence.toolResultSummary ?? grounding.context).slice(-12_000) || 'No external tool evidence was required.'}`,
            stepError
              ? `The previous attempt produced an invalid runtime action instead of a result:\n${stepError.slice(0, 1_000)}`
              : 'The previous provider completion produced no visible result.',
            'Return a complete visible task result now.',
            'Do not return an empty response. Do not request a child agent or tool unless the task explicitly requires one and the runtime has authorized it.',
          ].join('\n\n');
          this.ensureAgentSynthesisBudget(
            agent,
            recoveryPrompt,
            'agent.empty_output_recovery',
            options.correlationId ?? this.createCorrelationId(),
            1_536
          );
          this.emit({
            type: 'agent.output.empty.recovery.started',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              task,
              reason: stepError ? 'invalid_action_without_result' : 'model_returned_empty_visible_output',
            },
          });
          result = await this.completeAsAgent(
            agent,
            recoveryPrompt,
            'agent.empty_output_recovery',
            options.correlationId ?? this.createCorrelationId(),
            {
              isolatedContext: true,
              temperature: 0,
              maxOutputTokens: this.reasoningAwareCompletionTokenBudget(1_536),
            }
          );
          if (!result.trim()) {
            this.emit({
              type: 'agent.output.empty.recovery.failed',
              agentId,
              sessionId: ctx.sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: { task, reason: 'recovery_returned_empty_visible_output' },
            });
            throw new Error('Agent returned no visible result after an isolated completion recovery');
          }
          this.emit({
            type: 'agent.output.empty.recovery.completed',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              task,
              recoveredCharacters: result.length,
              recoveredFromInvalidAction: Boolean(stepError),
            },
          });
        }
      }
      let forcedExecutionContinuationAttempted = false;
      while (true) {
        const unresolvedToolIntent = this.containsUnresolvedToolIntent(result);
        const workspaceExecutionOpen = actorWorkspaceExecutionRequired
          && !this.analyzeWorkspaceExecutionClosure(
            grounding.toolCalls,
            undefined,
            false
          ).closed;
        if (!unresolvedToolIntent && !workspaceExecutionOpen) break;

        if (!unresolvedToolIntent) {
          if (forcedExecutionContinuationAttempted) break;
          forcedExecutionContinuationAttempted = true;
          result = await this.completeAsAgent(
            agent,
            [
              `Task reference:\n${this.isolatedAgentTaskObservation(task)}`,
              `Runtime-provided evidence:\n${(grounding.evidence.toolResultSummary ?? grounding.context).slice(-12_000)}`,
              'The workspace execution path is still open: a mutation and successful post-mutation verification have not both been observed.',
              'Produce the final task result from the evidence above. Do so only if the task is genuinely complete.',
              'Otherwise emit the next concrete authorized tool request. Runtime will execute it; do not merely describe the command or ask for permission.',
            ].join('\n\n'),
            'agent.output_execution_continue',
            options.correlationId ?? this.createCorrelationId(),
            {
              isolatedContext: true,
              temperature: 0,
              maxOutputTokens: 4_000,
            }
          );
          continue;
        }

        const recoveryPlans = this.extractUnresolvedToolPlans(agentId, result);
        const novelRecoveryPlans = recoveryPlans.filter(plan => {
          const cached = this.cachedToolPlanDecision(plan, grounding.toolCalls);
          if (!cached.skip) return true;
          this.emit({
            type: 'agent.output.tool_intent.recovery.skipped',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            data: {
              task,
              toolName: plan.toolName,
              params: plan.params,
              reason: cached.reason,
            },
          });
          return false;
        });
        if (novelRecoveryPlans.length > 0) {
          this.emit({
            type: 'agent.output.tool_intent.recovery.started',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            data: {
              task,
              tools: novelRecoveryPlans.map(plan => plan.toolName),
            },
          });
          const recovered = await this.runGroundingCheck(agentId, task, {
            ...options,
            intentTask: immutableActorTask,
            initialPlans: novelRecoveryPlans,
            priorToolCalls: grounding.toolCalls,
            skipInitialModelPlanning: true,
          });
          grounding = this.combineGroundingRuns([grounding, recovered]);
          activeGrounding = grounding;
          this.emit({
            type: recovered.toolLoop.successfulCalls > 0
              ? 'agent.output.tool_intent.recovery.completed'
              : 'agent.output.tool_intent.recovery.failed',
            agentId,
            sessionId: ctx.sessionId,
            correlationId: options.correlationId,
            data: {
              task,
              successfulCalls: recovered.toolLoop.successfulCalls,
              failedCalls: recovered.toolLoop.failedCalls,
              stopReason: recovered.toolLoop.stopReason,
            },
          });
          if (recovered.toolCalls.length > 0) {
            forcedExecutionContinuationAttempted = false;
            result = await this.completeAsAgent(
              agent,
              [
                `Task reference:\n${this.isolatedAgentTaskObservation(task)}`,
                `Runtime-provided evidence:\n${(grounding.evidence.toolResultSummary ?? grounding.context).slice(-12_000)}`,
                'Runtime executed the model-authored tool request through the authorized tool layer.',
                'Produce the final task result from the evidence above. Do so only if the task is genuinely complete.',
                'If another tool call is necessary, emit the next concrete authorized tool request instead of describing future work. Runtime will continue this execution path.',
              ].join('\n\n'),
              'agent.output_tool_continue',
              options.correlationId ?? this.createCorrelationId(),
              {
                isolatedContext: true,
                temperature: 0,
                maxOutputTokens: 4_000,
              }
            );
            continue;
          }
        }
        break;
      }
      if (this.containsUnresolvedToolIntent(result)) {
        if (!grounding.evidence.toolGrounded) {
          throw new Error('Agent returned an unexecuted tool request without runtime grounding evidence');
        }
        this.emit({
          type: 'agent.output.repair.started',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          data: { reason: 'unexecuted_tool_intent', task },
        });
        result = await this.completeAsAgent(
          agent,
          [
            `Task reference:\n${this.isolatedAgentTaskObservation(task)}`,
            `Runtime-provided evidence:\n${grounding.evidence.toolResultSummary ?? grounding.context}`,
            'Produce the final task result from the evidence above.',
            'Do not emit tool-call markup, JSON tool requests, or claim that a tool still needs to run.',
          ].join('\n\n'),
          'agent.output_repair',
          options.correlationId ?? this.createCorrelationId(),
          {
            isolatedContext: true,
            temperature: 0,
            maxOutputTokens: 2_000,
          }
        );
        if (this.containsUnresolvedToolIntent(result)) {
          throw new Error('Agent output repair still contained an unexecuted tool request');
        }
        this.emit({
          type: 'agent.output.repair.completed',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          data: { task },
        });
      }
      const evidenceContradictions = grounding.evidence.toolGrounded
        ? this.detectEvidenceContradictions(result, grounding.evidence)
        : [];
      for (const contradiction of evidenceContradictions) {
        grounding.warnings.push(contradiction);
        this.emit({
          type: 'agent.grounding.contradiction',
          agentId,
          sessionId: ctx.sessionId,
          correlationId: options.correlationId,
          data: { warning: contradiction, observedPaths: grounding.evidence.observedPaths.slice(0, 30) },
        });
      }

      agent.setRuntimeState('done');
      await this.transitionAgentFsm(agentId, 'S_responding', { correlationId: options.correlationId });

      const usageAfter = agent.getUsage();
      const usageDelta = this.usageDifference(usageBefore, usageAfter);
      this.recordTurnUsage(usageDelta);
      this.emit({ type: 'budget.updated', agentId, data: { ...usageDelta } });
      this.emit({ type: 'agent.status.changed', agentId, data: { from: 'thinking', to: 'done' } });

      let evidence: RunEvidence = {
        ...grounding.evidence,
        outputGrounded: grounding.evidence.toolGrounded
          ? evidenceContradictions.length === 0
            && this.resultIncludesEvidence(result || agent.getInfo().lastResult || '', grounding.evidence)
          : grounding.evidence.outputGrounded,
      };
      const warnings = [...grounding.warnings];
      if (grounding.evidence.toolGrounded && !evidence.outputGrounded && evidenceContradictions.length === 0) {
        result = this.attachRuntimeEvidence(result, grounding.evidence);
        evidence = { ...evidence, outputGrounded: true };
        warnings.push('Runtime appended structured tool evidence because the model response omitted concrete observed paths.');
        this.emit({
          type: 'agent.output.evidence.attached',
          agentId,
          data: {
            observedPathCount: grounding.evidence.observedPaths.length,
            correlationId: options.correlationId,
          },
        });
      }
      this.emit({
        type: 'agent.run.completed',
        agentId,
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          task,
          correlationId: options.correlationId,
          totalTokens: usageDelta.totalTokens,
          grounded: grounding.grounded && evidence.outputGrounded,
          evidence,
          warnings,
          toolLoop: grounding.toolLoop,
        },
      });
      await this.transitionAgentFsm(agentId, 'S_done', { correlationId: options.correlationId });
      this.settleAgentBudget(agentId, usageDelta, {
        success: true,
        evidenceGain: evidence.outputGrounded ? 0.95 : evidence.toolGrounded ? 0.5 : 0.15,
        uncertaintyReduction: grounding.grounded && evidence.outputGrounded ? 0.8 : 0.35,
        verificationGain: options.archetype === 'tester' ? (evidence.outputGrounded ? 0.9 : 0.45) : undefined,
      });
      await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(agent.getInfo()), {
        success: true,
        grounded: grounding.grounded && evidence.outputGrounded,
        totalTokens: usageDelta.totalTokens,
      }, options.patternId);

      const runResult: RunAgentResult = {
        agent: agent.getInfo(),
        result: result || agent.getInfo().lastResult || '',
        usage: usageDelta,
        toolCalls: grounding.toolCalls,
        evidence,
        grounded: grounding.grounded && evidence.outputGrounded,
        warnings,
        toolLoop: grounding.toolLoop,
      };
      await this.finalizeActorLifecycle(agentId, 'success', options.correlationId);
      return runResult;
    } catch (error) {
      const failure = error instanceof Error ? error : new Error(String(error));
      const usageDelta = this.usageDifference(usageBefore, agent.getUsage());
      this.recordTurnUsage(usageDelta);
      this.emit({
        type: 'budget.updated',
        agentId,
        correlationId: options.correlationId,
        data: { failed: true, ...usageDelta },
      });
      if (activeGrounding) {
        Object.assign(failure, {
          runtimeToolCalls: activeGrounding.toolCalls,
          runtimeGrounding: activeGrounding,
        });
      }
      Object.assign(failure, { runtimeUsage: usageDelta });
      const message = failure.message;
      agent.addToMemory('result', `Error: ${message}`);
      agent.setRuntimeState('failed');
      this.emit({ type: 'agent.status.changed', agentId, data: { from: 'thinking', to: 'failed' } });
      if (this.requireAgentFsm(agentId).getState() !== 'S_failed') {
        await this.transitionAgentFsm(agentId, 'S_failed', { failed: true, error: message, correlationId: options.correlationId });
      }
      this.emit({
        type: 'agent.run.failed',
        agentId,
        sessionId: ctx.sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          task,
          error: message,
          correlationId: options.correlationId,
          totalTokens: usageDelta.totalTokens,
        },
      });
      if (usageDelta.totalTokens > 0) {
        this.settleAgentBudget(agentId, usageDelta, {
          success: false,
          error: message,
          evidenceGain: activeGrounding?.evidence.toolGrounded ? 0.25 : 0,
          uncertaintyReduction: activeGrounding?.grounded ? 0.2 : 0,
        });
      } else {
        this.releaseAgentBudget(agentId, 'agent_run_failed_before_usage');
      }
      await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(agent.getInfo()), {
        success: false,
        grounded: false,
        totalTokens: usageDelta.totalTokens,
      }, options.patternId);
      await this.finalizeActorLifecycle(agentId, 'failure', options.correlationId);
      throw failure;
    }
  }

  getChildren(agentId: string): AgentInfo[] {
    const ctx = this.getContext();
    return ctx.manager.listAgentInfo()
      .filter(agent => agent.identity.parentId === agentId);
  }

  getActorLifecycle(actorId?: string): ActorLifecycleRecord | ActorLifecycleRecord[] | undefined {
    return actorId ? this.lifecycle.get(actorId) : this.lifecycle.list();
  }

  async getPersistedActors(actorKind?: ActorKind): Promise<PersistedActorSnapshot[]> {
    return this.getContext().memory.listActorSnapshots(actorKind);
  }

  async setActorLifecycle(
    actorId: string,
    action: ActorLifecycleAction,
    options: { cascade?: boolean; reason?: string; correlationId?: string } = {}
  ): Promise<ActorLifecycleRecord> {
    if (actorId === 'root') throw new Error('The root agent cannot be released or persisted as a derived actor');
    const record = this.lifecycle.get(actorId);
    if (!record) throw new Error(`Lifecycle actor "${actorId}" not found`);
    this.assertActorLifecycleManageable(record.actorKind, actorId);
    const decision = this.lifecycle.decide(actorId, 'manual', {
      action,
      correlationId: options.correlationId,
      reason: options.reason ?? `Manual lifecycle action: ${action}.`,
    });
    if (options.cascade !== undefined) decision.cascade = options.cascade;
    return this.applyActorLifecycleDecision(decision);
  }

  async restoreActor(actorId: string): Promise<AgentInfo | TeamRuntimeState> {
    const ctx = this.getContext();
    if (ctx.manager.getAgentById(actorId) || this.teams.get(actorId)) {
      throw new Error(`Actor "${actorId}" is already active`);
    }
    const snapshot = await ctx.memory.readActorSnapshot(actorId);
    if (!snapshot) throw new Error(`Persisted actor "${actorId}" not found`);
    let restored: AgentInfo | TeamRuntimeState;
    if (snapshot.actorKind === 'agent') {
      const spec = snapshot.restore as unknown as SpawnAgentSpec;
      restored = await this.spawnAgent({
        ...spec,
        instanceId: actorId,
        lifecycle: { ...snapshot.policy },
        lifecycleOrigin: 'restored',
      });
    } else {
      const spec = snapshot.restore as unknown as SpawnTeamSpec;
      restored = await this.spawnTeam({
        ...spec,
        instanceId: actorId,
        lifecycle: { ...snapshot.policy },
        lifecycleOrigin: 'restored',
      });
    }
    await ctx.memory.deleteActorSnapshot(actorId, snapshot.actorKind);
    this.lifecycle.markRestored(actorId);
    this.emit({
      type: 'actor.lifecycle.restored',
      agentId: actorId,
      sessionId: ctx.sessionId,
      data: { actorId, actorKind: snapshot.actorKind, persistedAt: snapshot.persistedAt },
    });
    return restored;
  }

  private async finalizeActorLifecycle(
    actorId: string,
    outcome: ActorLifecycleOutcome,
    correlationId?: string
  ): Promise<ActorLifecycleRecord | undefined> {
    const record = this.lifecycle.get(actorId);
    if (!record) return undefined;
    try {
      const decision = this.lifecycle.decide(actorId, outcome, { correlationId });
      return await this.applyActorLifecycleDecision(decision);
    } catch (error) {
      this.emit({
        type: 'actor.lifecycle.failed',
        agentId: actorId,
        sessionId: this.getContext().sessionId,
        correlationId,
        data: { actorId, error: error instanceof Error ? error.message : String(error) },
      });
      return this.lifecycle.get(actorId);
    }
  }

  private activateActorLifecycle(actorId: string, correlationId?: string): void {
    const current = this.lifecycle.get(actorId);
    if (!current || current.status === 'active') return;
    const record = this.lifecycle.markActive(actorId);
    this.emit({
      type: 'actor.lifecycle.activated',
      agentId: actorId,
      sessionId: this.getContext().sessionId,
      correlationId,
      data: {
        actorId,
        actorKind: record.actorKind,
        previousStatus: current.status,
      },
    });
  }

  private async applyActorLifecycleDecision(
    decision: ReturnType<ActorLifecycleRegistry['decide']>
  ): Promise<ActorLifecycleRecord> {
    const ctx = this.getContext();
    this.emit({
      type: 'actor.lifecycle.decided',
      agentId: decision.actorId,
      sessionId: ctx.sessionId,
      correlationId: decision.correlationId,
      data: { ...decision },
    });

    let snapshotPath: string | undefined;
    if (decision.action === 'persist') {
      snapshotPath = await this.persistActorSnapshot(decision.actorId, decision.actorKind);
    }

    if (decision.action !== 'retain_session') {
      if (decision.cascade) {
        const children = decision.actorKind === 'agent'
          ? [
            ...this.getChildren(decision.actorId).map(agent => ({ id: agent.identity.id, kind: 'agent' as const })),
            ...this.teams.list()
              .filter(team => team.identity.parentAgentId === decision.actorId)
              .map(team => ({ id: team.identity.id, kind: 'team' as const })),
          ]
          : (this.teams.get(decision.actorId)?.memberAgentIds ?? []).map(id => ({ id, kind: 'agent' as const }));
        for (const child of children) {
          const childRecord = this.lifecycle.get(child.id);
          if (!childRecord || childRecord.status === 'released' || childRecord.status === 'persisted') continue;
          const childAction = decision.actorKind === 'team' && decision.action === 'persist'
            ? 'release'
            : decision.action;
          const childDecision = this.lifecycle.decide(child.id, decision.outcome, {
            action: childAction,
            correlationId: decision.correlationId,
            reason: `${childAction === 'persist' ? 'Persisted' : 'Released'} with parent ${decision.actorId}.`,
          });
          childDecision.cascade = true;
          await this.applyActorLifecycleDecision(childDecision);
        }
      }
      if (decision.actorKind === 'agent') await this.releaseAgentRuntime(decision.actorId);
      else this.releaseTeamRuntime(decision.actorId);
    }

    const record = this.lifecycle.markApplied(decision.actorId, decision, snapshotPath);
    this.emit({
      type: 'actor.lifecycle.applied',
      agentId: decision.actorId,
      sessionId: ctx.sessionId,
      correlationId: decision.correlationId,
      data: {
        actorId: decision.actorId,
        actorKind: decision.actorKind,
        action: decision.action,
        status: record.status,
        snapshotPath,
      },
    });
    return record;
  }

  private async persistActorSnapshot(actorId: string, actorKind: ActorKind): Promise<string> {
    const ctx = this.getContext();
    const record = this.lifecycle.get(actorId);
    if (!record) throw new Error(`Lifecycle actor "${actorId}" not found`);
    const snapshot: PersistedActorSnapshot = {
      version: 1,
      actorId,
      actorKind,
      status: 'dormant',
      origin: record.origin,
      parentId: record.parentId,
      sessionId: ctx.sessionId,
      persistedAt: new Date().toISOString(),
      policy: { ...record.policy },
      restore: actorKind === 'agent'
        ? { ...(this.agentRestoreSpecs.get(actorId) ?? {}) }
        : { ...(this.teamRestoreSpecs.get(actorId) ?? {}) },
    };
    if (actorKind === 'agent') {
      const agent = ctx.manager.getAgentById(actorId);
      if (!agent) throw new Error(`Active agent "${actorId}" not found`);
      snapshot.agent = agent.getInfo();
    } else {
      const team = this.teams.get(actorId);
      if (!team) throw new Error(`Active team "${actorId}" not found`);
      snapshot.team = team;
    }
    const snapshotPath = await ctx.memory.writeActorSnapshot(snapshot);
    this.emit({
      type: 'actor.lifecycle.persisted',
      agentId: actorId,
      sessionId: ctx.sessionId,
      data: { actorId, actorKind, snapshotPath },
    });
    return snapshotPath;
  }

  private async releaseAgentRuntime(agentId: string): Promise<void> {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    if (!agent || agentId === 'root') return;
    const info = agent.getInfo();
    this.archivedAgentUsage.set(agentId, this.sumUsage([
      ...(this.archivedAgentUsage.has(agentId) ? [this.archivedAgentUsage.get(agentId)!] : []),
      this.toTokenUsage(info.usage),
    ]));
    this.archivedAgentInfo.set(agentId, info);
    for (const sessionId of ctx.manager.listSessions()) await agent.cleanup(sessionId);
    ctx.manager.removeAgent(info.name);
    this.agentBindings.delete(agentId);
    this.agentFsms.delete(agentId);
    this.agentBudgetAllocations.delete(agentId);
    this.agentBudgetLimits.delete(agentId);
    this.toolCallCounts.delete(agentId);
    this.agentRestoreSpecs.delete(agentId);
  }

  private releaseTeamRuntime(teamId: string): void {
    const team = this.teams.remove(teamId);
    if (!team) return;
    this.archivedTeamUsage.set(teamId, this.sumUsage([
      ...(this.archivedTeamUsage.has(teamId) ? [this.archivedTeamUsage.get(teamId)!] : []),
      team.tokenUsage,
    ]));
    this.archivedTeamSynthesisUsage.set(teamId, this.sumUsage([
      ...(this.archivedTeamSynthesisUsage.has(teamId) ? [this.archivedTeamSynthesisUsage.get(teamId)!] : []),
      team.synthesisUsage,
    ]));
    this.archivedTeamStates.set(teamId, team);
    this.teamMemberPlans.delete(teamId);
    this.releaseTeamSpawnReservation(teamId, 'team_runtime_released');
    this.teamRestoreSpecs.delete(teamId);
  }

  private assertActorLifecycleManageable(actorKind: ActorKind, actorId: string): void {
    if (actorKind === 'agent') {
      const state = this.getContext().manager.getAgentById(actorId)?.getInfo().state;
      if (!state) throw new Error(`Active agent "${actorId}" not found`);
      if (state === 'thinking' || state === 'calling_tool' || state === 'synthesizing' || state === 'waiting') {
        throw new Error(`Agent "${actorId}" cannot change lifecycle while state is ${state}`);
      }
      return;
    }
    const status = this.teams.get(actorId)?.status;
    if (!status) throw new Error(`Active team "${actorId}" not found`);
    if (status === 'running' || status === 'waiting' || status === 'synthesizing') {
      throw new Error(`Team "${actorId}" cannot change lifecycle while status is ${status}`);
    }
  }

  private resolveLifecyclePolicy(
    origin: ActorLifecycleOrigin,
    override?: Partial<ActorLifecyclePolicy>
  ): ActorLifecyclePolicy {
    const defaults = this.workspaceRuntimeConfig?.lifecycle;
    const mode: ActorLifecycleMode = origin === 'automatic_delegation'
      ? defaults?.automaticDelegation ?? 'release'
      : origin === 'team_member'
        ? defaults?.teamMember ?? 'retain_session'
        : origin === 'evolution'
          ? defaults?.evolutionCandidate ?? 'release'
          : defaults?.manual ?? 'retain_session';
    return {
      mode: override?.mode ?? mode,
      retainOnFailure: override?.retainOnFailure ?? defaults?.retainFailures ?? true,
      cascade: override?.cascade ?? defaults?.cascade ?? true,
    };
  }

  private lifecycleOriginForSource(source?: string, teamId?: string): ActorLifecycleOrigin {
    if (teamId) return 'team_member';
    if (source === 'cli' || source === 'server' || !source) return 'manual';
    if (source.startsWith('evo')) return 'evolution';
    return 'automatic_delegation';
  }

  private inheritParentLifecyclePolicy(
    parentId: string,
    origin: ActorLifecycleOrigin
  ): Partial<ActorLifecyclePolicy> | undefined {
    if (origin !== 'automatic_delegation' || parentId === 'root') return undefined;
    const parent = this.lifecycle.get(parentId);
    if (!parent) return undefined;
    return {
      mode: parent.policy.mode,
      retainOnFailure: parent.policy.retainOnFailure,
      cascade: parent.policy.cascade,
    };
  }

  getParent(agentId: string): AgentInfo | undefined {
    const ctx = this.getContext();
    const agent = ctx.manager.getAgentById(agentId);
    const parentId = agent?.getIdentity().parentId;
    return parentId ? ctx.manager.getAgentById(parentId)?.getInfo() : undefined;
  }

  getAgentTree(): AgentTreeNode {
    const ctx = this.getContext();
    const root = ctx.agent.getInfo();
    return this.buildAgentTree(root);
  }

  private toTokenUsage(usage: AgentUsage): TokenUsage {
    return {
      ...usage,
      inputTokens: usage.inputTokens ?? usage.promptTokens,
      outputTokens: usage.outputTokens ?? usage.completionTokens,
      thinkingTokens: usage.thinkingTokens ?? null,
      cachedInputTokens: usage.cachedInputTokens ?? null,
      cacheCreationInputTokens: usage.cacheCreationInputTokens ?? null,
    };
  }

  private usageDifference(before: AgentUsage, after: AgentUsage): TokenUsage {
    const nullableDelta = (previous: number | null | undefined, current: number | null | undefined): number | null => {
      if (previous === null && current === null) return null;
      if (previous === undefined && current === undefined) return null;
      return Math.max(0, (current ?? 0) - (previous ?? 0));
    };
    return this.toTokenUsage({
      llmCalls: after.llmCalls - before.llmCalls,
      promptTokens: after.promptTokens - before.promptTokens,
      completionTokens: after.completionTokens - before.completionTokens,
      totalTokens: after.totalTokens - before.totalTokens,
      inputTokens: (after.inputTokens ?? after.promptTokens) - (before.inputTokens ?? before.promptTokens),
      outputTokens: (after.outputTokens ?? after.completionTokens) - (before.outputTokens ?? before.completionTokens),
      thinkingTokens: nullableDelta(before.thinkingTokens, after.thinkingTokens),
      thinkingAccountingTokens: Math.max(
        0,
        (after.thinkingAccountingTokens ?? after.thinkingTokens ?? after.totalTokens)
          - (before.thinkingAccountingTokens ?? before.thinkingTokens ?? before.totalTokens)
      ),
      cachedInputTokens: nullableDelta(before.cachedInputTokens, after.cachedInputTokens),
      cacheCreationInputTokens: nullableDelta(before.cacheCreationInputTokens, after.cacheCreationInputTokens),
    });
  }

  private async processQueuedMessage(messageId: string): Promise<RuntimeMessage | undefined> {
    const ctx = this.getContext();
    const message = await ctx.queue.getMessage(messageId);
    if (!message) return undefined;
    const dequeued = await ctx.queue.dequeue({ to: message.to, kind: [message.kind], readyOnly: true });
    if (dequeued) this.deliverCommunicationContext(dequeued);
    return dequeued;
  }

  private deliverCommunicationContext(message: RuntimeMessage): void {
    const ctx = this.getContext();
    const recipient = ctx.manager.getAgentById(message.to);
    if (!recipient) return;
    const participantIds = new Set<string>([message.from, message.to]);
    for (const trace of ctx.communication.traces.list({
      sessionId: message.sessionId,
      correlationId: message.correlationId,
    })) {
      participantIds.add(trace.from.id);
      for (const actor of trace.to) participantIds.add(actor.id);
    }
    const participants = ctx.manager.listAgentInfo().filter(agent => participantIds.has(agent.identity.id));
    const task = this.extractMessageTask(message);
    const communicationContext = ctx.communication.buildContext({
      message,
      recipient: recipient.getInfo(),
      participants,
      task,
    });
    recipient.receiveCommunicationContext(communicationContext);
    this.emit({
      type: 'communication.context.delivered',
      agentId: recipient.id,
      sessionId: message.sessionId,
      correlationId: message.correlationId,
      nodeId: message.metadata?.nodeId,
      data: {
        messageId: message.id,
        protocolId: communicationContext.protocolId,
        traceCount: communicationContext.traces.length,
        participantIds: participants.map(agent => agent.identity.id),
      },
    });
  }

  private extractMessageTask(message: RuntimeMessage): string | undefined {
    if (!message.payload || typeof message.payload !== 'object') return undefined;
    const payload = message.payload as Record<string, unknown>;
    const value = payload.task ?? payload.input ?? payload.userTask;
    return typeof value === 'string' ? value : undefined;
  }

  private async transitionAgentFsm(
    agentId: string,
    state: Parameters<FSM['transition']>[0],
    data: Record<string, unknown> = {}
  ): Promise<void> {
    if (!state) return;
    const fsm = this.requireAgentFsm(agentId);
    const from = fsm.getState();
    try {
      await fsm.transition(state);
      this.emit({ type: 'agent.fsm.state', agentId, data: { from, state, ...data } });
    } catch (error) {
      this.emit({
        type: 'delegation.rejected',
        agentId,
        data: {
          reason: 'invalid_fsm_state',
          from,
          to: state,
          error: error instanceof Error ? error.message : String(error),
          ...data,
        },
      });
      throw error;
    }
  }

  private requireAgentFsm(agentId: string): FSM {
    const fsm = this.agentFsms.get(agentId);
    if (!fsm) throw new Error(`FSM for agent "${agentId}" not found`);
    return fsm;
  }

  private requireContextWindowManager(): ContextWindowManager {
    if (!this.contextWindowManager) throw new Error('ContextWindowManager is not initialized');
    return this.contextWindowManager;
  }

  private requireCommunicationManager(): AgentCommunicationManager {
    if (!this.communicationManager) throw new Error('AgentCommunicationManager is not initialized');
    return this.communicationManager;
  }

  private requireCandidatePlanner(): DefaultDelegationCandidatePlanner {
    if (!this.candidatePlanner) throw new Error('Delegation candidate planner is not initialized');
    return this.candidatePlanner;
  }

  private async prepareParentForDelegation(parentId: string, correlationId: string, task: string): Promise<void> {
    const parent = this.getContext().manager.getAgentById(parentId);
    if (!parent || parent.getState() === 'failed' || parent.getState() === 'stopped') return;
    const fsm = this.requireAgentFsm(parentId);
    if (fsm.getState() === 'S_done') {
      await this.transitionAgentFsm(parentId, 'S_ready', { correlationId, reason: 'manual_child_delegation' });
    }
    if (fsm.getState() === 'S_ready') {
      await this.transitionAgentFsm(parentId, 'S_task_received', { correlationId, task });
    }
    if (fsm.getState() === 'S_task_received') {
      await this.transitionAgentFsm(parentId, 'S_context_loading', { correlationId, task });
    }
    if (fsm.getState() === 'S_context_loading') {
      await this.transitionAgentFsm(parentId, 'S_planning', { correlationId, task });
    }
    if (fsm.getState() === 'S_planning') {
      await this.transitionAgentFsm(parentId, 'S_delegating', { correlationId, task });
    }
  }

  private async buildDelegationDecisionMetadata(decision: DelegationDecision): Promise<Record<string, unknown>> {
    const budget = this.getBudgetState();
    if (decision.action !== 'spawn_subagents') {
      return {
        budgetMode: budget.mode,
        remainingTokens: budget.remainingTokens,
        cacheUsed: false,
      };
    }

    const ctx = this.getContext();
    const agents = await Promise.all(decision.agents.map(async agent => {
      const [agentPattern, delegationPattern] = await Promise.all([
        ctx.memory.findAgentPattern(agent.archetype),
        ctx.memory.findDelegationPattern(agent.archetype, agent.task),
      ]);
      const patternIds = [
        typeof agentPattern?.id === 'string' ? agentPattern.id : undefined,
        typeof delegationPattern?.id === 'string' ? delegationPattern.id : undefined,
      ].filter((item): item is string => item !== undefined);

      return {
        ...agent,
        patternIds,
      };
    }));

    return {
      budgetMode: budget.mode,
      remainingTokens: budget.remainingTokens,
      cacheUsed: agents.some(agent => agent.patternIds.length > 0),
      agents,
    };
  }

  private async selectDelegationCandidate(
    parentId: string,
    task: string,
    decision: DelegationDecision,
    correlationId: string,
    scope: 'root' | 'agent',
    preserveRequestedPlan = false
  ): Promise<DelegationDecision> {
    if (decision.action !== 'spawn_subagents') return decision;
    const policy = this.getAgentPolicy(parentId);
    const budget = this.getBudgetState();
    const ctx = this.getContext();
    const parentToMProfile = ctx.manager.getAgentById(parentId)?.getIdentity().tomProfile;
    const tomSignals = this.deriveToMAnalysisSignals(parentId, correlationId, parentToMProfile);
    this.emit({
      type: 'tom.signals.collected',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: { ...tomSignals },
    });
    const tomAnalysis = this.tomPlanner.analyzeTask({
      task,
      parentId,
      parentProfile: parentToMProfile,
      signals: tomSignals,
    });
    const remainingAgentSlots = Math.min(
      policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      this.workspaceRuntimeConfig?.tom.maxAgentsPerDecision ?? 3
    );
    const tomEnabled = this.workspaceRuntimeConfig?.tom.enabled !== false;
    const stagedRootStep = scope === 'root'
      && (this.executionTrees.get(correlationId)?.steps.length ?? 0) === 0
      && this.requiresStagedDelegation(task);
    const completedPlans = tomEnabled
      ? this.tomPlanner.completePlans(
        tomAnalysis,
        decision.agents,
        this.workspaceRuntimeConfig?.tom.autoCompleteGaps === false || stagedRootStep || preserveRequestedPlan
          ? Math.min(decision.agents.length, remainingAgentSlots)
          : remainingAgentSlots
      )
      : decision.agents.slice(0, remainingAgentSlots);
    if (completedPlans.length === 0) {
      return {
        action: 'solve_directly',
        reason: `${decision.reason} Delegation skipped because no policy-approved agent slots remain.`,
      };
    }
    this.tomAnalyses.set(correlationId, tomAnalysis);
    this.emit({
      type: 'tom.task.analyzed',
      agentId: parentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        analysisId: tomAnalysis.id,
        rationale: tomAnalysis.rationale,
        requiresHigherOrderToM: tomAnalysis.requiresHigherOrderToM,
        gapCount: tomAnalysis.gaps.length,
        parentBeliefs: tomAnalysis.parentBeliefs,
        parentGoals: tomAnalysis.parentGoals,
        parentUncertainties: tomAnalysis.parentUncertainties,
        source: tomAnalysis.source,
        confidence: tomAnalysis.confidence,
        signals: tomAnalysis.signals,
      },
    });
    for (const gap of tomAnalysis.gaps) {
      this.emit({
        type: 'tom.gap.identified',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { ...gap, analysisId: tomAnalysis.id },
      });
    }
    if (tomAnalysis.requiresHigherOrderToM) {
      this.emit({
        type: 'tom.higher_order.required',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: { analysisId: tomAnalysis.id, reason: tomAnalysis.rationale },
      });
    }

    const explicitPlanCount = Math.min(decision.agents.length, completedPlans.length);
    const automaticallyCompletedGroundingPlans = new Set(
      completedPlans
        .slice(explicitPlanCount)
        .filter(plan =>
          /\bgrounded evidence\b/i.test(plan.tomProfile?.perspective ?? '')
          || (plan.tomProfile?.capabilityScope ?? []).some(capability =>
            /^(?:inspect_project|fs\.(?:list|read)|web\.(?:search|fetch))$/.test(capability)
          )
        )
        .map(plan => `${plan.archetype}\u0000${plan.name ?? ''}\u0000${plan.task}`)
    );
    const executableDecision = this.constrainDelegationToExecutablePlans(
      parentId,
      { ...decision, agents: completedPlans },
      correlationId,
      automaticallyCompletedGroundingPlans
    );
    if (executableDecision.action !== 'spawn_subagents') return executableDecision;
    const executablePlans = executableDecision.agents;

    const cacheHits = await Promise.all(executablePlans.map(async agent => {
      const [agentPattern, delegationPattern] = await Promise.all([
        ctx.memory.findAgentPattern(agent.archetype),
        ctx.memory.findDelegationPattern(agent.archetype, agent.task),
      ]);
      return Boolean(agentPattern || delegationPattern);
    }));
    const [agentPatterns, delegationPatterns] = await Promise.all([
      ctx.memory.getCachePatterns('agents'),
      ctx.memory.getCachePatterns('delegations'),
    ]);
    const archetypes = [...new Set(executablePlans.map(agent => agent.archetype))];
    const selection = await this.requireCandidatePlanner().select({
      parentId,
      correlationId,
      task,
      decision: executableDecision,
      allowedChildren: policy ? Math.max(0, policy.allowedChildren - policy.currentChildren) : 0,
      remainingTotalAgentsForTurn: this.getRemainingTotalAgentsForTurn(parentId, correlationId),
      budgetMode: budget.mode,
      remainingBudgetTokens: budget.remainingTokens,
      cacheUsed: cacheHits.some(Boolean),
      cachedPatterns: [...agentPatterns, ...delegationPatterns],
      allowedToolsByArchetype: Object.fromEntries(archetypes.map(archetype => [
        archetype,
        this.getAutomaticallyApprovedToolBindings(
          archetype,
          executablePlans.filter(plan => plan.archetype === archetype).map(plan => plan.task).join('\n'),
          executablePlans
            .filter(plan => plan.archetype === archetype)
            .flatMap(plan => plan.tools ?? [])
        ).map(binding => binding.name),
      ])),
      allowedSkillsByArchetype: Object.fromEntries(archetypes.map(archetype => [
        archetype,
        Array.from(new Set([
          ...this.getDefaultSkillBindings(archetype).filter(binding => binding.enabled).map(binding => binding.name),
          ...executablePlans
            .filter(plan => plan.archetype === archetype)
            .flatMap(plan => plan.skills ?? [])
            .filter(skill => skillRegistry.has(skill)),
        ])),
      ])),
      enforceMinimumToMCoverage: !preserveRequestedPlan
        && this.workspaceRuntimeConfig?.tom.enforceMinimumCoverage === true,
      parentToMProfile,
      tomAnalysis,
    });

    await this.recordEvolutionLifecycle(parentId, correlationId, scope, selection);
    this.emitDelegationCandidateEvents(parentId, correlationId, scope, selection);
    if (selection.selected?.tomCoverage) {
      this.emit({
        type: 'tom.delegation.coverage.evaluated',
        agentId: parentId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          analysisId: tomAnalysis.id,
          candidateId: selection.selected.id,
          ...selection.selected.tomCoverage,
        },
      });
    }
    return this.constrainDelegationToExecutablePlans(parentId, selection.decision, correlationId);
  }

  private deriveToMAnalysisSignals(
    parentId: string,
    correlationId: string,
    parentProfile?: ToMProfile
  ): ToMAnalysisSignals {
    const ctx = this.getContext();
    const current = this.communicationManager?.traces.list({
      sessionId: ctx.sessionId,
      correlationId,
      limit: this.workspaceRuntimeConfig?.communication.traceWindowSize ?? 200,
    }) ?? [];
    const traces = current;
    const participants = new Set(traces.flatMap(trace => [trace.from.id, ...trace.to.map(actor => actor.id)]));
    const failed = traces.filter(trace => trace.phase === 'failed');
    const cancelled = traces.filter(trace => trace.phase === 'cancelled');
    const toolResults = traces.filter(trace => trace.kind === 'tool.result' && trace.phase === 'completed');
    const evidence = traces.filter(trace =>
      trace.phase === 'completed'
      && ['tool.result', 'agent.result', 'team.result'].includes(trace.kind)
      && Boolean(trace.content?.trim())
    );
    const conflicting = traces.filter(trace =>
      /\b(conflict|contradict|disagree|inconsistent|unsupported|mismatch)\b|冲突|矛盾|不一致/.test(trace.content?.toLowerCase() ?? '')
    );
    const reliabilityConcerns = [
      ...(failed.length > 0 ? [`${failed.length} observable message(s) failed`] : []),
      ...(cancelled.length > 0 ? [`${cancelled.length} observable message(s) were cancelled`] : []),
    ];
    const evidenceOpportunity = traces.filter(trace =>
      ['agent.task', 'team.task', 'tool.call', 'tool.result', 'agent.result', 'team.result'].includes(trace.kind)
    ).length;
    const evidenceCoverage = evidenceOpportunity === 0
      ? 1
      : Math.min(1, (evidence.length + toolResults.length * 0.5) / evidenceOpportunity);
    const conflictLevel = participants.size === 0 ? 0 : Math.min(1, conflicting.length / participants.size);
    const profileUncertainty = parentProfile?.uncertainty.length ?? 0;
    const uncertaintyLevel = Math.min(1, profileUncertainty * 0.2 + failed.length * 0.15 + (1 - evidenceCoverage) * 0.35);
    return {
      traceCount: traces.length,
      participantCount: participants.size,
      failedTraceCount: failed.length,
      cancelledTraceCount: cancelled.length,
      toolResultCount: toolResults.length,
      evidenceTraceCount: evidence.length,
      conflictingTraceCount: conflicting.length,
      evidenceCoverage: Number(evidenceCoverage.toFixed(4)),
      conflictLevel: Number(conflictLevel.toFixed(4)),
      uncertaintyLevel: Number(uncertaintyLevel.toFixed(4)),
      observedKinds: [...new Set(traces.map(trace => trace.kind))],
      reliabilityConcerns,
    };
  }

  private async beforeDelegationScorerCall(
    input: DelegationCandidateInput,
    messages: LLMMessage[],
    options: LLMCompletionOptions
  ): Promise<LLMDelegationScorerInvocation> {
    const parent = this.getContext().manager.getAgentById(input.parentId);
    if (!parent) return { skip: true };

    const estimatedInputTokens = this.estimateTextTokens(
      messages.map(message => `${message.role}:${message.content}`).join('\n')
    );
    const activeAllocationId = this.agentBudgetAllocations.get(parent.id);
    const activeAllocation = activeAllocationId
      ? this.budgetMarket?.getAllocation(activeAllocationId)
      : undefined;
    const ownsAllocation = !activeAllocation || activeAllocation.status !== 'granted';
    const allocation = ownsAllocation
      ? await this.requestAgentBudget({
        parentId: parent.getIdentity().parentId ?? parent.id,
        requesterId: parent.id,
        archetype: parent.id === 'root' ? 'custom' : this.inferAgentArchetype(parent.getInfo()),
        correlationId: input.correlationId,
        requestedTokens: this.budgetRequestTokens(estimatedInputTokens, options.maxTokens ?? 700),
        minimumTokens: this.budgetMinimumTokens(estimatedInputTokens),
        priority: 'high',
        expectedUtility: 0.84,
        purpose: 'delegation.candidate_scoring',
      })
      : activeAllocation;

    if (allocation?.status === 'denied') {
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'budget_request_denied', allocationId: allocation.id },
      });
      return { skip: true };
    }

    const allocationRemaining = allocation?.status === 'granted'
      && !(this.hasUnlimitedBudgetSupply() && !ownsAllocation)
      ? Math.max(0, allocation.allocatedTokens - allocation.consumedTokens)
      : undefined;
    const agentRemaining = parent.getCompletionTokenLimit();
    const availableTokens = [allocationRemaining, agentRemaining]
      .filter((value): value is number => value !== undefined)
      .reduce<number | undefined>((minimum, value) => minimum === undefined ? value : Math.min(minimum, value), undefined);
    if (availableTokens !== undefined && availableTokens <= 0) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'delegation_scorer_allocation_exhausted');
      }
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'allocation_exhausted', availableTokens },
      });
      return { skip: true };
    }
    if (this.budgetAccountingDimension() === 'total_tokens'
      && availableTokens !== undefined
      && availableTokens <= estimatedInputTokens) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'delegation_scorer_input_exceeds_allocation');
      }
      this.emit({
        type: 'delegation.candidate.scoring.skipped',
        agentId: parent.id,
        correlationId: input.correlationId,
        data: { reason: 'insufficient_tokens_for_input', estimatedInputTokens, availableTokens },
      });
      return { skip: true };
    }

    const maxTokens = availableTokens === undefined
      ? options.maxTokens
      : Math.max(1, Math.min(
        options.maxTokens ?? 700,
        this.completionCapacity(availableTokens, estimatedInputTokens)
      ));
    return {
      options: { ...options, maxTokens },
      context: {
        parentId: parent.id,
        correlationId: input.correlationId,
        usageBefore: parent.getUsage(),
        allocation,
        ownsAllocation,
      } satisfies DelegationScorerBudgetContext,
    };
  }

  private afterDelegationScorerCall(
    completion: LLMCompletionResult,
    input: DelegationCandidateInput,
    hookContext?: unknown
  ): void {
    const context = hookContext as DelegationScorerBudgetContext | undefined;
    if (!context) return;
    const parent = this.getContext().manager.getAgentById(context.parentId);
    if (!parent) {
      this.releaseDelegationScorerBudget(context);
      return;
    }

    parent.recordRuntimeUsage(completion);
    const usage = this.usageDifference(context.usageBefore, parent.getUsage());
    if (context.ownsAllocation) {
      this.settleDirectBudget(parent.id, context.allocation, usage, context.correlationId);
    } else {
      this.consumeActiveAgentBudget(parent.id, usage, context.correlationId, 'delegation.candidate_scoring');
    }
    if (parent.id === 'root') this.recordTurnUsage(usage);
    this.emit({
      type: 'agent.llm.called',
      agentId: parent.id,
      correlationId: context.correlationId,
      data: {
        purpose: 'delegation.candidate_scoring',
        provider: this.getContext().llm?.name,
        model: completion.model ?? this.getContext().llm?.defaultModel,
        source: completion.usage?.source ?? 'estimated',
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        totalTokens: usage.totalTokens,
      },
    });
    this.emit({
      type: 'budget.updated',
      agentId: parent.id,
      correlationId: input.correlationId,
      data: { purpose: 'delegation.candidate_scoring', ...usage },
    });
  }

  private releaseDelegationScorerBudget(hookContext?: unknown): void {
    const context = hookContext as DelegationScorerBudgetContext | undefined;
    if (!context?.ownsAllocation || context.allocation?.status !== 'granted' || !this.budgetMarket) return;
    const released = this.budgetMarket.release(context.allocation.id, 'delegation_candidate_scoring_failed');
    if (released) {
      this.emit({
        type: 'budget.released',
        agentId: context.parentId,
        correlationId: context.correlationId,
        data: { allocationId: context.allocation.id, reason: 'delegation_candidate_scoring_failed' },
      });
    }
  }

  private emitDelegationCandidateEvents(
    parentId: string,
    correlationId: string,
    scope: 'root' | 'agent',
    selection: DelegationCandidateSelection
  ): void {
    for (const candidate of selection.candidates) {
      this.emit({
        type: 'delegation.candidate.generated',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          candidateId: candidate.id,
          source: candidate.source,
          agents: candidate.agents,
          expectedUtility: candidate.expectedUtility,
          expectedCostTokens: candidate.expectedCostTokens,
          score: candidate.score,
          scoreBreakdown: candidate.scoreBreakdown,
          investment: candidate.investment,
          lineage: candidate.lineage,
          rationale: candidate.rationale,
          tomCoverage: candidate.tomCoverage,
        },
      });
    }
    if (selection.selected) {
      this.emit({
        type: 'delegation.candidate.selected',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          candidateId: selection.selected.id,
          source: selection.selected.source,
          agents: selection.selected.agents,
          expectedUtility: selection.selected.expectedUtility,
          expectedCostTokens: selection.selected.expectedCostTokens,
          score: selection.selected.score,
          scoreBreakdown: selection.selected.scoreBreakdown,
          lineage: selection.selected.lineage,
          rationale: selection.selected.rationale,
          tomCoverage: selection.selected.tomCoverage,
          investment: selection.selected.investment,
        },
      });
    } else if (selection.rejectedReason) {
      this.emit({
        type: 'delegation.rejected',
        agentId: parentId,
        data: {
          correlationId,
          scope,
          reason: selection.rejectedReason,
        },
      });
    }
  }

  private async recordEvolutionLifecycle(
    parentId: string,
    correlationId: string,
    scope: 'root' | 'agent',
    selection: DelegationCandidateSelection
  ): Promise<void> {
    const ctx = this.getContext();
    const proposed = await this.enqueueMessage({
      kind: 'evo.propose',
      sessionId: ctx.sessionId,
      from: parentId,
      to: 'delegation.evolution',
      correlationId,
      payload: { scope, candidates: selection.candidates.map(candidate => ({ id: candidate.id, source: candidate.source, lineage: candidate.lineage })) },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(proposed.id);
    const evaluated = await this.enqueueMessage({
      kind: 'evo.evaluate',
      sessionId: ctx.sessionId,
      from: 'delegation.evolution',
      to: parentId,
      correlationId,
      parentMessageId: proposed.id,
      payload: {
        candidates: selection.candidates.map(candidate => ({
          id: candidate.id,
          score: candidate.score,
          scoreBreakdown: candidate.scoreBreakdown,
          investment: candidate.investment,
        })),
      },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(evaluated.id);
    const selected = await this.enqueueMessage({
      kind: 'evo.select',
      sessionId: ctx.sessionId,
      from: 'delegation.evolution',
      to: parentId,
      correlationId,
      parentMessageId: evaluated.id,
      payload: { selected: selection.selected?.id, rejectedReason: selection.rejectedReason },
      metadata: { agentId: parentId },
    });
    await this.processQueuedMessage(selected.id);
    await ctx.queue.ack(proposed.id);
    await ctx.queue.ack(evaluated.id);
    await ctx.queue.ack(selected.id);
    await ctx.memory.recordEvolutionRun({
      correlationId,
      parentId,
      scope,
      proposed: selection.candidates.map(candidate => ({
        id: candidate.id,
        source: candidate.source,
        score: candidate.score,
        scoreBreakdown: candidate.scoreBreakdown,
        investment: candidate.investment,
        lineage: candidate.lineage,
      })),
      selected: selection.selected?.id,
      rejectedReason: selection.rejectedReason,
    });
    this.emit({ type: 'evo.proposed', agentId: parentId, data: { correlationId, count: selection.candidates.length, scope } });
    this.emit({ type: 'evo.evaluated', agentId: parentId, data: { correlationId, count: selection.candidates.length, scope } });
    this.emit({ type: 'evo.selected', agentId: parentId, data: { correlationId, candidateId: selection.selected?.id, scope } });
    if (selection.selected?.source === 'mutated_from_cache') {
      this.emit({
        type: 'cache.mutated',
        agentId: parentId,
        data: { correlationId, candidateId: selection.selected.id, lineage: selection.selected.lineage },
      });
    }
  }

  private async transitionRootTurnState(state: Parameters<FSM['transition']>[0], data: Record<string, unknown> = {}): Promise<void> {
    const ctx = this.getContext();
    if (!state) return;
    await ctx.fsm.transition(state);
    this.emit({ type: 'turn.fsm.state', agentId: 'root', data: { state, ...data } });
  }

  private async recoverFailedRootTurn(correlationId: string, error: unknown): Promise<void> {
    const ctx = this.getContext();
    const message = error instanceof Error ? error.message : String(error);
    const failedState = ctx.fsm.getState();
    const tree = this.executionTrees.get(correlationId);
    if (tree?.status === 'running') {
      this.executionTrees.fail(correlationId, message);
      await this.persistRootExecutionTree(correlationId).catch(persistError => {
        this.emit({
          type: 'root.execution_tree.persistence.failed',
          agentId: 'root',
          correlationId,
          data: { error: persistError instanceof Error ? persistError.message : String(persistError) },
        });
      });
    }
    ctx.agent.setRuntimeState('idle');
    this.emit({
      type: 'root.turn.failed',
      agentId: 'root',
      correlationId,
      data: { error: message, failedState },
    });
    ctx.fsm.reset();
    this.emit({
      type: 'root.turn.recovered',
      agentId: 'root',
      correlationId,
      data: { from: failedState, to: ctx.fsm.getState(), runtimeState: 'idle' },
    });
  }

  private async startRootExecutionStep(
    correlationId: string,
    decision: RootExecutionStepDecision,
    dependsOn: string[] = []
  ): Promise<RootExecutionStep> {
    const ctx = this.getContext();
    const step = this.executionTrees.startStep(correlationId, { decision, dependsOn });
    const message = await this.enqueueMessage({
      kind: 'root.step.plan',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'runtime',
      correlationId,
      payload: {
        stepId: step.id,
        index: step.index,
        dependsOn: step.dependsOn,
        decision: step.decision,
      },
      metadata: { agentId: 'root', tags: ['root-execution-step'] },
    });
    await this.processQueuedMessage(message.id);
    await ctx.queue.ack(message.id);
    this.emit({
      type: 'root.step.started',
      agentId: 'root',
      correlationId,
      data: {
        stepId: step.id,
        index: step.index,
        dependsOn: step.dependsOn,
        decision: step.decision,
      },
    });
    await this.persistRootExecutionTree(correlationId);
    return step;
  }

  private async failRootExecutionStep(
    correlationId: string,
    step: RootExecutionStep,
    error: string,
    failTree: boolean
  ): Promise<RootExecutionStep> {
    const ctx = this.getContext();
    const tree = this.executionTrees.get(correlationId);
    if (!tree) throw new Error(`Execution tree not found: ${correlationId}`);
    const messages = await this.getMessages({ correlationId });
    const projected = this.executionActivityProjector.project({
      tree,
      step,
      messages,
      events: this.events,
    });
    const failedAt = Date.now();
    const activities: RootExecutionActivity[] = [...projected, {
      id: `${step.id}.failure`,
      kind: 'control',
      status: 'failed',
      label: 'Root step failed',
      actorId: 'root',
      summary: error.slice(0, 1200),
      startedAt: failedAt,
      completedAt: failedAt,
      data: { error },
    }];
    const events = this.events.filter(event =>
      (event.correlationId === correlationId || event.data?.correlationId === correlationId)
      && event.timestamp >= step.startedAt
      && event.timestamp <= failedAt
    );
    const actorIds = [...new Set(events
      .filter(event => event.type === 'agent.spawned')
      .map(event => event.agentId)
      .filter((id): id is string => Boolean(id)))];
    const teamIds = [...new Set(events
      .filter(event => event.type === 'team.created')
      .map(event => event.agentId)
      .filter((id): id is string => Boolean(id)))];
    const checkpoint = this.executionActivityProjector.checkpoint({
      tree,
      step,
      resultSummary: `Step failed: ${error}`,
      activities,
      actorIds,
      teamIds,
    });
    checkpoint.pending = [`Repair the failed step using this feedback: ${error.slice(0, 1000)}`];
    checkpoint.decisionBasis = `${step.decision.reason} Failure feedback: ${error.slice(0, 800)}`;
    const nodes = this.buildRootExecutionNodes(correlationId, step.index);
    const cache = this.buildExecutionCacheSnapshot({
      tree,
      step,
      nodes,
      actorIds,
      teamIds,
      resultSummary: `Step failed: ${error}`,
      checkpoint,
      activities,
      messages,
      stepStatus: 'failed',
    });
    const failed = this.executionTrees.failStep(correlationId, step.id, error, {
      failTree,
      activities,
      checkpoint,
      cache,
    });
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.cacheExecutionKnowledge !== false) {
      await ctx.memory.writeExecutionCacheSnapshot(
        cache,
        this.workspaceRuntimeConfig?.delegation.rootSteps.maxCachedSteps ?? 200
      );
      this.emit({
        type: 'execution.cache.snapshot.recorded',
        agentId: 'root',
        correlationId,
        data: {
          stepId: failed.id,
          pathId: cache.path.id,
          status: 'failed',
          actorObjects: cache.actors.length,
          feedbackObjects: cache.feedback.length,
        },
      });
      this.emit({
        type: 'execution.feedback.captured',
        agentId: 'root',
        correlationId,
        data: {
          stepId: failed.id,
          pathId: cache.path.id,
          count: cache.feedback.length,
          kinds: [...new Set(cache.feedback.map(item => item.kind))],
        },
      });
    }
    return failed;
  }

  private async completeRootExecutionStep(
    correlationId: string,
    step: RootExecutionStep,
    input: CompleteRootExecutionStepInput
  ): Promise<RootExecutionStep> {
    const ctx = this.getContext();
    const tree = this.executionTrees.get(correlationId);
    if (!tree) throw new Error(`Execution tree not found: ${correlationId}`);
    const messages = await this.getMessages({ correlationId });
    const collectedActivities = input.activities ?? this.executionActivityProjector.project({
      tree,
      step,
      messages,
      events: this.events,
    });
    const checkpoint = input.checkpoint ?? this.executionActivityProjector.checkpoint({
      tree,
      step,
      resultSummary: input.resultSummary,
      activities: collectedActivities,
      actorIds: input.actorIds,
      teamIds: input.teamIds,
    });
    const activities = [...collectedActivities, {
      id: `${step.id}.checkpoint`,
      kind: 'checkpoint' as const,
      status: 'completed' as const,
      label: 'Root state checkpoint',
      actorId: 'root',
      summary: checkpoint.decisionBasis,
      startedAt: checkpoint.createdAt,
      completedAt: checkpoint.createdAt,
      data: {
        completed: checkpoint.completed,
        pending: checkpoint.pending,
        evidence: checkpoint.evidence,
        stateFingerprint: checkpoint.stateFingerprint,
      },
    }];
    const cache = input.cache ?? this.buildExecutionCacheSnapshot({
      tree,
      step,
      nodes: input.nodes ?? tree.nodes,
      actorIds: input.actorIds ?? [],
      teamIds: input.teamIds ?? [],
      resultSummary: input.resultSummary,
      checkpoint,
      activities,
      messages,
    });
    const completed = this.executionTrees.completeStep(correlationId, step.id, {
      ...input,
      activities,
      checkpoint,
      cache,
    });
    const message = await this.enqueueMessage({
      kind: 'root.step.result',
      sessionId: ctx.sessionId,
      from: 'runtime',
      to: 'root',
      correlationId,
      payload: {
        stepId: completed.id,
        index: completed.index,
        actorIds: completed.actorIds,
        teamIds: completed.teamIds,
        resultSummary: completed.resultSummary,
        treeSnapshot: completed.treeSnapshot,
        activities: completed.activities,
        checkpoint: completed.checkpoint,
      },
      metadata: { agentId: 'root', tags: ['root-execution-step'] },
    });
    await this.processQueuedMessage(message.id);
    await ctx.queue.ack(message.id);
    this.emit({
      type: 'root.step.tree.updated',
      agentId: 'root',
      correlationId,
      data: {
        stepId: completed.id,
        index: completed.index,
        nodeCount: completed.treeSnapshot.length,
        activityCount: completed.activities.length,
        checkpointFingerprint: completed.checkpoint?.stateFingerprint,
      },
    });
    this.emit({
      type: 'root.step.completed',
      agentId: 'root',
      correlationId,
      data: {
        stepId: completed.id,
        index: completed.index,
        actorIds: completed.actorIds,
        teamIds: completed.teamIds,
      },
    });
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.cacheExecutionKnowledge !== false) {
      const cachePath = await ctx.memory.writeExecutionCacheSnapshot(
        cache,
        this.workspaceRuntimeConfig?.delegation.rootSteps.maxCachedSteps ?? 200
      );
      this.emit({
        type: 'execution.cache.snapshot.recorded',
        agentId: 'root',
        correlationId,
        data: {
          stepId: completed.id,
          pathId: cache.path.id,
          actorObjects: cache.actors.length,
          feedbackObjects: cache.feedback.length,
          cachePath,
        },
      });
      this.emit({
        type: 'execution.path.updated',
        agentId: 'root',
        correlationId,
        data: {
          stepId: completed.id,
          pathId: cache.path.id,
          status: cache.path.status,
          parentPathIds: cache.path.parentPathIds,
          observedPaths: cache.path.observedPaths,
          invalidPaths: cache.path.invalidPaths,
        },
      });
      if (cache.feedback.length > 0) {
        this.emit({
          type: 'execution.feedback.captured',
          agentId: 'root',
          correlationId,
          data: {
            stepId: completed.id,
            pathId: cache.path.id,
            count: cache.feedback.length,
            kinds: [...new Set(cache.feedback.map(item => item.kind))],
          },
        });
      }
    }
    await this.persistRootExecutionTree(correlationId);
    return completed;
  }

  private buildExecutionCacheSnapshot(input: {
    tree: RootExecutionTreeState;
    step: RootExecutionStep;
    nodes: RootExecutionNodeSnapshot[];
    actorIds: string[];
    teamIds: string[];
    resultSummary?: string;
    checkpoint: RootExecutionCheckpoint;
    activities: RootExecutionActivity[];
    messages: RuntimeMessage[];
    stepStatus?: 'completed' | 'failed';
  }): ExecutionCacheSnapshot {
    const { tree, step } = input;
    const now = Date.now();
    const pathId = `${step.id}.path`;
    const taskFingerprint = this.executionTaskFingerprint(tree.task);
    const stepStart = step.index === 1 ? tree.createdAt : step.startedAt;
    const stepMessages = input.messages.filter(message =>
      message.createdAt >= stepStart && message.createdAt <= now
    );
    const toolCalls = new Map(
      stepMessages
        .filter(message => message.kind === 'tool.call')
        .map(message => [message.id, message])
    );
    const observedPaths = new Set<string>();
    const invalidPaths = new Set<string>();
    const successfulTools = new Set<string>();
    const failedTools = new Set<string>();
    const feedback: ExecutionFeedbackRecord[] = [];
    const cachedToolFrontier: ExecutionCachedToolCall[] = [];
    let mutationObserved = false;
    let verificationObserved = false;
    const addFeedback = (
      kind: ExecutionFeedbackRecord['kind'],
      summary: string,
      options: Pick<ExecutionFeedbackRecord, 'actorId' | 'toolName' | 'path'> = {}
    ): void => {
      const normalizedSummary = summary.replace(/\s+/g, ' ').trim().slice(0, 1200);
      if (!normalizedSummary) return;
      const duplicate = feedback.some(item =>
        item.kind === kind
        && item.toolName === options.toolName
        && item.path === options.path
        && item.summary === normalizedSummary
      );
      if (duplicate) return;
      feedback.push({
        id: `${pathId}.feedback_${String(feedback.length + 1).padStart(3, '0')}`,
        kind,
        correlationId: tree.correlationId,
        stepId: step.id,
        pathId,
        actorId: options.actorId,
        toolName: options.toolName,
        path: options.path,
        summary: normalizedSummary,
        actionable: kind === 'tool_failure'
          || kind === 'actor_failure'
          || kind === 'external_feedback'
          || kind === 'unresolved_gap',
        createdAt: now,
      });
    };

    for (const message of stepMessages.filter(item => item.kind === 'tool.result')) {
      const resultPayload = message.payload as {
        success?: unknown;
        error?: unknown;
        result?: Record<string, unknown>;
      };
      const callMessage = message.parentMessageId ? toolCalls.get(message.parentMessageId) : undefined;
      const callPayload = (callMessage?.payload ?? {}) as {
        toolName?: unknown;
        params?: Record<string, unknown>;
        reason?: unknown;
      };
      const toolName = String(callPayload.toolName ?? callMessage?.to?.replace(/^tool\./, '') ?? 'unknown');
      const params = callPayload.params ?? {};
      const success = resultPayload.success === true;
      const result = resultPayload.result ?? {};
      const candidatePath = [
        typeof result.path === 'string' ? result.path : undefined,
        typeof params.path === 'string' ? params.path : undefined,
        typeof result.root === 'string' ? result.root : undefined,
      ].find((value): value is string => Boolean(value));
      const callRecord: ToolCallRecord = {
        toolName,
        params,
        result,
        success,
        error: typeof resultPayload.error === 'string' ? resultPayload.error : undefined,
        reason: typeof callPayload.reason === 'string' ? callPayload.reason : undefined,
        startedAt: callMessage?.createdAt,
        completedAt: message.createdAt,
      };
      cachedToolFrontier.push(this.compactExecutionCachedToolCall(callRecord));
      const semanticVerificationFailure = success
        && isWorkspaceVerificationCall(callRecord)
        && !isSuccessfulWorkspaceVerificationCall(callRecord);
      if (success && !semanticVerificationFailure) {
        successfulTools.add(toolName);
        if (candidatePath) observedPaths.add(this.normalizeCachedPath(candidatePath));
        if (Array.isArray(result.entries)) {
          const root = typeof result.root === 'string' ? result.root : '';
          for (const entry of result.entries.filter((item): item is string => typeof item === 'string').slice(0, 80)) {
            observedPaths.add(this.normalizeCachedPath(root && root !== '.' ? path.join(root, entry) : entry));
          }
        }
        if (isSuccessfulWorkspaceMutationCall(callRecord)) mutationObserved = true;
        if (isSuccessfulWorkspaceVerificationCall(callRecord)) verificationObserved = true;
      } else if (semanticVerificationFailure) {
        failedTools.add(toolName);
        const verifierDiagnostics = (
          result as {
            verifierDiagnostics?: unknown;
            candidateRollback?: unknown;
          }
        ).verifierDiagnostics;
        const candidateRollback = (
          result as { candidateRollback?: unknown }
        ).candidateRollback;
        addFeedback(
          'tool_failure',
          [
            `${toolName} completed but did not satisfy the verifier.`,
            verifierDiagnostics
              ? JSON.stringify(verifierDiagnostics)
              : JSON.stringify(result).slice(-4_000),
            candidateRollback
              ? `The latest workspace candidate was restored and is a failed path: ${JSON.stringify(candidateRollback)}`
              : '',
          ].join(' '),
          { actorId: message.metadata?.agentId, toolName }
        );
      } else if (!this.isAuthoritativeInvalidPathFailure(
        typeof resultPayload.error === 'string' ? resultPayload.error : undefined
      )) {
        failedTools.add(toolName);
        addFeedback(
          'tool_failure',
          `${toolName} failed without invalidating its path${candidatePath ? ` ${candidatePath}` : ''}: ${String(resultPayload.error ?? 'unknown error')}`,
          { actorId: message.metadata?.agentId, toolName }
        );
      } else {
        failedTools.add(toolName);
        if (candidatePath) invalidPaths.add(this.normalizeCachedPath(candidatePath));
        addFeedback(
          'tool_failure',
          `${toolName} failed${candidatePath ? ` for ${candidatePath}` : ''}: ${String(resultPayload.error ?? 'unknown error')}`,
          { actorId: message.metadata?.agentId, toolName, path: candidatePath }
        );
      }
    }

    if (observedPaths.size > 0) {
      addFeedback(
        'path_observation',
        `Authoritative paths observed by successful runtime tools: ${[...observedPaths].slice(0, 40).join(', ')}`
      );
    }
    if (mutationObserved) {
      addFeedback('workspace_mutation', 'At least one workspace mutation completed successfully in this path.');
    }
    if (verificationObserved) {
      addFeedback('workspace_verification', 'At least one workspace verification command completed successfully in this path.');
    }
    if (step.index === 1 && this.containsExternalExecutionFeedback(tree.task)) {
      addFeedback(
        'external_feedback',
        `External or user-supplied execution feedback that must be reconciled in this run: ${tree.task.slice(0, 2400)}`
      );
    }
    for (const activity of input.activities.filter(item =>
      (item.kind === 'agent' || item.kind === 'team') && item.status === 'failed'
    )) {
      addFeedback(
        'actor_failure',
        `${activity.label}: ${activity.summary ?? String(activity.data?.error ?? 'actor execution failed')}`,
        { actorId: activity.actorId }
      );
    }
    for (const pending of input.checkpoint.pending.filter(item =>
      !item.startsWith('Root must reassess accumulated state')
      && !item.startsWith('Await user clarification')
    )) {
      addFeedback('unresolved_gap', pending);
    }

    const events = this.events.filter(event =>
      (event.correlationId === tree.correlationId || event.data?.correlationId === tree.correlationId)
      && event.timestamp >= stepStart
      && event.timestamp <= now
    );
    for (const event of events.filter(item =>
      item.type === 'root.response.acceptance.unmet'
      || item.type === 'root.response.acceptance.repair.unmet'
    )) {
      const requirements = Array.isArray(event.data?.unmetRequirements)
        ? event.data.unmetRequirements
          .filter((item): item is string => typeof item === 'string')
          .join('; ')
        : '';
      const reason = typeof event.data?.reason === 'string'
        ? event.data.reason
        : '';
      addFeedback(
        event.type.endsWith('repair.unmet') ? 'unresolved_gap' : 'acceptance_feedback',
        [
          event.type.endsWith('repair.unmet')
            ? 'Final-response repair still has unmet acceptance requirements.'
            : 'Final-response acceptance feedback was routed into the repair step.',
          requirements,
          reason,
        ].filter(Boolean).join(' ')
      );
    }
    const generatedIds = new Set([
      ...input.actorIds,
      ...input.teamIds,
      ...events
        .filter(event => event.type === 'agent.spawned' || event.type === 'team.created')
        .map(event => event.agentId)
        .filter((id): id is string => Boolean(id)),
    ]);
    const actors: ExecutionCachedActor[] = input.nodes
      .filter(node => node.id !== tree.rootAgentId && generatedIds.has(node.id))
      .map(node => {
        const runEvent = [...events].reverse().find(event =>
          event.agentId === node.id && (event.type === 'agent.run.started' || event.type === 'team.member.started')
        );
        const spawnEvent = events.find(event =>
          event.agentId === node.id && (event.type === 'agent.spawned' || event.type === 'team.created')
        );
        const task = typeof runEvent?.data?.task === 'string' ? runEvent.data.task : undefined;
        const activeInfo = node.kind === 'agent'
          ? this.getContext().manager.getAgentById(node.id)?.getInfo() ?? this.archivedAgentInfo.get(node.id)
          : undefined;
        const generation = activeInfo?.identity.generation
          ?? this.actorGenerationFromNodes(node, input.nodes);
        return {
          id: `${pathId}.actor.${node.id}`,
          runtimeActorId: node.id,
          kind: node.kind,
          correlationId: tree.correlationId,
          stepId: step.id,
          pathId,
          name: node.name,
          role: node.role,
          parentId: node.parentId,
          teamId: node.teamId,
          generation,
          task,
          taskFingerprint: task ? this.executionTaskFingerprint(task) : undefined,
          definitionFingerprint: typeof spawnEvent?.data?.definitionFingerprint === 'string'
            ? spawnEvent.data.definitionFingerprint
            : node.definitionFingerprint,
          status: node.status === 'waiting' ? 'active' : node.status,
          createdAt: spawnEvent?.timestamp ?? step.startedAt,
          updatedAt: now,
        };
      });
    const parentPathIds = step.dependsOn.map(dependency => {
      const dependencyStep = tree.steps.find(item => item.id === dependency);
      return dependencyStep?.cache?.path.id ?? `${dependency}.path`;
    });
    const resumeState = this.resumedExecutionByCorrelation.get(tree.correlationId);
    if (step.index === 1
      && resumeState
      && !parentPathIds.includes(resumeState.anchorPathId)) {
      parentPathIds.push(resumeState.anchorPathId);
    }
    const inheritedToolFrontier = resumeState?.knowledge.paths
      .flatMap(path => path.toolFrontier ?? [])
      ?? [];
    const boundedToolFrontier = this.boundExecutionToolFrontier([
      ...inheritedToolFrontier,
      ...cachedToolFrontier,
    ]);
    if (inheritedToolFrontier.length > 0) {
      this.emit({
        type: 'execution.tool_frontier.carried_forward',
        agentId: 'root',
        correlationId: tree.correlationId,
        data: {
          stepId: step.id,
          inheritedCalls: inheritedToolFrontier.length,
          currentCalls: cachedToolFrontier.length,
          persistedCalls: boundedToolFrontier.length,
        },
      });
    }
    const hasFailures = failedTools.size > 0 || feedback.some(item => item.kind === 'actor_failure');
    const hasSuccess = successfulTools.size > 0 || input.actorIds.length > 0 || input.teamIds.length > 0;
    const pathStatus = input.stepStatus === 'failed'
      ? (hasSuccess ? 'partial' : 'failed')
      : hasFailures ? (hasSuccess ? 'partial' : 'failed') : 'completed';
    const stepRecord = {
      id: `${step.id}.cache`,
      correlationId: tree.correlationId,
      stepId: step.id,
      index: step.index,
      task: tree.task,
      taskFingerprint,
      pathId,
      dependsOn: [...step.dependsOn],
      action: step.decision.action,
      status: input.stepStatus ?? 'completed',
      actorIds: [...input.actorIds],
      teamIds: [...input.teamIds],
      feedbackIds: feedback.map(item => item.id),
      resultSummary: input.resultSummary?.slice(0, 5000),
      stateFingerprint: input.checkpoint.stateFingerprint,
      createdAt: step.startedAt,
      updatedAt: now,
    };
    return {
      step: stepRecord,
      path: {
        id: pathId,
        correlationId: tree.correlationId,
        stepId: step.id,
        parentPathIds,
        taskFingerprint,
        status: pathStatus,
        actorIds: actors.filter(item => item.kind === 'agent').map(item => item.runtimeActorId),
        teamIds: actors.filter(item => item.kind === 'team').map(item => item.runtimeActorId),
        observedPaths: [...observedPaths].slice(0, 120),
        invalidPaths: [...invalidPaths].slice(0, 80),
        successfulTools: [...successfulTools],
        failedTools: [...failedTools],
        mutationObserved,
        verificationObserved,
        toolFrontier: boundedToolFrontier,
        feedbackIds: feedback.map(item => item.id),
        summary: input.resultSummary?.slice(0, 5000),
        createdAt: step.startedAt,
        updatedAt: now,
      },
      actors,
      feedback,
    };
  }

  private compactExecutionCachedToolCall(
    call: ToolCallRecord
  ): ExecutionCachedToolCall {
    const compactText = (value: unknown, maxChars: number): unknown => {
      if (typeof value !== 'string' || value.length <= maxChars) return value;
      const head = Math.floor(maxChars * 0.6);
      const tail = maxChars - head;
      return `${value.slice(0, head)}\n[runtime_persisted_tool_evidence_compacted]\n${value.slice(-tail)}`;
    };
    const params = Object.fromEntries(
      Object.entries(call.params).map(([key, value]) => [
        key,
        compactText(value, ['content', 'oldText', 'newText'].includes(key) ? 4_000 : 2_000),
      ])
    );
    let result = call.result;
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      const record = result as Record<string, unknown>;
      if (call.toolName === 'fs.read') {
        const contentCompacted = typeof record.content === 'string'
          && record.content.length > 40_000;
        result = {
          ...record,
          content: compactText(record.content, 40_000),
          persistedContentCompacted: contentCompacted || undefined,
        };
      } else if (call.toolName === 'fs.list') {
        result = {
          ...record,
          entries: Array.isArray(record.entries) ? record.entries.slice(0, 160) : record.entries,
        };
      } else if (call.toolName === 'fs.search') {
        result = {
          ...record,
          matches: Array.isArray(record.matches) ? record.matches.slice(0, 80) : record.matches,
        };
      } else if (call.toolName === 'shell.exec') {
        const command = String(call.params.command ?? record.command ?? '');
        const stdout = typeof record.stdout === 'string' ? record.stdout : '';
        const verifierProbe = command.includes('ROY_VERIFIER_PROBE=1')
          || stdout.includes('VERIFIER_PROBE_');
        result = {
          ...record,
          stdout: verifierProbe
            ? this.compactVerifierProbeEvidenceText(stdout, 12_000)
            : compactText(record.stdout, 6_000),
          stderr: compactText(record.stderr, 8_000),
        };
      }
    }
    return {
      toolName: call.toolName,
      params,
      result,
      success: call.success,
      error: typeof call.error === 'string'
        ? String(compactText(call.error, 4_000))
        : undefined,
      reason: call.reason?.slice(0, 1_000),
      startedAt: call.startedAt,
      completedAt: call.completedAt,
    };
  }

  private compactVerifierProbeEvidenceText(
    output: string,
    maxChars: number
  ): string {
    if (!output.trim()) return output;
    const lines = output.split('\n').map(line => line.trim()).filter(Boolean);
    const mismatchTerms = new Set<string>();
    for (const line of lines) {
      if (!line.startsWith('VERIFIER_PROBE_MISMATCHES ')) continue;
      const payloadStart = line.indexOf('{');
      if (payloadStart < 0) continue;
      try {
        const payload = JSON.parse(line.slice(payloadStart)) as {
          mismatch_count?: unknown;
          mismatches?: Array<{ key?: unknown }>;
        };
        if (Number(payload.mismatch_count ?? 0) <= 0) continue;
        for (const mismatch of payload.mismatches ?? []) {
          const key = Array.isArray(mismatch.key) ? mismatch.key : [];
          for (const part of key.slice(0, 2)) {
            const normalized = String(part ?? '').trim().toLowerCase();
            if (normalized.length >= 3) mismatchTerms.add(normalized);
          }
        }
      } catch {
        // A clipped mismatch marker remains useful without relevance terms.
      }
    }
    const seenCandidates = new Set<string>();
    const candidates = lines
      .map((line, position) => {
        const marker = /^VERIFIER_PROBE_([A-Z_]+)\b/.exec(line)?.[1];
        if (!marker) return undefined;
        if (seenCandidates.has(line)) return undefined;
        seenCandidates.add(line);
        let priority = 20;
        let perItemLimit = 1_200;
        if (marker === 'EVIDENCE_VERSION' || marker === 'SCOPE') priority = 240;
        else if (['CALL', 'MISMATCHES', 'RESULT', 'REWARD'].includes(marker)) {
          priority = 220;
          perItemLimit = 1_800;
        } else if (marker === 'SPEC') {
          priority = 190;
          perItemLimit = 2_400;
        } else if (marker === 'TASK_INPUT') {
          priority = 180;
          perItemLimit = 2_400;
        } else if (marker === 'ARTIFACT') {
          priority = 80;
          perItemLimit = 2_400;
        }
        const payloadStart = line.indexOf('{');
        let evidencePath = '';
        if (payloadStart >= 0) {
          try {
            const payload = JSON.parse(line.slice(payloadStart)) as {
              path?: unknown;
              artifact?: unknown;
              mismatch_count?: unknown;
            };
            evidencePath = String(payload.path ?? payload.artifact ?? '').toLowerCase();
            if (marker === 'MISMATCHES'
              && Number(payload.mismatch_count ?? 0) === 0) {
              priority -= 100;
            }
          } catch {
            // Keep the marker even when its compact payload was truncated.
          }
        }
        if (marker === 'TASK_INPUT') {
          const relevantToMismatch = [...mismatchTerms].some(term =>
            evidencePath.includes(term)
          );
          if (relevantToMismatch) {
            priority = 235;
            perItemLimit = 3_200;
          } else if (/(?:manifest|index|dataset)/.test(evidencePath)) {
            priority = 225;
          }
        }
        if (evidencePath.includes('output')) priority += 50;
        if (/(?:qc|summary|report|journal|diagnostic|result|reward|log)/.test(evidencePath)) {
          priority += 60;
        }
        if (marker !== 'TASK_INPUT'
          && /(?:manifest|ocr|token|fixture|expected)/.test(evidencePath)) {
          priority -= 80;
        }
        if (/(?:reconstructed|prediction)/.test(evidencePath)) priority -= 20;
        return { line, position, priority, perItemLimit };
      })
      .filter((item): item is {
        line: string;
        position: number;
        priority: number;
        perItemLimit: number;
      } => Boolean(item))
      .sort((left, right) =>
        right.priority - left.priority || left.position - right.position
      );
    const selected: string[] = [];
    let remaining = Math.max(1, maxChars);
    for (const candidate of candidates) {
      if (remaining <= 0) break;
      const clipped = candidate.line.length <= candidate.perItemLimit
        ? candidate.line
        : `${candidate.line.slice(0, Math.max(0, candidate.perItemLimit - 28))}[diagnostic item compacted]`;
      if (clipped.length > remaining) continue;
      selected.push(clipped);
      remaining -= clipped.length + 1;
    }
    if (selected.length > 0) return selected.join('\n');
    if (output.length <= maxChars) return output;
    const head = Math.floor(maxChars * 0.4);
    return `${output.slice(0, head)}\n[runtime_verifier_probe_compacted]\n${output.slice(-(maxChars - head))}`;
  }

  private buildDeterministicVerifierDiagnosticResult(
    grounding: GroundingRunResult
  ): string {
    const diagnosticCalls = grounding.toolCalls.filter(call =>
      this.isFocusedVerifierDiagnosticCall(call)
    );
    const probeEvidence = diagnosticCalls
      .map(call => {
        const result = call.result as {
          stdout?: unknown;
          stderr?: unknown;
        } | undefined;
        return [result?.stdout, result?.stderr]
          .filter((value): value is string => typeof value === 'string' && value.length > 0)
          .join('\n');
      })
      .filter(Boolean)
      .map(output => this.compactVerifierProbeEvidenceText(output, 3_200))
      .join('\n');
    const observedSources = grounding.toolCalls
      .filter(call => call.toolName === 'fs.read' && call.success)
      .map(call => this.normalizeToolWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      )))
      .filter(sourcePath =>
        sourcePath
        && sourcePath !== '.'
        && !sourcePath.startsWith('.roy/official-verifier/')
      );
    return [
      '[runtime_verifier_diagnostic_evidence_closure]',
      'The runtime executed the focused verifier probe. The next repair member must consume these direct expected-versus-actual observations without another narrative diagnosis pass.',
      `Authoritative implementation snapshots observed: ${[...new Set(observedSources)].join(', ') || 'none'}`,
      probeEvidence || 'The focused probe completed without printable mismatch markers.',
    ].join('\n\n').slice(0, 3_900);
  }

  private boundExecutionToolFrontier(
    calls: ExecutionCachedToolCall[],
    maxCalls = 32,
    maxSerializedChars = 240_000
  ): ExecutionCachedToolCall[] {
    const unique: Array<{ call: ExecutionCachedToolCall; index: number }> = [];
    const seen = new Set<string>();
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const fingerprint = this.fingerprint({
        toolName: call.toolName,
        params: call.params,
        result: call.result,
        success: call.success,
        error: call.error,
      });
      if (seen.has(fingerprint)) continue;
      seen.add(fingerprint);
      unique.push({ call, index });
    }

    const mutatedPaths = new Set(unique
      .filter(item => isSuccessfulWorkspaceMutationCall(item.call))
      .map(item => this.normalizeToolWorkspacePath(String(item.call.params.path ?? '')))
      .filter(path => path && path !== '.'));
    const latestReadIndexByPath = new Map<string, number>();
    for (const item of unique) {
      if (item.call.toolName !== 'fs.read') continue;
      const target = this.normalizeToolWorkspacePath(String(
        (item.call.result as { path?: unknown } | undefined)?.path
          ?? item.call.params.path
          ?? ''
      ));
      if (target && !latestReadIndexByPath.has(target)) {
        latestReadIndexByPath.set(target, item.index);
      }
    }
    const recentThreshold = Math.max(0, calls.length - Math.min(12, Math.max(1, maxCalls)));
    const ranked = unique.map(item => {
      const call = item.call;
      const target = this.normalizeToolWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      ));
      let priority = item.index >= recentThreshold ? 600 : 0;
      if (workspaceCandidateRollbackFromCall(call)?.restored === true) priority += 1_000;
      if (isWorkspaceVerificationCall(call)) priority += 950;
      if (this.isFocusedVerifierDiagnosticCall(call)) priority += 925;
      if (isSuccessfulWorkspaceMutationCall(call)) priority += 900;
      if (call.toolName === 'shell.exec'
        && call.success
        && this.isDependencyInstallCommand(String(call.params.command ?? ''))) {
        priority += 1_500;
      }
      if (call.toolName === 'fs.read' && mutatedPaths.has(target)) priority += 875;
      if (call.toolName === 'fs.read' && target.startsWith('.roy/official-verifier/')) {
        priority += 825;
      }
      if (call.toolName === 'fs.read' && latestReadIndexByPath.get(target) === item.index) {
        priority += 700;
      }
      if (!call.success) priority += 650;
      return {
        ...item,
        priority,
        serializedChars: JSON.stringify(call).length,
      };
    }).sort((left, right) =>
      right.priority - left.priority || right.index - left.index
    );

    const selected: typeof ranked = [];
    let serializedChars = 0;
    for (const item of ranked) {
      if (selected.length >= Math.max(1, maxCalls)) break;
      if (selected.length > 0
        && serializedChars + item.serializedChars > maxSerializedChars) {
        continue;
      }
      selected.push(item);
      serializedChars += item.serializedChars;
    }
    return selected
      .sort((left, right) => left.index - right.index)
      .map(item => item.call);
  }

  private actorGenerationFromNodes(
    node: RootExecutionNodeSnapshot,
    nodes: RootExecutionNodeSnapshot[]
  ): number {
    let generation = 1;
    let parentId = node.parentId;
    const visited = new Set<string>([node.id]);
    while (parentId && parentId !== 'root' && !visited.has(parentId)) {
      visited.add(parentId);
      generation += 1;
      parentId = nodes.find(item => item.id === parentId)?.parentId;
    }
    return generation;
  }

  private executionTaskFingerprint(task: string): string {
    return createHash('sha256')
      .update(task.toLowerCase().replace(/\s+/g, ' ').trim())
      .digest('hex');
  }

  private executionTaskTerms(task: string): Set<string> {
    const stopTerms = new Set([
      'about', 'after', 'again', 'agent', 'benchmark', 'complete', 'continue',
      'directly', 'filesystem', 'horizon', 'implement', 'latest', 'possible',
      'round', 'runtime', 'should', 'state', 'still', 'system', 'terminal',
      'these', 'until', 'using', 'verifier', 'where', 'workspace',
    ]);
    return new Set(
      (task.toLowerCase().match(/[\p{L}\p{N}_./-]{4,}/gu) ?? [])
        .map(term => term.replace(/^[-./]+|[-./]+$/g, ''))
        .filter(term => term.length >= 4 && !stopTerms.has(term))
    );
  }

  private executionTaskSimilarity(left: string, right: string): number {
    if (this.executionTaskFingerprint(left) === this.executionTaskFingerprint(right)) return 1;
    const leftTerms = this.executionTaskTerms(left);
    const rightTerms = this.executionTaskTerms(right);
    const denominator = Math.min(leftTerms.size, rightTerms.size);
    if (denominator === 0) return 0;
    let overlap = 0;
    for (const term of leftTerms) {
      if (rightTerms.has(term)) overlap += 1;
    }
    return overlap / denominator;
  }

  private findExecutionResumeState(
    task: string,
    knowledge: ExecutionKnowledgeCacheState
  ): ExecutionResumeState | undefined {
    const externalContinuation = this.containsExternalExecutionFeedback(task);
    let relevantSteps = knowledge.steps
      .map(step => ({
        step,
        similarity: this.executionTaskSimilarity(task, step.task),
      }))
      .filter(item => item.similarity >= 0.3);
    if (relevantSteps.length === 0 && externalContinuation) {
      relevantSteps = knowledge.steps.map(step => ({ step, similarity: 0 }));
    }
    if (relevantSteps.length === 0) return undefined;

    const grouped = new Map<string, typeof relevantSteps>();
    for (const item of relevantSteps) {
      const values = grouped.get(item.step.correlationId) ?? [];
      values.push(item);
      grouped.set(item.step.correlationId, values);
    }
    const selected = [...grouped.entries()]
      .map(([correlationId, items]) => {
        const stepIds = new Set(items.map(item => item.step.stepId));
        const groupPaths = knowledge.paths.filter(path =>
          path.correlationId === correlationId && stepIds.has(path.stepId)
        );
        const groupFeedback = knowledge.feedback.filter(item =>
          item.correlationId === correlationId && stepIds.has(item.stepId)
        );
        return {
          correlationId,
          items,
          similarity: Math.max(...items.map(item => item.similarity)),
          updatedAt: Math.max(...items.map(item => item.step.updatedAt)),
          resumable: groupPaths.some(path =>
            path.status !== 'completed'
            || (path.mutationObserved && !path.verificationObserved)
          ) || groupFeedback.some(item => item.actionable),
          externalFeedback: groupFeedback.some(item =>
            item.actionable && item.kind === 'external_feedback'
          ),
        };
      })
      .sort((left, right) =>
        externalContinuation
          ? Number(right.externalFeedback) - Number(left.externalFeedback)
            || Number(right.resumable) - Number(left.resumable)
            || right.updatedAt - left.updatedAt
            || right.similarity - left.similarity
          : right.similarity - left.similarity || right.updatedAt - left.updatedAt
      )[0];
    if (!selected) return undefined;

    const selectedStepIds = new Set(selected.items.map(item => item.step.stepId));
    const selectedPaths = knowledge.paths.filter(path =>
      path.correlationId === selected.correlationId && selectedStepIds.has(path.stepId)
    );
    const openSelectedPaths = selectedPaths.filter(path =>
      path.status !== 'completed'
      || (path.mutationObserved && !path.verificationObserved)
    );
    const selectedFeedback = knowledge.feedback.filter(item =>
      item.correlationId === selected.correlationId && selectedStepIds.has(item.stepId)
    );
    const novelExternalContinuation = externalContinuation
      && !selected.items.some(item => item.similarity === 1);
    if (openSelectedPaths.length === 0
      && selectedFeedback.every(item => !item.actionable)
      && !novelExternalContinuation) {
      return undefined;
    }
    const anchorCandidates = openSelectedPaths.length > 0
      ? openSelectedPaths
      : selectedPaths;
    const anchor = anchorCandidates
      .sort((left, right) => right.updatedAt - left.updatedAt)[0];
    if (!anchor) return undefined;

    // Follow the persisted parent path chain across prior Runtime processes.
    // A Harbor continuation gets a new correlation id, but its authoritative
    // reads, verifier results, mutations, and rejected candidates remain part
    // of one causal path and must be available for cache invalidation.
    const pathsById = new Map(knowledge.paths.map(item => [item.id, item]));
    const relatedPathIds = new Set<string>();
    const pendingPathIds = [anchor.id];
    while (pendingPathIds.length > 0 && relatedPathIds.size < 16) {
      const pathId = pendingPathIds.shift()!;
      if (relatedPathIds.has(pathId)) continue;
      const cachedPath = pathsById.get(pathId);
      if (!cachedPath) continue;
      relatedPathIds.add(pathId);
      pendingPathIds.push(...cachedPath.parentPathIds);
    }
    const relatedPaths = knowledge.paths
      .filter(path => relatedPathIds.has(path.id))
      .sort((left, right) => left.updatedAt - right.updatedAt);
    const relatedStepIds = new Set(relatedPaths.map(path => path.stepId));
    const selectedKnowledge: ExecutionKnowledgeCacheState = {
      version: 1,
      updatedAt: knowledge.updatedAt,
      steps: knowledge.steps
        .filter(step => relatedStepIds.has(step.stepId))
        .sort((left, right) => left.updatedAt - right.updatedAt),
      paths: relatedPaths,
      actors: knowledge.actors.filter(actor => relatedStepIds.has(actor.stepId)),
      feedback: knowledge.feedback.filter(item => relatedStepIds.has(item.stepId)),
    };
    const openPaths = selectedKnowledge.paths.filter(path =>
      path.status !== 'completed'
      || (path.mutationObserved && !path.verificationObserved)
    );
    const actionableFeedback = selectedKnowledge.feedback.filter(item => item.actionable);
    return {
      sourceCorrelationId: selected.correlationId,
      anchorPathId: anchor.id,
      knowledge: selectedKnowledge,
      actionableFeedback: actionableFeedback.length,
      openPaths: openPaths.length,
    };
  }

  private compactExecutionResumeBrief(
    knowledge: ExecutionKnowledgeCacheState
  ): Record<string, unknown> {
    const feedback = [...knowledge.feedback]
      .filter(item => item.actionable
        || item.kind === 'workspace_mutation'
        || item.kind === 'workspace_verification')
      .sort((left, right) =>
        Number(right.actionable) - Number(left.actionable)
        || right.createdAt - left.createdAt
      )
      .slice(0, 12);
    const feedbackIds = new Set(feedback.map(item => item.id));
    return {
      updatedAt: knowledge.updatedAt,
      steps: knowledge.steps.slice(-6).map(step => ({
        id: step.stepId,
        index: step.index,
        action: step.action,
        status: step.status,
        pathId: step.pathId,
        resultSummary: step.resultSummary?.slice(0, 1200),
        stateFingerprint: step.stateFingerprint,
      })),
      paths: knowledge.paths.slice(-8).map(path => ({
        id: path.id,
        parentPathIds: path.parentPathIds,
        status: path.status,
        observedPaths: path.observedPaths.slice(0, 40),
        invalidPaths: path.invalidPaths.slice(0, 20),
        successfulTools: path.successfulTools,
        failedTools: path.failedTools,
        mutationObserved: path.mutationObserved,
        verificationObserved: path.verificationObserved,
        feedbackIds: path.feedbackIds.filter(id => feedbackIds.has(id)),
      })),
      feedback: feedback.map(item => ({
        kind: item.kind,
        toolName: item.toolName,
        path: item.path,
        actionable: item.actionable,
        summary: item.summary.slice(0, 1200),
        pathId: item.pathId,
      })),
    };
  }

  private formatExecutionResumeBrief(knowledge: ExecutionKnowledgeCacheState): string {
    if (knowledge.steps.length === 0) {
      return JSON.stringify({
        steps: [],
        attention: 'No prior execution ledger is available.',
      });
    }
    return JSON.stringify(this.compactExecutionResumeBrief(knowledge), null, 2);
  }

  private normalizeCachedPath(value: string): string {
    let normalized = value.trim().replaceAll('\\', '/').replace(/^\.\//, '').replace(/\/+$/, '');
    const workspaceRoot = this.workspaceRoot.replaceAll('\\', '/').replace(/\/+$/, '');
    if (normalized === workspaceRoot) normalized = '.';
    else if (normalized.startsWith(`${workspaceRoot}/`)) normalized = normalized.slice(workspaceRoot.length + 1);
    return normalized || '.';
  }

  private containsExternalExecutionFeedback(value: string): boolean {
    return /(?:<feedback>|<official_verifier_feedback>|(?:verifier|grader|reviewer|test)\s+(?:feedback|failure|reported)|previous\s+(?:attempt|round)[\s\S]{0,120}(?:failed|failure|error)|(?:验证器|评分器|评审|测试)[\s\S]{0,40}(?:反馈|失败|报错)|上一轮[\s\S]{0,100}(?:失败|报错|反馈))/i.test(value);
  }

  private async persistRootExecutionTree(correlationId: string): Promise<void> {
    const tree = this.executionTrees.get(correlationId);
    if (!tree || !this.memory) return;
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.persistEveryStep === false && tree.status === 'running') return;
    await this.memory.writeExecutionTree(tree);
  }

  private emitDelegationPlan(
    correlationId: string,
    plans: DelegationAgentPlan[],
    reason: string,
    decisionMetadata: Record<string, unknown>,
    stepId: string
  ): void {
    this.emit({
      type: 'delegation.plan.created',
      agentId: 'root',
      correlationId,
      data: { correlationId, stepId, count: plans.length, reason, agents: plans, ...decisionMetadata },
    });
    for (const plan of plans) {
      this.emit({
        type: 'delegation.subagent.selected',
        agentId: 'root',
        correlationId,
        data: {
          correlationId,
          stepId,
          archetype: plan.archetype,
          name: plan.name,
          tomLevel: plan.tomLevel,
          budgetTokens: plan.budgetTokens,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
          tomProfile: plan.tomProfile,
        },
      });
      this.emit({
        type: 'delegation.subagent.task_assigned',
        agentId: 'root',
        correlationId,
        data: {
          correlationId,
          stepId,
          archetype: plan.archetype,
          name: plan.name,
          task: plan.task,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
        },
      });
    }
  }

  private async executeRootDelegationRound(
    userTask: string,
    decision: Extract<DelegationDecision, { action: 'spawn_subagents' }>,
    correlationId: string
  ): Promise<RootDelegationRoundResult> {
    const plans = decision.agents;
    if (this.workspaceRuntimeConfig?.evolution.enabled
      && this.workspaceRuntimeConfig.evolution.mode === 'auto') {
      const evolution = await this.runEvolution({
        task: userTask,
        parentId: 'root',
        correlationId,
        profile: this.workspaceRuntimeConfig.evolution.profile,
        seedAgents: plans.map(plan => ({
          archetype: plan.archetype,
          name: plan.name,
          role: plan.existenceReason ?? plan.archetype,
          task: plan.task,
          tools: plan.tools,
          skills: plan.skills,
          budgetTokens: plan.budgetTokens,
          tomLevel: plan.tomLevel,
          perspective: plan.tomProfile?.perspective,
          groundingRequired: plan.archetype === 'researcher' || plan.archetype === 'tester',
        })),
      });
      return { subagents: [], teams: [], evolution };
    }

    const shouldCreateTeam = plans.length > 1
      && (decision.coordination === 'team'
        || (decision.coordination === undefined && this.workspaceRuntimeConfig?.teams.createForMultipleAgents !== false));
    if (shouldCreateTeam) {
      const teamPlan = decision.team;
      const team = await this.spawnTeam({
        parentAgentId: 'root',
        name: teamPlan?.name ?? this.deriveTeamName(plans),
        description: teamPlan?.description ?? userTask,
        task: teamPlan?.task ?? userTask,
        synthesisPolicy: teamPlan?.synthesisPolicy,
        tomLevel: teamPlan?.tomLevel,
        executionPolicy: teamPlan?.executionPolicy,
        members: plans.map((plan, index) => ({ ...plan, lead: index === 0 })),
        tomAnalysis: this.tomAnalyses.get(correlationId),
        correlationId,
        lifecycleOrigin: 'automatic_delegation',
      });
      const teamResult = await this.runTeam(team.identity.id, userTask, {
        correlationId,
        memberRecursiveDelegation: teamPlan?.memberDelegationPolicy !== 'deny',
      });
      return { subagents: [...teamResult.memberExecutions], teams: [teamResult] };
    }

    const subagents: RootMediatedSpawnResult[] = [];
    for (const plan of plans) {
      try {
        subagents.push(await this.handleSpawnCommand({
          archetype: plan.archetype,
          task: plan.task,
          parentId: 'root',
          name: plan.name,
          customRole: plan.role,
          customStyle: plan.style,
          tools: plan.tools,
          skills: plan.skills,
          tomLevel: plan.tomLevel,
          tomProfile: plan.tomProfile,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
          systemPrompt: plan.systemPrompt,
          budgetTokens: plan.budgetTokens,
          memoryScope: plan.memoryScope,
          correlationId,
          source: 'root',
          requireRootSynthesis: false,
          showSubagentOutput: false,
          disableRecursiveDelegation: this.getBudgetState().mode === 'limited',
        }));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!message.startsWith('Spawn rejected:')) throw error;
        this.emit({
          type: 'delegation.child.skipped',
          agentId: 'root',
          correlationId,
          data: {
            archetype: plan.archetype,
            name: plan.name,
            task: plan.task,
            reason: message.slice('Spawn rejected:'.length).trim(),
          },
        });
      }
    }
    return { subagents, teams: [] };
  }

  private collectDelegationRoundActorIds(round: RootDelegationRoundResult, startedAt: number): string[] {
    const correlationIds = new Set([
      ...round.subagents.map(item => item.correlationId),
      ...round.teams.map(item => item.correlationId),
    ]);
    return [...new Set([
      ...round.subagents.map(item => item.agent.identity.id),
      ...round.teams.flatMap(team => team.team.memberAgentIds),
      ...round.teams.flatMap(team => team.memberOutcomes
        .map(outcome => outcome.agentId)
        .filter((agentId): agentId is string => Boolean(agentId))),
      ...this.events
        .filter(event =>
          event.type === 'agent.spawned'
          && event.timestamp >= startedAt
          && event.correlationId
          && correlationIds.has(event.correlationId)
        )
        .map(event => event.agentId)
        .filter((agentId): agentId is string => Boolean(agentId)),
    ])];
  }

  private collectDelegationRoundTeamIds(round: RootDelegationRoundResult, startedAt: number): string[] {
    const correlationIds = new Set([
      ...round.subagents.map(item => item.correlationId),
      ...round.teams.map(item => item.correlationId),
    ]);
    return [...new Set([
      ...round.teams.map(item => item.team.identity.id),
      ...this.events
        .filter(event =>
          event.type === 'team.created'
          && event.timestamp >= startedAt
          && event.correlationId
          && correlationIds.has(event.correlationId)
        )
        .map(event => event.agentId)
        .filter((teamId): teamId is string => Boolean(teamId)),
    ])];
  }

  private delegationRoundHasWorkspaceMutation(round: RootDelegationRoundResult): boolean {
    const calls = [
      ...round.subagents.flatMap(item => item.subagentResult.toolCalls),
      ...round.teams.flatMap(team => team.members.flatMap(member => member.toolCalls)),
      ...round.teams.flatMap(team =>
        this.teamToolEvidenceCache.get(team.team.identity.id) ?? []
      ),
    ];
    return this.hasSuccessfulWorkspaceMutation(calls);
  }

  private shouldHandoffToRootExecution(input: {
    requiresWorkspaceMutation: boolean;
    requiresLongHorizon: boolean;
    roundMutationApplied: boolean;
    delegationRounds: number;
    maxRounds: number;
    exploratoryDelegationLimit: number;
  }): boolean {
    if (!input.requiresWorkspaceMutation) return false;
    if (input.roundMutationApplied) return true;
    return input.delegationRounds >= Math.min(
      input.maxRounds,
      input.exploratoryDelegationLimit
    );
  }

  private async decideRootContinuation(
    userTask: string,
    correlationId: string,
    steps: RootExecutionStep[],
    subagents: RootMediatedSpawnResult[],
    teams: TeamRunResult[]
  ): Promise<RootContinuationDecision> {
    const ctx = this.getContext();
    if (!ctx.llm) return { action: 'finalize', reason: 'No LLM is configured for root step reassessment.' };

    const evidenceFollowUp = this.buildRequiredEvidenceFollowUp(userTask, subagents);
    if (evidenceFollowUp) {
      this.emit({
        type: 'root.step.evidence_gap',
        agentId: 'root',
        correlationId,
        data: {
          target: evidenceFollowUp.target,
          requiredTool: 'fs.read',
          reason: evidenceFollowUp.reason,
        },
      });
      return {
        action: 'delegate_more',
        reason: evidenceFollowUp.reason,
        agents: [evidenceFollowUp.plan],
      };
    }

    const completedTasks = subagents.map(item => ({
      id: item.agent.identity.id,
      archetype: item.node.identity.archetype,
      task: item.subagentResult.agent.lastTask,
      grounded: item.subagentResult.grounded,
    }));
    const executionKnowledge = await ctx.memory.readExecutionKnowledge(
      userTask,
      this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
    );
    if (executionKnowledge.steps.length > 0) {
      this.emit({
        type: 'execution.cache.hit',
        agentId: 'root',
        correlationId,
        data: {
          scope: 'root.continuation',
          steps: executionKnowledge.steps.length,
          paths: executionKnowledge.paths.length,
          actors: executionKnowledge.actors.length,
          feedback: executionKnowledge.feedback.length,
        },
      });
    }
    const latestResults = teams.length > 0
      ? teams.slice(-2).map(item => ({
        kind: 'team',
        id: item.team.identity.id,
        name: item.team.identity.name,
        result: item.result.slice(-2400),
      }))
      : subagents.slice(-3).map(item => ({
        kind: 'agent',
        id: item.agent.identity.id,
        name: item.agent.identity.name,
        grounded: item.subagentResult.grounded,
        warnings: item.subagentResult.warnings.slice(-4),
        result: item.subagentResult.result.slice(-1800),
      }));
    const budget = this.getBudgetState();
    try {
      const raw = await this.completeJSONAsAgent<RootContinuationDecision>(ctx.agent, [
        {
          role: 'system',
          content: `You are Roy's dynamic root-step controller.
Reassess the original task after completed execution steps. The execution tree may grow only when the current evidence exposes a concrete unresolved gap.
Return exactly one strict JSON object and no surrounding prose.
Choose exactly one action:
{"action":"finalize","reason":"..."}
{"action":"ask_clarification","reason":"...","question":"..."}
{"action":"delegate_more","reason":"...","coordination":"independent","continuationPolicy":"reassess","agents":[{"archetype":"custom","name":"task-specific name","role":"task-specific responsibility","task":"...","tools":[],"skills":[],"tomLevel":0,"existenceReason":"..."}]}
{"action":"delegate_more","reason":"...","coordination":"team","continuationPolicy":"reassess","team":{"name":"task-specific team","description":"...","synthesisPolicy":"..."},"agents":[...]}
Do not repeat an existing agent task, cached failed path, or equivalent failed tool call unless the new task states the changed hypothesis. Generate the next actor structure from unresolved state rather than a fixed role list. Route each actionable feedback item to an actor that has the tools and responsibility to close it. Reuse a successful cached actor/team definition only as a template for a fresh runtime actor. Delegate only work that depends on prior-step results. Prefer finalize when evidence is sufficient. Use at most 1-3 agents.`,
        },
        {
          role: 'user',
          content: [
            `<original_task>${userTask}</original_task>`,
            `<acceptance_checklist>${this.buildTaskAcceptanceChecklist(userTask)}</acceptance_checklist>`,
            `<completed_steps>${JSON.stringify(steps.slice(-4).map(step => ({
              id: step.id,
              decision: step.decision,
              resultSummary: step.resultSummary?.slice(-1200),
            })), null, 2)}</completed_steps>`,
            `<execution_knowledge>${this.formatExecutionResumeBrief(executionKnowledge)}</execution_knowledge>`,
            `<latest_results>${JSON.stringify(latestResults, null, 2)}</latest_results>`,
            `<budget>${JSON.stringify({
              mode: budget.mode,
              limitTokens: budget.limitTokens,
              usedTokens: budget.usedTokens,
              remainingTokens: budget.remainingTokens,
            }, null, 2)}</budget>`,
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 1400 }, 'root.dynamic_step_decision', correlationId);
      return this.normalizeRootContinuation(raw, userTask, completedTasks.map(item => `${item.archetype}:${item.task ?? ''}`));
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'root.step.decision.fallback',
        agentId: 'root',
        correlationId,
        data: { reason: 'continuation_decision_failed', error: error instanceof Error ? error.message : String(error) },
      });
      return { action: 'finalize', reason: 'Root step reassessment failed, so Roy will synthesize completed results.' };
    }
  }

  private normalizeRootContinuation(
    value: unknown,
    userTask: string,
    completedTasks: string[]
  ): RootContinuationDecision {
    const input = value as Partial<RootContinuationDecision>;
    if (input.action === 'ask_clarification') {
      const reason = typeof input.reason === 'string'
        ? input.reason
        : 'The completed step exposed missing user input.';
      const question = typeof input.question === 'string' && input.question.trim()
        ? input.question.trim()
        : 'What additional constraint should Roy use before continuing?';
      if (this.taskRequiresWorkspaceMutation(userTask)
        && this.isToolPermissionClarification(reason, question)) {
        return {
          action: 'finalize',
          reason: 'Runtime tool policy, not conversational permission, governs workspace actions; proceed to the root execution phase.',
        };
      }
      return {
        action: 'ask_clarification',
        reason,
        question,
      };
    }
    if (input.action === 'delegate_more' && Array.isArray((input as { agents?: unknown[] }).agents)) {
      const seen = new Set(completedTasks.map(item => item.toLowerCase()));
      const webTask = this.taskNeedsWebAccess(userTask);
      const agents = (input as { agents: Array<Partial<DelegationAgentPlan>> }).agents
        .filter(item => this.isValidArchetype(String(item.archetype)))
        .map((item): DelegationAgentPlan => {
          const normalized = this.normalizeDelegationAgentPlan(item, userTask);
          if (!webTask) return normalized;
          return {
            ...normalized,
            task: `${normalized.task}\nThis is a continuation of the original public-web task. Use web.search/web.fetch rather than treating product or domain names as local file paths.`,
            tools: Array.from(new Set([
              ...(normalized.tools ?? []).filter(tool => !tool.startsWith('fs.')),
              'web.search',
              'web.fetch',
            ])),
            skills: Array.from(new Set([...(normalized.skills ?? []), 'use_tool_when_needed'])),
          };
        })
        .filter(item => !seen.has(`${item.archetype}:${item.task}`.toLowerCase()))
        .slice(0, 3);
      if (agents.length > 0) {
        return {
          action: 'delegate_more',
          reason: typeof input.reason === 'string' ? input.reason : 'A prior step exposed a concrete unresolved gap.',
          agents,
          coordination: this.normalizeCoordination(input, agents.length),
          team: this.normalizeDelegationTeamPlan(input, userTask, agents.length),
          continuationPolicy: this.normalizeContinuationPolicy(input),
        };
      }
    }
    return {
      action: 'finalize',
      reason: input.action === 'finalize' && typeof input.reason === 'string'
        ? input.reason
        : 'Completed steps provide enough information for Roy to synthesize the result.',
    };
  }

  private buildRequiredEvidenceFollowUp(
    userTask: string,
    subagents: RootMediatedSpawnResult[]
  ): { target: string; reason: string; plan: DelegationAgentPlan } | undefined {
    if (this.taskNeedsWebAccess(userTask)
      || this.taskRequiresWorkspaceMutation(userTask)) {
      return undefined;
    }
    const targets = [...new Set(
      [...userTask.matchAll(/(?:^|\s|[`'"])([./]?[a-zA-Z0-9_-]+\.(?:json|ya?ml|toml|md|ts|tsx|js|jsx|mjs|cjs))(?=\s|[,.!?;:`'"]|$)/g)]
        .map(match => match[1])
    )];
    if (targets.length === 0) return undefined;

    for (const target of targets) {
      const normalizedTarget = target.replace(/^\.\//, '').toLowerCase();
      const hasReadEvidence = subagents.some(item => item.subagentResult.toolCalls.some(call => {
        if (call.toolName !== 'fs.read' || !call.success) return false;
        const callPath = typeof call.params.path === 'string' ? call.params.path.toLowerCase() : '';
        return callPath === normalizedTarget || callPath.endsWith(`/${normalizedTarget}`);
      }));
      if (hasReadEvidence) continue;

      const verifierAlreadyAttempted = subagents.some(item => {
        const archetype = item.node.identity.archetype;
        const task = item.subagentResult.agent.lastTask?.toLowerCase() ?? '';
        return (archetype === 'tester' || archetype === 'critic') && task.includes(normalizedTarget);
      });
      if (verifierAlreadyAttempted) continue;

      const reason = `The task requires file-content evidence for ${target}, but completed steps contain no successful fs.read call for that target.`;
      return {
        target,
        reason,
        plan: {
          archetype: 'tester',
          name: 'EvidenceVerifier-1',
          task: `Read ${target} with fs.read, verify the claims made by prior agents against its actual content, and report concrete evidence and remaining limitations.`,
          tools: ['fs.read'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: `Close the unresolved file-content evidence gap for ${target}.`,
        },
      };
    }
    return undefined;
  }

  private buildRootExecutionNodes(correlationId: string, stepIndex: number): RootExecutionNodeSnapshot[] {
    const budget = this.getBudgetState();
    const events = this.events.filter(event => event.correlationId === correlationId || event.data?.correlationId === correlationId);
    const nodes = new Map<string, RootExecutionNodeSnapshot>();
    for (const event of events) {
      if (event.type === 'agent.spawned' && event.agentId) {
        const info = this.getContext().manager.getAgentById(event.agentId)?.getInfo()
          ?? this.archivedAgentInfo.get(event.agentId);
        const taskEvent = [...events].reverse().find(item =>
          item.agentId === event.agentId && item.type === 'agent.run.started'
        );
        const task = typeof taskEvent?.data?.task === 'string' ? taskEvent.data.task : undefined;
        nodes.set(event.agentId, {
          id: event.agentId,
          kind: 'agent',
          name: String(event.data?.name ?? event.agentId),
          role: String(event.data?.archetype ?? 'subagent'),
          parentId: typeof event.data?.parentId === 'string' ? event.data.parentId : 'root',
          generation: info?.identity.generation,
          definitionFingerprint: typeof event.data?.definitionFingerprint === 'string'
            ? event.data.definitionFingerprint
            : undefined,
          taskFingerprint: task ? this.executionTaskFingerprint(task) : undefined,
          status: 'active',
          createdAtStep: stepIndex,
          updatedAtStep: stepIndex,
          tokenUsage: budget.perAgent[event.agentId]?.totalTokens,
        });
      } else if (event.type === 'team.created' && event.agentId) {
        nodes.set(event.agentId, {
          id: event.agentId,
          kind: 'team',
          name: String(event.data?.name ?? event.agentId),
          role: 'subteam',
          parentId: typeof event.data?.parentAgentId === 'string' ? event.data.parentAgentId : 'root',
          status: 'active',
          createdAtStep: stepIndex,
          updatedAtStep: stepIndex,
          tokenUsage: budget.perTeam[event.agentId]?.totalTokens,
        });
      }
    }
    for (const node of nodes.values()) {
      const teamMembership = events.find(event => event.type === 'team.member.added' && event.agentId === node.id);
      if (node.kind === 'agent' && typeof teamMembership?.data?.teamId === 'string') {
        node.teamId = teamMembership.data.teamId;
        node.parentId = teamMembership.data.teamId;
      }
      const lifecycle = this.lifecycle.get(node.id);
      const failed = events.some(event => event.agentId === node.id && (event.type === 'agent.run.failed' || event.type === 'team.failed'));
      const completed = events.some(event => event.agentId === node.id && (event.type === 'agent.run.completed' || event.type === 'team.completed'));
      node.status = failed
        ? 'failed'
        : lifecycle?.status === 'released' || lifecycle?.status === 'persisted'
          ? 'released'
          : completed ? 'done' : 'active';
      node.updatedAtStep = stepIndex;
      node.tokenUsage = node.kind === 'agent'
        ? budget.perAgent[node.id]?.totalTokens ?? node.tokenUsage
        : budget.perTeam[node.id]?.totalTokens ?? node.tokenUsage;
    }
    return [...nodes.values()];
  }

  private summarizeDelegationRound(round: RootDelegationRoundResult): string {
    if (round.evolution) {
      return `Evolution ${round.evolution.id} selected ${round.evolution.selected?.genome.id ?? 'no genome'} with score ${round.evolution.selectedEvaluation?.score ?? 0}.`;
    }
    const teamSummary = round.teams.map(item => `${item.team.identity.name}: ${item.result.slice(0, 1200)}`);
    const agentSummary = round.subagents.map(item => [
      `${item.agent.identity.name} (${item.node.identity.archetype})`,
      `grounded=${item.subagentResult.grounded}`,
      `warnings=${item.subagentResult.warnings.join('; ') || 'none'}`,
      item.subagentResult.result.slice(0, 1200),
    ].join(' | '));
    return [...teamSummary, ...agentSummary].join('\n') || 'Delegation round completed without a visible result.';
  }

  private async decideDelegation(userInput: string, correlationId: string): Promise<DelegationDecision> {
    const ctx = this.getContext();
    const fallback = this.fallbackDelegationDecision(userInput);

    if (!ctx.llm) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: 'root',
        data: { correlationId, reason: 'llm_not_configured' },
      });
      return this.constrainDelegationToExecutablePlans(
        'root',
        this.applyBudgetConstraints(fallback),
        correlationId
      );
    }

    try {
      this.emit({ type: 'delegation.assess.started', agentId: 'root', data: { correlationId } });
      const rootContext = await ctx.memory.loadRootContext();
      const executionKnowledge = await ctx.memory.readExecutionKnowledge(
        userInput,
        this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
      );
      if (executionKnowledge.steps.length > 0) {
        this.emit({
          type: 'execution.cache.hit',
          agentId: 'root',
          correlationId,
          data: {
            scope: 'root.delegation',
            steps: executionKnowledge.steps.length,
            paths: executionKnowledge.paths.length,
            actors: executionKnowledge.actors.length,
            feedback: executionKnowledge.feedback.length,
          },
        });
      }
      const sessionWindow = await this.contextWindowManager?.build({
        sessionId: ctx.sessionId,
        agentId: 'root',
        agentKey: 'roy',
        role: 'root',
        task: userInput,
        memoryScope: this.getDefaultMemoryScope('root'),
      });
      const decision = await this.completeJSONAsAgent<DelegationDecision>(ctx.agent, [
        {
          role: 'system',
          content: `You are Roy's root delegation controller.
Decide whether the user request should be solved directly by Roy, clarified, delegated to independent agents, or delegated to an autonomously designed subteam.
Reason in terms of cognitive gaps: missing evidence, missing perspective, failure-mode uncertainty, implementation capability, verification, or belief reconciliation.
Use delegation only when the task benefits from grounded inspection, critique, planning, coding, testing, or summarization.
Do not spawn more than 3 subagents. Prefer 1-2 unless the task clearly needs more.
If later work depends on an earlier result, create only the immediately executable first-step agents. Roy will reassess after that step and grow the tree if needed.
For a long-horizon, multi-step, iterative, or recursive task, prefer a task-specific team of 2-3 complementary members, set continuationPolicy to reassess, and allow member delegation. A single root step may contain many descendant derivations; every member and descendant still needs a distinct gap, bounded task, and synthesis path.
When the task requests workspace changes, include an executor whose concrete task is to apply changes with fs.synthesize, fs.write, fs.replace, or shell.exec and verify them. Do not assign every agent to analysis or proposal writing.
Ask for clarification when the user request is too ambiguous to assign a concrete task safely.
Design agent names, roles, tasks, tools, and skills from the current task. Do not copy a fixed team template. Use a team only when members require an explicit coordination and synthesis boundary.
Return strict JSON matching one of:
{"action":"solve_directly","reason":"..."}
{"action":"ask_clarification","reason":"...","question":"..."}
{"action":"spawn_subagents","reason":"...","coordination":"independent","continuationPolicy":"reassess","agents":[{"archetype":"custom","name":"EvidenceMapper-1","role":"task-specific evidence mapper","task":"...","tools":["fs.read"],"skills":["use_tool_when_needed"],"tomLevel":0,"existenceReason":"which cognitive gap this agent fills"}]}
{"action":"spawn_subagents","reason":"...","coordination":"team","continuationPolicy":"finalize_after_round","team":{"name":"task-specific team name","description":"why this team exists","task":"shared objective","synthesisPolicy":"how member evidence must be combined","memberDelegationPolicy":"deny","executionPolicy":{"mode":"parallel","failureMode":"best_effort","maxConcurrency":2,"minimumSuccessfulMembers":1}},"agents":[{"archetype":"custom","name":"task-specific name","role":"task-specific responsibility","task":"non-overlapping member task","tools":[],"skills":[],"tomLevel":1,"existenceReason":"which cognitive gap this member fills"}]}
Set continuationPolicy to finalize_after_round when the user explicitly requires one delegation round, one team, or finalization immediately after team synthesis. Otherwise use reassess.
Set team.memberDelegationPolicy to deny only for a minimal or explicitly single-round team. For long-horizon work, use allow so members can form their own subteam when newly observed evidence warrants it.
Allowed archetypes: researcher, critic, planner, coder, summarizer, tester, custom.`,
        },
        {
          role: 'user',
          content: [
            `<user_task>${userInput}</user_task>`,
            `<acceptance_checklist>${this.buildTaskAcceptanceChecklist(userInput)}</acceptance_checklist>`,
            `<execution_knowledge>${this.formatExecutionKnowledge(executionKnowledge)}</execution_knowledge>`,
            `<memory_context>${this.formatPublicContext(rootContext).slice(0, 6000)}</memory_context>`,
            `<recent_session_context>${sessionWindow?.sessionContext || 'No prior turns in this session.'}</recent_session_context>`,
            `<budget_state>${JSON.stringify(this.getBudgetState(), null, 2)}</budget_state>`,
            `<runtime_capabilities>${JSON.stringify({
              tools: toolRegistry.list().map(tool => tool.name),
              skills: skillRegistry.list().map(skill => skill.name),
              webEnabled: this.workspaceRuntimeConfig?.tools.web.enabled !== false,
              automaticallyAuthorizedToolsByArchetype: Object.fromEntries(
                (['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'] as SubAgentArchetype[])
                  .map(archetype => [
                    archetype,
                    this.getAutomaticallyApprovedToolBindings(archetype, userInput)
                      .map(binding => binding.name),
                  ])
              ),
            }, null, 2)}</runtime_capabilities>`,
            '<runtime_policy>Subagents and teams must be runtime actors with identity, state, budget, messages, events, and lifecycle. Propose only registered tools and skills shown above; Runtime will intersect every request with parent-approved capabilities. Do not create a tool-grounded evidence collector when its archetype has no automatically authorized route for the required evidence. Pure semantic reasoning, critique, or synthesis may use a tool-free actor when the task does not require external or workspace grounding. Each agent must fill a distinct cognitive gap, receive a concrete non-overlapping task, and expose why it exists. Treat successful cached paths as authoritative, explicitly route actionable failure feedback to the actor that can repair it, and do not repeat an equivalent failed path. Reuse cached definitions only as structure for fresh runtime actors. If budget is limited, reduce the structure or solve directly.</runtime_policy>',
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 1800 }, 'root.delegation_decision', correlationId);
      let normalized = this.applyBudgetConstraints(this.normalizeDelegationDecision(decision, userInput));
      if (normalized.action === 'solve_directly') {
        normalized = await this.auditComplexDirectDecision(
          userInput,
          normalized,
          correlationId
        );
      }
      normalized = this.constrainDelegationToExecutablePlans('root', normalized, correlationId);
      this.emit({
        type: 'delegation.assess.completed',
        agentId: 'root',
        data: { correlationId, action: normalized.action, source: 'llm' },
      });
      return normalized;
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: 'root',
        data: {
          correlationId,
          reason: 'llm_decision_failed',
          error: error instanceof Error ? error.message : String(error),
        },
      });
      return this.constrainDelegationToExecutablePlans(
        'root',
        this.applyBudgetConstraints(fallback),
        correlationId
      );
    }
  }

  private async decideAgentDelegation(agent: AgentInfo, task: string, correlationId: string): Promise<DelegationDecision> {
    if (agent.identity.id === 'root') {
      return { action: 'solve_directly', reason: 'Root delegation is handled by the root turn controller.' };
    }

    const policy = this.getAgentPolicy(agent.identity.id);
    if (!policy?.spawnPolicy.canSpawn) {
      return { action: 'solve_directly', reason: 'Agent spawn policy does not allow child delegation.' };
    }
    if (policy.allowedChildren <= policy.currentChildren) {
      return { action: 'solve_directly', reason: 'Agent has no remaining child slots for this parent.' };
    }
    if (policy.depth >= policy.spawnPolicy.maxDepth) {
      return { action: 'solve_directly', reason: 'Agent is at the maximum recursive delegation depth.' };
    }
    if (this.getRemainingTotalAgentsForTurn(agent.identity.id, correlationId) <= 0) {
      return { action: 'solve_directly', reason: 'No agent slots remain for this turn.' };
    }

    const fallback = this.fallbackAgentDelegationDecision(agent, task);
    const ctx = this.getContext();
    if (!ctx.llm) {
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: agent.identity.id,
        data: { correlationId, reason: 'llm_not_configured_agent_delegation' },
      });
      const constrained = this.constrainDelegationToExecutablePlans(
        agent.identity.id,
        this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, fallback),
        correlationId
      );
      return this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
    }

    try {
      const executionKnowledge = await ctx.memory.readExecutionKnowledge(
        task,
        this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
      );
      if (executionKnowledge.steps.length > 0) {
        this.emit({
          type: 'execution.cache.hit',
          agentId: agent.identity.id,
          correlationId,
          data: {
            scope: 'agent.delegation',
            steps: executionKnowledge.steps.length,
            paths: executionKnowledge.paths.length,
            actors: executionKnowledge.actors.length,
            feedback: executionKnowledge.feedback.length,
          },
        });
      }
      this.emit({ type: 'delegation.assess.started', agentId: agent.identity.id, data: { correlationId, scope: 'agent' } });
      const decision = await this.completeJSONAsAgent<DelegationDecision>(
        ctx.manager.getAgentById(agent.identity.id)!,
        [
        {
          role: 'system',
          content: `You are ${agent.identity.name}'s delegation controller.
Decide whether this non-root agent should solve directly or delegate to 1-3 direct child agents. Delegate only to close an explicit evidence, perspective, risk, planning, implementation, verification, or synthesis gap in the parent agent's current model.
Only delegate when a child with a different specialty materially improves the result.
For two or more coupled gaps, prefer a team with a clear synthesis boundary. Team members may recursively create their own team when a newly discovered gap cannot be closed locally and depth/budget policy permits.
When the task requests workspace changes and this agent has fs.synthesize, fs.write, fs.replace, or shell.exec, it must execute the change itself or delegate a concrete implementation task. Do not return a proposed patch without applying it.
Treat successful cached paths as authoritative. Do not repeat a cached invalid path or equivalent failed call without a changed hypothesis. Route actionable cached feedback to a child with the capabilities to close it. Cached actors and teams are reusable definitions, not live instances.
Generate task-specific child definitions rather than selecting a fixed role template. Return strict JSON:
{"action":"solve_directly","reason":"..."}
{"action":"spawn_subagents","reason":"...","coordination":"independent","continuationPolicy":"reassess","agents":[{"archetype":"custom","name":"task-specific name","role":"task-specific responsibility","task":"...","tools":[],"skills":[],"tomLevel":1,"existenceReason":"which cognitive gap this child fills"}]}
{"action":"spawn_subagents","reason":"...","coordination":"team","team":{"name":"task-specific team name","description":"...","synthesisPolicy":"..."},"agents":[...]}
Allowed archetypes: researcher, critic, planner, coder, summarizer, tester, custom.`,
        },
        {
          role: 'user',
          content: [
            `<agent>${JSON.stringify(agent.identity, null, 2)}</agent>`,
            `<task>${task}</task>`,
            `<acceptance_checklist>${this.buildTaskAcceptanceChecklist(task)}</acceptance_checklist>`,
            `<policy>${JSON.stringify(policy, null, 2)}</policy>`,
            `<execution_knowledge>${this.formatExecutionKnowledge(executionKnowledge)}</execution_knowledge>`,
            '<runtime_policy>Delegate only to a direct child or a direct child team. The parent must synthesize child results before passing anything upward. A child team may contain multiple fresh actors and those actors may continue recursive team-first delegation within policy.</runtime_policy>',
          ].join('\n\n'),
        },
        ],
        { temperature: 0.1, maxTokens: 1000 },
        'agent.delegation_decision',
        correlationId
      );
      const normalized = this.normalizeAgentDelegationDecision(decision, task, fallback);
      const constrained = this.constrainDelegationToExecutablePlans(
        agent.identity.id,
        this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, normalized),
        correlationId
      );
      const selected = await this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
      this.emit({
        type: 'delegation.assess.completed',
        agentId: agent.identity.id,
        data: { correlationId, action: selected.action, source: 'llm', scope: 'agent' },
      });
      return selected.action === 'ask_clarification'
        ? { action: 'solve_directly', reason: selected.reason }
        : selected;
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'delegation.decision.fallback',
        agentId: agent.identity.id,
        data: {
          correlationId,
          reason: 'llm_agent_delegation_failed',
          error: error instanceof Error ? error.message : String(error),
        },
      });
      const constrained = this.constrainDelegationToExecutablePlans(
        agent.identity.id,
        this.applyAgentBudgetAndPolicyConstraints(agent.identity.id, fallback),
        correlationId
      );
      return this.selectDelegationCandidate(agent.identity.id, task, constrained, correlationId, 'agent');
    }
  }

  private fallbackAgentDelegationDecision(agent: AgentInfo, task: string): DelegationDecision {
    if (agent.identity.teamId && task.includes('<team_step_cache>')) {
      return {
        action: 'solve_directly',
        reason: 'The model did not select a recursive child, and this team member already received prior member state for its bounded role.',
      };
    }
    const archetype = this.inferAgentArchetype(agent);
    const lower = task.toLowerCase();
    const wantsReview = /\b(review|critique|risk|risks|failure|validate|audit)\b/.test(lower);
    const wantsPromptAudit = /\b(prompt|slot|slots|render|context)\b/.test(lower)
      && /\b(check|inspect|audit|review|validate)\b/.test(lower);
    const wantsTests = /\b(test|tests|verify|verification|regression)\b/.test(lower);

    const agents: DelegationAgentPlan[] = [];

    if (wantsPromptAudit && archetype !== 'custom') {
      agents.push({
        archetype: 'custom',
        name: 'PromptAuditor-1',
        task: `Inspect prompt/context correctness for parent ${agent.identity.name}: ${task}`,
        tools: ['fs.read'],
        skills: ['use_tool_when_needed'],
        tomLevel: 1,
      });
    }

    if (wantsReview && archetype !== 'critic') {
      agents.push({
        archetype: 'critic',
        name: 'Critic-1',
        task: `Review the parent agent task for risks, gaps, and grounding issues: ${task}`,
        tomLevel: 2,
      });
    }

    if (wantsTests && archetype !== 'tester') {
      agents.push({
        archetype: 'tester',
        name: 'Tester-1',
        task: `Evaluate test or verification needs for: ${task}`,
        tomLevel: 0,
      });
    }

    if (agents.length > 0) {
      return {
        action: 'spawn_subagents',
        reason: 'The task benefits from direct child specialists before the parent synthesizes upward.',
        agents: agents.slice(0, 3),
      };
    }

    return {
      action: 'solve_directly',
      reason: 'The task does not require a direct child specialist.',
    };
  }

  private normalizeAgentDelegationDecision(decision: unknown, task: string, fallback: DelegationDecision): DelegationDecision {
    const item = decision as Partial<DelegationDecision>;
    if (item.action === 'solve_directly') {
      return {
        action: 'solve_directly',
        reason: typeof item.reason === 'string' && item.reason.trim()
          ? item.reason.trim()
          : 'The agent can complete this task directly.',
      };
    }

    if (item.action === 'spawn_subagents' && Array.isArray((item as { agents?: unknown[] }).agents)) {
      const agents = (item as { agents: Array<Partial<DelegationAgentPlan>> }).agents
        .filter(plan => this.isValidArchetype(String(plan.archetype)))
        .slice(0, 3)
        .map((plan): DelegationAgentPlan => this.normalizeDelegationAgentPlan(plan, task));
      if (agents.length > 0) {
        return {
          action: 'spawn_subagents',
          reason: typeof item.reason === 'string' && item.reason.trim()
            ? item.reason.trim()
            : 'The agent benefits from a direct child specialist.',
          agents,
          coordination: this.normalizeCoordination(item, agents.length),
          team: this.normalizeDelegationTeamPlan(item, task, agents.length),
          continuationPolicy: this.normalizeContinuationPolicy(item),
        };
      }
    }

    return fallback;
  }

  private applyAgentBudgetAndPolicyConstraints(parentId: string, decision: DelegationDecision): DelegationDecision {
    if (decision.action !== 'spawn_subagents') return decision;
    const policy = this.getAgentPolicy(parentId);
    if (!policy) return { action: 'solve_directly', reason: 'Parent policy is unavailable.' };
    const allowed = Math.max(0, policy.allowedChildren - policy.currentChildren);
    if (allowed <= 0) {
      return { action: 'solve_directly', reason: 'No child slots remain for this parent.' };
    }
    const supportedAgents = decision.agents.filter(agent => agent.archetype !== 'custom' || policy.spawnPolicy.allowCustomAgents);
    if (supportedAgents.length === 0) {
      return { action: 'solve_directly', reason: 'Requested custom child agents are not allowed by this parent policy.' };
    }
    const boundedAgents = supportedAgents.slice(0, Math.min(allowed, 3));
    return {
      ...decision,
      agents: boundedAgents,
      coordination: boundedAgents.length > 1 ? decision.coordination : 'independent',
      team: boundedAgents.length > 1 ? decision.team : undefined,
    };
  }

  private normalizeDelegationDecision(decision: unknown, userInput: string): DelegationDecision {
    const item = decision as Partial<DelegationDecision>;
    if (item.action === 'ask_clarification') {
      const question = typeof (item as { question?: unknown }).question === 'string'
        && (item as { question: string }).question.trim()
        ? (item as { question: string }).question.trim()
        : 'What exactly would you like Roy to improve: code, architecture, documentation, tests, or runtime behavior?';
      const reason = typeof item.reason === 'string' && item.reason.trim()
        ? item.reason.trim()
        : 'The task is too ambiguous to safely delegate.';
      if (this.taskNeedsWebAccess(userInput)
        && /\b(?:tool|internet|network|browser|curl|wget|web access|permission|available)\b/i.test(`${reason} ${question}`)
        && this.workspaceRuntimeConfig?.tools.web.enabled !== false
        && toolRegistry.get('web.search')
        && toolRegistry.get('web.fetch')) {
        return this.fallbackDelegationDecision(userInput);
      }
      if (this.taskRequiresWorkspaceMutation(userInput)
        && this.isToolPermissionClarification(reason, question)) {
        return this.applyBudgetConstraints(this.fallbackDelegationDecision(userInput));
      }
      return {
        action: 'ask_clarification',
        reason,
        question,
      };
    }

    if (item.action === 'solve_directly') {
      return {
        action: 'solve_directly',
        reason: typeof item.reason === 'string' && item.reason.trim()
          ? item.reason.trim()
          : 'The task appears simple enough for Roy to answer directly.',
      };
    }

    if (item.action === 'spawn_subagents' && Array.isArray((item as { agents?: unknown[] }).agents)) {
      const agents = (item as { agents: Array<Partial<DelegationAgentPlan>> }).agents
        .filter(plan => this.isValidArchetype(String(plan.archetype)))
        .slice(0, 3)
        .map((plan): DelegationAgentPlan => this.normalizeDelegationAgentPlan(plan, userInput));

      if (agents.length > 0) {
        return {
          action: 'spawn_subagents',
          reason: typeof item.reason === 'string' && item.reason.trim()
            ? item.reason.trim()
            : 'The task benefits from delegated specialist work.',
          agents,
          coordination: this.normalizeCoordination(item, agents.length),
          team: this.normalizeDelegationTeamPlan(item, userInput, agents.length),
          continuationPolicy: this.normalizeContinuationPolicy(item),
        };
      }
    }

    return this.applyBudgetConstraints(this.fallbackDelegationDecision(userInput));
  }

  private countIndependentTaskObligations(task: string): number {
    const lineItems = task
      .split(/\r?\n/)
      .filter(line => /^\s*(?:[-*+]\s+|\d+[.)]\s+|(?:question|q)\s*\d+\s*[:.)])/i.test(line))
      .length;
    const inlineQuestions = (task.match(/\b(?:question|q)\s*\d+\s*[:.)]/gi) ?? []).length;
    const numberedClauses = (task.match(/(?:^|\s)\d+[.)]\s+(?=\S)/g) ?? []).length;
    return Math.max(lineItems, inlineQuestions, numberedClauses);
  }

  private async auditComplexDirectDecision(
    task: string,
    initial: Extract<DelegationDecision, { action: 'solve_directly' }>,
    correlationId: string
  ): Promise<DelegationDecision> {
    const threshold = Math.max(
      0,
      this.workspaceRuntimeConfig?.delegation.rootSteps.directDecisionAuditMinObligations ?? 4
    );
    const obligations = this.countIndependentTaskObligations(task);
    if (threshold === 0 || obligations < threshold) return initial;

    const ctx = this.getContext();
    if (!ctx.llm) return initial;
    this.emit({
      type: 'delegation.direct_decision.audit.started',
      agentId: 'root',
      correlationId,
      data: { obligations, threshold, initialReason: initial.reason },
    });
    try {
      const raw = await this.completeJSONAsAgent<DelegationDecision>(ctx.agent, [
        {
          role: 'system',
          content: `You are Roy's direct-solve complexity auditor.
The primary controller chose direct execution. Independently check whether the task contains multiple separable answer obligations with distinct knowledge, verification, or perspective gaps.
Do not delegate merely because the input is long or has many items. Delegate only when one or more concrete gaps benefit from bounded parallel specialists, and keep synthesis with Roy.
Return strict JSON as either {"action":"solve_directly","reason":"..."} or {"action":"spawn_subagents","reason":"...","coordination":"independent","continuationPolicy":"finalize_after_round","agents":[{"archetype":"custom","name":"...","role":"...","task":"...","tools":[],"skills":[],"existenceReason":"..."}]}. Use at most 3 agents.`,
        },
        {
          role: 'user',
          content: [
            `<task>${task}</task>`,
            `<independent_obligation_count>${obligations}</independent_obligation_count>`,
            `<initial_direct_reason>${initial.reason}</initial_direct_reason>`,
            `<runtime_budget>${JSON.stringify(this.getBudgetState(), null, 2)}</runtime_budget>`,
          ].join('\n\n'),
        },
      ], { temperature: 0.1, maxTokens: 1100 }, 'root.direct_decision_audit', correlationId);
      const rawAction = (raw as { action?: unknown })?.action;
      if (rawAction !== 'solve_directly' && rawAction !== 'spawn_subagents') {
        throw new Error('Direct-decision audit returned no explicit supported action');
      }
      let audited = this.normalizeDelegationDecision(raw, task);
      audited = this.applyBudgetConstraints(audited);
      this.emit({
        type: audited.action === 'spawn_subagents'
          ? 'delegation.direct_decision.audit.overridden'
          : 'delegation.direct_decision.audit.completed',
        agentId: 'root',
        correlationId,
        data: {
          obligations,
          initialAction: initial.action,
          auditedAction: audited.action,
          reason: audited.reason,
        },
      });
      return audited;
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'delegation.direct_decision.audit.failed',
        agentId: 'root',
        correlationId,
        data: {
          obligations,
          error: error instanceof Error ? error.message : String(error),
        },
      });
      return initial;
    }
  }

  private normalizeCoordination(value: unknown, agentCount: number): 'independent' | 'team' {
    if (agentCount <= 1) return 'independent';
    const coordination = (value as { coordination?: unknown }).coordination;
    if (coordination === 'independent' || coordination === 'team') return coordination;
    return 'team';
  }

  private normalizeContinuationPolicy(value: unknown): 'reassess' | 'finalize_after_round' {
    return (value as { continuationPolicy?: unknown }).continuationPolicy === 'finalize_after_round'
      ? 'finalize_after_round'
      : 'reassess';
  }

  private normalizeDelegationAgentPlan(
    plan: Partial<DelegationAgentPlan>,
    fallbackTask: string
  ): DelegationAgentPlan {
    const archetype = String(plan.archetype) as SubAgentArchetype;
    const suppliedTask =
      typeof plan.task === 'string' && plan.task.trim()
        ? plan.task.trim()
        : fallbackTask;
    const descriptor = [
      typeof plan.name === 'string' ? plan.name : '',
      typeof plan.role === 'string' ? plan.role : '',
      typeof plan.existenceReason === 'string' ? plan.existenceReason : '',
    ].filter(Boolean).join('\n');
    const evidenceTask = this.taskRequiresIndependentWebEvidence(
      suppliedTask,
      archetype,
      descriptor
    ) && !suppliedTask.includes('[runtime_independent_web_evidence]')
      ? [
        suppliedTask,
        '[runtime_independent_web_evidence]',
        'Treat shared cached web results as a baseline, not as independent corroboration. When cached searches cover a material factual conclusion, use non-equivalent follow-up queries that probe alternate wording, scope, dates, and counterexamples; prefer primary or authoritative sources and report conflicts instead of following the prior majority.',
      ].join('\n')
      : suppliedTask;
    const task = this.compactDelegatedTask(
      evidenceTask
    );
    const requestedTools = Array.isArray(plan.tools)
      ? plan.tools.filter((tool): tool is string =>
          typeof tool === 'string'
          && toolRegistry.has(tool)
          && (!tool.startsWith('web.') || this.taskNeedsWebAccess(task))
        )
      : [];
    const inferredTools = this.inferMinimumTaskTools(task);
    const tools = Array.from(new Set([...requestedTools, ...inferredTools]));
    const requestedSkills = Array.isArray(plan.skills)
      ? plan.skills.filter((skill): skill is string => typeof skill === 'string' && skillRegistry.has(skill))
      : [];
    const skills = tools.length > 0
      ? Array.from(new Set([...requestedSkills, 'use_tool_when_needed']))
      : requestedSkills.filter(skill => skill !== 'use_tool_when_needed');
    return {
      archetype,
      name: typeof plan.name === 'string' ? plan.name : undefined,
      role: typeof plan.role === 'string' ? plan.role : undefined,
      style: typeof plan.style === 'string' ? plan.style : undefined,
      description: typeof plan.description === 'string' ? plan.description : undefined,
      task,
      tools: tools.length > 0 ? tools : undefined,
      skills: skills.length > 0 ? skills : undefined,
      tomLevel: typeof plan.tomLevel === 'number' ? plan.tomLevel : undefined,
      budgetTokens: typeof plan.budgetTokens === 'number' ? plan.budgetTokens : undefined,
      cognitiveGapIds: Array.isArray(plan.cognitiveGapIds)
        ? plan.cognitiveGapIds.filter((item): item is string => typeof item === 'string')
        : undefined,
      existenceReason: typeof plan.existenceReason === 'string' ? plan.existenceReason : undefined,
      memoryScope: this.agentMemoryScope(plan.memoryScope),
      systemPrompt: typeof plan.systemPrompt === 'string' ? plan.systemPrompt : undefined,
    };
  }

  private inferMinimumTaskTools(task: string): string[] {
    const lower = task.toLowerCase();
    const tools: string[] = [];
    if (this.taskNeedsWebAccess(task)) tools.push('web.search', 'web.fetch');
    if (/\b(?:read|inspect|review|check|audit|verify|compare|cross-reference)\b/.test(lower)
      && /(?:\b(?:file|source|manifest|package\.json|exports?|apis?)\b|[./][a-z0-9_-]+\.(?:ts|tsx|js|jsx|json|md|ya?ml|toml)\b)/i.test(task)
      && !this.taskNeedsWebAccess(task)) {
      tools.push('fs.read', 'fs.search');
    }
    if (/\b(?:list|tree|structure|directories|workspace|repository|codebase)\b/.test(lower)
      && !this.taskNeedsWebAccess(task)) {
      tools.push('fs.list');
    }
    if (/\b(?:run|execute)\s+(?:the\s+)?(?:tests?|build)\b|\bnpm (?:test|run build)\b/.test(lower)) {
      tools.push('shell.exec');
    }
    if (/\b(?:terminal|shell|command line|cli|container)\b/.test(lower)
      || /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor)\b[\s\S]*\b(?:file|code|project|repository|workspace|artifact|solution)\b/.test(lower)) {
      tools.push('fs.list', 'fs.read', 'fs.search', 'fs.replace', 'fs.write', 'fs.synthesize', 'shell.exec');
    }
    return Array.from(new Set(tools)).filter(tool => toolRegistry.has(tool));
  }

  private normalizeDelegationTeamPlan(
    value: unknown,
    task: string,
    agentCount: number
  ): DelegationTeamPlan | undefined {
    if (agentCount <= 1 || this.normalizeCoordination(value, agentCount) !== 'team') return undefined;
    const raw = (value as { team?: unknown }).team;
    if (!raw || typeof raw !== 'object') return undefined;
    const team = raw as Record<string, unknown>;
    const name = typeof team.name === 'string' && team.name.trim() ? team.name.trim() : 'DelegatedTeam';
    const description = typeof team.description === 'string' && team.description.trim()
      ? team.description.trim()
      : `Coordinate task-specific agents for: ${task}`;
    const rawPolicy = team.executionPolicy && typeof team.executionPolicy === 'object'
      ? team.executionPolicy as Record<string, unknown>
      : undefined;
    const executionPolicy: Partial<TeamExecutionPolicy> | undefined = rawPolicy ? {
      mode: rawPolicy.mode === 'parallel' || rawPolicy.mode === 'sequential' ? rawPolicy.mode : undefined,
      failureMode: rawPolicy.failureMode === 'fail_fast' || rawPolicy.failureMode === 'best_effort'
        ? rawPolicy.failureMode
        : undefined,
      maxConcurrency: typeof rawPolicy.maxConcurrency === 'number' ? Math.max(1, Math.floor(rawPolicy.maxConcurrency)) : undefined,
      minimumSuccessfulMembers: typeof rawPolicy.minimumSuccessfulMembers === 'number'
        ? Math.max(1, Math.floor(rawPolicy.minimumSuccessfulMembers))
        : undefined,
    } : undefined;
    return {
      name,
      description,
      task: typeof team.task === 'string' && team.task.trim() ? team.task.trim() : task,
      synthesisPolicy: typeof team.synthesisPolicy === 'string' && team.synthesisPolicy.trim()
        ? team.synthesisPolicy.trim()
        : undefined,
      tomLevel: typeof team.tomLevel === 'number'
        ? Math.max(0, Math.min(3, Math.floor(team.tomLevel)))
        : undefined,
      executionPolicy,
      memberDelegationPolicy: team.memberDelegationPolicy === 'deny'
        || this.normalizeContinuationPolicy(value) === 'finalize_after_round'
        ? 'deny'
        : 'allow',
    };
  }

  private fallbackDelegationDecision(userInput: string): DelegationDecision {
    const lower = userInput.toLowerCase();
    const words = lower.trim().split(/\s+/).filter(Boolean);
    const ambiguousImprove = /\b(help|improve|fix|make better|enhance|optimi[sz]e)\b/.test(lower)
      && !/\b(code|repo|repository|project|architecture|test|tests|docs|documentation|memory|cache|runtime|cli|server|api|bug|risk|file|structure)\b/.test(lower);
    if (ambiguousImprove || (words.length <= 5 && /\b(help|improve|fix)\b/.test(lower))) {
      return {
        action: 'ask_clarification',
        reason: 'The request is too broad to select an agent or task safely.',
        question: 'What would you like Roy to improve: code, architecture, documentation, tests, memory/cache behavior, or CLI/API behavior?',
      };
    }

    const asksProjectInspection = /\b(inspect|analy[sz]e|review|audit|check|read|list)\b/.test(lower)
      && (
        /\b(repo|repository|project|codebase|architecture|structure|src|files?|filesystem)\b/.test(lower)
        || /(?:^|\s)[./]?[a-z0-9_-]+\.(?:json|ya?ml|toml|md|ts|tsx|js|jsx|mjs|cjs)(?:\s|$)/.test(lower)
      );
    const asksRisk = /\b(risk|risks|problem|bug|bugs|issue|issues|critique|review|regression|coupling)\b/.test(lower);
    const asksPlan = /\b(plan|steps|roadmap|refactor|design|phase|implement)\b/.test(lower);
    const asksCode = /\b(code|implement|fix|modify|change|patch)\b/.test(lower);
    const agents: DelegationAgentPlan[] = [];

    if (this.taskNeedsWebAccess(userInput)) {
      agents.push({
        archetype: 'researcher',
        name: 'WebResearcher-1',
        task: `Use web.search and web.fetch to collect task-relevant public evidence, then return a source-backed report for: ${userInput}`,
        tools: ['web.search', 'web.fetch'],
        skills: ['use_tool_when_needed'],
        tomLevel: 0,
        existenceReason: 'Collect and open current public-web evidence that Roy cannot establish from local memory.',
      });
    }

    if (asksProjectInspection && !this.taskNeedsWebAccess(userInput)) {
      agents.push({
        archetype: 'researcher',
        name: 'Researcher-1',
        task: `Inspect grounded project structure and collect concrete evidence for: ${userInput}`,
        tomLevel: 0,
      });
    }
    if (asksRisk) {
      agents.push({
        archetype: 'critic',
        name: `Critic-${agents.length + 1}`,
        task: `Identify architectural risks, hidden coupling, and failure modes for: ${userInput}`,
        tomLevel: 2,
      });
    }
    if (agents.length === 0 && asksPlan) {
      agents.push({
        archetype: 'planner',
        name: 'Planner-1',
        task: `Turn the user request into an actionable implementation plan: ${userInput}`,
        tomLevel: 1,
      });
    }
    if (agents.length === 0 && asksCode) {
      agents.push({
        archetype: 'coder',
        name: 'Coder-1',
        task: `Assess the coding change needed for: ${userInput}`,
        tomLevel: 0,
      });
    }

    if (agents.length > 0) {
      return {
        action: 'spawn_subagents',
        reason: 'The request is broad or evidence-seeking, so Roy should delegate specialist subtasks before synthesis.',
        agents: agents.slice(0, 3),
      };
    }

    return {
      action: 'solve_directly',
      reason: 'The request appears simple enough for Roy to answer without spawning subagents.',
    };
  }

  private requiresStagedDelegation(task: string): boolean {
    const normalized = task.toLowerCase().replace(/\s+/g, ' ');
    return /\b(after|once|based on|depending on)\b.{0,100}\b(result|finding|evidence|output|inspection)\b/.test(normalized)
      || /\b(first|initial)\b.{0,120}\b(then|next|afterward|subsequent)\b/.test(normalized)
      || /\b(then|next)\b.{0,100}\b(decide|determine|verify|delegate|spawn)\b/.test(normalized);
  }

  private requiresLongHorizonLoop(task: string): boolean {
    const normalized = task.toLowerCase().replace(/\s+/g, ' ');
    return this.requiresStagedDelegation(task)
      || /\b(long[- ]?(?:running|horizon|term)|multi[- ]?(?:step|stage|phase)|iterate|iteration|checkpoint|until complete|continue until|progressively|recursive)\b/.test(normalized)
      || /(长程|长期任务|多步骤|多阶段|逐步执行|持续执行|循环执行|直到完成|递归派生|检查点)/.test(normalized);
  }

  private shouldAssessAgentDelegation(
    agent: AgentInfo,
    task: string
  ): { assess: boolean; reason: string } {
    const inheritedParentMarker = /\bParent task\s*:/i.exec(task);
    const assignedResponsibility = inheritedParentMarker
      ? task.slice(0, inheritedParentMarker.index)
      : task;
    const explicitRecursiveResponsibility =
      /\b(?:delegate|delegation|spawn|subagent|sub-agent|subteam|sub-team|direct child|form a team|create a team)\b/i.test(assignedResponsibility)
      || /(?:派生|委派|子代理|子团队|组建团队)/.test(assignedResponsibility);
    const stagedAssignedResponsibility = this.requiresStagedDelegation(assignedResponsibility);
    const teamId = agent.identity.teamId;
    if (teamId) {
      const team = this.teams.get(teamId);
      if (team?.leadAgentId !== agent.identity.id) {
        return {
          assess: false,
          reason: 'This bounded team-member task should execute directly; recursive structure is coordinated by the team lead.',
        };
      }
      if (explicitRecursiveResponsibility || stagedAssignedResponsibility) {
        return {
          assess: true,
          reason: 'The team lead owns an explicit or staged recursive coordination responsibility.',
        };
      }
      return {
        assess: false,
        reason: 'The team lead has a bounded executable task and no concrete recursive coordination gap.',
      };
    }
    if (explicitRecursiveResponsibility || stagedAssignedResponsibility) {
      return {
        assess: true,
        reason: 'The standalone agent task explicitly requires delegation or depends on a later staged result.',
      };
    }
    return {
      assess: false,
      reason: 'The assigned agent task is a bounded leaf responsibility with no explicit descendant gap.',
    };
  }

  private buildTaskAcceptanceChecklist(task: string): string {
    const items = this.extractTaskAcceptanceItems(task);
    return JSON.stringify({
      items: items.map((item, index) => ({
        id: `acceptance_${String(index + 1).padStart(2, '0')}`,
        requirement: item,
        status: 'unverified',
      })),
      closurePolicy: [
        'Before finalizing, classify every item as verified, failed, or explicitly blocked using runtime evidence.',
        'Search for stale parallel declarations, call sites, generated metadata, configuration, and compatibility paths relevant to each item.',
        'A passing narrow check does not close sibling requirements; preserve failures and unverified items as actionable feedback.',
      ],
    }, null, 2);
  }

  private extractTaskAcceptanceItems(task: string): string[] {
    const lines = task
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(Boolean);
    const explicitItems: string[] = [];
    let requirementSection = false;
    let passiveReferenceSection = false;
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]!;
      const normativeLine = /\b(?:must|must not|required|preserve|keep|honou?r|support|pass only if)\b/i.test(line)
        || /(?:必须|不得|需要|保持|保留|支持|完成后|通过条件)/.test(line);
      const markdownSection = /^#{1,6}\s+/.test(line)
        ? line.replace(/^#{1,6}\s+/, '').replace(/:$/, '').trim()
        : undefined;
      const plainSectionLabel = line.match(/^([^`].*?):\s*$/)?.[1]?.trim();
      const knownSectionLabel = Boolean(plainSectionLabel
        && /^(?:requirements?|required outputs?|deliverables?|acceptance(?: criteria)?|output contract|artifacts?|(?:(?:official\s*\/\s*public|official|public)\s+)?references?)$/i.test(
          plainSectionLabel
        ));
      const plainSection = !normativeLine || knownSectionLabel
        ? plainSectionLabel
        : undefined;
      const section = markdownSection ?? plainSection;
      if (section) {
        passiveReferenceSection = /^(?:(?:official\s*\/\s*public|official|public)\s+)?references?$/i.test(section);
        requirementSection = !passiveReferenceSection
          && /\b(?:requirements?|required outputs?|deliverables?|acceptance|output contract|artifacts?)\b/i.test(section);
        continue;
      }
      if (passiveReferenceSection) continue;
      const listed = /^(?:[-*+]\s+|\d+[.)]\s+)/.test(line);
      const explicit = normativeLine || listed && (
        requirementSection
        || /\boutputs?\//i.test(line)
        || /\.(?:json|csv|md|ya?ml|toml|txt|py|ts|js)\b/i.test(line)
      );
      if (!explicit) continue;
      let requirement = line
        .replace(/^(?:[-*+]\s+|\d+[.)]\s+)/, '')
        .replace(/\s+/g, ' ');
      if (/\b(?:include|contain|with)\b.*:\s*$/i.test(requirement)) {
        const openingFenceIndex = lines.findIndex((candidate, candidateIndex) =>
          candidateIndex > index && candidate.startsWith('```')
        );
        if (openingFenceIndex === index + 1) {
          const closingFenceIndex = lines.findIndex((candidate, candidateIndex) =>
            candidateIndex > openingFenceIndex && candidate.startsWith('```')
          );
          if (closingFenceIndex > openingFenceIndex + 1) {
            const contract = lines
              .slice(openingFenceIndex + 1, closingFenceIndex)
              .join(' | ')
              .replace(/\s+/g, ' ');
            requirement = `${requirement} ${contract}`;
            index = closingFenceIndex;
          }
        }
      }
      explicitItems.push(requirement.slice(0, 500));
    }
    const items = [...new Set(explicitItems)].slice(0, 50);
    if (items.length === 0) {
      items.push(task.replace(/\s+/g, ' ').trim().slice(0, 1200));
    }
    return items;
  }

  private taskRequiresAcceptanceAudit(task: string): boolean {
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.requireAcceptanceAudit === false) {
      return false;
    }
    const threshold = Math.max(
      1,
      this.workspaceRuntimeConfig?.delegation.rootSteps.directDecisionAuditMinObligations ?? 4
    );
    return this.requiresLongHorizonLoop(task)
      || this.extractTaskAcceptanceItems(task).length >= threshold;
  }

  private ensureLongHorizonRecoveryContinuation(
    continuation: RootContinuationDecision,
    task: string,
    completedStep: RootExecutionStep,
    delegationRound: number,
    maxDelegationRounds: number,
    requiresLongHorizon: boolean,
    requiresWorkspaceMutation: boolean,
    correlationId: string
  ): RootContinuationDecision {
    if (!requiresLongHorizon
      || !requiresWorkspaceMutation
      || continuation.action !== 'finalize'
      || delegationRound >= maxDelegationRounds) {
      return continuation;
    }
    const pathState = completedStep.cache?.path;
    if (!pathState?.mutationObserved
      || (pathState.verificationObserved && pathState.status === 'completed')) {
      return continuation;
    }
    const actionableFeedback = completedStep.cache?.feedback
      .filter(item => item.actionable)
      .slice(0, 12)
      .map(item => item.summary) ?? [];
    const nextRound = delegationRound + 1;
    this.emit({
      type: 'root.step.long_horizon_recovery.required',
      agentId: 'root',
      correlationId,
      data: {
        stepId: completedStep.id,
        pathId: pathState.id,
        nextRound,
        pathStatus: pathState.status,
        mutationObserved: pathState.mutationObserved,
        verificationObserved: pathState.verificationObserved,
        actionableFeedback: actionableFeedback.length,
        rejectedFinalizeReason: continuation.reason,
      },
    });
    return {
      action: 'delegate_more',
      reason: [
        `Step ${completedStep.index} changed the workspace but did not close its execution path.`,
        pathState.verificationObserved
          ? 'Verification ran, but failed tools or unresolved feedback keep the path partial.'
          : 'No successful verification was observed after the mutation.',
        'Continue from cached state instead of repeating broad discovery.',
      ].join(' '),
      agents: [
        {
          archetype: 'coder',
          name: `RecoveryExecutor-${nextRound}`,
          role: 'cached-state recovery executor',
          task: [
            'Continue the long-horizon workspace task from the persisted execution cache.',
            'Do not restart broad project discovery and do not retry cached invalid paths without a changed hypothesis.',
            'Implement the next unresolved bounded slice, consume failed tool output as repair feedback, and run a focused check before reporting.',
            `Original task: ${task}`,
          ].join('\n'),
          tools: ['fs.list', 'fs.read', 'fs.search', 'fs.replace', 'fs.write', 'fs.synthesize', 'shell.exec'],
          skills: ['use_tool_when_needed', 'delegate_to_subagent'],
          tomLevel: 1,
          existenceReason: 'A cached mutated path remains partial or unverified and requires another bounded implementation slice.',
        },
        {
          archetype: 'tester',
          name: `RecoveryVerifier-${nextRound}`,
          role: 'independent cached-path verifier',
          task: [
            'Consume the preceding team member result and persisted execution cache.',
            'Verify the actual workspace after the recovery edit with the most relevant executable checks.',
            'Report exact failures, authoritative paths, and the smallest next repair; do not claim completion without successful tool evidence.',
            `Original task: ${task}`,
          ].join('\n'),
          tools: ['fs.list', 'fs.read', 'fs.search', 'shell.exec'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: 'The preceding mutation lacks a closed, independently verified execution path.',
        },
      ],
      coordination: 'team',
      continuationPolicy: 'reassess',
      team: {
        name: `LongHorizonRecoveryTeam-${nextRound}`,
        description: 'Continues a partial cached execution path through a bounded implementation and independent verification pass.',
        task,
        synthesisPolicy: 'Return completed mutations, verification evidence, remaining failed paths, and actionable feedback as the next cached checkpoint.',
        memberDelegationPolicy: 'allow',
        executionPolicy: {
          mode: 'sequential',
          failureMode: 'best_effort',
          maxConcurrency: 1,
          minimumSuccessfulMembers: 1,
        },
      },
    };
  }

  private buildVerifierGuidedResumeRecoveryDecision(
    task: string,
    resumeState: ExecutionResumeState
  ): Extract<DelegationDecision, { action: 'spawn_subagents' }> | undefined {
    if (this.workspaceRuntimeConfig?.teams.enabled === false) return undefined;
    const priorRecoveryActors = resumeState.knowledge.actors.filter(actor =>
      actor.name.startsWith('VerifierProbe-')
      || actor.name.startsWith('FocusedRepairer-')
      || actor.role === 'verifier-guided recovery executor'
    );
    const priorRecoveryRounds = new Set(
      priorRecoveryActors.map(actor => actor.stepId)
    ).size;
    const recoveryRound = priorRecoveryRounds + 1;

    const calls = resumeState.knowledge.paths
      .flatMap(path => path.toolFrontier ?? [])
      .sort((left, right) =>
        (left.completedAt ?? left.startedAt ?? 0)
        - (right.completedAt ?? right.startedAt ?? 0)
      );
    const authoritativeVerifierCommand = String(
      this.selectAuthoritativePriorVerification(calls, task)?.params.command ?? ''
    ).trim() || undefined;
    let latestScorecardIndex = -1;
    let latestScorecard: ReturnType<Runtime['latestVerifierScorecardFromCalls']>;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const scorecard = this.latestVerifierScorecardFromCalls([calls[index]!]);
      if (!scorecard) continue;
      latestScorecardIndex = index;
      latestScorecard = scorecard;
      break;
    }
    let rejectedCandidateIndex = -1;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const result = call.result as { synthesisRejected?: unknown } | undefined;
      if (workspaceCandidateRollbackFromCall(call) || result?.synthesisRejected === true) {
        rejectedCandidateIndex = index;
        break;
      }
    }
    const rejectedCandidate = rejectedCandidateIndex >= 0
      ? calls[rejectedCandidateIndex]
      : undefined;
    const rejectedRollback = rejectedCandidate
      ? workspaceCandidateRollbackFromCall(rejectedCandidate)
      : undefined;
    const acceptedProgressAfterRejection = Boolean(
      rejectedCandidate
      && latestScorecard
      && latestScorecardIndex > rejectedCandidateIndex
      && typeof rejectedRollback?.baselineReward === 'number'
      && latestScorecard.reward > rejectedRollback.baselineReward + 1e-12
    );
    const latestRejected = acceptedProgressAfterRejection
      ? undefined
      : rejectedCandidate;
    const unresolvedAcceptedScorecard = Boolean(
      latestScorecard
      && latestScorecard.reward < 1 - 1e-12
      && Object.values(latestScorecard.groups).some(score => score < 1)
    );
    if (!latestRejected && !unresolvedAcceptedScorecard) return undefined;
    const causalAnchorIndex = latestRejected
      ? rejectedCandidateIndex
      : latestScorecardIndex;
    let latestDiagnosticIndex = -1;
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const stdout = String(
        (call.result as { stdout?: unknown } | undefined)?.stdout ?? ''
      );
      if (this.isFocusedVerifierDiagnosticCall(call)
        && /VERIFIER_PROBE_EVIDENCE_VERSION [23]\b/.test(stdout)) {
        latestDiagnosticIndex = index;
        break;
      }
    }
    let netMutationDepthAfterDiagnostic = 0;
    if (latestDiagnosticIndex >= 0) {
      for (const call of calls.slice(latestDiagnosticIndex + 1)) {
        if (isSuccessfulWorkspaceMutationCall(call)) {
          netMutationDepthAfterDiagnostic += 1;
        }
        const laterRollback = workspaceCandidateRollbackFromCall(call);
        if (laterRollback?.restored === true && netMutationDepthAfterDiagnostic > 0) {
          netMutationDepthAfterDiagnostic -= 1;
        }
      }
    }
    const rejectedBeforeMutation = Boolean(
      latestRejected
      && !workspaceCandidateRollbackFromCall(latestRejected)
      && (latestRejected.result as { synthesisRejected?: unknown } | undefined)
        ?.synthesisRejected === true
    );
    const cachedDiagnostic = latestDiagnosticIndex >= 0
      && netMutationDepthAfterDiagnostic === 0
      && (
        latestDiagnosticIndex > causalAnchorIndex
        || rejectedBeforeMutation
      )
      ? calls[latestDiagnosticIndex]
      : undefined;
    const rollback = latestRejected
      ? workspaceCandidateRollbackFromCall(latestRejected)
      : undefined;
    const rejection = latestRejected?.result as {
      path?: unknown;
      reason?: unknown;
      generatedPreview?: unknown;
    } | undefined;
    const latestMutation = [...calls].reverse().find(call =>
      isSuccessfulWorkspaceMutationCall(call)
      && !this.normalizeCachedPath(String(call.params.path ?? '')).startsWith('.roy/')
    );
    const cachedTargetPath = this.normalizeCachedPath(String(
      rollback?.path
        ?? rejection?.path
        ?? latestMutation?.params.path
        ?? rejectedCandidate?.params.path
        ?? ''
    ));
    const externalFeedback = this.latestExternalVerificationFeedback(task);
    const targetPath = externalFeedback
      ? externalFeedback.path ?? ''
      : cachedTargetPath;
    const target = targetPath
      || (externalFeedback
        ? `the workspace declaration controlling this failure: ${externalFeedback.summary}`
        : 'the implementation target named by the task');
    const unresolvedGroups = Object.entries(latestScorecard?.groups ?? {})
      .filter(([, score]) => score < 1)
      .map(([group, score]) => `${group}=${score.toFixed(3)}`);
    const recoveryCapsule = JSON.stringify({
      target,
      sourceCorrelationId: resumeState.sourceCorrelationId,
      anchorPathId: resumeState.anchorPathId,
      recoveryRound,
      latestAcceptedScorecard: latestScorecard,
      recoveryTrigger: externalFeedback
        ? 'new_external_verifier_feedback'
        : latestRejected
          ? 'current_rejected_candidate'
          : 'incomplete_accepted_scorecard_after_progress',
      externalFeedback,
      unresolvedGroups,
      rejectedCandidate: latestRejected && !externalFeedback
        ? {
          strategy: latestRejected.params.strategy ?? 'complete',
          reason: typeof rejection?.reason === 'string'
            ? rejection.reason.slice(0, 600)
            : rollback?.reason,
          baselineReward: rollback?.baselineReward,
          candidateReward: rollback?.candidateReward,
          regressedGroups: rollback?.regressedGroups,
          improvedGroups: rollback?.improvedGroups,
        }
        : undefined,
      currentWorkspacePolicy: 'Re-observe the current source and run a fresh verifier probe; cached source bytes and rejected payloads are not authoritative.',
      authoritativeVerifierCommand,
    }, null, 2);
    const immutableAssignment = this.compactRecoveryAssignment(task);
    const focusedRecoveryMemory: AgentMemoryScope = {
      public: false,
      private: false,
      parentContext: true,
      sessionWindowTurns: 0,
    };
    const memberOrdinal = (recoveryRound - 1) * 3;
    const diagnosticAgent: DelegationAgentPlan = {
      archetype: 'tester',
      name: `VerifierProbe-${memberOrdinal + 1}`,
      role: 'verifier-guided diagnostic probe',
      task: [
        '[runtime_verifier_diagnostic_probe]',
        `Diagnose the unresolved official-verifier behavior for ${target}.`,
        'Read the immutable verifier source and current implementation. Use shell execution to reproduce the failing capability with a disposable diagnostic fixture when possible; do not merely rerun the aggregate score.',
        'Compare actual and expected outputs, identify the smallest causal mismatch, and report exact source anchors and a bounded repair hypothesis for the next member.',
        'Diagnose only groups listed as unresolved in the recovery capsule. Treat baseline score 1 groups as protected and ignore their probe observations unless the newest official scorecard proves a current regression.',
        'Do not mutate the implementation or verifier.',
        `<recovery_capsule>\n${recoveryCapsule}\n</recovery_capsule>`,
        `Immutable assignment:\n${immutableAssignment}`,
      ].join('\n\n'),
      tools: ['fs.read', 'fs.search', 'shell.exec'],
      skills: ['use_tool_when_needed'],
      tomLevel: 2,
      existenceReason: latestRejected
        ? 'A score-only repair was rejected; executable mismatch evidence is required before another mutation.'
        : 'The accepted scorecard still has unresolved groups after newer progress; fresh causal evidence is required without reviving obsolete failures.',
      memoryScope: focusedRecoveryMemory,
    };
    const repairAgent: DelegationAgentPlan = {
      archetype: 'coder',
      name: `FocusedRepairer-${memberOrdinal + 2}`,
      role: 'verifier-guided recovery executor',
      task: [
        externalFeedback
          ? `Consume the newest concrete external verifier feedback and repair ${target}: ${externalFeedback.summary}`
          : cachedDiagnostic
          ? `Consume the cached focused verifier diagnostic and repair ${target}.`
          : `Consume the diagnostic member result and repair ${target}.`,
        'Inspect the authoritative current snapshot and use the smallest coherent fs.replace or focused fs.synthesize patch supported by the diagnostic evidence.',
        'Preserve all verifier groups already passing at the accepted baseline. Run the focused reproduction and official verifier after the mutation; consume failures as feedback before reporting.',
        authoritativeVerifierCommand
          ? `Configured authoritative verifier command:\n\`\`\`bash\n${authoritativeVerifierCommand}\n\`\`\``
          : '',
        'Do not perform another broad whole-file rewrite and do not modify immutable verifier evidence.',
        `<recovery_capsule>\n${recoveryCapsule}\n</recovery_capsule>`,
        `Immutable assignment:\n${immutableAssignment}`,
      ].join('\n\n'),
      tools: ['fs.read', 'fs.search', 'fs.replace', 'fs.write', 'fs.synthesize', 'shell.exec'],
      skills: ['use_tool_when_needed'],
      tomLevel: 2,
      existenceReason: externalFeedback
        ? 'A newer concrete verifier failure supersedes the stale cached causal target and supports a direct localized repair.'
        : cachedDiagnostic
        ? 'A fresh cached diagnostic already identifies the unresolved behavior; continue directly with its localized repair.'
        : latestRejected
          ? 'The diagnostic hypothesis must be turned into a localized, verified workspace mutation.'
          : 'The remaining accepted-scorecard gap must be turned into one localized, verified workspace mutation.',
      memoryScope: focusedRecoveryMemory,
    };
    const verifierAgent: DelegationAgentPlan = {
      archetype: 'tester',
      name: `RecoveryVerifier-${memberOrdinal + 3}`,
      role: 'independent recovery verifier',
      task: [
        'Independently verify the workspace state produced by the recovery executor.',
        'Run the configured official verifier and inspect its scorecard. Confirm that accepted baseline groups remain passing and report exact remaining mismatches.',
        authoritativeVerifierCommand
          ? `Configured authoritative verifier command:\n\`\`\`bash\n${authoritativeVerifierCommand}\n\`\`\``
          : '',
        'Do not mutate the implementation or verifier and do not claim closure unless the official verifier succeeds.',
        `<recovery_capsule>\n${recoveryCapsule}\n</recovery_capsule>`,
        `Immutable assignment:\n${immutableAssignment}`,
      ].join('\n\n'),
      tools: ['fs.read', 'fs.search', 'shell.exec'],
      skills: ['use_tool_when_needed'],
      tomLevel: 1,
      existenceReason: 'The localized repair needs independent objective validation before root acceptance.',
      memoryScope: focusedRecoveryMemory,
    };
    const instrumentableGradeVerifier = Boolean(
      authoritativeVerifierCommand
      && /\.roy\/official-verifier\/grade\.py(?:\s|$)/i.test(
        authoritativeVerifierCommand
      )
    );
    const recoveryAgents = externalFeedback
      ? [repairAgent, verifierAgent]
      : cachedDiagnostic
        ? [repairAgent, verifierAgent]
        : instrumentableGradeVerifier
          ? [diagnosticAgent, repairAgent, verifierAgent]
          : [repairAgent, verifierAgent];
    return {
      action: 'spawn_subagents',
      reason: externalFeedback
        ? 'New concrete external verifier feedback supersedes the stale cached recovery frontier; repair its authoritative target directly, then independently verify it.'
        : cachedDiagnostic
        ? 'The prior recovery stopped after producing a fresh focused diagnostic; resume directly at localized repair and independent verification.'
        : latestRejected && instrumentableGradeVerifier
          ? 'The persisted root repair strategy produced a verifier rollback or an invalid patch; change the causal hypothesis through a bounded diagnostic, localized repair, and independent verification team.'
          : latestRejected
            ? 'The persisted repair candidate was rejected, but the authoritative verifier is not compatible with the specialized grade.py probe; re-read the exact verifier and source, apply one localized repair, and independently rerun its declared command.'
          : 'A newer accepted verifier scorecard superseded the old rejection but remains incomplete; diagnose only its unresolved groups, then apply and independently verify one localized repair.',
      coordination: 'team',
      continuationPolicy: 'reassess',
      agents: recoveryAgents,
      team: {
        name: recoveryRound === 1
          ? 'VerifierGuidedRecoveryTeam'
          : `VerifierGuidedRecoveryTeam-${recoveryRound}`,
        description: 'Changes repair strategy after a rejected candidate by reproducing the mismatch, applying one localized repair, and independently verifying the accepted state.',
        task,
        synthesisPolicy: 'Return the reproduced mismatch, exact mutation, official verifier scorecard before and after, preserved passing groups, and any remaining causal blocker.',
        memberDelegationPolicy: 'deny',
        executionPolicy: {
          mode: 'sequential',
          failureMode: 'best_effort',
          maxConcurrency: 1,
          minimumSuccessfulMembers: 1,
        },
      },
    };
  }

  private latestExternalVerificationFeedback(
    task: string
  ): { summary: string; path?: string } | undefined {
    const capsuleMatch = /<recovery_capsule>\s*([\s\S]*?)\s*<\/recovery_capsule>/i.exec(
      task
    );
    if (capsuleMatch) {
      try {
        const capsule = JSON.parse(capsuleMatch[1]!) as {
          recoveryTrigger?: unknown;
          externalFeedback?: {
            summary?: unknown;
            path?: unknown;
          };
        };
        const summary = typeof capsule.externalFeedback?.summary === 'string'
          ? capsule.externalFeedback.summary.trim().slice(0, 1_200)
          : '';
        const path = typeof capsule.externalFeedback?.path === 'string'
          ? this.extractTaskDiagnosticSourcePaths(
            capsule.externalFeedback.path
          )[0]
          : undefined;
        if (capsule.recoveryTrigger === 'new_external_verifier_feedback'
          && summary) {
          return { summary, path };
        }
      } catch {
        // Fall through to the full external-verifier feedback parser.
      }
    }
    const failureMarker = task.toLowerCase().lastIndexOf('## verification failed');
    if (failureMarker < 0) return undefined;
    const latestFailure = task.slice(failureMarker);
    const feedbackMatch = /(?:^|\n)Verifier feedback:\s*\n([\s\S]*?)(?=\n\s*<official_verifier_feedback>|\n\s*##\s+Required local repair verification|\s*$)/i.exec(
      latestFailure
    );
    const feedback = feedbackMatch?.[1]?.trim();
    if (!feedback) return undefined;
    const summary = feedback
      .split(/\r?\n/)
      .map(line => line.trim())
      .find(line => line && !/^unchanged\b/i.test(line))
      ?.slice(0, 1_200);
    if (!summary) return undefined;
    const path = this.extractTaskDiagnosticSourcePaths(feedback)[0]
      ?? this.extractTaskDiagnosticInputPaths(feedback)[0];
    return {
      summary,
      path,
    };
  }

  private buildLongHorizonTeamDecision(
    task: string,
    requiresWorkspaceMutation: boolean
  ): Extract<DelegationDecision, { action: 'spawn_subagents' }> {
    const agents: DelegationAgentPlan[] = requiresWorkspaceMutation
      ? [
        {
          archetype: 'researcher',
          name: 'PathSteward-1',
          role: 'execution state and path steward',
          task: `Establish the first grounded checkpoint for this long-horizon task. Inspect authoritative workspace paths, identify cached invalid paths and actionable feedback, and pass a bounded execution map to the team: ${task}`,
          tools: ['fs.list', 'fs.read', 'fs.search'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: 'Maintain authoritative path, cache, and unresolved-feedback state for the team.',
        },
        {
          archetype: 'coder',
          name: 'Executor-2',
          role: 'workspace executor',
          task: `Inspect the actual workspace, implement the next complete bounded slice of the requested change, use failed tool output to repair it, and do not stop at a proposal: ${task}`,
          tools: ['fs.list', 'fs.read', 'fs.search', 'fs.replace', 'fs.write', 'fs.synthesize', 'shell.exec'],
          skills: ['use_tool_when_needed', 'delegate_to_subagent'],
          tomLevel: 1,
          existenceReason: 'Apply the requested workspace mutation with runtime tools.',
        },
        {
          archetype: 'tester',
          name: 'Verifier-3',
          role: 'independent closure verifier',
          task: `Verify the actual resulting workspace state for this task. Run relevant checks, distinguish authoritative and invalid paths, and return concrete repair feedback for any failure: ${task}`,
          tools: ['fs.list', 'fs.read', 'fs.search', 'shell.exec'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: 'Close the mutation-and-verification loop and produce actionable feedback.',
        },
      ]
      : [
        {
          archetype: 'researcher',
          name: 'EvidenceSteward-1',
          role: 'grounded evidence and path steward',
          task: `Establish the first grounded checkpoint, authoritative evidence paths, cached failures to avoid, and unresolved gaps for: ${task}`,
          tools: ['fs.list', 'fs.read', 'fs.search'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: 'Create durable evidence and path state for a long-horizon task.',
        },
        {
          archetype: 'tester',
          name: 'CheckpointVerifier-2',
          role: 'checkpoint and feedback verifier',
          task: `Independently verify the first bounded checkpoint, challenge unsupported claims, and return actionable feedback that later steps and descendants can consume for: ${task}`,
          tools: ['fs.list', 'fs.read', 'fs.search', 'shell.exec'],
          skills: ['use_tool_when_needed'],
          tomLevel: 1,
          existenceReason: 'Validate the checkpoint and expose concrete next-step feedback.',
        },
      ];
    return {
      action: 'spawn_subagents',
      reason: 'The request describes a staged or long-running task, so Roy will start with a checkpointed team whose members and descendants can extend one execution step.',
      agents,
      coordination: 'team',
      continuationPolicy: 'reassess',
      team: {
        name: 'LongHorizonCheckpointTeam',
        description: 'Maintains grounded state, executes bounded work, verifies closure, and propagates feedback across recursive derivations.',
        task,
        synthesisPolicy: 'Synthesize authoritative paths, completed work, failed paths, verification evidence, and unresolved feedback into the next root checkpoint.',
        memberDelegationPolicy: 'allow',
        executionPolicy: {
          mode: 'sequential',
          failureMode: 'best_effort',
          maxConcurrency: 1,
          minimumSuccessfulMembers: 1,
        },
      },
    };
  }

  private overrideExecutableTaskClarification(
    decision: DelegationDecision,
    task: string,
    requiresLongHorizon: boolean,
    requiresWorkspaceMutation: boolean,
    correlationId: string
  ): DelegationDecision {
    if (decision.action !== 'ask_clarification') {
      return decision;
    }
    if (this.isSelfContainedAnswerTask(task)) {
      this.emit({
        type: 'delegation.clarification.overridden',
        agentId: 'root',
        correlationId,
        data: {
          reason: 'The task already supplies a concrete question and explicit answer contract, so no additional user input is needed.',
          rejectedQuestion: decision.question,
          replacementAction: 'solve_directly',
        },
      });
      return {
        action: 'solve_directly',
        reason: 'The request is self-contained and declares a concrete answer contract; solve it directly instead of requesting unrelated clarification.',
      };
    }
    if (!requiresLongHorizon || !requiresWorkspaceMutation) return decision;
    const replacement = this.buildLongHorizonTeamDecision(task, true);
    this.emit({
      type: 'delegation.clarification.overridden',
      agentId: 'root',
      correlationId,
      data: {
        reason: 'Self-contained workspace evidence must be gathered with authorized tools instead of asking the user.',
        rejectedQuestion: decision.question,
      },
    });
    return replacement;
  }

  private isSelfContainedAnswerTask(task: string): boolean {
    if (!this.inferExplicitRootOutputContract(task) || task.trim().length < 80) {
      return false;
    }
    const hasConcreteAnswerRequest =
      /(?:\?|？)\s*(?:\n|$)|\b(?:solve|answer|identify|determine|calculate|choose|select|write|compose|produce)\b|(?:求解|回答|识别|确定|计算|选择|写出|生成)/i.test(
        task
      );
    const declaresMissingInput =
      /\b(?:tbd|todo|placeholder|ask the user to provide|wait for (?:the )?user input)\b|(?:待补充|占位符|请用户提供|等待用户输入)/i.test(
        task
      );
    return hasConcreteAnswerRequest && !declaresMissingInput;
  }

  private ensureLongHorizonTeamDecision(
    decision: Extract<DelegationDecision, { action: 'spawn_subagents' }>,
    task: string,
    requiresWorkspaceMutation: boolean,
    correlationId?: string
  ): Extract<DelegationDecision, { action: 'spawn_subagents' }> {
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.teamFirstLongHorizon === false
      || this.workspaceRuntimeConfig?.teams.enabled === false) {
      return decision;
    }
    const available = Math.min(
      3,
      this.getRemainingTotalAgentsForTurn('root', correlationId)
    );
    if (available < 2) return decision;
    const agents = [...decision.agents.slice(0, available)];
    const bootstrapAgents = this.buildLongHorizonTeamDecision(task, requiresWorkspaceMutation).agents;
    const additions = requiresWorkspaceMutation
      ? [
        ...bootstrapAgents.filter(agent => agent.archetype === 'coder'),
        ...bootstrapAgents.filter(agent => agent.archetype === 'tester'),
        ...bootstrapAgents.filter(agent => agent.archetype !== 'coder' && agent.archetype !== 'tester'),
      ]
      : [
        ...bootstrapAgents.filter(agent => agent.archetype === 'tester'),
        ...bootstrapAgents.filter(agent => agent.archetype !== 'tester'),
      ];
    const resumesGroundedPath = Boolean(
      decision.team?.name?.toLowerCase().includes('recovery')
      || decision.agents.some(agent =>
        /\b(?:cached-state|cached-path|verifier-guided recovery)\b/i.test(
          [agent.role, agent.existenceReason].filter(Boolean).join(' ')
        )
      )
    );
    const requiredPhases = requiresWorkspaceMutation
      ? resumesGroundedPath ? [1, 2] : [1, 2, 0]
      : [2, 0];
    for (const requiredPhase of requiredPhases) {
      if (agents.some(agent => this.longHorizonMemberPhase(agent) === requiredPhase)) {
        continue;
      }
      const addition = additions.find(agent =>
        this.longHorizonMemberPhase(agent) === requiredPhase
      );
      if (!addition) continue;
      if (agents.length < available) {
        agents.push(addition);
        continue;
      }
      const phaseCounts = new Map<number, number>();
      for (const agent of agents) {
        const phase = this.longHorizonMemberPhase(agent);
        phaseCounts.set(phase, (phaseCounts.get(phase) ?? 0) + 1);
      }
      const replaceable = agents
        .map((agent, index) => ({
          agent,
          index,
          phase: this.longHorizonMemberPhase(agent),
        }))
        .filter(item => (phaseCounts.get(item.phase) ?? 0) > 1)
        .sort((left, right) =>
          Number(right.agent.archetype === 'custom') - Number(left.agent.archetype === 'custom')
          || right.index - left.index
        )[0];
      if (replaceable) agents[replaceable.index] = addition;
    }
    const coveredPhases = new Set(agents.map(agent => this.longHorizonMemberPhase(agent)));
    for (const addition of additions) {
      if (agents.length >= available) break;
      const phase = this.longHorizonMemberPhase(addition);
      if (!requiredPhases.includes(phase)) continue;
      if (coveredPhases.has(phase)) continue;
      agents.push(addition);
      coveredPhases.add(phase);
    }
    if (agents.length < 2) return decision;
    const orderedAgents = agents
      .map((agent, index) => ({ agent, index, phase: this.longHorizonMemberPhase(agent) }))
      .sort((left, right) => left.phase - right.phase || left.index - right.index)
      .map(item => item.agent);
    const orderedPhases = orderedAgents.map(agent => this.longHorizonMemberPhase(agent));
    const hasPhaseDependency = new Set(orderedPhases).size > 1;
    const requestedExecutionPolicy = decision.team?.executionPolicy;
    if (hasPhaseDependency && (
      requestedExecutionPolicy?.mode !== 'sequential'
      || requestedExecutionPolicy.maxConcurrency !== 1
      || orderedAgents.some((agent, index) => agent !== agents[index])
    )) {
      this.emit({
        type: 'delegation.team.dependencies.normalized',
        agentId: 'root',
        correlationId,
        data: {
          phases: orderedAgents.map(agent => ({
            name: agent.name ?? agent.archetype,
            archetype: agent.archetype,
            phase: this.longHorizonMemberPhase(agent),
          })),
          requestedMode: requestedExecutionPolicy?.mode,
          appliedMode: 'sequential',
          reason: 'Long-horizon evidence, implementation, and verification phases must consume prior phase results.',
        },
      });
    }
    return {
      ...decision,
      agents: orderedAgents,
      coordination: 'team',
      continuationPolicy: 'reassess',
      team: {
        name: decision.team?.name ?? 'LongHorizonCheckpointTeam',
        description: decision.team?.description
          ?? 'Coordinates grounded state, execution, verification, and recursive descendant feedback for a long-horizon task.',
        task: decision.team?.task ?? task,
        synthesisPolicy: decision.team?.synthesisPolicy
          ?? 'Synthesize authoritative paths, completed work, failed paths, verification evidence, and unresolved feedback into the next root checkpoint.',
        tomLevel: decision.team?.tomLevel,
        executionPolicy: hasPhaseDependency
          ? {
            ...requestedExecutionPolicy,
            mode: 'sequential',
            failureMode: requestedExecutionPolicy?.failureMode ?? 'best_effort',
            maxConcurrency: 1,
            minimumSuccessfulMembers: Math.min(
              orderedAgents.length,
              requestedExecutionPolicy?.minimumSuccessfulMembers ?? 1
            ),
          }
          : requestedExecutionPolicy ?? {
            mode: 'parallel',
            failureMode: 'best_effort',
            maxConcurrency: Math.min(orderedAgents.length, 2),
            minimumSuccessfulMembers: 1,
          },
        memberDelegationPolicy: 'allow',
      },
    };
  }

  private longHorizonMemberPhase(plan: DelegationAgentPlan): 0 | 1 | 2 {
    if (plan.role === 'verifier-guided diagnostic probe') return 0;
    if (plan.archetype === 'researcher' || plan.archetype === 'planner') return 0;
    if (plan.archetype === 'coder') return 1;
    if (plan.archetype === 'tester'
      || plan.archetype === 'critic'
      || plan.archetype === 'summarizer') return 2;
    const intent = [
      plan.name,
      plan.role,
      plan.existenceReason,
    ].filter(Boolean).join(' ').toLowerCase();
    const leadingTaskIntent = plan.task.trim().slice(0, 500).toLowerCase();
    const verifiesOnly = (
      /\b(?:verifier|verification|validator|tester|reviewer|critic|acceptance)\b/.test(intent)
      || /^(?:after\b[\s\S]{0,120}\bcompletes?\b[\s,;:-]*)?(?:run|perform|execute)?\s*(?:the\s+)?(?:verification|validation|tests?|review|acceptance|audit)\b/.test(leadingTaskIntent)
    )
      && !/\b(?:implement|implementation|build|create|write|modify|edit|patch|repair|fix|migrate|executor|coder|engineer)\b/.test(intent);
    if (verifiesOnly) return 2;
    const combinedIntent = `${intent} ${leadingTaskIntent}`;
    const evidenceDeliverableOnly =
      /\b(?:explor|research|evidence|path steward|mapper|analyst)\w*\b/.test(intent)
      && /\b(?:write|create|produce|return|summari[sz]e|document)\b[\s\S]{0,120}\b(?:report|summary|findings|analysis|plan|inventory|map|documentation|notes?|project structure|implementation)\b/.test(
        leadingTaskIntent
      )
      && !/\b(?:implement|build|modify|edit|patch|repair|fix|migrate)\b/.test(
        leadingTaskIntent
      );
    const implementationIntent =
      /\b(?:executor|coder|engineer|implementation (?:owner|agent|specialist|lead))\b/.test(intent)
      || /\b(?:implement|build|modify|edit|patch|repair|fix|migrate)\b/.test(
        leadingTaskIntent
      )
      || /\b(?:create|write)\b[\s\S]{0,100}\b(?:source|code|pipeline|application|module|package|cli|service|required artifacts?|production files?)\b/.test(
        combinedIntent
      );
    if (implementationIntent && !evidenceDeliverableOnly) {
      return 1;
    }
    return 0;
  }

  private applyBudgetConstraints(decision: DelegationDecision): DelegationDecision {
    if (decision.action !== 'spawn_subagents') return decision;
    const budget = this.getBudgetState();
    if (budget.mode !== 'limited') return decision;

    const remaining = budget.remainingTokens ?? 0;
    if (remaining <= 500) {
      return {
        action: 'solve_directly',
        reason: `Budget constrained: only ${remaining} tokens remain, so Roy will avoid spawning subagents and answer directly.`,
      };
    }

    if (remaining <= 2000 && decision.agents.length > 1) {
      return {
        action: 'spawn_subagents',
        reason: `${decision.reason} Budget constrained: reduced delegation to one subagent with ${remaining} tokens remaining.`,
        agents: decision.agents.slice(0, 1),
        coordination: 'independent',
        continuationPolicy: decision.continuationPolicy,
      };
    }

    return decision;
  }

  private constrainDelegationToExecutablePlans(
    parentId: string,
    decision: DelegationDecision,
    correlationId: string,
    forcedGroundingPlanKeys: ReadonlySet<string> = new Set()
  ): DelegationDecision {
    if (decision.action !== 'spawn_subagents') return decision;
    const rejected: Array<{
      name?: string;
      archetype: SubAgentArchetype;
      reason: string;
      taskSummary: string;
      requestedTools: string[];
      automaticallyAuthorizedTools: string[];
    }> = [];
    const agents = decision.agents.filter(plan => {
      const planKey = `${plan.archetype}\u0000${plan.name ?? ''}\u0000${plan.task}`;
      if (!forcedGroundingPlanKeys.has(planKey)
        && !this.taskRequiresGrounding(plan.archetype, plan.task)) return true;
      const requested = Array.from(new Set([
        ...(plan.tools ?? []),
        ...this.inferMinimumTaskTools(plan.task),
      ]));
      const approved = this.getAutomaticallyApprovedToolBindings(
        plan.archetype,
        plan.task,
        requested
      );
      if (this.hasExecutableGroundingPath(plan.task, approved)) return true;
      rejected.push({
        name: plan.name,
        archetype: plan.archetype,
        reason: 'grounding_required_but_no_automatically_authorized_tool_path',
        taskSummary: plan.task.slice(0, 1000),
        requestedTools: requested,
        automaticallyAuthorizedTools: approved.map(binding => binding.name),
      });
      return false;
    });

    if (rejected.length > 0) {
      this.emit({
        type: 'delegation.plan.infeasible',
        agentId: parentId,
        correlationId,
        data: {
          rejected,
          retainedAgents: agents.map(agent => agent.name ?? agent.archetype),
        },
      });
    }
    if (agents.length === 0) {
      return {
        action: 'solve_directly',
        reason: `${decision.reason} Delegation was skipped because every grounded child plan lacked an executable, automatically authorized tool path.`,
      };
    }
    if (agents.length === decision.agents.length) return decision;

    const team = agents.length > 1 && decision.team
      ? {
        ...decision.team,
        executionPolicy: decision.team.executionPolicy
          ? {
            ...decision.team.executionPolicy,
            minimumSuccessfulMembers: Math.min(
              agents.length,
              decision.team.executionPolicy.minimumSuccessfulMembers ?? 1
            ),
          }
          : undefined,
      }
      : undefined;
    return {
      ...decision,
      agents,
      coordination: agents.length > 1 ? decision.coordination : 'independent',
      team,
    };
  }

  private hasExecutableGroundingPath(task: string, bindings: ToolBinding[]): boolean {
    const names = new Set(bindings.filter(binding => binding.enabled).map(binding => binding.name));
    if (this.taskNeedsWebAccess(task)) {
      return names.has('web.search')
        && (!this.taskRequiresFetchedWebEvidence(task) || names.has('web.fetch'));
    }
    if (this.taskRequiresWorkspaceMutation(task)) {
      return names.has('fs.write') || names.has('fs.synthesize') || names.has('shell.exec');
    }
    if (/\b(?:run|execute)\b[\s\S]{0,100}\b(?:tests?|build|checks?|commands?)\b|(?:运行|执行)[\s\S]{0,80}(?:测试|构建|检查|命令)/i.test(task)) {
      return names.has('shell.exec');
    }
    if (/\b(?:project|files?|filesystem|repository|repo|codebase|workspace|source|manifest)\b|(?:项目|文件|仓库|代码库|工作区|源码)/i.test(task)) {
      return names.has('fs.list') || names.has('fs.read') || names.has('shell.exec');
    }
    return names.size > 0;
  }

  private async runRootSoloReasoning(userInput: string, correlationId: string): Promise<string> {
    const ctx = this.getContext();
    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('thinking');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'thinking', correlationId } });
    const grounding = await this.runGroundingCheck('root', userInput, { correlationId, archetype: 'custom' });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.solo_reasoning', correlationId } });
    const response = await this.completeAsRoot(this.buildGroundedTask(userInput, grounding), 'root.solo_reasoning', correlationId);
    const groundedResponse = await this.enforceRootEvidenceBoundary(
      response,
      userInput,
      [grounding.evidence],
      correlationId,
      'root.solo_reasoning'
    );
    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });
    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle', correlationId } });
    this.emit({ type: 'root.solo.completed', agentId: 'root', data: { correlationId, totalTokens: usageDelta.totalTokens } });
    if (groundedResponse.trim()) return groundedResponse;
    this.emit({
      type: 'root.completion.fallback',
      agentId: 'root',
      correlationId,
      data: { purpose: 'root.solo_reasoning', reason: 'model_returned_empty_visible_output' },
    });
    return '[runtime_root_completion_fallback]\nRoy could not produce visible output for this turn. No stale response was reused.';
  }

  private inferExplicitRootOutputContract(task: string): {
    markers: string[];
    instruction: string;
  } | undefined {
    const outputCue = /\b(?:end|finish|return|respond|output|provide|emit|containing)\b/i;
    const instructions: string[] = [];
    const markers = new Set<string>();
    for (const line of task.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!outputCue.test(trimmed)) continue;
      const lineMarkers = [...trimmed.matchAll(/\b([A-Z][A-Z0-9_]{2,})\s*:/g)]
        .map(match => match[1]);
      if (lineMarkers.length === 0) continue;
      lineMarkers.forEach(marker => markers.add(marker));
      instructions.push(trimmed.slice(0, 1000));
    }
    if (markers.size === 0) return undefined;
    return {
      markers: [...markers],
      instruction: instructions.join('\n'),
    };
  }

  private responseSatisfiesExplicitOutputContract(
    response: string,
    contract: { markers: string[] }
  ): boolean {
    return contract.markers.every(marker => {
      const escaped = marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const match = new RegExp(`(?:^|\\n)\\s*${escaped}\\s*:\\s*`, 'gi');
      const matches = [...response.matchAll(match)];
      const last = matches.at(-1);
      if (!last || last.index === undefined) return false;
      const value = response.slice(last.index + last[0].length).trim();
      return Boolean(value) && !/^<[^>]+>$/.test(value.split(/\r?\n/, 1)[0].trim());
    });
  }

  private responseSelfDeclaresIncomplete(response: string): boolean {
    return /\b(?:does not|doesn't|did not|didn't|fails? to|unable to)\s+(?:fully\s+)?(?:satisfy|address|answer|include|incorporate|cover|complete)\b/i.test(response)
      || /\b(?:omitted?|missing|unmet|unaddressed|incomplete)\s+(?:requirements?|questions?|items?|parts?|details?)\b/i.test(response)
      || /(?:未能|没有|尚未|无法)(?:完整|全部)?(?:满足|回答|包括|纳入|覆盖|完成)|(?:遗漏|缺少|未满足|未覆盖)(?:的)?(?:要求|问题|项目|部分)/.test(response);
  }

  private acceptanceEvidenceSupportsAnswer(evidence: string, answer: string): boolean {
    const normalize = (value: string): string => value
      .normalize('NFKC')
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const normalizedEvidence = normalize(evidence);
    const normalizedAnswer = normalize(answer);
    return Boolean(normalizedAnswer)
      && ` ${normalizedEvidence} `.includes(` ${normalizedAnswer} `);
  }

  private shouldAuditRootResponse(response: string, userTask: string): boolean {
    if (this.workspaceRuntimeConfig?.delegation.rootSteps.requireAcceptanceAudit === false) {
      return false;
    }
    const threshold = Math.max(
      2,
      this.workspaceRuntimeConfig?.delegation.rootSteps.directDecisionAuditMinObligations ?? 4
    );
    const requiresEveryExplicitItem =
      /\b(?:every|each|all)\b[\s\S]{0,100}\b(?:question|requirement|item|answer|deliverable|constraint)s?\b/i.test(userTask)
      || /(?:每个|每一|所有|全部)[\s\S]{0,60}(?:问题|要求|项目|答案|交付物|约束)/.test(userTask);
    return this.responseSelfDeclaresIncomplete(response)
      || (
        requiresEveryExplicitItem
        && this.countIndependentTaskObligations(userTask) >= threshold
      );
  }

  private taskHasMultipleFactualQuestions(userTask: string): boolean {
    const threshold = Math.max(
      2,
      this.workspaceRuntimeConfig?.delegation.rootSteps.directDecisionAuditMinObligations ?? 4
    );
    return this.countIndependentTaskObligations(userTask) >= threshold
      && /\b(?:questions?|who|what|which|when|where|how many|in which)\b/i.test(userTask);
  }

  private async resolveRootResponseAcceptanceReferences(
    userTask: string,
    correlationId: string,
    supportingToolEvidence: string[]
  ): Promise<RootResponseAcceptanceReference[]> {
    if (!this.taskHasMultipleFactualQuestions(userTask)) return [];
    const evidenceCorpus = supportingToolEvidence
      .filter(item => item.trim())
      .join('\n\n')
      .slice(0, 32_000);
    this.emit({
      type: 'root.response.acceptance.references.started',
      agentId: 'root',
      correlationId,
      data: {
        independentObligations: this.countIndependentTaskObligations(userTask),
        candidateVisible: false,
        evidenceCharacters: evidenceCorpus.length,
      },
    });
    try {
      const raw = await this.completeJSONAsAgent<{ references?: unknown }>(
        this.getContext().agent,
        [
          {
            role: 'system',
            content: [
              'You are an independent factual requirement resolver.',
              'You cannot see any candidate response. Resolve each numbered factual question in the original task independently to avoid confirmation bias.',
              'The supplied runtime tool evidence is your only factual source. Do not use unsupported memory to fill a missing answer.',
              'If the evidence supports materially different scope interpretations or historical milestones, preserve all of them as accepted alternatives.',
              'Return the precise canonical answer plus common aliases, alternate titles, stage names, or equivalent forms that would unambiguously answer that exact question.',
              'Do not substitute a related entity. If the evidence is insufficient, return low confidence rather than inventing certainty.',
              'Return concise strict JSON only with a references array.',
            ].join('\n'),
          },
          {
            role: 'user',
            content: JSON.stringify({
              originalTask: userTask,
              runtimeToolEvidence: evidenceCorpus || 'No runtime tool evidence was available.',
              requiredSchema: {
                references: [{
                  requirement: 'the exact numbered factual question',
                  acceptedAnswers: ['canonical answer', 'unambiguous common alias'],
                  confidence: 'number from 0 to 1',
                }],
              },
            }),
          },
        ],
        { temperature: 0, maxTokens: 1_600 },
        'root.response_acceptance.reference_resolution',
        correlationId
      );
      const references = Array.isArray(raw.references)
        ? raw.references
          .filter((item): item is Record<string, unknown> =>
            Boolean(item) && typeof item === 'object'
          )
          .map(item => {
            const acceptedAnswers = Array.isArray(item.acceptedAnswers)
              ? item.acceptedAnswers
                .filter((answer): answer is string =>
                  typeof answer === 'string' && answer.trim().length >= 2
                )
                .map(answer => answer.trim().slice(0, 200))
                .slice(0, 12)
              : [];
            const evidenceGroundedAnswers = evidenceCorpus
              ? acceptedAnswers.filter(answer =>
                this.acceptanceEvidenceSupportsAnswer(evidenceCorpus, answer)
              )
              : acceptedAnswers;
            const evidenceGrounded = Boolean(
              evidenceCorpus
              && evidenceGroundedAnswers.length > 0
            );
            const reportedConfidence = typeof item.confidence === 'number'
              && Number.isFinite(item.confidence)
              ? Math.max(0, Math.min(1, item.confidence))
              : 0;
            return {
              requirement: typeof item.requirement === 'string'
              ? item.requirement.trim().slice(0, 800)
              : '',
              acceptedAnswers: evidenceGroundedAnswers,
              confidence: evidenceGrounded
                ? reportedConfidence
                : Math.min(reportedConfidence, 0.49),
              evidenceGrounded,
            };
          })
          .filter(item => item.requirement && item.acceptedAnswers.length > 0)
          .slice(0, 24)
        : [];
      this.emit({
        type: 'root.response.acceptance.references.completed',
        agentId: 'root',
        correlationId,
        data: {
          references: references.length,
          highConfidenceReferences: references.filter(item => item.confidence >= 0.65).length,
          evidenceGroundedReferences: references.filter(item => item.evidenceGrounded).length,
          candidateVisible: false,
        },
      });
      return references;
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'root.response.acceptance.references.failed',
        agentId: 'root',
        correlationId,
        data: { error: error instanceof Error ? error.message : String(error) },
      });
      return [];
    }
  }

  private async auditRootResponseAcceptance(
    response: string,
    userTask: string,
    correlationId: string,
    phase: 'candidate' | 'repaired',
    references: RootResponseAcceptanceReference[]
  ): Promise<RootResponseAcceptanceAudit> {
    const raw = await this.completeJSONAsAgent<{
      complete?: unknown;
      unmetRequirements?: unknown;
      reason?: unknown;
      obligations?: unknown;
    }>(
      this.getContext().agent,
      [
        {
          role: 'system',
          content: [
            'You are Roy\'s final-response acceptance auditor.',
            'Compare the candidate only against explicit, user-visible requirements in the original task.',
            'Treat every numbered or bulleted question, deliverable, constraint, and output contract as a separate obligation.',
            'For every obligation, quote the exact candidate text that satisfies it. Missing quoted evidence means the obligation is unmet.',
            'When an obligation asks for a factual answer, independently check whether the quoted answer actually answers the question; flag contradictions or uncertain substitutions instead of checking only keyword presence.',
            'A blind reference resolver ran without seeing the candidate. Treat its high-confidence acceptedAnswers as the comparison baseline and explicitly flag candidate conflicts.',
            'Judge objective coverage and factual consistency, not subjective writing style or any hidden evaluator.',
            'Do not expose chain-of-thought. Return concise strict JSON only.',
          ].join('\n'),
        },
        {
          role: 'user',
          content: JSON.stringify({
            originalTask: userTask,
            candidateResponse: response,
            blindFactualReferences: references,
            requiredSchema: {
              complete: 'boolean: true only when every explicit obligation is satisfied',
              unmetRequirements: ['concise explicit requirements that remain unmet'],
              reason: 'one concise evidence-based explanation',
              obligations: [{
                requirement: 'the explicit numbered or bulleted obligation',
                satisfied: 'boolean',
                candidateEvidence: 'an exact concise quote from the candidate, or an empty string',
                factualConcern: 'a concise contradiction or uncertainty, omitted when none',
              }],
            },
          }),
        },
      ],
      { temperature: 0, maxTokens: 1_400 },
      `root.response_acceptance.${phase}`,
      correlationId
    );
    const obligations = Array.isArray(raw.obligations)
      ? raw.obligations
        .filter((item): item is Record<string, unknown> =>
          Boolean(item) && typeof item === 'object'
        )
        .map(item => ({
          requirement: typeof item.requirement === 'string'
            ? item.requirement.trim().slice(0, 800)
            : '',
          satisfied: item.satisfied === true,
          candidateEvidence: typeof item.candidateEvidence === 'string'
            ? item.candidateEvidence.trim().slice(0, 1_200)
            : '',
          factualConcern: typeof item.factualConcern === 'string'
            && item.factualConcern.trim()
            ? item.factualConcern.trim().slice(0, 800)
            : undefined,
        }))
        .filter(item => item.requirement)
        .slice(0, 24)
      : [];
    const expectedObligations = this.countIndependentTaskObligations(userTask);
    const structuredUnmet = obligations
      .filter(item =>
        !item.satisfied
        || !item.candidateEvidence
        || Boolean(item.factualConcern)
      )
      .map(item => [
        item.requirement,
        item.factualConcern,
        !item.candidateEvidence ? 'No exact candidate evidence was supplied.' : '',
      ].filter(Boolean).join(' — '));
    const unmetRequirements = Array.isArray(raw.unmetRequirements)
      ? raw.unmetRequirements
        .filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
        .map(item => item.trim().slice(0, 800))
        .slice(0, 24)
      : [];
    unmetRequirements.push(...structuredUnmet);
    if (expectedObligations >= 2 && obligations.length < expectedObligations) {
      unmetRequirements.push(
        `Acceptance evidence covered ${obligations.length} of ${expectedObligations} explicit obligations.`
      );
    }
    const normalizedResponse = response.toLocaleLowerCase();
    for (const reference of references.filter(item => item.confidence >= 0.65)) {
      const matched = reference.acceptedAnswers.some(answer =>
        normalizedResponse.includes(answer.toLocaleLowerCase())
      );
      if (!matched) {
        unmetRequirements.push(
          `${reference.requirement} — candidate contains none of the blind reference answers: ${reference.acceptedAnswers.join(' / ')}`
        );
      }
    }
    const normalizedUnmet = [...new Set(unmetRequirements)].slice(0, 24);
    const complete = raw.complete === true
      && normalizedUnmet.length === 0
      && (expectedObligations < 2 || obligations.length >= expectedObligations);
    return {
      complete,
      unmetRequirements: normalizedUnmet,
      reason: typeof raw.reason === 'string' && raw.reason.trim()
        ? raw.reason.trim().slice(0, 1_200)
        : complete
          ? 'Every explicit response obligation is covered.'
          : 'The acceptance auditor found incomplete explicit obligations.',
      obligations,
    };
  }

  private async enforceRootResponseAcceptance(
    response: string,
    userTask: string,
    correlationId: string,
    supportingResults: string[],
    supportingToolEvidence: string[] = []
  ): Promise<string> {
    if (!this.shouldAuditRootResponse(response, userTask)) return response;
    this.emit({
      type: 'root.response.acceptance.audit.started',
      agentId: 'root',
      correlationId,
      data: {
        candidateCharacters: response.length,
        independentObligations: this.countIndependentTaskObligations(userTask),
        selfDeclaredIncomplete: this.responseSelfDeclaresIncomplete(response),
      },
    });
    const references = await this.resolveRootResponseAcceptanceReferences(
      userTask,
      correlationId,
      supportingToolEvidence
    );
    let audit: RootResponseAcceptanceAudit;
    try {
      audit = await this.auditRootResponseAcceptance(
        response,
        userTask,
        correlationId,
        'candidate',
        references
      );
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      this.emit({
        type: 'root.response.acceptance.audit.failed',
        agentId: 'root',
        correlationId,
        data: { error: error instanceof Error ? error.message : String(error) },
      });
      return response;
    }
    if (audit.complete) {
      this.emit({
        type: 'root.response.acceptance.audit.completed',
        agentId: 'root',
        correlationId,
        data: {
          complete: true,
          reason: audit.reason,
          evaluatedObligations: audit.obligations.length,
        },
      });
      return response;
    }

    const tree = this.executionTrees.get(correlationId);
    const previousStepId = tree?.steps.at(-1)?.id;
    let repairStep: RootExecutionStep | undefined;
    try {
      repairStep = await this.startRootExecutionStep(correlationId, {
        action: 'finalize',
        reason: 'Final-response acceptance found unmet explicit requirements; repair only the missing obligations using accumulated evidence.',
        agentCount: 0,
      }, previousStepId ? [previousStepId] : []);
    } catch (error) {
      this.emit({
        type: 'root.response.acceptance.repair.untracked',
        agentId: 'root',
        correlationId,
        data: { reason: error instanceof Error ? error.message : String(error) },
      });
    }
    this.emit({
      type: 'root.response.acceptance.unmet',
      agentId: 'root',
      correlationId,
      data: {
        stepId: repairStep?.id,
        unmetRequirements: audit.unmetRequirements,
        reason: audit.reason,
      },
    });
    this.emit({
      type: 'root.response.acceptance.repair.started',
      agentId: 'root',
      correlationId,
      data: {
        stepId: repairStep?.id,
        unmetRequirements: audit.unmetRequirements,
      },
    });

    const supportingEvidence = supportingResults
      .filter(result => result.trim())
      .map((result, index) => `<delegated_result index="${index + 1}">\n${result}\n</delegated_result>`)
      .join('\n\n')
      .slice(0, 32_000);
    const authoritativeToolEvidence = supportingToolEvidence
      .filter(result => result.trim())
      .map((result, index) => `<tool_evidence index="${index + 1}">\n${result}\n</tool_evidence>`)
      .join('\n\n')
      .slice(0, 32_000);
    const usageBefore = this.getContext().agent.getUsage();
    try {
      const repaired = await this.completeAsRoot(
        [
          'Repair the candidate into one complete final response to the original user task.',
          'Close every listed unmet requirement. Preserve correct completed content and the requested output format.',
          'Acceptance feedback identifies coverage gaps; it is not itself a factual source.',
          'Use runtime tool evidence as the factual authority. Never replace a candidate fact with an answer absent from that evidence.',
          'Use delegated results only as secondary supporting reports. Do not mention this audit or expose chain-of-thought.',
          `<original_task>\n${userTask}\n</original_task>`,
          `<acceptance_feedback>\n${JSON.stringify(audit, null, 2)}\n</acceptance_feedback>`,
          `<candidate_response>\n${response}\n</candidate_response>`,
          supportingEvidence
            ? `<supporting_evidence>\n${supportingEvidence}\n</supporting_evidence>`
            : '',
          authoritativeToolEvidence
            ? `<authoritative_tool_evidence>\n${authoritativeToolEvidence}\n</authoritative_tool_evidence>`
            : '',
        ].filter(Boolean).join('\n\n'),
        'root.response_acceptance.repair',
        correlationId
      );
      const usageDelta = this.usageDifference(
        usageBefore,
        this.getContext().agent.getUsage()
      );
      this.recordTurnUsage(usageDelta);
      this.emit({
        type: 'budget.updated',
        agentId: 'root',
        correlationId,
        data: { purpose: 'root.response_acceptance.repair', ...usageDelta },
      });
      const candidate = repaired.trim() || response;
      const repairedAudit = await this.auditRootResponseAcceptance(
        candidate,
        userTask,
        correlationId,
        'repaired',
        references
      );
      this.emit({
        type: repairedAudit.complete
          ? 'root.response.acceptance.repair.completed'
          : 'root.response.acceptance.repair.unmet',
        agentId: 'root',
        correlationId,
        data: {
          stepId: repairStep?.id,
          complete: repairedAudit.complete,
          unmetRequirements: repairedAudit.unmetRequirements,
          reason: repairedAudit.reason,
          repairedCharacters: candidate.length,
          retainedOriginal: !repairedAudit.complete,
        },
      });
      if (repairStep) {
        await this.completeRootExecutionStep(correlationId, repairStep, {
          resultSummary: candidate,
        });
      }
      return repairedAudit.complete ? candidate : response;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.emit({
        type: 'root.response.acceptance.repair.failed',
        agentId: 'root',
        correlationId,
        data: { stepId: repairStep?.id, error: message },
      });
      if (repairStep) {
        await this.failRootExecutionStep(correlationId, repairStep, message, false);
      }
      this.rethrowRetryableLLMTransportError(error);
      return response;
    }
  }

  private async enforceExplicitRootOutputContract(
    response: string,
    userTask: string,
    correlationId: string
  ): Promise<string> {
    const contract = this.inferExplicitRootOutputContract(userTask);
    if (!contract || this.responseSatisfiesExplicitOutputContract(response, contract)) {
      return response;
    }
    const tree = this.executionTrees.get(correlationId);
    const previousStepId = tree?.steps.at(-1)?.id;
    let repairStep: RootExecutionStep | undefined;
    try {
      repairStep = await this.startRootExecutionStep(correlationId, {
        action: 'finalize',
        reason: 'The candidate response did not satisfy the user-declared output contract; repair formatting without adding claims.',
        agentCount: 0,
      }, previousStepId ? [previousStepId] : []);
    } catch (error) {
      this.emit({
        type: 'root.output_contract.repair.untracked',
        agentId: 'root',
        correlationId,
        data: { reason: error instanceof Error ? error.message : String(error) },
      });
    }
    this.emit({
      type: 'root.output_contract.repair.started',
      agentId: 'root',
      correlationId,
      data: {
        stepId: repairStep?.id,
        markers: contract.markers,
        candidateCharacters: response.length,
      },
    });
    try {
      const repaired = await this.completeJSONAsAgent<{ finalResponse?: unknown }>(
        this.getContext().agent,
        [
          {
            role: 'system',
            content: [
              'You repair a root agent response so it obeys the user-declared output contract.',
              'Preserve the candidate answer and its claims. Do not solve a different task or add unsupported facts.',
              'Return JSON only with one string field named finalResponse.',
            ].join('\n'),
          },
          {
            role: 'user',
            content: JSON.stringify({
              originalTask: userTask,
              requiredMarkers: contract.markers,
              outputInstruction: contract.instruction,
              candidateResponse: response,
              requirement: 'The repaired response must contain every required marker at the beginning of a line with a non-placeholder value after the colon.',
            }),
          },
        ],
        { temperature: 0, maxTokens: 512 },
        'root.output_contract_repair',
        correlationId
      );
      const candidate = typeof repaired.finalResponse === 'string'
        ? repaired.finalResponse.trim()
        : '';
      if (!candidate || !this.responseSatisfiesExplicitOutputContract(candidate, contract)) {
        throw new Error('Structured repair did not satisfy the explicit output contract');
      }
      this.emit({
        type: 'root.output_contract.repair.completed',
        agentId: 'root',
        correlationId,
        data: {
          stepId: repairStep?.id,
          markers: contract.markers,
          repairedCharacters: candidate.length,
        },
      });
      if (repairStep) {
        await this.completeRootExecutionStep(correlationId, repairStep, {
          resultSummary: candidate,
        });
      }
      return candidate;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.emit({
        type: 'root.output_contract.repair.failed',
        agentId: 'root',
        correlationId,
        data: {
          stepId: repairStep?.id,
          markers: contract.markers,
          error: message,
        },
      });
      if (repairStep) {
        await this.failRootExecutionStep(correlationId, repairStep, message, false);
      }
      this.rethrowRetryableLLMTransportError(error);
      return response;
    }
  }

  private async collectRuntimeLLMStream(
    provider: LLMProvider,
    messages: LLMMessage[],
    options: LLMCompletionOptions,
    context: { actorId: string; purpose: string; correlationId: string; teamId?: string }
  ): Promise<{ content: string; usage?: ModelTokenUsage; finishReason?: string }> {
    const retryConfig = this.workspaceRuntimeConfig?.llm;
    const maxAttempts = Math.max(1, Math.floor(retryConfig?.streamMaxAttempts ?? 3));
    const initialDelayMs = Math.max(0, Math.floor(retryConfig?.retryInitialDelayMs ?? 250));
    const maxDelayMs = Math.max(initialDelayMs, Math.floor(retryConfig?.retryMaxDelayMs ?? 2_000));
    let discardedPartialCharacters = 0;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const chunks: string[] = [];
      let usage: ModelTokenUsage | undefined;
      let finishReason: string | undefined;
      try {
        for await (const chunk of provider.stream(messages, options)) {
          if (chunk.content) chunks.push(chunk.content);
          if (chunk.usage) usage = chunk.usage;
          if (chunk.finishReason) finishReason = chunk.finishReason;
        }
        if (attempt > 1) {
          this.emit({
            type: 'llm.stream.recovered',
            agentId: context.actorId,
            correlationId: context.correlationId,
            data: { ...context, attempt, discardedPartialCharacters },
          });
        }
        return { content: chunks.join(''), usage, finishReason };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const retryable = this.isRetryableLLMStreamError(error);
        const willRetry = retryable && attempt < maxAttempts;
        const discardedThisAttempt = chunks.join('').length;
        discardedPartialCharacters += discardedThisAttempt;
        this.emit({
          type: willRetry ? 'llm.stream.retrying' : 'llm.stream.failed',
          agentId: context.actorId,
          correlationId: context.correlationId,
          data: {
            ...context,
            attempt,
            maxAttempts,
            error: message,
            retryable,
            discardedPartialCharacters: discardedThisAttempt,
          },
        });
        if (!willRetry) throw error;
        const delayMs = Math.min(maxDelayMs, initialDelayMs * (2 ** (attempt - 1)));
        if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    throw new Error('LLM stream exhausted without a result');
  }

  private isRetryableLLMStreamError(error: unknown): boolean {
    const value = error as { code?: unknown; status?: unknown; message?: unknown } | undefined;
    const code = typeof value?.code === 'string' ? value.code.toUpperCase() : '';
    const status = typeof value?.status === 'number' ? value.status : undefined;
    const message = String(value?.message ?? error ?? '').toLowerCase();
    if (['ECONNRESET', 'ECONNREFUSED', 'EPIPE', 'ETIMEDOUT', 'UND_ERR_SOCKET', 'UND_ERR_CONNECT_TIMEOUT'].includes(code)) {
      return true;
    }
    if (status !== undefined && [408, 409, 425, 429, 500, 502, 503, 504].includes(status)) return true;
    return /premature close|socket hang up|connection (?:error|(?:was )?(?:closed|reset))|stream (?:was )?(?:closed|terminated)|fetch failed|network error|timed? ?out|temporarily unavailable|service unavailable/.test(message);
  }

  private rethrowRetryableLLMTransportError(error: unknown): void {
    if (this.isRetryableLLMStreamError(error)) throw error;
  }

  private async completeAsRoot(prompt: string, purpose: string, correlationId: string): Promise<string> {
    const ctx = this.getContext();
    if (!ctx.llm) {
      const message = 'Error: LLM not configured';
      ctx.agent.recordRuntimeCompletion(message, {
        content: message,
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      });
      return message;
    }

    const communicationSummary = ctx.agent.getCommunicationContext()?.rendered;
    const systemContent = [
      'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      'You are not DeepSeek, Claude, OpenAI, Anthropic, or any model provider.',
      'The model provider is only your inference backend.',
      `Purpose: ${purpose}.`,
      `Correlation: ${correlationId}.`,
      communicationSummary && communicationSummary.length > 1200
        ? `${communicationSummary.slice(0, 1200)}\n...[communication context truncated]`
        : communicationSummary,
    ].join('\n');
    let effectivePrompt = prompt;
    const availableTokens = this.budgetMarket?.getState().availableTokens;
    if (availableTokens !== undefined && this.budgetAccountingDimension() === 'total_tokens') {
      const outputReserve = Math.min(512, Math.max(64, Math.floor(availableTokens * 0.2)));
      const systemEstimate = this.estimateTextTokens(systemContent);
      const promptBudget = Math.max(0, availableTokens - outputReserve - systemEstimate);
      const promptEstimate = this.estimateTextTokens(effectivePrompt);
      if (promptEstimate > promptBudget && promptBudget > 0) {
        const maxChars = Math.max(64, Math.floor(effectivePrompt.length * promptBudget / promptEstimate));
        effectivePrompt = `${effectivePrompt.slice(0, maxChars)}\n...[budget-constrained context truncation]`;
        this.emit({
          type: 'budget.context.truncated',
          agentId: 'root',
          correlationId,
          data: { purpose, originalTokens: promptEstimate, allowedTokens: promptBudget, outputReserve },
        });
      }
    }

    const initialMessages: LLMMessage[] = [
      { role: 'system', content: systemContent },
      { role: 'user', content: effectivePrompt },
    ];
    const continuationLimit = Math.max(
      1,
      Math.floor(this.workspaceRuntimeConfig?.llm.streamContinuationMaxSegments ?? 4)
    );
    const continuationContext = this.compactRootContinuationContext(effectivePrompt);
    let content = '';

    for (let segment = 1; segment <= continuationLimit; segment += 1) {
      const messages: LLMMessage[] = segment === 1
        ? initialMessages
        : [
          { role: 'system', content: systemContent },
          {
            role: 'user',
            content: [
              'Continue a response that the provider cut off only because its output window ended.',
              'Continue exactly where the draft stops. Do not restart, summarize, or repeat completed text.',
              'Finish every requirement from the original request and preserve its output contract.',
              `<original_context>\n${continuationContext}\n</original_context>`,
            ].join('\n\n'),
          },
          {
            role: 'assistant',
            content: content.slice(-16_000),
          },
          {
            role: 'user',
            content: 'Continue from the final character above and complete the response. Return continuation text only.',
          },
        ];
      const estimatedInputTokens = this.estimateTextTokens(messages.map(message => message.content).join('\n'));
      const visibleCompletionTokens = this.rootVisibleCompletionTokenBudget(purpose);
      const completionTokens = this.reasoningAwareCompletionTokenBudget(visibleCompletionTokens);
      let allocation: BudgetAllocation | undefined;
      const usageBefore = ctx.agent.getUsage();
      try {
        allocation = await this.requestAgentBudget({
          parentId: 'root',
          requesterId: 'root',
          archetype: 'custom',
          correlationId,
          requestedTokens: Math.max(
            this.budgetRequestTokens(estimatedInputTokens, completionTokens),
            this.workspaceRuntimeConfig?.budgetMarket.defaultRequestsByArchetype.root ?? 2400
          ),
          minimumTokens: this.budgetMinimumTokens(estimatedInputTokens),
          priority: purpose.includes('synthesis') ? 'high' : 'medium',
          expectedUtility: purpose.includes('synthesis') ? 0.9 : 0.78,
          purpose: segment === 1 ? purpose : `${purpose}.continuation`,
        });
        if (allocation?.status === 'denied') {
          throw new Error(`Root completion rejected by budget market: ${allocation.reason}`);
        }
        const maxTokens = allocation?.status === 'granted'
          ? Math.max(1, this.completionCapacity(allocation.allocatedTokens, estimatedInputTokens))
          : completionTokens;
        const completion = await this.collectRuntimeLLMStream(
          ctx.llm,
          messages,
          { temperature: 0.2, maxTokens },
          {
            actorId: 'root',
            purpose: segment === 1 ? purpose : `${purpose}.continuation`,
            correlationId,
          }
        );
        content = this.mergeCompletionContinuation(content, completion.content);
        ctx.agent.recordRuntimeCompletion(completion.content, {
          content: completion.content,
          usage: completion.usage ?? this.estimateModelUsage(messages, completion.content),
        });
        this.settleDirectBudget(
          'root',
          allocation,
          this.usageDifference(usageBefore, ctx.agent.getUsage()),
          correlationId
        );
        if (!this.isTruncatedFinishReason(completion.finishReason)) {
          if (segment > 1) {
            this.emit({
              type: 'llm.stream.continuation.completed',
              agentId: 'root',
              correlationId,
              data: {
                purpose,
                segments: segment,
                characters: content.length,
                finishReason: completion.finishReason,
              },
            });
          }
          return content;
        }
        this.emit({
          type: 'llm.stream.truncated',
          agentId: 'root',
          correlationId,
          data: {
            purpose,
            finishReason: completion.finishReason,
            characters: content.length,
            maxTokens,
            segment,
            continuationLimit,
          },
        });
        if (segment < continuationLimit) {
          this.emit({
            type: 'llm.stream.continuation.started',
            agentId: 'root',
            correlationId,
            data: {
              purpose,
              completedSegments: segment,
              characters: content.length,
            },
          });
        }
      } catch (error) {
        if (allocation?.status === 'granted') {
          this.budgetMarket?.release(allocation.id, 'root_completion_failed');
        }
        if (segment === 1 || !content) throw error;
        this.emit({
          type: 'llm.stream.continuation.failed',
          agentId: 'root',
          correlationId,
          data: {
            purpose,
            segment,
            characters: content.length,
            error: error instanceof Error ? error.message : String(error),
          },
        });
        return content;
      }
    }
    this.emit({
      type: 'llm.stream.continuation.exhausted',
      agentId: 'root',
      correlationId,
      data: { purpose, segments: continuationLimit, characters: content.length },
    });
    return content;
  }

  private async completeJSONAsAgent<T>(
    agent: BaseAgent,
    messages: LLMMessage[],
    options: LLMCompletionOptions,
    purpose: string,
    correlationId: string
  ): Promise<T> {
    const ctx = this.getContext();
    if (!ctx.llm) throw new Error('LLM not configured');
    const usageBefore = agent.getUsage();
    const estimatedInput = this.estimateTextTokens(messages.map(message => `${message.role}:${message.content}`).join('\n'));
    const requestedCompletionTokens = this.reasoningAwareCompletionTokenBudget(options.maxTokens ?? 512);
    const ownsAllocation = agent.id === 'root' || !this.agentBudgetAllocations.has(agent.id);
    const allocation = ownsAllocation
      ? await this.requestAgentBudget({
        parentId: agent.getIdentity().parentId ?? agent.id,
        requesterId: agent.id,
        archetype: agent.id === 'root' ? 'custom' : this.inferAgentArchetype(agent.getInfo()),
        correlationId,
        requestedTokens: this.budgetRequestTokens(estimatedInput, requestedCompletionTokens),
        minimumTokens: this.budgetMinimumTokens(estimatedInput),
        priority: purpose.includes('delegation') ? 'high' : 'medium',
        expectedUtility: 0.82,
        purpose,
      })
      : undefined;
    if (allocation?.status === 'denied') throw new Error(`JSON completion rejected by budget market: ${allocation.reason}`);

    const activeAllocationId = this.agentBudgetAllocations.get(agent.id);
    const activeAllocation = activeAllocationId ? this.budgetMarket?.getAllocation(activeAllocationId) : undefined;
    const effectiveAllocation = allocation ?? activeAllocation;
    const allocationRemaining = effectiveAllocation?.status === 'granted'
      && !(this.hasUnlimitedBudgetSupply() && !ownsAllocation)
      ? Math.max(0, effectiveAllocation.allocatedTokens - effectiveAllocation.consumedTokens)
      : undefined;
    const agentRemaining = agent.getCompletionTokenLimit();
    const availableTokens = [allocationRemaining, agentRemaining]
      .filter((value): value is number => value !== undefined)
      .reduce<number | undefined>(
        (minimum, value) => minimum === undefined ? value : Math.min(minimum, value),
        undefined
      );
    if (availableTokens !== undefined && availableTokens <= 0) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'json_completion_allocation_exhausted');
      }
      throw new Error('JSON completion rejected: active allocation is exhausted');
    }
    if (this.budgetAccountingDimension() === 'total_tokens'
      && availableTokens !== undefined
      && availableTokens <= estimatedInput) {
      if (ownsAllocation && allocation?.status === 'granted') {
        this.budgetMarket?.release(allocation.id, 'json_completion_input_exceeds_allocation');
      }
      throw new Error(`JSON completion rejected: estimated input ${estimatedInput} exceeds remaining allocation ${availableTokens}`);
    }
    const boundedOptions: LLMCompletionOptions = {
      ...options,
      maxTokens: availableTokens === undefined
        ? requestedCompletionTokens
        : Math.max(1, Math.min(
          requestedCompletionTokens,
          this.completionCapacity(availableTokens, estimatedInput)
        )),
    };

    try {
      const maxAttempts = Math.max(
        1,
        Math.floor(this.workspaceRuntimeConfig?.llm.jsonMaxAttempts ?? 2)
      );
      const allowStructuredContentRetry = this.getBudgetState().mode !== 'limited';
      let value: T | undefined;
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        const attemptMessages = attempt === 1 ? messages : [
          ...messages,
          {
            role: 'user' as const,
            content: 'The previous response was incomplete or invalid. Return one complete, concise JSON object only. Do not include analysis, markdown, or prose outside the JSON object.',
          },
        ];
        try {
          if (ctx.llm.completeJSONWithUsage) {
            const result = await ctx.llm.completeJSONWithUsage<T>(attemptMessages, boundedOptions);
            value = result.value;
            agent.recordRuntimeUsage(result.completion);
          } else {
            value = await ctx.llm.completeJSON<T>(attemptMessages, boundedOptions);
            const output = JSON.stringify(value);
            agent.recordRuntimeUsage({
              content: output,
              usage: this.estimateModelUsage(attemptMessages, output),
            });
          }
          if (attempt > 1) {
            this.emit({
              type: 'llm.json.recovered',
              agentId: agent.id,
              correlationId,
              data: { purpose, attempt },
            });
          }
          break;
        } catch (error) {
          const retryable = this.isRetryableLLMStreamError(error)
            || (
              allowStructuredContentRetry
              && this.isRetryableJSONCompletionError(error)
            );
          const willRetry = retryable && attempt < maxAttempts;
          this.emit({
            type: willRetry ? 'llm.json.retrying' : 'llm.json.failed',
            agentId: agent.id,
            correlationId,
            data: {
              purpose,
              attempt,
              maxAttempts,
              retryable,
              error: (error instanceof Error ? error.message : String(error)).slice(0, 1000),
            },
          });
          if (!willRetry) throw error;
          const retryConfig = this.workspaceRuntimeConfig?.llm;
          const initialDelayMs = Math.max(0, retryConfig?.retryInitialDelayMs ?? 250);
          const maxDelayMs = Math.max(initialDelayMs, retryConfig?.retryMaxDelayMs ?? 2_000);
          const delayMs = Math.min(maxDelayMs, initialDelayMs * (2 ** (attempt - 1)));
          if (delayMs > 0) await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
      if (value === undefined) throw new Error('JSON completion exhausted without a value');
      const usage = this.usageDifference(usageBefore, agent.getUsage());
      if (ownsAllocation) {
        this.settleDirectBudget(agent.id, allocation, usage, correlationId);
      } else {
        this.consumeActiveAgentBudget(agent.id, usage, correlationId, purpose);
      }
      if (agent.id === 'root') {
        this.recordTurnUsage(usage);
        this.emit({ type: 'budget.updated', agentId: agent.id, correlationId, data: { purpose, ...usage } });
      }
      return value;
    } catch (error) {
      if (allocation?.status === 'granted') this.budgetMarket?.release(allocation.id, 'json_completion_failed');
      throw error;
    }
  }

  private isRetryableJSONCompletionError(error: unknown): boolean {
    const message = error instanceof Error ? error.message : String(error);
    return message.startsWith('Failed to parse JSON response:')
      || message === 'Empty JSON response'
      || this.isRetryableLLMStreamError(error);
  }

  private async synthesizeDelegatedResults(
    userTask: string,
    results: RootMediatedSpawnResult[],
    correlationId: string,
    teamResults: TeamRunResult[] = [],
    rootExecution?: GroundingRunResult
  ): Promise<string> {
    const executionClosure = rootExecution
      ? this.analyzeWorkspaceExecutionClosure(
        rootExecution.toolCalls,
        rootExecution.acceptanceAudit,
        this.taskRequiresAcceptanceAudit(userTask)
      )
      : undefined;
    const verifierDrivenWorkspaceRun = Boolean(
      rootExecution
      && (
        /\.roy\/official-verifier\//i.test(userTask)
        || /<official_verifier_feedback>/i.test(userTask)
        || /\b(?:harbor|continue_until_timeout)\b/i.test(userTask)
      )
    );
    const failedDirectExecution = Boolean(
      rootExecution
      && results.length === 0
      && teamResults.length === 0
      && executionClosure
      && !executionClosure.closed
    );
    if (verifierDrivenWorkspaceRun || failedDirectExecution) {
      const finalResponse = this.buildDeterministicExecutionHandoff(
        rootExecution!,
        executionClosure!
      );
      this.emit({
        type: 'root.synthesis.skipped',
        agentId: 'root',
        correlationId,
        data: {
          reason: verifierDrivenWorkspaceRun
            ? 'verifier_driven_workspace_state_is_authoritative'
            : 'failed_direct_execution_requires_compact_handoff',
          ...executionClosure,
        },
      });
      return finalResponse;
    }
    const ctx = this.getContext();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'root.synthesis',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'root',
      correlationId,
      payload: {
        userTask,
        subagentIds: results.map(result => result.agent.identity.id),
        teamIds: teamResults.map(result => result.team.identity.id),
      },
      metadata: { agentId: 'root', tomLevel: ctx.agent.getIdentity().tomProfile.level },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('synthesizing');
    this.emit({
      type: 'root.synthesis.started',
      agentId: 'root',
      data: { correlationId, subagentIds: results.map(result => result.agent.identity.id) },
    });
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.multi_agent_synthesis', correlationId } });
    let response = '';
    try {
      response = await this.completeAsRoot(
        this.buildMultiAgentSynthesisPrompt(userTask, results, teamResults, rootExecution),
        'root.multi_agent_synthesis',
        correlationId
      );
    } catch (error) {
      this.emit({
        type: 'root.synthesis.recovered',
        agentId: 'root',
        correlationId,
        data: {
          reason: error instanceof Error ? error.message : String(error),
          recovery: 'deterministic_delegated_result',
          teamCount: teamResults.length,
          subagentCount: results.length,
        },
      });
    }
    let finalResponse = response.trim()
      ? response
      : this.buildRootSynthesisFallback(
        userTask,
        teamResults.map(result => result.result).filter(Boolean),
        results.map(result => result.subagentResult.result).filter(Boolean)
      );
    if (!response.trim()) {
      this.emit({
        type: 'root.synthesis.fallback',
        agentId: 'root',
        correlationId,
        data: {
          reason: 'model_returned_empty_visible_output',
          teamCount: teamResults.length,
          subagentCount: results.length,
        },
      });
    }
    finalResponse = await this.enforceRootEvidenceBoundary(
      finalResponse,
      userTask,
      [
        ...results.map(result => result.subagentResult.evidence),
        ...(rootExecution ? [rootExecution.evidence] : []),
      ],
      correlationId,
      'root.multi_agent_synthesis'
    );
    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });
    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle', correlationId } });
    this.emit({
      type: 'root.synthesis.completed',
      agentId: 'root',
      data: { correlationId, totalTokens: usageDelta.totalTokens, subagentCount: results.length },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return finalResponse;
  }

  private buildDeterministicExecutionHandoff(
    execution: GroundingRunResult,
    closure: WorkspaceExecutionClosureStatus
  ): string {
    const latestVerification = [...execution.toolCalls].reverse().find(call =>
      isWorkspaceVerificationCall(call)
    );
    const latestMutation = [...execution.toolCalls].reverse().find(call =>
      isSuccessfulWorkspaceMutationCall(call)
    );
    const latestFailure = [
      latestVerification?.error,
      latestVerification?.result
        ? JSON.stringify(latestVerification.result)
        : undefined,
      execution.warnings.at(-1),
    ].filter(Boolean).join('\n').slice(-4_000);
    return [
      '[runtime_execution_handoff]',
      closure.closed
        ? 'The workspace mutation, fresh verification, and required acceptance audit are complete.'
        : 'The workspace execution remains open and has been checkpointed for focused continuation.',
      `closure=${JSON.stringify(closure)}`,
      latestMutation
        ? `latest_mutation=${latestMutation.toolName} ${JSON.stringify(latestMutation.params).slice(0, 1200)}`
        : 'latest_mutation=none',
      latestVerification
        ? `latest_verification=${latestVerification.toolName} ${JSON.stringify(latestVerification.params).slice(0, 1200)}`
        : 'latest_verification=none',
      latestFailure ? `causal_frontier:\n${latestFailure}` : '',
      'The next continuation should reuse cached paths and repair only the newest unresolved verifier evidence.',
    ].filter(Boolean).join('\n\n');
  }

  private buildMultiAgentSynthesisPrompt(
    userTask: string,
    results: RootMediatedSpawnResult[],
    teamResults: TeamRunResult[] = [],
    rootExecution?: GroundingRunResult
  ): string {
    const reports = results.map(result => {
      const warnings = result.subagentResult.warnings.length > 0
        ? result.subagentResult.warnings.map(item => `- ${item}`).join('\n')
        : 'None';
      return `<subagent_report>
id: ${result.agent.identity.id}
name: ${result.agent.identity.name}
archetype: ${result.agent.identity.role}
tom: ToM-${result.agent.identity.tomProfile.level}
tom_profile: ${JSON.stringify(result.agent.identity.tomProfile)}
existence_reason: ${result.node.identity.existenceReason ?? 'bounded specialist contribution'}
cognitive_gaps: ${result.node.identity.cognitiveGapIds.join(', ') || 'none'}
tokens: ${result.subagentResult.usage.totalTokens}
grounded: ${result.subagentResult.grounded}
tool_grounded: ${result.subagentResult.evidence.toolGrounded}
output_grounded: ${result.subagentResult.evidence.outputGrounded}
tool_calls: ${result.subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
observed_paths:
${result.subagentResult.evidence.observedPaths.slice(0, 40).map(item => `- ${item}`).join('\n') || '- none'}
observed_urls:
${(result.subagentResult.evidence.observedUrls ?? []).slice(0, 20).map(item => `- ${item}`).join('\n') || '- none'}
task_relevant_observed_urls:
${(result.subagentResult.evidence.relevantObservedUrls ?? []).slice(0, 20).map(item => `- ${item}`).join('\n') || '- none'}
discovered_urls:
${(result.subagentResult.evidence.discoveredUrls ?? []).slice(0, 20).map(item => `- ${item}`).join('\n') || '- none'}
tool_result_summary:
${result.subagentResult.evidence.toolResultSummary?.slice(0, 6000) || 'none'}
warnings:
${warnings}
content:
${result.subagentResult.result}
</subagent_report>`;
    }).join('\n\n');

    const teamReports = teamResults.map(result => `<team_report>
id: ${result.team.identity.id}
name: ${result.team.identity.name}
tom: ToM-${result.team.identity.tomLevel}
tom_profile: ${JSON.stringify(result.team.identity.tomProfile)}
members: ${result.team.memberAgentIds.join(', ')}
tokens: ${result.usage.totalTokens}
content:
${result.result}
</team_report>`).join('\n\n');

    const executionReport = rootExecution
      ? `<root_execution_report>
mutation_applied: ${this.hasSuccessfulWorkspaceMutation(rootExecution.toolCalls)}
verification_ran: ${this.hasSuccessfulWorkspaceVerification(rootExecution.toolCalls)}
acceptance_audit_required: ${rootExecution.acceptanceAudit?.required === true}
acceptance_audit_passed: ${rootExecution.acceptanceAudit?.passed === true}
acceptance_items:
${rootExecution.acceptanceAudit?.items.map(item => `- ${item.id}: ${item.status} — ${item.evidence}`).join('\n') || '- none'}
tool_calls:
${rootExecution.toolCalls.map(call => `- ${call.toolName}: ${call.success ? 'success' : `failed (${call.error ?? 'unknown error'})`}`).join('\n') || '- none'}
runtime_evidence:
${rootExecution.evidence.toolResultSummary?.slice(0, 12_000) || 'none'}
warnings:
${rootExecution.warnings.map(item => `- ${item}`).join('\n') || '- none'}
</root_execution_report>`
      : '';

    return `The user requested:
<user_task>
${userTask}
</user_task>

Roy delegated this task to ${results.length} subagent(s). Synthesize their results into one final user-facing response.
Use concrete evidence from grounded reports. If a report is ungrounded or missing concrete tool output, say so and avoid overstating it.
Compare each agent's belief scope and perspective against the cognitive gaps it was created to fill. Preserve unresolved uncertainty instead of forcing agreement.
For web-grounded work, cite only observed_urls from the reports. Never introduce a URL or factual detail from model memory. Search-result discovered_urls are not opened evidence.
When a root_execution_report is present, treat its successful tool results as the authoritative final workspace state. Do not repeat an earlier claim that changes were only proposed when the root execution report proves they were applied. Never claim completion when mutation_applied is false or when acceptance_audit_required is true and acceptance_audit_passed is false.

${teamReports ? `The following subteam reports have already aggregated their direct members. Treat them as the primary delegation result.\n\n${teamReports}` : ''}

${reports}

${executionReport}

Produce the final response to the user as Roy, the root agent.`;
  }

  private buildRootSynthesisFallback(userTask: string, primaryResults: string[], secondaryResults: string[]): string {
    const selected = primaryResults.find(result => result.trim())
      ?? secondaryResults.find(result => result.trim());
    return [
      '[runtime_root_synthesis_fallback]',
      'Roy\'s synthesis backend returned no visible text. The runtime is returning the best completed delegated result without adding new claims.',
      `User task: ${userTask}`,
      selected ? `Delegated result:\n${selected.slice(0, 16000)}` : 'No non-empty delegated result was available.',
    ].join('\n\n');
  }

  private async synthesizeEvolutionResult(
    task: string,
    run: EvolutionRunResult,
    correlationId: string,
    rootExecution?: GroundingRunResult
  ): Promise<string> {
    const selected = run.selected;
    const execution = run.selectedExecution;
    const evaluation = run.selectedEvaluation;
    if (!execution) {
      return `Roy could not complete the evolutionary delegation run. Run ${run.id} produced no executable result.`;
    }
    const response = await this.completeAsRoot(
      [
        'Synthesize the selected evolutionary agent/team result into the final answer to the user.',
        'Do not expose hidden chain-of-thought. Explain observable evidence, limitations, and relevant disagreements.',
        `<user_task>${task}</user_task>`,
        `<evolution_run>${JSON.stringify({
          runId: run.id,
          profile: run.profile,
          candidateCount: run.metrics.candidateCount,
          selectedGenome: selected?.genome,
          evaluation,
          metrics: run.metrics,
        }, null, 2)}</evolution_run>`,
        `<selected_result>${execution.result}</selected_result>`,
        `<warnings>${execution.warnings.join('\n') || 'none'}</warnings>`,
        rootExecution
          ? `<root_execution_report>\n${this.summarizeRootExecutionClosure(rootExecution)}\n</root_execution_report>`
          : '',
      ].join('\n\n'),
      'root.evolution_synthesis',
      correlationId
    );
    if (response.trim()) return response;
    this.emit({
      type: 'root.synthesis.fallback',
      agentId: 'root',
      correlationId,
      data: { reason: 'model_returned_empty_visible_output', source: 'evolution', runId: run.id },
    });
    return this.buildRootSynthesisFallback(task, [execution.result], []);
  }

  private resolveEvolutionRunOptions(
    profileOverride?: EvolutionProfile,
    override?: RunEvolutionInput['options']
  ): EvolutionRunOptions {
    const configured = this.workspaceRuntimeConfig?.evolution;
    if (!configured) throw new Error('Evolution workspace configuration is unavailable');
    const profile = profileOverride ?? override?.profile ?? configured.profile;
    const profileAblations: Record<EvolutionProfile, EvolutionAblations> = {
      solo: {
        withoutSubagents: true, withoutToMProfile: true, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      fixed_subagents: {
        withoutSubagents: false, withoutToMProfile: true, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      tom_subteam: {
        withoutSubagents: false, withoutToMProfile: false, withoutBudgetMarket: true,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      budget_market: {
        withoutSubagents: false, withoutToMProfile: false, withoutBudgetMarket: false,
        withoutEvoMutation: true, withoutPatternMemory: true,
      },
      evo_team: { ...configured.ablations },
    };
    const ablations = { ...profileAblations[profile], ...override?.ablations };
    return {
      profile,
      populationSize: clampInteger(override?.populationSize ?? configured.populationSize, 1, 10),
      generations: ablations.withoutEvoMutation ? 0 : clampInteger(override?.generations ?? configured.generations, 0, 5),
      topK: clampInteger(override?.topK ?? configured.topK, 1, 5),
      maxExecutedCandidates: clampInteger(override?.maxExecutedCandidates ?? configured.maxExecutedCandidates, 1, 10),
      integrationMinimumScore: clamp01(override?.integrationMinimumScore ?? configured.integrationMinimumScore),
      patternSimilarityThreshold: clamp01(override?.patternSimilarityThreshold ?? configured.patternSimilarityThreshold),
      useLlmJudge: override?.useLlmJudge ?? configured.useLlmJudge,
      ablations,
    };
  }

  private validateEvolutionConfigPatch(
    patch: Partial<Omit<WorkspaceRuntimeConfig['evolution'], 'ablations'>> & {
      ablations?: Partial<WorkspaceRuntimeConfig['evolution']['ablations']>;
    }
  ): void {
    if (patch.enabled !== undefined && typeof patch.enabled !== 'boolean') throw new Error('Evolution enabled must be boolean');
    if (patch.mode !== undefined && patch.mode !== 'manual' && patch.mode !== 'auto') throw new Error('Evolution mode must be manual or auto');
    if (patch.profile !== undefined
      && !['solo', 'fixed_subagents', 'tom_subteam', 'budget_market', 'evo_team'].includes(patch.profile)) {
      throw new Error('Unsupported evolution profile');
    }
    for (const key of ['populationSize', 'generations', 'topK', 'maxExecutedCandidates'] as const) {
      const value = patch[key];
      if (value !== undefined && (!Number.isInteger(value) || value < (key === 'generations' ? 0 : 1))) {
        throw new Error(`Evolution ${key} must be ${key === 'generations' ? 'a non-negative' : 'a positive'} integer`);
      }
    }
    for (const key of ['integrationMinimumScore', 'patternSimilarityThreshold'] as const) {
      const value = patch[key];
      if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 1)) {
        throw new Error(`Evolution ${key} must be between 0 and 1`);
      }
    }
    if (patch.useLlmJudge !== undefined && typeof patch.useLlmJudge !== 'boolean') {
      throw new Error('Evolution useLlmJudge must be boolean');
    }
    if (patch.ablations) {
      for (const [key, value] of Object.entries(patch.ablations)) {
        if (!['withoutSubagents', 'withoutToMProfile', 'withoutBudgetMarket', 'withoutEvoMutation', 'withoutPatternMemory'].includes(key)
          || typeof value !== 'boolean') {
          throw new Error(`Invalid evolution ablation ${key}`);
        }
      }
    }
  }

  private async runSoloEvolutionBaseline(
    runId: string,
    correlationId: string,
    task: string,
    options: EvolutionRunOptions
  ): Promise<EvolutionRunResult> {
    const ctx = this.getContext();
    const startedAt = Date.now();
    if (options.ablations.withoutBudgetMarket) this.evolutionBudgetBypassCorrelations.add(correlationId);
    const usageBefore = ctx.agent.getUsage();
    this.emit({ type: 'evo.fsm.transition', agentId: 'root', correlationId, data: { runId, from: 'S_evo_idle', to: 'S_evo_execute', profile: 'solo' } });
    try {
      const result = await this.completeAsRoot(
        `Solve the following task directly as Roy without creating subagents or teams. Return a concrete answer and state limitations.\n\n${task}`,
        'evo.solo_baseline',
        correlationId
      );
      const usage = this.usageDifference(usageBefore, ctx.agent.getUsage());
      const completedAt = Date.now();
      const artifact: EvolutionExecutionArtifact = {
        candidateId: 'solo_root', actorKind: 'agent', actorId: 'root', success: Boolean(result.trim()), result,
        usage: this.tokenUsageToEvolutionUsage(usage), wallClockMs: completedAt - startedAt,
        agentIds: [], teamIds: [], toolCalls: 0, successfulToolCalls: 0, unresolvedToolIntents: 0,
        groundedResults: 0, totalResults: 1, failedActors: result.trim() ? 0 : 1,
        recoveredFailures: 0, warnings: [],
      };
      const answerQuality = result.trim() ? Math.min(1, 0.5 + Math.log10(result.length + 1) / 5) : 0;
      const run: EvolutionRunResult = {
        id: runId, correlationId, task, profile: 'solo', state: 'S_evo_done',
        candidates: [], executions: [artifact], evaluations: [], selected: undefined,
        selectedExecution: artifact, selectedEvaluation: undefined,
        metrics: {
          taskSuccess: Boolean(result.trim()), answerQuality, toolSuccessRate: 0,
          agentsSpawned: 0, teamsSpawned: 0, totalTokens: usage.totalTokens,
          thinkingTokens: usage.thinkingTokens, wallClockMs: completedAt - startedAt,
          budgetRequested: 0, budgetAllocated: 0, failureRecoveryCount: 0,
          candidateCount: 0, executedCandidateCount: 1, cacheHits: 0, mutationsApplied: 0,
        },
        ablations: options.ablations, startedAt, completedAt,
      };
      this.evolutionRuns.push(run);
      await ctx.memory.recordEvolutionRun(run as unknown as Record<string, unknown>);
      this.emit({ type: 'evo.run.completed', agentId: 'root', correlationId, data: { runId, profile: 'solo', metrics: run.metrics } });
      return structuredClone(run);
    } finally {
      this.evolutionBudgetBypassCorrelations.delete(correlationId);
    }
  }

  private async findRelevantEvolutionPatterns(task: string, threshold: number): Promise<EvolutionPattern[]> {
    const patterns = await this.getContext().memory.getEvolutionPatterns();
    const invalidPatternIds: string[] = [];
    const structurallyValid = patterns.filter(pattern => {
      try {
        validateTeamGenome(pattern.genome);
        return true;
      } catch {
        invalidPatternIds.push(pattern.id);
        return false;
      }
    });
    if (invalidPatternIds.length > 0) {
      await this.getContext().memory.deprecateEvolutionPatterns(invalidPatternIds);
      this.emit({
        type: 'evo.pattern.deprecated',
        agentId: 'root',
        data: { patternIds: invalidPatternIds, reason: 'genome_preflight_validation_failed' },
      });
    }
    const embeddings = new HashTaskEmbeddingProvider();
    return structurallyValid
      .map(pattern => ({ pattern, similarity: embeddings.similarity(task, pattern.taskSignature) }))
      .filter(item => item.pattern.status !== 'deprecated' && item.similarity >= threshold)
      .sort((left, right) => {
        const leftScore = left.similarity * 0.7 + left.pattern.averageScore * 0.3;
        const rightScore = right.similarity * 0.7 + right.pattern.averageScore * 0.3;
        return rightScore - leftScore;
      })
      .map(item => item.pattern);
  }

  private async createEvolutionSeeds(
    task: string,
    parentId: string,
    correlationId: string,
    options: EvolutionRunOptions
  ): Promise<EvolutionSeedAgent[]> {
    let decision: DelegationDecision;
    if (parentId === 'root') {
      decision = await this.decideDelegation(task, correlationId);
      decision = await this.selectDelegationCandidate(parentId, task, decision, correlationId, 'root');
    } else {
      const parent = this.getContext().manager.getAgentById(parentId)?.getInfo();
      if (!parent) throw new Error(`Evolution parent agent "${parentId}" not found`);
      decision = await this.decideAgentDelegation(parent, task, correlationId);
    }
    const plans = decision.action === 'spawn_subagents' && decision.agents.length > 0
      ? decision.agents
      : this.defaultEvolutionPlans(task);
    const limit = Math.max(1, Math.min(options.populationSize, this.workspaceRuntimeConfig?.teams.maxMembersPerTeam ?? 5));
    return plans.slice(0, limit).map(plan => this.normalizeEvolutionSeed({
      archetype: plan.archetype,
      name: plan.name,
      role: plan.existenceReason ?? plan.archetype,
      task: plan.task,
      tools: plan.tools,
      skills: plan.skills,
      budgetTokens: plan.budgetTokens,
      tomLevel: options.ablations.withoutToMProfile ? 0 : plan.tomLevel,
      perspective: options.ablations.withoutToMProfile ? undefined : plan.tomProfile?.perspective,
      groundingRequired: plan.archetype === 'researcher' || plan.archetype === 'tester',
    }));
  }

  private normalizeEvolutionSeed(seed: EvolutionSeedAgent): EvolutionSeedAgent {
    const tools = seed.tools && seed.tools.length > 0
      ? seed.tools
      : this.getDefaultToolBindings(seed.archetype).map(binding => binding.name);
    const skills = seed.skills && seed.skills.length > 0
      ? seed.skills
      : this.getDefaultSkillBindings(seed.archetype).map(binding => binding.name);
    return {
      ...seed,
      tools: [...new Set(tools)],
      skills: [...new Set(skills)],
      groundingRequired: seed.groundingRequired
        ?? (seed.archetype === 'researcher' || seed.archetype === 'tester'),
    };
  }

  private defaultEvolutionPlans(task: string): DelegationAgentPlan[] {
    const lower = task.toLowerCase();
    if (/\b(implement|code|fix|patch|refactor)\b/.test(lower)) {
      return [
        { archetype: 'planner', task: `Decompose the implementation and its dependencies: ${task}`, tomLevel: 1, existenceReason: 'implementation planning' },
        { archetype: 'coder', task: `Produce the bounded implementation analysis or change: ${task}`, tomLevel: 0, existenceReason: 'implementation capability' },
        { archetype: 'tester', task: `Verify behavior, regressions, and failure paths for: ${task}`, tomLevel: 0, existenceReason: 'verification capability' },
      ];
    }
    if (/\b(inspect|analy[sz]e|review|risk|architecture|project|repo|structure)\b/.test(lower)) {
      return [
        { archetype: 'researcher', task: `Collect grounded project evidence for: ${task}`, tomLevel: 0, existenceReason: 'missing project evidence' },
        { archetype: 'critic', task: `Stress-test claims, risks, and evidence gaps for: ${task}`, tomLevel: 2, existenceReason: 'missing failure-mode perspective' },
      ];
    }
    return [{ archetype: 'custom', name: 'TaskSpecialist', task, tomLevel: 1, existenceReason: 'bounded specialist capability not covered by a built-in role' }];
  }

  private async instantiateEvolutionCandidate(
    candidate: EvolutionCandidate,
    parentId: string,
    correlationId: string,
    options: EvolutionRunOptions
  ): Promise<{ kind: 'agent' | 'team'; actorId: string }> {
    const members = candidate.genome.members;
    if (members.length === 1) {
      const member = members[0];
      const agent = await this.spawnAgent({
        parentId,
        name: member.name,
        customRole: member.role,
        archetype: member.archetype,
        tomLevel: member.tomProfile.level,
        description: candidate.genome.purpose,
        task: member.task,
        tools: member.toolPolicy.map(tool => tool.name),
        skills: member.skills,
        budgetTokens: options.ablations.withoutBudgetMarket ? undefined : member.budgetPolicy.requestedTokens,
        systemPrompt: member.rolePrompt,
        outputContract: {
          format: member.outputContract.format,
          requiredFields: member.outputContract.requiredFields,
          groundingRequired: member.outputContract.groundingRequired,
        },
        correlationId,
        tomProfile: this.genomeToRuntimeToM(member.tomProfile, member.id, parentId, member.role, member.toolPolicy.map(tool => tool.name)),
        cacheHits: candidate.lineage.parentPatternIds,
        cognitiveGapIds: member.tomProfile.uncertainty.map((_, index) => `${member.id}_gap_${index + 1}`),
        existenceReason: candidate.rationale,
        lifecycle: { mode: 'retain_session' },
        lifecycleOrigin: 'evolution',
      });
      return { kind: 'agent', actorId: agent.identity.id };
    }
    const team = await this.spawnTeam({
      parentAgentId: parentId,
      name: candidate.genome.name,
      description: candidate.genome.purpose,
      task: candidate.genome.taskSignature,
      tomLevel: options.ablations.withoutToMProfile ? 0 : candidate.genome.tomLevel,
      correlationId,
      members: members.map((member, index) => ({
        archetype: member.archetype,
        name: member.name,
        role: member.role,
        task: member.task,
        tools: member.toolPolicy.map(tool => tool.name),
        skills: member.skills,
        budgetTokens: options.ablations.withoutBudgetMarket ? undefined : member.budgetPolicy.requestedTokens,
        tomLevel: member.tomProfile.level,
        tomProfile: this.genomeToRuntimeToM(member.tomProfile, member.id, parentId, member.role, member.toolPolicy.map(tool => tool.name)),
        systemPrompt: member.rolePrompt,
        existenceReason: candidate.rationale,
        lead: index === 0,
      })),
      executionPolicy: {
        mode: candidate.genome.coordinationPolicy === 'parallel' ? 'parallel' : 'sequential',
        failureMode: 'best_effort',
        maxConcurrency: Math.min(3, members.length),
        minimumSuccessfulMembers: 1,
      },
      lifecycle: { mode: 'retain_session' },
      lifecycleOrigin: 'evolution',
    });
    return { kind: 'team', actorId: team.identity.id };
  }

  private async executeEvolutionCandidate(
    candidate: EvolutionCandidate,
    actor: { kind: 'agent' | 'team'; actorId?: string },
    correlationId: string,
    details: Map<string, { agent?: RunAgentResult; team?: TeamRunResult }>
  ): Promise<EvolutionExecutionArtifact> {
    const startedAt = Date.now();
    try {
      if (!actor.actorId) throw new Error('Evolution candidate actor id is missing');
      if (actor.kind === 'agent') {
        const member = candidate.genome.members[0];
        const result = await this.runAgent(actor.actorId, member.task, {
          correlationId,
          archetype: member.archetype,
          disableRecursiveDelegation: true,
        });
        details.set(candidate.id, { agent: result });
        const groundingSatisfied = !member.outputContract.groundingRequired
          || (result.evidence.toolGrounded && result.evidence.outputGrounded);
        return {
          candidateId: candidate.id, actorKind: 'agent', actorId: actor.actorId,
          success: Boolean(result.result.trim()) && groundingSatisfied, result: result.result,
          usage: this.tokenUsageToEvolutionUsage(result.usage), wallClockMs: Date.now() - startedAt,
          agentIds: [actor.actorId], teamIds: [], toolCalls: result.toolCalls.length,
          successfulToolCalls: result.toolCalls.filter(call => call.success).length,
          unresolvedToolIntents: this.containsUnresolvedToolIntent(result.result) ? 1 : 0,
          groundedResults: groundingSatisfied ? 1 : 0, totalResults: 1,
          failedActors: result.agent.state === 'failed' ? 1 : 0, recoveredFailures: 0,
          warnings: [
            ...result.warnings,
            ...(!groundingSatisfied ? ['Candidate did not satisfy its grounding-required output contract.'] : []),
          ],
        };
      }
      const result = await this.runTeam(actor.actorId, candidate.genome.taskSignature, { correlationId });
      details.set(candidate.id, { team: result });
      const toolCalls = result.members.flatMap(member => member.toolCalls);
      const unresolvedToolIntents = result.members.filter(member => this.containsUnresolvedToolIntent(member.result)).length
        + (this.containsUnresolvedToolIntent(result.result) ? 1 : 0);
      const failedActors = result.memberOutcomes.filter(outcome => outcome.status === 'failed').length;
      const groundingRequired = candidate.genome.members.some(member => member.outputContract.groundingRequired);
      const groundedResults = result.members.filter(member => (
        member.evidence.outputGrounded && (!groundingRequired || member.evidence.toolGrounded)
      )).length;
      const groundingSatisfied = !groundingRequired || groundedResults > 0;
      return {
        candidateId: candidate.id, actorKind: 'team', actorId: actor.actorId,
        success: result.team.status === 'done' && Boolean(result.result.trim()) && groundingSatisfied, result: result.result,
        usage: this.tokenUsageToEvolutionUsage(result.usage), wallClockMs: Date.now() - startedAt,
        agentIds: result.members.map(member => member.agent.identity.id), teamIds: [actor.actorId],
        toolCalls: toolCalls.length, successfulToolCalls: toolCalls.filter(call => call.success).length,
        unresolvedToolIntents,
        groundedResults,
        totalResults: result.members.length, failedActors,
        recoveredFailures: result.team.status === 'done' ? failedActors : 0,
        warnings: [
          ...result.members.flatMap(member => member.warnings),
          ...(!groundingSatisfied ? ['Team did not satisfy its grounding-required output contract.'] : []),
        ],
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const team = actor.actorId ? this.getTeamState(actor.actorId) : undefined;
      const agent = actor.actorId ? this.getContext().manager.getAgentById(actor.actorId)?.getInfo() : undefined;
      return {
        ...this.failedEvolutionArtifact(candidate, actor.actorId ?? 'execution-failed', message),
        actorKind: actor.kind,
        wallClockMs: Date.now() - startedAt,
        agentIds: team?.memberAgentIds ?? (agent ? [agent.identity.id] : []),
        teamIds: team ? [team.identity.id] : [],
      };
    }
  }

  private failedEvolutionArtifact(candidate: EvolutionCandidate, actorId: string, error: string): EvolutionExecutionArtifact {
    return {
      candidateId: candidate.id,
      actorKind: candidate.genome.members.length === 1 ? 'agent' : 'team',
      actorId,
      success: false,
      result: '',
      usage: this.tokenUsageToEvolutionUsage(this.sumUsage([])),
      wallClockMs: 0,
      agentIds: [], teamIds: [], toolCalls: 0, successfulToolCalls: 0, unresolvedToolIntents: 0,
      groundedResults: 0, totalResults: candidate.genome.members.length,
      failedActors: candidate.genome.members.length, recoveredFailures: 0,
      warnings: [error],
    };
  }

  private async archiveEvolutionCandidateActors(
    artifact: EvolutionExecutionArtifact,
    runId: string,
    candidateId: string,
    correlationId: string
  ): Promise<void> {
    const ctx = this.getContext();
    const actorIds = [...artifact.teamIds, ...artifact.agentIds];
    for (const actorId of actorIds) {
      const lifecycle = this.lifecycle.get(actorId);
      if (!lifecycle || lifecycle.status === 'released' || lifecycle.status === 'persisted') continue;
      const record = await this.setActorLifecycle(actorId, 'release', {
        cascade: true,
        correlationId,
        reason: `Evolution candidate ${candidateId} execution completed; runtime actor is no longer required.`,
      });
      this.emit({
        type: 'evo.candidate.actor.archived',
        agentId: actorId,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          runId,
          candidateId,
          actorKind: lifecycle.actorKind,
          lifecycleStatus: record.status,
          totalTokens: lifecycle.actorKind === 'team'
            ? artifact.usage.totalTokens
            : artifact.agentIds.includes(actorId) ? artifact.usage.totalTokens : 0,
        },
      });
    }
  }

  private createEvolutionJudge(
    parentId: string,
    correlationId: string,
    onUsage: (usage: TokenUsage) => void
  ): EvolutionJudge {
    return {
      name: 'llm_judge',
      evaluate: async (task, candidate, execution) => {
        const agent = this.getContext().manager.getAgentById(parentId) ?? this.getContext().agent;
        const usageBefore = agent.getUsage();
        this.emit({ type: 'evo.judge.started', agentId: agent.id, correlationId, data: { candidateId: candidate.id } });
        const judged = await this.completeJSONAsAgent<Partial<EvolutionEvaluationDimensions> & { rationale?: string }>(
          agent,
          [
            {
              role: 'system',
              content: 'Evaluate an executed agent/team candidate. Return strict JSON with optional 0..1 fields taskSuccess, answerQuality, completeness, costEfficiency, novelty, toolUse, consistency, tomCoverage, plus rationale. Check whether technical conclusions follow from the supplied evidence, distinguish facts from inference, and penalize unsupported or factually incorrect claims. A long, grounded-looking answer is not necessarily correct.',
            },
            {
              role: 'user',
              content: JSON.stringify({ task, genome: candidate.genome, execution: { ...execution, result: execution.result.slice(0, 8000) } }),
            },
          ],
          { temperature: 0, maxTokens: 600 },
          'evo.candidate_evaluation',
          correlationId
        );
        const usage = this.usageDifference(usageBefore, agent.getUsage());
        onUsage(usage);
        this.emit({ type: 'evo.judge.completed', agentId: agent.id, correlationId, data: { candidateId: candidate.id, usage } });
        return this.normalizeEvolutionJudgeResult(judged);
      },
    };
  }

  private normalizeEvolutionJudgeResult(
    value: Partial<EvolutionEvaluationDimensions> & { rationale?: string }
  ): Partial<EvolutionEvaluationDimensions> & { rationale?: string } {
    const result: Partial<EvolutionEvaluationDimensions> & { rationale?: string } = {};
    for (const key of ['taskSuccess', 'answerQuality', 'completeness', 'costEfficiency', 'novelty', 'toolUse', 'consistency', 'tomCoverage'] as const) {
      const score = value[key];
      if (typeof score === 'number' && Number.isFinite(score)) result[key] = clamp01(score);
    }
    if (typeof value.rationale === 'string') result.rationale = value.rationale;
    return result;
  }

  private async integrateEvolutionPattern(
    selected: EvolutionCandidate,
    evaluation: EvolutionEvaluationResult,
    execution: EvolutionExecutionArtifact
  ): Promise<string> {
    const ctx = this.getContext();
    const structure = selected.genome.members.map(member => ({
      archetype: member.archetype,
      role: member.role,
      tools: member.toolPolicy.map(tool => tool.name),
    }));
    const patternId = `evo_pattern_${this.fingerprint({ task: selected.genome.taskSignature, structure }).slice(0, 16)}_v1`;
    const [agentPatterns, teamPatterns] = await Promise.all([
      ctx.memory.getCachePatterns('agents'),
      ctx.memory.getCachePatterns('teams'),
    ]);
    const agentPatternIds = agentPatterns
      .filter(pattern => selected.genome.members.some(member => pattern.archetype === member.archetype))
      .map(pattern => String(pattern.id ?? ''))
      .filter(Boolean);
    const teamPatternIds = execution.actorKind === 'team'
      ? teamPatterns
        .filter(pattern => pattern.name === selected.genome.name || pattern.key === this.safeAgentKey(selected.genome.name))
        .map(pattern => String(pattern.id ?? ''))
        .filter(Boolean)
      : [];
    const links = { agentPatternIds, teamPatternIds };
    const pattern = await ctx.memory.upsertEvolutionPattern({
      id: patternId,
      name: selected.genome.name,
      taskSignature: selected.genome.taskSignature,
      genome: selected.genome,
      historicalScores: [],
      lineage: selected.lineage,
      linkedPatterns: links,
      evaluation,
      tokenCost: execution.usage.totalTokens,
    });
    await ctx.memory.linkEvolutionPattern(pattern.id, links);
    this.emit({
      type: 'evo.candidate.integrated', agentId: execution.actorId,
      data: { candidateId: selected.id, patternId: pattern.id, score: evaluation.score, linkedPatterns: links },
    });
    return pattern.id;
  }

  private async recordEvolutionLifecycleMessage(
    state: string,
    parentId: string,
    correlationId: string,
    payload: Record<string, unknown>
  ): Promise<void> {
    const kind = state === 'S_evo_propose' ? 'evo.propose'
      : state === 'S_evo_instantiate' ? 'evo.instantiate'
        : state === 'S_evo_execute' ? 'evo.execute'
          : state === 'S_evo_evaluate' ? 'evo.evaluate'
            : state === 'S_evo_select' ? 'evo.select'
              : state === 'S_evo_mutate' ? 'evo.mutate'
                : state === 'S_evo_integrate' || state === 'S_evo_done' ? 'evo.integrate'
                  : undefined;
    if (!kind) return;
    const message = await this.enqueueMessage({
      kind,
      sessionId: this.getContext().sessionId,
      from: parentId,
      to: 'runtime.evolution',
      correlationId,
      payload,
      metadata: { agentId: parentId, tags: ['evolution'] },
    });
    await this.processQueuedMessage(message.id);
    await this.getContext().queue.ack(message.id);
  }

  private genomeToRuntimeToM(
    profile: GenomeToMProfile,
    subjectAgentId: string,
    parentId: string,
    purpose: string,
    capabilities: string[]
  ): ToMProfile {
    return normalizeToMProfile({
      level: profile.level,
      subjectAgentId,
      beliefScope: [...profile.beliefScope],
      goalModel: [...profile.goalModel],
      uncertainty: [...profile.uncertainty],
      perspective: profile.perspective,
      observesAgents: [...profile.observesAgents ?? []],
      modelsAgents: [...profile.modelsAgents ?? []],
      capabilityScope: [...capabilities],
      cognitiveGaps: profile.uncertainty.map((_, index) => `${subjectAgentId}_gap_${index + 1}`),
      models: profile.level >= 1 ? [{
        targetId: parentId,
        targetType: 'agent',
        goalModel: [...profile.goalModel],
        uncertaintyModel: [...profile.uncertainty],
      }] : [],
      purpose,
    }, { level: profile.level, subjectAgentId, purpose });
  }

  private buildEvolutionMetrics(input: {
    lifecycle: EvolutionLifecycleResult;
    usage: TokenUsage;
    startedAt: number;
    completedAt: number;
    cacheHits: number;
    budgetRequested: number;
    budgetAllocated: number;
  }): EvolutionRunResult['metrics'] {
    const selectedEvaluation = input.lifecycle.selectedEvaluation;
    const toolCalls = input.lifecycle.executions.reduce((sum, execution) => sum + execution.toolCalls, 0);
    const successfulToolCalls = input.lifecycle.executions.reduce((sum, execution) => sum + execution.successfulToolCalls, 0);
    return {
      taskSuccess: selectedEvaluation?.success ?? false,
      answerQuality: selectedEvaluation?.dimensions.answerQuality ?? 0,
      toolSuccessRate: toolCalls > 0 ? successfulToolCalls / toolCalls : 0,
      agentsSpawned: input.lifecycle.executions.reduce((sum, execution) => sum + execution.agentIds.length, 0),
      teamsSpawned: input.lifecycle.executions.reduce((sum, execution) => sum + execution.teamIds.length, 0),
      totalTokens: input.usage.totalTokens,
      thinkingTokens: input.usage.thinkingTokens,
      wallClockMs: input.completedAt - input.startedAt,
      budgetRequested: input.budgetRequested,
      budgetAllocated: input.budgetAllocated,
      failureRecoveryCount: input.lifecycle.executions.reduce((sum, execution) => sum + execution.recoveredFailures, 0),
      candidateCount: input.lifecycle.candidates.length,
      executedCandidateCount: input.lifecycle.executions.length,
      selectedGenomeId: input.lifecycle.selected?.genome.id,
      selectedGenomeScore: selectedEvaluation?.score,
      cacheHits: input.cacheHits,
      mutationsApplied: input.lifecycle.candidates.filter(candidate => candidate.lineage.operators.length > 0).length,
    };
  }

  private tokenUsageToEvolutionUsage(usage: TokenUsage): EvolutionExecutionArtifact['usage'] {
    return {
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thinkingTokens: usage.thinkingTokens,
      totalTokens: usage.totalTokens,
      cachedInputTokens: usage.cachedInputTokens,
      cacheCreationInputTokens: usage.cacheCreationInputTokens,
    };
  }

  private evolutionUsageToTokenUsage(usage: EvolutionExecutionArtifact['usage']): TokenUsage {
    return {
      llmCalls: usage.totalTokens > 0 ? 1 : 0,
      promptTokens: usage.inputTokens,
      completionTokens: usage.outputTokens,
      totalTokens: usage.totalTokens,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thinkingTokens: usage.thinkingTokens,
      thinkingAccountingTokens: usage.thinkingTokens ?? usage.totalTokens,
      cachedInputTokens: usage.cachedInputTokens ?? null,
      cacheCreationInputTokens: usage.cacheCreationInputTokens ?? null,
    };
  }

  private sumUsage(items: TokenUsage[]): TokenUsage {
    return items.reduce<TokenUsage>((total, item) => ({
      llmCalls: total.llmCalls + item.llmCalls,
      promptTokens: total.promptTokens + item.promptTokens,
      completionTokens: total.completionTokens + item.completionTokens,
      totalTokens: total.totalTokens + item.totalTokens,
      inputTokens: total.inputTokens + item.inputTokens,
      outputTokens: total.outputTokens + item.outputTokens,
      thinkingTokens: total.thinkingTokens === null && item.thinkingTokens === null
        ? null
        : Number(total.thinkingTokens ?? 0) + Number(item.thinkingTokens ?? 0),
      thinkingAccountingTokens: (total.thinkingAccountingTokens ?? 0) + (item.thinkingAccountingTokens ?? item.thinkingTokens ?? item.totalTokens),
      cachedInputTokens: total.cachedInputTokens === null && item.cachedInputTokens === null
        ? null
        : Number(total.cachedInputTokens ?? 0) + Number(item.cachedInputTokens ?? 0),
      cacheCreationInputTokens: total.cacheCreationInputTokens === null && item.cacheCreationInputTokens === null
        ? null
        : Number(total.cacheCreationInputTokens ?? 0) + Number(item.cacheCreationInputTokens ?? 0),
    }), {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      thinkingTokens: null,
      thinkingAccountingTokens: 0,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
    });
  }

  private subtractTokenUsage(after: TokenUsage, before: TokenUsage): TokenUsage {
    const thinkingTokens = after.thinkingTokens === null && before.thinkingTokens === null
      ? null
      : Math.max(0, Number(after.thinkingTokens ?? 0) - Number(before.thinkingTokens ?? 0));
    const estimatedCostUsd = Math.max(0, Number(after.estimatedCostUsd ?? 0) - Number(before.estimatedCostUsd ?? 0));
    return {
      llmCalls: Math.max(0, after.llmCalls - before.llmCalls),
      promptTokens: Math.max(0, after.promptTokens - before.promptTokens),
      completionTokens: Math.max(0, after.completionTokens - before.completionTokens),
      totalTokens: Math.max(0, after.totalTokens - before.totalTokens),
      inputTokens: Math.max(0, after.inputTokens - before.inputTokens),
      outputTokens: Math.max(0, after.outputTokens - before.outputTokens),
      thinkingTokens,
      thinkingAccountingTokens: Math.max(
        0,
        (after.thinkingAccountingTokens ?? after.thinkingTokens ?? after.totalTokens)
          - (before.thinkingAccountingTokens ?? before.thinkingTokens ?? before.totalTokens)
      ),
      cachedInputTokens: after.cachedInputTokens === null && before.cachedInputTokens === null
        ? null
        : Math.max(0, Number(after.cachedInputTokens ?? 0) - Number(before.cachedInputTokens ?? 0)),
      cacheCreationInputTokens: after.cacheCreationInputTokens === null && before.cacheCreationInputTokens === null
        ? null
        : Math.max(0, Number(after.cacheCreationInputTokens ?? 0) - Number(before.cacheCreationInputTokens ?? 0)),
      estimatedCostUsd: estimatedCostUsd > 0 ? estimatedCostUsd : undefined,
    };
  }

  private async synthesizeChildResult(
    parentId: string,
    userTask: string,
    childAgent: AgentInfo,
    childResult: RunAgentResult,
    correlationId: string,
    parentMessageId: string
  ): Promise<string> {
    if (parentId === 'root') {
      return this.synthesizeSubagentResult(userTask, childAgent, childResult, correlationId, parentMessageId);
    }

    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) {
      throw new Error(`Parent agent "${parentId}" not found`);
    }
    const parentIdentity = parent.getIdentity();
    if (this.requireAgentFsm(parentId).getState() === 'S_delegating') {
      await this.transitionAgentFsm(parentId, 'S_waiting_children', { correlationId, childId: childAgent.identity.id });
    }
    const synthesisMessage = await this.enqueueMessage({
      kind: 'agent.synthesis',
      sessionId: ctx.sessionId,
      from: parentId,
      to: parentId,
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        parentId,
        childId: childAgent.identity.id,
      },
      metadata: {
        agentId: parentId,
        tomLevel: parentIdentity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = parent.getUsage();
    const from = parent.getState();
    parent.setRuntimeState('synthesizing');
    await this.transitionAgentFsm(parentId, 'S_synthesizing', { correlationId, childId: childAgent.identity.id });
    this.emit({
      type: 'agent.synthesis.started',
      agentId: parentId,
      data: {
        correlationId,
        childId: childAgent.identity.id,
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from, to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: parentId, data: { purpose: 'agent.child_synthesis', correlationId } });

    const synthesisPrompt = this.buildParentChildSynthesisPrompt(parent.getInfo(), userTask, childAgent, childResult);
    this.ensureAgentSynthesisBudget(parent, synthesisPrompt, 'agent.child_synthesis', correlationId);
    const response = await this.completeAsAgent(
      parent,
      synthesisPrompt,
      'agent.child_synthesis',
      correlationId
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: parentId, data: { ...usageDelta } });

    parent.setRuntimeState('done');
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from: 'synthesizing', to: 'done', correlationId } });
    await this.transitionAgentFsm(parentId, 'S_responding', { correlationId, childId: childAgent.identity.id });
    await this.transitionAgentFsm(parentId, 'S_done', { correlationId, childId: childAgent.identity.id });
    this.emit({
      type: 'agent.synthesis.completed',
      agentId: parentId,
      data: {
        correlationId,
        childId: childAgent.identity.id,
        totalTokens: usageDelta.totalTokens,
      },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return response || parent.getInfo().lastResult || '';
  }

  private async synthesizeDirectChildResults(
    parentId: string,
    userTask: string,
    childResults: RootMediatedSpawnResult[],
    correlationId: string,
    parentMessageId: string,
    teamResult?: TeamRunResult
  ): Promise<string> {
    if (childResults.length === 1) {
      return this.synthesizeChildResult(
        parentId,
        userTask,
        childResults[0].agent,
        childResults[0].subagentResult,
        correlationId,
        parentMessageId
      );
    }

    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(parentId);
    if (!parent) {
      throw new Error(`Parent agent "${parentId}" not found`);
    }
    if (parentId === 'root') {
      return this.synthesizeDelegatedResults(userTask, childResults, correlationId, teamResult ? [teamResult] : []);
    }

    const parentIdentity = parent.getIdentity();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'agent.synthesis',
      sessionId: ctx.sessionId,
      from: parentId,
      to: parentId,
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        parentId,
        childIds: childResults.map(result => result.agent.identity.id),
      },
      metadata: {
        agentId: parentId,
        tomLevel: parentIdentity.tomProfile.level,
      },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = parent.getUsage();
    const from = parent.getState();
    parent.setRuntimeState('synthesizing');
    await this.transitionAgentFsm(parentId, 'S_synthesizing', {
      correlationId,
      childIds: childResults.map(result => result.agent.identity.id),
    });
    this.emit({
      type: 'agent.synthesis.started',
      agentId: parentId,
      data: {
        correlationId,
        childIds: childResults.map(result => result.agent.identity.id),
      },
    });
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from, to: 'synthesizing', correlationId } });
    this.emit({ type: 'agent.llm.called', agentId: parentId, data: { purpose: 'agent.multi_child_synthesis', correlationId } });

    const synthesisPrompt = this.buildParentMultiChildSynthesisPrompt(parent.getInfo(), userTask, childResults, teamResult);
    this.ensureAgentSynthesisBudget(parent, synthesisPrompt, 'agent.multi_child_synthesis', correlationId);
    const response = await this.completeAsAgent(
      parent,
      synthesisPrompt,
      'agent.multi_child_synthesis',
      correlationId
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: parentId, data: { ...usageDelta } });

    parent.setRuntimeState('done');
    this.emit({ type: 'agent.status.changed', agentId: parentId, data: { from: 'synthesizing', to: 'done', correlationId } });
    await this.transitionAgentFsm(parentId, 'S_responding', { correlationId, childIds: childResults.map(result => result.agent.identity.id) });
    await this.transitionAgentFsm(parentId, 'S_done', { correlationId, childIds: childResults.map(result => result.agent.identity.id) });
    this.emit({
      type: 'agent.synthesis.completed',
      agentId: parentId,
      data: {
        correlationId,
        childIds: childResults.map(result => result.agent.identity.id),
        totalTokens: usageDelta.totalTokens,
      },
    });
    await ctx.queue.ack(synthesisMessage.id);
    return response || parent.getInfo().lastResult || '';
  }

  private async runAgentDelegatedChildren(
    agentId: string,
    task: string,
    decision: Extract<DelegationDecision, { action: 'spawn_subagents' }>,
    usageBefore: AgentUsage,
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      disableRecursiveDelegation?: boolean;
      nodeId?: string;
      patternId?: string;
    }
  ): Promise<RunAgentResult> {
    const ctx = this.getContext();
    const parent = ctx.manager.getAgentById(agentId);
    if (!parent) {
      throw new Error(`Agent "${agentId}" not found`);
    }
    const correlationId = options.correlationId ?? this.createCorrelationId();
    const plans = decision.agents;
    await this.transitionAgentFsm(agentId, 'S_planning', { correlationId, count: plans.length });
    this.emit({
      type: 'delegation.plan.created',
      agentId,
      data: {
        correlationId,
        scope: 'agent',
        count: plans.length,
        agents: plans,
      },
    });
    await this.transitionAgentFsm(agentId, 'S_delegating', { correlationId, count: plans.length });

    const childResults: RootMediatedSpawnResult[] = [];
    for (const plan of plans) {
      this.emit({
        type: 'delegation.subagent.selected',
        agentId,
        data: {
          correlationId,
          archetype: plan.archetype,
          name: plan.name,
          tomLevel: plan.tomLevel,
          tomProfile: plan.tomProfile,
          cognitiveGapIds: plan.cognitiveGapIds,
          existenceReason: plan.existenceReason,
          budgetTokens: plan.budgetTokens,
          scope: 'agent',
        },
      });
      this.emit({
        type: 'delegation.subagent.task_assigned',
        agentId,
        data: {
          correlationId,
          archetype: plan.archetype,
          name: plan.name,
          task: plan.task,
          scope: 'agent',
        },
      });
    }

    let teamResult: TeamRunResult | undefined;
    const shouldCreateTeam = plans.length > 1
      && (decision.coordination === 'team'
        || (decision.coordination === undefined && this.workspaceRuntimeConfig?.teams.createForMultipleAgents !== false));
    if (shouldCreateTeam) {
      const teamPlan = decision.team;
      const team = await this.spawnTeam({
        parentAgentId: agentId,
        name: teamPlan?.name ?? this.deriveTeamName(plans),
        description: teamPlan?.description ?? task,
        task: teamPlan?.task ?? task,
        synthesisPolicy: teamPlan?.synthesisPolicy,
        tomLevel: teamPlan?.tomLevel,
        executionPolicy: teamPlan?.executionPolicy,
        members: plans.map((plan, index) => ({ ...plan, lead: index === 0 })),
        tomAnalysis: this.tomAnalyses.get(correlationId),
        correlationId,
        lifecycleOrigin: 'automatic_delegation',
      });
      teamResult = await this.runTeam(team.identity.id, task, {
        correlationId,
        memberRecursiveDelegation: teamPlan?.memberDelegationPolicy !== 'deny',
      });
      childResults.push(...teamResult.memberExecutions);
    } else {
      for (const plan of plans) {
        try {
          const result = await this.handleSpawnCommand({
            archetype: plan.archetype,
            task: plan.task,
            parentId: agentId,
            name: plan.name,
            customRole: plan.role,
            customStyle: plan.style,
            tools: plan.tools,
            skills: plan.skills,
            tomLevel: plan.tomLevel,
            tomProfile: plan.tomProfile,
            cognitiveGapIds: plan.cognitiveGapIds,
            existenceReason: plan.existenceReason,
            systemPrompt: plan.systemPrompt,
            budgetTokens: plan.budgetTokens,
            memoryScope: plan.memoryScope,
            correlationId,
            source: agentId,
            requireRootSynthesis: false,
            showSubagentOutput: false,
            disableRecursiveDelegation: false,
          });
          childResults.push(result);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (!message.startsWith('Spawn rejected:')) throw error;
          this.emit({
            type: 'delegation.child.skipped',
            agentId,
            correlationId,
            data: {
              archetype: plan.archetype,
              name: plan.name,
              task: plan.task,
              reason: message.slice('Spawn rejected:'.length).trim(),
            },
          });
        }
      }
    }

    await this.transitionAgentFsm(agentId, 'S_waiting_children', { correlationId, completed: childResults.length });
    if (childResults.length === 0) {
      return this.runAgent(agentId, task, { ...options, disableRecursiveDelegation: true });
    }

    const synthesis = await this.synthesizeDirectChildResults(
      agentId,
      task,
      childResults,
      correlationId,
      childResults[0].messages.find(message => message.kind === 'agent.result')?.id ?? options.parentMessageId ?? '',
      teamResult
    );
    const usageAfter = parent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    const evidence = this.mergeChildEvidence(childResults.map(result => result.subagentResult));
    const warnings = childResults.flatMap(result => result.subagentResult.warnings);
    this.emit({
      type: 'agent.run.completed',
      agentId,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        task,
        correlationId,
        delegated: true,
        childIds: childResults.map(result => result.agent.identity.id),
        totalTokens: usageDelta.totalTokens,
        grounded: childResults.every(result => result.subagentResult.grounded),
        evidence,
        warnings,
      },
    });
    this.emit({
      type: 'delegation.completed',
      agentId,
      data: {
        correlationId,
        scope: 'agent',
        subagentIds: childResults.map(result => result.agent.identity.id),
        totalSubagents: childResults.length,
      },
    });
    const groundedChildren = childResults.filter(result => result.subagentResult.grounded).length;
    const groundedRatio = childResults.length > 0 ? groundedChildren / childResults.length : 0;
    this.settleAgentBudget(agentId, usageDelta, {
      success: true,
      evidenceGain: evidence.outputGrounded ? 0.9 : evidence.toolGrounded ? 0.5 : 0.15,
      uncertaintyReduction: groundedRatio,
      conflictResolution: childResults.length > 1 ? groundedRatio : undefined,
    });
    await ctx.memory.recordAgentPatternOutcome(options.archetype ?? this.inferAgentArchetype(parent.getInfo()), {
      success: true,
      grounded: childResults.every(result => result.subagentResult.grounded),
      totalTokens: usageDelta.totalTokens,
    }, options.patternId);

    return {
      agent: parent.getInfo(),
      result: synthesis,
      usage: usageDelta,
      toolCalls: childResults.flatMap(result => result.subagentResult.toolCalls),
      evidence,
      grounded: childResults.every(result => result.subagentResult.grounded),
      warnings,
    };
  }

  private mergeChildEvidence(results: RunAgentResult[]): RunEvidence {
    return {
      toolGrounded: results.some(result => result.evidence.toolGrounded),
      outputGrounded: results.some(result => result.evidence.outputGrounded),
      observedPaths: Array.from(new Set(results.flatMap(result => result.evidence.observedPaths))),
      toolResultSummary: results.map(result => result.evidence.toolResultSummary).filter(Boolean).join('\n'),
    };
  }

  private async transitionTeamFsm(
    teamId: string,
    state: TeamFSMState,
    data: Record<string, unknown> = {}
  ): Promise<TeamRuntimeState> {
    const team = this.teams.get(teamId);
    if (!team) throw new Error(`Team "${teamId}" not found`);
    const from = team.fsmState;
    try {
      const updated = this.teams.transitionFsm(teamId, state, typeof data.error === 'string' ? data.error : undefined);
      this.emit({
        type: 'team.fsm.transition',
        agentId: updated.identity.id,
        sessionId: this.getContext().sessionId,
        correlationId: updated.correlationId,
        data: { teamId, from, to: state, status: updated.status, ...data },
      });
      return updated;
    } catch (error) {
      this.emit({
        type: 'team.fsm.invalid_transition',
        agentId: team.identity.id,
        sessionId: this.getContext().sessionId,
        correlationId: team.correlationId,
        data: { teamId, from, to: state, error: error instanceof Error ? error.message : String(error) },
      });
      throw error;
    }
  }

  private async persistTeamTopology(team: TeamRuntimeState): Promise<void> {
    const ctx = this.getContext();
    const plannedMembers = (this.teamMemberPlans.get(team.identity.id) ?? []).map(member => ({
      archetype: member.archetype,
      name: member.name,
      role: member.role,
      task: member.task,
      tools: member.tools,
      skills: member.skills,
      tomLevel: member.tomLevel,
      tomProfile: member.tomProfile,
      cognitiveGapIds: member.cognitiveGapIds,
      existenceReason: member.existenceReason,
      lead: member.lead ?? false,
    }));
    const members = team.memberAgentIds.map(agentId => {
      const agent = ctx.manager.getAgentById(agentId)?.getInfo();
      return {
        id: agentId,
        name: agent?.identity.name,
        role: agent?.identity.role,
        parentId: agent?.identity.parentId,
        teamId: agent?.identity.teamId,
        task: team.memberTasks[agentId],
      };
    });
    await ctx.memory.writeTeamTopology(this.safeAgentKey(team.identity.name), {
      type: 'subteam',
      teamId: team.identity.id,
      parentAgentId: team.identity.parentAgentId,
      leadAgentId: team.leadAgentId,
      tomLevel: team.identity.tomLevel,
      tomProfile: team.identity.tomProfile,
      status: team.status,
      fsmState: team.fsmState,
      members,
      plannedMembers,
      tokenUsage: team.tokenUsage,
      executionPolicy: team.executionPolicy,
      memberStatuses: team.memberStatuses,
      memberErrors: team.memberErrors,
      updatedAt: new Date().toISOString(),
    });
  }

  private async persistTeamRunArtifacts(input: {
    team: TeamRuntimeState;
    task: string;
    result?: string;
    correlationId: string;
    usage: TokenUsage;
    success: boolean;
  }): Promise<void> {
    const ctx = this.getContext();
    const teamKey = this.safeAgentKey(input.team.identity.name);
    const operations: Array<{ name: string; run: () => Promise<void> }> = [];
    if (input.success && input.result !== undefined) {
      operations.push({
        name: 'append_team_session',
        run: () => ctx.memory.appendTeamSession(teamKey, {
          timestamp: Date.now(),
          sessionId: ctx.sessionId,
          correlationId: input.correlationId,
          teamId: input.team.identity.id,
          task: input.task,
          result: input.result,
          memberAgentIds: input.team.memberAgentIds,
          memberStatuses: input.team.memberStatuses,
          memberErrors: input.team.memberErrors,
          partial: Object.keys(input.team.memberErrors).length > 0,
          tokenUsage: input.usage,
          cumulativeTokenUsage: input.team.tokenUsage,
        }),
      });
    }
    operations.push(
      {
        name: 'persist_team_topology',
        run: () => this.persistTeamTopology(input.team),
      },
      {
        name: 'record_team_pattern_outcome',
        run: () => ctx.memory.recordTeamPatternOutcome(teamKey, {
          success: input.success,
          totalTokens: input.usage.totalTokens,
          memberCount: input.team.memberAgentIds.length,
          failedMemberCount: Object.keys(input.team.memberErrors).length,
        }),
      }
    );
    for (const operation of operations) {
      try {
        await operation.run();
      } catch (error) {
        this.emit({
          type: 'team.persistence.failed',
          agentId: input.team.identity.id,
          sessionId: ctx.sessionId,
          correlationId: input.correlationId,
          data: {
            teamId: input.team.identity.id,
            parentAgentId: input.team.identity.parentAgentId,
            operation: operation.name,
            error: error instanceof Error ? error.message : String(error),
          },
        });
      }
    }
  }

  private async completeAsTeam(
    team: TeamRuntimeState,
    task: string,
    members: RunAgentResult[],
    failures: Array<TeamExecutionOutcome<unknown>>,
    correlationId: string
  ): Promise<{ content: string; usage: TokenUsage }> {
    const ctx = this.getContext();
    const orderedMembers = [...members].sort((left, right) => {
      if (left.agent.identity.id === team.leadAgentId) return -1;
      if (right.agent.identity.id === team.leadAgentId) return 1;
      return 0;
    });
    const hasStructuredVerifierEvidence = orderedMembers.some(member =>
      member.toolCalls.some(call =>
        isSuccessfulWorkspaceVerificationCall(call)
        || this.isFocusedVerifierDiagnosticCall(call)
      )
    );
    const hasStructuredMutationOrFailure = failures.length > 0
      || orderedMembers.some(member =>
        member.toolCalls.some(call => isSuccessfulWorkspaceMutationCall(call))
      );
    const canAggregateSequentialClosure =
      team.executionPolicy.mode === 'sequential'
      && orderedMembers.length > 0
      && orderedMembers.every(member => member.evidence.toolGrounded)
      && hasStructuredVerifierEvidence
      && hasStructuredMutationOrFailure;
    if (canAggregateSequentialClosure) {
      this.emit({
        type: 'team.synthesis.started',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          memberAgentIds: team.memberAgentIds,
          leadAgentId: team.leadAgentId,
          failedMembers: failures.length,
          mode: 'structured_sequential_closure',
        },
      });
      this.emit({
        type: 'team.synthesis.recovered',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          reason: 'structured_sequential_closure',
          recovery: 'deterministic_evidence_aggregation',
          completedMembers: members.length,
          failedMembers: failures.length,
        },
      });
      this.emit({
        type: 'team.synthesis.completed',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          totalTokens: 0,
          deterministic: true,
          reason: 'structured_sequential_closure',
        },
      });
      return {
        content: this.buildStructuredSequentialTeamClosure(
          team,
          task,
          orderedMembers,
          failures
        ),
        usage: this.zeroTokenUsage(),
      };
    }
    const teamKey = this.safeAgentKey(team.identity.name);
    const [teamDefinition, teamMemory, rootContext] = await Promise.all([
      ctx.memory.readTeamDoc(teamKey, 'team'),
      ctx.memory.readTeamDoc(teamKey, 'memory'),
      ctx.memory.loadRootContext(),
    ]);
    const publicContext = [
      `<execution_knowledge>${this.formatExecutionKnowledge(rootContext.executionKnowledge)}</execution_knowledge>`,
      rootContext.projectMemory,
      rootContext.constraints,
      rootContext.decisions,
    ]
      .filter(Boolean)
      .join('\n\n')
      .slice(0, 6000);
    this.emit({
      type: 'team.context.loaded',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        teamId: team.identity.id,
        teamDefinitionChars: teamDefinition.length,
        teamMemoryChars: teamMemory.length,
        publicContextChars: publicContext.length,
      },
    });
    const reports = orderedMembers.map(member => [
      `<member id="${member.agent.identity.id}" name="${member.agent.identity.name}">`,
      `team_role: ${member.agent.identity.id === team.leadAgentId ? 'lead' : 'member'}`,
      `tokens: ${member.usage.totalTokens}`,
      `grounded: ${member.grounded}`,
      `tool_grounded: ${member.evidence.toolGrounded}`,
      `output_grounded: ${member.evidence.outputGrounded}`,
      `observed_paths: ${JSON.stringify(member.evidence.observedPaths)}`,
      `observed_urls: ${JSON.stringify(member.evidence.observedUrls ?? [])}`,
      `discovered_urls: ${JSON.stringify(member.evidence.discoveredUrls ?? [])}`,
      `failed_tool_calls: ${JSON.stringify(member.toolCalls.filter(call => !call.success).map(call => ({
        toolName: call.toolName,
        params: call.params,
        error: call.error,
      })))}`,
      `warnings: ${JSON.stringify(member.warnings)}`,
      `tool_result_summary:\n${(member.evidence.toolResultSummary ?? 'none').slice(0, 5000)}`,
      `tom_profile: ${JSON.stringify(member.agent.identity.tomProfile)}`,
      'member_report:',
      member.result,
      '</member>',
    ].join('\n')).join('\n\n');
    const failureReports = failures.map(failure => [
      `<member_failure key="${failure.key}">`,
      failure.error ?? 'unknown member execution failure',
      '</member_failure>',
    ].join('\n')).join('\n\n');
    const cachedTeamToolEvidence = this.formatTeamCachedToolEvidence(
      team.identity.id
    );
    const prompt = [
      `Team task: ${task}`,
      `<acceptance_checklist>\n${this.buildTaskAcceptanceChecklist(task)}\n</acceptance_checklist>`,
      `You are ${team.identity.name}, a subteam actor in Roy.`,
      `Description: ${team.identity.description}`,
      `Parent-defined synthesis policy: ${team.synthesisPolicy ?? 'Aggregate evidence, preserve disagreements, and return one grounded result.'}`,
      `ToM level: ${team.identity.tomLevel}`,
      `ToM profile: ${JSON.stringify(team.identity.tomProfile, null, 2)}`,
      `Lead agent: ${team.leadAgentId ?? 'not assigned'}`,
      `<team_definition>\n${teamDefinition}\n</team_definition>`,
      `<team_private_memory>\n${teamMemory}\n</team_private_memory>`,
      `<public_context>\n${publicContext}\n</public_context>`,
      'Aggregate direct member reports into one grounded result for the parent agent.',
      'Reconcile member beliefs explicitly, preserve unresolved uncertainty, and explain how the final result covers the team cognitive gaps.',
      'Give the lead report coordination priority, but verify it against all available member evidence.',
      'If member failures are present, state their impact and do not imply full team completion.',
      'Reconcile every acceptance-checklist item against structured tool evidence. Keep failed and unverified items explicit for the next cached step.',
      'Before declaring an item verified, inspect stale parallel declarations, call sites, generated metadata, configuration, and compatibility paths that can contradict the primary edit.',
      'Produce explicit completed-state, authoritative-path, invalid-path, verification, and actionable-feedback sections so the runtime can carry them into later steps.',
      'Do not repeat a cached failed path or equivalent failed call unless a member used a changed hypothesis and recorded the new outcome.',
      'The structured observed_paths, observed_urls, and tool_result_summary fields are the authoritative evidence boundary.',
      'Do not infer file contents merely because a path was observed. Do not invent example values and present them as observations.',
      'A member report claim is usable only when supported by that member structured evidence or clearly labeled as analysis.',
      'Do not claim any tool call, file read, command output, or project fact absent from the structured member evidence.',
      'The cached team tool frontier is authoritative partial execution evidence from members that may have failed before returning a narrative report. Reconcile it explicitly; never describe a recorded mutation or verification as unattempted.',
      `<cached_team_tool_frontier>\n${cachedTeamToolEvidence}\n</cached_team_tool_frontier>`,
      reports,
      failureReports,
    ].join('\n\n');
    const systemPrompt = [
      `You are ${team.identity.name}, a formal subteam actor in the Roy autonomous agent system.`,
      'You are not Roy and you are not the model provider.',
      `Parent agent: ${team.identity.parentAgentId}.`,
      `Correlation: ${correlationId}.`,
    ].join('\n');
    this.emit({
      type: 'team.synthesis.started',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: {
        teamId: team.identity.id,
        memberAgentIds: team.memberAgentIds,
        leadAgentId: team.leadAgentId,
        failedMembers: failures.length,
      },
    });
    if (!ctx.llm) {
      const usage = this.zeroTokenUsage();
      this.emit({
        type: 'team.synthesis.completed',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: team.identity.id, totalTokens: 0, limited: true, reason: 'llm_not_configured' },
      });
      return {
        content: this.buildTeamSynthesisFallback(team, task, members, failures, 'llm_not_configured'),
        usage,
      };
    }
    const estimatedPromptTokens = this.estimateTextTokens(`${systemPrompt}\n${prompt}`);
    const completionTokenBudget = this.reasoningAwareCompletionTokenBudget(1024);
    const allocation = await this.requestTeamSynthesisBudget({
      team,
      correlationId,
      promptTokens: estimatedPromptTokens,
      completionTokens: completionTokenBudget,
    });
    if (allocation?.status === 'denied') {
      const usage = this.zeroTokenUsage();
      this.emit({
        type: 'team.synthesis.recovered',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          reason: allocation.reason,
          recovery: 'deterministic_member_aggregation',
          completedMembers: members.length,
          failedMembers: failures.length,
        },
      });
      this.emit({
        type: 'team.synthesis.completed',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: { teamId: team.identity.id, totalTokens: 0, limited: true, reason: allocation.reason },
      });
      return {
        content: this.buildTeamSynthesisFallback(team, task, members, failures, allocation.reason),
        usage,
      };
    }
    const maxCompletionTokens = allocation
      ? Math.max(1, this.completionCapacity(allocation.grantedTokens, estimatedPromptTokens))
      : completionTokenBudget;
    let completion: { content: string; usage?: ModelTokenUsage };
    try {
      completion = await this.collectRuntimeLLMStream(
        ctx.llm,
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        { temperature: 0.2, maxTokens: maxCompletionTokens },
        { actorId: team.identity.id, teamId: team.identity.id, purpose: 'team.synthesis', correlationId }
      );
    } catch (error) {
      this.releaseTeamSynthesisBudget(team.identity.id, allocation, correlationId, 'team_synthesis_failed');
      const message = error instanceof Error ? error.message : String(error);
      this.emit({
        type: 'team.synthesis.recovered',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          reason: 'llm_synthesis_failed',
          error: message.slice(0, 1000),
          recovery: 'deterministic_member_aggregation',
          completedMembers: members.length,
          failedMembers: failures.length,
        },
      });
      this.emit({
        type: 'team.synthesis.completed',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          totalTokens: 0,
          limited: true,
          reason: 'llm_synthesis_failed',
        },
      });
      return {
        content: this.buildTeamSynthesisFallback(
          team,
          task,
          orderedMembers,
          failures,
          `llm_synthesis_failed: ${message.slice(0, 240)}`
        ),
        usage: this.zeroTokenUsage(),
      };
    }
    let content = completion.content;
    if (!content.trim()) {
      content = this.buildTeamSynthesisFallback(team, task, orderedMembers, failures);
      this.emit({
        type: 'team.synthesis.fallback',
        agentId: team.identity.id,
        sessionId: ctx.sessionId,
        correlationId,
        data: {
          teamId: team.identity.id,
          reason: 'model_returned_empty_visible_output',
          memberCount: orderedMembers.length,
        },
      });
    }
    const normalizedUsage = completion.usage ?? this.estimateModelUsage([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt },
    ], content);
    const usage = this.toTokenUsage({
      llmCalls: 1,
      promptTokens: normalizedUsage.promptTokens,
      completionTokens: normalizedUsage.completionTokens,
      totalTokens: normalizedUsage.totalTokens,
      inputTokens: normalizedUsage.inputTokens ?? normalizedUsage.promptTokens,
      outputTokens: normalizedUsage.outputTokens ?? normalizedUsage.completionTokens,
      thinkingTokens: normalizedUsage.thinkingTokens ?? null,
      cachedInputTokens: normalizedUsage.cachedInputTokens ?? null,
      cacheCreationInputTokens: normalizedUsage.cacheCreationInputTokens ?? null,
    });
    this.settleTeamSynthesisBudget(team.identity.id, allocation, usage, correlationId);
    this.recordTurnUsage(usage);
    this.emit({
      type: 'budget.updated',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId: team.identity.id, ...usage },
    });
    this.emit({
      type: 'team.synthesis.completed',
      agentId: team.identity.id,
      sessionId: ctx.sessionId,
      correlationId,
      data: { teamId: team.identity.id, totalTokens: usage.totalTokens },
    });
    return { content, usage };
  }

  private buildTeamSynthesisFallback(
    team: TeamRuntimeState,
    task: string,
    members: RunAgentResult[],
    failures: Array<TeamExecutionOutcome<unknown>>,
    reason = 'model_returned_empty_visible_output'
  ): string {
    const reports = members.map(member => {
      const report = member.result.trim() || 'No visible narrative response was produced.';
      return `### ${member.agent.identity.name}\n${report.slice(0, 2400)}`;
    }).join('\n\n');
    const observedPaths = Array.from(new Set(members.flatMap(member => member.evidence.observedPaths))).slice(0, 80);
    const limitations = [
      ...failures.map(failure => `${failure.key}: ${failure.error ?? 'member execution failed'}`),
      ...members.flatMap(member => member.warnings),
    ];
    const cachedTeamToolEvidence = this.formatTeamCachedToolEvidence(
      team.identity.id
    );
    return [
      '[runtime_team_synthesis_fallback]',
      `# ${team.identity.name} Result`,
      `Task: ${task}`,
      `The synthesis model was unavailable (${reason}). The following member reports are preserved for diagnosis and are not accepted as a verified team conclusion.`,
      `Parent-defined synthesis policy: ${team.synthesisPolicy ?? 'not provided'}`,
      '## Unverified Member Reports',
      reports || 'No member report was available.',
      '## Runtime Evidence',
      observedPaths.length > 0
        ? observedPaths.map(item => `- ${item}`).join('\n')
        : 'No structured filesystem paths were observed.',
      '## Cached Partial Execution Frontier',
      cachedTeamToolEvidence,
      '## Limitations',
      limitations.length > 0
        ? limitations.map(item => `- ${item}`).join('\n')
        : '- The model returned no visible team synthesis, so the runtime preserved member reports and evidence without adding new claims.',
    ].join('\n\n');
  }

  private formatTeamCachedToolEvidence(
    teamId: string,
    maxCalls = 32,
    maxChars = 12_000
  ): string {
    const calls = this.boundExecutionToolFrontier(
      this.teamToolEvidenceCache.get(teamId) ?? [],
      maxCalls,
      240_000
    );
    if (calls.length === 0) {
      return 'No cached partial tool execution was recorded for this team.';
    }
    const lines = calls.map(call => {
      const result = call.result && typeof call.result === 'object'
        ? call.result as {
          path?: unknown;
          exitCode?: unknown;
          timedOut?: unknown;
          stdout?: unknown;
          stderr?: unknown;
        }
        : undefined;
      const details = {
        path: result?.path,
        exitCode: result?.exitCode,
        timedOut: result?.timedOut,
        stdout: typeof result?.stdout === 'string'
          ? result.stdout.slice(-800)
          : undefined,
        stderr: typeof result?.stderr === 'string'
          ? result.stderr.slice(-1_200)
          : undefined,
        error: call.error ? String(call.error).slice(-1_200) : undefined,
      };
      return [
        `- ${call.success ? 'ok' : 'failed'} ${call.toolName} ${JSON.stringify(call.params).slice(0, 500)}`,
        JSON.stringify(details),
      ].join('\n  ');
    });
    const rendered = [
      `Recorded calls: ${calls.length}; successful: ${calls.filter(call => call.success).length}; failed: ${calls.filter(call => !call.success).length}.`,
      ...lines,
    ].join('\n');
    return rendered.length <= maxChars
      ? rendered
      : `${rendered.slice(0, Math.floor(maxChars * 0.45))}\n[runtime_compacted_team_tool_frontier]\n${rendered.slice(-Math.ceil(maxChars * 0.55))}`;
  }

  private buildStructuredSequentialTeamClosure(
    team: TeamRuntimeState,
    task: string,
    members: RunAgentResult[],
    failures: Array<TeamExecutionOutcome<unknown>>
  ): string {
    const memberEvidence = members.map((member, index) => {
      const successfulTools = member.toolCalls
        .filter(call => call.success)
        .map(call => call.toolName);
      const failedTools = member.toolCalls
        .filter(call => !call.success)
        .map(call => `${call.toolName}: ${call.error ?? 'failed'}`);
      return [
        `### ${index + 1}. ${member.agent.identity.name}`,
        `Grounded: ${member.grounded}; successful tools: ${successfulTools.join(', ') || 'none'}`,
        member.evidence.observedPaths.length > 0
          ? `Observed paths: ${member.evidence.observedPaths.join(', ')}`
          : 'Observed paths: none',
        failedTools.length > 0
          ? `Failed tools: ${failedTools.join('; ')}`
          : '',
        'Structured tool evidence:',
        (member.evidence.toolResultSummary ?? 'No structured tool summary.').slice(-3_600),
        'Member report:',
        (member.result.trim() || 'No visible member report.').slice(-1_800),
      ].filter(Boolean).join('\n');
    }).join('\n\n');
    const failureEvidence = failures.length > 0
      ? failures.map(failure =>
        `- ${failure.key}: ${failure.error ?? 'member execution failed'}`
      ).join('\n')
      : '- None.';
    const observedPaths = Array.from(new Set(
      members.flatMap(member => member.evidence.observedPaths)
    )).slice(0, 80);
    return [
      '[runtime_structured_sequential_team_closure]',
      `# ${team.identity.name} Evidence Closure`,
      `Task reference: ${this.compactDelegatedTask(task, 1_200)}`,
      `Synthesis policy: ${team.synthesisPolicy ?? 'Preserve authoritative member evidence in execution order.'}`,
      'The runtime aggregated this sequential mutation-and-verification chain directly from structured tool results. No additional model inference or unsupported completion claim was added.',
      '## Ordered Member Evidence',
      memberEvidence,
      '## Member Failures',
      failureEvidence,
      '## Authoritative Paths',
      observedPaths.length > 0
        ? observedPaths.map(item => `- ${item}`).join('\n')
        : '- No filesystem path was recorded.',
      '## Closure Rule',
      'Treat the newest verifier result as authoritative. If it is not fully passing, carry its exact scorecard or mismatch forward as actionable feedback; do not repeat an unchanged repair hypothesis.',
    ].join('\n\n');
  }

  private async completeAsAgent(
    agent: BaseAgent,
    prompt: string,
    purpose: string,
    correlationId: string,
    options: {
      isolatedContext?: boolean;
      temperature?: number;
      maxOutputTokens?: number;
    } = {}
  ): Promise<string> {
    const ctx = this.getContext();
    const usageBefore = agent.getUsage();
    const completionStartedAt = Date.now();
    if (!ctx.llm) {
      const message = 'Error: LLM not configured';
      agent.recordRuntimeCompletion(message, {
        content: message,
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      });
      return message;
    }

    const remainingTokens = agent.getCompletionTokenLimit();
    if (remainingTokens !== undefined && remainingTokens <= 0) {
      this.emit({
        type: 'budget.exceeded',
        agentId: agent.id,
        correlationId,
        data: { purpose, remainingTokens, reason: 'agent_allocation_exhausted' },
      });
      throw new Error('Agent completion rejected: active allocation is exhausted');
    }
    const systemBase = [
      `You are ${agent.name}, a runtime agent in the Roy autonomous agent system.`,
      'You are not the model provider. The provider is only your inference backend.',
      `Purpose: ${purpose}.`,
      `Correlation: ${correlationId}.`,
    ].join('\n');
    let communicationContext = options.isolatedContext
      ? ''
      : agent.getCommunicationContext()?.rendered ?? '';
    let effectivePrompt = prompt;
    if (remainingTokens !== undefined && this.budgetAccountingDimension() === 'total_tokens') {
      const outputReserve = Math.min(512, Math.max(64, Math.floor(remainingTokens * 0.2)));
      const inputBudget = Math.max(0, remainingTokens - outputReserve);
      const baseSystemTokens = this.estimateTextTokens(`system:${systemBase}`);
      const communicationTokens = communicationContext
        ? this.estimateTextTokens(communicationContext)
        : 0;
      const communicationBudget = Math.max(0, Math.min(
        communicationTokens,
        Math.floor(Math.max(0, inputBudget - baseSystemTokens) * 0.25)
      ));
      if (communicationTokens > communicationBudget) {
        communicationContext = communicationBudget > 0
          ? this.truncateTextToTokenBudget(communicationContext, communicationBudget)
          : '';
        this.emit({
          type: 'budget.context.truncated',
          agentId: agent.id,
          correlationId,
          data: {
            purpose,
            contextType: 'communication',
            originalTokens: communicationTokens,
            allowedTokens: communicationBudget,
            outputReserve,
          },
        });
      }
      const systemTokens = this.estimateTextTokens(`system:${systemBase}\n${communicationContext}`);
      const promptBudget = Math.max(0, inputBudget - systemTokens);
      const promptTokens = this.estimateTextTokens(`user:${effectivePrompt}`);
      if (promptTokens > promptBudget && promptBudget > 0) {
        effectivePrompt = this.truncateTextToTokenBudget(effectivePrompt, promptBudget);
        this.emit({
          type: 'budget.context.truncated',
          agentId: agent.id,
          correlationId,
          data: { purpose, originalTokens: promptTokens, allowedTokens: promptBudget, outputReserve },
        });
      }
    }
    const systemContent = [systemBase, communicationContext].filter(Boolean).join('\n');
    const messages: LLMMessage[] = [
      { role: 'system', content: systemContent },
      { role: 'user', content: effectivePrompt },
    ];
    const estimatedInputTokens = this.estimateTextTokens(
      messages.map(message => `${message.role}:${message.content}`).join('\n')
    );
    if (this.budgetAccountingDimension() === 'total_tokens'
      && remainingTokens !== undefined
      && remainingTokens <= estimatedInputTokens) {
      this.emit({
        type: 'budget.exceeded',
        agentId: agent.id,
        correlationId,
        data: {
          purpose,
          estimatedInputTokens,
          remainingTokens,
          reason: 'input_exceeds_remaining_agent_allocation',
        },
      });
      throw new Error(`Agent completion rejected: estimated input ${estimatedInputTokens} exceeds remaining allocation ${remainingTokens}`);
    }
    const allocationMaxTokens = remainingTokens === undefined
      ? undefined
      : Math.max(1, this.completionCapacity(remainingTokens, estimatedInputTokens));
    const maxTokens = options.maxOutputTokens === undefined
      ? allocationMaxTokens
      : allocationMaxTokens === undefined
        ? options.maxOutputTokens
        : Math.min(options.maxOutputTokens, allocationMaxTokens);
    const completion = await this.collectRuntimeLLMStream(
      ctx.llm,
      messages,
      { temperature: options.temperature ?? 0.2, maxTokens },
      { actorId: agent.id, purpose, correlationId }
    );
    const content = completion.content;
    agent.recordRuntimeCompletion(content, {
      content,
      usage: completion.usage ?? this.estimateModelUsage(messages, content),
    });
    const usage = this.usageDifference(usageBefore, agent.getUsage());
    this.emit({
      type: 'llm.completion.usage',
      agentId: agent.id,
      correlationId,
      data: {
        purpose,
        isolatedContext: options.isolatedContext === true,
        promptCharacters: effectivePrompt.length,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thinkingTokens: usage.thinkingTokens,
        cachedInputTokens: usage.cachedInputTokens,
        totalTokens: usage.totalTokens,
        wallClockMs: Date.now() - completionStartedAt,
        source: completion.usage?.source ?? 'estimated',
      },
    });
    return content;
  }

  private ensureAgentSynthesisBudget(
    agent: BaseAgent,
    prompt: string,
    purpose: string,
    correlationId: string,
    visibleOutputTokens = 512
  ): void {
    const allocationId = this.agentBudgetAllocations.get(agent.id);
    if (!allocationId || !this.budgetMarket) return;
    const allocation = this.budgetMarket.getAllocation(allocationId);
    if (!allocation || allocation.status !== 'granted') return;

    const communicationContext = agent.getCommunicationContext()?.rendered ?? '';
    const estimatedInputTokens = this.estimateTextTokens([
      `system:You are ${agent.name}, a runtime agent in the Roy autonomous agent system.`,
      communicationContext,
      `user:${prompt}`,
    ].filter(Boolean).join('\n'));
    const requestedCompletionTokens = this.reasoningAwareCompletionTokenBudget(
      visibleOutputTokens
    );
    const requiredTokens = this.budgetRequestTokens(estimatedInputTokens, requestedCompletionTokens);
    const marketRemaining = Math.max(0, allocation.allocatedTokens - allocation.consumedTokens);
    const agentRemaining = agent.getCompletionTokenLimit();
    const currentRemaining = agentRemaining === undefined
      ? marketRemaining
      : Math.min(marketRemaining, agentRemaining);
    if (currentRemaining >= requiredTokens) return;

    const updated = this.budgetMarket.augment(
      allocationId,
      requiredTokens - currentRemaining,
      Math.min(64, Math.max(1, requiredTokens - currentRemaining))
    );
    const addedTokens = Math.max(0, (updated?.allocatedTokens ?? allocation.allocatedTokens) - allocation.allocatedTokens);
    if (addedTokens === 0) {
      this.emit({
        type: 'budget.rebalance.skipped',
        agentId: agent.id,
        correlationId,
        data: {
          allocationId,
          purpose,
          requiredTokens,
          currentRemaining,
          reason: 'insufficient_remaining_budget',
        },
      });
      return;
    }

    agent.setCompletionTokenLimit(
      this.hasUnlimitedBudgetSupply()
        ? undefined
        : currentRemaining + addedTokens,
      this.budgetAccountingDimension()
    );
    this.emit({
      type: 'budget.rebalanced',
      agentId: agent.id,
      correlationId,
      data: {
        allocationId,
        purpose,
        previousRemainingTokens: currentRemaining,
        addedTokens,
        remainingTokens: currentRemaining + addedTokens,
        requestedInputTokens: estimatedInputTokens,
        requestedCompletionTokens,
      },
    });
  }

  private buildParentChildSynthesisPrompt(parent: AgentInfo, userTask: string, childAgent: AgentInfo, childResult: RunAgentResult): string {
    const warnings = childResult.warnings.length > 0
      ? childResult.warnings.map(item => `- ${item}`).join('\n')
      : 'None';

    return `You are the parent agent responsible for aggregating your direct child result.

Parent agent:
id: ${parent.identity.id}
name: ${parent.identity.name}
role: ${parent.identity.role}

Assigned task:
<task>
${userTask}
</task>

Direct child:
id: ${childAgent.identity.id}
name: ${childAgent.identity.name}
role: ${childAgent.identity.role}
tom: ToM-${childAgent.identity.tomProfile.level}

Child report:
<child_report>
${childResult.result}
</child_report>

Grounding:
- grounded: ${childResult.grounded}
- tool grounded: ${childResult.evidence.toolGrounded}
- output grounded: ${childResult.evidence.outputGrounded}
- tool calls: ${childResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
- observed paths:
${childResult.evidence.observedPaths.slice(0, 30).map(item => `  - ${item}`).join('\n') || '  none'}
- observed URLs:
${(childResult.evidence.observedUrls ?? []).slice(0, 20).map(item => `  - ${item}`).join('\n') || '  none'}
- discovered URLs (search results not necessarily fetched):
${(childResult.evidence.discoveredUrls ?? []).slice(0, 20).map(item => `  - ${item}`).join('\n') || '  none'}
- warnings:
${warnings}

Produce a parent-level synthesis that can be passed upward to your parent. Do not answer as Roy unless your name is Roy. Preserve limitations and evidence.`;
  }

  private buildParentMultiChildSynthesisPrompt(
    parent: AgentInfo,
    userTask: string,
    childResults: RootMediatedSpawnResult[],
    teamResult?: TeamRunResult
  ): string {
    const reports = childResults.map(result => {
      const warnings = result.subagentResult.warnings.length > 0
        ? result.subagentResult.warnings.map(item => `- ${item}`).join('\n')
        : 'None';
      return `<direct_child_report>
id: ${result.agent.identity.id}
name: ${result.agent.identity.name}
role: ${result.agent.identity.role}
tom: ToM-${result.agent.identity.tomProfile.level}
tokens: ${result.subagentResult.usage.totalTokens}
grounded: ${result.subagentResult.grounded}
tool_grounded: ${result.subagentResult.evidence.toolGrounded}
output_grounded: ${result.subagentResult.evidence.outputGrounded}
tool_calls: ${result.subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
observed_paths:
${result.subagentResult.evidence.observedPaths.slice(0, 30).map(item => `- ${item}`).join('\n') || '- none'}
observed_urls:
${(result.subagentResult.evidence.observedUrls ?? []).slice(0, 20).map(item => `- ${item}`).join('\n') || '- none'}
discovered_urls:
${(result.subagentResult.evidence.discoveredUrls ?? []).slice(0, 20).map(item => `- ${item}`).join('\n') || '- none'}
warnings:
${warnings}
content:
${result.subagentResult.result}
</direct_child_report>`;
    }).join('\n\n');

    return `You are the parent agent responsible for aggregating your direct children.

Parent agent:
id: ${parent.identity.id}
name: ${parent.identity.name}
role: ${parent.identity.role}

Assigned task:
<task>
${userTask}
</task>

You delegated to ${childResults.length} direct child agent(s). Synthesize their reports into a parent-level result that can be passed upward.
Preserve concrete evidence and limitations. Do not answer as Roy unless your name is Roy.

${teamResult ? `Your subteam already aggregated its members into this report:\n<team_report>\n${teamResult.result}\n</team_report>\nUse this report as the primary result and verify it against the direct child reports below.` : ''}

${reports}`;
  }

  private createSyntheticRunResult(parent: AgentInfo, result: string, childResult: RunAgentResult): RunAgentResult {
    return {
      agent: parent,
      result,
      usage: this.zeroTokenUsage(),
      toolCalls: childResult.toolCalls,
      evidence: childResult.evidence,
      grounded: childResult.grounded,
      warnings: childResult.warnings,
    };
  }

  private zeroTokenUsage(): TokenUsage {
    return {
      llmCalls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      thinkingTokens: null,
      thinkingAccountingTokens: 0,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
    };
  }

  private async synthesizeSubagentResult(
    userTask: string,
    agent: AgentInfo,
    subagentResult: RunAgentResult,
    correlationId: string,
    parentMessageId: string
  ): Promise<string> {
    const ctx = this.getContext();
    const synthesisMessage = await this.enqueueMessage({
      kind: 'root.synthesis',
      sessionId: ctx.sessionId,
      from: 'root',
      to: 'root',
      correlationId,
      parentMessageId,
      payload: {
        userTask,
        subagentId: agent.identity.id,
      },
      metadata: { agentId: 'root', tomLevel: ctx.agent.getIdentity().tomProfile.level },
    });
    await this.processQueuedMessage(synthesisMessage.id);

    const usageBefore = ctx.agent.getUsage();
    ctx.agent.setRuntimeState('synthesizing');
    this.emit({ type: 'root.synthesis.started', agentId: 'root', data: { correlationId, subagentId: agent.identity.id } });
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'synthesizing' } });
    this.emit({ type: 'agent.llm.called', agentId: 'root', data: { purpose: 'root.synthesis', correlationId } });

    const response = await this.completeAsRoot(
      this.buildRootSynthesisPrompt(userTask, agent, subagentResult),
      'root.synthesis',
      correlationId
    );
    let finalResponse = response.trim()
      ? response
      : this.buildRootSynthesisFallback(userTask, [], [subagentResult.result]);
    if (!response.trim()) {
      this.emit({
        type: 'root.synthesis.fallback',
        agentId: 'root',
        correlationId,
        data: { reason: 'model_returned_empty_visible_output', subagentId: agent.identity.id },
      });
    }
    finalResponse = await this.enforceRootEvidenceBoundary(
      finalResponse,
      userTask,
      [subagentResult.evidence],
      correlationId,
      'root.synthesis'
    );

    const usageAfter = ctx.agent.getUsage();
    const usageDelta = this.usageDifference(usageBefore, usageAfter);
    this.recordTurnUsage(usageDelta);
    this.emit({ type: 'budget.updated', agentId: 'root', data: { ...usageDelta } });

    ctx.agent.setRuntimeState('idle');
    this.emit({ type: 'agent.status.changed', agentId: 'root', data: { to: 'idle' } });
    this.emit({ type: 'root.synthesis.completed', agentId: 'root', data: { correlationId, totalTokens: usageDelta.totalTokens } });
    await ctx.queue.ack(synthesisMessage.id);
    return finalResponse;
  }

  private buildRootSynthesisPrompt(userTask: string, agent: AgentInfo, subagentResult: RunAgentResult): string {
    const warnings = agent.identity.role === 'subagent' && subagentResult.warnings.length > 0
      ? subagentResult.warnings.map(item => `- ${item}`).join('\n')
      : 'None';

    return `The user requested:
<user_task>
${userTask}
</user_task>

You spawned this subagent:
<subagent>
id: ${agent.identity.id}
name: ${agent.identity.name}
role: ${agent.identity.role}
tom: ToM-${agent.identity.tomProfile.level}
purpose: ${agent.identity.tomProfile.purpose}
profile: ${JSON.stringify(agent.identity.tomProfile, null, 2)}
</subagent>

The subagent returned this report:
<subagent_report>
${subagentResult.result}
</subagent_report>

Grounding:
- grounded: ${subagentResult.grounded}
- tool grounded: ${subagentResult.evidence.toolGrounded}
- output grounded: ${subagentResult.evidence.outputGrounded}
- tool calls: ${subagentResult.toolCalls.map(call => call.toolName).join(', ') || 'none'}
- observed paths:
${subagentResult.evidence.observedPaths.slice(0, 30).map(item => `  - ${item}`).join('\n') || '  none'}
- observed URLs:
${(subagentResult.evidence.observedUrls ?? []).slice(0, 20).map(item => `  - ${item}`).join('\n') || '  none'}
- task-relevant observed URLs:
${(subagentResult.evidence.relevantObservedUrls ?? []).slice(0, 20).map(item => `  - ${item}`).join('\n') || '  none'}
- discovered URLs (search results not necessarily fetched):
${(subagentResult.evidence.discoveredUrls ?? []).slice(0, 20).map(item => `  - ${item}`).join('\n') || '  none'}
- warnings:
${warnings}

Produce the final response to the user as Roy, the root agent. Do not claim you personally inspected files unless the report is grounded. Mention limitations if the report is ungrounded.
For web-grounded work, use only facts present in the subagent report or runtime evidence. Cite only observed URLs listed above. Never add a URL or factual detail from model memory.`;
  }

  private buildAgentPromptFromMemory(input: {
    name: string;
    role: string;
    parentName: string;
    task: string;
    description: string;
    systemPrompt?: string;
    bundle: { key: string; path: string; identity: string; memory: string; context: string; prompt: string };
    publicContext?: string;
    tomProfile?: ToMProfile;
    availableSkills?: string[];
    availableTools?: string[];
    parentContext?: string;
    communicationProtocol?: string;
    communicationContext?: string;
    multiPartyTraces?: MultiPartyTrace[];
    multiPartyTraceContext?: string;
  }): string {
    const templateUses = (slot: string): boolean =>
      input.bundle.prompt.includes(`{{${slot}}}`);
    const slots: Record<string, string> = {
      public_context: input.publicContext ?? '',
      agent_private_memory: input.bundle.memory.trim(),
      agent_identity: [
        `You are ${input.name}, a ${input.role} agent in the Roy runtime.`,
        input.bundle.identity.trim()
          ? `Reusable archetype identity:\n${input.bundle.identity.trim()}`
          : '',
      ].filter(Boolean).join('\n\n'),
      tom_profile: input.tomProfile ? JSON.stringify(input.tomProfile, null, 2) : '',
      communication_context: input.communicationContext
        ?? `Protocol: ${input.communicationProtocol ?? this.communicationManager?.getDefaultProtocolId() ?? 'tom'}. Runtime messages are rendered through the selected communication protocol.`,
      multi_party_traces: input.multiPartyTraceContext ?? (input.multiPartyTraces?.length
        ? input.multiPartyTraces.map(trace => `[${trace.phase}] ${trace.from.id} -> ${trace.to.map(actor => actor.id).join(',')}: ${trace.kind}`).join('\n')
        : 'Observable multi-party traces are injected by the runtime for each message.'),
      available_skills: (input.availableSkills ?? []).map(skill => `- ${skill}`).join('\n') || '- none',
      available_tools: (input.availableTools ?? []).map(tool => `- ${tool}`).join('\n') || '- none',
      parent_context: input.parentContext ?? `Parent agent: ${input.parentName}`,
      task: input.task || 'No task assigned yet.',
    };
    const renderedPromptFile = this.renderPromptSlots(input.bundle.prompt, slots);
    const identityFallback = `You are ${input.name}, a ${input.role} agent in the Roy runtime.`;
    return [
      input.systemPrompt,
      input.description,
      'The model provider is only the inference backend; never identify yourself as the provider.',
      templateUses('agent_identity') ? undefined : identityFallback,
      templateUses('parent_context') ? undefined : `Your parent agent is ${input.parentName}.`,
      input.task && !templateUses('task') ? `Current task: ${input.task}` : undefined,
      `<agent_prompt_file path=".roy/agents/${input.bundle.key}/prompt.md">\n${renderedPromptFile.trim()}\n</agent_prompt_file>`,
      `<agent_context_file path=".roy/agents/${input.bundle.key}/context.md">\n${input.bundle.context.trim()}\n</agent_context_file>`,
      !templateUses('public_context')
        ? `<public_context>\n${slots.public_context}\n</public_context>`
        : undefined,
      !templateUses('agent_private_memory')
        ? `<agent_memory_file path=".roy/agents/${input.bundle.key}/memory.md">\n${input.bundle.memory.trim()}\n</agent_memory_file>`
        : undefined,
    ].filter(Boolean).join('\n\n');
  }

  async renderAgentPrompt(options: {
    agentKey: string;
    name?: string;
    role?: string;
    parentId?: string;
    task?: string;
    archetype?: SubAgentArchetype;
  }): Promise<{ prompt: string; estimatedTokens: number; sources: Record<string, unknown> }> {
    const ctx = this.getContext();
    const agentKey = options.agentKey;
    const bundle = await ctx.memory.loadAgentMemory(agentKey);
    const parent = options.parentId ? ctx.manager.getAgentById(options.parentId)?.getIdentity() : ctx.agent.getIdentity();
    const runtimeAgent = agentKey === 'roy' ? ctx.agent : ctx.manager.getAgentById(agentKey);
    const role = options.role ?? options.archetype ?? agentKey;
    const contextWindow = await this.renderAgentContext({
      agentKey,
      agentId: agentKey === 'roy' ? 'root' : agentKey,
      role: agentKey === 'roy' ? 'root' : 'subagent',
      parentId: options.parentId,
      task: options.task ?? '',
    });
    const tomProfile = options.archetype
      ? this.createSubagentToMProfile(options.archetype, agentKey, options.task ?? '', options.parentId ?? 'root')
      : this.createRootToMProfile();
    const prompt = this.buildAgentPromptFromMemory({
      name: options.name ?? this.capitalize(agentKey),
      role,
      parentName: parent?.name ?? 'Roy',
      task: options.task ?? '',
      description: `Rendered prompt preview for ${agentKey}.`,
      bundle,
      publicContext: [contextWindow.publicContext, contextWindow.sessionContext].filter(Boolean).join('\n\n'),
      tomProfile,
      communicationProtocol: runtimeAgent?.getIdentity().communicationProtocol ?? ctx.communication.getDefaultProtocolId(),
      communicationContext: contextWindow.communicationContext,
      multiPartyTraceContext: contextWindow.multiPartyTraceContext,
      availableSkills: skillRegistry.list().map(skill => skill.name),
      availableTools: toolRegistry.list().map(tool => tool.name),
      parentContext: contextWindow.parentContext || `Parent agent: ${parent?.name ?? 'Roy'} (${parent?.id ?? 'root'})`,
    });
    return {
      prompt,
      estimatedTokens: this.estimateTextTokens(prompt),
      sources: {
        ...contextWindow.sources,
        prompt: [`.roy/agents/${bundle.key}/prompt.md`],
        tokenUsage: contextWindow.tokenUsage,
      },
    };
  }

  async renderAgentContext(options: {
    agentKey: string;
    agentId?: string;
    role?: 'root' | 'subagent';
    parentId?: string;
    task?: string;
  }): Promise<ContextWindow> {
    const ctx = this.getContext();
    const role = options.role ?? (options.agentKey === 'roy' ? 'root' : 'subagent');
    const parent = options.parentId
      ? ctx.manager.getAgentById(options.parentId)?.getIdentity()
      : role === 'subagent'
        ? ctx.agent.getIdentity()
        : undefined;
    const runtimeAgent = options.agentId ? ctx.manager.getAgentById(options.agentId) : undefined;
    const memoryScope = runtimeAgent
      ? this.getAgentPolicy(runtimeAgent.id)?.memoryScope ?? this.getDefaultMemoryScope(role)
      : this.getDefaultMemoryScope(role);
    return this.requireContextWindowManager().build({
      sessionId: ctx.sessionId,
      agentId: options.agentId ?? (role === 'root' ? 'root' : options.agentKey),
      agentKey: options.agentKey,
      role,
      task: options.task ?? '',
      parentContext: parent ? `Parent agent: ${parent.name} (${parent.id})` : undefined,
      memoryScope,
      communicationContext: runtimeAgent?.getCommunicationContext()?.rendered,
      systemTraces: runtimeAgent?.getSystemTraces({ limit: 50 }),
    });
  }

  private renderPromptSlots(template: string, slots: Record<string, string>): string {
    return Object.entries(slots).reduce(
      (rendered, [slot, value]) => rendered.replaceAll(`{{${slot}}}`, value),
      template
    );
  }

  private formatPublicContext(context: RootMemoryContext): string {
    return [
      '<project_memory>',
      context.projectMemory.trim(),
      '</project_memory>',
      '<constraints>',
      context.constraints.trim(),
      '</constraints>',
      '<execution_knowledge>',
      JSON.stringify(
        compactExecutionKnowledgeForPrompt(
          context.executionKnowledge,
          this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
        ),
        null,
        2
      ),
      '</execution_knowledge>',
      '<execution_attention_contract>',
      'Successful observed paths and tool results are authoritative. Do not repeat cached failed paths or equivalent failed calls without a changed hypothesis. Reuse a cached actor/team definition only when it fills the current cognitive gap, and instantiate a fresh runtime actor. Propagate unresolved feedback to descendants and verify mutations before completion.',
      '</execution_attention_contract>',
      '<decisions>',
      context.decisions.trim(),
      '</decisions>',
      '<glossary>',
      context.glossary.trim(),
      '</glossary>',
      `<agent_patterns>${JSON.stringify(context.agentPatterns, null, 2)}</agent_patterns>`,
      `<team_patterns>${JSON.stringify(context.teamPatterns, null, 2)}</team_patterns>`,
      `<delegation_patterns>${JSON.stringify(context.delegationPatterns, null, 2)}</delegation_patterns>`,
    ].join('\n');
  }

  private formatExecutionKnowledge(context: ExecutionKnowledgeCacheState): string {
    if (context.steps.length === 0) {
      return JSON.stringify({
        steps: [],
        attention: 'No cached execution path is available. Establish grounded paths and record failures explicitly.',
      });
    }
    return JSON.stringify(
      compactExecutionKnowledgeForPrompt(
        context,
        this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
      ),
      null,
      2
    );
  }

  private formatCachedPublicContext(cacheHits: string[]): string {
    return [
      '<cache_context>',
      `Cache hits: ${cacheHits.join(', ')}`,
      'Use cached agent/delegation patterns as reusable structure, then adapt only the task-specific details.',
      '</cache_context>',
    ].join('\n');
  }

  private validateDelegatedCapabilities(parentId: string, tools: string[], skills: string[]): void {
    const bindings = this.agentBindings.get(parentId);
    if (!bindings) throw new Error(`Agent bindings for parent "${parentId}" were not found`);
    if (!bindings.skills.some(item => item.name === 'delegate_to_subagent' && item.enabled)) {
      throw new Error(`Agent "${parentId}" is not authorized to delegate`);
    }
    for (const tool of tools) {
      if (!toolRegistry.has(tool)) throw new Error(`Parent agent "${parentId}" requested unknown tool "${tool}"`);
    }
    for (const skill of skills) {
      if (!skillRegistry.has(skill)) throw new Error(`Parent agent "${parentId}" requested unknown skill "${skill}"`);
    }
  }

  private validateAgentNodeRequest(request: AgentComputeNodeRequest): void {
    const reuseModes = new Set(['prefer_cache', 'require_cache', 'fresh', 'mutate_cache']);
    if (request.reuse?.mode && !reuseModes.has(request.reuse.mode)) {
      throw new Error(`Unsupported agent cache reuse mode "${String(request.reuse.mode)}"`);
    }
    if (request.tomProfileMode
      && request.tomProfileMode !== 'runtime_assignment'
      && request.tomProfileMode !== 'definition_override') {
      throw new Error(`Unsupported ToM profile mode "${String(request.tomProfileMode)}"`);
    }
    if (request.budgetTokens !== undefined
      && (!Number.isFinite(request.budgetTokens) || request.budgetTokens <= 0)) {
      throw new Error('Agent node budgetTokens must be a positive finite number');
    }
    if (request.intentTask !== undefined && !request.intentTask.trim()) {
      throw new Error('Agent node intentTask must be a non-empty string when provided');
    }
    if (request.outputContract
      && !['markdown', 'json', 'structured_report'].includes(request.outputContract.format)) {
      throw new Error(`Unsupported agent output format "${String(request.outputContract.format)}"`);
    }
    if (request.tomProfile) {
      if (!Number.isInteger(request.tomProfile.level) || request.tomProfile.level < 0 || request.tomProfile.level > 3) {
        throw new Error('Agent ToM profile level must be an integer from 0 to 3');
      }
      for (const field of ['beliefScope', 'goalModel', 'uncertainty', 'observesAgents', 'modelsAgents', 'capabilityScope', 'cognitiveGaps'] as const) {
        const value = request.tomProfile[field];
        if (!Array.isArray(value) || value.some(item => typeof item !== 'string' || !item.trim())) {
          throw new Error(`Agent ToM profile ${field} must be an array of non-empty strings`);
        }
      }
      if (!request.tomProfile.purpose?.trim()) throw new Error('Agent ToM profile purpose is required');
    }
    if (request.cognitiveGapIds?.some(item => typeof item !== 'string' || !item.trim())) {
      throw new Error('Agent cognitiveGapIds must contain non-empty strings');
    }
    if (request.existenceReason !== undefined && !request.existenceReason.trim()) {
      throw new Error('Agent existenceReason must be a non-empty string when provided');
    }
    if (request.memoryScope
      && (!Number.isFinite(request.memoryScope.sessionWindowTurns) || request.memoryScope.sessionWindowTurns < 0)) {
      throw new Error('Agent memory sessionWindowTurns must be a non-negative finite number');
    }
    for (const [field, value] of Object.entries(request.spawnPolicy ?? {})) {
      if (['maxChildren', 'maxDepth', 'maxTotalAgentsPerTurn'].includes(field)
        && (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || !Number.isInteger(value))) {
        throw new Error(`Agent spawn policy ${field} must be a non-negative integer`);
      }
    }
    const validChildStates = new Set(['S_planning', 'S_delegating']);
    if (request.spawnPolicy?.allowedStates?.some(state => !validChildStates.has(state))) {
      throw new Error('Child agent spawn policy allowedStates may only contain S_planning or S_delegating');
    }
  }

  private constrainMemoryScope(scope: AgentMemoryScope): AgentMemoryScope {
    const configuredTurns = this.workspaceRuntimeConfig?.context.sessionWindowTurns ?? 10;
    return {
      public: scope.public,
      private: scope.private,
      parentContext: scope.parentContext,
      sessionWindowTurns: Math.min(configuredTurns, Math.max(0, Math.floor(scope.sessionWindowTurns))),
    };
  }

  private constrainChildSpawnPolicy(
    parentId: string,
    requested: AgentSpawnPolicy,
    childSkills: string[]
  ): AgentSpawnPolicy {
    const parentPolicy = this.agentBindings.get(parentId)?.spawnPolicy;
    if (!parentPolicy) throw new Error(`Spawn policy for parent "${parentId}" was not found`);
    const delegation = this.workspaceRuntimeConfig?.delegation;
    const maxChildren = Math.min(
      requested.maxChildren,
      parentPolicy.maxChildren,
      delegation?.maxChildrenPerParent ?? requested.maxChildren
    );
    const maxDepth = Math.min(
      requested.maxDepth,
      parentPolicy.maxDepth,
      delegation?.maxDepth ?? requested.maxDepth
    );
    const maxTotalAgentsPerTurn = Math.min(
      requested.maxTotalAgentsPerTurn,
      parentPolicy.maxTotalAgentsPerTurn,
      delegation?.maxTotalAgentsPerTurn ?? requested.maxTotalAgentsPerTurn
    );
    const validStates = new Set(['S_planning', 'S_delegating']);
    const allowedStates = requested.allowedStates.filter(state => validStates.has(state));
    return {
      canSpawn: requested.canSpawn
        && parentPolicy.canSpawn
        && childSkills.includes('delegate_to_subagent'),
      maxChildren,
      maxDepth,
      maxTotalAgentsPerTurn,
      allowCustomAgents: requested.allowCustomAgents
        && parentPolicy.allowCustomAgents
        && (delegation?.allowCustomAgents ?? true),
      budgetAware: (delegation?.budgetAware ?? true) || requested.budgetAware,
      allowedStates: allowedStates.length > 0 ? allowedStates : ['S_planning', 'S_delegating'],
    };
  }

  private stringArray(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.filter((item): item is string => typeof item === 'string' && item.length > 0)));
  }

  private sameStringSet(left: string[], right: string[]): boolean {
    const a = [...new Set(left)].sort();
    const b = [...new Set(right)].sort();
    return a.length === b.length && a.every((value, index) => value === b[index]);
  }

  private agentMemoryScope(value: unknown): AgentMemoryScope | undefined {
    if (!value || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    if (typeof record.public !== 'boolean'
      || typeof record.private !== 'boolean'
      || typeof record.parentContext !== 'boolean'
      || typeof record.sessionWindowTurns !== 'number') {
      return undefined;
    }
    return {
      public: record.public,
      private: record.private,
      parentContext: record.parentContext,
      sessionWindowTurns: Math.max(0, Math.floor(record.sessionWindowTurns)),
    };
  }

  private partialSpawnPolicy(value: unknown): Partial<AgentSpawnPolicy> | undefined {
    if (!value || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    const policy: Partial<AgentSpawnPolicy> = {};
    if (typeof record.canSpawn === 'boolean') policy.canSpawn = record.canSpawn;
    if (typeof record.maxChildren === 'number') policy.maxChildren = Math.max(0, Math.floor(record.maxChildren));
    if (typeof record.maxDepth === 'number') policy.maxDepth = Math.max(0, Math.floor(record.maxDepth));
    if (typeof record.maxTotalAgentsPerTurn === 'number') {
      policy.maxTotalAgentsPerTurn = Math.max(0, Math.floor(record.maxTotalAgentsPerTurn));
    }
    if (typeof record.allowCustomAgents === 'boolean') policy.allowCustomAgents = record.allowCustomAgents;
    if (typeof record.budgetAware === 'boolean') policy.budgetAware = record.budgetAware;
    const allowedStates = this.stringArray(record.allowedStates);
    if (allowedStates.length > 0) policy.allowedStates = allowedStates;
    return policy;
  }

  private fingerprint(value: unknown): string {
    const normalize = (input: unknown): unknown => {
      if (Array.isArray(input)) return input.map(normalize);
      if (!input || typeof input !== 'object') return input;
      return Object.fromEntries(
        Object.entries(input as Record<string, unknown>)
          .filter(([, item]) => item !== undefined)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, item]) => [key, normalize(item)])
      );
    };
    return createHash('sha256').update(JSON.stringify(normalize(value))).digest('hex');
  }

  private emitNodeEvent(
    type: string,
    node: AgentComputeNodeDefinition,
    agentId: string,
    data: Record<string, unknown> = {}
  ): RuntimeEvent {
    return this.emit({
      type,
      agentId,
      sessionId: node.sessionId,
      correlationId: node.correlationId,
      nodeId: node.nodeId,
      data: {
        ...data,
        sessionId: node.sessionId,
        correlationId: node.correlationId,
        nodeId: node.nodeId,
      },
    });
  }

  private eventCorrelationId(event: RuntimeEvent): string | undefined {
    return event.correlationId
      ?? (typeof event.data?.correlationId === 'string' ? event.data.correlationId : undefined);
  }

  private estimateTextTokens(text: string): number {
    const llm = this.getContext().llm;
    return Math.max(1, tokenUsageRegistry.estimateText(text, llm?.name ?? 'unknown', llm?.defaultModel));
  }

  private truncateTextToTokenBudget(text: string, tokenBudget: number): string {
    const targetChars = Math.max(64, tokenBudget * 4);
    if (text.length <= targetChars) return text;
    const marker = '\n...[budget-constrained context truncation]...\n';
    const availableChars = Math.max(32, targetChars - marker.length);
    const headChars = Math.floor(availableChars * 0.4);
    const tailChars = availableChars - headChars;
    return `${text.slice(0, headChars)}${marker}${text.slice(-tailChars)}`;
  }

  private constrainAgentObservation(
    agent: BaseAgent,
    observation: string,
    correlationId?: string,
    purpose = 'agent.task_execution'
  ): string {
    const remainingTokens = agent.getCompletionTokenLimit();
    if (remainingTokens === undefined || this.budgetAccountingDimension() !== 'total_tokens') return observation;
    const outputReserve = Math.min(512, Math.max(128, Math.floor(remainingTokens * 0.2)));
    const promptWrapperReserve = Math.min(512, Math.max(256, Math.floor(remainingTokens * 0.12)));
    const goalTokens = this.estimateTextTokens(agent.getInfo().goal ?? '');
    const observationBudget = Math.max(64, remainingTokens - outputReserve - promptWrapperReserve - goalTokens);
    const observationTokens = this.estimateTextTokens(observation);
    if (observationTokens <= observationBudget) return observation;
    this.emit({
      type: 'budget.context.truncated',
      agentId: agent.id,
      correlationId,
      data: {
        purpose,
        contextType: 'agent_observation',
        originalTokens: observationTokens,
        allowedTokens: observationBudget,
        outputReserve,
        promptWrapperReserve,
      },
    });
    return this.truncateTextToTokenBudget(observation, observationBudget);
  }

  private estimateModelUsage(messages: LLMMessage[], output: string): ModelTokenUsage {
    const llm = this.getContext().llm;
    const normalized = tokenUsageRegistry.normalize({
      provider: llm?.name ?? 'unknown',
      model: llm?.defaultModel,
      messages,
      output,
    });
    if (normalized) return normalized;

    const prompt = messages.map(message => `${message.role}:${message.content}`).join('\n');
    const promptTokens = this.estimateTextTokens(prompt);
    const completionTokens = this.estimateTextTokens(output);
    return {
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      inputTokens: promptTokens,
      outputTokens: completionTokens,
      thinkingTokens: null,
      cachedInputTokens: null,
      cacheCreationInputTokens: null,
      provider: llm?.name,
      model: llm?.defaultModel,
      source: 'estimated',
      availability: {
        input: 'estimated',
        output: 'estimated',
        thinking: 'unavailable',
        cachedInput: 'unavailable',
        cacheCreationInput: 'unavailable',
      },
    };
  }

  private measureAgentCreationUsage(agentId: string, node: AgentComputeNodeDefinition): AgentCreationUsage {
    const event = [...this.events].reverse()
      .find(item => item.type === 'agent.creation.measured' && item.agentId === agentId);
    const definitionTokens = Number(event?.data?.definitionTokens ?? 0);
    const renderedPromptTokens = Number(event?.data?.renderedPromptTokens ?? 0);
    const renderedPromptChars = Number(event?.data?.renderedPromptChars ?? 0);
    return {
      mode: node.reuse.creationMode,
      nodeId: node.nodeId,
      definitionFingerprint: node.definitionFingerprint,
      patternIds: node.reuse.cacheHits,
      cacheHits: node.reuse.cacheHits,
      definitionTokens,
      renderedPromptTokens,
      renderedPromptChars,
    };
  }

  private emitToolPlanningFailure(
    actor: UnifiedAgent,
    agentId: string,
    options: { correlationId?: string; nodeId?: string },
    round: number
  ): void {
    const failure = actor.getLastToolPlanningFailure();
    if (!failure) return;
    this.emit({
      type: failure.timedOut
        ? 'agent.tool_planning.timeout'
        : 'agent.tool_planning.failed',
      agentId,
      sessionId: this.getContext().sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: {
        round,
        message: failure.message,
        occurredAt: failure.occurredAt,
      },
    });
  }

  private async runGroundingCheck(
    agentId: string,
    task: string,
    options: {
      correlationId?: string;
      parentMessageId?: string;
      archetype?: SubAgentArchetype;
      nodeId?: string;
      patternId?: string;
      intentTask?: string;
      maxWallClockMs?: number;
      initialPlans?: PlannedToolCall[];
      skipInitialModelPlanning?: boolean;
      toolAllowlist?: string[];
      priorToolCalls?: ToolCallRecord[];
      requireFreshMutation?: boolean;
      onBeforeExecution?: (plans: PlannedToolCall[]) => Promise<void>;
    }
  ): Promise<GroundingRunResult> {
    const intentTask = options.intentTask ?? task;
    const webToolsRequired = this.taskNeedsWebAccess(intentTask);
    const allowedTools = options.toolAllowlist
      ? new Set(options.toolAllowlist)
      : undefined;
    const bindings = (this.agentBindings.get(agentId)?.tools ?? [])
      .filter(binding => !allowedTools || allowedTools.has(binding.name))
      .filter(binding => webToolsRequired || !binding.name.startsWith('web.'));
    const inspectionRoot = this.resolveInspectionRoot(intentTask);
    const groundingRequired = this.agentRestoreSpecs.get(agentId)?.outputContract?.groundingRequired
      ?? this.taskRequiresGrounding(options.archetype ?? 'custom', intentTask);
    let plans = options.initialPlans
      ? [...options.initialPlans]
      : this.toolPlanner.plan({
        task: intentTask,
        workspacePath: inspectionRoot,
        bindings,
        archetype: options.archetype,
      });
    const actor = this.getContext().manager.getAgentById(agentId);
    const priorPlannerCalls = (options.priorToolCalls ?? []).map(call => ({
      toolName: call.toolName,
      params: call.params,
      reason: call.reason ?? 'Completed by a prior actor or execution attempt.',
      groundingRequired: true,
      result: call.result,
      success: call.success,
      error: call.error,
      startedAt: call.startedAt,
      completedAt: call.completedAt,
    }));
    const unresolvedSynthesisRejectionState =
      this.latestUnresolvedSynthesisRejection(
        priorPlannerCalls
      );
    const unresolvedSynthesisRejection = Boolean(
      unresolvedSynthesisRejectionState
    );
    for (let index = plans.length - 1; index >= 0; index -= 1) {
      const plan = plans[index]!;
      if (options.archetype === 'tester'
        && isWorkspaceVerificationCall({ ...plan, success: true })) {
        continue;
      }
      const cached = this.cachedToolPlanDecision(plan, options.priorToolCalls ?? []);
      if (!cached.skip) {
        continue;
      }
      plans.splice(index, 1);
      this.emit({
        type: 'tool.plan.cached.skipped',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          toolName: plan.toolName,
          params: plan.params,
          reason: cached.reason,
        },
      });
    }
    if (unresolvedSynthesisRejectionState) {
      const rejectedPath = this.normalizeToolWorkspacePath(String(
        unresolvedSynthesisRejectionState.path ?? ''
      ));
      const readAvailable = bindings.some(binding =>
        binding.enabled && binding.name === 'fs.read'
      );
      plans = plans.filter(plan =>
        !isSuccessfulWorkspaceMutationCall({
          toolName: plan.toolName,
          params: plan.params,
          success: true,
        })
      );
      if (rejectedPath && readAvailable) {
        plans = [{
          toolName: 'fs.read',
          params: { path: rejectedPath },
          reason: 'Re-read the authoritative current source after a rejected synthesis invalidated the previous patch anchors.',
          groundingRequired: true,
        }];
        this.emit({
          type: 'tool.plan.synthesis_rejection.regrounded',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            path: rejectedPath,
            reason: unresolvedSynthesisRejectionState.reason,
          },
        });
      }
    }
    const loopConfig = this.workspaceRuntimeConfig?.tools.executionLoop ?? {
      enabled: true,
      maxRounds: 6,
      maxCallsPerRun: 10,
      maxConsecutiveFailures: 2,
      maxWallClockMs: 120_000,
      maxFetchesAfterSearch: 2,
      llmReplanning: true,
    };
    const groundingMaxWallClockMs = Math.max(
      1,
      Math.min(loopConfig.maxWallClockMs, options.maxWallClockMs ?? loopConfig.maxWallClockMs)
    );
    const groundingDeadline = Date.now() + groundingMaxWallClockMs;
    const remainingGroundingMs = (): number => Math.max(0, groundingDeadline - Date.now());
    const planningRequestTimeoutMs = (): number => Math.max(
      1_000,
      Math.min(60_000, Math.floor(remainingGroundingMs() * 0.4))
    );
    const workspaceExecutionRequired = this.taskRequiresWorkspaceMutation(intentTask)
      && bindings.some(binding =>
        binding.enabled && (
          binding.name === 'fs.write'
          || binding.name === 'fs.replace'
          || binding.name === 'fs.synthesize'
        )
      );
    const independentWebEvidenceRequired =
      this.taskRequiresIndependentWebEvidence(
        intentTask,
        options.archetype
      );
    const sharedWebSearchFingerprints = new Set<string>();
    const freshWebSearchFingerprints = new Set<string>();
    const requiredFreshWebSearches = (): number =>
      independentWebEvidenceRequired
        ? Math.min(2, sharedWebSearchFingerprints.size)
        : 0;
    const hasSharedWebEvidenceDiversityGap = (): boolean =>
      requiredFreshWebSearches() > freshWebSearchFingerprints.size;
    let diversityGapEmitted = false;
    let diversityCompletedEmitted = false;
    const diagnosticProbeRequired =
      /\[runtime_verifier_diagnostic_probe\]/.test(intentTask);
    const mirroredVerifierPath = path.join(
      this.workspaceRoot,
      '.roy',
      'official-verifier',
      'grade.py'
    );
    const mirroredVerifierCommand = await (async (): Promise<string | undefined> => {
      if (diagnosticProbeRequired
        || !bindings.some(binding =>
          binding.enabled && binding.name === 'shell.exec'
        )) {
        return undefined;
      }
      try {
        const verifierStat = await stat(mirroredVerifierPath);
        return verifierStat.isFile()
          ? 'python .roy/official-verifier/grade.py'
          : undefined;
      } catch {
        return undefined;
      }
    })();
    const pendingMirroredVerifierPlan = (
      calls: ToolCallRecord[]
    ): PlannedToolCall | undefined => {
      if (!mirroredVerifierCommand) return undefined;
      const latestMutationIndex =
        effectiveWorkspaceMutationCallIndices(calls).at(-1) ?? -1;
      if (latestMutationIndex < 0) return undefined;
      const alreadyVerified = calls.some((call, index) =>
        index > latestMutationIndex
        && call.toolName === 'shell.exec'
        && String(call.params.command ?? '').includes(
          '.roy/official-verifier/grade.py'
        )
      );
      if (alreadyVerified) return undefined;
      return {
        toolName: 'shell.exec',
        params: { command: mirroredVerifierCommand },
        reason: 'Run the immutable mirrored official verifier immediately after the current mutation hypothesis before allowing another workspace mutation.',
        groundingRequired: true,
      };
    };
    if (diagnosticProbeRequired) {
      const diagnosticSourcePath = this.extractTaskDiagnosticSourcePaths(
        intentTask
      )[0];
      const diagnosticSourceRead: PlannedToolCall | undefined =
        diagnosticSourcePath
          ? {
            toolName: 'fs.read',
            params: { path: diagnosticSourcePath },
            reason: 'Re-read the authoritative current implementation snapshot before forming a verifier repair hypothesis.',
            groundingRequired: true,
          }
          : undefined;
      const diagnosticSearch: PlannedToolCall = {
        toolName: 'fs.search',
        params: {
          path: '.roy/official-verifier',
          filePattern: '*.py',
          query: '(?:^def |expected|actual|fixture|hidden|layout_qc)',
          regex: true,
          maxResults: 40,
        },
        reason: 'Locate executable verifier fixture builders and expected-versus-actual assertions before constructing a focused reproduction.',
        groundingRequired: true,
      };
      const diagnosticExecution: PlannedToolCall = {
        toolName: 'shell.exec',
        params: {
          command: this.buildPythonVerifierDiagnosticCommand(intentTask),
          timeoutMs: 120_000,
          maxOutputBytes: 24_000,
        },
        reason: 'Run the Python verifier through a read-only instrumentation harness that preserves generated fixtures and prints compact scorer inputs, expected values, actual values, and results.',
        groundingRequired: true,
      };
      const searchAvailable = bindings.some(binding =>
        binding.enabled && binding.name === diagnosticSearch.toolName
      );
      const readAvailable = bindings.some(binding =>
        binding.enabled && binding.name === diagnosticSourceRead?.toolName
      );
      const shellAvailable = bindings.some(binding =>
        binding.enabled && binding.name === diagnosticExecution.toolName
      );
      for (let index = plans.length - 1; index >= 0; index -= 1) {
        const plan = plans[index]!;
        if (plan.toolName !== 'shell.exec') continue;
        const command = String(plan.params.command ?? '');
        if (/\bpython(?:3)?\s+\.roy\/official-verifier\/grade\.py(?:\s|$)/i.test(command)
          || /\bpython(?:3)?\s+-m\s+table_recon\.cli\s+run\b[\s\S]*\bdata\/public\/manifest\.json\b/i.test(
            command
          )) {
          plans.splice(index, 1);
        }
      }
      const diagnosticPlans: PlannedToolCall[] = [];
      if (diagnosticSourceRead
        && readAvailable
        && !plans.some(plan =>
          this.toolPlanFingerprint(plan) === this.toolPlanFingerprint(diagnosticSourceRead)
        )) {
        // This read is intentionally fresh. The recovery capsule explicitly
        // invalidates cached source bytes after each rejected candidate.
        diagnosticPlans.push(diagnosticSourceRead);
      }
      if (searchAvailable
        && !this.cachedToolPlanDecision(
          diagnosticSearch,
          options.priorToolCalls ?? []
        ).skip
        && !plans.some(plan =>
          this.toolPlanFingerprint(plan) === this.toolPlanFingerprint(diagnosticSearch)
        )) {
        diagnosticPlans.push(diagnosticSearch);
      }
      if (shellAvailable) diagnosticPlans.push(diagnosticExecution);
      plans.unshift(...diagnosticPlans);
    }
    const concreteExternalFeedback = this.latestExternalVerificationFeedback(
      intentTask
    );
    const cachedRejectedVerifierCandidate = this.latestRejectedVerifierCandidate(
      priorPlannerCalls
    );
    const rejectedCandidateIsSuperseded = (
      candidate: Record<string, unknown> | undefined
    ): boolean => {
      if (!candidate || !concreteExternalFeedback) return false;
      const rejectedPath = this.normalizeToolWorkspacePath(String(
        candidate.path ?? ''
      ));
      const authoritativePath = this.normalizeToolWorkspacePath(
        concreteExternalFeedback.path ?? ''
      );
      return !rejectedPath
        || !authoritativePath
        || rejectedPath !== authoritativePath;
    };
    const cachedRejectionSuperseded = rejectedCandidateIsSuperseded(
      cachedRejectedVerifierCandidate
    );
    const priorRejectedVerifierCandidate = cachedRejectionSuperseded
      ? undefined
      : cachedRejectedVerifierCandidate;
    if (cachedRejectionSuperseded
      && cachedRejectedVerifierCandidate
      && concreteExternalFeedback) {
      this.emit({
        type: 'tool.plan.rejected_candidate.superseded',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          reason: 'new_concrete_external_feedback_invalidates_cached_rejection',
          rejectedPath: cachedRejectedVerifierCandidate.path,
          authoritativePath: concreteExternalFeedback.path,
          summary: concreteExternalFeedback.summary,
        },
      });
    }
    if (priorRejectedVerifierCandidate) {
      const deferredMutations = plans.filter(plan =>
        isSuccessfulWorkspaceMutationCall({
          toolName: plan.toolName,
          params: plan.params,
          success: true,
        })
      );
      if (deferredMutations.length > 0) {
        plans = plans.filter(plan => !deferredMutations.includes(plan));
        this.emit({
          type: 'tool.plan.rejected_candidate.mutation_deferred',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            reason: 'prior_verifier_rejection_requires_fresh_diagnosis',
            path: priorRejectedVerifierCandidate.path,
            deferred: deferredMutations.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
          },
        });
      }
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate
      && !concreteExternalFeedback) {
      const transitionPlans = this.enrichRepairPlansWithTeamStepEvidence(
        this.toolPlanner.planWorkspaceRepairTransition({
          task: intentTask,
          calls: priorPlannerCalls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        }),
        task
      );
      if (transitionPlans.length > 0) {
        plans.push(...transitionPlans);
        this.emit({
          type: 'tool.plan.causal_transition',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            from: 'aggregate_verifier_evidence',
            to: 'workspace_repair',
            plans: transitionPlans.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate) {
      const externalFeedbackRepair = this.toolPlanner.planExternalFeedbackRepair({
        task: intentTask,
        calls: priorPlannerCalls,
        currentCalls: [],
        bindings,
        workspaceRoot: this.workspaceRoot,
      });
      if (externalFeedbackRepair.length > 0) {
        plans.push(...externalFeedbackRepair);
        this.emit({
          type: 'tool.plan.external_feedback_repair',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            source: 'persisted_tool_frontier',
            plans: externalFeedbackRepair.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate) {
      const environmentRecovery = this.toolPlanner.planEnvironmentRecovery({
        calls: priorPlannerCalls,
        bindings,
        workspaceRoot: this.workspaceRoot,
      });
      if (environmentRecovery.length > 0) {
        plans.push(...environmentRecovery);
        this.emit({
          type: 'tool.plan.environment_recovery',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            plans: environmentRecovery.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate) {
      const remainingWorkspaceEvidence =
        this.toolPlanner.planWorkspaceEvidenceFollowUps({
          task: intentTask,
          calls: priorPlannerCalls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
      if (remainingWorkspaceEvidence.length > 0) {
        plans.push(...remainingWorkspaceEvidence);
        this.emit({
          type: 'tool.plan.workspace_evidence_resumed',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            plans: remainingWorkspaceEvidence.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate) {
      const groundedImplementation =
        this.toolPlanner.planGroundedImplementationTransition({
          task: intentTask,
          calls: priorPlannerCalls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
      if (groundedImplementation.length > 0) {
        plans.push(...groundedImplementation);
        this.emit({
          type: 'tool.plan.grounded_implementation_transition',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            plans: groundedImplementation.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    const needsModelPlannedAction = bindings.some(binding =>
      binding.enabled && (
        binding.name === 'shell.exec'
        || binding.name === 'fs.write'
        || binding.name === 'fs.replace'
        || binding.name === 'fs.synthesize'
      )
    ) && (
      workspaceExecutionRequired
      ||
      /\b(?:terminal|shell|command line|cli|container)\b/i.test(intentTask)
      || /\b(?:implement|modify|edit|create|write|patch|repair|fix|refactor)\b[\s\S]*\b(?:file|code|project|repository|workspace|artifact|solution)\b/i.test(intentTask)
    );
    const hasInitialInspection = plans.some(plan =>
      plan.toolName === 'fs.list'
      || plan.toolName === 'fs.read'
      || plan.toolName === 'fs.search'
    );
    const hasCausalWorkspaceMutation = plans.some(plan =>
      isSuccessfulWorkspaceMutationCall({
        toolName: plan.toolName,
        params: plan.params,
        success: true,
      })
    );
    const initialWorkspaceEvidenceSaturated = workspaceExecutionRequired
      && priorPlannerCalls.some(call =>
        call.toolName === 'fs.read' && call.success
      )
      && this.toolPlanner.planWorkspaceEvidenceFollowUps({
        task: intentTask,
        calls: priorPlannerCalls,
        bindings,
        workspaceRoot: this.workspaceRoot,
      }).length === 0;
    if (needsModelPlannedAction
      && loopConfig.enabled
      && loopConfig.llmReplanning
      && !options.skipInitialModelPlanning
      && !hasInitialInspection
      && !hasCausalWorkspaceMutation
      && !unresolvedSynthesisRejection
      && remainingGroundingMs() > 1_000
      && actor instanceof UnifiedAgent) {
      const modelPlans = await actor.planNextToolRound({
        task,
        intentTask,
        executionRequired: workspaceExecutionRequired,
        diagnosticProbeRequired,
        requiredDiagnosticAfterCallIndex: priorPlannerCalls.length - 1,
        workspaceEvidenceSaturated: initialWorkspaceEvidenceSaturated,
        round: 0,
        remainingCalls: loopConfig.maxCallsPerRun,
        requestTimeoutMs: planningRequestTimeoutMs(),
        tools: bindings
          .filter(binding => binding.enabled)
          .map(binding => {
            const metadata = toolRegistry.getMetadata(binding.name);
            return {
              name: binding.name,
              description: metadata?.description,
              parameters: metadata?.parameters as Record<string, unknown> | undefined,
            };
          }),
        calls: priorPlannerCalls,
        requiredMutationAfterCallIndex: options.requireFreshMutation
          ? priorPlannerCalls.length - 1
          : undefined,
      });
      this.emitToolPlanningFailure(actor, agentId, options, 0);
      const plannedFingerprints = new Set(plans.map(plan => this.toolPlanFingerprint(plan)));
      plans.push(...modelPlans.filter(plan => !plannedFingerprints.has(this.toolPlanFingerprint(plan))));
    }
    if (plans.length === 0
      && workspaceExecutionRequired
      && !priorRejectedVerifierCandidate
      && actor instanceof UnifiedAgent) {
      const groundedTransition = actor.planGroundedExecutionTransition({
        task: intentTask,
        round: 0,
        tools: bindings
          .filter(binding => binding.enabled)
          .map(binding => ({ name: binding.name })),
        calls: priorPlannerCalls,
      });
      if (groundedTransition.length > 0) {
        plans.push(...groundedTransition);
        this.emit({
          type: 'tool.plan.grounded_execution_transition',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            plans: groundedTransition.map(plan => ({
              toolName: plan.toolName,
              params: plan.params,
            })),
            priorToolCalls: priorPlannerCalls.length,
          },
        });
      }
    }
    plans = this.deferRepeatedNoGainMutationTargets(
      plans,
      priorPlannerCalls,
      agentId,
      options
    );
    plans = this.keepSingleWorkspaceMutationHypothesis(plans, agentId, options);
    const pendingInitialMirroredVerifier = pendingMirroredVerifierPlan(
      priorPlannerCalls
    );
    if (pendingInitialMirroredVerifier && !unresolvedSynthesisRejection) {
      plans = [pendingInitialMirroredVerifier];
      this.emit({
        type: 'tool.plan.mirrored_verifier_barrier',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          phase: 'initial',
          command: mirroredVerifierCommand,
          reason: 'prior_workspace_mutation_requires_objective_feedback',
        },
      });
    }
    if (plans.length === 0) {
      if (priorRejectedVerifierCandidate) {
        this.emit({
          type: 'agent.tool_loop.rejected_candidate.closed',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            round: 0,
            reason: 'prior_verifier_rejection_requires_fresh_diagnosis',
            path: priorRejectedVerifierCandidate.path,
            candidateReason: priorRejectedVerifierCandidate.reason,
            baselineReward: priorRejectedVerifierCandidate.baselineReward,
            candidateReward: priorRejectedVerifierCandidate.candidateReward,
          },
        });
      }
      const warning = groundingRequired && !priorRejectedVerifierCandidate
        ? 'Grounding was required, but no authorized tool call could be planned for this task.'
        : undefined;
      if (warning) {
        this.emit({
          type: 'agent.grounding.warning',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          data: { warning, reason: 'no_grounding_tool_plan' },
        });
      }
      return {
        toolCalls: [],
        grounded: Boolean(priorRejectedVerifierCandidate) || !groundingRequired,
        warnings: warning ? [warning] : [],
        context: priorRejectedVerifierCandidate
          ? 'The latest verifier-rejected workspace hypothesis was rolled back. A fresh diagnostic phase is required before another mutation.'
          : '',
        evidence: {
          toolGrounded: Boolean(priorRejectedVerifierCandidate),
          outputGrounded: Boolean(priorRejectedVerifierCandidate) || !groundingRequired,
          observedPaths: typeof priorRejectedVerifierCandidate?.path === 'string'
            ? [priorRejectedVerifierCandidate.path]
            : [],
        },
        toolLoop: {
          rounds: [],
          totalCalls: 0,
          successfulCalls: 0,
          failedCalls: 0,
          stopReason: 'completed',
          startedAt: Date.now(),
          completedAt: Date.now(),
        },
      };
    }

    await options.onBeforeExecution?.(plans);
    const loop = new AgentToolExecutionLoop({
      maxRounds: loopConfig.enabled ? loopConfig.maxRounds : 1,
      maxCalls: loopConfig.enabled ? loopConfig.maxCallsPerRun : Math.max(1, plans.length),
      maxConsecutiveFailures: loopConfig.maxConsecutiveFailures,
      maxWallClockMs: Math.max(1, remainingGroundingMs()),
    });
    const liveGroundingCalls: ToolCallRecord[] = [...priorPlannerCalls];
    const toolLoop = await loop.run({
      task,
      initialPlans: plans,
      fingerprint: plan => this.toolPlanFingerprint(plan),
      execute: async (plan, _round) => {
        if (plan.toolName === 'shell.exec') {
          const remainingMs = remainingGroundingMs();
          const executionReserveMs = Math.max(
            1_000,
            Math.min(30_000, Math.floor(remainingMs * 0.2))
          );
          const deadlineTimeoutMs = Math.max(1_000, remainingMs - executionReserveMs);
          const configuredDefaultTimeoutMs = this.workspaceRuntimeConfig?.tools.shell.defaultTimeoutMs
            ?? 10_000;
          const requestedTimeoutMs = plan.params.timeoutMs === undefined
            ? configuredDefaultTimeoutMs
            : Number(plan.params.timeoutMs);
          if (Number.isFinite(requestedTimeoutMs)
            && requestedTimeoutMs > 0
            && deadlineTimeoutMs < requestedTimeoutMs) {
            plan.params = {
              ...plan.params,
              timeoutMs: deadlineTimeoutMs,
            };
            this.emit({
              type: 'tool.deadline.applied',
              agentId,
              sessionId: this.getContext().sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: {
                toolName: plan.toolName,
                requestedTimeoutMs,
                effectiveTimeoutMs: deadlineTimeoutMs,
                remainingGroundingMs: remainingMs,
              },
            });
          }
        }
        const cacheKey = this.sharedReadOnlyToolCacheKey(plan);
        const authorized = bindings.some(binding =>
          binding.enabled && binding.name === plan.toolName
        );
        let result: ToolResult;
        let cacheSource: SharedReadOnlyToolCacheEntry | undefined;
        if (cacheKey && authorized) {
          cacheSource = this.sharedReadOnlyToolResultCache.get(cacheKey);
          if (!cacheSource) {
            const pending = this.sharedReadOnlyToolRequests.get(cacheKey);
            if (pending) {
              const pendingResult = await pending;
              if (pendingResult.success) {
                cacheSource = this.sharedReadOnlyToolResultCache.get(cacheKey);
              }
            }
          }
        }
        if (cacheSource) {
          result = {
            success: true,
            result: cacheSource.call.result,
            metadata: {
              sharedReadOnlyToolCache: true,
              sourceAgentId: cacheSource.sourceAgentId,
              cachedAt: cacheSource.cachedAt,
            },
          };
          this.emit({
            type: 'tool.plan.shared_cache.reused',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              toolName: plan.toolName,
              params: plan.params,
              cacheKey,
              sourceAgentId: cacheSource.sourceAgentId,
              sourceCorrelationId: cacheSource.correlationId,
              ageMs: Math.max(0, Date.now() - cacheSource.cachedAt),
            },
          });
          if (plan.toolName === 'web.search'
            && cacheSource.sourceAgentId !== agentId) {
            sharedWebSearchFingerprints.add(this.toolPlanFingerprint(plan));
          }
        } else {
          const execute = this.executeToolForAgent(agentId, plan.toolName, plan.params, {
            reason: plan.reason,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            synthesisTask: intentTask,
            groundingCalls: liveGroundingCalls,
          });
          if (cacheKey && authorized) {
            this.sharedReadOnlyToolRequests.set(cacheKey, execute);
          }
          try {
            result = await execute;
          } finally {
            if (cacheKey
              && this.sharedReadOnlyToolRequests.get(cacheKey) === execute) {
              this.sharedReadOnlyToolRequests.delete(cacheKey);
            }
          }
          if (cacheKey && result.success) {
            this.storeSharedReadOnlyToolResult(cacheKey, {
              call: {
                toolName: plan.toolName,
                params: { ...plan.params },
                reason: plan.reason,
                result: result.result,
                success: true,
                completedAt: Date.now(),
              },
              sourceAgentId: agentId,
              correlationId: options.correlationId,
              cachedAt: Date.now(),
            });
            this.emit({
              type: 'tool.result.shared_cache.stored',
              agentId,
              sessionId: this.getContext().sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: {
                toolName: plan.toolName,
                params: plan.params,
                cacheKey,
                entries: this.sharedReadOnlyToolResultCache.size,
              },
            });
          }
          if (plan.toolName === 'web.search' && result.success) {
            freshWebSearchFingerprints.add(this.toolPlanFingerprint(plan));
          }
        }
        liveGroundingCalls.push({
          toolName: plan.toolName,
          params: plan.params,
          reason: plan.reason,
          result: result.result,
          success: result.success,
          error: result.error,
        });
        return { result: result.result, success: result.success, error: result.error };
      },
      planNext: async context => {
        const combinedCalls = [...priorPlannerCalls, ...context.calls];
        const pendingMirroredVerifier = pendingMirroredVerifierPlan(
          combinedCalls
        );
        if (pendingMirroredVerifier) {
          this.emit({
            type: 'tool.plan.mirrored_verifier_barrier',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              phase: 'continuation',
              command: mirroredVerifierCommand,
              reason: 'current_workspace_mutation_requires_objective_feedback',
            },
          });
          return [pendingMirroredVerifier];
        }
        const postMutationVerification = this.toolPlanner.planPostMutationVerification({
          task: intentTask,
          calls: combinedCalls,
          bindings,
        });
        if (postMutationVerification.length > 0) {
          this.emit({
            type: 'tool.plan.post_mutation_verification',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              plans: postMutationVerification.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
            },
          });
          return postMutationVerification.slice(0, context.remainingCalls);
        }
        const rejectedCandidate = this.latestRejectedVerifierCandidate(
          context.calls
        ) ?? (rejectedCandidateIsSuperseded(
          this.latestRejectedVerifierCandidate(
            [...priorPlannerCalls, ...context.calls]
          )
        )
          ? undefined
          : this.latestRejectedVerifierCandidate(
            [...priorPlannerCalls, ...context.calls]
          ));
        if (rejectedCandidate) {
          this.emit({
            type: 'agent.tool_loop.rejected_candidate.closed',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              round: context.round,
              reason: 'verifier_rejected_current_hypothesis_require_fresh_diagnosis',
              path: rejectedCandidate.path,
              candidateReason: rejectedCandidate.reason,
              baselineReward: rejectedCandidate.baselineReward,
              candidateReward: rejectedCandidate.candidateReward,
            },
          });
          return [];
        }
        const authoritativeExternalFeedbackRepair = concreteExternalFeedback
          ? this.toolPlanner.planExternalFeedbackRepair({
            task: intentTask,
            calls: [...priorPlannerCalls, ...context.calls],
            currentCalls: context.calls,
            bindings,
            workspaceRoot: this.workspaceRoot,
          })
          : [];
        if (authoritativeExternalFeedbackRepair.length > 0) {
          // A new external verifier result is the authoritative frontier for
          // this continuation. Resolve it before mining older tool failures;
          // those failures may already have been superseded by later passing
          // commands from the prior phase.
          this.emit({
            type: 'tool.plan.external_feedback_repair',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              plans: authoritativeExternalFeedbackRepair.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
            },
          });
          const eligible = this.deferRepeatedNoGainMutationTargets(
            authoritativeExternalFeedbackRepair,
            combinedCalls,
            agentId,
            options
          );
          if (eligible.length > 0) {
            return eligible.slice(0, context.remainingCalls);
          }
        }
        const causalTransitionPlans = this.enrichRepairPlansWithTeamStepEvidence(
          this.toolPlanner.planWorkspaceRepairTransition({
            task: intentTask,
            calls: [...priorPlannerCalls, ...context.calls],
            bindings,
            workspaceRoot: this.workspaceRoot,
          }),
          task
        );
        if (causalTransitionPlans.length > 0) {
          this.emit({
            type: 'tool.plan.causal_transition',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              from: 'aggregate_verifier_evidence',
              to: 'workspace_repair',
              plans: causalTransitionPlans.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
              priorToolCalls: priorPlannerCalls.length + context.calls.length,
            },
          });
          const eligible = this.deferRepeatedNoGainMutationTargets(
            causalTransitionPlans,
            combinedCalls,
            agentId,
            options
          );
          if (eligible.length > 0) {
            return eligible.slice(0, context.remainingCalls);
          }
        }
        const environmentRecovery = this.toolPlanner.planEnvironmentRecovery({
          calls: combinedCalls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
        if (environmentRecovery.length > 0) {
          // Environment recovery already declines to act while a newer,
          // source-localized failure remains. Give it priority over historical
          // failure follow-ups so a cleared traceback cannot hide a missing
          // test runner and trigger unrelated source mutations.
          this.emit({
            type: 'tool.plan.environment_recovery',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              plans: environmentRecovery.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
              priorToolCalls: combinedCalls.length,
            },
          });
          return environmentRecovery.slice(0, context.remainingCalls);
        }
        const failureContextPlans = this.toolPlanner.planWorkspaceFailureFollowUps({
          calls: [...priorPlannerCalls, ...context.calls],
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
        if (failureContextPlans.length > 0) {
          return failureContextPlans.slice(0, context.remainingCalls);
        }
        const externalFeedbackRepair = this.toolPlanner.planExternalFeedbackRepair({
          task: intentTask,
          calls: [...priorPlannerCalls, ...context.calls],
          currentCalls: context.calls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
        if (externalFeedbackRepair.length > 0) {
          this.emit({
            type: 'tool.plan.external_feedback_repair',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              plans: externalFeedbackRepair.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
            },
          });
          const eligible = this.deferRepeatedNoGainMutationTargets(
            externalFeedbackRepair,
            combinedCalls,
            agentId,
            options
          );
          if (eligible.length > 0) {
            return eligible.slice(0, context.remainingCalls);
          }
        }
        if (diagnosticProbeRequired
          && context.calls.some(call => this.isFocusedVerifierDiagnosticCall(call))) {
          this.emit({
            type: 'agent.tool_loop.diagnostic_closed',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              round: context.round,
              reason: 'focused_verifier_probe_completed',
            },
          });
          return [];
        }
        if (options.archetype === 'tester'
          && !diagnosticProbeRequired
          && context.calls.some(call =>
            isSuccessfulWorkspaceVerificationCall(call)
          )) {
          this.emit({
            type: 'agent.tool_loop.verification_closed',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              round: context.round,
              reason: 'fresh_authoritative_verifier_completed',
            },
          });
          return [];
        }
        const workspaceEvidencePlans = this.toolPlanner.planWorkspaceEvidenceFollowUps({
          task: intentTask,
          calls: combinedCalls,
          bindings,
          workspaceRoot: this.workspaceRoot,
        });
        if (workspaceEvidencePlans.length > 0) {
          return workspaceEvidencePlans.slice(0, context.remainingCalls);
        }
        const deterministic = this.toolPlanner.planWebFollowUps({
          task: intentTask,
          calls: context.calls,
          bindings,
          maxFetches: loopConfig.maxFetchesAfterSearch,
        });
        if (deterministic.length > 0) return deterministic.slice(0, context.remainingCalls);
        if (diversityGapEmitted
          && !diversityCompletedEmitted
          && !hasSharedWebEvidenceDiversityGap()) {
          diversityCompletedEmitted = true;
          this.emit({
            type: 'agent.web_evidence.diversification.completed',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              sharedSearches: sharedWebSearchFingerprints.size,
              freshSearches: freshWebSearchFingerprints.size,
              requiredFreshSearches: requiredFreshWebSearches(),
            },
          });
        }
        if (this.toolPlanner.hasSufficientWebEvidence(intentTask, context.calls)
          && !hasSharedWebEvidenceDiversityGap()) {
          return [];
        }
        if (this.hasClosedReadOnlyWorkspaceEvidence(intentTask, combinedCalls)) {
          this.emit({
            type: 'agent.tool_loop.evidence_closed',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              round: context.round,
              successfulReads: combinedCalls.filter(call =>
                call.toolName === 'fs.read' && call.success
              ).length,
              reason: 'explicit_workspace_evidence_covered',
            },
          });
          return [];
        }
        if (this.hasUnresolvedSynthesisRejection(combinedCalls)) {
          this.emit({
            type: 'agent.tool_loop.rejected_candidate.stopped',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              reason: 'A synthesized candidate was rejected and no newer causal evidence supports another generation.',
              priorToolCalls: combinedCalls.length,
            },
          });
          return [];
        }
        if (!loopConfig.enabled || !loopConfig.llmReplanning || !(actor instanceof UnifiedAgent)) return [];
        if (!this.shouldReplanToolLoop(intentTask, context.calls)) return [];
        if (remainingGroundingMs() <= 1_000) {
          this.emit({
            type: 'agent.tool_planning.time_budget.exhausted',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              round: context.round,
              remainingMs: remainingGroundingMs(),
              groundingMaxWallClockMs,
            },
          });
          return [];
        }
        const diversityPlanningTask = hasSharedWebEvidenceDiversityGap()
          ? [
            task,
            '[runtime_shared_web_frontier]',
            `Another actor already executed ${sharedWebSearchFingerprints.size} equivalent web search(es), so their cached results do not constitute an independent check. Request ${requiredFreshWebSearches() - freshWebSearchFingerprints.size} non-equivalent web.search call(s) now. Probe alternate wording, scope, dates, or counterexamples for the most material uncertainties; do not repeat these cached query fingerprints: ${[...sharedWebSearchFingerprints].join(', ')}.`,
          ].join('\n')
          : task;
        if (hasSharedWebEvidenceDiversityGap() && !diversityGapEmitted) {
          diversityGapEmitted = true;
          this.emit({
            type: 'agent.web_evidence.diversification.required',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              sharedSearches: sharedWebSearchFingerprints.size,
              freshSearches: freshWebSearchFingerprints.size,
              requiredFreshSearches: requiredFreshWebSearches(),
            },
          });
        }
        const llmPlans = await actor.planNextToolRound({
          task: diversityPlanningTask,
          intentTask,
          executionRequired: workspaceExecutionRequired,
          diagnosticProbeRequired,
          requiredDiagnosticAfterCallIndex: priorPlannerCalls.length - 1,
          workspaceEvidenceSaturated: workspaceExecutionRequired
            && workspaceEvidencePlans.length === 0
            && combinedCalls.some(call =>
              call.toolName === 'fs.read' && call.success
            ),
          round: context.round,
          remainingCalls: context.remainingCalls,
          requestTimeoutMs: planningRequestTimeoutMs(),
          tools: bindings
            .filter(binding => binding.enabled)
            .map(binding => {
              const metadata = toolRegistry.getMetadata(binding.name);
              return {
                name: binding.name,
                description: metadata?.description,
                parameters: metadata?.parameters as Record<string, unknown> | undefined,
              };
          }),
          calls: [...priorPlannerCalls, ...context.calls],
          requiredMutationAfterCallIndex: options.requireFreshMutation
            ? priorPlannerCalls.length - 1
            : undefined,
        });
        if (hasSharedWebEvidenceDiversityGap()) {
          this.emit({
            type: 'agent.web_evidence.diversification.planned',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              plans: llmPlans.map(plan => ({
                toolName: plan.toolName,
                params: plan.params,
              })),
            },
          });
        }
        this.emitToolPlanningFailure(actor, agentId, options, context.round);
        const alignedPlans = llmPlans.filter(plan => {
          const parallelMutation = findParallelSourceMutation(
            plan,
            [...priorPlannerCalls, ...context.calls]
          );
          if (parallelMutation) {
            this.emit({
              type: 'tool.path.authoritative_rejected',
              agentId,
              sessionId: this.getContext().sessionId,
              correlationId: options.correlationId,
              nodeId: options.nodeId,
              data: {
                toolName: plan.toolName,
                ...parallelMutation,
              },
            });
            return false;
          }
          if (plan.toolName === 'web.fetch') {
            return this.toolPlanner.isWebCandidateAligned(task, String(plan.params.url ?? ''));
          }
          if (plan.toolName === 'web.search') {
            return this.toolPlanner.isWebCandidateAligned(task, String(plan.params.query ?? ''));
          }
          return true;
        });
        return this.deferRepeatedNoGainMutationTargets(
          alignedPlans,
          combinedCalls,
          agentId,
          options
        );
      },
      onRoundStarted: (round, roundPlans) => {
        this.emit({
          type: 'agent.tool_loop.round.started',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            round,
            plans: roundPlans.map(plan => ({ toolName: plan.toolName, params: plan.params, reason: plan.reason })),
          },
        });
      },
      onRoundCompleted: round => {
        this.emit({
          type: 'agent.tool_loop.round.completed',
          agentId,
          sessionId: this.getContext().sessionId,
          correlationId: options.correlationId,
          nodeId: options.nodeId,
          data: {
            round: round.round,
            calls: round.calls.length,
            successful: round.calls.filter(call => call.success).length,
            failed: round.calls.filter(call => !call.success).length,
          },
        });
      },
    });
    if (diversityGapEmitted
      && !diversityCompletedEmitted
      && !hasSharedWebEvidenceDiversityGap()) {
      diversityCompletedEmitted = true;
      this.emit({
        type: 'agent.web_evidence.diversification.completed',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options.correlationId,
        nodeId: options.nodeId,
        data: {
          sharedSearches: sharedWebSearchFingerprints.size,
          freshSearches: freshWebSearchFingerprints.size,
          requiredFreshSearches: requiredFreshWebSearches(),
        },
      });
    }
    this.emit({
      type: 'agent.tool_loop.completed',
      agentId,
      sessionId: this.getContext().sessionId,
      correlationId: options.correlationId,
      nodeId: options.nodeId,
      data: {
        rounds: toolLoop.rounds.length,
        totalCalls: toolLoop.totalCalls,
        successfulCalls: toolLoop.successfulCalls,
        failedCalls: toolLoop.failedCalls,
        stopReason: toolLoop.stopReason,
        durationMs: toolLoop.completedAt - toolLoop.startedAt,
      },
    });

    const toolCalls: ToolCallRecord[] = toolLoop.rounds.flatMap(round => round.calls.map(call => ({
      toolName: call.toolName,
      params: call.params,
      result: call.result,
      success: call.success,
      error: call.error,
      reason: call.reason,
      round: round.round,
      startedAt: call.startedAt,
      completedAt: call.completedAt,
    })));
    const warnings: string[] = [];
    const observedPaths: string[] = [];
    const observedUrls: string[] = [];
    const discoveredUrls: string[] = [];
    const summaries: string[] = [];
    const contexts: string[] = [];

    for (const call of toolCalls) {
      if (!call.success) {
        const warning = `${call.toolName === 'fs.list' ? 'Project inspection tool ' : 'Tool '}${call.toolName} failed: ${call.error ?? 'unknown error'}`;
        warnings.push(warning);
        this.emit({ type: 'agent.grounding.warning', agentId, data: { warning, correlationId: options.correlationId } });
        continue;
      }

      if (call.toolName === 'fs.list') {
        const entries = Array.isArray((call.result as { entries?: unknown } | undefined)?.entries)
          ? (call.result as { entries: unknown[] }).entries.filter((item): item is string => typeof item === 'string')
          : [];
        observedPaths.push(...entries.slice(0, 80));
        summaries.push(entries.slice(0, 80).join('\n'));
        contexts.push(`Filesystem listing:\n${entries.join('\n')}`);
      } else if (call.toolName === 'fs.read') {
        const read = call.result as { path?: unknown; content?: unknown } | undefined;
        if (typeof read?.path === 'string') observedPaths.push(read.path);
        const readPath = this.normalizeToolWorkspacePath(String(
          read?.path ?? call.params.path ?? ''
        ));
        const diagnosticSourcePath = diagnosticProbeRequired
          ? this.extractTaskDiagnosticSourcePaths(intentTask)[0]
          : undefined;
        const contentLimit = diagnosticSourcePath
          && readPath === diagnosticSourcePath
          ? 24_000
          : 8_000;
        const content = typeof read?.content === 'string'
          ? read.content.slice(0, contentLimit)
          : '';
        if (diagnosticSourcePath && readPath === diagnosticSourcePath) {
          this.emit({
            type: 'agent.verifier_diagnostic.source_context.loaded',
            agentId,
            sessionId: this.getContext().sessionId,
            correlationId: options.correlationId,
            nodeId: options.nodeId,
            data: {
              path: readPath,
              contentChars: content.length,
              availableChars: typeof read?.content === 'string'
                ? read.content.length
                : 0,
            },
          });
        }
        summaries.push(`${String(read?.path ?? 'file')}: ${content.slice(0, 1000)}`);
        contexts.push(`File read result for ${String(read?.path ?? 'file')}:\n${content}`);
      } else if (call.toolName === 'fs.write') {
        const written = call.result as { path?: unknown; bytes?: unknown; mode?: unknown } | undefined;
        if (typeof written?.path === 'string') observedPaths.push(written.path);
        const summary = `Wrote ${String(written?.bytes ?? 'unknown')} bytes to ${String(written?.path ?? 'file')} (${String(written?.mode ?? 'unknown')}).`;
        summaries.push(summary);
        contexts.push(summary);
      } else if (call.toolName === 'fs.synthesize') {
        const synthesized = call.result as {
          path?: unknown;
          bytes?: unknown;
          operation?: unknown;
        } | undefined;
        if (typeof synthesized?.path === 'string') observedPaths.push(synthesized.path);
        const summary = `Synthesized ${String(synthesized?.bytes ?? 'unknown')} bytes to ${String(synthesized?.path ?? 'file')} (${String(synthesized?.operation ?? 'unknown')}).`;
        summaries.push(summary);
        contexts.push(summary);
      } else if (call.toolName === 'fs.replace') {
        const replaced = call.result as { path?: unknown; replacements?: unknown; bytes?: unknown } | undefined;
        if (typeof replaced?.path === 'string') observedPaths.push(replaced.path);
        const summary = `Replaced ${String(replaced?.replacements ?? 'unknown')} occurrence(s) in ${String(replaced?.path ?? 'file')} (${String(replaced?.bytes ?? 'unknown')} bytes after edit).`;
        summaries.push(summary);
        contexts.push(summary);
      } else if (call.toolName === 'fs.search') {
        const searched = call.result as {
          query?: unknown;
          filesSearched?: unknown;
          matches?: Array<{ path?: unknown; line?: unknown; preview?: unknown }>;
          truncated?: unknown;
        } | undefined;
        const matches = Array.isArray(searched?.matches) ? searched.matches : [];
        const lines = matches.slice(0, 100).map(match => {
          if (typeof match.path === 'string') observedPaths.push(match.path);
          return `${String(match.path ?? 'file')}:${String(match.line ?? '?')}: ${String(match.preview ?? '').slice(0, 500)}`;
        });
        const summary = `Workspace search for ${JSON.stringify(String(searched?.query ?? call.params.query ?? ''))} across ${String(searched?.filesSearched ?? 'unknown')} file(s):\n${lines.join('\n') || 'No matches.'}${searched?.truncated ? '\n[truncated]' : ''}`;
        summaries.push(summary);
        contexts.push(summary);
      } else if (call.toolName === 'shell.exec') {
        const shell = call.result as { command?: unknown; stdout?: unknown; stderr?: unknown } | undefined;
        const rawCommand = String(shell?.command ?? call.params.command ?? 'command');
        const output = [shell?.stdout, shell?.stderr].filter(value => typeof value === 'string' && value).join('\n');
        const command = this.compactShellCommandForEvidence(
          rawCommand
        );
        const evidenceOutput = rawCommand.includes('ROY_VERIFIER_PROBE=1')
          || output.includes('VERIFIER_PROBE_EVIDENCE_VERSION')
          ? this.compactVerifierProbeEvidenceText(output, 12_000)
          : output;
        summaries.push(`${command}: ${evidenceOutput.slice(0, 2400)}`);
        contexts.push(`Command result for ${command}:\n${evidenceOutput.slice(0, 12_000)}`);
      } else if (call.toolName === 'web.search') {
        const search = call.result as {
          query?: unknown;
          provider?: unknown;
          results?: Array<{ title?: unknown; url?: unknown; snippet?: unknown; source?: unknown }>;
        } | undefined;
        const results = Array.isArray(search?.results) ? search.results : [];
        const lines = results.slice(0, 10).map(item => {
          const url = typeof item.url === 'string' ? item.url : '';
          if (url) discoveredUrls.push(url);
          return `- ${String(item.title ?? 'Untitled')} (${url})\n  ${String(item.snippet ?? '').slice(0, 700)}`;
        });
        summaries.push(`Web search (${String(search?.provider ?? 'unknown')}): ${String(search?.query ?? task)}\n${lines.join('\n')}`);
        contexts.push(`Web search results:\n${lines.join('\n')}`);
      } else if (call.toolName === 'web.fetch') {
        const page = call.result as { finalUrl?: unknown; title?: unknown; text?: unknown; contentType?: unknown } | undefined;
        const url = typeof page?.finalUrl === 'string' ? page.finalUrl : String(call.params.url ?? '');
        if (url) observedUrls.push(url);
        const text = typeof page?.text === 'string' ? page.text.slice(0, 8000) : '';
        summaries.push(`Web page: ${String(page?.title ?? url)} (${url})\n${text.slice(0, 1800)}`);
        contexts.push(`Web page evidence from ${url}:\nTitle: ${String(page?.title ?? 'unknown')}\n${text}`);
      }
    }

    const successful = toolCalls.filter(call => call.success);
    const successfulWebFetchCalls = toolCalls.filter(call => call.toolName === 'web.fetch' && call.success);
    const successfulWebFetches = successfulWebFetchCalls.length;
    const requiredWebFetches = this.requiredWebFetchCount(intentTask);
    const relevantObservedUrls = successfulWebFetchCalls
      .filter(call => this.toolPlanner.webEvidenceScore(task, call) >= 6)
      .map(call => String(
        (call.result as { finalUrl?: unknown } | undefined)?.finalUrl ?? call.params.url ?? ''
      ))
      .filter(Boolean);
    const relevantWebDocuments = new Set(relevantObservedUrls.map(url => this.canonicalWebDocumentUrl(url)));
    const groundedWebSourceCount = relevantWebDocuments.size;
    if (groundedWebSourceCount < requiredWebFetches) {
      const warning = `The task required ${requiredWebFetches} task-relevant opened web source(s), but only ${groundedWebSourceCount} distinct relevant document(s) were fetched successfully (${successfulWebFetches} total fetches).`;
      warnings.push(warning);
      this.emit({
        type: 'agent.grounding.warning',
        agentId,
        sessionId: this.getContext().sessionId,
        correlationId: options.correlationId,
        data: {
          warning,
          reason: 'web_source_not_fetched',
          requiredWebFetches,
          successfulWebFetches,
          groundedWebSourceCount,
        },
      });
    }
    return {
      toolCalls,
      grounded: plans.every(plan => !plan.groundingRequired || toolCalls.some(call => call.toolName === plan.toolName && call.success))
        && groundedWebSourceCount >= requiredWebFetches,
      warnings,
      evidence: {
        toolGrounded: successful.length > 0,
        outputGrounded: false,
        observedPaths: Array.from(new Set(observedPaths)),
        observedUrls: Array.from(new Set(observedUrls)),
        relevantObservedUrls: Array.from(new Set(relevantObservedUrls)),
        discoveredUrls: Array.from(new Set(discoveredUrls)),
        toolResultSummary: summaries.filter(Boolean).join('\n\n'),
      },
      context: contexts.join('\n\n'),
      toolLoop,
    };
  }

  private shouldReplanToolLoop(
    task: string,
    calls: Array<{ toolName: string }>
  ): boolean {
    const requestsFollowUpWorkspaceEvidence =
      /\b(?:read|inspect|review|audit|verify|compare|cross-reference|open|search)\b/i.test(task)
      && (
        /\b(?:files?|csvs?|datasets?|inputs?|rules?|tests?|source|workspace|repository|codebase)\b/i.test(task)
        || /(?:^|[\s`'"])[./]?[a-z0-9_-]+(?:\/[a-z0-9_.-]+)*\.(?:csv|json|ya?ml|toml|md|py|ts|tsx|js|jsx|mjs|cjs)(?=$|[\s,.;:`'"])/i.test(task)
      )
      && calls.some(call =>
        call.toolName === 'fs.list'
        || call.toolName === 'fs.read'
        || call.toolName === 'fs.search'
      );
    return this.taskRequiresWorkspaceMutation(task)
      || requestsFollowUpWorkspaceEvidence
      || calls.some(call =>
        call.toolName.startsWith('web.')
        || call.toolName === 'shell.exec'
        || call.toolName === 'fs.write'
        || call.toolName === 'fs.replace'
        || call.toolName === 'fs.synthesize'
      )
      || /\b(?:multi-step|continue|iterate|until|cross-check|multiple sources|independent sources)\b/i.test(task);
  }

  private hasClosedReadOnlyWorkspaceEvidence(
    task: string,
    calls: Array<Pick<ToolCallRecord, 'toolName' | 'params' | 'result' | 'success'>>
  ): boolean {
    if (this.taskRequiresWorkspaceMutation(task)
      || !/\b(?:inspect|read|review|audit|summari[sz]e|report|identify|analy[sz]e)\b/i.test(
        task
      )) {
      return false;
    }
    const successfulReads = calls.filter(call =>
      call.toolName === 'fs.read' && call.success
    );
    if (successfulReads.length === 0
      || !calls.some(call => call.toolName === 'fs.list' && call.success)) {
      return false;
    }
    const readPaths = successfulReads.map(call =>
      this.normalizeToolWorkspacePath(String(
        (call.result as { path?: unknown } | undefined)?.path
          ?? call.params.path
          ?? ''
      )).toLowerCase()
    );
    const namedFiles = [...task.matchAll(
      /(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh|json|jsonl|csv|ya?ml|toml|ini|cfg|md|txt|xml))(?=$|[.\s`'"),:;])/gi
    )].map(match => this.normalizeToolWorkspacePath(String(match[1])).toLowerCase());
    if (namedFiles.some(named =>
      !readPaths.some(observed =>
        observed === named || observed.endsWith(`/${named}`)
      )
    )) {
      return false;
    }
    if (/\b(?:source|code|implementation|codebase)\b/i.test(task)
      && !readPaths.some(path =>
        /\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh)$/i.test(path)
      )) {
      return false;
    }
    if (/\b(?:manifest|dataset|inputs?|data)\b/i.test(task)
      && !readPaths.some(path =>
        /(?:^|\/)data\/|manifest|dataset|input/i.test(path)
      )) {
      return false;
    }
    if (/\b(?:ocr|tokens?)\b/i.test(task)
      && !readPaths.some(path => /ocr|tokens?/.test(path))) {
      return false;
    }
    return true;
  }

  private isFocusedVerifierDiagnosticCall(
    call: Pick<ToolCallRecord, 'toolName' | 'params' | 'success'>
  ): boolean {
    if (call.toolName !== 'shell.exec' || !call.success) return false;
    const command = String(call.params.command ?? '').trim();
    if (!command) return false;
    if (/\bROY_VERIFIER_PROBE=1\b/.test(command)) return true;
    if (/\bpython(?:3)?\s+\.roy\/official-verifier\/grade\.py(?:\s|$)/i.test(
      command
    )) {
      return false;
    }
    if (/\bpython(?:3)?\s+-m\s+table_recon\.cli\s+run\b[\s\S]*\bdata\/public\/manifest\.json\b/i.test(
      command
    )) {
      return false;
    }
    return /\b(?:python3?|pytest|unittest|node|npm|npx)\b/i.test(command)
      && /\b(?:actual|expected|fixture|hidden|manifest|repro|test|audit|table_recon|verifier)\b/i.test(
        command
      );
  }

  private selectAuthoritativePriorVerification(
    calls: ToolCallRecord[],
    task = ''
  ): ToolCallRecord | undefined {
    const candidates = calls
      .map((call, index) => ({ call, index }))
      // Command authority is independent of whether the prior candidate
      // passed. A failed official verifier remains the command that an
      // independent tester must rerun after the next mutation.
      .filter(item => item.call.success && isWorkspaceVerificationCall(item.call))
      .map(item => {
        const command = String(item.call.params.command ?? '');
        return {
          ...item,
          authority: this.verificationCommandAuthority(command),
        };
      })
      .sort((left, right) =>
        right.authority - left.authority || right.index - left.index
      );
    const cached = candidates[0];
    const taskCommand = this.extractAuthoritativeVerificationCommand(task);
    if (taskCommand
      && this.verificationCommandAuthority(taskCommand) >= (cached?.authority ?? 0)) {
      return {
        toolName: 'shell.exec',
        params: { command: taskCommand },
        success: true,
        reason: 'Authoritative verification command extracted from the immutable task.',
      };
    }
    return cached?.call;
  }

  private verificationCommandAuthority(command: string): number {
    const normalized = command.toLowerCase();
    if (normalized.includes('.roy/official-verifier/')
      && !normalized.includes('roy_verifier_probe=1')) {
      return 100;
    }
    if (/\b(?:pytest|unittest|npm test|pnpm test|yarn test|cargo test|go test)\b/.test(normalized)) {
      return 80;
    }
    if (/\b(?:grade|verifier|verify|test|check)\b/.test(normalized)) {
      return normalized.includes('roy_verifier_probe=1') ? 30 : 60;
    }
    return 10;
  }

  private extractAuthoritativeVerificationCommand(task: string): string | undefined {
    const commands = [...task.matchAll(
      /```(?:bash|sh|shell)?\s*\n([\s\S]*?)```/gi
    )]
      .flatMap(match => String(match[1] ?? '')
        .split('\n')
        .map(line => line.trim())
        .filter(line => line
          && !line.startsWith('#')
          && !line.includes('ROY_VERIFIER_PROBE=1')))
      .filter(command =>
        /\b(?:grade|verifier|verify|pytest|unittest|npm test|pnpm test|yarn test|cargo test|go test|check)\b/i.test(
          command
        )
      )
      .map((command, index) => ({
        command,
        index,
        authority: this.verificationCommandAuthority(command),
      }))
      .sort((left, right) =>
        right.authority - left.authority || right.index - left.index
      );
    return commands[0]?.command;
  }

  private latestUnresolvedSynthesisRejection(
    calls: Array<Pick<ToolCallRecord, 'toolName' | 'params' | 'result' | 'success'>>
  ): { path?: string; reason?: string } | undefined {
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      const result = call.result as {
        synthesisRejected?: unknown;
        path?: unknown;
        reason?: unknown;
      } | undefined;
      const rejected = result?.synthesisRejected === true;
      if (!rejected) continue;
      if (calls.slice(index + 1).some(later =>
        later.success && (
          this.isFocusedVerifierDiagnosticCall(later)
          || later.toolName === 'fs.read'
          || later.toolName === 'fs.search'
          || isSuccessfulWorkspaceMutationCall(later)
        )
      )) {
        return undefined;
      }
      return {
        path: typeof result.path === 'string'
          ? result.path
          : typeof call.params.path === 'string'
            ? call.params.path
            : undefined,
        reason: typeof result.reason === 'string' ? result.reason : undefined,
      };
    }
    return undefined;
  }

  private hasUnresolvedSynthesisRejection(
    calls: Array<Pick<ToolCallRecord, 'toolName' | 'params' | 'result' | 'success'>>
  ): boolean {
    return Boolean(this.latestUnresolvedSynthesisRejection(calls));
  }

  private latestRejectedVerifierCandidate(
    calls: Array<Pick<ToolCallRecord, 'toolName' | 'params' | 'result' | 'success'>>
  ): Record<string, unknown> | undefined {
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]!;
      if (this.isFocusedVerifierDiagnosticCall(call)
        || isSuccessfulWorkspaceMutationCall(call)) {
        return undefined;
      }
      const candidateRollback = (
        call.result as { candidateRollback?: unknown } | undefined
      )?.candidateRollback;
      if (candidateRollback && typeof candidateRollback === 'object') {
        return candidateRollback as Record<string, unknown>;
      }
    }
    return undefined;
  }

  private pythonExecutable(): string {
    return process.env.ROY_PYTHON_EXECUTABLE?.trim() || 'python3';
  }

  private shellExecutable(executable: string): string {
    if (/^[A-Za-z0-9_./-]+$/.test(executable)) return executable;
    return `'${executable.replace(/'/g, `'"'"'`)}'`;
  }

  private buildPythonVerifierDiagnosticCommand(task = ''): string {
    const taskInputPaths = this.extractTaskDiagnosticInputPaths(task);
    const diagnosticScopes = this.extractVerifierDiagnosticScopes(task);
    const acceptedReward = this.extractAcceptedVerifierReward(task);
    const script = [
      'import functools',
      'import json',
      'import os',
      'import runpy',
      'import shutil',
      'import tempfile',
      '',
      'retained = []',
      '',
      'class RetainedTemporaryDirectory:',
      '    def __init__(self, suffix=None, prefix=None, dir=None, **kwargs):',
      '        self.name = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)',
      '        retained.append(self.name)',
      '    def __enter__(self):',
      '        return self.name',
      '    def __exit__(self, exc_type, exc_value, traceback):',
      '        return False',
      '    def cleanup(self):',
      '        return None',
      '',
      'tempfile.TemporaryDirectory = RetainedTemporaryDirectory',
      'namespace = runpy.run_path(".roy/official-verifier/grade.py", run_name="roy_verifier_probe")',
      '',
      'def compact(value):',
      '    try:',
      '        rendered = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)',
      '    except Exception:',
      '        rendered = repr(value)',
      '    if len(rendered) <= 6000:',
      '        return rendered',
      '    return rendered[:3000] + "\\n[verifier_probe_compacted]\\n" + rendered[-3000:]',
      '',
      `task_input_candidates = ${JSON.stringify(taskInputPaths)}`,
      'task_input_seen = set()',
      'task_input_records = []',
      'task_input_queue = [(candidate, 0) for candidate in task_input_candidates]',
      'task_input_files_remaining = 10',
      'task_input_bytes_remaining = 16000',
      'workspace_root = os.path.realpath(os.getcwd())',
      'text_suffixes = {".json", ".jsonl", ".csv", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml"}',
      'while task_input_queue and task_input_files_remaining > 0 and task_input_bytes_remaining > 0:',
      '    raw_path, depth = task_input_queue.pop(0)',
      '    candidate = os.path.normpath(str(raw_path))',
      '    absolute = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(workspace_root, candidate))',
      '    if absolute != workspace_root and not absolute.startswith(workspace_root + os.sep):',
      '        continue',
      '    if absolute in task_input_seen or not os.path.isfile(absolute):',
      '        continue',
      '    suffix = os.path.splitext(absolute)[1].lower()',
      '    if suffix not in text_suffixes:',
      '        continue',
      '    task_input_seen.add(absolute)',
      '    try:',
      '        size = os.path.getsize(absolute)',
      '        with open(absolute, "r", encoding="utf-8", errors="replace") as handle:',
      '            content = handle.read(min(4000, task_input_bytes_remaining))',
      '    except OSError:',
      '        continue',
      '    relative = os.path.relpath(absolute, workspace_root)',
      '    task_input_records.append({"path": relative, "size": size, "content": content})',
      '    task_input_files_remaining -= 1',
      '    task_input_bytes_remaining -= len(content)',
      '    if depth >= 2 or suffix != ".json" or size > 262144:',
      '        continue',
      '    try:',
      '        with open(absolute, "r", encoding="utf-8", errors="replace") as handle:',
      '            payload = json.load(handle)',
      '    except Exception:',
      '        continue',
      '    discovered = []',
      '    def collect_paths(value):',
      '        if isinstance(value, str) and os.path.splitext(value)[1].lower() in text_suffixes:',
      '            discovered.append(value)',
      '        elif isinstance(value, list):',
      '            for item in value[:200]:',
      '                collect_paths(item)',
      '        elif isinstance(value, dict):',
      '            for item in list(value.values())[:200]:',
      '                collect_paths(item)',
      '    collect_paths(payload)',
      '    dependency_candidates = []',
      '    for discovered_path in discovered[:20]:',
      '        if os.path.isabs(discovered_path):',
      '            dependency_candidates.append((discovered_path, depth + 1))',
      '            continue',
      '        # Manifests commonly mix paths relative to the manifest with',
      '        # paths relative to the workspace root. Queue both; canonical',
      '        # path deduplication keeps the traversal bounded.',
      '        dependency_candidates.append((os.path.join(os.path.dirname(absolute), discovered_path), depth + 1))',
      '        dependency_candidates.append((os.path.join(workspace_root, discovered_path), depth + 1))',
      '    # Follow dependencies before unrelated paths mentioned later in the',
      '    # task (which are often expected output artifacts). This preserves',
      '    # the causal input chain without increasing the evidence budget.',
      '    task_input_queue = dependency_candidates + task_input_queue',
      '',
      'def mismatch_summary(args):',
      '    if len(args) < 2 or not isinstance(args[0], dict) or not isinstance(args[1], list):',
      '        return None',
      '    actual, expected = args[0], args[1]',
      '    mismatches = []',
      '    key_fields = ("document_id", "table_id", "row_index", "col_index")',
      '    for item in expected:',
      '        if not isinstance(item, dict) or not all(field in item for field in key_fields):',
      '            continue',
      '        key = tuple(item[field] for field in key_fields)',
      '        expected_value = item.get("text", item.get("value"))',
      '        actual_value = actual.get(key)',
      '        if str(actual_value).strip().lower() != str(expected_value).strip().lower():',
      '            mismatches.append({"key": key, "expected": expected_value, "actual": actual_value})',
      '    return {',
      '        "actual_items": len(actual),',
      '        "expected_items": len(expected),',
      '        "mismatch_count": len(mismatches),',
      '        "mismatches": mismatches[:40],',
      '    }',
      '',
      'probe_observations = []',
      `diagnostic_scopes = ${JSON.stringify(diagnosticScopes)}`,
      `accepted_reward = ${acceptedReward === undefined ? 'None' : String(acceptedReward)}`,
      '',
      'def instrument(label, function):',
      '    @functools.wraps(function)',
      '    def wrapped(*args, **kwargs):',
      '        result = function(*args, **kwargs)',
      '        summary = mismatch_summary(args)',
      '        probe_observations.append({',
      '            "label": label,',
      '            "summary": summary,',
      '            "args": None if summary is not None else args,',
      '            "kwargs": kwargs or None,',
      '            "result": result,',
      '        })',
      '        return result',
      '    return wrapped',
      '',
      'entrypoint = namespace.get("grade") or namespace.get("main")',
      'if not callable(entrypoint):',
      '    raise RuntimeError("No callable grade or main entrypoint found in the Python verifier")',
      'function_globals = getattr(entrypoint, "__globals__", namespace)',
      'markers = ("compare", "score", "metric", "fraction", "evaluate", "check")',
      'for name, value in list(function_globals.items()):',
      '    if callable(value) and name not in ("grade", "main") and any(marker in name.lower() for marker in markers):',
      '        function_globals[name] = instrument(name, value)',
      '',
      'reward = entrypoint()',
      '# Emit the causal core before optional retained artifacts. Shell',
      '# transports are permitted to retain only the output head, so the',
      '# scorer mismatch, relevant task inputs, and reward come first.',
      'scoped_observations = []',
      'for observation in probe_observations:',
      '    summary = observation.get("summary")',
      '    mismatches = summary.get("mismatches", []) if isinstance(summary, dict) else []',
      '    keys = [item.get("key", []) for item in mismatches if isinstance(item, dict)]',
      '    if any(',
      '        scope in str(key[0]).lower()',
      '        for scope in diagnostic_scopes',
      '        for key in keys',
      '        if isinstance(key, (list, tuple)) and key',
      '    ):',
      '        scoped_observations.append(observation)',
      'current_reward = float(reward) if isinstance(reward, (int, float)) else None',
      'workspace_regressed = accepted_reward is not None and current_reward is not None and current_reward + 1e-12 < accepted_reward',
      'selected_observations = probe_observations if workspace_regressed else (scoped_observations if scoped_observations else probe_observations)',
      'print("VERIFIER_PROBE_SCOPE", compact({"requested": diagnostic_scopes, "matched_observations": len(scoped_observations), "total_observations": len(probe_observations), "accepted_reward": accepted_reward, "current_reward": current_reward, "expanded_for_workspace_regression": workspace_regressed}))',
      'for observation in selected_observations[-20:]:',
      '    print("VERIFIER_PROBE_CALL", observation["label"])',
      '    if observation["summary"] is not None:',
      '        print("VERIFIER_PROBE_MISMATCHES", compact(observation["summary"]))',
      '    elif observation["args"] is not None:',
      '        print("VERIFIER_PROBE_ARGS", compact(observation["args"]))',
      '    if observation["kwargs"] is not None:',
      '        print("VERIFIER_PROBE_KWARGS", compact(observation["kwargs"]))',
      '    print("VERIFIER_PROBE_RESULT", compact(observation["result"]))',
      'for task_input_record in task_input_records:',
      '    print("VERIFIER_PROBE_TASK_INPUT", compact(task_input_record))',
      'print("VERIFIER_PROBE_REWARD", compact(reward))',
      'print("VERIFIER_PROBE_EVIDENCE_VERSION", 3)',
      'mirror_root = os.path.join(".roy", "diagnostics", f"verifier-probe-{os.getpid()}")',
      'os.makedirs(mirror_root, exist_ok=True)',
      'mirror_suffixes = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".gz"}',
      'mirror_paths = []',
      'mirror_files_remaining = 64',
      'mirror_bytes_remaining = 8 * 1024 * 1024',
      'for retained_index, directory in enumerate(retained, start=1):',
      '    for root, directories, files in os.walk(directory):',
      '        directories.sort()',
      '        files.sort()',
      '        for filename in files:',
      '            if mirror_files_remaining <= 0 or mirror_bytes_remaining <= 0:',
      '                break',
      '            source = os.path.join(root, filename)',
      '            suffix = os.path.splitext(filename)[1].lower()',
      '            try:',
      '                size = os.path.getsize(source)',
      '            except OSError:',
      '                continue',
      '            if suffix not in mirror_suffixes or size > 2 * 1024 * 1024 or size > mirror_bytes_remaining:',
      '                continue',
      '            relative = os.path.relpath(source, directory)',
      '            destination_relative = os.path.join(f"retained-{retained_index}", relative)',
      '            destination = os.path.join(mirror_root, destination_relative)',
      '            try:',
      '                os.makedirs(os.path.dirname(destination), exist_ok=True)',
      '                shutil.copyfile(source, destination)',
      '                os.chmod(destination, 0o444)',
      '            except OSError:',
      '                continue',
      '            mirror_paths.append(destination_relative)',
      '            mirror_files_remaining -= 1',
      '            mirror_bytes_remaining -= size',
      '        if mirror_files_remaining <= 0 or mirror_bytes_remaining <= 0:',
      '            break',
      'print("VERIFIER_PROBE_MIRROR", compact({"root": mirror_root, "paths": mirror_paths}))',
      'artifact_budget = 12000',
      'artifact_suffixes = {".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"}',
      'artifact_paths = []',
      'for directory in retained:',
      '    for root, directories, files in os.walk(directory):',
      '        directories.sort()',
      '        files.sort()',
      '        for filename in files:',
      '            if artifact_budget <= 0:',
      '                break',
      '            path = os.path.join(root, filename)',
      '            if os.path.splitext(filename)[1].lower() not in artifact_suffixes:',
      '                continue',
      '            try:',
      '                if os.path.getsize(path) > 65536:',
      '                    continue',
      '                with open(path, "r", encoding="utf-8", errors="replace") as handle:',
      '                    content = handle.read(min(4000, artifact_budget))',
      '            except OSError:',
      '                continue',
      '            relative = os.path.relpath(path, directory)',
      '            print("VERIFIER_PROBE_ARTIFACT", compact({"directory": directory, "path": relative, "content": content}))',
      '            artifact_paths.append(relative)',
      '            artifact_budget -= len(content)',
      '        if artifact_budget <= 0:',
      '            break',
      'try:',
      '    with open(".roy/official-verifier/grade.py", "r", encoding="utf-8") as handle:',
      '        verifier_lines = handle.read().splitlines()',
      '    spec_budget = 6000',
      '    artifact_terms = sorted({os.path.basename(path) for path in artifact_paths if path.startswith("outputs/")})',
      '    emitted_spec_lines = set()',
      '    for term in artifact_terms:',
      '        for index, line in enumerate(verifier_lines):',
      '            if spec_budget <= 0:',
      '                break',
      '            if term not in line or index in emitted_spec_lines:',
      '                continue',
      '            start = max(0, index - 2)',
      '            end = min(len(verifier_lines), index + 7)',
      '            emitted_spec_lines.update(range(start, end))',
      '            context = "\\n".join(f"{line_number + 1}: {verifier_lines[line_number]}" for line_number in range(start, end))',
      '            print("VERIFIER_PROBE_SPEC", compact({"artifact": term, "content": context}))',
      '            spec_budget -= len(context)',
      'except OSError:',
      '    pass',
      'print("VERIFIER_PROBE_RETAINED_DIRS", compact(retained))',
    ].join('\n');
    const encoded = Buffer.from(script, 'utf8').toString('base64');
    const python = this.shellExecutable(this.pythonExecutable());
    return `ROY_VERIFIER_PROBE=1 ${python} -c "import base64;exec(base64.b64decode('${encoded}'))"`;
  }

  private extractVerifierDiagnosticScopes(task: string): string[] {
    const capsuleMatch = /<recovery_capsule>\s*([\s\S]*?)\s*<\/recovery_capsule>/i.exec(
      task
    );
    if (!capsuleMatch) return [];
    try {
      const capsule = JSON.parse(capsuleMatch[1]!) as {
        unresolvedGroups?: unknown;
      };
      const groups = Array.isArray(capsule.unresolvedGroups)
        ? capsule.unresolvedGroups.map(item => String(item).toLowerCase())
        : [];
      const scopes: string[] = [];
      if (groups.some(group => /\bg_public/.test(group))) scopes.push('public');
      if (groups.some(group => /\bg_hidden/.test(group))) scopes.push('hidden');
      return scopes;
    } catch {
      return [];
    }
  }

  private extractAcceptedVerifierReward(task: string): number | undefined {
    const capsuleMatch = /<recovery_capsule>\s*([\s\S]*?)\s*<\/recovery_capsule>/i.exec(
      task
    );
    if (!capsuleMatch) return undefined;
    try {
      const capsule = JSON.parse(capsuleMatch[1]!) as {
        latestAcceptedScorecard?: { reward?: unknown };
      };
      const reward = capsule.latestAcceptedScorecard?.reward;
      return typeof reward === 'number' && Number.isFinite(reward)
        ? reward
        : undefined;
    } catch {
      return undefined;
    }
  }

  private extractTaskDiagnosticInputPaths(task: string): string[] {
    const paths = [...task.matchAll(
      /(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+\.(?:json|jsonl|csv|md|txt|ya?ml|toml|ini|cfg|xml))(?=$|[\s`'"),:;])/gi
    )]
      .map(match => this.normalizeToolWorkspacePath(match[1]!))
      .filter(path => path !== '.'
        && !path.startsWith('.roy/official-verifier/')
        && !path.startsWith('outputs/'));
    const rank = (value: string): number => {
      let score = 0;
      if (/(?:^|\/)(?:data|input|fixture|sample)s?(?:\/|$)/i.test(value)) score += 30;
      if (/(?:manifest|config|metadata|meta|schema|expected)/i.test(value)) score += 20;
      if (/\.json$/i.test(value)) score += 10;
      return score;
    };
    return Array.from(new Set(paths))
      .sort((left, right) => rank(right) - rank(left) || left.localeCompare(right))
      .slice(0, 8);
  }

  private extractTaskDiagnosticSourcePaths(task: string): string[] {
    const paths = [...task.matchAll(
      /(?:^|[\s`'"(])((?:\.{1,2}\/)?(?:[A-Za-z0-9._@-]+\/)*[A-Za-z0-9._@-]+\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|php|sh))(?=$|[\s`'"),:;])/gi
    )]
      .map(match => this.normalizeToolWorkspacePath(match[1]!))
      .filter(path => path !== '.'
        && !path.startsWith('.roy/')
        && !path.startsWith('outputs/'));
    const rank = (value: string): number => {
      let score = 0;
      if (/(?:^|\/)src(?:\/|$)/i.test(value)) score += 30;
      if (/(?:audit|implementation|solution|main|app)/i.test(value)) score += 10;
      return score;
    };
    return Array.from(new Set(paths))
      .sort((left, right) => rank(right) - rank(left) || left.localeCompare(right))
      .slice(0, 4);
  }

  private toolPlanFingerprint(
    plan: Pick<PlannedToolCall, 'toolName' | 'params'>
  ): string {
    if (plan.toolName === 'web.fetch') {
      return `${plan.toolName}:${this.canonicalWebDocumentUrl(String(plan.params.url ?? ''))}`;
    }
    if (plan.toolName === 'web.search') {
      const query = String(plan.params.query ?? '')
        .toLocaleLowerCase()
        .replace(/[^\p{L}\p{N}]+/gu, ' ')
        .trim();
      return `${plan.toolName}:${query}`;
    }
    return workspaceToolIntentFingerprint({
      toolName: plan.toolName,
      params: plan.params,
    });
  }

  private sharedReadOnlyToolCacheKey(
    plan: Pick<PlannedToolCall, 'toolName' | 'params'>
  ): string | undefined {
    if (plan.toolName !== 'web.search' && plan.toolName !== 'web.fetch') {
      return undefined;
    }
    return this.toolPlanFingerprint(plan);
  }

  private storeSharedReadOnlyToolResult(
    key: string,
    entry: SharedReadOnlyToolCacheEntry
  ): void {
    const maxEntries = 256;
    this.sharedReadOnlyToolResultCache.delete(key);
    this.sharedReadOnlyToolResultCache.set(key, entry);
    while (this.sharedReadOnlyToolResultCache.size > maxEntries) {
      const oldest = this.sharedReadOnlyToolResultCache.keys().next().value as
        | string
        | undefined;
      if (!oldest) break;
      this.sharedReadOnlyToolResultCache.delete(oldest);
    }
  }

  private cachedToolPlanDecision(
    plan: PlannedToolCall,
    priorCalls: ToolCallRecord[]
  ): { skip: boolean; reason?: string } {
    const fingerprint = this.toolPlanFingerprint(plan);
    let previousIndex = -1;
    for (let index = priorCalls.length - 1; index >= 0; index -= 1) {
      if (this.toolPlanFingerprint(priorCalls[index]!) === fingerprint
        || completedWorkspaceReadCoversPlan(priorCalls[index]!, {
          toolName: plan.toolName,
          params: plan.params,
        })) {
        previousIndex = index;
        break;
      }
    }
    if (previousIndex < 0) return { skip: false };
    const previous = priorCalls[previousIndex]!;
    const laterCalls = priorCalls.slice(previousIndex + 1);
    const effectiveLaterMutations = effectiveWorkspaceMutationCallIndices(laterCalls)
      .map(index => laterCalls[index]!);

    if (plan.toolName === 'fs.read') {
      const target = this.normalizeToolWorkspacePath(String(plan.params.path ?? ''));
      const invalidated = effectiveLaterMutations.some(call =>
        this.workspaceMutationTouchesPath(call, target)
      );
      return invalidated
        ? { skip: false }
        : { skip: true, reason: 'cached_file_read_still_current' };
    }
    if (plan.toolName === 'fs.list') {
      const listedEntries = new Set(
        Array.isArray((previous.result as { entries?: unknown } | undefined)?.entries)
          ? (previous.result as { entries: unknown[] }).entries
            .filter((entry): entry is string => typeof entry === 'string')
            .map(entry => this.normalizeToolWorkspacePath(entry))
          : []
      );
      const structureInvalidated = effectiveLaterMutations.some(call => {
        if (!call.success) return false;
        if (call.toolName === 'fs.replace') return false;
        if (call.toolName === 'fs.write' || call.toolName === 'fs.synthesize') {
          const writtenPath = this.normalizeToolWorkspacePath(String(call.params.path ?? ''));
          return Boolean(writtenPath && !listedEntries.has(writtenPath));
        }
        return call.toolName === 'shell.exec'
          && isSuccessfulWorkspaceMutationCall(call);
      });
      return structureInvalidated
        ? { skip: false }
        : { skip: true, reason: 'cached_workspace_listing_still_current' };
    }
    if (plan.toolName === 'fs.search') {
      const searchRoot = this.normalizeToolWorkspacePath(String(plan.params.path ?? '.'));
      const invalidated = effectiveLaterMutations.some(call =>
        this.workspaceMutationTouchesPath(call, searchRoot, true)
      );
      return invalidated
        ? { skip: false }
        : { skip: true, reason: 'cached_workspace_search_still_current' };
    }
    if (plan.toolName === 'shell.exec'
      && this.isDependencyInstallCommand(String(plan.params.command ?? ''))) {
      const dependencyManifestInvalidated = effectiveLaterMutations.some(call =>
        (call.toolName === 'fs.write'
          || call.toolName === 'fs.replace'
          || call.toolName === 'fs.synthesize')
        && this.isDependencyManifestPath(String(
          (call.result as { path?: unknown } | undefined)?.path
            ?? call.params.path
            ?? ''
        ))
      );
      if (dependencyManifestInvalidated) {
        return {
          skip: false,
          reason: 'dependency_manifest_changed_after_cached_install',
        };
      }
    }
    if (isWorkspaceVerificationCall({ ...plan, success: true })) {
      const timedOut = Boolean(
        (previous.result as { timedOut?: unknown } | undefined)?.timedOut
      );
      if (timedOut) return { skip: false };
      const mutationAfterPrevious = effectiveLaterMutations.length > 0;
      return mutationAfterPrevious
        ? { skip: false }
        : {
          skip: true,
          reason: previous.success
            ? 'cached_verification_still_current'
            : 'cached_failed_verification_without_later_mutation',
        };
    }
    return {
      skip: previous.success,
      reason: previous.success ? 'equivalent_successful_call_already_completed' : undefined,
    };
  }

  private workspaceMutationTouchesPath(
    call: ToolCallRecord,
    target: string,
    includeDescendants = false
  ): boolean {
    if (!isSuccessfulWorkspaceMutationCall(call)) return false;
    if (call.toolName === 'shell.exec') return true;
    const mutated = this.normalizeToolWorkspacePath(String(call.params.path ?? ''));
    if (!mutated || !target) return true;
    return mutated === target
      || (includeDescendants && (
        target === '.'
        || mutated.startsWith(`${target.replace(/\/+$/, '')}/`)
      ));
  }

  private isDependencyInstallCommand(command: string): boolean {
    const normalized = command.trim().replace(/^(?:cd\s+\S+\s*&&\s*)+/i, '');
    return /^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?|uv)\s+install\b/i.test(normalized)
      || /^(?:npm|pnpm|yarn|bun)\s+(?:install|ci)\b/i.test(normalized);
  }

  private isDependencyManifestPath(candidatePath: string): boolean {
    const path = this.normalizeToolWorkspacePath(candidatePath).toLowerCase();
    return /(?:^|\/)(?:pyproject\.toml|setup\.cfg|setup\.py|requirements[^/]*\.txt|package\.json|package-lock\.json|pnpm-lock\.ya?ml|yarn\.lock|bun\.lockb?|cargo\.toml|cargo\.lock|go\.mod|go\.sum|pom\.xml|build\.gradle|gemfile|gemfile\.lock)$/.test(
      path
    );
  }

  private normalizeToolWorkspacePath(value: string): string {
    let normalized = value.trim().replace(/\\/g, '/').replace(/^\.\/+/, '').replace(/\/+/g, '/');
    normalized = normalized.replace(/\/+$/, '');
    const workspaceRoot = this.workspaceRoot.replace(/\\/g, '/').replace(/\/+$/, '');
    if (normalized === workspaceRoot) return '.';
    if (normalized.startsWith(`${workspaceRoot}/`)) {
      normalized = normalized.slice(workspaceRoot.length + 1);
    }
    return normalized || '.';
  }

  private buildGroundedTask(task: string, grounding: { context: string; warnings: string[] }): string {
    if (!grounding.context && grounding.warnings.length === 0) return task;
    return [
      '[runtime_grounding_provided]',
      task,
      '\nGrounding rules:\n- Treat the runtime-provided tool output as authoritative.\n- Mention only files, technologies, commands, test results, compatibility claims, and API semantics supported by that output.\n- A page title, URL, or navigation entry proves only that the page exists; it does not establish the page body, compatibility range, or API behavior.\n- If prior assumptions conflict with the tool output, discard the assumptions.\n- When credible sources support multiple plausible interpretations, state the ambiguity and preserve the supported alternatives instead of silently choosing one.\n- Label anything not established by the tool output as unverified; do not invent a replacement structure.',
      grounding.context ? `\nGrounding context:\n${grounding.context}` : '',
      grounding.warnings.length > 0 ? `\nGrounding warnings:\n${grounding.warnings.join('\n')}` : '',
    ].filter(Boolean).join('\n');
  }

  private resultIncludesEvidence(result: string, evidence: RunEvidence): boolean {
    if (!result.trim()) return false;
    const normalized = result.toLowerCase();
    if (evidence.observedPaths.slice(0, 80).some(item => normalized.includes(item.toLowerCase()))) return true;
    if ((evidence.observedUrls ?? []).slice(0, 30).some(item => {
      try {
        const url = new URL(item);
        return normalized.includes(item.toLowerCase()) || normalized.includes(url.hostname.toLowerCase());
      } catch {
        return false;
      }
    })) return true;
    const evidenceTerms = (evidence.toolResultSummary ?? '')
      .toLowerCase()
      .split(/[^a-z0-9._/-]+/)
      .filter(term => term.length >= 4)
      .slice(0, 30);
    return evidenceTerms.some(term => normalized.includes(term));
  }

  private taskRequiresWorkspaceMutation(task: string): boolean {
    return taskRequestsWorkspaceMutation(task);
  }

  private taskRequiresRootWorkspaceMutation(task: string): boolean {
    return this.taskRequiresWorkspaceMutation(task)
      || this.containsExternalExecutionFeedback(task);
  }

  private buildRootExecutionClosureTask(
    userTask: string,
    subagents: RootMediatedSpawnResult[],
    teamResults: TeamRunResult[]
  ): string {
    const delegatedFindings = teamResults.length > 0
      ? teamResults.slice(-3).map(result =>
        `Team ${result.team.identity.name}:\n${result.result.slice(-3500)}`
      ).join('\n\n')
      : subagents.slice(-4).map(result =>
        `${result.agent.identity.name} (${result.node.identity.archetype}):\n${result.subagentResult.result.slice(-2500)}`
      ).join('\n\n');
    return [
      '[runtime_execution_phase]',
      `Original task:\n${userTask}`,
      `<acceptance_checklist>\n${this.buildTaskAcceptanceChecklist(userTask)}\n</acceptance_checklist>`,
      delegatedFindings
        ? `Delegated findings and proposals:\n${delegatedFindings}`
        : 'No delegated report was available. Inspect the workspace directly.',
      [
        'Execution contract:',
        '- Work on the configured workspace now; do not stop at analysis or a proposed patch.',
        '- The listed tools are already bound to this actor. Request the tool calls directly; Runtime enforces approval policy. Do not ask the user for tool permission.',
        '- Inspect current files as needed, apply the required changes with fs.synthesize, fs.write, fs.replace, or shell.exec, and run relevant verification commands.',
        '- Resolve the authoritative source root from project metadata before writing. Do not create a parallel package outside an existing src/, lib/, packages/, or configured package directory.',
        '- Continue through failed verification when another bounded repair is possible.',
        '- Finish only after the requested workspace mutation has been attempted and the resulting state has been verified or a concrete blocking error has been observed.',
      ].join('\n'),
    ].join('\n\n');
  }

  private buildRootAcceptanceAuditTask(
    userTask: string,
    priorExecution: GroundingRunResult
  ): string {
    const priorEvidence = String(
      priorExecution.evidence.toolResultSummary ?? ''
    )
      .split(/\r?\n/)
      .filter(line =>
        !/(?:^|[\s"'`])\.roy\/(?:agents|cache|sessions|teams|traces)(?:\/|\b)/i.test(
          line
        )
      )
      .join('\n')
      .slice(-12_000);
    const observedProjectPaths = priorExecution.evidence.observedPaths
      .filter(path =>
        !/^\.roy\/(?!official-verifier(?:\/|$))/i.test(
          this.normalizeToolWorkspacePath(path)
        )
      )
      .slice(-120);
    return [
      '[runtime_acceptance_audit_phase]',
      'This is a read-only final audit. Do not modify, edit, write, patch, install, or create files during this audit.',
      `Original task:\n${userTask}`,
      `<acceptance_checklist>\n${this.buildTaskAcceptanceChecklist(userTask)}\n</acceptance_checklist>`,
      [
        'Audit contract:',
        '- Establish the authoritative workspace root and actual project layout before selecting paths.',
        '- Cross-check every parallel manifest, requirements file, generated declaration, configuration source, compatibility path, and relevant call site that could contradict the primary edit.',
        '- Use fs.search for stale declarations or forbidden patterns when text search is applicable; do not assume rg, grep, npm, or another ecosystem command exists.',
        '- Choose verification commands from observed project metadata instead of assuming a language or package manager.',
        '- Run the broadest relevant executable checks that fit the remaining time, preserving their real exit status.',
        '- Return concrete evidence for every acceptance item. A narrow passing check does not verify unrelated items.',
        '- Treat .roy/agents, .roy/cache, .roy/sessions, .roy/teams, and .roy/traces as Runtime control-plane metadata, not project acceptance evidence. Do not inspect them.',
        '- Do not infer a top-level agents/ path from Runtime actor metadata. Read only project paths established by workspace inventory or the authoritative project-path list below.',
        '- The only .roy subtree that may contain project acceptance evidence is the read-only .roy/official-verifier mirror.',
      ].join('\n'),
      `Authoritative observed project paths:\n${observedProjectPaths.join('\n') || 'No project paths were retained.'}`,
      `Prior execution evidence:\n${priorEvidence || 'No prior textual evidence.'}`,
    ].join('\n\n');
  }

  private async runRootAcceptanceAudit(
    userTask: string,
    priorExecution: GroundingRunResult,
    correlationId: string,
    maxWallClockMs: number
  ): Promise<GroundingRunResult> {
    const task = this.buildRootAcceptanceAuditTask(userTask, priorExecution);
    const grounding = await this.runGroundingCheck('root', task, {
      correlationId,
      archetype: 'tester',
      intentTask: task,
      maxWallClockMs,
      toolAllowlist: ['fs.list', 'fs.read', 'fs.search', 'shell.exec'],
    });
    const globalCalls = [...priorExecution.toolCalls, ...grounding.toolCalls];
    const items = this.extractTaskAcceptanceItems(userTask);
    const expectedIds = items.map((_, index) =>
      `acceptance_${String(index + 1).padStart(2, '0')}`
    );
    const inventoryObserved = globalCalls.some(call =>
      call.toolName === 'fs.list' && call.success
    );
    const stateInspected = globalCalls.some(call =>
      call.success && (
        call.toolName === 'fs.read'
        || call.toolName === 'fs.search'
        || call.toolName === 'shell.exec'
      )
    );
    const auditStateInspected = grounding.toolCalls.some(call =>
      call.success && (
        call.toolName === 'fs.list'
        || call.toolName === 'fs.read'
        || call.toolName === 'fs.search'
        || call.toolName === 'shell.exec'
      )
    );
    const verificationPassed = this.analyzeWorkspaceExecutionClosure(globalCalls).verificationPassed;
    const toolEvidenceSufficient = inventoryObserved
      && stateInspected
      && auditStateInspected
      && verificationPassed;
    let normalizedItems: WorkspaceAcceptanceAuditItem[] = expectedIds.map(id => ({
      id,
      status: 'unverified',
      evidence: 'No acceptance classification was returned.',
    }));
    let classificationReason: string;
    try {
      const response = await this.completeJSONAsAgent<{
        items?: Array<{ id?: unknown; status?: unknown; evidence?: unknown }>;
        reason?: unknown;
      }>(
        this.getContext().agent,
        [
          {
            role: 'system',
            content: [
              'You are Roy\'s conservative final acceptance auditor.',
              'Classify every checklist item from authoritative runtime tool evidence only.',
              'Use verified only when the evidence directly proves the complete requirement and no observed sibling declaration contradicts it.',
              'Use failed when evidence contradicts the requirement, blocked for a concrete external blocker, and unverified when evidence is incomplete.',
              'Return JSON only: {"items":[{"id":"acceptance_01","status":"verified|failed|blocked|unverified","evidence":"..."}],"reason":"..."}.',
            ].join('\n'),
          },
          {
            role: 'user',
            content: [
              `<acceptance_checklist>${this.buildTaskAcceptanceChecklist(userTask)}</acceptance_checklist>`,
              `<global_tool_evidence>${[
                priorExecution.evidence.toolResultSummary,
                grounding.evidence.toolResultSummary,
              ].filter(Boolean).join('\n\n').slice(-24_000) || 'none'}</global_tool_evidence>`,
              `<audit_warnings>${grounding.warnings.join('\n') || 'none'}</audit_warnings>`,
              `<audit_structure>${JSON.stringify({
                inventoryObserved,
                stateInspected,
                auditStateInspected,
                verificationPassed,
                observedPaths: Array.from(new Set([
                  ...priorExecution.evidence.observedPaths,
                  ...grounding.evidence.observedPaths,
                ])).slice(0, 120),
              })}</audit_structure>`,
            ].join('\n\n'),
          },
        ],
        { temperature: 0, maxTokens: Math.min(4096, 300 + items.length * 140) },
        'root.acceptance_audit',
        correlationId
      );
      const responseById = new Map(
        (response.items ?? [])
          .filter(item => typeof item.id === 'string')
          .map(item => [String(item.id), item])
      );
      normalizedItems = expectedIds.map(id => {
        const item = responseById.get(id);
        const status = item?.status === 'verified'
          || item?.status === 'failed'
          || item?.status === 'blocked'
          || item?.status === 'unverified'
          ? item.status
          : 'unverified';
        return {
          id,
          status,
          evidence: typeof item?.evidence === 'string'
            ? item.evidence.slice(0, 1000)
            : 'No direct evidence was supplied.',
        };
      });
      classificationReason = typeof response.reason === 'string'
        ? response.reason.slice(0, 2000)
        : 'Acceptance items classified from runtime evidence.';
    } catch (error) {
      this.rethrowRetryableLLMTransportError(error);
      classificationReason = `Acceptance classification failed: ${error instanceof Error ? error.message : String(error)}`;
    }
    const missingItemIds = normalizedItems
      .filter(item => item.status !== 'verified')
      .map(item => item.id);
    const audit: WorkspaceAcceptanceAudit = {
      required: true,
      performed: true,
      passed: toolEvidenceSufficient && missingItemIds.length === 0,
      items: normalizedItems,
      missingItemIds,
      toolEvidenceSufficient,
      reason: classificationReason,
    };
    this.emit({
      type: audit.passed
        ? 'root.acceptance.audit.completed'
        : 'root.acceptance.audit.unmet',
      agentId: 'root',
      correlationId,
      data: {
        ...audit,
        inventoryObserved,
        stateInspected,
        auditStateInspected,
        verificationPassed,
        toolCalls: globalCalls.map(call => ({
          toolName: call.toolName,
          success: call.success,
        })),
      },
    });
    return {
      ...grounding,
      acceptanceAudit: audit,
      warnings: audit.passed
        ? grounding.warnings
        : [
          ...grounding.warnings,
          `Final acceptance audit remains open: ${missingItemIds.join(', ') || 'insufficient global tool evidence'}. ${classificationReason}`,
        ],
    };
  }

  private collectDelegatedExecutionGrounding(
    subagents: RootMediatedSpawnResult[],
    teamResults: TeamRunResult[],
    correlationId: string
  ): GroundingRunResult | undefined {
    const results: RunAgentResult[] = [];
    const seenResults = new Set<RunAgentResult>();
    const addResult = (result: RunAgentResult): void => {
      if (seenResults.has(result)) return;
      seenResults.add(result);
      results.push(result);
    };
    for (const subagent of subagents) addResult(subagent.subagentResult);
    for (const team of teamResults) {
      for (const member of team.members) addResult(member);
    }
    const cachedTeamCalls = [...this.teamToolEvidenceCache.entries()]
      .filter(([teamId]) =>
        this.teamToolEvidenceCorrelations.get(teamId) === correlationId
      )
      .flatMap(([, calls]) => calls);
    if (results.length === 0 && cachedTeamCalls.length === 0) return undefined;

    const seenCalls = new Set<ToolCallRecord>();
    const indexedCalls = results.flatMap((result, resultIndex) =>
      result.toolCalls
        .filter(call => {
          if (seenCalls.has(call)) return false;
          seenCalls.add(call);
          return true;
        })
        .map((call, callIndex) => ({
          call,
          originalIndex: resultIndex * 10_000 + callIndex,
        }))
    );
    for (const call of cachedTeamCalls) {
      if (seenCalls.has(call)) continue;
      seenCalls.add(call);
      indexedCalls.push({
        call,
        originalIndex: indexedCalls.length + results.length * 10_000,
      });
    }
    const allTimestamped = indexedCalls.every(item =>
      Number.isFinite(item.call.completedAt ?? item.call.startedAt)
    );
    if (allTimestamped) {
      indexedCalls.sort((left, right) =>
        (left.call.completedAt ?? left.call.startedAt ?? 0)
        - (right.call.completedAt ?? right.call.startedAt ?? 0)
        || left.originalIndex - right.originalIndex
      );
    }
    const toolCalls = indexedCalls.map(item => item.call);
    const cachedObservedPaths = toolCalls.flatMap(call => {
      const result = call.result as {
        path?: unknown;
        entries?: unknown;
        matches?: Array<{ path?: unknown }>;
      } | undefined;
      return [
        ...(typeof result?.path === 'string' ? [result.path] : []),
        ...(Array.isArray(result?.entries)
          ? result.entries.filter((entry): entry is string => typeof entry === 'string')
          : []),
        ...(Array.isArray(result?.matches)
          ? result.matches
            .map(match => match.path)
            .filter((entry): entry is string => typeof entry === 'string')
          : []),
      ];
    });
    const cachedCallSummary = cachedTeamCalls.length > 0
      ? [
        'Cached team tool frontier:',
        ...cachedTeamCalls.slice(-40).map(call =>
          `- ${call.success ? 'ok' : 'failed'} ${call.toolName} ${JSON.stringify(call.params).slice(0, 300)}`
        ),
      ].join('\n')
      : '';
    const startedAt = Math.min(
      ...results.map(result => result.toolLoop?.startedAt).filter((value): value is number => Number.isFinite(value)),
      Date.now()
    );
    const completedAt = Math.max(
      ...results.map(result => result.toolLoop?.completedAt).filter((value): value is number => Number.isFinite(value)),
      startedAt
    );
    return {
      toolCalls,
      grounded: results.some(result => result.grounded) || toolCalls.some(call => call.success),
      warnings: [
        ...results.flatMap(result => result.warnings),
        ...cachedTeamCalls
          .filter(call => !call.success)
          .map(call => `${call.toolName} failed: ${call.error ?? 'unknown error'}`),
      ],
      context: results
        .map(result => result.evidence.toolResultSummary)
        .filter((value): value is string => Boolean(value))
        .concat(cachedCallSummary ? [cachedCallSummary] : [])
        .join('\n\n'),
      evidence: {
        toolGrounded: results.some(result => result.evidence.toolGrounded)
          || toolCalls.some(call => call.success),
        outputGrounded: results.every(result => result.evidence.outputGrounded),
        observedPaths: Array.from(new Set(
          [
            ...results.flatMap(result => result.evidence.observedPaths),
            ...cachedObservedPaths,
          ]
        )),
        observedUrls: Array.from(new Set(
          results.flatMap(result => result.evidence.observedUrls ?? [])
        )),
        relevantObservedUrls: Array.from(new Set(
          results.flatMap(result => result.evidence.relevantObservedUrls ?? [])
        )),
        discoveredUrls: Array.from(new Set(
          results.flatMap(result => result.evidence.discoveredUrls ?? [])
        )),
        toolResultSummary: results
          .map(result => result.evidence.toolResultSummary)
          .filter((value): value is string => Boolean(value))
          .concat(cachedCallSummary ? [cachedCallSummary] : [])
          .join('\n\n'),
      },
      toolLoop: {
        rounds: [],
        totalCalls: toolCalls.length,
        successfulCalls: toolCalls.filter(call => call.success).length,
        failedCalls: toolCalls.filter(call => !call.success).length,
        stopReason: results.some(result => result.toolLoop?.stopReason === 'max_wall_clock')
          ? 'max_wall_clock'
          : 'completed',
        startedAt,
        completedAt,
      },
    };
  }

  private compactShellCommandForEvidence(command: string): string {
    const normalized = command.trim();
    if (!normalized) return 'command';
    if (normalized.includes('ROY_VERIFIER_PROBE=1')
      || normalized.includes('VERIFIER_PROBE_EVIDENCE_VERSION')) {
      return `ROY_VERIFIER_PROBE=1 [runtime-generated verifier probe omitted; chars=${normalized.length}]`;
    }
    if (normalized.length <= 600) return normalized;
    const fingerprint = this.fingerprint(normalized).slice(0, 16);
    return [
      normalized.slice(0, 360),
      `[runtime_compacted_shell_command chars=${normalized.length} fingerprint=${fingerprint}]`,
      normalized.slice(-120),
    ].join('\n');
  }

  private collectResumedToolGrounding(
    correlationId: string
  ): GroundingRunResult | undefined {
    const resumeState = this.resumedExecutionByCorrelation.get(correlationId);
    if (!resumeState) return undefined;
    const toolCalls = this.restoredToolCallsFromResume(correlationId);
    if (toolCalls.length === 0) return undefined;
    const observedPaths = Array.from(new Set(
      resumeState.knowledge.paths.flatMap(path => path.observedPaths)
    )).slice(0, 160);
    const warnings = resumeState.knowledge.feedback
      .filter(item => item.actionable)
      .sort((left, right) => left.createdAt - right.createdAt)
      .slice(-16)
      .map(item => item.summary);
    const summary = [
      'Persisted causal tool frontier:',
      ...toolCalls.slice(-32).map(call =>
        `- ${call.success ? 'ok' : 'failed'} ${call.toolName} ${JSON.stringify(call.params).slice(0, 500)}`
      ),
      warnings.length > 0
        ? `Persisted actionable feedback:\n${warnings.join('\n')}`
        : '',
    ].filter(Boolean).join('\n');
    const startedAt = toolCalls
      .map(call => call.startedAt)
      .filter((value): value is number => Number.isFinite(value))
      .at(0) ?? Date.now();
    const completedAt = [...toolCalls]
      .reverse()
      .map(call => call.completedAt)
      .find((value): value is number => Number.isFinite(value))
      ?? startedAt;
    this.emit({
      type: 'execution.tool_frontier.resumed',
      agentId: 'root',
      correlationId,
      data: {
        sourceCorrelationId: resumeState.sourceCorrelationId,
        paths: resumeState.knowledge.paths.length,
        toolCalls: toolCalls.length,
        observedPaths: observedPaths.length,
      },
    });
    return {
      toolCalls,
      grounded: toolCalls.some(call => call.success),
      warnings,
      context: summary,
      evidence: {
        toolGrounded: toolCalls.some(call => call.success),
        outputGrounded: false,
        observedPaths,
        toolResultSummary: summary,
      },
      toolLoop: {
        rounds: [],
        totalCalls: toolCalls.length,
        successfulCalls: toolCalls.filter(call => call.success).length,
        failedCalls: toolCalls.filter(call => !call.success).length,
        stopReason: 'completed',
        startedAt,
        completedAt,
      },
    };
  }

  private restoredToolCallsFromResume(
    correlationId: string,
    maxCalls = 96
  ): ToolCallRecord[] {
    const resumeState = this.resumedExecutionByCorrelation.get(correlationId);
    if (!resumeState) return [];
    const persistedFrontier = resumeState.knowledge.paths
      .flatMap(path => path.toolFrontier ?? [])
      .sort((left, right) =>
        (left.completedAt ?? left.startedAt ?? 0)
        - (right.completedAt ?? right.startedAt ?? 0)
      );
    const toolCalls = this.boundExecutionToolFrontier(
      persistedFrontier,
      maxCalls,
      720_000
    ).map(call => ({
        toolName: call.toolName,
        params: call.params,
        result: call.result,
        success: call.success,
        error: call.error,
        reason: call.reason ?? 'Restored from the persisted causal tool frontier.',
        startedAt: call.startedAt,
        completedAt: call.completedAt,
      }));
    return toolCalls;
  }

  private async runRequiredRootExecution(
    userTask: string,
    subagents: RootMediatedSpawnResult[],
    teamResults: TeamRunResult[],
    correlationId: string
  ): Promise<GroundingRunResult> {
    const attempts: GroundingRunResult[] = [];
    const liveDelegatedExecution = this.collectDelegatedExecutionGrounding(
      subagents,
      teamResults,
      correlationId
    );
    const resumedExecution = this.collectResumedToolGrounding(correlationId);
    const delegatedExecution = resumedExecution && liveDelegatedExecution
      ? this.combineGroundingRuns([resumedExecution, liveDelegatedExecution])
      : resumedExecution ?? liveDelegatedExecution;
    const combineWithDelegatedExecution = (): GroundingRunResult => this.combineGroundingRuns(
      delegatedExecution ? [delegatedExecution, ...attempts] : attempts
    );
    const progressWindow = Math.max(
      1,
      this.workspaceRuntimeConfig?.delegation.rootSteps.maxExecutionClosureAttempts ?? 3
    );
    const maxStalledIterations = Math.max(
      1,
      this.workspaceRuntimeConfig?.delegation.rootSteps.maxStalledIterations ?? 3
    );
    let maxAttempts = progressWindow;
    let stalledIterations = 0;
    let acceptanceAuditInvalidated = false;
    const auditRequired = this.taskRequiresAcceptanceAudit(userTask);
    const deferInitialAcceptanceToExternalVerifier =
      this.shouldDeferInitialAcceptanceToExternalVerifier(userTask);
    if (delegatedExecution) {
      const delegatedClosure = this.analyzeWorkspaceExecutionClosure(
        delegatedExecution.toolCalls
      );
      if (delegatedClosure.closed) {
        this.emit({
          type: 'root.execution.delegated.closure.reused',
          agentId: 'root',
          correlationId,
          data: {
            ...delegatedClosure,
            auditRequired,
            deferInitialAcceptanceToExternalVerifier,
            toolCalls: delegatedExecution.toolCalls.length,
          },
        });
        if (!auditRequired || deferInitialAcceptanceToExternalVerifier) {
          if (deferInitialAcceptanceToExternalVerifier) {
            this.emit({
              type: 'root.execution.external_verifier.handoff',
              agentId: 'root',
              correlationId,
              data: {
                reason: 'The external continue-until-timeout protocol has not supplied verifier feedback yet; hand off the closed delegated mutation instead of speculating across unverified acceptance items.',
                mutationApplied: delegatedClosure.mutationApplied,
                verificationPassed: delegatedClosure.verificationPassed,
                toolCalls: delegatedExecution.toolCalls.length,
              },
            });
          }
          return delegatedExecution;
        }
        const auditRemainingMs = this.remainingRootExecutionTimeMs(correlationId);
        if (auditRemainingMs > 5_000) {
          const audit = await this.runRootAcceptanceAudit(
            userTask,
            delegatedExecution,
            correlationId,
            Math.max(1_000, auditRemainingMs)
          );
          attempts.push(audit);
          const auditedExecution = combineWithDelegatedExecution();
          const auditedClosure = this.analyzeWorkspaceExecutionClosure(
            auditedExecution.toolCalls,
            auditedExecution.acceptanceAudit,
            true
          );
          if (auditedClosure.closed) return auditedExecution;
        }
      }
    }
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const remainingMs = this.remainingRootExecutionTimeMs(correlationId);
      if (attempt > 1 && remainingMs <= 5_000) {
        this.emit({
          type: 'root.execution.time_budget.exhausted',
          agentId: 'root',
          correlationId,
          data: { attempt, maxAttempts, remainingMs },
        });
        break;
      }
      const remainingAttempts = Math.max(1, maxAttempts - attempt + 1);
      const attemptWallClockMs = Math.max(
        1_000,
        Math.floor(remainingMs / (remainingAttempts + (auditRequired ? 1 : 0)))
      );
      this.emit({
        type: 'root.execution.time_budget.allocated',
        agentId: 'root',
        correlationId,
        data: {
          attempt,
          maxAttempts,
          remainingMs,
          attemptWallClockMs,
          auditRequired,
        },
      });
      const priorExecution = attempts.length > 0
        ? combineWithDelegatedExecution()
        : undefined;
      const priorClosure = priorExecution
        ? this.analyzeWorkspaceExecutionClosure(
            priorExecution.toolCalls,
            priorExecution.acceptanceAudit,
            auditRequired
          )
        : undefined;
      if (priorClosure?.verificationPassed
        && priorExecution?.acceptanceAudit?.performed
        && !priorExecution.acceptanceAudit.passed
        && !this.acceptanceAuditRequiresMutation(priorExecution.acceptanceAudit)) {
        this.emit({
          type: 'root.execution.acceptance.non_mutating_frontier',
          agentId: 'root',
          correlationId,
          data: {
            attempt,
            reason: 'Acceptance is unverified or blocked, but no audit item contains evidence of a code defect; another mutation would be causally unsupported.',
            missingItemIds: priorExecution.acceptanceAudit.missingItemIds,
            statuses: priorExecution.acceptanceAudit.items.map(item => ({
              id: item.id,
              status: item.status,
            })),
          },
        });
        break;
      }
      let baseTask = priorExecution
        ? this.buildRootExecutionRepairTask(userTask, priorExecution, attempt)
        : this.buildRootExecutionClosureTask(userTask, subagents, teamResults);
      if (!priorExecution && delegatedExecution && delegatedExecution.toolCalls.length > 0) {
        const delegatedClosure = this.analyzeWorkspaceExecutionClosure(
          delegatedExecution.toolCalls
        );
        baseTask = [
          baseTask,
          '<delegated_execution_state>',
          JSON.stringify({
            toolCalls: delegatedExecution.toolCalls.length,
            successfulToolCalls: delegatedExecution.toolCalls.filter(call => call.success).length,
            mutationApplied: delegatedClosure.mutationApplied,
            verificationPassedAfterLatestMutation: delegatedClosure.verificationPassed,
            observedPaths: delegatedExecution.evidence.observedPaths.slice(0, 120),
          }),
          '</delegated_execution_state>',
          [
            'Global closure attention:',
            '- Treat successful delegated mutations as existing workspace state; inspect them before deciding whether another edit is necessary.',
            '- If the delegated state already implements the request, prioritize an independent root verification instead of repeating the edit.',
            '- Repair only concrete remaining defects, then run verification after the latest mutation performed by any actor.',
          ].join('\n'),
        ].join('\n\n');
      }
      const executionKnowledge = this.resumedExecutionByCorrelation.get(correlationId)?.knowledge
        ?? await this.getContext().memory.readExecutionKnowledge(
          userTask,
          this.workspaceRuntimeConfig?.delegation.rootSteps.maxFeedbackItemsInPrompt ?? 24
        );
      const task = [
        baseTask,
        `<execution_resume_ledger>\n${this.formatExecutionResumeBrief(executionKnowledge)}\n</execution_resume_ledger>`,
        [
          'Execution attention:',
          '- Use successful cached paths as the authoritative starting point.',
          '- Do not retry a cached invalid path or equivalent failed call unless this attempt changes the hypothesis.',
          '- Consume actionable feedback, preserve useful state from prior paths, and append new verification evidence.',
        ].join('\n'),
      ].join('\n\n');
      const executionIntentTask = priorExecution?.acceptanceAudit
        ? [
          userTask,
          '<runtime_acceptance_repair_targets>',
          ...priorExecution.acceptanceAudit.items
            .filter(item => item.status === 'failed')
            .map(item => `${item.id}: ${item.evidence}`),
          '</runtime_acceptance_repair_targets>',
        ].join('\n')
        : userTask;
      const current = await this.runGroundingCheck(
        'root',
        task,
        {
          correlationId,
          archetype: 'coder',
          intentTask: executionIntentTask,
          maxWallClockMs: attemptWallClockMs,
          priorToolCalls: priorExecution?.toolCalls ?? delegatedExecution?.toolCalls,
          requireFreshMutation: this.shouldRequireFreshAcceptanceMutation(
            priorClosure,
            acceptanceAuditInvalidated,
            priorExecution?.acceptanceAudit
          ),
        }
      );
      attempts.push(current);
      let combined = combineWithDelegatedExecution();
      if (this.hasSuccessfulWorkspaceMutation(current.toolCalls)) {
        acceptanceAuditInvalidated = true;
        combined.acceptanceAudit = undefined;
        const extendedMaxAttempts = attempt + progressWindow;
        if (extendedMaxAttempts > maxAttempts) {
          const previousMaxAttempts = maxAttempts;
          maxAttempts = extendedMaxAttempts;
          this.emit({
            type: 'root.execution.progress_horizon.extended',
            agentId: 'root',
            correlationId,
            data: {
              attempt,
              previousMaxAttempts,
              maxAttempts,
              progressWindow,
              reason: 'A new successful workspace mutation opened another bounded convergence window.',
            },
          });
        }
      }
      if (acceptanceAuditInvalidated) combined.acceptanceAudit = undefined;
      let closure = this.analyzeWorkspaceExecutionClosure(
        combined.toolCalls,
        combined.acceptanceAudit,
        auditRequired
      );
      if (this.shouldRunRootAcceptanceAudit(
        closure,
        combined.acceptanceAudit,
        auditRequired,
        acceptanceAuditInvalidated
      )) {
        const auditRemainingMs = this.remainingRootExecutionTimeMs(correlationId);
        if (auditRemainingMs > 5_000) {
          const audit = await this.runRootAcceptanceAudit(
            userTask,
            combined,
            correlationId,
            Math.max(1_000, auditRemainingMs)
          );
          attempts.push(audit);
          acceptanceAuditInvalidated = false;
          combined = combineWithDelegatedExecution();
          closure = this.analyzeWorkspaceExecutionClosure(
            combined.toolCalls,
            combined.acceptanceAudit,
            auditRequired
          );
        }
      }
      this.emit({
        type: 'root.execution.attempt.completed',
        agentId: 'root',
        correlationId,
        data: {
          attempt,
          maxAttempts,
          ...closure,
          verificationRan: closure.verificationPassed,
          toolCalls: current.toolCalls.length,
          acceptanceAuditToolCalls: combined.acceptanceAudit
            ? attempts.at(-1)?.toolCalls.length ?? 0
            : 0,
          stopReason: attempts.at(-1)?.toolLoop.stopReason ?? current.toolLoop.stopReason,
        },
      });
      if (closure.closed) return combined;
      const attemptAdvanced = this.rootExecutionAttemptAdvanced(
        current,
        priorExecution ?? delegatedExecution
      );
      if (!attemptAdvanced) {
        stalledIterations += 1;
        const stableCausalFrontier = priorExecution ?? delegatedExecution;
        this.emit({
          type: 'root.execution.no_progress.detected',
          agentId: 'root',
          correlationId,
          data: {
            attempt,
            maxAttempts,
            stalledIterations,
            maxStalledIterations,
            stateUnchanged: Boolean(stableCausalFrontier?.toolCalls.length),
            toolCallsProduced: current.toolCalls.length,
            reason: stableCausalFrontier?.toolCalls.length
              ? 'The grounded repair planner produced no new successful action or failure evidence from an unchanged causal frontier; repeating it would be redundant.'
              : 'The grounded repair planner exhausted its internal correction attempts without producing a new tool action.',
            latestFailure: priorExecution?.warnings.at(-1),
            ...closure,
          },
        });
        if (stableCausalFrontier?.toolCalls.length) break;
        if (stalledIterations >= maxStalledIterations) break;
      } else {
        stalledIterations = 0;
      }
    }
    return combineWithDelegatedExecution();
  }

  private shouldDeferInitialAcceptanceToExternalVerifier(task: string): boolean {
    const externalLoopDeclared =
      /\bcontinue(?:[-_ ]+)until(?:[-_ ]+)timeout\b/i.test(task);
    if (!externalLoopDeclared) return false;
    return !/<official_verifier_feedback>|<recovery_capsule>|\bVERIFICATION FAILED\b/i.test(
      task
    );
  }

  private rootExecutionAttemptAdvanced(
    current: GroundingRunResult,
    prior?: GroundingRunResult
  ): boolean {
    if (current.toolCalls.some(call => call.success)) return true;
    if (current.toolCalls.length === 0) return false;
    const priorFailures = new Set(
      (prior?.toolCalls ?? [])
        .filter(call => !call.success)
        .map(call => this.failedToolCallEvidenceFingerprint(call))
    );
    return current.toolCalls.some(call =>
      !call.success && !priorFailures.has(this.failedToolCallEvidenceFingerprint(call))
    );
  }

  private failedToolCallEvidenceFingerprint(
    call: Pick<ToolCallRecord, 'toolName' | 'params' | 'result' | 'error'>
  ): string {
    return createHash('sha256').update(JSON.stringify({
      toolName: call.toolName,
      params: call.params,
      error: call.error ?? null,
      result: call.result ?? null,
    })).digest('hex');
  }

  private buildRootExecutionRepairTask(
    userTask: string,
    priorExecution: GroundingRunResult,
    attempt: number
  ): string {
    return [
      '[runtime_execution_repair_phase]',
      `Original task:\n${userTask}`,
      `Execution attempt ${attempt - 1} did not satisfy the required mutation-and-verification closure.`,
      [
        `Closure status: ${JSON.stringify(this.analyzeWorkspaceExecutionClosure(
          priorExecution.toolCalls,
          priorExecution.acceptanceAudit,
          this.taskRequiresAcceptanceAudit(userTask)
        ))}`,
        priorExecution.acceptanceAudit
          ? [
            'Acceptance repair targets:',
            ...priorExecution.acceptanceAudit.items
              .filter(item => item.status !== 'verified')
              .map(item => `- ${item.id} [${item.status}]: ${item.evidence}`),
          ].join('\n')
          : '',
        `Prior tool evidence:\n${priorExecution.evidence.toolResultSummary?.slice(-10_000) || 'No textual tool evidence was available.'}`,
        priorExecution.warnings.length > 0
          ? `Prior warnings:\n${priorExecution.warnings.slice(-8).join('\n')}`
          : '',
      ].filter(Boolean).join('\n\n'),
      [
        'Recovery contract:',
        '- Inspect the actual configured workspace state and continue the original implementation; do not create marker or progress files.',
        '- Use failed command output to repair remaining defects. Do not repeat an equivalent failed call.',
        '- Do not hide failure with shell status masking.',
        '- Apply any remaining workspace changes and run a relevant verification command whose exit status is preserved.',
        '- Finish only when the original task is implemented and verification succeeds, or after a concrete blocking error is observed.',
      ].join('\n'),
    ].join('\n\n');
  }

  private buildAgentExecutionClosureTask(
    originalTask: string,
    priorExecution: GroundingRunResult,
    closure: WorkspaceExecutionClosureStatus,
    continuation: number
  ): string {
    const nextAction = !closure.mutationApplied
      ? 'Use the authoritative evidence already collected and apply the implementation mutation now.'
      : !closure.verificationAttemptedAfterMutation
        ? 'Run the most relevant executable verification against the mutated workspace now.'
        : 'Use the latest failed verification as causal feedback, inspect only its reported source location, repair it, and verify again.';
    return [
      '[runtime_agent_execution_closure_phase]',
      `Original task:\n${this.compactDelegatedTask(originalTask)}`,
      `Continuation ${continuation}; closure state: ${JSON.stringify(closure)}`,
      nextAction,
      [
        'Execution contract:',
        '- Reuse prior tool evidence as cached state; do not restart workspace discovery or repeat equivalent reads.',
        '- A natural-language claim cannot satisfy this phase. The required state change must be observed through an authorized mutation tool.',
        '- Preserve the real verification exit status and use any failure as repair feedback.',
        '- Continue with one concrete highest-value tool action; do not finish while the closure state is open.',
      ].join('\n'),
      priorExecution.warnings.length > 0
        ? `Latest warnings:\n${priorExecution.warnings.slice(-4).join('\n')}`
        : '',
    ].filter(Boolean).join('\n\n');
  }

  private agentExecutionContinuationAdvanced(run: GroundingRunResult): boolean {
    return run.toolCalls.some(call =>
      call.success
      || (
        call.toolName === 'shell.exec'
        && isWorkspaceVerificationCall(call)
        && Boolean(call.error)
      )
    );
  }

  private combineGroundingRuns(runs: GroundingRunResult[]): GroundingRunResult {
    const first = runs[0];
    const last = runs[runs.length - 1];
    if (!first || !last) {
      throw new Error('Cannot combine an empty set of grounding runs');
    }
    const toolCalls = runs.flatMap(run => run.toolCalls);
    return {
      toolCalls,
      grounded: runs.some(run => run.grounded),
      warnings: runs.flatMap(run => run.warnings),
      context: runs.map(run => run.context).filter(Boolean).join('\n\n'),
      evidence: {
        toolGrounded: runs.some(run => run.evidence.toolGrounded),
        outputGrounded: runs.every(run => run.evidence.outputGrounded),
        observedPaths: Array.from(new Set(runs.flatMap(run => run.evidence.observedPaths))),
        observedUrls: Array.from(new Set(runs.flatMap(run => run.evidence.observedUrls ?? []))),
        relevantObservedUrls: Array.from(new Set(
          runs.flatMap(run => run.evidence.relevantObservedUrls ?? [])
        )),
        discoveredUrls: Array.from(new Set(
          runs.flatMap(run => run.evidence.discoveredUrls ?? [])
        )),
        toolResultSummary: runs
          .map(run => run.evidence.toolResultSummary)
          .filter((summary): summary is string => Boolean(summary))
          .join('\n\n'),
      },
      toolLoop: {
        rounds: runs.flatMap(run => run.toolLoop.rounds),
        totalCalls: runs.reduce((sum, run) => sum + run.toolLoop.totalCalls, 0),
        successfulCalls: runs.reduce((sum, run) => sum + run.toolLoop.successfulCalls, 0),
        failedCalls: runs.reduce((sum, run) => sum + run.toolLoop.failedCalls, 0),
        stopReason: last.toolLoop.stopReason,
        startedAt: first.toolLoop.startedAt,
        completedAt: last.toolLoop.completedAt,
      },
      acceptanceAudit: [...runs].reverse()
        .find(run => run.acceptanceAudit)?.acceptanceAudit,
    };
  }

  private remainingRootExecutionTimeMs(correlationId: string): number {
    const tree = this.executionTrees.get(correlationId);
    if (!tree) return this.workspaceRuntimeConfig?.tools.executionLoop.maxWallClockMs ?? 120_000;
    const configuredFinalizationReserveMs = Math.max(
      0,
      this.workspaceRuntimeConfig?.delegation.rootSteps.finalizationReserveMs ?? 30_000
    );
    const finalizationReserveMs = Math.min(
      configuredFinalizationReserveMs,
      Math.floor(tree.loop.maxWallClockMs * 0.2)
    );
    const elapsedMs = Math.max(0, Date.now() - tree.createdAt);
    return Math.max(1_000, tree.loop.maxWallClockMs - elapsedMs - finalizationReserveMs);
  }

  private hasSuccessfulWorkspaceMutation(calls: ToolCallRecord[]): boolean {
    return hasEffectiveWorkspaceMutationCall(calls);
  }

  private hasSuccessfulWorkspaceVerification(calls: ToolCallRecord[]): boolean {
    return calls.some(call => isSuccessfulWorkspaceVerificationCall(call));
  }

  private analyzeWorkspaceExecutionClosure(
    calls: ToolCallRecord[],
    acceptanceAudit?: WorkspaceAcceptanceAudit,
    acceptanceAuditRequired = false
  ): WorkspaceExecutionClosureStatus {
    const lastMutationCallIndex = effectiveWorkspaceMutationCallIndices(calls).at(-1) ?? -1;
    const allVerificationCalls = calls
      .map((call, index) => ({ call, index }))
      .filter(item =>
        item.index >= lastMutationCallIndex
        && isWorkspaceVerificationCall(item.call)
      );
    const lastVerification = allVerificationCalls.at(-1);
    const mutationApplied = lastMutationCallIndex >= 0;
    const verificationAttemptedAfterMutation =
      mutationApplied && allVerificationCalls.length > 0;
    const latestVerificationByIntent = new Map<string, ToolCallRecord>();
    for (const item of allVerificationCalls) {
      latestVerificationByIntent.set(
        this.toolPlanFingerprint(item.call),
        item.call
      );
    }
    const availableLatestVerifications = [...latestVerificationByIntent.values()]
      .filter(call => !isUnavailableWorkspaceVerificationCall(call));
    const unresolvedVerificationFailures = availableLatestVerifications
      .filter(call => !isSuccessfulWorkspaceVerificationCall(call)).length;
    const verificationPassed = verificationAttemptedAfterMutation
      && unresolvedVerificationFailures === 0
      && availableLatestVerifications
        .some(call => isSuccessfulWorkspaceVerificationCall(call));
    const acceptanceItems = acceptanceAudit?.items.length ?? 0;
    const acceptanceItemsVerified = acceptanceAudit?.items
      .filter(item => item.status === 'verified').length ?? 0;
    const acceptanceItemsFailed = acceptanceAudit?.items
      .filter(item => item.status === 'failed' || item.status === 'blocked').length ?? 0;
    const acceptanceAuditPerformed = acceptanceAudit?.performed === true;
    const acceptanceAuditPassed = !acceptanceAuditRequired || acceptanceAudit?.passed === true;
    return {
      mutationApplied,
      verificationAttemptedAfterMutation,
      verificationPassed,
      acceptanceAuditRequired,
      acceptanceAuditPerformed,
      acceptanceAuditPassed,
      acceptanceItems,
      acceptanceItemsVerified,
      acceptanceItemsFailed,
      closed: mutationApplied && verificationPassed && acceptanceAuditPassed,
      lastMutationCallIndex,
      lastVerificationCallIndex: lastVerification?.index ?? -1,
      failedVerificationCallsAfterMutation: allVerificationCalls
        .filter(item =>
          !isUnavailableWorkspaceVerificationCall(item.call)
          && !isSuccessfulWorkspaceVerificationCall(item.call)
        ).length,
      unresolvedVerificationFailures,
    };
  }

  private shouldRunRootAcceptanceAudit(
    closure: WorkspaceExecutionClosureStatus,
    acceptanceAudit: WorkspaceAcceptanceAudit | undefined,
    auditRequired: boolean,
    acceptanceAuditInvalidated: boolean
  ): boolean {
    return closure.mutationApplied
      && closure.verificationPassed
      && auditRequired
      && (acceptanceAuditInvalidated || !acceptanceAudit);
  }

  private shouldRequireFreshAcceptanceMutation(
    priorClosure: WorkspaceExecutionClosureStatus | undefined,
    acceptanceAuditInvalidated: boolean,
    acceptanceAudit?: WorkspaceAcceptanceAudit
  ): boolean {
    return !acceptanceAuditInvalidated
      && Boolean(
        priorClosure?.acceptanceAuditPerformed
        && !priorClosure.acceptanceAuditPassed
        && this.acceptanceAuditRequiresMutation(acceptanceAudit)
      );
  }

  private acceptanceAuditRequiresMutation(
    acceptanceAudit: WorkspaceAcceptanceAudit | undefined
  ): boolean {
    return acceptanceAudit?.items.some(item => item.status === 'failed') === true;
  }

  private summarizeRootExecutionClosure(execution: GroundingRunResult): string {
    const closure = this.analyzeWorkspaceExecutionClosure(
      execution.toolCalls,
      execution.acceptanceAudit,
      execution.acceptanceAudit?.required === true
    );
    const successfulCalls = execution.toolCalls.filter(call => call.success).length;
    return [
      `Root execution closure: mutationApplied=${closure.mutationApplied}, `
        + `verificationAttemptedAfterMutation=${closure.verificationAttemptedAfterMutation}, `
        + `verificationPassed=${closure.verificationPassed}, closed=${closure.closed}.`,
      `Tool calls: ${successfulCalls}/${execution.toolCalls.length} succeeded.`,
      execution.evidence.toolResultSummary?.slice(0, 4000) || 'No runtime tool evidence was produced.',
      execution.warnings.length > 0 ? `Warnings: ${execution.warnings.join('; ')}` : '',
    ].filter(Boolean).join('\n');
  }

  private isToolPermissionClarification(reason: string, question: string): boolean {
    return /\b(?:permission|approval|allowed|authorized|may i|can i|do i have permission)\b[\s\S]{0,120}\b(?:read|inspect|write|edit|modify|execute|run|shell|terminal|tool|files?)\b/i.test(`${reason}\n${question}`)
      || /(?:是否|能否|可以|允许|授权)[\s\S]{0,80}(?:读取|检查|写入|修改|执行|运行|终端|工具|文件)/.test(`${reason}\n${question}`);
  }

  private taskRequiresGrounding(archetype: SubAgentArchetype, task: string): boolean {
    void archetype;
    if (this.taskNeedsWebAccess(task)) return true;
    return /\b(?:filesystem|repository|repo|codebase|workspace|source files?|package\.json|manifest|runtime trace|tool)[ -]?(?:grounded|evidence)\b|\busing (?:filesystem|repository|repo|workspace|source|tool) evidence\b/i.test(task)
      || /\bcollect (?:concrete|grounded) evidence\b[\s\S]{0,180}\b(?:tools?|source files?|filesystem|repository|repo|workspace)\b/i.test(task)
      || /\b(?:inspect|read|open|list|search|audit|modify|edit|write|patch|implement|run|execute|verify|test)\b[\s\S]{0,160}\b(?:actual|local|current)?\s*(?:project(?:\s+structure)?|files?|filesystem|repository|repo|codebase|workspace|source|tests?|build|commands?|cli|container)\b/i.test(task)
      || /(?:检查|读取|打开|列出|搜索|审计|修改|编辑|写入|修复|实现|运行|执行|验证|测试)[\s\S]{0,100}(?:文件|仓库|代码库|工作区|源码|测试|构建|命令|CLI|容器)/i.test(task);
  }

  private taskRequiresFetchedWebEvidence(task: string): boolean {
    return this.taskNeedsWebAccess(task)
      && /\b(?:open|fetch|read|inspect|compare|verify)\b[\s\S]*\b(?:pages?|websites?|urls?|sources?|documentation)\b/i.test(task);
  }

  private requiredWebFetchCount(task: string): number {
    if (!this.taskRequiresFetchedWebEvidence(task)) return 0;
    const explicitDocuments = new Set(
      (task.match(/https?:\/\/[^\s`'"<>),]+/gi) ?? []).map(url => this.canonicalWebDocumentUrl(url))
    );
    if (explicitDocuments.size >= 2) return 2;
    if (/\b(?:at least|minimum of)\s+(?:two|2)\b|\b(?:two|2)\s+(?:independent|relevant|public)?\s*(?:pages?|websites?|urls?|sources?)\b/i.test(task)) {
      return 2;
    }
    if (/\bboth\s+(?:pages?|websites?|urls?|sources?|documents?)\b/i.test(task)) return 2;
    return 1;
  }

  private canonicalWebDocumentUrl(input: string): string {
    try {
      const url = new URL(input);
      url.hash = '';
      return url.toString().replace(/\/$/, '');
    } catch {
      return input;
    }
  }

  private async enforceRootEvidenceBoundary(
    response: string,
    task: string,
    evidences: RunEvidence[],
    correlationId: string,
    purpose: string
  ): Promise<string> {
    const observedUrls = Array.from(new Set(evidences.flatMap(evidence => evidence.observedUrls ?? [])));
    if (observedUrls.length === 0 && !this.taskNeedsWebAccess(task)) return response;
    const unsupported = this.findUnsupportedResponseUrls(response, observedUrls);
    if (unsupported.length === 0) return response;

    this.emit({
      type: 'root.synthesis.grounding.warning',
      agentId: 'root',
      correlationId,
      data: { purpose, reason: 'unsupported_urls', unsupportedUrls: unsupported, observedUrls },
    });
    const evidenceSummary = evidences
      .map(evidence => evidence.toolResultSummary ?? '')
      .filter(Boolean)
      .join('\n\n')
      .slice(0, 16_000);
    let repaired: string;
    try {
      repaired = await this.completeAsRoot(
        [
          'Rewrite the draft as a strictly evidence-grounded final answer.',
          'Use only facts in the runtime evidence. Cite only URLs in <allowed_urls>.',
          'Do not mention, cite, reconstruct, or replace unsupported URLs. Preserve explicit limitations.',
          `<task>${task}</task>`,
          `<allowed_urls>\n${observedUrls.map(url => `- ${url}`).join('\n')}\n</allowed_urls>`,
          `<runtime_evidence>\n${evidenceSummary || 'No textual evidence summary was available.'}\n</runtime_evidence>`,
          `<draft>\n${response}\n</draft>`,
        ].join('\n\n'),
        `${purpose}.grounding_repair`,
        correlationId
      );
    } catch (error) {
      this.emit({
        type: 'root.synthesis.grounding.repair_skipped',
        agentId: 'root',
        correlationId,
        data: { purpose, reason: error instanceof Error ? error.message : String(error) },
      });
      repaired = this.removeUnsupportedUrls(response, unsupported);
    }
    const candidate = repaired.trim() || response;
    const remainingUnsupported = this.findUnsupportedResponseUrls(candidate, observedUrls);
    if (remainingUnsupported.length === 0) {
      this.emit({
        type: 'root.synthesis.grounding.repaired',
        agentId: 'root',
        correlationId,
        data: { purpose, removedUnsupportedUrls: unsupported },
      });
      return candidate;
    }

    this.emit({
      type: 'root.synthesis.grounding.repair_failed',
      agentId: 'root',
      correlationId,
      data: { purpose, unsupportedUrls: remainingUnsupported },
    });
    return [
      'Roy removed unverified citations from the generated synthesis. The remaining answer is limited to runtime-observed sources.',
      this.removeUnsupportedUrls(candidate, remainingUnsupported),
    ].join('\n\n').trim();
  }

  private findUnsupportedResponseUrls(response: string, observedUrls: string[]): string[] {
    const allowed = new Set(observedUrls.map(url => this.normalizeEvidenceUrl(url)));
    return Array.from(new Set(
      (response.match(/https?:\/\/[^\s<>'"`\])}]+/gi) ?? [])
        .map(url => url.replace(/[.,;:!?]+$/, ''))
        .filter(url => !allowed.has(this.normalizeEvidenceUrl(url)))
    ));
  }

  private normalizeEvidenceUrl(input: string): string {
    try {
      const url = new URL(input);
      return url.toString().replace(/\/$/, '');
    } catch {
      return input.replace(/\/$/, '');
    }
  }

  private removeUnsupportedUrls(response: string, unsupported: string[]): string {
    return unsupported.reduce(
      (current, url) => current.split(url).join('[unverified URL removed]'),
      response
    );
  }

  private detectEvidenceContradictions(result: string, evidence: RunEvidence): string[] {
    if (!result.trim() || evidence.observedPaths.length === 0) return [];
    const observed = evidence.observedPaths.map(item =>
      this.normalizeToolWorkspacePath(item).toLowerCase()
    );
    const normalized = result.toLowerCase();
    const hasNodeEvidence = observed.some(item => item === 'package.json' || /\.(?:ts|tsx|js)$/.test(item));
    const hasRustEvidence = observed.some(item => item === 'cargo.toml' || item.endsWith('.rs'));
    const claimsRust = /\bcargo\.(?:toml|lock)\b|\brust-toolchain(?:\.toml)?\b|(?:^|[\s`'"(])[^\s`'"()]+\.rs\b|\brust (?:project|codebase|crate|toolchain)\b/im.test(normalized);
    const contradictions: string[] = [];
    if (hasNodeEvidence && !hasRustEvidence && claimsRust) {
      contradictions.push('The model report claims a Rust/Cargo project, but runtime filesystem evidence contains Node/TypeScript markers and no Rust project markers.');
    }
    const evidenceText = [
      ...evidence.observedPaths,
      ...observed,
      evidence.toolResultSummary ?? '',
    ].join('\n').toLowerCase();
    const unsupportedPaths = Array.from(result.matchAll(/`([^`\n]{1,180})`/g))
      .map(match => match[1].trim().replace(/^["']|["']$/g, '').replace(/^\.\//, '').replace(/[,:;.)]+$/, ''))
      .filter(candidate => this.looksLikeConcreteProjectPath(candidate))
      .filter(candidate => {
        const normalizedCandidate =
          this.normalizeToolWorkspacePath(candidate).toLowerCase();
        return !evidenceText.includes(normalizedCandidate)
          && !observed.some(item => item === normalizedCandidate || item.endsWith(`/${normalizedCandidate}`));
      });
    const uniqueUnsupportedPaths = Array.from(new Set(unsupportedPaths));
    if (uniqueUnsupportedPaths.length >= 2) {
      contradictions.push(
        `The model report references concrete project paths not present in runtime evidence: ${uniqueUnsupportedPaths.slice(0, 6).join(', ')}.`
      );
    }
    return contradictions;
  }

  private looksLikeConcreteProjectPath(value: string): boolean {
    if (!value || /\s|[*{}<>]|^(?:https?:|npm |pnpm |yarn )/i.test(value)) return false;
    if (/^(?:src|test|tests|docs|config|scripts|lib|app|packages)\//i.test(value)) return true;
    return /(?:^|\/)[A-Za-z0-9_.-]+\.(?:ts|tsx|js|jsx|json|ya?ml|toml|md|txt|lock|env)$/i.test(value);
  }

  private attachRuntimeEvidence(result: string, evidence: RunEvidence): string {
    const observedPaths = evidence.observedPaths.slice(0, 80);
    const observedUrls = (evidence.observedUrls ?? []).slice(0, 30);
    const structuredEvidence = [
      ...observedPaths.map(item => `- path: ${item}`),
      ...observedUrls.map(item => `- url: ${item}`),
    ];
    const evidenceBlock = structuredEvidence.length > 0
      ? structuredEvidence.join('\n')
      : (evidence.toolResultSummary ?? 'No structured evidence summary was available.').slice(0, 4000);
    return [
      result.trim(),
      '## Runtime-Verified Evidence',
      evidenceBlock,
    ].filter(Boolean).join('\n\n');
  }

  private containsUnresolvedToolIntent(result: string): boolean {
    if (!result.trim()) return false;
    return /<tool_call>[\s\S]*?<\/tool_call>/i.test(result)
      || /<tool_calls>[\s\S]*?<\/tool_calls>/i.test(result)
      || /<function>\s*<name>\s*(?:web\.(?:search|fetch)|fs\.(?:list|read|search|replace|write|synthesize)|shell\.exec)\s*<\/name>[\s\S]*?<\/function>/i.test(result)
      || /<tool_name>[\s\S]*?<\/tool_name>/i.test(result)
      || /<function_calls>[\s\S]*?<\/function_calls>/i.test(result)
      || /<invocation\s+name=["'](?:web\.(?:search|fetch)|fs\.(?:list|read|search|replace|write|synthesize)|shell\.exec)["'][\s\S]*?<\/invocation>/i.test(result)
      || /<invoke\s+name=["'](?:web\.(?:search|fetch)|fs\.(?:list|read|search|replace|write|synthesize)|shell\.exec)["'][\s\S]*?<\/invoke>/i.test(result)
      || /```(?:tool|json)?\s*\n\s*(?:web\.(?:search|fetch)|fs\.(?:list|read|search|replace|write|synthesize)|shell\.exec)\b[\s\S]*?```/i.test(result)
      || /\{\s*"(?:tool_name|tool|function)"\s*:\s*"[^"\n]+"[\s\S]*?\}/i.test(result);
  }

  private extractUnresolvedToolPlans(agentId: string, result: string): PlannedToolCall[] {
    const authorized = new Set(
      (this.agentBindings.get(agentId)?.tools ?? [])
        .filter(binding => binding.enabled)
        .map(binding => binding.name)
    );
    const candidates: Array<{ toolName: string; params: Record<string, unknown> }> = [];
    const parseParams = (body: string): Record<string, unknown> => {
      const argumentText = body.match(/<arguments?>\s*([\s\S]*?)\s*<\/arguments?>/i)?.[1]
        ?? body.match(/<params?>\s*([\s\S]*?)\s*<\/params?>/i)?.[1];
      if (argumentText) {
        try {
          const parsed = JSON.parse(argumentText);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
          }
        } catch {
          // Fall through to simple XML field extraction.
        }
      }
      const params: Record<string, unknown> = {};
      for (const field of [
        'path',
        'query',
        'command',
        'content',
        'oldText',
        'newText',
        'cwd',
        'url',
      ]) {
        const match = body.match(new RegExp(`<${field}>\\s*([\\s\\S]*?)\\s*</${field}>`, 'i'));
        if (match?.[1] !== undefined) params[field] = match[1];
      }
      for (const match of body.matchAll(
        /<parameter\s+name=["']([^"']+)["']([^>]*)>([\s\S]*?)<\/parameter>/gi
      )) {
        const name = String(match[1] ?? '').trim();
        if (!name) continue;
        const attributes = String(match[2] ?? '');
        const rawValue = decodeBasicXmlEntities(String(match[3] ?? '').trim());
        if (/\bstring=["']true["']/i.test(attributes)) {
          params[name] = rawValue;
          continue;
        }
        try {
          params[name] = JSON.parse(rawValue);
        } catch {
          params[name] = rawValue;
        }
      }
      return params;
    };
    for (const match of result.matchAll(/<tool_call>([\s\S]*?)<\/tool_call>/gi)) {
      const body = match[1] ?? '';
      const toolName = body.match(/<tool_name>\s*([^<]+?)\s*<\/tool_name>/i)?.[1]?.trim();
      if (toolName) candidates.push({ toolName, params: parseParams(body) });
    }
    for (const match of result.matchAll(/<function>([\s\S]*?)<\/function>/gi)) {
      const body = match[1] ?? '';
      const toolName = body.match(/<name>\s*([^<]+?)\s*<\/name>/i)?.[1]?.trim();
      if (toolName) candidates.push({ toolName, params: parseParams(body) });
    }
    for (const match of result.matchAll(/<(?:invocation|invoke)\s+name=["']([^"']+)["'][^>]*>([\s\S]*?)<\/(?:invocation|invoke)>/gi)) {
      candidates.push({
        toolName: String(match[1] ?? '').trim(),
        params: parseParams(match[2] ?? ''),
      });
    }
    for (const match of result.matchAll(/```(?:tool|json)?\s*\n\s*((?:web\.(?:search|fetch)|fs\.(?:list|read|search|replace|write|synthesize)|shell\.exec))\s*\n([\s\S]*?)```/gi)) {
      try {
        const parsed = JSON.parse((match[2] ?? '').trim());
        candidates.push({
          toolName: String(match[1] ?? '').trim(),
          params: parsed && typeof parsed === 'object' && !Array.isArray(parsed)
            ? parsed as Record<string, unknown>
            : {},
        });
      } catch {
        // Invalid model-authored JSON remains unexecuted and will fail grounding.
      }
    }
    const trimmed = result.trim().replace(/^```json\s*/i, '').replace(/\s*```$/i, '');
    if (trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed) as {
          tool_name?: unknown;
          tool?: unknown;
          function?: unknown;
          arguments?: unknown;
          params?: unknown;
        };
        const toolName = String(parsed.tool_name ?? parsed.tool ?? parsed.function ?? '').trim();
        const rawParams = parsed.arguments ?? parsed.params;
        const params = typeof rawParams === 'string'
          ? JSON.parse(rawParams)
          : rawParams;
        candidates.push({
          toolName,
          params: params && typeof params === 'object' && !Array.isArray(params)
            ? params as Record<string, unknown>
            : {},
        });
      } catch {
        // Other structured forms may already have been recovered above.
      }
    }
    const seen = new Set<string>();
    return candidates
      .filter(candidate => candidate.toolName && authorized.has(candidate.toolName))
      .filter(candidate => {
        const validation = toolRegistry.get(candidate.toolName)?.validate?.(candidate.params);
        if (validation?.valid !== false) return true;
        this.emit({
          type: 'agent.output.tool_intent.invalid',
          agentId,
          sessionId: this.getContext().sessionId,
          data: {
            toolName: candidate.toolName,
            params: candidate.params,
            errors: validation.errors ?? [],
          },
        });
        return false;
      })
      .map(candidate => ({
        toolName: candidate.toolName,
        params: candidate.params,
        reason: 'Execute the model-authored tool intent through Runtime authorization and feed the result back before finalizing.',
        groundingRequired: true,
      }))
      .filter(plan => {
        const fingerprint = this.toolPlanFingerprint(plan);
        if (seen.has(fingerprint)) return false;
        seen.add(fingerprint);
        return true;
      });
  }

  private resolveInspectionRoot(task: string): string {
    const match = task.match(/(?:\.{1,2}\/|\/)[A-Za-z0-9._/@-]+/);
    if (!match) return this.workspaceRoot;
    const workspaceRoot = this.workspaceRoot;
    const candidate = path.resolve(workspaceRoot, match[0]);
    const relative = path.relative(workspaceRoot, candidate);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      return workspaceRoot;
    }
    return candidate;
  }

  private buildAgentTree(agent: AgentInfo): AgentTreeNode {
    return {
      agent,
      children: this.getChildren(agent.identity.id).map(child => this.buildAgentTree(child)),
    };
  }

  private buildRuntimeAgentActorTree(agent: AgentInfo, visited: Set<string>): RuntimeAgentActorNode {
    const key = `agent:${agent.identity.id}`;
    if (visited.has(key)) return { type: 'agent', agent, children: [] };
    const nextVisited = new Set(visited).add(key);
    const teamChildren = this.teams.list()
      .filter(team => team.identity.parentAgentId === agent.identity.id)
      .map(team => this.buildRuntimeTeamActorTree(team, nextVisited));
    const agentChildren = this.getChildren(agent.identity.id)
      .filter(child => !child.identity.teamId)
      .map(child => this.buildRuntimeAgentActorTree(child, nextVisited));
    return {
      type: 'agent',
      agent,
      children: [...teamChildren, ...agentChildren],
    };
  }

  private buildRuntimeTeamActorTree(team: TeamRuntimeState, visited: Set<string>): RuntimeTeamActorNode {
    const key = `team:${team.identity.id}`;
    if (visited.has(key)) return { type: 'team', team, children: [] };
    const nextVisited = new Set(visited).add(key);
    const ctx = this.getContext();
    const children = team.memberAgentIds
      .map(agentId => ctx.manager.getAgentById(agentId)?.getInfo())
      .filter((agent): agent is AgentInfo => Boolean(agent))
      .map(agent => this.buildRuntimeAgentActorTree(agent, nextVisited));
    return { type: 'team', team, children };
  }

  private createAgentId(archetype: SubAgentArchetype, sequence: number): string {
    return `agent_${archetype}_${String(sequence).padStart(3, '0')}`;
  }

  private createUniqueAgentName(archetype: SubAgentArchetype, requestedName: string, sequence: number): string {
    const defaultPattern = new RegExp(`^${this.capitalize(archetype)}-\\d+$`);
    return defaultPattern.test(requestedName)
      ? `${this.capitalize(archetype)}-${sequence}`
      : `${requestedName}-${sequence}`;
  }

  private createCorrelationId(): string {
    const sequence = ++this.delegationSequence;
    const timestamp = new Date().toISOString().replace(/\D/g, '').slice(0, 17);
    const session = (this.ctx?.sessionId ?? 'bootstrap')
      .replace(/[^a-zA-Z0-9_-]/g, '_')
      .slice(0, 32);
    return `del_${timestamp}_${session}_${String(sequence).padStart(3, '0')}_${randomUUID().slice(0, 8)}`;
  }

  private createRootToMProfile(): ToMProfile {
    return normalizeToMProfile({
      level: 1,
      subjectAgentId: 'root',
      beliefScope: ['user intent', 'runtime state', 'available agent capabilities', 'budget and policy constraints'],
      goalModel: ['Provide a reliable final response and delegate only when another perspective closes a cognitive gap.'],
      uncertainty: ['Which evidence or specialist perspective is missing for the current task?'],
      perspective: 'root coordinator and user-intent modeler',
      observesAgents: [],
      modelsAgents: [],
      capabilityScope: ['task assessment', 'delegation', 'team synthesis', 'memory coordination'],
      cognitiveGaps: [],
      models: [
        {
          targetId: 'user',
          targetType: 'user',
          goalModel: ['develop Roy into a Theory-of-Mind based multi-agent runtime'],
          intentModel: ['validate controlled subagent spawning and message-mediated execution'],
        },
      ],
      purpose: 'Understand user intent and decide how to answer or delegate.',
    }, {
      level: 1,
      subjectAgentId: 'root',
      purpose: 'Understand user intent and decide how to answer or delegate.',
    });
  }

  private createSubagentToMProfile(archetype: SubAgentArchetype, subjectAgentId: string, task: string, parentId = 'root'): ToMProfile {
    const analysis = this.tomPlanner.analyzeTask({
      task,
      parentId,
      parentProfile: this.ctx?.manager.getAgentById(parentId)?.getIdentity().tomProfile,
    });
    const plan = this.tomPlanner.completePlans(analysis, [{ archetype, task }], 1)[0];
    const profile = normalizeToMProfile(plan?.tomProfile, {
      level: this.defaultToMLevel(archetype),
      subjectAgentId: subjectAgentId || archetype,
      purpose: this.defaultToMPurpose(archetype, task),
    });
    profile.subjectAgentId = subjectAgentId || archetype;
    return profile;
  }

  private defaultToMLevel(archetype: SubAgentArchetype): ToMProfile['level'] {
    switch (archetype) {
      case 'critic':
        return 2;
      case 'planner':
        return 1;
      case 'summarizer':
        return 0;
      case 'researcher':
      case 'coder':
      case 'tester':
      case 'custom':
      default:
        return 0;
    }
  }

  private defaultToMPurpose(archetype: SubAgentArchetype, task: string): string {
    switch (archetype) {
      case 'researcher':
        return 'Collect grounded facts from the project context.';
      case 'critic':
        return "Evaluate another agent or design result against Roy's goal and user intent.";
      case 'planner':
        return 'Turn context into a sequence of actionable steps.';
      case 'coder':
        return 'Implement scoped code changes.';
      case 'summarizer':
        return 'Condense results into a clear summary.';
      case 'tester':
        return 'Validate behavior and identify regressions.';
      case 'custom':
      default:
        return `Complete the assigned task: ${task}`;
    }
  }

  private isValidArchetype(value: string): value is SubAgentArchetype {
    return ['researcher', 'critic', 'planner', 'coder', 'summarizer', 'tester', 'custom'].includes(value);
  }

  private handleQueueTransition(transition: QueueTransition): void {
    const message = transition.message;
    const communicationTrace = this.communicationManager?.recordTransition(transition);
    if (communicationTrace && this.ctx) {
      for (const actor of [communicationTrace.from, ...communicationTrace.to]) {
        const info = this.ctx.manager.getAgentById(actor.id)?.getInfo();
        if (!info) continue;
        actor.name = info.identity.name;
        actor.parentId = info.identity.parentId;
        actor.teamId = info.identity.teamId;
        actor.type = info.identity.role === 'subteam' ? 'team' : 'agent';
      }
      const participantIds = new Set([communicationTrace.from.id, ...communicationTrace.to.map(actor => actor.id)]);
      for (const info of this.ctx.manager.listAgentInfo()) {
        const agent = this.ctx.manager.getAgentById(info.identity.id);
        if (agent && participantIds.has(agent.id)) agent.receiveSystemTrace(communicationTrace);
      }
    }
    this.emit({
      type: transition.type,
      agentId: message.metadata?.agentId,
      sessionId: message.sessionId,
      correlationId: message.correlationId,
      nodeId: message.metadata?.nodeId,
      data: {
        messageId: message.id,
        kind: message.kind,
        from: message.from,
        to: message.to,
        status: message.status,
        sessionId: message.sessionId,
        turnId: message.turnId,
        traceId: message.traceId,
        correlationId: message.correlationId,
        parentMessageId: message.parentMessageId,
        error: transition.error,
        reason: transition.reason,
        communicationProtocol: message.metadata?.communicationProtocol,
        communicationTraceId: communicationTrace?.id,
      },
    });
  }

  private capitalize(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  private async drainAgentOutput(
    queue: { isEmpty(recipient: string): boolean; receive(recipient: string): Promise<{ sender: string; content: unknown } | undefined> },
    sender: string
  ): Promise<string> {
    const chunks: string[] = [];
    while (!queue.isEmpty('env')) {
      const message = await queue.receive('env');
      if (!message) break;
      if (message.sender === sender) {
        chunks.push(String(message.content));
      }
    }
    return chunks.join('');
  }
}

function decodeBasicXmlEntities(value: string): string {
  return value
    .replace(/&quot;/gi, '"')
    .replace(/&apos;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&amp;/gi, '&');
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function finiteRatio(value: unknown): number | undefined {
  const number = finiteNumber(value);
  return number === undefined ? undefined : Math.max(0, Math.min(1, number));
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const defined = values.filter((value): value is number => value !== undefined);
  return defined.length > 0 ? defined.reduce((sum, value) => sum + value, 0) / defined.length : undefined;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value)) return minimum;
  return Math.max(minimum, Math.min(maximum, Math.floor(value)));
}

function teamCompletionRatio(team: TeamRuntimeState): number {
  if (team.memberAgentIds.length === 0) return 0;
  const completed = Object.values(team.memberStatuses).filter(status => status === 'completed').length;
  return completed / team.memberAgentIds.length;
}

export const runtime = Runtime.getInstance();
export default Runtime;
