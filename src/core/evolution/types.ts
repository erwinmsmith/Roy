export type EvolutionFSMState =
  | 'S_evo_idle'
  | 'S_evo_propose'
  | 'S_evo_instantiate'
  | 'S_evo_execute'
  | 'S_evo_evaluate'
  | 'S_evo_select'
  | 'S_evo_mutate'
  | 'S_evo_integrate'
  | 'S_evo_done'
  | 'S_evo_failed';

export type EvolutionProfile =
  | 'solo'
  | 'fixed_subagents'
  | 'tom_subteam'
  | 'budget_market'
  | 'evo_team';

export interface EvolutionAblations {
  withoutSubagents: boolean;
  withoutToMProfile: boolean;
  withoutBudgetMarket: boolean;
  withoutEvoMutation: boolean;
  withoutPatternMemory: boolean;
}

export interface EvolutionRunOptions {
  profile: EvolutionProfile;
  populationSize: number;
  generations: number;
  topK: number;
  maxExecutedCandidates: number;
  integrationMinimumScore: number;
  patternSimilarityThreshold: number;
  useLlmJudge: boolean;
  ablations: EvolutionAblations;
}

export type GenomeMemoryPolicy = 'none' | 'local' | 'shared' | 'episodic';
export type GenomeCoordinationPolicy =
  | 'parallel'
  | 'sequential'
  | 'debate'
  | 'critic_refine'
  | 'leader_worker';

export interface GenomeToolPolicy {
  name: string;
  permission: 'read_only' | 'write' | 'execute';
  required?: boolean;
}

export interface GenomeToMProfile {
  level: 0 | 1 | 2 | 3;
  beliefScope: string[];
  goalModel: string[];
  uncertainty: string[];
  perspective?: string;
  observesAgents?: string[];
  modelsAgents?: string[];
}

export interface AgentGenome {
  id: string;
  archetype: 'researcher' | 'critic' | 'planner' | 'coder' | 'summarizer' | 'tester' | 'custom';
  name?: string;
  role: string;
  task: string;
  rolePrompt: string;
  taskDecompositionStyle: string;
  toolPolicy: GenomeToolPolicy[];
  skills: string[];
  memoryPolicy: GenomeMemoryPolicy;
  tomProfile: GenomeToMProfile;
  budgetPolicy: {
    requestedTokens: number;
    maxTokens?: number;
  };
  evaluationCriteria: string[];
  outputContract: {
    format: 'markdown' | 'json' | 'structured_report';
    groundingRequired: boolean;
    requiredFields: string[];
  };
}

export interface TeamGenome {
  id: string;
  name: string;
  taskSignature: string;
  purpose: string;
  tomLevel: 0 | 1 | 2 | 3;
  members: AgentGenome[];
  coordinationPolicy: GenomeCoordinationPolicy;
  synthesisPolicy: string;
  budgetPolicy: {
    requestedTokens: number;
    maxTokens?: number;
  };
  evaluationCriteria: string[];
}

export type EvolutionCandidateSource =
  | 'default'
  | 'cache_hit'
  | 'mutated_from_cache'
  | 'generated'
  | 'custom_generated';

export interface EvolutionLineage {
  parentCandidateId?: string;
  parentPatternIds: string[];
  operators: string[];
  generation: number;
}

export interface EvolutionCandidate {
  id: string;
  source: EvolutionCandidateSource;
  genome: TeamGenome;
  rationale: string;
  expectedCostTokens: number;
  expectedUtility: number;
  lineage: EvolutionLineage;
}

export interface EvolutionTokenUsage {
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number | null;
  totalTokens: number;
  cachedInputTokens?: number | null;
  cacheCreationInputTokens?: number | null;
}

export interface EvolutionExecutionArtifact {
  candidateId: string;
  actorKind: 'agent' | 'team';
  actorId: string;
  success: boolean;
  result: string;
  usage: EvolutionTokenUsage;
  wallClockMs: number;
  agentIds: string[];
  teamIds: string[];
  toolCalls: number;
  successfulToolCalls: number;
  unresolvedToolIntents: number;
  groundedResults: number;
  totalResults: number;
  failedActors: number;
  recoveredFailures: number;
  warnings: string[];
}

export interface EvolutionEvaluationDimensions {
  taskSuccess: number;
  answerQuality: number;
  completeness: number;
  costEfficiency: number;
  novelty: number;
  toolUse: number;
  consistency: number;
  tomCoverage: number;
}

export interface EvolutionEvaluationResult {
  candidateId: string;
  score: number;
  dimensions: EvolutionEvaluationDimensions;
  tokenUsed: number;
  success: boolean;
  rationale: string;
  evaluator: string;
}

export interface EvolutionMetrics {
  taskSuccess: boolean;
  answerQuality: number;
  toolSuccessRate: number;
  agentsSpawned: number;
  teamsSpawned: number;
  totalTokens: number;
  thinkingTokens: number | null;
  wallClockMs: number;
  budgetRequested: number;
  budgetAllocated: number;
  failureRecoveryCount: number;
  candidateCount: number;
  executedCandidateCount: number;
  selectedGenomeId?: string;
  selectedGenomeScore?: number;
  cacheHits: number;
  mutationsApplied: number;
}

export interface EvolutionPattern {
  id: string;
  name: string;
  taskSignature: string;
  genome: TeamGenome;
  historicalScores: EvolutionEvaluationResult[];
  usageCount: number;
  successCount: number;
  averageScore: number;
  averageTokenCost: number;
  status: 'candidate' | 'active' | 'deprecated';
  lineage: EvolutionLineage;
  linkedPatterns: {
    agentPatternIds: string[];
    teamPatternIds: string[];
  };
  createdAt: string;
  updatedAt: string;
}

export interface EvolutionRunResult {
  id: string;
  correlationId: string;
  task: string;
  profile: EvolutionProfile;
  state: EvolutionFSMState;
  candidates: EvolutionCandidate[];
  executions: EvolutionExecutionArtifact[];
  evaluations: EvolutionEvaluationResult[];
  selected: EvolutionCandidate | undefined;
  selectedExecution: EvolutionExecutionArtifact | undefined;
  selectedEvaluation: EvolutionEvaluationResult | undefined;
  integratedPatternId?: string;
  metrics: EvolutionMetrics;
  ablations: EvolutionAblations;
  startedAt: number;
  completedAt: number;
  error?: string;
}

export interface EvolutionSeedAgent {
  archetype: AgentGenome['archetype'];
  name?: string;
  role?: string;
  task: string;
  tools?: string[];
  skills?: string[];
  budgetTokens?: number;
  tomLevel?: number;
  perspective?: string;
  groundingRequired?: boolean;
}

export interface EvolutionProposalInput {
  runId: string;
  task: string;
  parentId: string;
  agents: EvolutionSeedAgent[];
  patterns: EvolutionPattern[];
  availableTokens?: number;
  availableAgentSlots?: number;
  options: EvolutionRunOptions;
}

export interface EvolutionJudge {
  readonly name: string;
  evaluate(
    task: string,
    candidate: EvolutionCandidate,
    execution: EvolutionExecutionArtifact
  ): Promise<Partial<EvolutionEvaluationDimensions> & { rationale?: string }>;
}

export interface EvolutionCandidateEvaluator {
  readonly name: string;
  evaluate(
    task: string,
    candidate: EvolutionCandidate,
    execution: EvolutionExecutionArtifact
  ): Promise<EvolutionEvaluationResult>;
}

export interface EvolutionSelectionPolicy {
  readonly name: string;
  select(
    candidates: EvolutionCandidate[],
    evaluations: EvolutionEvaluationResult[],
    topK: number
  ): EvolutionCandidate[];
}

export interface GenomeMutationContext {
  runId: string;
  generation: number;
  task: string;
  availableTokens?: number;
}

export interface GenomeMutationOperator {
  readonly name: string;
  supports(genome: TeamGenome, context: GenomeMutationContext): boolean;
  mutate(genome: TeamGenome, context: GenomeMutationContext): TeamGenome;
}

export interface EvolutionLifecycleHooks {
  onTransition?(from: EvolutionFSMState, to: EvolutionFSMState, data?: Record<string, unknown>): void | Promise<void>;
  onCandidateRejected?(candidate: EvolutionCandidate, reason: string): void | Promise<void>;
  instantiate(candidate: EvolutionCandidate): Promise<void>;
  execute(candidate: EvolutionCandidate): Promise<EvolutionExecutionArtifact>;
  integrate(
    selected: EvolutionCandidate,
    evaluation: EvolutionEvaluationResult,
    execution: EvolutionExecutionArtifact
  ): Promise<string | undefined>;
}
