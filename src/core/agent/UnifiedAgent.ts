// UnifiedAgent - Full-featured agent that integrates FSM, Actions, Tools, Prompts, Memory

import type { LLMMessage } from '../llm/types.js';
import { BaseAgent, type AgentConfig } from './BaseAgent.js';
import type { Plan, Planner } from '../actions/Planner.js';
import { actionRegistry } from '../actions/index.js';
import { toolRegistry } from '../tools/index.js';
import { skillRegistry } from '../skills/index.js';
import { contextManager } from '../memory/context.js';
import { buildPrompt } from '../prompts/builder.js';
import {
  conversationalTemplate,
  fsmDiagnoseTemplate,
  fsmDecideTemplate,
  fsmDeriveTemplate,
  fsmVerifyTemplate,
  actionTemplate,
} from '../prompts/index.js';
import { logger } from '../utils/logger.js';
import type { PlannedToolCall } from '../tools/planner.js';
import type { ToolLoopCallRecord } from '../tools/executionLoop.js';
import {
  isSuccessfulWorkspaceMutationCall as isSuccessfulWorkspaceMutation,
  isSuccessfulWorkspaceVerificationCall as isSuccessfulWorkspaceVerification,
  isWorkspaceVerificationCall,
  taskRequestsWorkspaceMutation as requestsWorkspaceMutation,
  workspaceToolIntentFingerprint,
} from '../tools/executionIntent.js';

export type AgentMode = 'conversational' | 'action' | 'hybrid';

export interface UnifiedAgentConfig extends AgentConfig {
  mode?: AgentMode;
  planner?: Planner;
  useContextManager?: boolean;
  allowedActions?: string[];
  allowedTools?: string[];
  allowedSkills?: string[];
}

export interface AgentToolRoundPlanningInput {
  task: string;
  executionRequired?: boolean;
  requiredMutationAfterCallIndex?: number;
  round: number;
  remainingCalls: number;
  requestTimeoutMs?: number;
  tools: Array<{ name: string; description?: string; parameters?: Record<string, unknown> }>;
  calls: ToolLoopCallRecord[];
}

export class UnifiedAgent extends BaseAgent {
  private mode: AgentMode;
  private planner?: Planner;
  private useContextManager: boolean;
  private sessionId: string = '';
  private allowedActions?: Set<string>;
  private allowedTools?: Set<string>;
  private allowedSkills?: Set<string>;
  private lastToolPlanningFailure?: {
    message: string;
    timedOut: boolean;
    occurredAt: number;
  };

  constructor(config: UnifiedAgentConfig) {
    super({
      id: config.id,
      name: config.name,
      goal: config.goal,
      example: config.example,
      llm: config.llm,
      fsm: config.fsm,
      role: config.role,
      parentId: config.parentId,
      teamId: config.teamId,
      generation: config.generation,
      tomLevel: config.tomLevel,
      description: config.description,
      tomProfile: config.tomProfile,
      communicationProtocol: config.communicationProtocol,
    });
    this.mode = config.mode ?? 'hybrid';
    this.planner = config.planner;
    this.useContextManager = config.useContextManager ?? true;
    this.allowedActions = config.allowedActions ? new Set(config.allowedActions) : undefined;
    this.allowedTools = config.allowedTools ? new Set(config.allowedTools) : undefined;
    this.allowedSkills = config.allowedSkills ? new Set(config.allowedSkills) : undefined;
  }

  /**
   * Plan another observe/act round after runtime tools have returned evidence.
   * Runtime remains responsible for authorization and execution.
   */
  async planNextToolRound(input: AgentToolRoundPlanningInput): Promise<PlannedToolCall[]> {
    this.lastToolPlanningFailure = undefined;
    if (!this.llm || input.remainingCalls <= 0 || input.tools.length === 0) return [];
    const authorized = new Set(input.tools
      .filter(tool => !this.allowedTools || this.allowedTools.has(tool.name))
      .map(tool => tool.name));
    if (authorized.size === 0) return [];
    const executionRequired = (input.executionRequired ?? requestsWorkspaceMutation(input.task))
      && (authorized.has('fs.write') || authorized.has('fs.replace') || authorized.has('shell.exec'));
    const mutationApplied = input.calls.some(call => isSuccessfulWorkspaceMutation(call));
    const freshMutationRequired = input.requiredMutationAfterCallIndex !== undefined;
    const freshMutationApplied = !freshMutationRequired || input.calls
      .slice(Math.max(0, Number(input.requiredMutationAfterCallIndex) + 1))
      .some(call => isSuccessfulWorkspaceMutation(call));
    const mutationRequirementSatisfied = mutationApplied && freshMutationApplied;
    const lastMutationIndex = findLastToolCallIndex(input.calls, call =>
      isSuccessfulWorkspaceMutation(call)
    );
    const lastVerificationIndex = findLastToolCallIndex(input.calls, call =>
      isWorkspaceVerificationCall(call)
    );
    const verificationAttempted = lastVerificationIndex >= 0;
    const verificationPassed = freshMutationApplied
      && lastVerificationIndex > lastMutationIndex
      && isSuccessfulWorkspaceVerification(input.calls[lastVerificationIndex]!);
    const latestVerificationFailed = lastVerificationIndex > lastMutationIndex
      && !isSuccessfulWorkspaceVerification(input.calls[lastVerificationIndex]!);
    const inspectedAfterLatestFailure = latestVerificationFailed
      && input.calls.slice(lastVerificationIndex + 1).some(call =>
        call.success && (
          call.toolName === 'fs.read'
          || call.toolName === 'fs.search'
        )
      );
    const successfulInspection = input.calls.some(call =>
      call.success && (
        call.toolName === 'fs.list'
        || call.toolName === 'fs.read'
        || call.toolName === 'fs.search'
      )
    );
    const recentCalls = input.calls.slice(-8);
    const detailedObservationIndices = new Set<number>([recentCalls.length - 1]);
    for (let index = recentCalls.length - 1; index >= 0; index -= 1) {
      const call = recentCalls[index]!;
      if (isWorkspaceVerificationCall(call)
        && !isSuccessfulWorkspaceVerification(call)) {
        detailedObservationIndices.add(index);
        break;
      }
    }
    for (let index = recentCalls.length - 1; index >= 0; index -= 1) {
      if (isSuccessfulWorkspaceMutation(recentCalls[index]!)) {
        detailedObservationIndices.add(index);
        break;
      }
    }
    const observations = recentCalls.map((call, index) => {
      const detailed = detailedObservationIndices.has(index);
      return {
        toolName: call.toolName,
        params: compactToolPlanningParams(call.params, detailed),
        success: call.success,
        error: call.error
          ? compactTail(String(call.error), detailed ? 3_000 : 400)
          : undefined,
        result: compactToolObservation(call.result, call.toolName, detailed),
      };
    });
    const messages: LLMMessage[] = [
      {
        role: 'system',
        content: [
          `You plan authorized tool calls for ${this.name}.`,
          'Continue only when another tool call is necessary to answer the task with concrete evidence.',
          'The listed tools are bound to this actor. Request the call directly and let Runtime enforce approval policy; never ask the user conversationally for tool permission.',
          executionRequired
            ? 'This is an execution task. Do not finish after analysis or a proposed patch. Apply the workspace change, then run relevant verification.'
            : '',
          executionRequired && !mutationRequirementSatisfied
            ? successfulInspection
              ? freshMutationRequired
                ? 'A prior acceptance audit found unmet requirements. Existing mutations and passing commands do not close this repair phase. Request a new focused fs.replace, fs.write, or mutating shell.exec call that fixes the reported unmet item.'
                : 'The workspace layout has been grounded. Request fs.replace, fs.write, or a mutating shell.exec call that advances the actual task.'
              : 'No authoritative workspace inspection has succeeded yet. Recover failed paths with fs.list, fs.read, or fs.search before requesting a mutation.'
            : '',
          executionRequired && latestVerificationFailed
            ? inspectedAfterLatestFailure
              ? 'The newest verification failed after the latest mutation, and relevant source evidence has already been inspected. Apply a focused repair now; do not rerun verification or broaden inspection before changing the workspace.'
              : 'The newest verification failed after the latest mutation. Preserve its detailed causal frontier and inspect only the reported source location before applying a focused repair.'
            : '',
          executionRequired && mutationApplied && !verificationPassed && !latestVerificationFailed
            ? 'At least one workspace mutation succeeded, but no verification has passed. Continue any remaining edits or repairs, then request a relevant test, build, lint, typecheck, or targeted assertion.'
            : '',
          'When verification fails, use its output to repair the workspace before retrying. Never hide a failing exit status with `|| true`, `; true`, `|| :`, or equivalent shell constructs.',
          'Before creating or replacing a module, use observed package metadata and directory layout to identify the authoritative source root. Never create a parallel top-level package when the project installs from src/, lib/, packages/, or another configured source directory.',
          'Keep mutation payloads bounded: prefer fs.replace for focused edits. Limit one fs.write/fs.replace payload to 6000 characters. For a larger full-file rewrite, write one bounded chunk with fs.write mode=overwrite, then append later chunks with mode=append in subsequent tool rounds before verification.',
          'When malformed quoting or embedded newlines make exact oldText fragile, inspect the reported lines and call fs.replace with path, startLine, endLine, and newText instead of guessing escaped oldText repeatedly.',
          'Use fs.write or fs.replace for file content. Do not embed multiline source, Markdown backticks, or shell-sensitive content inside python -c, echo, sed, or similar shell.exec writers; shell substitution can execute unintended commands and corrupt the edit.',
          'Never repeat an equivalent call. Search snippets are discovery evidence; fetch relevant result pages before making source-backed claims.',
          'Reject search results that do not match the core entities and topic in the task. Never fetch an irrelevant result merely because it is available.',
          'After two web.search calls, do not keep reformulating the same search. Fetch a likely official public URL if one can be identified, or finish with an explicit evidence limitation.',
          'Return JSON only: {"action":"call_tools"|"finish","reason":"...","calls":[{"toolName":"...","params":{...}}]}.',
        ].join('\n'),
      },
      {
        role: 'user',
        content: [
          `Task:\n${compactToolPlanningTask(input.task)}`,
          `Completed tool round: ${input.round}`,
          `Remaining call capacity: ${input.remainingCalls}`,
          `Execution state: ${JSON.stringify({
            executionRequired,
            successfulInspection,
            mutationApplied,
            freshMutationRequired,
            freshMutationApplied,
            verificationAttempted,
            verificationPassed,
            latestVerificationFailed,
            inspectedAfterLatestFailure,
          })}`,
          `Authorized tools:\n${JSON.stringify(input.tools, null, 2)}`,
          `Tool observations:\n${JSON.stringify(observations, null, 2)}`,
        ].join('\n\n'),
      },
    ];
    try {
      type PlanningResponse = {
        action?: string;
        reason?: string;
        calls?: Array<{ toolName?: unknown; params?: unknown }>;
      };
      let planningMessages = messages;
      let response: PlanningResponse = {};
      let plannedCalls: PlannedToolCall[] = [];
      const planningDeadline = input.requestTimeoutMs === undefined
        ? undefined
        : Date.now() + Math.max(1, input.requestTimeoutMs);
      const maxPlanningAttempts = executionRequired ? 3 : 1;
      for (let attempt = 0; attempt < maxPlanningAttempts; attempt += 1) {
        let rejectedDestructiveRepairOverwrite = false;
        const remainingPlanningMs = planningDeadline === undefined
          ? undefined
          : Math.max(0, planningDeadline - Date.now());
        if (remainingPlanningMs === 0) break;
        try {
          response = await this.completeJSONWithAccounting<PlanningResponse>(
            planningMessages,
            {
              temperature: 0,
              maxTokens: executionRequired
                ? attempt === 0 ? 4096 : 8192
                : 640,
              timeoutMs: remainingPlanningMs,
            }
          );
        } catch (error) {
          const recoveredMutation = recoverTruncatedFsWriteResponse(
            error,
            authorized,
            input.calls
          );
          if (recoveredMutation) {
            response = recoveredMutation;
          } else {
            const canRetry = executionRequired
              && attempt + 1 < maxPlanningAttempts
              && isRetryableToolPlanningResponseError(error)
              && (planningDeadline === undefined || Date.now() < planningDeadline);
            if (!canRetry) throw error;
            planningMessages = [
              ...planningMessages,
              {
                role: 'user',
                content: [
                  'The previous tool plan was incomplete or was not valid JSON.',
                  'Return one complete compact JSON object only, with no analysis or markdown.',
                  'Request at most one mutation call in this response.',
                  'Keep fs.write content or fs.replace replacement text within 6000 characters.',
                  'For a larger rewrite, emit only the next bounded chunk and continue with fs.write mode=append in a later tool round.',
                ].join('\n'),
              },
            ];
            continue;
          }
        }
        plannedCalls = normalizePlannedToolCalls(
          response,
          authorized,
          input.remainingCalls,
          input.round
        ).filter(call => !shouldSuppressRepeatedPlannedCall(call, input.calls));
        if (latestVerificationFailed) {
          if (inspectedAfterLatestFailure) {
            rejectedDestructiveRepairOverwrite = plannedCalls.some(call =>
              isDestructiveRepairOverwrite(call, input.calls)
            );
            plannedCalls = plannedCalls.filter(call =>
              isSuccessfulWorkspaceMutation({ ...call, success: true })
              && !isDestructiveRepairOverwrite(call, input.calls)
            );
          } else {
            const directRepair = plannedCalls.find(call =>
              isSuccessfulWorkspaceMutation({ ...call, success: true })
            );
            const targetedInspection = plannedCalls.find(call =>
              call.toolName === 'fs.read' || call.toolName === 'fs.search'
            );
            plannedCalls = directRepair
              ? [directRepair]
              : targetedInspection
                ? [targetedInspection]
                : [];
          }
        }
        const plannedInspection = plannedCalls.some(call =>
          call.toolName === 'fs.list'
          || call.toolName === 'fs.read'
          || call.toolName === 'fs.search'
        );
        const advancesExecution = !executionRequired
          || (latestVerificationFailed
            ? inspectedAfterLatestFailure
              ? plannedCalls.some(call =>
                isSuccessfulWorkspaceMutation({ ...call, success: true })
              )
              : plannedInspection || plannedCalls.some(call =>
                isSuccessfulWorkspaceMutation({ ...call, success: true })
              )
          : mutationRequirementSatisfied
              ? verificationPassed || plannedCalls.some(call =>
                isSuccessfulWorkspaceMutation({ ...call, success: true })
                || isSuccessfulWorkspaceVerification({ ...call, success: true })
              )
              : ((!successfulInspection && plannedInspection)
                || plannedCalls.some(call => isSuccessfulWorkspaceMutation({
                  ...call,
                  success: true,
                }))));
        if (advancesExecution) break;
        planningMessages = [
          ...planningMessages,
          { role: 'assistant', content: JSON.stringify(response) },
          {
            role: 'user',
            content: [
              'Runtime rejected this plan because the execution contract is incomplete.',
              !mutationRequirementSatisfied
                ? successfulInspection
                  ? freshMutationRequired
                    ? 'The previous acceptance audit is still open. Request a new focused mutation that addresses one of its failed or unverified items now.'
                    : 'The workspace is already grounded. Request a concrete fs.replace, fs.write, or mutating shell.exec call now.'
                  : 'The previous inspection failed or was absent. Request a corrected fs.list, fs.read, or fs.search call before mutating.'
                : '',
              latestVerificationFailed && inspectedAfterLatestFailure
                ? rejectedDestructiveRepairOverwrite
                  ? 'The latest verifier failure concerns an existing file. Preserve working code: use fs.replace for a focused repair instead of overwriting that file, then verify.'
                  : 'The latest verifier failure and its relevant source evidence are already grounded. Request a concrete fs.replace, fs.write for a new file, or mutating shell.exec repair before any further verification.'
                : '',
              mutationRequirementSatisfied && !verificationPassed && !latestVerificationFailed
                ? 'Finish, read-only, masked-failure, and repeated plans are insufficient. Request a concrete remaining edit or repair, or a distinct verification command whose exit status is preserved.'
                : '',
              'Return the required call_tools JSON. Do not ask for permission.',
            ].filter(Boolean).join('\n'),
          },
        ];
      }
      if (executionRequired) {
        const plannedInspection = plannedCalls.some(call =>
          call.toolName === 'fs.list'
          || call.toolName === 'fs.read'
          || call.toolName === 'fs.search'
        );
        const advancesExecution = latestVerificationFailed
          ? inspectedAfterLatestFailure
            ? plannedCalls.some(call =>
              isSuccessfulWorkspaceMutation({ ...call, success: true })
            )
            : plannedInspection || plannedCalls.some(call =>
              isSuccessfulWorkspaceMutation({ ...call, success: true })
            )
          : mutationRequirementSatisfied
            ? verificationPassed || plannedCalls.some(call =>
              isSuccessfulWorkspaceMutation({ ...call, success: true })
              || isSuccessfulWorkspaceVerification({ ...call, success: true })
            )
            : ((!successfulInspection && plannedInspection)
              || plannedCalls.some(call => isSuccessfulWorkspaceMutation({
                ...call,
                success: true,
              })));
        if (!advancesExecution) return [];
      }
      return plannedCalls;
    } catch (error) {
      logger.warn(`Agent ${this.name} could not plan another tool round:`, error);
      const message = error instanceof Error ? error.message : String(error);
      this.lastToolPlanningFailure = {
        message: message.slice(0, 2_000),
        timedOut: /\b(?:timed?\s*out|timeout|deadline|aborted?)\b/i.test(message),
        occurredAt: Date.now(),
      };
      if (authorized.has('web.fetch')) {
        const urls = extractPlannerFallbackUrls(message);
        if (urls.length > 0) {
          return urls.slice(0, input.remainingCalls).map(url => ({
            toolName: 'web.fetch',
            params: { url },
            reason: 'Recovered a concrete public URL from a non-JSON tool-planning response.',
            groundingRequired: true,
          }));
        }
      }
      return [];
    }
  }

  getLastToolPlanningFailure(): {
    message: string;
    timedOut: boolean;
    occurredAt: number;
  } | undefined {
    return this.lastToolPlanningFailure
      ? { ...this.lastToolPlanningFailure }
      : undefined;
  }

  /**
   * Main step - orchestrates FSM, prompts, planning, and execution
   */
  async step(observation: string): Promise<void> {
    if (this.fsm?.isTerminal()) {
      this.fsm.reset();
    }

    this.state = 'thinking';
    this.addToMemory('observation', observation);

    if (!this.llm) {
      const errorMsg = 'Error: LLM not configured';
      logger.warn(`Agent ${this.name} has no LLM configured`);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
      this.state = 'idle';
      return;
    }

    // === FSM Integration ===
    await this.updateFSMContext(observation);

    // Pre-step transition
    await this.maybeTransition();

    // === FSM State-Based Prompt Selection ===
    const systemPrompt = this.buildFSMPrompt(observation);
    const messages = this.buildMessages(systemPrompt, observation);
    const tokensBefore = this.usage.totalTokens;

    // === Determine Action Mode ===
    const shouldAct = await this.shouldExecuteAction(observation);

    if (shouldAct && this.mode !== 'conversational') {
      await this.executeActionMode(observation, messages);
    } else {
      await this.executeConversationalMode(messages);
    }

    // Track cost
    if (this.fsm) {
      this.fsm.addCost(this.usage.totalTokens - tokensBefore);
      this.fsm.addToTrace(`[${this.fsm.getStateName()}] Output complete`);
      await this.maybeTransition();
    }

    logger.info(`Turn completed for agent ${this.name}`);
    this.state = 'idle';
  }

  /**
   * Update FSM context based on observation
   */
  private async updateFSMContext(observation: string): Promise<void> {
    if (!this.fsm) return;

    this.fsm.setUncertainty(this.calculateUncertainty(observation));
    this.fsm.setConflict(this.analyzeConflict(observation));
    this.fsm.setEvidence(this.analyzeEvidence(observation));

    if (this.useContextManager && this.sessionId) {
      const context = contextManager.get(this.name, this.sessionId);
      if (context) {
        const lines = context.content.split('\n').length;
        this.fsm.setUncertainty(Math.min(1, this.fsm.getContext().uncertainty + lines / 100));
      }
    }
  }

  /**
   * Build prompt based on FSM state
   */
  private buildFSMPrompt(currentObservation: string): string {
    const baseGoal = this.goal || 'You are a helpful assistant.';
    const examplePart = this.example ? `\nExamples:\n${this.example}` : '';

    if (!this.fsm) {
      return buildPrompt(conversationalTemplate.template, {
        agent_identity: this.buildIdentityPrompt(),
        agent_goal: baseGoal,
        agent_example: examplePart,
      });
    }

    const state = this.fsm.getState();
    const ctx = this.fsm.getContext();
    const remainingBudget = ctx.budget === null ? 'unlimited' : String(ctx.budget - ctx.cost);

    switch (state) {
      case 'S_solo':
        return buildPrompt(conversationalTemplate.template, {
          agent_identity: this.buildIdentityPrompt(),
          agent_goal: baseGoal,
          agent_example: examplePart,
        });

      case 'S_diagnose':
        return buildPrompt(fsmDiagnoseTemplate.template, {
          trace: ctx.trace.join('\n') || 'No trace yet',
          state,
        });

      case 'S_decide': {
        const candidates = this.formatCapabilitiesForPrompt();
        return buildPrompt(fsmDecideTemplate.template, {
          current_state: state,
          budget: remainingBudget,
          candidates: candidates || 'No candidate actions available',
          uncertainty: String(ctx.uncertainty),
          conflict: String(ctx.conflict),
        });
      }

      case 'S_derive':
        return buildPrompt(fsmDeriveTemplate.template, {
          parent_unit: this.name,
          trace: ctx.trace.join('\n'),
          budget: remainingBudget,
          state,
        });

      case 'S_execute': {
        return buildPrompt(actionTemplate.template, {
          agent_goal: baseGoal,
          agent_actions: this.formatCapabilitiesForPrompt(),
          agent_example: examplePart,
        });
      }

      case 'S_verify': {
        const history = this.getHistory();
        const lastMsg = history[history.length - 1];
        return buildPrompt(fsmVerifyTemplate.template, {
          question: currentObservation,
          answer: lastMsg?.content ?? '',
          trace: ctx.trace.join('\n'),
          state,
        });
      }

      case 'S_final':
        return buildPrompt(conversationalTemplate.template, {
          agent_identity: this.buildIdentityPrompt(),
          agent_goal: baseGoal,
          agent_example: 'Provide a clear, final answer summarizing all reasoning.',
        });

      case 'S_backtrack':
        return buildPrompt(fsmDeriveTemplate.template, {
          parent_unit: this.name,
          trace: ctx.trace.join('\n'),
          budget: remainingBudget,
          state: 'S_backtrack',
        });

      default:
        return buildPrompt(conversationalTemplate.template, {
          agent_identity: this.buildIdentityPrompt(),
          agent_goal: baseGoal,
          agent_example: examplePart,
        });
    }
  }

  /**
   * Build messages for LLM
   */
  private buildMessages(systemPrompt: string, observation: string): LLMMessage[] {
    const history = this.getHistory();
    const priorHistory = history.at(-1)?.role === 'user' && history.at(-1)?.content === observation
      ? history.slice(0, -1)
      : history;
    return [
      { role: 'system', content: systemPrompt },
      ...priorHistory,
      { role: 'user', content: observation },
    ];
  }

  /**
   * Decide whether to execute an action
   */
  private async shouldExecuteAction(observation: string): Promise<boolean> {
    if (observation.includes('[runtime_grounding_provided]') || /\nGrounding context:\n/.test(observation)) return false;
    const task = this.extractPrimaryTask(observation);
    const lowerObs = task.toLowerCase();
    if (/^(?:stress-test|critique|evaluate|assess|challenge|reconcile|synthesize|summarize)\b/.test(lowerObs)) {
      return false;
    }
    const hasActionIndicator = /\b(?:run|execute|perform|search|find|calculate|create|update|delete|fetch|inspect|list|read|status|check)\b/.test(lowerObs)
      || /\bget me\b/.test(lowerObs)
      || /\brun\s+(?:the\s+)?(?:tests?|build)\b/.test(lowerObs);
    const hasCapabilities = this.getCapabilityNames().length > 0;

    return (hasActionIndicator && hasCapabilities) || this.mode === 'action';
  }

  /**
   * Execute in action mode with planner
   */
  private async executeActionMode(
    observation: string,
    messages: LLMMessage[]
  ): Promise<void> {
    let plan: Plan | null = null;

    if (this.planner) {
      try {
        plan = await this.planner.plan({
          agentInfo: {
            name: this.name,
            goal: this.goal,
            actions: this.formatCapabilitiesForPrompt(),
          },
          observation,
          availableActions: this.getCapabilityNames(),
          history: messages.slice(0, -1),
        });
      } catch (error) {
        logger.error('Planner error:', error);
      }
    }

    if (!plan) {
      plan = await this.decideActionWithLLM(observation, messages);
    }

    if (!plan) {
      await this.executeConversationalMode(messages);
      return;
    }

    try {
      const result = await this.executeCapability(plan.action, plan.params);

      if (result.success) {
        const response = this.formatCapabilityResult(result.result);
        this.addToMemory('action', `${plan.action}: ${response}`);
        this.fsm?.addToTrace(`Action executed: ${plan.action}`);
        await this.executeConversationalMode(
          this.buildActionSynthesisMessages(observation, messages, plan.action, response)
        );
      } else {
        if (plan.action === 'use_tool_when_needed'
          && result.error?.includes('toolName is required when needed is not false')) {
          this.addToMemory(
            'result',
            'The optional tool-use action was incomplete, so no tool was executed and reasoning continued.'
          );
          this.fsm?.addToTrace('Skipped malformed optional tool-use action');
          await this.executeConversationalMode([
            ...messages,
            {
              role: 'system',
              content: 'The attempted optional tool action omitted a tool name. No tool was executed. Answer the task directly from the available context without emitting tool-call markup.',
            },
          ]);
          return;
        }
        const errorMsg = `Action error: ${result.error}`;
        this.addToMemory('result', errorMsg);
        if (this.messageQueue) {
          await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
        } else {
          process.stdout.write(errorMsg);
        }
      }
    } catch (error) {
      const errorMsg = `Error: ${error}`;
      logger.error(`Action execution error:`, error);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
    }
  }

  private formatCapabilityResult(result: unknown): string {
    if (result === undefined) return 'Action completed successfully';
    if (typeof result === 'string') return result;
    try {
      return JSON.stringify(result, null, 2);
    } catch {
      return String(result);
    }
  }

  private buildActionSynthesisMessages(
    observation: string,
    messages: LLMMessage[],
    action: string,
    response: string
  ): LLMMessage[] {
    const limit = this.getCompletionTokenLimit();
    const inputTokenBudget = limit === undefined ? 4000 : Math.max(64, Math.floor(limit * 0.62));
    const inputCharBudget = inputTokenBudget * 3;
    const systemSource = messages.find(message => message.role === 'system')?.content
      ?? `You are ${this.name}.`;
    const systemBudget = Math.min(1800, Math.max(160, Math.floor(inputCharBudget * 0.25)));
    const task = this.extractPrimaryTask(observation);
    const instruction = [
      `Original task:\n${task}`,
      `Authorized capability "${action}" completed.`,
      'Capability result:',
      'Use this evidence to answer the original task.',
      'Return a user-facing conclusion, not raw tool output or another tool request.',
    ].join('\n\n');
    const resultBudget = Math.max(128, inputCharBudget - systemBudget - instruction.length - 64);
    return [
      { role: 'system', content: systemSource.slice(0, systemBudget) },
      { role: 'user', content: `${instruction}\n\n${response.slice(0, resultBudget)}` },
    ];
  }

  /**
   * Decide action using LLM
   */
  private async decideActionWithLLM(
    observation: string,
    _messages: LLMMessage[]
  ): Promise<Plan | null> {
    const task = this.extractPrimaryTask(observation);
    const prompt = `Based on the observation, choose the best action to execute.

Available capabilities:
${this.formatCapabilitiesForPrompt()}

Observation: ${task}

Return a JSON object with:
- action: the name of the action, tool, or skill to execute (or "none" if no action needed)
- params: parameters for the action (or empty object)
- reasoning: why you chose this action

Tool-use policy:
- If the task needs external evidence, filesystem inspection, command output, or validation, use a registered tool directly or the "use_tool_when_needed" skill.
- If no tool is needed, return {"action":"none","params":{}}
- Do not call tools just to appear active.`;

    const decisionMessages: LLMMessage[] = [
      {
        role: 'system',
        content: `You are the capability selector for ${this.name}. Choose only from the explicitly authorized capabilities below.`,
      },
      { role: 'user', content: prompt },
    ];

    try {
      const response = await this.completeJSONWithAccounting<{
        action: string;
        params: Record<string, unknown>;
        reasoning?: string;
      }>(decisionMessages, { temperature: 0, maxTokens: 512 });

      if (response.action === 'none' || !response.action) {
        return null;
      }

      return {
        action: response.action,
        params: response.params || {},
        reasoning: response.reasoning,
        confidence: 0.8,
      };
    } catch (error) {
      logger.error('LLM action decision error:', error);
      return null;
    }
  }

  private extractPrimaryTask(observation: string): string {
    return observation
      .split(/\n(?:Grounding context:|Grounding warnings:)|\n\n<system_communication_context/)[0]
      .trim();
  }

  /**
   * Execute an action, tool, or skill by name.
   */
  private async executeCapability(
    name: string,
    params: Record<string, unknown>
  ): Promise<{ success: boolean; result?: unknown; error?: string; metadata?: Record<string, unknown> }> {
    if (actionRegistry.has(name)) {
      if (this.allowedActions && !this.allowedActions.has(name)) {
        return { success: false, error: `Action "${name}" is not authorized for agent "${this.id}"` };
      }
      return actionRegistry.execute(name, params);
    }

    if (toolRegistry.has(name)) {
      if (this.allowedTools && !this.allowedTools.has(name)) {
        return { success: false, error: `Tool "${name}" is not authorized for agent "${this.id}"` };
      }
      return toolRegistry.execute(name, params);
    }

    if (skillRegistry.has(name)) {
      if (this.allowedSkills && !this.allowedSkills.has(name)) {
        return { success: false, error: `Skill "${name}" is not authorized for agent "${this.id}"` };
      }
      const validation = skillRegistry.get(name)?.validate?.({ action: name, params });
      if (validation && !validation.valid) {
        return {
          success: false,
          error: `Validation failed: ${validation.errors?.join(', ')}`,
        };
      }

      return skillRegistry.execute(
        name,
        { action: name, params },
        {
          agentId: this.id,
          sessionId: this.sessionId,
          variables: {
            mode: this.mode,
            fsmState: this.fsm?.getStateName(),
          },
        }
      );
    }

    return {
      success: false,
      error: `Capability "${name}" not found`,
    };
  }

  /**
   * List all executable capability names.
   */
  private getCapabilityNames(): string[] {
    return [
      ...actionRegistry.keys().filter(name => !this.allowedActions || this.allowedActions.has(name)),
      ...toolRegistry.keys().filter(name => !this.allowedTools || this.allowedTools.has(name)),
      ...skillRegistry.keys().filter(name => !this.allowedSkills || this.allowedSkills.has(name)),
    ];
  }

  /**
   * Format registered capabilities for planner and LLM prompts.
   */
  private formatCapabilitiesForPrompt(): string {
    const lines: string[] = [];

    const actions = actionRegistry.list()
      .filter(action => !this.allowedActions || this.allowedActions.has(action.name));
    if (actions.length > 0) {
      lines.push('Actions:', ...actions.map(action => `- ${action.name}: ${action.description || 'No description'}`));
    }

    const tools = toolRegistry.list()
      .filter(tool => !this.allowedTools || this.allowedTools.has(tool.name));
    if (tools.length > 0) {
      lines.push(
        'Tools:',
        ...tools.map(tool => `- ${tool.name}: ${tool.description || 'No description'}`)
      );
    }

    const skills = skillRegistry.listManifests()
      .filter(skill => !this.allowedSkills || this.allowedSkills.has(skill.name));
    if (skills.length > 0) {
      lines.push(
        'Skills:',
        ...skills.map(skill => {
          const tags = skill.tags.length > 0 ? ` [${skill.tags.join(', ')}]` : '';
          return `- ${skill.name}: ${skill.description}${tags}`;
        })
      );
    }

    return lines.join('\n') || 'No executable capabilities registered';
  }

  /**
   * Execute in conversational mode
   */
  private async executeConversationalMode(messages: LLMMessage[]): Promise<void> {
    try {
      let fullResponse = '';
      this.state = 'synthesizing';

      const boundedMessages = this.compactMessagesForActiveAllocation(messages);
      const estimatedInputTokens = this.estimateMessageTokens(boundedMessages);
      for await (const chunk of this.llm!.stream(boundedMessages, this.completionOptions({}, estimatedInputTokens))) {
        if (chunk.usage) {
          this.recordUsage(chunk);
        }

        fullResponse += chunk.content;

        if (this.messageQueue) {
          await this.messageQueue.send(
            this.name,
            'env',
            chunk.content,
            { stream: !chunk.done, done: chunk.done }
          );
        } else {
          process.stdout.write(chunk.content);
        }
      }

      this.addToMemory('result', fullResponse);

      if (this.fsm) {
        this.fsm.addToTrace(`Response: ${fullResponse.substring(0, 50)}...`);
      }
    } catch (error) {
      const errorMsg = `Error: ${error}`;
      logger.error(`Agent ${this.name} error:`, error);
      this.addToMemory('result', errorMsg);
      if (this.messageQueue) {
        await this.messageQueue.send(this.name, 'env', errorMsg, { done: true });
      }
    }
  }

  private compactMessagesForActiveAllocation(messages: LLMMessage[]): LLMMessage[] {
    const limit = this.getCompletionTokenLimit();
    if (limit === undefined || messages.length === 0) return messages;
    const outputReserve = Math.min(512, Math.max(128, Math.floor(limit * 0.25)));
    const inputBudget = Math.max(32, limit - outputReserve);
    const estimated = this.estimateMessageTokens(messages);
    if (estimated <= inputBudget) return messages;

    const system = messages.find(message => message.role === 'system');
    const user = [...messages].reverse().find(message => message.role === 'user') ?? messages.at(-1);
    if (!user) return messages;
    const wrapperReserve = 16;
    const contentBudget = Math.max(16, inputBudget - wrapperReserve);
    let systemBudget = system ? Math.max(8, Math.floor(contentBudget * 0.42)) : 0;
    let userBudget = Math.max(8, contentBudget - systemBudget);
    let compact: LLMMessage[] = [];
    for (let attempt = 0; attempt < 4; attempt += 1) {
      compact = [];
      if (system) compact.push({ role: 'system', content: this.truncateForTokenBudget(system.content, systemBudget) });
      compact.push({ role: user.role, content: this.truncateForTokenBudget(user.content, userBudget) });
      const compactEstimate = this.estimateMessageTokens(compact);
      if (compactEstimate <= inputBudget) return compact;
      const scale = Math.max(0.1, Math.min(0.9, (inputBudget - wrapperReserve) / compactEstimate));
      systemBudget = system ? Math.max(8, Math.floor(systemBudget * scale)) : 0;
      userBudget = Math.max(8, Math.floor(userBudget * scale));
    }
    return compact;
  }

  private truncateForTokenBudget(content: string, tokenBudget: number): string {
    const maxChars = Math.max(32, tokenBudget * 4);
    if (content.length <= maxChars) return content;
    const marker = '\n...[active allocation context truncation]...\n';
    const available = Math.max(16, maxChars - marker.length);
    const head = Math.floor(available * 0.65);
    return `${content.slice(0, head)}${marker}${content.slice(-(available - head))}`;
  }

  /**
   * Calculate uncertainty metric (0-1)
   */
  private calculateUncertainty(text: string): number {
    const len = text.length;
    const questionCount = (text.match(/\?/g) || []).length;
    const complexWords = (text.match(/\b(however|although|therefore|because|perhaps|might|may|could)\b/gi) || []).length;
    return Math.min(1.0, (len / 2000) + (questionCount * 0.15) + (complexWords * 0.1));
  }

  /**
   * Analyze conflict in text (0-1)
   */
  private analyzeConflict(text: string): number {
    const conflictIndicators = ['but', 'however', 'although', 'conflict', 'disagree', 'contradict', 'versus', 'or', 'vs'];
    const lowerText = text.toLowerCase();
    let conflicts = 0;
    for (const indicator of conflictIndicators) {
      const regex = new RegExp(`\\b${indicator}\\b`, 'gi');
      const matches = lowerText.match(regex);
      if (matches) conflicts += matches.length;
    }
    return Math.min(1.0, conflicts * 0.15);
  }

  /**
   * Analyze evidence in text (0-1)
   */
  private analyzeEvidence(text: string): number {
    const evidenceIndicators = [
      'because', 'therefore', 'evidence', 'proves', 'shows',
      'since', 'thus', 'according to', 'data shows', 'research shows',
      'studies show', 'resulting in', 'leads to',
    ];
    const lowerText = text.toLowerCase();
    let evidence = 0;
    for (const indicator of evidenceIndicators) {
      if (lowerText.includes(indicator)) evidence++;
    }
    return Math.min(1.0, evidence * 0.15);
  }

  /**
   * Initialize with session ID
   */
  async initialize(sessionId: string): Promise<void> {
    this.sessionId = sessionId;
    await super.initialize(sessionId);

    if (this.useContextManager) {
      contextManager.upsert(this.name, sessionId, `# Session: ${sessionId}\n\n`);
    }
  }

  /**
   * Cleanup
   */
  async cleanup(sessionId?: string): Promise<void> {
    const targetSessionId = sessionId ?? this.sessionId;
    if (targetSessionId && this.useContextManager) {
      contextManager.delete(this.name, targetSessionId);
    }
    if (!sessionId || this.sessionId === targetSessionId) this.sessionId = '';
    await super.cleanup(targetSessionId || undefined);
  }

  /**
   * Run the agent main loop
   */
  async run(): Promise<void> {
    if (!this.messageQueue) {
      throw new Error('Message queue not set');
    }

    logger.info(`Agent ${this.name} started running in ${this.mode} mode`);

    while (this.state !== 'stopped') {
      try {
        const message = await this.messageQueue.receive(this.name);

        if (message) {
          await this.step(String(message.content));
        }

        await new Promise(resolve => setTimeout(resolve, 10));
      } catch (error) {
        logger.error(`Agent ${this.name} run error:`, error);
        this.state = 'idle';
      }
    }

    logger.info(`Agent ${this.name} stopped`);
  }

  /**
   * Get context from ContextManager
   */
  getContext(): string {
    if (!this.sessionId || !this.useContextManager) {
      return '';
    }
    const doc = contextManager.get(this.name, this.sessionId);
    return doc?.content ?? '';
  }
}

function normalizePlannedToolCalls(
  response: {
    action?: string;
    reason?: string;
    calls?: Array<{ toolName?: unknown; params?: unknown }>;
  },
  authorized: Set<string>,
  remainingCalls: number,
  round: number
): PlannedToolCall[] {
  if (response.action !== 'call_tools' || !Array.isArray(response.calls)) return [];
  return response.calls
    .filter(call => typeof call.toolName === 'string' && authorized.has(call.toolName))
    .filter(call => !isFragileShellFileWriter(call, authorized))
    .slice(0, remainingCalls)
    .map(call => ({
      toolName: String(call.toolName),
      params: call.params && typeof call.params === 'object' && !Array.isArray(call.params)
        ? call.params as Record<string, unknown>
        : {},
      reason: response.reason?.trim() || `Agent requested another tool round after observing round ${round}.`,
      groundingRequired: true,
    }));
}

function findLastToolCallIndex(
  calls: ToolLoopCallRecord[],
  predicate: (call: ToolLoopCallRecord) => boolean
): number {
  for (let index = calls.length - 1; index >= 0; index -= 1) {
    if (predicate(calls[index]!)) return index;
  }
  return -1;
}

function isDestructiveRepairOverwrite(
  planned: PlannedToolCall,
  completed: ToolLoopCallRecord[]
): boolean {
  if (planned.toolName !== 'fs.write'
    || String(planned.params.mode ?? 'overwrite') !== 'overwrite') {
    return false;
  }
  const target = normalizePlannedWorkspacePath(String(planned.params.path ?? ''));
  if (!target) return false;
  return completed.some(call => {
    if (!call.success
      || (call.toolName !== 'fs.read'
        && call.toolName !== 'fs.write'
        && call.toolName !== 'fs.replace')) {
      return false;
    }
    return normalizePlannedWorkspacePath(String(call.params.path ?? '')) === target;
  });
}

function normalizePlannedWorkspacePath(value: string): string {
  return value.trim().replace(/\\/g, '/').replace(/^(?:\.\/)+/, '').replace(/\/+/g, '/');
}

function isFragileShellFileWriter(
  call: { toolName?: unknown; params?: unknown },
  authorized: Set<string>
): boolean {
  if (call.toolName !== 'shell.exec'
    || (!authorized.has('fs.write') && !authorized.has('fs.replace'))) {
    return false;
  }
  const params = call.params && typeof call.params === 'object' && !Array.isArray(call.params)
    ? call.params as Record<string, unknown>
    : {};
  const command = String(params.command ?? '');
  return /\b(?:python|python3|node)\b[\s\S]*(?:writeFile(?:Sync)?|write_text|write_bytes|writelines|open\s*\([^)]*['"][wa]['"])/i.test(command);
}

function shouldSuppressRepeatedPlannedCall(
  planned: PlannedToolCall,
  completed: ToolLoopCallRecord[]
): boolean {
  const fingerprint = plannedToolCallFingerprint(planned);
  let previousIndex = -1;
  for (let index = completed.length - 1; index >= 0; index -= 1) {
    if (plannedToolCallFingerprint(completed[index]!) === fingerprint) {
      previousIndex = index;
      break;
    }
  }
  if (previousIndex < 0) return false;
  const mutationAfterPrevious = completed
    .slice(previousIndex + 1)
    .some(call => isSuccessfulWorkspaceMutation(call));
  const repeatCanObserveNewWorkspaceState = planned.toolName === 'fs.list'
    || planned.toolName === 'fs.read'
    || planned.toolName === 'fs.search'
    || isSuccessfulWorkspaceVerification({ ...planned, success: true });
  return !repeatCanObserveNewWorkspaceState || !mutationAfterPrevious;
}

function plannedToolCallFingerprint(
  call: Pick<PlannedToolCall, 'toolName' | 'params'>
): string {
  return workspaceToolIntentFingerprint({
    toolName: call.toolName,
    params: call.params,
  });
}

function isRetryableToolPlanningResponseError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.startsWith('Failed to parse JSON response')
    || message === 'Empty JSON response';
}

function recoverTruncatedFsWriteResponse(
  error: unknown,
  authorized: Set<string>,
  completed: ToolLoopCallRecord[]
): {
  action: 'call_tools';
  reason: string;
  calls: Array<{ toolName: 'fs.write'; params: Record<string, unknown> }>;
} | undefined {
  if (!authorized.has('fs.write')) return undefined;
  const message = error instanceof Error ? error.message : String(error);
  const prefix = 'Failed to parse JSON response:';
  if (!message.startsWith(prefix)) return undefined;
  const raw = message.slice(prefix.length);
  const writeIndex = raw.search(/"toolName"\s*:\s*"fs\.write"/);
  if (writeIndex < 0) return undefined;
  const writePayload = raw.slice(writeIndex);
  const pathField = extractPartialJsonStringField(writePayload, 'path');
  const contentField = extractPartialJsonStringField(writePayload, 'content');
  const filePath = pathField?.value.trim() ?? '';
  if (!filePath || !contentField || contentField.value.length < 32) return undefined;

  let content = contentField.value.slice(0, 6_000);
  if (!contentField.complete) {
    const lastNewline = content.lastIndexOf('\n');
    const completeLines = lastNewline >= 0 ? content.slice(0, lastNewline + 1) : '';
    if (completeLines.trim().length >= 16) content = completeLines;
  }
  if (content.trim().length < 16) return undefined;

  const priorRecoveredChunk = [...completed].reverse().find(call =>
    call.toolName === 'fs.write'
    && call.success
    && String(call.params.path ?? '') === filePath
    && call.reason?.includes('truncated structured fs.write')
  );
  const priorContent = String(priorRecoveredChunk?.params.content ?? '');
  const restartsFromPriorPrefix = priorContent.length > 0
    && content.startsWith(priorContent.slice(0, Math.min(160, priorContent.length)));
  const mode = priorRecoveredChunk && !restartsFromPriorPrefix
    ? 'append'
    : 'overwrite';
  return {
    action: 'call_tools',
    reason: 'Recovered a bounded source chunk from a truncated structured fs.write response so generated implementation work is not discarded.',
    calls: [{
      toolName: 'fs.write',
      params: {
        path: filePath,
        content,
        mode,
        createDirectories: true,
      },
    }],
  };
}

function extractPartialJsonStringField(
  input: string,
  field: string
): { value: string; complete: boolean } | undefined {
  const fieldMatch = new RegExp(`"${field}"\\s*:\\s*"`).exec(input);
  if (!fieldMatch || fieldMatch.index === undefined) return undefined;
  let index = fieldMatch.index + fieldMatch[0].length;
  let value = '';
  while (index < input.length) {
    const char = input[index]!;
    if (char === '"') return { value, complete: true };
    if (char !== '\\') {
      value += char;
      index += 1;
      continue;
    }
    index += 1;
    if (index >= input.length) break;
    const escaped = input[index]!;
    const simpleEscapes: Record<string, string> = {
      '"': '"',
      '\\': '\\',
      '/': '/',
      b: '\b',
      f: '\f',
      n: '\n',
      r: '\r',
      t: '\t',
    };
    if (escaped === 'u') {
      const hex = input.slice(index + 1, index + 5);
      if (!/^[0-9a-f]{4}$/i.test(hex)) break;
      value += String.fromCharCode(Number.parseInt(hex, 16));
      index += 5;
      continue;
    }
    value += simpleEscapes[escaped] ?? escaped;
    index += 1;
  }
  return { value, complete: false };
}

function compactToolPlanningTask(task: string): string {
  const maxChars = 12_000;
  if (task.length <= maxChars) return task;
  const headChars = 7_000;
  const tailChars = maxChars - headChars;
  return [
    task.slice(0, headChars),
    '[runtime_compacted_middle_for_tool_planning]',
    task.slice(-tailChars),
  ].join('\n');
}

function compactToolObservation(
  result: unknown,
  toolName: string,
  latest = true
): unknown {
  if (result && typeof result === 'object' && toolName === 'web.search') {
    const search = result as {
      query?: unknown;
      provider?: unknown;
      results?: Array<{ title?: unknown; url?: unknown; snippet?: unknown }>;
    };
    return {
      query: search.query,
      provider: search.provider,
      results: (search.results ?? []).slice(0, 5).map(item => ({
        title: String(item.title ?? '').slice(0, 180),
        url: String(item.url ?? '').slice(0, 500),
        snippet: String(item.snippet ?? '').slice(0, 320),
      })),
    };
  }
  if (result && typeof result === 'object' && toolName === 'web.fetch') {
    const page = result as {
      finalUrl?: unknown;
      title?: unknown;
      text?: unknown;
      links?: Array<{ text?: unknown; url?: unknown }>;
    };
    return {
      finalUrl: page.finalUrl,
      title: String(page.title ?? '').slice(0, 240),
      text: String(page.text ?? '').slice(0, 1200),
      links: (page.links ?? []).slice(0, 8).map(link => ({
        text: String(link.text ?? '').slice(0, 120),
        url: String(link.url ?? '').slice(0, 500),
      })),
    };
  }
  if (result && typeof result === 'object' && toolName === 'shell.exec') {
    const shell = result as {
      command?: unknown;
      stdout?: unknown;
      stderr?: unknown;
      exitCode?: unknown;
      timedOut?: unknown;
    };
    return {
      command: String(shell.command ?? '').slice(0, latest ? 800 : 250),
      stdout: compactTail(String(shell.stdout ?? ''), latest ? 1_500 : 250),
      stderr: compactTail(String(shell.stderr ?? ''), latest ? 3_000 : 500),
      exitCode: shell.exitCode,
      timedOut: shell.timedOut,
    };
  }
  if (result && typeof result === 'object' && toolName === 'fs.read') {
    const read = result as {
      path?: unknown;
      content?: unknown;
      bytes?: unknown;
      truncated?: unknown;
      startLine?: unknown;
      endLine?: unknown;
      totalLines?: unknown;
    };
    return {
      path: read.path,
      content: String(read.content ?? '').slice(0, latest ? 6_000 : 800),
      bytes: read.bytes,
      truncated: read.truncated,
      startLine: read.startLine,
      endLine: read.endLine,
      totalLines: read.totalLines,
    };
  }
  if (result && typeof result === 'object' && toolName === 'fs.search') {
    const search = result as {
      query?: unknown;
      filesSearched?: unknown;
      matches?: Array<{ path?: unknown; line?: unknown; column?: unknown; preview?: unknown }>;
      truncated?: unknown;
    };
    return {
      query: search.query,
      filesSearched: search.filesSearched,
      matches: (search.matches ?? []).slice(0, latest ? 20 : 5).map(match => ({
        path: match.path,
        line: match.line,
        column: match.column,
        preview: String(match.preview ?? '').slice(0, latest ? 350 : 180),
      })),
      truncated: search.truncated,
    };
  }
  return compactObservationValue(result, 0);
}

function compactToolPlanningParams(
  params: Record<string, unknown>,
  latest: boolean
): Record<string, unknown> {
  const maxStringChars = latest ? 1_000 : 250;
  return Object.fromEntries(Object.entries(params).map(([key, value]) => {
    if (typeof value !== 'string') return [key, compactObservationValue(value, 0)];
    if (value.length <= maxStringChars) return [key, value];
    const headChars = Math.floor(maxStringChars * 0.6);
    return [
      key,
      `${value.slice(0, headChars)}...[${value.length - maxStringChars} chars compacted]...${value.slice(-(maxStringChars - headChars))}`,
    ];
  }));
}

function compactTail(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `[${value.length - maxChars} earlier chars compacted]\n${value.slice(-maxChars)}`;
}

function extractPlannerFallbackUrls(message: string): string[] {
  const matches = message.match(/https?:\/\/[^\s<>'"`\])}]+/gi) ?? [];
  return Array.from(new Set(matches
    .map(url => url.replace(/[.,;:!?]+$/, ''))
    .filter(url => {
      try {
        const parsed = new URL(url);
        return parsed.protocol === 'https:' && !parsed.username && !parsed.password;
      } catch {
        return false;
      }
    })))
    .slice(0, 3);
}

function compactObservationValue(value: unknown, depth: number): unknown {
  if (value === undefined || value === null || typeof value === 'boolean' || typeof value === 'number') return value;
  if (typeof value === 'string') return value.length > 500 ? `${value.slice(0, 500)}...[truncated]` : value;
  if (depth >= 3) return '[nested value omitted]';
  if (Array.isArray(value)) return value.slice(0, 6).map(item => compactObservationValue(item, depth + 1));
  if (typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .slice(0, 12)
      .map(([key, item]) => [key, compactObservationValue(item, depth + 1)]));
  }
  return String(value).slice(0, 500);
}

export default UnifiedAgent;
