import { describe, expect, it } from 'vitest';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime, { type DelegationDecision } from '../src/core/runtime/Runtime.js';
import type { LLMCompletionOptions, LLMCompletionResult, LLMJSONCompletionResult, LLMMessage, LLMProvider, LLMStreamChunk } from '../src/core/llm/types.js';

class RootDelegationLLM implements LLMProvider {
  readonly name = 'root-delegation-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'complete',
      usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
    };
  }

  async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    let content = 'Roy direct response.';
    if (text.includes('fs.list failed')) {
      content = 'Final synthesis: the subagent could not inspect the requested directory, so Roy reports the failure and suggests checking the path.';
    } else if (text.includes('Synthesize their results into one final user-facing response')) {
      content = 'Final synthesis from Researcher-1 and Critic-2.';
    } else if (text.includes('definitely-not-real-dir')) {
      content = 'Researcher report: unable to inspect ./definitely-not-real-dir.';
    } else if (text.includes('architectural risks') || text.includes('failure modes')) {
      content = 'Critic report: coupling and runtime observability risks.';
    } else if (text.includes('grounded project structure')) {
      content = 'Researcher report: observed README.md, package.json, src/, tests/.';
    }
    yield {
      content,
      done: true,
      usage: { promptTokens: 20, completionTokens: 7, totalTokens: 27 },
    };
  }

  async completeJSON<T>(messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller") && text.includes('Help me improve this')) {
      return {
        action: 'ask_clarification',
        reason: 'The request is too ambiguous to assign a safe delegation task.',
        question: 'What would you like Roy to improve: code, architecture, documentation, tests, memory/cache behavior, or CLI/API behavior?',
      } satisfies DelegationDecision as T;
    }

    if (text.includes("Roy's root delegation controller") && text.includes('definitely-not-real-dir')) {
      return {
        action: 'spawn_subagents',
        reason: 'The request explicitly asks for project inspection by a subagent.',
        agents: [
          {
            archetype: 'researcher',
            name: 'Researcher-1',
            task: 'Inspect ./definitely-not-real-dir and report whether it exists.',
            tomLevel: 0,
          },
        ],
      } satisfies DelegationDecision as T;
    }

    if (text.includes("Roy's root delegation controller") && text.includes('current project structure')) {
      return {
        action: 'spawn_subagents',
        reason: 'Project structure inspection should use a researcher.',
        agents: [
          {
            archetype: 'researcher',
            name: 'Researcher-1',
            task: 'Inspect the current project structure and summarize risks.',
            tomLevel: 0,
          },
        ],
      } satisfies DelegationDecision as T;
    }

    if (text.includes("Roy's root delegation controller") && text.includes('architectural risks')) {
      return {
        action: 'spawn_subagents',
        reason: 'Architecture risk analysis needs grounded inspection and critique.',
        agents: [
          {
            archetype: 'researcher',
            name: 'Researcher-1',
            task: 'Inspect grounded project structure and collect concrete evidence for architectural risk analysis.',
            tomLevel: 0,
          },
          {
            archetype: 'critic',
            name: 'Critic-2',
            task: 'Identify architectural risks and hidden coupling from the project evidence.',
            tomLevel: 2,
          },
        ],
      } satisfies DelegationDecision as T;
    }

    if (!text.includes("Roy's root delegation controller")) {
      return { action: 'none', params: {} } as T;
    }

    return {
      action: 'solve_directly',
      reason: 'Simple conversational task.',
    } satisfies DelegationDecision as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages, options);
    return { value, completion: { content: JSON.stringify(value), usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 } } };
  }

  isConfigured(): boolean {
    return true;
  }
}

class EmptyVisibleRootSynthesisLLM extends RootDelegationLLM {
  override async *stream(messages: LLMMessage[], options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes('Synthesize their results into one final user-facing response')) {
      yield {
        content: '',
        done: true,
        usage: { promptTokens: 100, completionTokens: 512, totalTokens: 612, thinkingTokens: 512 },
      };
      return;
    }
    yield* super.stream(messages, options);
  }
}

class InfeasibleGroundedChildLLM extends RootDelegationLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'Collect repository evidence.',
        agents: [{
          archetype: 'researcher',
          name: 'UnavailableCollector-1',
          task: 'Inspect the actual repository files and return grounded source evidence.',
          tools: [],
          skills: [],
        }],
      } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class InternalKnowledgeChildLLM extends RootDelegationLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's root delegation controller")) {
      return {
        action: 'spawn_subagents',
        reason: 'An independent knowledge pass improves answer coverage.',
        continuationPolicy: 'finalize_after_round',
        agents: [{
          archetype: 'custom',
          name: 'KnowledgeSolver-1',
          task: 'Search your internal knowledge for the supplied trivia answers and return a complete answer map.',
          tools: [],
          skills: [],
        }],
      } as T;
    }
    if (text.includes("'s delegation controller")) {
      return { action: 'solve_directly', reason: 'The bounded knowledge task can be completed directly.' } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class DirectDecisionAuditLLM extends RootDelegationLLM {
  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const text = messages.map(message => String(message.content)).join('\n');
    if (text.includes("Roy's direct-solve complexity auditor")) {
      return {
        action: 'spawn_subagents',
        reason: 'The independent obligations expose separate knowledge gaps.',
        coordination: 'independent',
        continuationPolicy: 'finalize_after_round',
        agents: [
          {
            archetype: 'custom',
            name: 'CoverageA-1',
            task: 'Solve obligations 1-5 from supplied prompt knowledge.',
            tools: [],
            skills: [],
            existenceReason: 'Cover the first independent answer group.',
          },
          {
            archetype: 'custom',
            name: 'CoverageB-2',
            task: 'Solve obligations 6-10 from supplied prompt knowledge.',
            tools: [],
            skills: [],
            existenceReason: 'Cover the second independent answer group.',
          },
        ],
      } as T;
    }
    if (text.includes("Roy's root delegation controller")) {
      return { action: 'solve_directly', reason: 'Initially judged answerable in one pass.' } as T;
    }
    if (text.includes('delegation controller')) {
      return { action: 'solve_directly', reason: 'The bounded child task is direct.' } as T;
    }
    return super.completeJSON<T>(messages);
  }
}

class DeepSeekBudgetProbeLLM extends RootDelegationLLM {
  override readonly name = 'deepseek';
  override readonly defaultModel = 'deepseek-v4-flash';
  lastStreamMaxTokens?: number;

  override async *stream(messages: LLMMessage[], options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    this.lastStreamMaxTokens = options?.maxTokens;
    yield* super.stream(messages, options);
  }
}

describe('Runtime root-controlled delegation', () => {
  it('returns a delegated-result fallback when root synthesis has no visible output', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-root-fallback-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-root-fallback-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new EmptyVisibleRootSynthesisLLM(),
    });

    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');

    expect(result.finalResponse).toContain('[runtime_root_synthesis_fallback]');
    expect(result.finalResponse).toContain('Delegated result:');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({ type: 'root.synthesis.fallback' }));

    await runtime.shutdown();
  });

  it('assesses a complex task, spawns subagents, waits for results, and synthesizes', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-delegation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-delegation-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(3);
    expect(result.subagents.map(item => item.node.identity.archetype)).toEqual([
      'researcher',
      'critic',
      'summarizer',
    ]);
    expect(result.finalResponse).toBe('Final synthesis from Researcher-1 and Critic-2.');

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(0);
    for (const item of result.subagents) {
      expect(runtime.getActorLifecycle(item.agent.identity.id)).toMatchObject({
        status: 'released',
        lastDecision: { action: 'release' },
      });
    }

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('delegation.decision');
    expect(eventTypes).toContain('delegation.plan.created');
    expect(eventTypes).toContain('delegation.subagent.selected');
    expect(eventTypes).toContain('delegation.subagent.task_assigned');
    expect(eventTypes).toContain('delegation.completed');
    expect(eventTypes).toContain('agent.spawned');
    expect(eventTypes).toContain('agent.run.started');
    expect(eventTypes).toContain('agent.run.completed');
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');
    expect(eventTypes).toContain('memory.update.propose.completed');
    expect(eventTypes).toContain('tom.task.analyzed');
    expect(eventTypes).toContain('tom.signals.collected');
    expect(eventTypes).toContain('tom.gap.identified');
    expect(eventTypes).toContain('tom.delegation.coverage.evaluated');
    expect(eventTypes).toContain('tom.team.profile.created');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('user.input');
    const teamId = result.teams[0].team.identity.id;
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === teamId)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === teamId)).toHaveLength(3);
    expect(messages.map(message => message.kind)).toContain('root.synthesis');
    expect(messages.map(message => message.kind)).toContain('root.final_response');

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBeGreaterThan(0);
    for (const item of result.subagents) {
      expect(budget.perAgent[item.agent.identity.id].totalTokens).toBeGreaterThan(0);
    }
    expect(result.subagents.every(item => item.agent.identity.tomProfile.cognitiveGaps.length > 0)).toBe(true);
    const tomState = runtime.getToMState(result.correlationId);
    expect(tomState.analyses).toHaveLength(1);
    expect(tomState.analyses[0].requiresHigherOrderToM).toBe(true);
    expect(tomState.teams[0].profile.modelsAgents).toEqual(expect.arrayContaining([
      'root',
      'Researcher-1',
      'Critic-2',
    ]));
    expect(result.usage.total.totalTokens).toBeGreaterThan(0);
    const selectedCandidate = runtime.getEvents().find(event => event.type === 'delegation.candidate.selected');
    expect(selectedCandidate?.data?.investment).toMatchObject({
      model: 'weighted_reasoning_investment_v1',
    });
    expect(runtime.getEvents().some(event => event.type === 'budget.outcome.recorded')).toBe(true);

    await runtime.shutdown();
  });

  it('solves simple turns directly without spawning subagents', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-solo-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-solo-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('who are you?');

    expect(result.decision.action).toBe('solve_directly');
    expect(result.subagents).toHaveLength(0);
    expect(result.finalResponse).toBe('Roy direct response.');
    expect(runtime.getAgentTree().children).toHaveLength(0);
    expect(runtime.getEvents().map(event => event.type)).toContain('root.solo.completed');
    expect(runtime.getEvents().map(event => event.type)).toContain('delegation.skipped');

    await runtime.shutdown();
  });

  it('drops a grounded child plan when policy exposes no executable tool path', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-infeasible-child-plan-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tools: {
        approval: {
          readOnly: 'deny',
          write: 'deny',
          execute: 'deny',
          overrides: {},
        },
      },
      agents: {
        defaultToolsByArchetype: {
          researcher: [],
          critic: [],
          planner: [],
          coder: [],
          summarizer: [],
          tester: [],
          custom: [],
        },
      },
      tom: {
        autoCompleteGaps: false,
        enforceMinimumCoverage: false,
        minimumCoverage: 0,
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'infeasible-child-plan-test',
      workspaceCwd,
      llmProvider: new InfeasibleGroundedChildLLM(),
    });

    const result = await runtime.handleUserTurn('Summarize one conclusion from the supplied prompt.');

    expect(result.decision.action).toBe('solve_directly');
    expect(result.subagents).toHaveLength(0);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'delegation.plan.infeasible',
      data: expect.objectContaining({
        rejected: [expect.objectContaining({
          name: 'UnavailableCollector-1',
          reason: 'grounding_required_but_no_automatically_authorized_tool_path',
        })],
      }),
    }));
    await runtime.shutdown();
  });

  it('keeps internal-knowledge search executable without a web tool path', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-internal-knowledge-child-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tools: {
        approval: {
          readOnly: 'deny',
          write: 'deny',
          execute: 'deny',
          overrides: {},
        },
      },
      agents: {
        defaultToolsByArchetype: {
          researcher: [],
          critic: [],
          planner: [],
          coder: [],
          summarizer: [],
          tester: [],
          custom: [],
        },
      },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'internal-knowledge-child-test',
      workspaceCwd,
      llmProvider: new InternalKnowledgeChildLLM(),
    });

    const result = await runtime.handleUserTurn(
      'Answer the supplied trivia questions from internal knowledge, then synthesize the result.'
    );

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(1);
    expect(result.subagents[0].subagentResult.result).not.toBe('');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'delegation.plan.infeasible',
      data: expect.objectContaining({
        rejected: expect.arrayContaining([
          expect.objectContaining({
            reason: 'grounding_required_but_no_automatically_authorized_tool_path',
            automaticallyAuthorizedTools: [],
          }),
        ]),
        retainedAgents: ['KnowledgeSolver-1'],
      }),
    }));
    await runtime.shutdown();
  });

  it('audits a direct decision when a task has many independent obligations', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-direct-decision-audit-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { autoCompleteGaps: false, minimumCoverage: 0 },
      teams: { createForMultipleAgents: false },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'direct-decision-audit-test',
      workspaceCwd,
      llmProvider: new DirectDecisionAuditLLM(),
    });

    const task = Array.from({ length: 10 }, (_, index) =>
      `${index + 1}. Answer independent supplied question ${index + 1}.`
    ).join('\n');
    const result = await runtime.handleUserTurn(task);

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents.length).toBeGreaterThanOrEqual(1);
    expect(runtime.getEvents().map(event => event.type))
      .toContain('delegation.direct_decision.audit.overridden');
    await runtime.shutdown();
  });

  it('reserves reasoning capacity for visible output on direct reasoning models', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-reasoning-reserve-'));
    const llm = new DeepSeekBudgetProbeLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-reasoning-reserve-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: llm,
    });

    const result = await runtime.handleUserTurn('who are you?');

    expect(result.finalResponse).toBe('Roy direct response.');
    expect(llm.lastStreamMaxTokens).toBeGreaterThanOrEqual(3584);
    await runtime.shutdown();
  });

  it('asks for clarification on ambiguous tasks without spawning subagents', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-clarify-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-clarify-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('Help me improve this.');

    expect(result.decision.action).toBe('ask_clarification');
    expect(result.subagents).toHaveLength(0);
    expect(result.finalResponse).toContain('What would you like Roy to improve');
    expect(runtime.getAgentTree().children).toHaveLength(0);
    expect(runtime.getEvents().map(event => event.type)).toContain('delegation.skipped');

    await runtime.shutdown();
  });

  it('records cache participation in delegation decisions and still creates fresh runtime instances', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-cache-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(path.join(workspaceCwd, '.roy', 'config.json'), JSON.stringify({
      tom: { enforceMinimumCoverage: true },
    }));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-cache-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });
    const result = await runtime.handleUserTurn('Inspect the current project structure and summarize the risks.');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(3);
    expect(result.subagents[0].agent.identity.id).toBe('agent_researcher_002');
    expect(result.subagents[0].creationUsage.mode).toBe('cache_hit');
    expect(result.subagents[0].creationUsage.definitionTokens).toBe(0);

    const decisionEvent = runtime.getEvents()
      .find(event => event.type === 'delegation.decision'
        && event.agentId === 'root'
        && event.data?.correlationId === result.correlationId);
    expect(decisionEvent?.data?.cacheUsed).toBe(true);
    const hits = runtime.getEvents()
      .filter(event => event.type === 'cache.hit' && event.data?.correlationId === result.correlationId);
    expect(hits.map(event => event.data?.patternId)).toContain('agent_pattern_researcher_v1');

    const agentPatterns = await runtime.getCachePatterns('agents');
    const researcherPatterns = agentPatterns.filter(pattern => pattern.archetype === 'researcher');
    expect(researcherPatterns.some(pattern => {
      const profile = pattern.tomProfile as { beliefScope?: unknown[]; cognitiveGaps?: unknown[] } | undefined;
      return (profile?.beliefScope?.length ?? 0) > 0 && (profile?.cognitiveGaps?.length ?? 0) > 0;
    })).toBe(true);
    const delegationPatterns = await runtime.getCachePatterns('delegations');
    expect(delegationPatterns.some(pattern => {
      const profile = pattern.tomProfile as { perspective?: string } | undefined;
      return typeof profile?.perspective === 'string'
        && Array.isArray(pattern.cognitiveGapIds)
        && pattern.cognitiveGapIds.length > 0;
    })).toBe(true);

    await runtime.shutdown();
  });

  it('reduces delegation under constrained budget', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-budget-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-budget-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });
    runtime.setBudget(2000);

    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(1);
    expect(result.decision.reason).toContain('Budget constrained');
    const decisionEvent = runtime.getEvents()
      .find(event => event.type === 'delegation.decision' && event.data?.correlationId === result.correlationId);
    expect(decisionEvent?.data?.budgetMode).toBe('limited');
    await runtime.shutdown();
  });

  it('scores delegation candidates and respects maxTotalAgentsPerTurn', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-max-total-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        version: 1,
        memoryUpdates: 'suggest',
        delegation: {
          maxChildrenPerParent: 5,
          maxDepth: 3,
          maxTotalAgentsPerTurn: 1,
          allowCustomAgents: true,
          budgetAware: true,
        },
      }, null, 2) + '\n',
      'utf8'
    );
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-max-total-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('Analyze this repo and find architectural risks');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(1);
    expect(runtime.getAgentTree().children).toHaveLength(0);
    expect(runtime.getActorLifecycle(result.subagents[0].agent.identity.id)).toMatchObject({ status: 'released' });
    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('delegation.candidate.generated');
    expect(eventTypes).toContain('delegation.candidate.selected');
    const selected = runtime.getEvents().find(event => event.type === 'delegation.candidate.selected');
    expect((selected?.data?.agents as unknown[] | undefined)?.length).toBe(1);

    await expect(runtime.handleSpawnCommand({
      archetype: 'critic',
      task: 'Try to exceed the same turn total agent limit',
      correlationId: result.correlationId,
    })).rejects.toThrow('max_total_agents_per_turn_exceeded');

    await runtime.shutdown();
  });

  it('records tool errors and lets root recover when subagent inspection fails', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-phase2-failure-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'phase2-failure-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });

    const result = await runtime.handleUserTurn('Use a subagent to inspect a nonexistent directory named ./definitely-not-real-dir');

    expect(result.decision.action).toBe('spawn_subagents');
    expect(result.subagents).toHaveLength(1);
    expect(result.subagents[0].subagentResult.grounded).toBe(false);
    expect(result.subagents[0].subagentResult.warnings.some(warning => warning.includes('fs.list failed'))).toBe(true);
    expect(result.finalResponse).toContain('could not inspect');
    expect(runtime.getEvents().map(event => event.type)).toContain('tool.error');

    await runtime.shutdown();
  });

  it('removes requested web tools from local workspace agent tasks', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-local-tool-routing-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'local-tool-routing-test',
      workspaceCwd,
      fsmEnabled: true,
      llmProvider: new RootDelegationLLM(),
    });
    const bindings = (runtime as unknown as {
      getAutomaticallyApprovedToolBindings: (
        archetype: string,
        task: string,
        requestedTools: string[]
      ) => Array<{ name: string }>;
    }).getAutomaticallyApprovedToolBindings(
      'custom',
      'Read /app/src/dq_audit/audit.py and inspect the local verifier.',
      ['web.search', 'web.fetch', 'fs.read']
    );

    expect(bindings.map(binding => binding.name)).toEqual(['fs.read']);

    await runtime.shutdown();
  });
});
