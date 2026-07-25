import { describe, it, expect } from 'vitest';
import { mkdir, mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import Runtime, { type RunAgentResult } from '../src/core/runtime/Runtime.js';
import type { LLMProvider, LLMMessage, LLMCompletionOptions, LLMCompletionResult, LLMJSONCompletionResult, LLMStreamChunk } from '../src/core/llm/types.js';

class EchoLLM implements LLMProvider {
  readonly name = 'echo-test';
  readonly defaultModel = 'test-model';

  async complete(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<LLMCompletionResult> {
    return {
      content: 'echo complete',
      usage: {
        promptTokens: 5,
        completionTokens: 3,
        totalTokens: 8,
      },
    };
  }

  async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'subagent ', done: false };
    yield {
      content: 'result',
      done: true,
      usage: {
        promptTokens: 7,
        completionTokens: 2,
        totalTokens: 9,
      },
    };
  }

  async completeJSON<T>(_messages: LLMMessage[], _options?: LLMCompletionOptions): Promise<T> {
    return { action: 'none', params: {} } as T;
  }

  async completeJSONWithUsage<T>(messages: LLMMessage[], options?: LLMCompletionOptions): Promise<LLMJSONCompletionResult<T>> {
    const value = await this.completeJSON<T>(messages, options);
    return { value, completion: { content: JSON.stringify(value), usage: { promptTokens: 3, completionTokens: 1, totalTokens: 4 } } };
  }

  isConfigured(): boolean {
    return true;
  }
}

class OversizedUsageLLM extends EchoLLM {
  override async *stream(
    _messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield {
      content: 'Completed a productive long-horizon slice beyond the initial accounting estimate.',
      done: true,
      usage: {
        promptTokens: 420,
        completionTokens: 180,
        totalTokens: 600,
      },
    };
  }
}

class SaturatedClosurePlanningLLM extends EchoLLM {
  sawSaturatedEvidence = false;

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    if (prompt.includes('You plan authorized tool calls')
      && prompt.includes('"workspaceEvidenceSaturated":true')) {
      this.sawSaturatedEvidence = true;
      return {
        action: 'call_tools',
        reason: 'Apply the explicit implementation repair from the closed evidence phase.',
        calls: [{
          toolName: 'fs.replace',
          params: {
            path: 'app.ts',
            oldText: 'export const value = "stub";',
            newText: 'export const value = "implemented";',
            expectedReplacements: 1,
          },
        }],
      } as T;
    }
    return { action: 'finish', reason: 'No action selected.', calls: [] } as T;
  }
}

class DiversifyingWebLLM extends EchoLLM {
  sawSharedFrontier = false;

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    if (prompt.includes('[runtime_shared_web_frontier]')) {
      this.sawSharedFrontier = true;
      return {
        action: 'call_tools',
        reason: 'Probe the shared conclusions with independent search formulations.',
        calls: [
          {
            toolName: 'web.search',
            params: { query: 'Hamlet authorship Shakespeare primary evidence' },
          },
          {
            toolName: 'web.search',
            params: { query: 'Macbeth authorship Shakespeare primary evidence' },
          },
        ],
      } as T;
    }
    return { action: 'finish', reason: 'No additional tool call is needed.', calls: [] } as T;
  }
}

class ContradictoryArchitectureLLM extends EchoLLM {
  override async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'This is a Rust project built around Cargo.toml, Cargo.lock, and src/main.rs.', done: false };
    yield {
      content: '',
      done: true,
      usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 },
    };
  }
}

class FabricatedPathsLLM extends EchoLLM {
  override async *stream(_messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield { content: 'The repository contains `package.json`, `src/fabricated/worker.ts`, and `config/missing.yaml`.', done: false };
    yield { content: '', done: true, usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 } };
  }
}

class AbsoluteObservedPathsLLM extends EchoLLM {
  constructor(private readonly workspaceRoot: string) {
    super();
  }

  override async *stream(
    _messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    yield {
      content: `The grounded project files are \`${this.workspaceRoot}/package.json\` and \`${this.workspaceRoot}/index.ts\`.`,
      done: true,
      usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 },
    };
  }
}

class MarkdownToolIntentLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The runtime evidence confirms package.json and the project source tree.'
      : '```tool\nfs.read\n{"path":"package.json"}\n```';
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

class XmlToolIntentRecoveryLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The recovered runtime call read evidence.txt and confirmed the value.'
      : '<tool_call><tool_name>fs.read</tool_name><path>evidence.txt</path></tool_call>';
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

class FunctionXmlToolIntentRecoveryLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The runtime executed the function-style request and grounded evidence.txt.'
      : [
        'I need one more source observation.',
        '<function>',
        '<name>fs.read</name>',
        '<params>',
        '<path>evidence.txt</path>',
        '</params>',
        '</function>',
      ].join('\n');
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

class NativeXmlToolIntentRecoveryLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => String(message.content)).join('\n');
    const content = prompt.includes('Produce the final task result from the evidence above.')
      ? 'The native runtime request read evidence.txt and confirmed runtime-grounded.'
      : [
        '<tool_calls>',
        '<invoke name="fs.read">',
        '<parameter name="path" string="true">evidence.txt</parameter>',
        '</invoke>',
        '</tool_calls>',
      ].join('\n');
    yield { content, done: true, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } };
  }
}

class CapturingStreamLLM extends EchoLLM {
  readonly streamedMessages: LLMMessage[][] = [];

  override async *stream(messages: LLMMessage[], _options?: LLMCompletionOptions): AsyncGenerator<LLMStreamChunk, void, unknown> {
    this.streamedMessages.push(messages.map(message => ({ ...message })));
    yield {
      content: 'Completed the assigned bounded task.',
      done: true,
      usage: { promptTokens: 20, completionTokens: 6, totalTokens: 26 },
    };
  }
}

class EmptyThenRecoveredLLM extends EchoLLM {
  readonly streamedPrompts: string[] = [];

  override async *stream(
    messages: LLMMessage[],
    _options?: LLMCompletionOptions
  ): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    this.streamedPrompts.push(prompt);
    const recovered = prompt.includes('Return a complete visible task result now.');
    yield {
      content: recovered ? 'Recovered complete visible result.' : '',
      done: true,
      finishReason: 'stop',
      usage: {
        promptTokens: 20,
        completionTokens: recovered ? 6 : 0,
        totalTokens: recovered ? 26 : 20,
      },
    };
  }
}

class FileSynthesisLLM extends EchoLLM {
  readonly streamedPrompts: string[] = [];

  override async completeJSON<T>(messages: LLMMessage[]): Promise<T> {
    const prompt = messages.map(message => message.content).join('\n');
    if (prompt.includes('"toolName":"fs.synthesize"')
      || prompt.includes('"toolName": "fs.synthesize"')) {
      return {
        action: 'call_tools',
        reason: 'Verify the synthesized implementation.',
        calls: [{ toolName: 'shell.exec', params: { command: 'npm test' } }],
      } as T;
    }
    return {
      action: 'call_tools',
      reason: 'Generate the complete implementation outside structured tool JSON.',
      calls: [{
        toolName: 'fs.synthesize',
        params: {
          path: 'app.js',
          instructions: 'Replace the stub with the complete assigned implementation.',
        },
      }],
    } as T;
  }

  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    this.streamedPrompts.push(prompt);
    yield {
      content: prompt.includes('[runtime_workspace_file_synthesis]')
        ? 'export function value() { return 42; }\n'
        : 'Implemented app.js and verified it with npm test.',
      done: true,
      usage: { promptTokens: 40, completionTokens: 12, totalTokens: 52 },
    };
  }
}

class ToolMarkupSynthesisLLM extends EchoLLM {
  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    yield {
      content: prompt.includes('[runtime_workspace_file_synthesis]')
        ? [
          'Let me inspect the verifier.',
          '<｜｜DSML｜｜tool_calls>',
          '<｜｜DSML｜｜invoke name="bash">',
          '<｜｜DSML｜｜parameter name="command">cat .roy/official-verifier/grade.py</｜｜DSML｜｜parameter>',
          '</｜｜DSML｜｜invoke>',
          '</｜｜DSML｜｜tool_calls>',
        ].join('\n')
        : 'No source mutation was performed.',
      done: true,
      usage: { promptTokens: 40, completionTokens: 20, totalTokens: 60 },
    };
  }
}

class FocusedPatchSynthesisLLM extends EchoLLM {
  readonly streamedPrompts: string[] = [];

  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    this.streamedPrompts.push(prompt);
    yield {
      content: prompt.includes('[runtime_workspace_file_patch_synthesis]')
        ? [
          '--- a/app.js',
          '+++ b/app.js',
          '@@ -14,1 +14,1 @@',
          '-export function value() { return 41; }',
          '+export function value() { return 42; }',
          ' export function keep() { return "stable"; }',
        ].join('\n')
        : 'Focused patch completed.',
      done: true,
      usage: { promptTokens: 32, completionTokens: 20, totalTokens: 52 },
    };
  }
}

class RecoveringFocusedPatchSynthesisLLM extends EchoLLM {
  readonly streamedPrompts: string[] = [];

  override async *stream(messages: LLMMessage[]): AsyncGenerator<LLMStreamChunk, void, unknown> {
    const prompt = messages.map(message => message.content).join('\n');
    this.streamedPrompts.push(prompt);
    yield {
      content: prompt.includes('[runtime_workspace_file_patch_recovery]')
        ? [
          '--- a/app.js',
          '+++ b/app.js',
          '@@ -1,2 +1,2 @@',
          '-export function value() { return 41; }',
          '+export function value() { return 42; }',
          ' export function keep() { return "stable"; }',
        ].join('\n')
        : [
          '--- a/app.js',
          '+++ b/app.js',
          '@@ -1,1 +1,1 @@',
          '-export function missing() { return 0; }',
          '+export function value() { return 42; }',
        ].join('\n'),
      done: true,
      usage: { promptTokens: 32, completionTokens: 20, totalTokens: 52 },
    };
  }
}

describe('Runtime controlled subagent spawning', () => {
  it('treats unlimited per-agent allocations as renewable accounting leases', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-renewable-lease-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'renewable-lease-test',
      llmProvider: new OversizedUsageLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const worker = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'summarizer',
      name: 'LongHorizonWorker',
      description: 'Complete one bounded productive slice.',
      task: 'Return one grounded summary.',
      budgetTokens: 256,
    });

    const result = await runtime.runAgent(
      worker.identity.id,
      'Return one grounded summary.',
      {
        correlationId: 'renewable-lease-correlation',
        archetype: 'summarizer',
        disableRecursiveDelegation: true,
      }
    );

    expect(result.result).toContain('productive long-horizon slice');
    expect(result.usage.totalTokens).toBe(600);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'budget.rebalanced',
      agentId: worker.identity.id,
      data: expect.objectContaining({
        purpose: 'agent_run_settlement',
        reason: 'unlimited_active_actor_lease_renewal',
        requiredConsumedTokens: 600,
      }),
    }));
    expect(runtime.getEvents().some(event =>
      event.type === 'budget.exceeded'
      && event.agentId === worker.identity.id
    )).toBe(false);
    await runtime.shutdown();
  });

  it('uses Python 3 for verifier probes and safely quotes an explicit interpreter path', () => {
    const previous = process.env.ROY_PYTHON_EXECUTABLE;
    const runtime = new Runtime();
    const buildCommand = () => (
      runtime as unknown as {
        buildPythonVerifierDiagnosticCommand: () => string;
      }
    ).buildPythonVerifierDiagnosticCommand();

    try {
      delete process.env.ROY_PYTHON_EXECUTABLE;
      expect(buildCommand()).toMatch(/^ROY_VERIFIER_PROBE=1 python3 -c /);

      process.env.ROY_PYTHON_EXECUTABLE = "/opt/Python 3's/bin/python";
      expect(buildCommand()).toMatch(
        /^ROY_VERIFIER_PROBE=1 '\/opt\/Python 3'"'"'s\/bin\/python' -c /
      );
    } finally {
      if (previous === undefined) {
        delete process.env.ROY_PYTHON_EXECUTABLE;
      } else {
        process.env.ROY_PYTHON_EXECUTABLE = previous;
      }
    }
  });

  it('requires new causal evidence before retrying a rejected synthesis', () => {
    const runtime = new Runtime();
    const unresolved = (runtime as unknown as {
      hasUnresolvedSynthesisRejection: (
        calls: Array<Record<string, unknown>>
      ) => boolean;
    }).hasUnresolvedSynthesisRejection.bind(runtime);
    const rejected = [{
      toolName: 'fs.synthesize',
      params: { path: 'src/app.py', strategy: 'patch' },
      success: false,
      result: {
        synthesisRejected: true,
        path: 'src/app.py',
        reason: 'source anchor mismatch',
      },
    }];

    expect(unresolved(rejected)).toBe(true);
    const rejectionState = (runtime as unknown as {
      latestUnresolvedSynthesisRejection: (
        calls: Array<Record<string, unknown>>
      ) => { path?: string; reason?: string } | undefined;
    }).latestUnresolvedSynthesisRejection(rejected);
    expect(rejectionState).toEqual({
      path: 'src/app.py',
      reason: 'source anchor mismatch',
    });
    expect(unresolved([
      ...rejected,
      {
        toolName: 'shell.exec',
        params: {
          command: 'ROY_VERIFIER_PROBE=1 python .roy/runtime/verifier_probe.py',
        },
        success: true,
        result: { stdout: 'VERIFIER_PROBE_MISMATCHES expected=A actual=B' },
      },
    ])).toBe(false);
  });

  it('does not route a local repair through web tools because verifier logs contain URLs', () => {
    const runtime = new Runtime();
    const needsWeb = (task: string) => (runtime as unknown as {
      taskNeedsWebAccess: (value: string) => boolean;
    }).taskNeedsWebAccess(task);

    expect(needsWeb([
      'Work directly in /app. Repair src/dq_audit/audit.py and run the local verifier.',
      '---',
      '## VERIFICATION FAILED — CONTINUE WORKING',
      '<official_verifier_feedback>',
      'WARNING: see https://docs.pytest.org/en/stable/how-to/capture-warnings.html',
      '</official_verifier_feedback>',
    ].join('\n'))).toBe(false);
    expect(needsWeb([
      'Repair the current workspace package and rerun its tests.',
      'Latest command output:',
      'WARNING: use a virtual environment: https://pip.pypa.io/warnings/venv',
    ].join('\n'))).toBe(false);
    expect(needsWeb([
      '[runtime_acceptance_audit_phase]',
      'Original task:',
      'Work directly in /app, implement the local data pipeline, and run its tests.',
      '<acceptance_checklist>',
      '- Verify the required local artifacts.',
      '</acceptance_checklist>',
      'Audit contract:',
      '- Cross-check independent configuration sources and verify local files.',
    ].join('\n'))).toBe(false);
    expect(needsWeb(
      'Use public web sources to compare the official Node.js and MDN documentation.'
    )).toBe(true);
    expect(needsWeb(
      'Read https://nodejs.org/api/globals.html and summarize the fetch section.'
    )).toBe(true);
    expect(needsWeb([
      'Repair the local document reconstruction pipeline and run its CLI.',
      'Official/public references:',
      '- OpenCV: https://opencv.org/',
      '- TableBank: https://github.com/doc-analysis/TableBank',
    ].join('\n'))).toBe(false);
  });

  it('keeps acceptance items focused on obligations instead of headings, inputs, and passive references', () => {
    const runtime = new Runtime();
    const extract = (task: string) => (runtime as unknown as {
      extractTaskAcceptanceItems: (value: string) => string[];
    }).extractTaskAcceptanceItems(task);

    const items = extract([
      'The input contains:',
      '- customers',
      '- orders',
      'Required outputs:',
      '- outputs/reconstructed_tables.csv',
      '- outputs/layout_qc.json',
      'The implementation must assign OCR tokens to cells by geometry.',
      'Official/public references:',
      '- OpenCV: https://opencv.org/',
      '- TableBank: https://github.com/doc-analysis/TableBank',
    ].join('\n'));

    expect(items).toEqual([
      'outputs/reconstructed_tables.csv',
      'outputs/layout_qc.json',
      'The implementation must assign OCR tokens to cells by geometry.',
    ]);
  });

  it('continues workspace evidence collection after a directory listing when the task requests file reads', () => {
    const runtime = new Runtime();
    const shouldReplan = (task: string, calls: Array<{ toolName: string }>) =>
      (runtime as unknown as {
        shouldReplanToolLoop: (
          value: string,
          completed: Array<{ toolName: string }>
        ) => boolean;
      }).shouldReplanToolLoop(task, calls);

    expect(shouldReplan(
      'List /app, then read the CSV inputs and rules/expectations.yml before reporting.',
      [{ toolName: 'fs.list' }]
    )).toBe(true);
    expect(shouldReplan(
      'List the workspace top-level entries.',
      [{ toolName: 'fs.list' }]
    )).toBe(false);
  });

  it('removes the tool-use skill from a delegation plan that has no tools', () => {
    const runtime = new Runtime();
    const normalized = (runtime as unknown as {
      normalizeDelegationAgentPlan: (
        plan: Record<string, unknown>,
        fallbackTask: string
      ) => { tools?: string[]; skills?: string[] };
    }).normalizeDelegationAgentPlan({
      archetype: 'custom',
      task: 'Answer the supplied trivia questions from model knowledge.',
      tools: [],
      skills: ['use_tool_when_needed'],
    }, 'Answer the question.');

    expect(normalized.tools).toBeUndefined();
    expect(normalized.skills).toBeUndefined();
  });

  it('marks delegated public-web fact verification as independent of shared cached evidence', () => {
    const runtime = new Runtime();
    const normalized = (runtime as unknown as {
      normalizeDelegationAgentPlan: (
        plan: Record<string, unknown>,
        fallbackTask: string
      ) => { task: string };
    }).normalizeDelegationAgentPlan({
      archetype: 'researcher',
      name: 'IndependentReviewer-1',
      role: 'factuality reviewer',
      task: 'Use public web sources to verify the precise canonical answers.',
      tools: ['web.search', 'web.fetch'],
    }, 'Verify the supplied claims.');

    expect(normalized.task).toContain('[runtime_independent_web_evidence]');
    expect(normalized.task).toContain('shared cached web results');
  });

  it('allows a semantic researcher to reason without pretending it has an external tool path', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-semantic-researcher-'));
    await writeFile(path.join(workspaceCwd, '.roy-config-placeholder'), '');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'semantic-researcher-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'SemanticResearcher-1',
      tomLevel: 0,
      description: 'Reason over the supplied word list only.',
      task: 'Infer which supplied words match the clue using only the prompt.',
      tools: [],
      skills: [],
      outputContract: { format: 'markdown', groundingRequired: false },
    });

    const result = await runtime.runAgent(
      researcher.identity.id,
      'Given only these words and the clue, rank the most likely matches.',
      { disableRecursiveDelegation: true, archetype: 'researcher' }
    );

    expect(result.toolCalls).toHaveLength(0);
    expect(result.grounded).toBe(true);
    expect(result.result).toBe('subagent result');
    expect(result.warnings).not.toContain(expect.stringContaining('no authorized tool call'));
    await runtime.shutdown();
  });

  it('rejects an implementation result when no grounded mutation-and-verification progress exists', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-open-execution-'));
    await writeFile(path.join(workspaceCwd, 'app.ts'), 'export const value = "stub";\n');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'open-execution-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const coder = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'coder',
      name: 'ClosureCoder-1',
      description: 'Implements and verifies one workspace source change.',
      task: 'Implement the workspace code in app.ts and verify the result.',
      tools: ['fs.list', 'fs.read', 'fs.search', 'fs.replace', 'fs.write', 'shell.exec'],
    });

    await expect(runtime.runAgent(
      coder.identity.id,
      'Implement the workspace code in app.ts and verify the result.',
      { disableRecursiveDelegation: true, archetype: 'coder' }
    )).rejects.toThrow('no grounded mutation-and-verification progress');
    expect(await readFile(path.join(workspaceCwd, 'app.ts'), 'utf8')).toBe(
      'export const value = "stub";\n'
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.execution.no_progress',
      agentId: coder.identity.id,
    }));
    expect(runtime.getEvents()).not.toContainEqual(expect.objectContaining({
      type: 'agent.run.completed',
      agentId: coder.identity.id,
    }));
    await runtime.shutdown();
  });

  it('resumes remaining evidence and carries saturation into closure model planning', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-saturated-closure-'));
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: { readOnly: 'auto', write: 'auto', execute: 'auto' },
        },
      })
    );
    await writeFile(path.join(workspaceCwd, 'app.ts'), 'export const value = "stub";\n');
    await writeFile(path.join(workspaceCwd, 'runtime.json'), '{"mode":"modern"}\n');
    const llm = new SaturatedClosurePlanningLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'saturated-closure-test',
      llmProvider: llm,
      workspaceCwd,
    });
    const task = [
      'app.ts: Fix the implementation using runtime.json while preserving its exported interface.',
      'Inspect the named runtime contract before editing.',
    ].join(' ');
    const coder = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'coder',
      name: 'SaturatedClosureCoder-1',
      description: task,
      task,
      tools: ['fs.list', 'fs.read', 'fs.replace'],
    });
    const priorToolCalls = [
      {
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
        success: true,
        result: { root: '.', maxDepth: 4, entries: ['app.ts'] },
      },
      {
        toolName: 'fs.read',
        params: { path: 'app.ts' },
        success: true,
        result: {
          path: 'app.ts',
          content: 'export const value = "stub";\n',
          truncated: false,
        },
      },
    ];

    const grounding = await (runtime as unknown as {
      runGroundingCheck: (
        agentId: string,
        assignedTask: string,
        options: {
          archetype: 'coder';
          intentTask: string;
          priorToolCalls: typeof priorToolCalls;
        }
      ) => Promise<RunAgentResult>;
    }).runGroundingCheck(coder.identity.id, task, {
      archetype: 'coder',
      intentTask: task,
      priorToolCalls,
    });

    expect(llm.sawSaturatedEvidence).toBe(true);
    expect(grounding.toolCalls).toContainEqual(expect.objectContaining({
      toolName: 'fs.read',
      params: { path: 'runtime.json' },
      success: true,
    }));
    expect(grounding.toolCalls).toContainEqual(expect.objectContaining({
      toolName: 'fs.replace',
      success: true,
    }));
    expect(await readFile(path.join(workspaceCwd, 'app.ts'), 'utf8')).toBe(
      'export const value = "implemented";\n'
    );
    await runtime.shutdown();
  });

  it('carries resumed causal evidence into an implementation closure continuation', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-closure-frontier-'));
    await writeFile(path.join(workspaceCwd, 'app.ts'), 'export const value = "stub";\n');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'closure-frontier-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const task = 'Repair app.ts and verify the implementation.';
    const coder = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'coder',
      name: 'ClosureFrontierCoder-1',
      description: task,
      task,
      tools: ['fs.read', 'fs.replace', 'shell.exec'],
    });
    const resumedDiagnostic = {
      toolName: 'shell.exec',
      params: { command: 'ROY_VERIFIER_PROBE=1 python -c "print(1)"' },
      success: true,
      result: { exitCode: 0, stdout: 'VERIFIER_PROBE_RESULT 0.5' },
    };
    let groundingRuns = 0;
    let continuationPriorCalls: Array<{ toolName: string; params: Record<string, unknown> }> = [];
    (
      runtime as unknown as {
        runGroundingCheck: (
          agentId: string,
          task: string,
          options: {
            priorToolCalls?: Array<{
              toolName: string;
              params: Record<string, unknown>;
            }>;
          }
        ) => Promise<Record<string, unknown>>;
      }
    ).runGroundingCheck = async (_agentId, _task, options) => {
      groundingRuns += 1;
      const toolCalls = groundingRuns === 1
        ? [{
          toolName: 'fs.read',
          params: { path: 'app.ts' },
          success: true,
          result: { path: 'app.ts', content: 'export const value = "stub";\n' },
        }]
        : [{
          toolName: 'fs.replace',
          params: { path: 'app.ts', oldText: '"stub"', newText: '"ready"' },
          success: true,
          result: { path: 'app.ts', replaced: true },
        }, {
          toolName: 'shell.exec',
          params: { command: 'npm test' },
          success: true,
          result: { command: 'npm test', exitCode: 0 },
        }];
      if (groundingRuns === 2) {
        continuationPriorCalls = options.priorToolCalls ?? [];
      }
      const now = Date.now();
      return {
        toolCalls,
        grounded: true,
        warnings: [],
        context: '',
        evidence: {
          toolGrounded: true,
          outputGrounded: true,
          observedPaths: ['app.ts'],
        },
        toolLoop: {
          rounds: [],
          totalCalls: toolCalls.length,
          successfulCalls: toolCalls.length,
          failedCalls: 0,
          stopReason: 'completed',
          startedAt: now,
          completedAt: now,
        },
      };
    };

    await runtime.runAgent(coder.identity.id, task, {
      disableRecursiveDelegation: true,
      archetype: 'coder',
      priorToolCalls: [resumedDiagnostic],
    });

    expect(groundingRuns).toBe(2);
    expect(continuationPriorCalls).toEqual(expect.arrayContaining([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: expect.objectContaining({
          command: expect.stringContaining('ROY_VERIFIER_PROBE=1'),
        }),
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        params: expect.objectContaining({ path: 'app.ts' }),
      }),
    ]));
    await runtime.shutdown();
  });

  it('does not demand a workspace mutation from a read-only evidence member whose task references the parent implementation', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-read-only-parent-task-'));
    await writeFile(path.join(workspaceCwd, 'app.ts'), 'export const value = "stub";\n');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'read-only-parent-task-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'PathSteward-1',
      description: 'Inspect the authoritative path for the parent implementation task.',
      task: [
        'Inspect app.ts and report its current state.',
        'Original task: Implement the workspace code in app.ts and verify the result.',
      ].join('\n'),
      tools: ['fs.read'],
    });

    await expect(runtime.runAgent(
      researcher.identity.id,
      [
        'Inspect app.ts and report its current state.',
        'Original task: Implement the workspace code in app.ts and verify the result.',
      ].join('\n'),
      { disableRecursiveDelegation: true, archetype: 'researcher' }
    )).resolves.toMatchObject({
      toolCalls: expect.arrayContaining([
        expect.objectContaining({ toolName: 'fs.read', success: true }),
      ]),
    });
    expect(runtime.getEvents()).not.toContainEqual(expect.objectContaining({
      type: 'agent.execution.no_progress',
      agentId: researcher.identity.id,
    }));
    await runtime.shutdown();
  });

  it('synthesizes an observed source file through a separate payload channel and verifies it', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-file-synthesis-'));
    await writeFile(path.join(workspaceCwd, 'app.js'), 'throw new Error("stub");\n');
    await writeFile(
      path.join(workspaceCwd, 'package.json'),
      JSON.stringify({
        name: 'runtime-file-synthesis-test',
        type: 'module',
        scripts: { test: 'node --check app.js' },
      })
    );
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: {
            write: 'auto',
            execute: 'auto',
          },
        },
      })
    );
    const llm = new FileSynthesisLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'file-synthesis-test',
      llmProvider: llm,
      workspaceCwd,
    });
    const coder = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'coder',
      name: 'SynthesisCoder-1',
      description: 'Implement one observed source file and verify it.',
      task: 'Inspect and implement app.js, then verify it with npm test.',
      tools: ['fs.read', 'fs.synthesize', 'shell.exec'],
    });

    const grounding = await (runtime as unknown as {
      runGroundingCheck: (
        agentId: string,
        task: string,
        options: Record<string, unknown>
      ) => Promise<{ toolCalls: Array<{ toolName: string; success: boolean }> }>;
    }).runGroundingCheck(
      coder.identity.id,
      'Inspect and implement app.js, then verify it with npm test.',
      { archetype: 'coder', correlationId: 'file-synthesis-correlation' }
    );

    expect(grounding.toolCalls).toEqual(expect.arrayContaining([
      expect.objectContaining({ toolName: 'fs.read', success: true }),
      expect.objectContaining({ toolName: 'fs.synthesize', success: true }),
      expect.objectContaining({ toolName: 'shell.exec', success: true }),
    ]));
    expect(await readFile(path.join(workspaceCwd, 'app.js'), 'utf8')).toBe(
      'export function value() { return 42; }\n'
    );
    const synthesisPrompt = llm.streamedPrompts.find(prompt =>
      prompt.includes('[runtime_workspace_file_synthesis]')
    ) ?? '';
    expect(synthesisPrompt).toContain('Current target snapshot');
    expect(synthesisPrompt).toContain('throw new Error("stub")');
    expect(synthesisPrompt.length).toBeLessThan(70_000);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'tool.synthesis.started',
      'tool.synthesis.completed',
    ]));
    const normalizePath = (runtime as unknown as {
      normalizeToolWorkspacePath: (value: string) => string;
    }).normalizeToolWorkspacePath;
    expect(normalizePath.call(runtime, path.join(workspaceCwd, 'app.js'))).toBe('app.js');
    const compactAssignment = (runtime as unknown as {
      compactFileSynthesisAssignment: (value: string) => string;
    }).compactFileSynthesisAssignment;
    const compacted = compactAssignment.call(runtime, [
      'Implement the immutable task.',
      '<team_step_cache>',
      'large repeated derived member context',
      '</team_step_cache>',
      'Preserve the acceptance criteria.',
    ].join('\n'));
    expect(compacted).toContain('Implement the immutable task.');
    expect(compacted).toContain('Preserve the acceptance criteria.');
    expect(compacted).not.toContain('derived member context');
    const compactEvidence = (runtime as unknown as {
      compactFileSynthesisEvidence: (
        calls: Array<Record<string, unknown>>,
        targetPath: string,
        maxChars: number
      ) => string;
    }).compactFileSynthesisEvidence;
    const verifierEvidence = compactEvidence.call(
      runtime,
      [{
        toolName: 'fs.read',
        params: { path: '.roy/official-verifier/grade.py' },
        success: true,
        result: {
          path: '.roy/official-verifier/grade.py',
          content: `${'head\n'.repeat(1_200)}UNIQUE_HIDDEN_ASSERTION\n${'tail\n'.repeat(1_200)}`,
          truncated: false,
        },
      }],
      'app.js',
      30_000
    );
    expect(verifierEvidence).toContain('UNIQUE_HIDDEN_ASSERTION');
    await runtime.shutdown();
  });

  it('applies a focused runtime-generated patch without rewriting working source', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-file-patch-'));
    await writeFile(
      path.join(workspaceCwd, 'app.js'),
      [
        'export function value() { return 41; }',
        'export function keep() { return "stable"; }',
        '',
      ].join('\n')
    );
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: {
            readOnly: 'auto',
            write: 'auto',
          },
        },
      })
    );
    const llm = new FocusedPatchSynthesisLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'file-patch-synthesis-test',
      llmProvider: llm,
      workspaceCwd,
    });
    const observed = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'app.js' },
      { correlationId: 'file-patch-synthesis-correlation' }
    );
    const patched = await runtime.executeToolForAgent(
      'root',
      'fs.synthesize',
      {
        path: 'app.js',
        instructions: 'Correct value while preserving the stable API.',
        strategy: 'patch',
      },
      {
        correlationId: 'file-patch-synthesis-correlation',
        synthesisTask: 'Repair app.js without rewriting working behavior.',
        groundingCalls: [{
          toolName: 'fs.read',
          params: { path: 'app.js' },
          reason: 'Observe the authoritative current source.',
          groundingRequired: true,
          success: observed.success,
          result: observed.result,
          error: observed.error,
        }],
      }
    );

    expect(patched).toMatchObject({
      success: true,
      result: expect.objectContaining({
        path: 'app.js',
        synthesized: true,
        strategy: 'patch',
      }),
    });
    expect(await readFile(path.join(workspaceCwd, 'app.js'), 'utf8')).toBe([
      'export function value() { return 42; }',
      'export function keep() { return "stable"; }',
      '',
    ].join('\n'));
    expect(llm.streamedPrompts).toContainEqual(
      expect.stringContaining('[runtime_workspace_file_patch_synthesis]')
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.synthesis.completed',
      data: expect.objectContaining({
        strategy: 'patch',
      }),
    }));
    await runtime.shutdown();
  });

  it('applies exact non-overlapping patch anchors despite stale line numbers and hunk order', async () => {
    const runtime = new Runtime();
    const applyPatch = (runtime as unknown as {
      applyUnifiedPatchToContent: (
        current: string,
        patch: string
      ) => { content?: string; error?: string };
    }).applyUnifiedPatchToContent.bind(runtime);
    const current = [
      'def earlier():',
      '    value = 1',
      '    return value',
      '',
      'def later():',
      '    flag = False',
      '    return flag',
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -40,3 +40,3 @@',
      ' def later():',
      '-    flag = False',
      '+    flag = True',
      '     return flag',
      '@@ -44,3 +44,3 @@',
      ' def earlier():',
      '-    value = 1',
      '+    value = 2',
      '     return value',
    ].join('\n');

    expect(applyPatch(current, patch)).toEqual({
      content: [
        'def earlier():',
        '    value = 2',
        '    return value',
        '',
        'def later():',
        '    flag = True',
        '    return flag',
        '',
      ].join('\n'),
    });
  });

  it('normalizes an unprefixed blank context line in a focused patch', () => {
    const runtime = new Runtime();
    const applyPatch = (runtime as unknown as {
      applyUnifiedPatchToContent: (
        current: string,
        patch: string
      ) => { content?: string; error?: string };
    }).applyUnifiedPatchToContent.bind(runtime);
    const current = [
      'def value():',
      '    answer = 41',
      '',
      '    return answer',
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -1,4 +1,4 @@',
      ' def value():',
      '-    answer = 41',
      '+    answer = 42',
      '',
      '     return answer',
    ].join('\n');

    expect(applyPatch(current, patch)).toEqual({
      content: [
        'def value():',
        '    answer = 42',
        '',
        '    return answer',
        '',
      ].join('\n'),
    });
  });

  it('applies a focused patch with stale edge context while preserving deletion anchors', () => {
    const runtime = new Runtime();
    const applyPatch = (runtime as unknown as {
      applyUnifiedPatchToContent: (
        current: string,
        patch: string
      ) => { content?: string; error?: string };
    }).applyUnifiedPatchToContent.bind(runtime);
    const current = [
      'def value():',
      '    answer = 41',
      '    return answer',
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -80,4 +80,4 @@',
      ' # stale generated edge context',
      ' def value():',
      '-    answer = 41',
      '+    answer = 42',
      '     return answer',
    ].join('\n');

    expect(applyPatch(current, patch)).toEqual({
      content: [
        'def value():',
        '    answer = 42',
        '    return answer',
        '',
      ].join('\n'),
    });
  });

  it('applies wider edge fuzz only around an exact unique deletion anchor', () => {
    const runtime = new Runtime();
    const applyPatch = (runtime as unknown as {
      applyUnifiedPatchToContent: (
        current: string,
        patch: string
      ) => { content?: string; error?: string };
    }).applyUnifiedPatchToContent.bind(runtime);
    const current = [
      'def unique_value():',
      '    answer = 41',
      '    return answer',
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -80,7 +80,7 @@',
      ...Array.from(
        { length: 18 },
        (_, index) => ` # stale generated edge ${index + 1}`
      ),
      '-    answer = 41',
      '+    answer = 42',
      '     return answer',
    ].join('\n');

    expect(applyPatch(current, patch)).toEqual({
      content: [
        'def unique_value():',
        '    answer = 42',
        '    return answer',
        '',
      ].join('\n'),
    });
  });

  it('recovers independently anchored change blocks separated by stale context', () => {
    const runtime = new Runtime();
    const applyPatch = (runtime as unknown as {
      applyUnifiedPatchToContent: (
        current: string,
        patch: string
      ) => { content?: string; error?: string };
    }).applyUnifiedPatchToContent.bind(runtime);
    const current = [
      'def first_value():',
      '    first = 41',
      '    return first',
      '',
      'def second_value():',
      '    second = False',
      '    return second',
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -80,9 +80,9 @@',
      ' def first_value():',
      '-    first = 41',
      '+    first = 42',
      ' # stale context omitted from the current snapshot',
      ' def second_value():',
      '-    second = False',
      '+    second = True',
      '     return second',
    ].join('\n');

    expect(applyPatch(current, patch)).toEqual({
      content: [
        'def first_value():',
        '    first = 42',
        '    return first',
        '',
        'def second_value():',
        '    second = True',
        '    return second',
        '',
      ].join('\n'),
    });
  });

  it('builds patch recovery context around exact rejected anchors', () => {
    const runtime = new Runtime();
    const focusedSnapshot = (runtime as unknown as {
      focusedPatchRecoverySnapshot: (
        current: string,
        patch: string,
        maxChars?: number
      ) => string;
    }).focusedPatchRecoverySnapshot.bind(runtime);
    const current = [
      ...Array.from({ length: 180 }, (_, index) => `before_${index} = ${index}`),
      'def repair_target():',
      '    answer = 41',
      '    return answer',
      ...Array.from({ length: 180 }, (_, index) => `after_${index} = ${index}`),
      '',
    ].join('\n');
    const patch = [
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -999,3 +999,3 @@',
      ' def repair_target():',
      '-    answer = 41',
      '+    answer = 42',
      '     return answer',
    ].join('\n');

    const focused = focusedSnapshot(current, patch, 3_000);
    expect(focused).toContain('def repair_target():');
    expect(focused).toContain('    answer = 41');
    expect(focused).not.toContain('before_0 = 0');
    expect(focused).not.toContain('after_179 = 179');
    expect(focused.length).toBeLessThanOrEqual(3_100);
  });

  it('extracts a unified diff from model prose and a trailing Markdown fence', () => {
    const runtime = new Runtime();
    const extract = (runtime as unknown as {
      extractUnifiedDiffPayload: (content: string) => string;
    }).extractUnifiedDiffPayload.bind(runtime);

    expect(extract([
      'Here is the focused repair:',
      '```diff',
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -1,1 +1,1 @@',
      '-value = 1',
      '+value = 2',
      '```',
      'This preserves the remaining behavior.',
    ].join('\n'))).toBe([
      '--- a/app.py',
      '+++ b/app.py',
      '@@ -1,1 +1,1 @@',
      '-value = 1',
      '+value = 2',
    ].join('\n'));
  });

  it('retries a rejected focused patch once with exact anchor feedback', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-file-patch-recovery-'));
    await writeFile(
      path.join(workspaceCwd, 'app.js'),
      [
        'export function value() { return 41; }',
        'export function keep() { return "stable"; }',
        '',
      ].join('\n')
    );
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: {
            readOnly: 'auto',
            write: 'auto',
          },
        },
      })
    );
    const llm = new RecoveringFocusedPatchSynthesisLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'file-patch-synthesis-recovery-test',
      llmProvider: llm,
      workspaceCwd,
    });
    const observed = await runtime.executeToolForAgent(
      'root',
      'fs.read',
      { path: 'app.js' },
      { correlationId: 'file-patch-synthesis-recovery-correlation' }
    );
    const patched = await runtime.executeToolForAgent(
      'root',
      'fs.synthesize',
      {
        path: 'app.js',
        instructions: 'Correct only the returned value.',
        strategy: 'patch',
      },
      {
        correlationId: 'file-patch-synthesis-recovery-correlation',
        synthesisTask: 'Repair app.js without rewriting working behavior.',
        groundingCalls: [{
          toolName: 'fs.read',
          params: { path: 'app.js' },
          reason: 'Observe the authoritative current source.',
          groundingRequired: true,
          success: observed.success,
          result: observed.result,
          error: observed.error,
        }],
      }
    );

    expect(patched).toMatchObject({
      success: true,
      result: expect.objectContaining({ strategy: 'patch' }),
    });
    expect(await readFile(path.join(workspaceCwd, 'app.js'), 'utf8')).toContain(
      'return 42'
    );
    expect(llm.streamedPrompts).toContainEqual(
      expect.stringContaining('[runtime_workspace_file_patch_recovery]')
    );
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.synthesis.retrying',
      data: expect.objectContaining({
        strategy: 'patch',
        reason: expect.stringContaining('anchor'),
      }),
    }));
    await runtime.shutdown();
  });

  it('keeps one workspace mutation hypothesis in a combined causal frontier', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-single-mutation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'single-mutation-hypothesis-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const prioritize = (runtime as unknown as {
      keepSingleWorkspaceMutationHypothesis: (
        plans: Array<{
          toolName: string;
          params: Record<string, unknown>;
          reason: string;
          groundingRequired: boolean;
        }>,
        agentId?: string,
        options?: { correlationId?: string }
      ) => Array<{ toolName: string; params: Record<string, unknown> }>;
    }).keepSingleWorkspaceMutationHypothesis.bind(runtime);

    const selected = prioritize([
      {
        toolName: 'fs.read',
        params: { path: 'src/app.py' },
        reason: 'Observe the target.',
        groundingRequired: true,
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app.py',
          strategy: 'patch',
          instructions: 'Apply the focused verifier repair.',
        },
        reason: 'Execute the causal repair hypothesis.',
        groundingRequired: true,
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app.py',
          instructions: 'Repair the implementation.',
        },
        reason: 'Generic model fallback.',
        groundingRequired: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        reason: 'Verify the selected repair.',
        groundingRequired: true,
      },
    ], 'root', { correlationId: 'single-mutation-hypothesis-correlation' });

    expect(selected).toEqual([
      expect.objectContaining({ toolName: 'fs.read' }),
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({ strategy: 'patch' }),
      }),
      expect.objectContaining({ toolName: 'shell.exec' }),
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.plan.mutation_hypothesis.deferred',
      data: expect.objectContaining({
        reason: 'verify_current_workspace_hypothesis_before_another_mutation',
        deferred: [
          expect.objectContaining({
            toolName: 'fs.synthesize',
            params: expect.not.objectContaining({ strategy: 'patch' }),
          }),
        ],
      }),
    }));
    await runtime.shutdown();
  });

  it('rejects tool-protocol markup from file synthesis without corrupting the existing source', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-rejected-synthesis-'));
    const original = 'def value():\n    return 41\n';
    await writeFile(path.join(workspaceCwd, 'app.py'), original);
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'rejected-synthesis-test',
      llmProvider: new ToolMarkupSynthesisLLM(),
      workspaceCwd,
    });
    const coder = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'coder',
      name: 'RejectedSynthesisCoder',
      description: 'Attempt one grounded synthesis while preserving the existing source on rejection.',
      task: 'Repair app.py using grounded evidence.',
      tools: ['fs.read', 'fs.synthesize'],
    });
    const executeSynthesis = (runtime as unknown as {
      executeSynthesizedFileForAgent: (
        agent: unknown,
        params: Record<string, unknown>,
        task: string,
        calls: Array<Record<string, unknown>>,
        correlationId: string
      ) => Promise<{
        success: boolean;
        error?: string;
        metadata?: Record<string, unknown>;
      }>;
    }).executeSynthesizedFileForAgent.bind(runtime);

    const result = await executeSynthesis(
      (runtime as unknown as {
        getContext: () => {
          manager: { getAgentById: (id: string) => unknown };
        };
      }).getContext().manager.getAgentById(coder.identity.id),
      {
        path: 'app.py',
        instructions: 'Repair the implementation without changing its public API.',
      },
      'Repair app.py using grounded evidence.',
      [{
        toolName: 'fs.read',
        params: { path: 'app.py' },
        success: true,
        result: { path: 'app.py', content: original },
      }],
      'rejected-synthesis-correlation'
    );

    expect(result).toMatchObject({
      success: false,
      error: expect.stringContaining('rejected before mutation'),
      metadata: expect.objectContaining({
        synthesisRejected: true,
        workspacePreserved: true,
      }),
    });
    expect(await readFile(path.join(workspaceCwd, 'app.py'), 'utf8')).toBe(original);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.synthesis.rejected',
      data: expect.objectContaining({ workspacePreserved: true }),
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.synthesis.retrying',
      data: expect.objectContaining({
        reason: expect.stringContaining('tool-protocol markup'),
        workspacePreserved: true,
      }),
    }));
    await runtime.shutdown();
  });

  it('keeps web tool enablement scoped to each runtime workspace', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-web-scope-'));
    const bootstrap = new Runtime();
    await bootstrap.initialize({
      sessionId: 'web-scope-bootstrap',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    expect(bootstrap.getAgentPolicy('root')?.tools.map(tool => tool.name)).toEqual(expect.arrayContaining([
      'web.search', 'web.fetch',
    ]));
    await bootstrap.shutdown();

    const configPath = path.join(workspaceCwd, '.roy', 'config.json');
    const workspaceConfig = JSON.parse(await readFile(configPath, 'utf8')) as { tools: { web: { enabled: boolean } } };
    workspaceConfig.tools.web.enabled = false;
    await writeFile(configPath, `${JSON.stringify(workspaceConfig, null, 2)}\n`, 'utf8');

    const disabled = new Runtime();
    await disabled.initialize({
      sessionId: 'web-scope-disabled',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    expect(disabled.getAgentPolicy('root')?.tools.map(tool => tool.name)).not.toContain('web.search');
    expect(disabled.getAgentPolicy('root')?.tools.map(tool => tool.name)).not.toContain('web.fetch');
    await expect(disabled.executeToolForAgent('root', 'web.fetch', { url: 'https://example.com' })).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining('not authorized'),
    });
    await disabled.shutdown();
  });

  it('shares successful idempotent web evidence across agents without repeating the request', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-shared-web-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'shared-web-cache-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const internal = runtime as unknown as {
      runtimeToolOverrides: Map<string, {
        name: string;
        description: string;
        execute: (params: Record<string, unknown>) => Promise<{
          success: boolean;
          result: unknown;
        }>;
      }>;
      workspaceRuntimeConfig: {
        tools: {
          executionLoop: {
            maxRounds: number;
            llmReplanning: boolean;
          };
        };
      };
      runGroundingCheck: (
        agentId: string,
        task: string,
        options: Record<string, unknown>
      ) => Promise<{
        toolCalls: Array<{
          toolName: string;
          success: boolean;
          result?: unknown;
        }>;
        context: string;
      }>;
    };
    internal.workspaceRuntimeConfig.tools.executionLoop.maxRounds = 1;
    internal.workspaceRuntimeConfig.tools.executionLoop.llmReplanning = false;
    let executions = 0;
    internal.runtimeToolOverrides.set('web.search', {
      name: 'web.search',
      description: 'Deterministic test web search.',
      execute: async params => {
        executions += 1;
        return {
          success: true,
          result: {
            query: params.query,
            provider: 'test',
            results: [{
              title: 'Hamlet',
              url: 'https://example.test/hamlet',
              snippet: 'Hamlet was written by William Shakespeare.',
            }],
          },
        };
      },
    });
    const task = 'Use web.search to answer: Who wrote Hamlet?';
    const first = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: task,
      task,
      tools: ['web.search'],
    });
    const second = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: task,
      task,
      tools: ['web.search'],
    });
    const firstRun = await internal.runGroundingCheck(first.identity.id, task, {
      archetype: 'researcher',
      correlationId: 'shared-web-first',
      initialPlans: [{
        toolName: 'web.search',
        params: { query: 'Who Wrote Hamlet?' },
        reason: 'Resolve the public fact.',
        groundingRequired: true,
      }],
      skipInitialModelPlanning: true,
    });
    const secondRun = await internal.runGroundingCheck(second.identity.id, task, {
      archetype: 'researcher',
      correlationId: 'shared-web-second',
      initialPlans: [{
        toolName: 'web.search',
        params: { query: ' who wrote hamlet ' },
        reason: 'Resolve the same public fact independently.',
        groundingRequired: true,
      }],
      skipInitialModelPlanning: true,
    });

    expect(executions).toBe(1);
    expect(firstRun.toolCalls).toEqual([
      expect.objectContaining({ toolName: 'web.search', success: true }),
    ]);
    expect(secondRun.toolCalls).toEqual([
      expect.objectContaining({ toolName: 'web.search', success: true }),
    ]);
    expect(secondRun.context).toContain('William Shakespeare');
    const fingerprint = (
      runtime as unknown as {
        toolPlanFingerprint: (
          plan: { toolName: string; params: Record<string, unknown> }
        ) => string;
      }
    ).toolPlanFingerprint.bind(runtime);
    expect(fingerprint({
      toolName: 'web.search',
      params: { query: '中国首都' },
    })).not.toBe(fingerprint({
      toolName: 'web.search',
      params: { query: '日本首都' },
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'tool.plan.shared_cache.reused',
      agentId: second.identity.id,
      data: expect.objectContaining({
        sourceAgentId: first.identity.id,
      }),
    }));
    const concurrentPlan = {
      archetype: 'researcher',
      initialPlans: [{
        toolName: 'web.search',
        params: { query: 'Who wrote Macbeth?' },
        reason: 'Resolve another public fact concurrently.',
        groundingRequired: true,
      }],
      skipInitialModelPlanning: true,
    };
    await Promise.all([
      internal.runGroundingCheck(first.identity.id, task, {
        ...concurrentPlan,
        correlationId: 'shared-web-concurrent-first',
      }),
      internal.runGroundingCheck(second.identity.id, task, {
        ...concurrentPlan,
        correlationId: 'shared-web-concurrent-second',
      }),
    ]);
    expect(executions).toBe(2);
    expect(runtime.getEvents().filter(event =>
      event.type === 'tool.plan.shared_cache.reused'
    )).toHaveLength(2);
    await runtime.shutdown();
  });

  it('requires a verification actor to diversify a shared web-search frontier', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-diversified-web-'));
    const llm = new DiversifyingWebLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'diversified-web-test',
      llmProvider: llm,
      workspaceCwd,
    });
    const internal = runtime as unknown as {
      runtimeToolOverrides: Map<string, {
        name: string;
        description: string;
        execute: (params: Record<string, unknown>) => Promise<{
          success: boolean;
          result: unknown;
        }>;
      }>;
      runGroundingCheck: (
        agentId: string,
        task: string,
        options: Record<string, unknown>
      ) => Promise<{
        toolCalls: Array<{
          toolName: string;
          success: boolean;
          params: Record<string, unknown>;
        }>;
      }>;
    };
    const queries: string[] = [];
    internal.runtimeToolOverrides.set('web.search', {
      name: 'web.search',
      description: 'Deterministic test web search.',
      execute: async params => {
        queries.push(String(params.query));
        return {
          success: true,
          result: {
            query: params.query,
            provider: 'test',
            results: [{
              title: 'Authorship evidence',
              url: 'https://example.test/authorship',
              snippet: 'The requested play is attributed to William Shakespeare.',
            }],
          },
        };
      },
    });
    const task = [
      'Use web.search to fact-check these precise canonical answers.',
      '1. Who wrote Hamlet?',
      '2. Who wrote Macbeth?',
      '[runtime_independent_web_evidence]',
    ].join('\n');
    const primary = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: task,
      task,
      tools: ['web.search'],
    });
    const verifier = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: task,
      task,
      tools: ['web.search'],
    });
    const initialPlans = [
      {
        toolName: 'web.search',
        params: { query: 'Who wrote Hamlet?' },
        reason: 'Resolve the first fact.',
        groundingRequired: true,
      },
      {
        toolName: 'web.search',
        params: { query: 'Who wrote Macbeth?' },
        reason: 'Resolve the second fact.',
        groundingRequired: true,
      },
    ];
    await internal.runGroundingCheck(primary.identity.id, task, {
      archetype: 'researcher',
      correlationId: 'diversified-web-primary',
      initialPlans,
      skipInitialModelPlanning: true,
    });
    const verified = await internal.runGroundingCheck(verifier.identity.id, task, {
      archetype: 'researcher',
      correlationId: 'diversified-web-verifier',
      initialPlans,
      skipInitialModelPlanning: true,
    });

    expect(llm.sawSharedFrontier).toBe(true);
    expect(queries).toEqual(expect.arrayContaining([
      'Who wrote Hamlet?',
      'Who wrote Macbeth?',
      'Hamlet authorship Shakespeare primary evidence',
      'Macbeth authorship Shakespeare primary evidence',
    ]));
    expect(verified.toolCalls.filter(call =>
      call.toolName === 'web.search'
    )).toHaveLength(4);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.web_evidence.diversification.required',
      agentId: verifier.identity.id,
    }));
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.web_evidence.diversification.completed',
      agentId: verifier.identity.id,
    }));
    await runtime.shutdown();
  });

  it('detects delegated references whose concrete parent objects were omitted', () => {
    const runtime = new Runtime();
    const internal = runtime as unknown as {
      delegatedTaskHasUnresolvedReference: (task: string) => boolean;
      completeDelegatedTaskReference: (
        task: string,
        parentId: string,
        correlationId: string
      ) => { task: string; source: string } | undefined;
      executionTrees: {
        begin: (input: {
          correlationId: string;
          sessionId: string;
          task: string;
          maxSteps: number;
        }) => unknown;
      };
    };
    const hasUnresolvedReference =
      internal.delegatedTaskHasUnresolvedReference.bind(runtime);

    expect(hasUnresolvedReference(
      'Research the precise canonical answer to each of the five trivia questions.'
    )).toBe(true);
    expect(hasUnresolvedReference([
      'Research these five questions:',
      '1. Who wrote Hamlet?',
      '2. Who wrote Macbeth?',
    ].join('\n'))).toBe(false);
    expect(hasUnresolvedReference(
      'Inspect all referenced files: src/a.ts and src/b.ts.'
    )).toBe(false);
    internal.executionTrees.begin({
      correlationId: 'referential-assignment',
      sessionId: 'test',
      task: [
        'Answer every question.',
        '1. Who wrote Hamlet?',
        '2. Who wrote Macbeth?',
      ].join('\n'),
      maxSteps: 4,
    });
    const completed = internal.completeDelegatedTaskReference.call(
      runtime,
      'Research each of the two questions.',
      'root',
      'referential-assignment'
    );
    expect(completed?.source).toBe('execution_tree');
    expect(completed?.task).toContain('[runtime_referenced_parent_assignment]');
    expect(completed?.task).toContain('Who wrote Hamlet?');
    expect(completed?.task).toContain('Who wrote Macbeth?');
  });

  it('does not impose a hidden lifetime cap on filesystem evidence calls', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-fs-call-cap-'));
    await writeFile(path.join(workspaceCwd, 'evidence.txt'), 'reusable evidence\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'fs-call-cap-test',
      llmProvider: new EchoLLM(),
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      tomLevel: 0,
      description: 'Inspect evidence repeatedly across a long execution path.',
      task: 'Inspect evidence repeatedly across a long execution path.',
      tools: ['fs.read'],
    });

    const outcomes = [];
    for (let index = 0; index < 25; index += 1) {
      outcomes.push(await runtime.executeToolForAgent(agent.identity.id, 'fs.read', {
        path: 'evidence.txt',
        startLine: 1,
        endLine: 1,
      }));
    }

    expect(outcomes.every(outcome => outcome.success)).toBe(true);
    expect(outcomes.some(outcome => outcome.error?.includes('Tool call limit reached'))).toBe(false);
    await runtime.shutdown();
  });

  it('renders an immutable assigned task once and sends only an observation reference', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-assignment-reference-'));
    const llm = new CapturingStreamLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'assignment-reference-test',
      llmProvider: llm,
      fsmEnabled: false,
      workspaceCwd,
    });
    const task = [
      'Analyze the distinctive zephyr ledger scenario.',
      'Return the relevant conclusion from the supplied assignment.',
    ].join(' ');
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      tomLevel: 0,
      description: task,
      task,
      tools: [],
      skills: [],
      outputContract: { format: 'markdown', groundingRequired: false },
    });

    await runtime.runAgent(agent.identity.id, task, {
      disableRecursiveDelegation: true,
      archetype: 'custom',
    });

    const messages = llm.streamedMessages.at(-1) ?? [];
    const rendered = messages.map(message => message.content).join('\n');
    expect(rendered.match(/distinctive zephyr ledger scenario/g)).toHaveLength(1);
    expect(messages.at(-1)?.content).toContain('[runtime_current_assignment]');
    expect(messages.at(-1)?.content).not.toContain('distinctive zephyr ledger scenario');
    await runtime.shutdown();
  });

  it('recovers an empty leaf-agent completion before marking the actor successful', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-empty-agent-output-'));
    const llm = new EmptyThenRecoveredLLM();
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'empty-agent-output-test',
      llmProvider: llm,
      fsmEnabled: false,
      workspaceCwd,
    });
    const task = 'Produce one complete bounded semantic analysis.';
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      tomLevel: 0,
      description: task,
      task,
      tools: [],
      skills: [],
      outputContract: { format: 'markdown', groundingRequired: false },
    });

    const result = await runtime.runAgent(agent.identity.id, task, {
      archetype: 'custom',
      disableRecursiveDelegation: true,
    });

    expect(result.result).toBe('Recovered complete visible result.');
    expect(result.agent.error).toBeUndefined();
    expect(llm.streamedPrompts).toHaveLength(2);
    expect(llm.streamedPrompts.at(-1)).toContain('[runtime_current_assignment]');
    expect(llm.streamedPrompts.at(-1)).toContain(task);
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'agent.output.empty.recovery.started',
      'agent.output.empty.recovery.completed',
      'agent.run.completed',
    ]));
    await runtime.shutdown();
  });

  it('repairs Markdown tool requests instead of presenting them as final output', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-markdown-tool-intent-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"tool-repair-test"}\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'markdown-tool-intent-test',
      llmProvider: new MarkdownToolIntentLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect package exports.', task: 'Inspect package exports.',
    });

    const result = await runtime.runAgent(agent.identity.id, 'Inspect package exports.', {
      archetype: 'researcher', disableRecursiveDelegation: true,
    });

    expect(result.result).not.toContain('```tool');
    expect(result.result).toContain('package.json');
    expect(runtime.getEvents().map(event => event.type)).toContain('agent.output.repair.completed');
    await runtime.shutdown();
  });

  it('executes an authorized unresolved tool intent before repairing the answer', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-tool-intent-recovery-'));
    await writeFile(path.join(workspaceCwd, 'evidence.txt'), 'runtime-grounded\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'tool-intent-recovery-test',
      llmProvider: new XmlToolIntentRecoveryLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      name: 'IntentRecovery-1',
      tomLevel: 0,
      description: 'Resolve the attached fact.',
      task: 'Resolve the attached fact.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(agent.identity.id, 'Resolve the attached fact.', {
      archetype: 'custom',
      disableRecursiveDelegation: true,
    });

    expect(result.toolCalls).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        success: true,
        params: { path: 'evidence.txt' },
      }),
    ]);
    expect(result.result).toContain('evidence.txt');
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'agent.output.tool_intent.recovery.started',
      'agent.output.tool_intent.recovery.completed',
    ]));
    expect(runtime.getEvents().map(event => event.type)).not.toContain('agent.output.repair.completed');
    await runtime.shutdown();
  });

  it('executes function/name/params tool markup before caching a member result', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-function-tool-intent-'));
    await writeFile(path.join(workspaceCwd, 'evidence.txt'), 'runtime-grounded\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'function-tool-intent-recovery-test',
      llmProvider: new FunctionXmlToolIntentRecoveryLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'FunctionIntentRecovery-1',
      tomLevel: 0,
      description: 'Inspect the remaining evidence and report it.',
      task: 'Inspect the remaining evidence and report it.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(
      agent.identity.id,
      'Inspect the remaining evidence and report it.',
      { archetype: 'researcher', disableRecursiveDelegation: true }
    );

    expect(result.toolCalls).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        success: true,
        params: { path: 'evidence.txt' },
      }),
    ]);
    expect(result.result).not.toContain('<function>');
    expect(result.result).toContain('evidence.txt');
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'agent.output.tool_intent.recovery.started',
      'agent.output.tool_intent.recovery.completed',
      'agent.run.completed',
    ]));
    await runtime.shutdown();
  });

  it('does not execute recovered tool markup with invalid required parameters', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-invalid-tool-intent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'invalid-tool-intent-recovery-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const extract = (runtime as unknown as {
      extractUnresolvedToolPlans: (
        agentId: string,
        result: string
      ) => Array<{ toolName: string; params: Record<string, unknown> }>;
    }).extractUnresolvedToolPlans.bind(runtime);

    expect(extract(
      'root',
      '<tool_call><tool_name>fs.read</tool_name></tool_call>'
    )).toEqual([]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.output.tool_intent.invalid',
      agentId: 'root',
      data: expect.objectContaining({
        toolName: 'fs.read',
        params: {},
        errors: expect.arrayContaining([
          expect.stringContaining('path must be a non-empty string'),
        ]),
      }),
    }));
    await runtime.shutdown();
  });

  it('executes native invoke/parameter XML even after earlier grounding calls', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-native-tool-intent-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"seed-grounded"}\n', 'utf8');
    await writeFile(path.join(workspaceCwd, 'evidence.txt'), 'runtime-grounded\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'native-tool-intent-recovery-test',
      llmProvider: new NativeXmlToolIntentRecoveryLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'custom',
      name: 'NativeIntentRecovery-1',
      tomLevel: 0,
      description: 'Inspect package.json, then resolve the attached fact.',
      task: 'Inspect package.json, then resolve the attached fact.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(
      agent.identity.id,
      'Inspect package.json, then resolve the attached fact.',
      { archetype: 'custom', disableRecursiveDelegation: true }
    );

    expect(result.toolCalls).toEqual(expect.arrayContaining([
      expect.objectContaining({
        toolName: 'fs.read',
        success: true,
        params: expect.objectContaining({ path: 'package.json' }),
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        success: true,
        params: { path: 'evidence.txt' },
      }),
    ]));
    expect(result.result).toContain('runtime-grounded');
    expect(runtime.getEvents().map(event => event.type)).toEqual(expect.arrayContaining([
      'agent.output.tool_intent.recovery.started',
      'agent.output.tool_intent.recovery.completed',
    ]));
    await runtime.shutdown();
  });

  it('rejects a model report that contradicts runtime filesystem evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-contradiction-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"typescript-project"}\n', 'utf8');
    await writeFile(path.join(workspaceCwd, 'index.ts'), 'export const value = 1;\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-contradiction-test',
      llmProvider: new ContradictoryArchitectureLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: 'Inspect repository architecture from filesystem evidence.',
      task: 'Inspect repository architecture from filesystem evidence.',
      outputContract: { format: 'markdown', groundingRequired: true },
    });
    const result = await runtime.runAgent(
      agent.identity.id,
      'Inspect repository architecture from filesystem evidence.',
      { archetype: 'researcher' }
    );

    expect(result.evidence.toolGrounded).toBe(true);
    expect(result.evidence.outputGrounded).toBe(false);
    expect(result.grounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('claims a Rust/Cargo project'));
    expect(runtime.getEvents().map(event => event.type)).toContain('agent.grounding.contradiction');

    await runtime.shutdown();
  });

  it('rejects multiple concrete project paths that are absent from runtime evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-path-contradiction-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"typescript-project"}\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-path-contradiction-test',
      llmProvider: new FabricatedPathsLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect repository architecture from filesystem evidence.',
      task: 'Inspect repository architecture from filesystem evidence.',
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(agent.identity.id, agent.identity.description ?? 'Inspect repository.', {
      archetype: 'researcher', disableRecursiveDelegation: true,
    });

    expect(result.grounded).toBe(false);
    expect(result.evidence.outputGrounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('src/fabricated/worker.ts'));
    await runtime.shutdown();
  });

  it('accepts absolute workspace paths when tools recorded their relative equivalents', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-absolute-path-'));
    await writeFile(path.join(workspaceCwd, 'package.json'), '{"name":"typescript-project"}\n', 'utf8');
    await writeFile(path.join(workspaceCwd, 'index.ts'), 'export const value = 1;\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-absolute-path-test',
      llmProvider: new AbsoluteObservedPathsLLM(workspaceCwd),
      fsmEnabled: false,
      workspaceCwd,
    });
    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: 'Inspect repository architecture from filesystem evidence.',
      task: 'Inspect repository architecture from filesystem evidence.',
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(
      agent.identity.id,
      agent.identity.description ?? 'Inspect repository.',
      { archetype: 'researcher', disableRecursiveDelegation: true }
    );

    expect(result.evidence.outputGrounded).toBe(true);
    expect(result.warnings).not.toContainEqual(
      expect.stringContaining('not present in runtime evidence')
    );
    await runtime.shutdown();
  });

  it('does not mark a grounding-required agent as grounded when no tool can be planned', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-grounding-required-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'grounding-required-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const agent = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'tester',
      tomLevel: 0,
      description: 'Verify behavior without an authorized tool.',
      task: 'Verify behavior against tests and failure cases.',
      tools: [],
      outputContract: { format: 'markdown', groundingRequired: true },
    });
    const result = await runtime.runAgent(
      agent.identity.id,
      'Verify behavior against tests and failure cases.',
      { archetype: 'tester', disableRecursiveDelegation: true }
    );

    expect(result.toolCalls).toHaveLength(0);
    expect(result.evidence).toMatchObject({ toolGrounded: false, outputGrounded: false });
    expect(result.grounded).toBe(false);
    expect(result.warnings).toContainEqual(expect.stringContaining('no authorized tool call'));

    await runtime.shutdown();
  });

  it('instruments a Python verifier to expose focused scorer inputs without deleting fixtures', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-python-verifier-probe-'));
    const verifierDirectory = path.join(workspaceCwd, '.roy', 'official-verifier');
    await mkdir(verifierDirectory, { recursive: true });
    await writeFile(
      path.join(verifierDirectory, 'grade.py'),
      [
        'import tempfile',
        'from pathlib import Path',
        '',
        'def reconstruction_fraction(actual, expected):',
        '    return 1.0 if actual == expected else 0.0',
        '',
        'def grade():',
        '    with tempfile.TemporaryDirectory(prefix="fixture_probe_") as directory:',
        '        Path(directory, "fixture.txt").write_text("retained", encoding="utf-8")',
        '        output = Path(directory, "outputs")',
        '        output.mkdir()',
        '        Path(output, "layout_qc.json").write_text(\'{"cropped_pages_detected": []}\', encoding="utf-8")',
        '        actual = {("public_doc", "table", 0, 0): "wrong"}',
        '        expected = [{"document_id": "public_doc", "table_id": "table", "row_index": 0, "col_index": 0, "text": "right"}]',
        '        public_result = reconstruction_fraction(actual, expected)',
        '        hidden_actual = {("hidden_doc", "table", 0, 0): "hidden_wrong"}',
        '        hidden_expected = [{"document_id": "hidden_doc", "table_id": "table", "row_index": 0, "col_index": 0, "text": "hidden_right"}]',
        '        reconstruction_fraction(hidden_actual, hidden_expected)',
        '        return public_result',
        '',
      ].join('\n'),
      'utf8'
    );
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          shell: {
            mode: 'unrestricted',
            shell: '/bin/sh',
          },
          approval: {
            execute: 'auto',
          },
        },
      }),
      'utf8'
    );
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'python-verifier-probe-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    await mkdir(path.join(workspaceCwd, 'data'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, 'data', 'tokens.json'),
      JSON.stringify({ tokens: [{ text: 'causal-input-token' }] }),
      'utf8'
    );
    await writeFile(
      path.join(workspaceCwd, 'data', 'metadata.json'),
      JSON.stringify({ source: 'workspace-root-relative-input' }),
      'utf8'
    );
    await writeFile(
      path.join(workspaceCwd, 'data', 'manifest.json'),
      JSON.stringify({
        token_file: 'tokens.json',
        metadata_file: 'data/metadata.json',
      }),
      'utf8'
    );
    await mkdir(path.join(workspaceCwd, 'outputs'), { recursive: true });
    await Promise.all(['a.json', 'b.json', 'c.json', 'd.json'].map(filename =>
      writeFile(
        path.join(workspaceCwd, 'outputs', filename),
        JSON.stringify({ generated: 'x'.repeat(3_900) }),
        'utf8'
      )
    ));
    const command = (runtime as unknown as {
      buildPythonVerifierDiagnosticCommand: (task?: string) => string;
    }).buildPythonVerifierDiagnosticCommand(
      [
        'Repair the implementation using data/manifest.json as the authoritative input.',
        'Expected artifacts: outputs/a.json outputs/b.json outputs/c.json outputs/d.json.',
        '<recovery_capsule>',
        JSON.stringify({ unresolvedGroups: ['G_public_reconstruction=0.500'] }),
        '</recovery_capsule>',
      ].join('\n')
    );
    const diagnosticSources = (
      runtime as unknown as {
        extractTaskDiagnosticSourcePaths: (task: string) => string[];
      }
    ).extractTaskDiagnosticSourcePaths(
      'Diagnose src/table_recon/audit.py against .roy/official-verifier/grade.py.'
    );
    expect(diagnosticSources).toEqual(['src/table_recon/audit.py']);
    const diagnosticScopes = (
      runtime as unknown as {
        extractVerifierDiagnosticScopes: (task: string) => string[];
      }
    ).extractVerifierDiagnosticScopes([
      '<recovery_capsule>',
      JSON.stringify({
        latestAcceptedScorecard: { reward: 0.998125 },
        unresolvedGroups: [
          'G_public_reconstruction=0.625',
        ],
      }),
      '</recovery_capsule>',
    ].join('\n'));
    expect(diagnosticScopes).toEqual(['public']);
    expect((
      runtime as unknown as {
        extractAcceptedVerifierReward: (task: string) => number | undefined;
      }
    ).extractAcceptedVerifierReward([
      '<recovery_capsule>',
      JSON.stringify({
        latestAcceptedScorecard: { reward: 0.998125 },
      }),
      '</recovery_capsule>',
    ].join('\n'))).toBe(0.998125);
    const probe = await runtime.executeToolForAgent(
      'root',
      'shell.exec',
      { command, timeoutMs: 30_000, maxOutputBytes: 24_000 },
      { correlationId: 'python-verifier-probe-correlation' }
    );
    const stdout = String((probe.result as { stdout?: unknown } | undefined)?.stdout ?? '');

    expect(probe.error).toBeUndefined();
    expect(probe).toMatchObject({ success: true });
    expect(stdout).toContain('VERIFIER_PROBE_EVIDENCE_VERSION 3');
    expect(stdout).toContain('VERIFIER_PROBE_CALL reconstruction_fraction');
    expect(stdout).toContain('"actual": "wrong"');
    expect(stdout).toContain('"expected": "right"');
    expect(stdout).toContain('"requested": ["public"]');
    expect(stdout
      .split('\n')
      .filter(line => line.startsWith('VERIFIER_PROBE_MISMATCHES '))
      .join('\n')).not.toContain('hidden_wrong');
    expect(stdout).toContain('VERIFIER_PROBE_ARTIFACT');
    expect(stdout).toContain('layout_qc.json');
    expect(stdout).toContain('cropped_pages_detected');
    expect(stdout).toContain('VERIFIER_PROBE_SPEC');
    expect(stdout).toContain('VERIFIER_PROBE_TASK_INPUT');
    expect(stdout).toContain('data/manifest.json');
    expect(stdout).toContain('data/tokens.json');
    expect(stdout).toContain('causal-input-token');
    expect(stdout).toContain('data/metadata.json');
    expect(stdout).toContain('workspace-root-relative-input');
    expect(stdout).toContain('VERIFIER_PROBE_RETAINED_DIRS');
    expect(stdout).toContain('VERIFIER_PROBE_MIRROR');
    const probeDirectories = await readdir(
      path.join(workspaceCwd, '.roy', 'diagnostics')
    );
    const mirror = probeDirectories.find(directory =>
      directory.startsWith('verifier-probe-')
    );
    expect(mirror).toBeDefined();
    expect(await readFile(
      path.join(
        workspaceCwd,
        '.roy',
        'diagnostics',
        mirror!,
        'retained-1',
        'fixture.txt'
      ),
      'utf8'
    )).toBe('retained');
    const compactProbe = (
      runtime as unknown as {
        compactVerifierProbeEvidenceText: (output: string, maxChars: number) => string;
      }
    ).compactVerifierProbeEvidenceText([
      'VERIFIER_PROBE_EVIDENCE_VERSION 2',
      `VERIFIER_PROBE_ARTIFACT ${JSON.stringify({
        path: 'hidden_input_manifest.json',
        content: 'x'.repeat(8_000),
      })}`,
      'VERIFIER_PROBE_TASK_INPUT {"path":"data/manifest.json","content":"{\\"token_file\\":\\"tokens.json\\"}"}',
      'VERIFIER_PROBE_SPEC {"artifact":"layout_qc.json","content":"qc.get(\\"cropped_pages_detected\\", 0) >= 1"}',
      'VERIFIER_PROBE_ARTIFACT {"path":"outputs/layout_qc.json","content":"{\\"cropped_pages_detected\\": []}"}',
      'VERIFIER_PROBE_REWARD 0.04',
    ].join('\n'), 2_000);
    expect(compactProbe).toContain('VERIFIER_PROBE_EVIDENCE_VERSION 2');
    expect(compactProbe).toContain('VERIFIER_PROBE_TASK_INPUT');
    expect(compactProbe).toContain('VERIFIER_PROBE_SPEC');
    expect(compactProbe).toContain('outputs/layout_qc.json');
    expect(compactProbe).not.toContain('x'.repeat(1_000));
    const relevanceCompactProbe = (
      runtime as unknown as {
        compactVerifierProbeEvidenceText: (output: string, maxChars: number) => string;
      }
    ).compactVerifierProbeEvidenceText([
      'VERIFIER_PROBE_EVIDENCE_VERSION 2',
      'VERIFIER_PROBE_MISMATCHES {"mismatch_count":1,"mismatches":[{"key":["public_region_status","status_counts",0,0],"expected":"Region","actual":"31"}]}',
      `VERIFIER_PROBE_TASK_INPUT ${JSON.stringify({
        path: 'data/public/public_invoice_alpha_ocr.json',
        content: `IRRELEVANT_GEOMETRY_${'i'.repeat(900)}`,
      })}`,
      `VERIFIER_PROBE_TASK_INPUT ${JSON.stringify({
        path: 'data/public/public_region_status_ocr.json',
        content: `RELEVANT_GEOMETRY_${'r'.repeat(900)}`,
      })}`,
      'VERIFIER_PROBE_REWARD 0.99',
    ].join('\n'), 1_600);
    expect(relevanceCompactProbe).toContain('RELEVANT_GEOMETRY');
    expect(relevanceCompactProbe).not.toContain('IRRELEVANT_GEOMETRY');
    await runtime.shutdown();
  });

  it('does not let cached teammate task markers change an agent immutable intent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-immutable-agent-intent-'));
    await writeFile(path.join(workspaceCwd, 'README.md'), '# Immutable assignment\n', 'utf8');
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'immutable-agent-intent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const assignedTask = 'Read README.md and report its current heading.';
    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 0,
      description: assignedTask,
      task: assignedTask,
      tools: ['fs.read', 'fs.search', 'shell.exec'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });
    const contaminatedObservation = [
      assignedTask,
      '<team_step_cache>',
      'Previous member assignment: [runtime_verifier_diagnostic_probe]',
      'Previous member ran a verifier probe.',
      '</team_step_cache>',
    ].join('\n');
    const result = await runtime.runAgent(researcher.identity.id, contaminatedObservation, {
      archetype: 'researcher',
      disableRecursiveDelegation: true,
    });

    expect(result.toolCalls).toContainEqual(expect.objectContaining({
      toolName: 'fs.read',
      params: expect.objectContaining({ path: 'README.md' }),
      success: true,
    }));
    expect(result.toolCalls.some(call =>
      call.toolName === 'shell.exec'
      && String(call.params.command ?? '').includes('ROY_VERIFIER_PROBE')
    )).toBe(false);
    expect(runtime.getEvents().some(event =>
      event.type.startsWith('agent.verifier_diagnostic.')
      && event.agentId === researcher.identity.id
    )).toBe(false);
    await runtime.shutdown();
  });

  it('requires a tester to rerun cached verification instead of accepting another actor result', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-fresh-tester-verification-'));
    await writeFile(path.join(workspaceCwd, 'app.js'), 'export const value = 1;\n', 'utf8');
    await writeFile(
      path.join(workspaceCwd, 'package.json'),
      JSON.stringify({
        name: 'fresh-tester-verification-test',
        type: 'module',
        scripts: { test: 'node --check app.js' },
      }),
      'utf8'
    );
    await mkdir(path.join(workspaceCwd, '.roy'), { recursive: true });
    await writeFile(
      path.join(workspaceCwd, '.roy', 'config.json'),
      JSON.stringify({
        tools: {
          approval: {
            execute: 'auto',
          },
        },
      }),
      'utf8'
    );
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'fresh-tester-verification-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });
    const task = [
      'Verify the actual workspace independently.',
      '## Required Verification Command',
      '```bash',
      'npm test',
      '```',
    ].join('\n');
    const tester = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'tester',
      tomLevel: 0,
      description: task,
      task,
      tools: ['shell.exec'],
      outputContract: { format: 'markdown', groundingRequired: true },
    });

    const result = await runtime.runAgent(tester.identity.id, task, {
      archetype: 'tester',
      disableRecursiveDelegation: true,
      priorToolCalls: [{
        toolName: 'shell.exec',
        params: { command: 'npm test' },
        success: true,
        result: { command: 'npm test', exitCode: 0 },
      }],
    });

    expect(result.toolCalls).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: expect.objectContaining({ command: 'npm test' }),
        success: true,
      }),
    ]);
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.verification.evidence.completed',
      agentId: tester.identity.id,
      data: expect.objectContaining({ attempts: 1, successful: 1 }),
    }));
    expect(result.result).toContain('[runtime_independent_verification_closure]');
    expect(runtime.getEvents()).toContainEqual(expect.objectContaining({
      type: 'agent.verification.summary.deterministic',
      agentId: tester.identity.id,
    }));
    expect(runtime.getEvents().some(event =>
      event.type === 'agent.llm.called'
      && event.agentId === tester.identity.id
    )).toBe(false);
    const authoritative = (
      runtime as unknown as {
        selectAuthoritativePriorVerification: (
          calls: RunAgentResult['toolCalls'],
          task?: string
        ) => RunAgentResult['toolCalls'][number] | undefined;
      }
    ).selectAuthoritativePriorVerification([
      {
        toolName: 'shell.exec',
        params: {
          command: 'python -m table_recon.cli run --manifest data/public/manifest.json --out-dir outputs',
        },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        result: {
          candidateRollback: {
            restored: true,
            reason: 'reward_regression',
          },
        },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'npm test' },
        success: true,
      },
    ]);
    expect(authoritative?.params.command).toBe('python .roy/official-verifier/grade.py');
    const taskAuthoritative = (
      runtime as unknown as {
        selectAuthoritativePriorVerification: (
          calls: RunAgentResult['toolCalls'],
          task?: string
        ) => RunAgentResult['toolCalls'][number] | undefined;
      }
    ).selectAuthoritativePriorVerification(
      [{
        toolName: 'shell.exec',
        params: {
          command: 'python -m table_recon.cli run --manifest data/public/manifest.json --out-dir outputs',
        },
        success: true,
      }],
      [
        'Public reproduction:',
        '```bash',
        'python -m table_recon.cli run --manifest data/public/manifest.json --out-dir outputs',
        '```',
        'Authoritative acceptance:',
        '```bash',
        'python .roy/official-verifier/grade.py',
        '```',
      ].join('\n')
    );
    expect(taskAuthoritative?.params.command).toBe(
      'python .roy/official-verifier/grade.py'
    );
    await runtime.shutdown();
  });

  it('requires fresh diagnosis after a verifier rolls back the current hypothesis', () => {
    const runtime = new Runtime();
    const latestRejected = (runtime as unknown as {
      latestRejectedVerifierCandidate: (
        calls: RunAgentResult['toolCalls']
      ) => Record<string, unknown> | undefined;
    }).latestRejectedVerifierCandidate.bind(runtime);
    const calls: RunAgentResult['toolCalls'] = [
      {
        toolName: 'fs.synthesize',
        params: { path: 'app.py', instructions: 'repair', strategy: 'patch' },
        result: { path: 'app.py', synthesized: true },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        result: {
          candidateRollback: {
            restored: true,
            path: 'app.py',
            reason: 'reward_regression',
            baselineReward: 0.8,
            candidateReward: 0.7,
          },
        },
        success: true,
      },
    ];

    expect(latestRejected(calls)).toEqual(expect.objectContaining({
      path: 'app.py',
      reason: 'reward_regression',
    }));
    expect(latestRejected([
      ...calls,
      {
        toolName: 'shell.exec',
        params: {
          command: 'ROY_VERIFIER_PROBE=1 python diagnostic.py',
        },
        result: { stdout: 'new causal evidence' },
        success: true,
      },
    ])).toBeUndefined();
  });

  it('passes the preceding sequential diagnosis into a focused repair plan', () => {
    const runtime = new Runtime();
    const enrich = (runtime as unknown as {
      enrichRepairPlansWithTeamStepEvidence: (
        plans: Array<{
          toolName: string;
          params: Record<string, unknown>;
          reason: string;
          groundingRequired: boolean;
        }>,
        task: string
      ) => Array<{
        toolName: string;
        params: Record<string, unknown>;
      }>;
    }).enrichRepairPlansWithTeamStepEvidence.bind(runtime);
    const diagnosis = [
      'The first grid boundary is a page-border decoy.',
      'Repair _assign_tokens_to_grid at its sufficient-lines branch.',
      'diagnostic-detail-'.repeat(220),
      'diagnostic-tail-anchor',
    ].join(' ');
    const task = [
      'Repair app.py.',
      '<team_step_cache>',
      JSON.stringify({
        completedMembers: [{
          agentId: 'agent_tester_001',
          result: diagnosis,
        }],
      }),
      '</team_step_cache>',
    ].join('\n');
    const [plan] = enrich([{
      toolName: 'fs.synthesize',
      params: {
        path: 'app.py',
        instructions: `Apply the smallest grounded repair. ${'baseline-contract-'.repeat(220)}`,
        strategy: 'patch',
      },
      reason: 'repair',
      groundingRequired: true,
    }], task);

    const instructions = String(plan?.params.instructions);
    expect(instructions).toContain('The first grid boundary is a page-border decoy.');
    expect(instructions).toContain('diagnostic-tail-anchor');
    expect(instructions).toContain(
      'immediately preceding sequential team member'
    );
    expect(instructions.length).toBeLessThanOrEqual(4_000);
  });

  it('spawns, registers, runs, and tracks a subagent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-subagent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'subagent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const spawned = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      tomLevel: 2,
      description: 'Inspect runtime state',
      task: 'Inspect runtime state',
      budgetTokens: 8000,
    });

    expect(spawned.identity.id).toBe('agent_researcher_001');
    expect(spawned.identity.parentId).toBe('root');

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].agent.identity.id).toBe(spawned.identity.id);

    const result = await runtime.runAgent(spawned.identity.id, 'Check token accounting');
    expect(result.result).toBe('subagent result');
    expect(result.usage.totalTokens).toBeGreaterThanOrEqual(13);

    const budget = runtime.getBudgetState();
    expect(budget.usedTokens).toBe(result.usage.totalTokens);
    expect(budget.perAgent[spawned.identity.id].totalTokens).toBe(result.usage.totalTokens);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.spawned');
    expect(eventTypes).toContain('budget.allocated');
    expect(eventTypes).toContain('agent.run.completed');

    await runtime.shutdown();
  });

  it('runs controlled spawn through root-mediated messages and synthesis', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-mediated-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'mediated-spawn-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });

    expect(result.correlationId).toMatch(/^del_/);
    expect(result.agent.identity.tomProfile.level).toBe(0);
    expect(result.subagentResult.grounded).toBe(true);
    expect(result.subagentResult.toolCalls.map(call => call.toolName)).toContain('fs.list');
    expect(result.finalResponse).toBe('subagent result');

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toEqual([
      'user.command.spawn',
      'agent.create.request',
      'budget.request',
      'budget.grant',
      'agent.create.approved',
      'agent.task',
      'tool.approval.request',
      'tool.approval.resolved',
      'tool.call',
      'tool.result',
      'agent.result',
      'root.synthesis',
      'budget.request',
      'budget.grant',
      'root.final_response',
    ]);

    const budget = runtime.getBudgetState();
    expect(budget.perAgent.root.totalTokens).toBe(9);
    expect(budget.perAgent[result.agent.identity.id].totalTokens).toBe(result.subagentResult.usage.totalTokens);

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');
    expect(eventTypes).toContain('agent.result.sent');
    expect(eventTypes).toContain('memory.pattern.updated');
    expect((await runtime.getConversation(undefined, 20)).some(entry => entry.role === 'agent')).toBe(true);
    const memoryState = await runtime.getMemoryState();
    expect(memoryState.agentMemories.map(memory => memory.id)).toContain('researcher');
    expect(memoryState.patterns.agents).toBe(1);
    expect(memoryState.patterns.delegations).toBe(1);
    const signals = await runtime.collectMemorySignals();
    expect(signals.counts.agentResults).toBe(1);
    expect(signals.candidateSignals).toContain('researcher.tool_policy');
    expect(signals.candidateSignals).toContain('public.project_structure');
    expect(signals.candidateSignals).toContain('roy.delegation_lesson');
    const proposals = await runtime.listMemoryProposals();
    expect(proposals.map(proposal => proposal.target.section)).toContain('tool-policy');
    expect(proposals.map(proposal => proposal.target.section)).toContain('project-structure');
    expect(proposals.map(proposal => proposal.target.section)).toContain('delegation-lessons');
    expect(proposals[0].id).toMatch(/^mem_prop_\d{17}_[a-f0-9]{4}$/);

    const prompt = await readFile(path.join(workspaceCwd, '.roy', 'agents', 'researcher', 'prompt.md'), 'utf8');
    expect(prompt).toContain('{{public_context}}');
    expect(prompt).toContain('{{agent_private_memory}}');
    expect(prompt).toContain('{{agent_identity}}');
    expect(prompt).toContain('{{tom_profile}}');
    expect(prompt).toContain('{{available_skills}}');
    expect(prompt).toContain('{{available_tools}}');
    expect(prompt).toContain('{{parent_context}}');
    expect(prompt).toContain('{{task}}');
    expect(result.subagentResult.evidence.toolGrounded).toBe(true);
    expect(result.subagentResult.evidence.outputGrounded).toBe(true);
    expect(result.subagentResult.result).toContain('## Runtime-Verified Evidence');
    expect(result.creationUsage.mode).toBe('generated');
    expect(result.creationUsage.definitionTokens).toBeGreaterThan(0);
    expect(result.creationUsage.renderedPromptTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('emits cache hits on repeated controlled spawn', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-cache-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'cache-hit-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });
    const second = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure again',
    });

    const hits = runtime.getEvents()
      .filter(event => event.type === 'cache.hit' && event.data?.correlationId === second.correlationId)
      .map(event => event.data?.patternId);
    expect(hits).toContain('agent_pattern_researcher_v1');
    expect(hits).toContain('delegation_project_inspection_researcher_v1');
    expect(second.creationUsage.cacheHits).toHaveLength(2);
    expect(second.creationUsage.mode).toBe('cache_hit');
    expect(second.creationUsage.definitionTokens).toBe(0);
    expect(second.creationUsage.renderedPromptTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('injects custom agent name and role into rendered prompts', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-custom-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'custom-agent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const rendered = await runtime.renderAgentPrompt({
      agentKey: 'custom',
      name: 'Singer-1',
      role: 'performer',
      task: 'Introduce yourself briefly.',
      archetype: 'custom',
    });

    expect(rendered.prompt).toContain('Singer-1');
    expect(rendered.prompt).toContain('performer');
    expect(rendered.prompt).toContain('Introduce yourself briefly.');
    expect(rendered.prompt.match(/Introduce yourself briefly\./g)).toHaveLength(1);
    expect(rendered.prompt.match(/<execution_knowledge>/g)).toHaveLength(1);
    expect(rendered.prompt.match(/<agent_memory_file/g)).toBeNull();

    await runtime.shutdown();
  });

  it('exposes built-in archetype skills, tools, and spawn policies', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-archetypes-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'archetype-policy-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const profiles = runtime.getAgentArchetypeProfiles();
    const researcher = profiles.find(profile => profile.archetype === 'researcher');
    const critic = profiles.find(profile => profile.archetype === 'critic');

    expect(researcher?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(researcher?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(critic?.tools.map(tool => tool.name)).toEqual(['fs.read', 'fs.search']);
    expect(critic?.skills.map(skill => skill.name)).toContain('delegate_to_subagent');
    expect(researcher?.spawnPolicy.maxChildren).toBe(5);
    expect(researcher?.spawnPolicy.maxDepth).toBe(3);

    await runtime.shutdown();
  });

  it('grants web tools only to an agent whose assigned task requires web evidence', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-web-capability-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'web-capability-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const localAgent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Inspect local project files.', task: 'Inspect local project files.',
    });
    const webAgent = await runtime.spawnAgent({
      parentId: 'root', archetype: 'researcher', tomLevel: 0,
      description: 'Search the web for the latest official Node.js documentation.',
      task: 'Search the web for the latest official Node.js documentation.',
    });

    expect(runtime.getAgentPolicy(localAgent.identity.id)?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(runtime.getAgentPolicy(webAgent.identity.id)?.tools.map(tool => tool.name)).toEqual([
      'fs.list', 'fs.read', 'fs.search', 'web.search', 'web.fetch',
    ]);
    await runtime.shutdown();
  });

  it('binds parent-approved tools and skills, and stores them in cache patterns', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-bindings-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'binding-cache-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'researcher',
      task: 'Inspect the project structure',
    });
    const policy = runtime.getAgentPolicy(result.agent.identity.id);
    expect(policy?.tools.map(tool => tool.name)).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(policy?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);

    const agentPatterns = await runtime.getCachePatterns('agents');
    const researcherPattern = agentPatterns.find(pattern => pattern.id === 'agent_pattern_researcher_v1');
    expect(researcherPattern?.tools).toEqual(['fs.list', 'fs.read', 'fs.search']);
    expect(researcherPattern?.skills).toEqual(['use_tool_when_needed', 'delegate_to_subagent']);
    expect(researcherPattern?.spawnPolicy).toMatchObject({
      maxChildren: 5,
      maxDepth: 3,
      budgetAware: true,
    });

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.create.requested');
    expect(eventTypes).toContain('spawn.policy.checked');
    expect(eventTypes).toContain('agent.create.approved');
    expect(eventTypes).toContain('agent.instance.created');
    expect(eventTypes).toContain('agent.tool.bound');
    expect(eventTypes).toContain('agent.skill.bound');

    await runtime.shutdown();
  });

  it('creates custom agents with custom identity, role, and explicit bindings', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-custom-spawn-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'custom-spawn-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const result = await runtime.handleSpawnCommand({
      archetype: 'custom',
      name: 'PromptAuditor-1',
      customRole: 'prompt inspector',
      task: 'Introduce yourself briefly.',
      tools: ['fs.read'],
      skills: ['use_tool_when_needed'],
    });

    expect(result.agent.identity.name).toBe('PromptAuditor-1');
    expect(result.agent.identity.description).toContain('Introduce yourself briefly.');
    const policy = runtime.getAgentPolicy(result.agent.identity.id);
    expect(policy?.tools.map(tool => tool.name)).toEqual(['fs.read']);
    expect(policy?.skills.map(skill => skill.name)).toEqual(['use_tool_when_needed']);

    const prompt = await readFile(path.join(workspaceCwd, '.roy', 'agents', 'promptauditor-1', 'prompt.md'), 'utf8');
    expect(prompt).toContain('{{agent_identity}}');

    await runtime.shutdown();
  });

  it('keeps a custom agent name from colliding with a built-in archetype pattern', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-pattern-namespace-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'pattern-namespace-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    await runtime.handleSpawnCommand({
      archetype: 'custom',
      name: 'Critic',
      task: 'Recommend semantic candidates from the supplied prompt.',
      tools: [],
      skills: [],
    });
    await expect(runtime.handleSpawnCommand({
      archetype: 'critic',
      name: 'ArchitectureCritic',
      task: 'Critique the supplied architecture evidence.',
    })).resolves.toBeDefined();

    const patterns = await runtime.getCachePatterns('agents');
    expect(patterns.map(pattern => pattern.id)).toEqual(expect.arrayContaining([
      'agent_pattern_custom-critic_v1',
      'agent_pattern_critic_v1',
    ]));
    await runtime.shutdown();
  });

  it('rejects the sixth direct child under the default parent child limit', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-child-limit-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'child-limit-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    for (let index = 1; index <= 5; index += 1) {
      await runtime.spawnAgent({
        parentId: 'root',
        archetype: 'researcher',
        name: `Researcher-${index}`,
        tomLevel: 0,
        description: `task ${index}`,
        task: `task ${index}`,
      });
    }

    await expect(runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-6',
      tomLevel: 0,
      description: 'task 6',
      task: 'task 6',
    })).rejects.toThrow('max_children_exceeded');

    const rejected = runtime.getEvents().find(event => event.type === 'spawn.policy.rejected');
    expect(rejected?.data?.reason).toBe('max_children_exceeded');
    expect(runtime.getChildren('root')).toHaveLength(5);

    await runtime.shutdown();
  });

  it('supports creating a subsubagent under a subagent parent', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-subsubagent-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'subsubagent-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const critic = await runtime.spawnAgent({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      tomLevel: 2,
      description: 'Review Researcher-1 output',
      task: 'Review Researcher-1 output',
    });

    const tree = runtime.getAgentTree();
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].agent.identity.id).toBe(researcher.identity.id);
    expect(tree.children[0].children).toHaveLength(1);
    expect(tree.children[0].children[0].agent.identity.id).toBe(critic.identity.id);

    await runtime.shutdown();
  });

  it('routes subsubagent results through parent synthesis before root final synthesis', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-parent-synthesis-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'parent-synthesis-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const result = await runtime.handleSpawnCommand({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      task: 'Review Researcher-1 output',
    });

    const messages = await runtime.getMessages({ correlationId: result.correlationId });
    expect(messages.map(message => message.kind)).toContain('agent.synthesis');
    const parentResult = messages.find(message => message.kind === 'agent.result' && message.from === researcher.identity.id && message.to === 'root');
    expect(parentResult).toBeDefined();

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('agent.synthesis.started');
    expect(eventTypes).toContain('agent.synthesis.completed');
    expect(eventTypes).toContain('root.synthesis.started');
    expect(eventTypes).toContain('root.synthesis.completed');

    const parentEvents = runtime.getEvents().filter(event => event.agentId === researcher.identity.id);
    expect(parentEvents.some(event => event.type === 'agent.fsm.state' && event.data?.state === 'S_synthesizing')).toBe(true);
    expect(runtime.getBudgetState().perAgent[researcher.identity.id].totalTokens).toBeGreaterThan(0);
    expect(result.finalResponse).toBe('subagent result');

    await runtime.shutdown();
  });

  it('lets a non-root agent recursively delegate to a direct child during its run', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-recursive-delegation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'recursive-delegation-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const result = await runtime.runAgent(
      researcher.identity.id,
      'Review project risks and grounding gaps with a direct child critic.',
      { correlationId: 'del_recursive_test', archetype: 'researcher' }
    );

    const tree = runtime.getAgentTree();
    expect(tree.children[0].agent.identity.id).toBe(researcher.identity.id);
    expect(tree.children[0].children).toHaveLength(3);
    expect(tree.children[0].children[0].agent.identity.id).toBe('agent_critic_002');
    expect(tree.children[0].children.map(child => child.agent.identity.tomProfile.cognitiveGaps).flat().length).toBeGreaterThan(0);

    expect(result.agent.identity.id).toBe(researcher.identity.id);
    expect(result.result).toBe('subagent result');
    expect(result.usage.totalTokens).toBeGreaterThan(0);

    const messages = await runtime.getMessages({ correlationId: 'del_recursive_test' });
    expect(messages.map(message => message.kind)).toContain('agent.create.request');
    expect(messages.map(message => message.kind)).toContain('agent.task');
    expect(messages.map(message => message.kind)).toContain('agent.result');
    expect(messages.map(message => message.kind)).toContain('agent.synthesis');

    const eventTypes = runtime.getEvents().map(event => event.type);
    expect(eventTypes).toContain('delegation.decision');
    expect(eventTypes).toContain('delegation.plan.created');
    expect(eventTypes).toContain('delegation.completed');
    expect(eventTypes).toContain('agent.synthesis.completed');
    expect(runtime.getEvents().some(event =>
      event.type === 'budget.rebalanced'
      && event.agentId === researcher.identity.id
      && event.data?.purpose === 'agent.multi_child_synthesis'
    )).toBe(true);
    expect(runtime.getEvents().some(event =>
      event.type === 'budget.context.truncated'
      && event.agentId === researcher.identity.id
      && event.data?.purpose === 'agent.multi_child_synthesis'
    )).toBe(false);
    expect(runtime.getBudgetState().mode).toBe('unlimited');

    await runtime.shutdown();
  });

  it('does not treat inherited parent delegation guidance as a child assignment', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-inherited-delegation-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'inherited-delegation-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Bounded Researcher',
      tomLevel: 0,
      description: 'Collect evidence for two facts.',
      task: 'Collect evidence for two facts.',
    });
    await runtime.runAgent(
      researcher.identity.id,
      [
        'Collect evidence for two facts and return a bounded report.',
        'Parent task: Use normal delegation judgment and delegate factual research when useful.',
      ].join('\n'),
      { correlationId: 'del_inherited_guidance', archetype: 'researcher' }
    );

    expect(runtime.getAgentTree().children[0].children).toHaveLength(0);
    expect(runtime.getEvents().some(event =>
      event.type === 'agent.delegation.assessment.skipped'
      && event.agentId === researcher.identity.id
    )).toBe(true);
    expect(runtime.getEvents().some(event =>
      event.type === 'delegation.assess.started'
      && event.agentId === researcher.identity.id
    )).toBe(false);

    await runtime.shutdown();
  });

  it('lets a non-root parent aggregate multiple direct children', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-multi-child-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'multi-child-delegation-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const result = await runtime.runAgent(
      researcher.identity.id,
      'Delegate project risk review and test-coverage verification to direct children, then aggregate them.',
      { correlationId: 'del_multi_child_test', archetype: 'researcher' }
    );

    const children = runtime.getChildren(researcher.identity.id);
    expect(children).toHaveLength(3);
    expect(children.map(child => child.identity.name)).toEqual(expect.arrayContaining([
      expect.stringMatching(/^Critic-/),
      expect.stringMatching(/^Tester-/),
      expect.stringMatching(/^Researcher-/),
    ]));
    expect(result.result).toBe('subagent result');

    const messages = await runtime.getMessages({ correlationId: 'del_multi_child_test' });
    const team = runtime.getTeams()[0];
    expect(messages.filter(message => message.kind === 'agent.task' && message.from === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'agent.result' && message.to === team.identity.id)).toHaveLength(3);
    expect(messages.filter(message => message.kind === 'team.result' && message.to === researcher.identity.id)).toHaveLength(1);
    expect(messages.filter(message => message.kind === 'agent.synthesis' && message.from === researcher.identity.id)).toHaveLength(1);

    const synthesisEvent = runtime.getEvents().find(event => event.type === 'agent.synthesis.completed' && event.agentId === researcher.identity.id);
    expect(synthesisEvent?.data?.childIds).toEqual(children.map(child => child.identity.id));
    expect(runtime.getBudgetState().perAgent[researcher.identity.id].totalTokens).toBeGreaterThan(0);

    await runtime.shutdown();
  });

  it('rejects child creation when the parent is failed', async () => {
    const workspaceCwd = await mkdtemp(path.join(tmpdir(), 'roy-runtime-invalid-fsm-'));
    const runtime = new Runtime();
    await runtime.initialize({
      sessionId: 'invalid-fsm-test',
      llmProvider: new EchoLLM(),
      fsmEnabled: false,
      workspaceCwd,
    });

    const researcher = await runtime.spawnAgent({
      parentId: 'root',
      archetype: 'researcher',
      name: 'Researcher-1',
      tomLevel: 0,
      description: 'Inspect project',
      task: 'Inspect project',
    });
    const agent = runtime.getContext().manager.getAgentById(researcher.identity.id);
    agent?.setRuntimeState('failed');

    await expect(runtime.spawnAgent({
      parentId: researcher.identity.id,
      archetype: 'critic',
      name: 'Critic-1',
      tomLevel: 2,
      description: 'Review failed researcher',
      task: 'Review failed researcher',
    })).rejects.toThrow('invalid_fsm_state');

    const rejected = runtime.getEvents().find(event => event.type === 'delegation.rejected');
    expect(rejected?.data?.reason).toBe('invalid_fsm_state');

    await runtime.shutdown();
  });
});
