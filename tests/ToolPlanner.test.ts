import { describe, expect, it } from 'vitest';
import { AgentToolPlanner } from '../src/core/tools/planner.js';

describe('AgentToolPlanner', () => {
  it('reads package.json when a package export inspection needs manifest evidence', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect this package exports and identify one concrete architecture risk.',
      workspacePath: '/workspace',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans.map(plan => plan.toolName)).toEqual(['fs.list', 'fs.read']);
    expect(plans[1].params).toEqual({ path: 'package.json' });
    expect(plans.every(plan => plan.groundingRequired)).toBe(true);
  });

  it('does not guess an npm verification command without observed project metadata', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Verify the claims against tests and failure cases.',
      workspacePath: '/workspace',
      archetype: 'tester',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    });

    expect(plans).toEqual([]);
  });

  it('honors an explicitly requested npm verification command', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Run npm test and report the real exit status.',
      workspacePath: '/workspace',
      archetype: 'tester',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'shell.exec', params: { command: 'npm test' }, groundingRequired: true }),
    ]);
  });

  it('executes a task-declared required CLI command before optional file reads', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Implement the workspace pipeline.',
        '## Required Command',
        '```bash',
        'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        '```',
        'Inspect configs/public_audit.yml when repairing failures.',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'coder',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list' }),
      expect.objectContaining({
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'configs/public_audit.yml' },
      }),
    ]);
  });

  it('does not execute explanatory prose found in a shell-styled task block', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        '## Required verification notes',
        '```bash',
        'The verifier independently reruns validation on the cleaned data.',
        '```',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'tester',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([]);
  });

  it('reruns a task-declared acceptance command after every newer mutation', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Implement the workspace pipeline.',
      '## Required Command',
      '```bash',
      'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
      '```',
    ].join('\n');
    const calls = [
      {
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
        success: false,
      },
      {
        toolName: 'fs.synthesize',
        params: { path: 'src/dq_audit/audit.py', instructions: 'Implement it.' },
        success: true,
      },
    ];

    expect(planner.planPostMutationVerification({
      task,
      calls,
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
      }),
    ]);

    expect(planner.planPostMutationVerification({
      task,
      calls: [
        ...calls,
        {
          toolName: 'shell.exec',
          params: {
            command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
          },
          success: true,
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([]);
  });

  it('does not reinstall an already prepared environment after source-only mutations', () => {
    const planner = new AgentToolPlanner();
    const task = [
      '## Required Commands',
      '```bash',
      'python -m pip install -e .',
      'python -m support_rag.cli answer --question "hello"',
      '```',
    ].join('\n');

    const plans = planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install -e .' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli answer --question "hello"' },
          success: false,
        },
        {
          toolName: 'fs.synthesize',
          params: { path: 'src/support_rag/chain.py', instructions: 'Migrate imports.' },
          success: true,
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: {
          command: 'python -m support_rag.cli answer --question "hello"',
        },
      }),
    ]);
  });

  it('invalidates a successful dependency install when its manifest changes afterward', () => {
    const planner = new AgentToolPlanner();
    const task = [
      '## Required Commands',
      '```bash',
      'cd /app',
      'python -m pip install -e .',
      'python -m support_rag.cli answer --question "hello"',
      '```',
    ].join('\n');

    const plans = planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'shell.exec',
          params: { command: 'cd /app' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install -e .' },
          success: true,
        },
        {
          toolName: 'fs.synthesize',
          params: {
            path: 'pyproject.toml',
            instructions: 'Upgrade dependencies.',
            strategy: 'patch',
          },
          success: true,
          result: { path: 'pyproject.toml' },
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: { command: 'python -m pip install -e .' },
        reason: expect.stringContaining('Refresh the active environment'),
      }),
    ]);

    const afterInstall = planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'fs.synthesize',
          params: {
            path: 'pyproject.toml',
            instructions: 'Upgrade dependencies.',
            strategy: 'patch',
          },
          success: true,
          result: { path: 'pyproject.toml' },
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install -e .' },
          success: true,
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    }).map(plan => plan.params.command);
    expect(afterInstall).not.toContain('python -m pip install -e .');
    expect(afterInstall).toContain(
      'python -m support_rag.cli answer --question "hello"'
    );
  });

  it('retains acceptance commands beyond the initial planning batch', () => {
    const planner = new AgentToolPlanner();
    const task = [
      '## Helpful Commands',
      '```bash',
      'cd /app',
      'python -m pip install -e .',
      'python -m support_rag.cli answer --question "hello"',
      'python -m support_rag.cli route --ticket "invoice"',
      'python -m support_rag.cli replay --history data/history',
      'python -m pytest -q',
      '```',
    ].join('\n');

    const plans = planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'shell.exec',
          params: { command: 'cd /app' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install -e .' },
          success: true,
        },
        {
          toolName: 'fs.synthesize',
          params: { path: 'src/support_rag/chain.py', instructions: 'Migrate it.' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli answer --question "hello"' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli route --ticket "invoice"' },
          success: true,
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans.map(plan => plan.params.command)).toEqual([
      'python -m support_rag.cli replay --history data/history',
    ]);

    const afterReplay = planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'shell.exec',
          params: { command: 'cd /app' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install -e .' },
          success: true,
        },
        {
          toolName: 'fs.synthesize',
          params: { path: 'src/support_rag/chain.py', instructions: 'Migrate it.' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli answer --question "hello"' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli route --ticket "invoice"' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli replay --history data/history' },
          success: true,
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    });
    expect(afterReplay.map(plan => plan.params.command)).toEqual([
      'python -m pytest -q',
    ]);
  });

  it('does not repair source code from a pytest no-tests-collected exit', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Implement the requested behavior.',
      '```bash',
      'python -m app.cli smoke',
      'python -m pytest -q',
      '```',
    ].join('\n');
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/cli.py',
          instructions: 'Implement the behavior.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/app/cli.py' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m app.cli smoke' },
        success: true,
        result: { exitCode: 0, stdout: 'ok' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: { exitCode: 5, stdout: 'no tests ran in 0.01s' },
      },
    ];

    expect(planner.planPostMutationVerification({
      task,
      calls,
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([]);
    expect(planner.planWorkspaceRepairTransition({
      task,
      calls,
      bindings: [{ name: 'fs.synthesize', enabled: true }],
    })).toEqual([]);
  });

  it('stops the post-mutation verification frontier at the first failure', () => {
    const planner = new AgentToolPlanner();
    const task = [
      '## Helpful Commands',
      '```bash',
      'python -m app.cli answer',
      'python -m app.cli replay',
      'python -m pytest -q',
      '```',
    ].join('\n');

    expect(planner.planPostMutationVerification({
      task,
      calls: [
        {
          toolName: 'fs.synthesize',
          params: {
            path: 'src/app/chain.py',
            instructions: 'Migrate the chain.',
            strategy: 'patch',
          },
          success: true,
          result: { path: 'src/app/chain.py' },
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m app.cli answer' },
          success: false,
          result: {
            exitCode: 1,
            stderr: 'src/app/chain.py:12: import failed',
          },
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([]);
  });

  it('retries a failed acceptance command after its missing runtime tool is installed', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Modify src/app/chain.py and verify it.',
      '```bash',
      'python -m pytest -q',
      '```',
    ].join('\n');
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/chain.py',
          instructions: 'Migrate the chain.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/app/chain.py' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          exitCode: 1,
          stderr: '/usr/local/bin/python: No module named pytest',
        },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pip install pytest' },
        success: true,
        result: { exitCode: 0 },
      },
    ];

    expect(planner.planPostMutationVerification({
      task,
      calls,
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([
      expect.objectContaining({
        params: { command: 'python -m pytest -q' },
        reason: expect.stringContaining('missing runtime tool was just installed'),
      }),
    ]);

    expect(planner.planPostMutationVerification({
      task,
      calls: [
        ...calls,
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pytest -q' },
          success: false,
          result: {
            exitCode: 1,
            stderr: 'src/app/chain.py:12: assertion failed',
          },
        },
      ],
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([]);
  });

  it('turns external dependency feedback into ordered manifest repairs', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Continue the workspace migration.',
      '## VERIFICATION FAILED — CONTINUE WORKING',
      '<official_verifier_feedback>',
      'project metadata still targets the legacy runtime dependency version',
      '</official_verifier_feedback>',
    ].join('\n');
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
    ];
    const calls = [
      {
        toolName: 'fs.read',
        params: { path: '.roy/official-verifier/test_outputs.py' },
        success: true,
        result: {
          path: '.roy/official-verifier/test_outputs.py',
          content: 'raise AssertionError("project metadata still targets the legacy runtime dependency version")',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'requirements.txt' },
        success: true,
        result: {
          path: 'requirements.txt',
          content: 'runtime==0.0.1\ncompat<2\n',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'pyproject.toml' },
        success: true,
        result: {
          path: 'pyproject.toml',
          content: '[project]\ndependencies = ["runtime>=0.3"]\n',
        },
      },
    ];

    const first = planner.planExternalFeedbackRepair({
      task,
      calls,
      bindings,
      workspaceRoot: '/app',
    });
    expect(first).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'requirements.txt',
          strategy: 'patch',
          instructions: expect.stringContaining(
            'project metadata still targets the legacy runtime dependency version'
          ),
        }),
      }),
    ]);

    const priorTurnMutation = {
      toolName: 'fs.synthesize',
      params: first[0]!.params,
      success: true,
    };
    expect(planner.planExternalFeedbackRepair({
      task,
      calls: [...calls, priorTurnMutation],
      currentCalls: [],
      bindings,
      workspaceRoot: '/app',
    })).toEqual([
      expect.objectContaining({
        params: expect.objectContaining({ path: 'requirements.txt' }),
      }),
    ]);

    const second = planner.planExternalFeedbackRepair({
      task,
      calls: [
        ...calls,
        priorTurnMutation,
      ],
      currentCalls: [priorTurnMutation],
      bindings,
      workspaceRoot: '/app',
    });
    expect(second).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({ path: 'pyproject.toml' }),
      }),
    ]);

    expect(planner.planExternalFeedbackRepair({
      task,
      calls: [
        {
          toolName: 'fs.list',
          params: { path: '.', maxDepth: 3 },
          success: true,
          result: { entries: ['pyproject.toml', 'requirements.txt'] },
        },
        calls[2]!,
      ],
      currentCalls: [],
      bindings,
      workspaceRoot: '/app',
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'requirements.txt' },
      }),
    ]);
  });

  it('lets the newest recovery capsule supersede stale dependency feedback', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Migrate the application dependencies and runtime.',
      '<official_verifier_feedback>',
      'project metadata still targets the legacy runtime dependency version',
      '</official_verifier_feedback>',
      '<recovery_capsule>',
      JSON.stringify({
        recoveryTrigger: 'new_external_verifier_feedback',
        externalFeedback: {
          summary: 'router structure violation: classify_ticket is not registered with @tool',
        },
        authoritativeVerifierCommand:
          'python -m pytest -q .roy/official-verifier/test_outputs.py',
      }),
      '</recovery_capsule>',
      'Read `.roy/official-verifier/test_outputs.py` and repair the current source.',
    ].join('\n');
    const bindings = [
      { name: 'fs.list', enabled: true },
      { name: 'fs.read', enabled: true },
      { name: 'fs.search', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];

    expect(planner.plan({
      task,
      workspacePath: '/app',
      bindings,
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.search',
        params: {
          path: '.',
          query: 'classify_ticket',
          maxResults: 20,
        },
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: '.roy/official-verifier/test_outputs.py' },
      }),
    ]);

    const plans = planner.planExternalFeedbackRepair({
      task,
      bindings,
      workspaceRoot: '/app',
      calls: [
        {
          toolName: 'fs.read',
          params: { path: 'requirements.txt' },
          success: true,
          result: {
            path: 'requirements.txt',
            content: 'langchain==1.3.4\n',
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/support_rag/router.py' },
          success: true,
          result: {
            path: 'src/support_rag/router.py',
            content: 'def classify_ticket(ticket: str) -> str:\n    return "billing"\n',
          },
        },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/router.py',
          strategy: 'patch',
          instructions: expect.stringContaining(
            'classify_ticket is not registered with @tool'
          ),
        }),
      }),
    ]);

    const nextPlans = planner.planExternalFeedbackRepair({
      task,
      bindings,
      workspaceRoot: '/app',
      calls: [
        {
          toolName: 'fs.read',
          params: { path: 'src/support_rag/router.py' },
          success: true,
          result: {
            path: 'src/support_rag/router.py',
            content: '@tool\ndef classify_ticket(ticket: str) -> str:\n    return "billing"\n',
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/support_rag/retriever.py' },
          success: true,
          result: {
            path: 'src/support_rag/retriever.py',
            content: 'class FAQRetriever:\n    def search(self, query: str):\n        return []\n',
          },
        },
        {
          toolName: 'shell.exec',
          params: {
            command: 'python -m pytest -q .roy/official-verifier/test_outputs.py',
          },
          success: false,
          result: {
            command: 'python -m pytest -q .roy/official-verifier/test_outputs.py',
            exitCode: 1,
            stdout:
              "AttributeError: 'FAQRetriever' object has no attribute '_get_relevant_documents'",
          },
        },
      ],
    });

    expect(nextPlans).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/retriever.py',
          instructions: expect.stringContaining(
            "FAQRetriever' object has no attribute '_get_relevant_documents"
          ),
        }),
      }),
    ]);
  });

  it('repairs implementation code instead of benchmark config named by verifier feedback', () => {
    const planner = new AgentToolPlanner();
    const plans = planner.planExternalFeedbackRepair({
      task: [
        'Continue repairing the implementation.',
        '## VERIFICATION FAILED — CONTINUE WORKING',
        '<official_verifier_feedback>',
        'configs/public_audit.yml exposes an invalid row-count result from the audit pipeline',
        'Requirement already satisfied: jsonschema>=4.18 (from great-expectations)',
        'Installing build dependencies: finished with status done',
        'Successfully installed dq-audit-0.1.0',
        '</official_verifier_feedback>',
      ].join('\n'),
      calls: [
        {
          toolName: 'fs.read',
          params: { path: 'configs/public_audit.yml' },
          success: true,
          result: {
            path: 'configs/public_audit.yml',
            content: 'input: data/public.csv\n',
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/dq_audit/audit.py' },
          success: true,
          result: {
            path: 'src/dq_audit/audit.py',
            content: 'def run_audit():\n    return None\n',
          },
        },
      ],
      bindings: [{ name: 'fs.synthesize', enabled: true }],
      workspaceRoot: '/app',
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({ path: 'src/dq_audit/audit.py' }),
      }),
    ]);
    expect(planner.planExternalFeedbackRepair({
      task: [
        'Continue repairing the implementation.',
        '## VERIFICATION FAILED — CONTINUE WORKING',
        '<official_verifier_feedback>',
        'configs/public_audit.yml exposes an invalid row-count result from the audit pipeline',
        'Requirement already satisfied: jsonschema>=4.18 (from great-expectations)',
        'Installing build dependencies: finished with status done',
        'Successfully installed dq-audit-0.1.0',
        '</official_verifier_feedback>',
      ].join('\n'),
      calls: [{
        toolName: plans[0]!.toolName,
        params: plans[0]!.params,
        success: false,
        error: 'Focused patch returned no diff.',
      }, {
        toolName: 'fs.read',
        params: { path: 'src/dq_audit/audit.py' },
        success: true,
        result: {
          path: 'src/dq_audit/audit.py',
          content: 'def run_audit():\n    return None\n',
        },
      }],
      bindings: [{ name: 'fs.synthesize', enabled: true }],
      workspaceRoot: '/app',
    })).toEqual([]);
  });

  it('follows explicit files and text dependencies from a grounded manifest without another model plan', () => {
    const planner = new AgentToolPlanner();
    const plans = planner.planWorkspaceEvidenceFollowUps({
      task: [
        'Inspect src/table_recon/audit.py and data/public/manifest.json.',
        'Summarize the OCR inputs and src/table_recon/cli.py.',
      ].join(' '),
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.read', enabled: true }],
      calls: [
        {
          toolName: 'fs.list',
          params: { path: '.', maxDepth: 4 },
          success: true,
          result: {
            entries: [
              'src/table_recon/audit.py',
              'src/table_recon/cli.py',
              'data/public/manifest.json',
              'data/public/invoice_ocr.json',
              'data/public/invoice.png',
            ],
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/table_recon/audit.py' },
          success: true,
          result: { path: 'src/table_recon/audit.py', content: 'def run(): pass' },
        },
        {
          toolName: 'fs.read',
          params: { path: 'data/public/manifest.json' },
          success: true,
          result: {
            path: 'data/public/manifest.json',
            content: JSON.stringify({
              documents: [{
                image_path: 'data/public/invoice.png',
                tokens_path: 'data/public/invoice_ocr.json',
              }],
            }),
          },
        },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'src/table_recon/cli.py' },
      }),
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'data/public/invoice_ocr.json' },
      }),
    ]);
    expect(JSON.stringify(plans)).not.toContain('invoice.png');
  });

  it('treats slash-separated manifest filenames as alternatives instead of a nested file path', () => {
    const plans = new AgentToolPlanner().planWorkspaceEvidenceFollowUps({
      task: 'Inspect setup.py/pyproject.toml and summarize the dependency metadata.',
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.read', enabled: true }],
      calls: [{
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
        success: true,
        result: {
          entries: ['pyproject.toml', 'src/example/__init__.py'],
        },
      }],
    });

    expect(plans.map(plan => plan.params.path)).toContain('pyproject.toml');
    expect(plans.map(plan => plan.params.path)).not.toContain('setup.py/pyproject.toml');
  });

  it('reads repair source before generated outputs in mutation-task evidence follow-ups', () => {
    const plans = new AgentToolPlanner().planWorkspaceEvidenceFollowUps({
      task: [
        'Repair src/table_recon/audit.py using data/public/manifest.json.',
        'Required outputs: outputs/layout_qc.json and outputs/reconstructed_tables.csv.',
      ].join(' '),
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.read', enabled: true }],
      calls: [{
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
        success: true,
        result: {
          entries: [
            'src/table_recon/audit.py',
            'data/public/manifest.json',
            'outputs/layout_qc.json',
            'outputs/reconstructed_tables.csv',
          ],
        },
      }],
    });

    expect(plans.map(plan => plan.params.path)).toEqual([
      'src/table_recon/audit.py',
      'data/public/manifest.json',
      'outputs/layout_qc.json',
    ]);
  });

  it('resolves artifact basenames beneath the output directory declared by the task', () => {
    const plans = new AgentToolPlanner().planWorkspaceEvidenceFollowUps({
      task: [
        'Implement the data audit.',
        '## Required Artifacts',
        'Write all of these files under `outputs/`:',
        '- `cleaned_customers.csv`',
        '- `validation_report.json`',
      ].join('\n'),
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.read', enabled: true }],
      calls: [{
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
        success: true,
        result: {
          entries: [
            'outputs/cleaned_customers.csv',
            'outputs/validation_report.json',
          ],
        },
      }],
    });

    expect(plans.map(plan => plan.params.path)).toEqual([
      'outputs/cleaned_customers.csv',
      'outputs/validation_report.json',
    ]);
  });

  it('follows file dependencies from a grounded YAML configuration', () => {
    const plans = new AgentToolPlanner().planWorkspaceEvidenceFollowUps({
      task: 'Inspect configs/public_audit.yml and implement the configured audit.',
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.read', enabled: true }],
      calls: [
        {
          toolName: 'fs.list',
          params: { path: '.', maxDepth: 4 },
          success: true,
          result: {
            entries: [
              'configs/public_audit.yml',
              'data/public/messy_customers.csv',
              'data/public/messy_orders.csv',
              'rules/public_expectations.yml',
            ],
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'configs/public_audit.yml' },
          success: true,
          result: {
            path: 'configs/public_audit.yml',
            content: [
              'customers_path: data/public/messy_customers.csv',
              'orders_path: data/public/messy_orders.csv',
              'rules_path: rules/public_expectations.yml',
            ].join('\n'),
          },
        },
      ],
    });

    expect(plans.map(plan => plan.params.path)).toEqual([
      'rules/public_expectations.yml',
      'data/public/messy_customers.csv',
      'data/public/messy_orders.csv',
    ]);
  });

  it('turns a verifier traceback into a bounded source read', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: { command: 'pytest -q' },
        success: false,
        result: {
          cwd: '/app',
          stdout: '',
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/src/dq_audit/audit.py", line 612',
            '    batch = batch_def.get_batch()',
            'IndentationError: unexpected indent',
          ].join('\n'),
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/dq_audit/audit.py',
          startLine: 587,
          endLine: 637,
        },
      }),
    ]);
  });

  it('turns a path-only verifier failure into a localized repair frontier', () => {
    const planner = new AgentToolPlanner();
    const task = 'Migrate the application and satisfy the official verifier.';
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const failure = {
      toolName: 'shell.exec',
      params: {
        command: 'python -m pytest -q .roy/official-verifier/test_outputs.py',
      },
      success: false,
      result: {
        cwd: '/app',
        exitCode: 1,
        stderr: [
          'AssertionError: source violations in calls:',
          'src/support_rag/retriever.py calls .get_relevant_documents()',
        ].join(' '),
      },
    };

    expect(planner.planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings,
      calls: [failure],
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'src/support_rag/retriever.py' },
      }),
    ]);

    const read = {
      toolName: 'fs.read',
      params: { path: 'src/support_rag/retriever.py' },
      success: true,
      result: {
        path: 'src/support_rag/retriever.py',
        content: [
          'class FAQRetriever:',
          '    def get_relevant_documents(self, query):',
          '        return []',
        ].join('\n'),
        truncated: false,
      },
    };
    expect(planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: [failure, read],
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/retriever.py',
          instructions: expect.stringContaining('.get_relevant_documents()'),
          strategy: 'patch',
        }),
      }),
    ]);
  });

  it('prioritizes the implementation path over verifier traceback frames', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: {
          command: 'python -m pytest -q .roy/official-verifier/test_outputs.py',
        },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stdout: [
            'message = source violations in stubs:',
            'src/support_rag/models.py contains forbidden shortcut ROUTE_KEYWORDS',
            '/tests/test_outputs.py:254: in test_langchain_runtime_migration',
            'E AssertionError: source violations in stubs:',
            'src/support_rag/models.py contains forbidden shortcut ROUTE_KEYWORDS',
            'FAILED .roy/official-verifier/test_outputs.py::test_langchain_runtime_migration',
          ].join('\n'),
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'src/support_rag/models.py' },
      }),
    ]);
  });

  it('repairs a freshly read manifest after the environment rejects a dependency', () => {
    const planner = new AgentToolPlanner();
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'pyproject.toml',
          instructions: 'Upgrade the runtime dependencies.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'pyproject.toml' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pip install -e .' },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: [
            'ERROR: Could not find a version that satisfies the requirement',
            'langchain-community>=0.3.0 (from support-rag)',
            'ERROR: No matching distribution found for langchain-community>=0.3.0',
          ].join(' '),
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'pyproject.toml' },
        success: true,
        result: {
          path: 'pyproject.toml',
          content: [
            '[project]',
            'dependencies = [',
            '  "langchain>=1.3.0",',
            '  "langchain-community>=0.3.0",',
            ']',
          ].join('\n'),
          truncated: false,
        },
      },
    ];

    expect(planner.planWorkspaceRepairTransition({
      task: 'Upgrade the application runtime and verify installation.',
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls,
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'pyproject.toml',
          instructions: expect.stringContaining('langchain-community'),
          strategy: 'patch',
        }),
      }),
    ]);
  });

  it('routes failed global acceptance feedback back to its observed source component', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Migrate the application runtime and preserve its CLI/API behavior.',
      '<runtime_acceptance_repair_targets>',
      'acceptance_03: Migrated replay always returns account with confidence 0.0.',
      'acceptance_07: Router ignores markers and confidences from config.',
      '</runtime_acceptance_repair_targets>',
    ].join('\n');
    const calls = [
      {
        toolName: 'fs.read',
        params: { path: 'src/support_rag/router.py' },
        success: true,
        result: {
          path: 'src/support_rag/router.py',
          content: 'def route_ticket(ticket):\n    return {"route": "account", "confidence": 0.0}\n',
          truncated: false,
        },
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/support_rag/router.py',
          instructions: 'Migrate the router.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/support_rag/router.py' },
      },
      {
        toolName: 'fs.read',
        params: { path: 'src/support_rag/router.py' },
        success: true,
        result: {
          path: 'src/support_rag/router.py',
          content: 'def route_ticket(ticket):\n    return {"route": "account", "confidence": 0.0}\n',
          truncated: false,
        },
      },
    ];

    expect(planner.planGroundedImplementationTransition({
      task,
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.synthesize', enabled: true }],
      calls,
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/router.py',
          instructions: expect.stringContaining('confidence 0.0'),
          strategy: 'patch',
        }),
      }),
    ]);
  });

  it('uses shell error text to inspect an imported workspace module', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: {
          command: 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs',
        },
        success: false,
        error: [
          'Traceback (most recent call last):',
          '  File "/app/src/dq_audit/cli.py", line 6, in <module>',
          '    from .audit import run_audit',
          "ImportError: cannot import name 'run_audit' from 'dq_audit.audit' (/app/src/dq_audit/audit.py)",
        ].join('\n'),
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: { path: 'src/dq_audit/audit.py' },
      }),
    ]);
  });

  it('repairs a stale importing consumer after a provider compatibility patch regresses verified groups', () => {
    const planner = new AgentToolPlanner();
    const calls = [
      {
        toolName: 'fs.read',
        params: { path: '.roy/official-verifier/test_outputs.py' },
        success: true,
        result: {
          path: '.roy/official-verifier/test_outputs.py',
          content: 'assert source_violations["stubs"] == []',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'src/support_rag/chain.py' },
        success: true,
        result: {
          path: 'src/support_rag/chain.py',
          content: 'from support_rag.models import LegacyAnswerLLM\n',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'src/support_rag/models.py' },
        success: true,
        result: {
          path: 'src/support_rag/models.py',
          content: 'class AnswerModel:\n    pass\n',
        },
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/support_rag/models.py',
          instructions: 'Restore the missing imported symbol.',
          strategy: 'patch',
        },
        success: true,
        result: {
          path: 'src/support_rag/models.py',
          synthesized: true,
        },
      },
      {
        toolName: 'shell.exec',
        params: {
          command: 'python -m pytest -q .roy/official-verifier/test_outputs.py',
        },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/src/support_rag/chain.py", line 7, in <module>',
            '    from support_rag.models import LegacyAnswerLLM',
            "ImportError: cannot import name 'LegacyAnswerLLM' from 'support_rag.models' (/app/src/support_rag/models.py)",
          ].join('\n'),
          verifierDiagnostics: [{
            path: '/logs/verifier/scorecard.json',
            content: JSON.stringify({
              groups: {
                imports: 1,
                router_structure: 1,
                stubs: 0,
              },
              weights: {
                imports: 0.1,
                router_structure: 0.2,
                stubs: 0.1,
              },
            }),
          }],
          candidateRollback: {
            restored: true,
            path: 'src/support_rag/models.py',
            reason: 'reward_regression',
            baselineReward: 0.3889,
            candidateReward: 0.2778,
            baselineGroups: {
              imports: 1,
              router_structure: 1,
              stubs: 1,
            },
            candidateGroups: {
              imports: 1,
              router_structure: 1,
              stubs: 0,
            },
            regressedGroups: [{
              group: 'stubs',
              before: 1,
              after: 0,
            }],
          },
        },
      },
    ];

    expect(planner.planWorkspaceRepairTransition({
      task: 'Migrate the application and preserve all passing verifier groups.',
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls,
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/chain.py',
          instructions: expect.stringMatching(
            /(?=.*Preserve the currently passing capabilities exactly: imports, router_structure, stubs)(?=.*stale importing consumer)/s
          ),
          strategy: 'patch',
        }),
        reason: expect.stringContaining('stale consumer'),
      }),
    ]);
  });

  it('does not let a later missing test runner hide an earlier source traceback', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [
        {
          toolName: 'fs.synthesize',
          params: {
            path: 'src/support_rag/router.py',
            instructions: 'Migrate the router.',
            strategy: 'patch',
          },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli route --ticket invoice' },
          success: false,
          result: {
            cwd: '/app',
            exitCode: 1,
            stderr: [
              'Traceback (most recent call last):',
              '  File "/app/src/support_rag/models.py", line 7, in <module>',
              '    from langchain.llms.base import LLM',
              "ModuleNotFoundError: No module named 'langchain.llms'",
            ].join('\n'),
          },
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pytest -q' },
          success: false,
          result: {
            cwd: '/app',
            exitCode: 1,
            stderr: '/usr/local/bin/python: No module named pytest',
          },
        },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/support_rag/models.py',
          startLine: 1,
          endLine: 32,
        },
      }),
    ]);
  });

  it('follows an abstract-instantiation failure to the class definition', () => {
    const planner = new AgentToolPlanner();
    const failure = {
      toolName: 'shell.exec',
      params: { command: 'python -m support_rag.cli answer' },
      success: false,
      result: {
        cwd: '/app',
        exitCode: 1,
        stderr: [
          'Traceback (most recent call last):',
          '  File "/app/src/support_rag/chain.py", line 42, in build_answer_chain',
          '    llm = LegacyAnswerLLM()',
          "TypeError: Can't instantiate abstract class LegacyAnswerLLM with abstract methods _generate, _llm_type",
        ].join('\n'),
      },
    };
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.search', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const searchPlan = planner.planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings,
      calls: [failure],
    });

    expect(searchPlan).toEqual([
      expect.objectContaining({
        toolName: 'fs.search',
        params: {
          path: '.',
          filePattern: '*.py',
          query: 'class\\s+LegacyAnswerLLM\\b',
          regex: true,
          maxResults: 20,
        },
        reason: expect.stringContaining('definition instead of patching the caller'),
      }),
    ]);

    const definitionRead = planner.planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings,
      calls: [
        failure,
        {
          toolName: 'fs.search',
          params: searchPlan[0]!.params,
          success: true,
          result: {
            matches: [{
              path: 'src/support_rag/models.py',
              line: 24,
              preview: 'class LegacyAnswerLLM(BaseLLM):',
            }],
          },
        },
      ],
    });
    expect(definitionRead).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/support_rag/models.py',
          startLine: 9,
          endLine: 104,
        },
        reason: expect.stringContaining('semantic cause'),
      }),
    ]);
  });

  it('repairs a grounded abstract class definition rather than its caller frame', () => {
    const planner = new AgentToolPlanner();
    const failure = {
      toolName: 'shell.exec',
      params: { command: 'python -m support_rag.cli answer' },
      success: false,
      result: {
        cwd: '/app',
        exitCode: 1,
        stderr: [
          'Traceback (most recent call last):',
          '  File "/app/src/support_rag/chain.py", line 42, in build_answer_chain',
          '    llm = LegacyAnswerLLM()',
          "TypeError: Can't instantiate abstract class LegacyAnswerLLM with abstract methods _generate, _llm_type",
        ].join('\n'),
      },
    };
    const plans = planner.planWorkspaceRepairTransition({
      task: 'Repair src/support_rag/models.py and preserve the CLI.',
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [
        failure,
        {
          toolName: 'fs.read',
          params: {
            path: 'src/support_rag/models.py',
            startLine: 9,
            endLine: 104,
          },
          success: true,
          result: {
            path: 'src/support_rag/models.py',
            content: 'class LegacyAnswerLLM(BaseLLM):\n    def _call(self, prompt): return prompt\n',
            truncated: false,
          },
        },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/models.py',
          strategy: 'patch',
          instructions: expect.stringMatching(/abstract class LegacyAnswerLLM[\s\S]*_generate, _llm_type/i),
        }),
      }),
    ]);
  });

  it('inspects an imported runtime signature before repairing an unexpected keyword', () => {
    const planner = new AgentToolPlanner();
    const failure = {
      toolName: 'shell.exec',
      params: { command: 'python -m support_rag.cli route --ticket invoice' },
      success: false,
      result: {
        command: 'python -m support_rag.cli route --ticket invoice',
        cwd: '/app',
        exitCode: 1,
        stderr: [
          'Traceback (most recent call last):',
          '  File "/app/src/support_rag/router.py", line 88, in build_router_agent',
          '    agent = create_agent(llm=llm, tools=tools)',
          "TypeError: create_agent() got an unexpected keyword argument 'llm'",
        ].join('\n'),
      },
    };
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.search', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const searchPlan = planner.planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings,
      calls: [failure],
    });
    expect(searchPlan).toEqual([
      expect.objectContaining({
        toolName: 'fs.search',
        params: {
          path: 'src/support_rag',
          filePattern: 'router.py',
          query: 'create_agent',
          maxResults: 20,
        },
      }),
    ]);
    const searchCall = {
      toolName: 'fs.search',
      params: searchPlan[0]!.params,
      success: true,
      result: {
        matches: [
          {
            path: 'src/support_rag/router.py',
            line: 8,
            preview: 'from langchain.agents import create_agent',
          },
          {
            path: 'src/support_rag/router.py',
            line: 88,
            preview: 'agent = create_agent(llm=llm, tools=tools)',
          },
        ],
      },
    };
    const signaturePlan = planner.planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings,
      calls: [failure, searchCall],
    });
    const signatureCommand =
      'python -c "import inspect; from langchain.agents import create_agent; print(inspect.signature(create_agent))"';
    expect(signaturePlan).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: {
          command: signatureCommand,
          maxOutputBytes: 8_000,
        },
        reason: expect.stringContaining('do not guess replacement keyword names'),
      }),
    ]);
    const sourceRead = {
      toolName: 'fs.read',
      params: {
        path: 'src/support_rag/router.py',
        startLine: 63,
        endLine: 113,
      },
      success: true,
      result: {
        path: 'src/support_rag/router.py',
        content: [
          'def build_router_agent(path):',
          '    return create_agent(llm=model, tools=tools)',
        ].join('\n'),
        truncated: false,
      },
    };
    expect(planner.planWorkspaceRepairTransition({
      task: 'Repair src/support_rag/router.py and rerun the CLI.',
      workspaceRoot: '/app',
      bindings,
      calls: [failure, searchCall, sourceRead],
    })).toEqual([]);

    const plans = planner.planWorkspaceRepairTransition({
      task: 'Repair src/support_rag/router.py and rerun the CLI.',
      workspaceRoot: '/app',
      bindings,
      calls: [
        failure,
        searchCall,
        {
          toolName: 'shell.exec',
          params: signaturePlan[0]!.params,
          success: true,
          result: {
            command: signatureCommand,
            stdout: '(model, tools=None, *, system_prompt=None, context_schema=None)\n',
            exitCode: 0,
          },
        },
        sourceRead,
      ],
    });
    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({
          path: 'src/support_rag/router.py',
          strategy: 'patch',
          instructions: expect.stringMatching(
            /unexpected keyword argument 'llm'[\s\S]*system_prompt=None, context_schema=None/
          ),
        }),
      }),
    ]);
  });

  it('bootstraps an explicitly invoked missing development tool after source failures clear', () => {
    const planner = new AgentToolPlanner();
    const bindings = [{ name: 'shell.exec', enabled: true }];
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/support_rag/router.py',
          instructions: 'Repair the router.',
          strategy: 'patch',
        },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m support_rag.cli route --ticket invoice' },
        success: true,
        result: {
          command: 'python -m support_rag.cli route --ticket invoice',
          cwd: '/app',
          exitCode: 0,
          stdout: '{"route":"billing"}',
        },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          command: 'python -m pytest -q',
          cwd: '/app',
          exitCode: 1,
          stderr: '/usr/local/bin/python: No module named pytest',
        },
      },
    ];

    expect(planner.planEnvironmentRecovery({
      calls,
      bindings,
      workspaceRoot: '/app',
    })).toEqual([
      expect.objectContaining({
        toolName: 'shell.exec',
        params: { command: 'python -m pip install pytest' },
        reason: expect.stringContaining('development-tool module is absent'),
      }),
    ]);
    expect(planner.planEnvironmentRecovery({
      calls: [
        ...calls,
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pip install pytest' },
          success: true,
        },
      ],
      bindings,
      workspaceRoot: '/app',
    })).toEqual([]);
  });

  it('does not install a development tool while a newer source repair remains', () => {
    const planner = new AgentToolPlanner();
    expect(planner.planEnvironmentRecovery({
      workspaceRoot: '/app',
      bindings: [{ name: 'shell.exec', enabled: true }],
      calls: [
        {
          toolName: 'fs.synthesize',
          params: {
            path: 'src/support_rag/chain.py',
            instructions: 'Migrate the chain.',
            strategy: 'patch',
          },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m support_rag.cli answer' },
          success: false,
          result: {
            cwd: '/app',
            exitCode: 1,
            stderr: [
              'Traceback (most recent call last):',
              '  File "/app/src/support_rag/models.py", line 5, in <module>',
              "ModuleNotFoundError: No module named 'langchain.llms'",
            ].join('\n'),
          },
        },
        {
          toolName: 'shell.exec',
          params: { command: 'python -m pytest -q' },
          success: false,
          result: {
            command: 'python -m pytest -q',
            cwd: '/app',
            exitCode: 1,
            stderr: '/usr/local/bin/python: No module named pytest',
          },
        },
      ],
    })).toEqual([]);

    expect(planner.planEnvironmentRecovery({
      bindings: [{ name: 'shell.exec', enabled: true }],
      calls: [{
        toolName: 'shell.exec',
        params: { command: 'python -m business_package.cli' },
        success: false,
        result: {
          command: 'python -m business_package.cli',
          stderr: '/usr/bin/python: No module named business_package',
        },
      }],
    })).toEqual([]);
  });

  it('uses a non-perfect official verifier score and nested diagnostic traceback', () => {
    const plans = new AgentToolPlanner().planWorkspaceFailureFollowUps({
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [{
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          cwd: '/app',
          stdout: '0.040000000000\n',
          stderr: '',
          exitCode: 0,
          verifierDiagnostics: [{
            path: '/logs/verifier/grade.log',
            content: JSON.stringify({
              public_log_tail: [
                'Traceback (most recent call last):',
                '  File "/app/src/table_recon/audit.py", line 433, in run_audit',
                '    expected.get("tables", [])',
                "AttributeError: 'list' object has no attribute 'get'",
              ].join('\n'),
            }),
          }],
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.read',
        params: {
          path: 'src/table_recon/audit.py',
          startLine: 408,
          endLine: 458,
        },
      }),
    ]);
  });

  it('transitions a freshly inspected localized traceback into a minimal patch', () => {
    const planner = new AgentToolPlanner();
    const task = 'Repair src/table_recon/audit.py and rerun the CLI.';
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const calls = [
      {
        toolName: 'shell.exec',
        params: {
          command: 'python -m table_recon.cli run --manifest data/public/manifest.json --out-dir outputs',
        },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/src/table_recon/audit.py", line 191, in run_audit',
            '    if cell_boxes and texts:',
            "UnboundLocalError: cannot access local variable 'cell_boxes'",
          ].join('\n'),
        },
      },
      {
        toolName: 'fs.read',
        params: {
          path: 'src/table_recon/audit.py',
          startLine: 166,
          endLine: 216,
        },
        success: true,
        result: {
          path: 'src/table_recon/audit.py',
          content: 'def run_audit(...):\n    if cell_boxes and texts:\n        pass\n',
          truncated: false,
        },
      },
    ];

    const plans = planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls,
    });

    expect(plans).toEqual([expect.objectContaining({
      toolName: 'fs.synthesize',
      params: expect.objectContaining({
        path: 'src/table_recon/audit.py',
        strategy: 'patch',
        instructions: expect.stringMatching(/cell_boxes[\s\S]*every control-flow path/),
      }),
      reason: expect.stringContaining('exact traceback location'),
    })]);
    expect(planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: [
        ...calls,
        {
          toolName: plans[0]!.toolName,
          params: plans[0]!.params,
          success: false,
          result: { synthesisRejected: true },
        },
      ],
    })).toEqual([]);
  });

  it('does not let a newer environment-only failure mask a grounded source repair', () => {
    const planner = new AgentToolPlanner();
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/chain.py',
          instructions: 'Migrate the first source slice.',
          strategy: 'patch',
        },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m app.cli replay' },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/src/app/models.py", line 5, in <module>',
            '    from legacy.models import LegacyModel',
            "ModuleNotFoundError: No module named 'legacy.models'",
          ].join('\n'),
        },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: '/usr/local/bin/python: No module named pytest',
        },
      },
      {
        toolName: 'fs.read',
        params: {
          path: 'src/app/models.py',
          startLine: 1,
          endLine: 40,
        },
        success: true,
        result: {
          path: 'src/app/models.py',
          content: 'from legacy.models import LegacyModel\n',
          truncated: false,
        },
      },
    ];

    const plans = planner.planWorkspaceRepairTransition({
      task: 'Migrate the application and run its acceptance commands.',
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls,
    });

    expect(plans).toEqual([expect.objectContaining({
      toolName: 'fs.synthesize',
      params: expect.objectContaining({
        path: 'src/app/models.py',
        strategy: 'patch',
        instructions: expect.stringContaining('legacy.models'),
      }),
      reason: expect.stringContaining('environment-only'),
    })]);
  });

  it('transitions cached complete source evidence into the next explicit implementation slice', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Migrate the current project implementation.',
      '- src/app/chain.py: Replace LegacyChain.predict with RunnableSequence.invoke.',
      '- src/app/models.py: Replace LegacyModel with CoreModel.',
      'Run `python -m app.cli answer` and verify the result.',
    ].join('\n');
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const calls = [
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'pyproject.toml',
          instructions: 'Update runtime dependencies.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'pyproject.toml' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: '/usr/bin/python: No module named pytest',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: '/app/src/app/models.py' },
        success: true,
        result: {
          path: '/app/src/app/models.py',
          content: 'class LegacyModel:\n    pass\n',
          truncated: false,
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'src/app/chain.py' },
        success: true,
        result: {
          path: 'src/app/chain.py',
          content: 'class LegacyChain:\n    def predict(self): pass\n',
          truncated: false,
        },
      },
    ];

    const plans = planner.planGroundedImplementationTransition({
      task,
      calls,
      bindings,
      workspaceRoot: '/app',
    });

    expect(plans).toEqual([expect.objectContaining({
      toolName: 'fs.synthesize',
      params: expect.objectContaining({
        path: 'src/app/chain.py',
        strategy: 'patch',
        instructions: expect.stringMatching(/LegacyChain[\s\S]*preserve public interfaces/i),
      }),
      reason: expect.stringContaining('cached inspection to execution'),
    })]);

    const nextPlans = planner.planGroundedImplementationTransition({
      task,
      calls: [
        ...calls,
        {
          toolName: plans[0]!.toolName,
          params: plans[0]!.params,
          success: true,
          result: { path: 'src/app/chain.py' },
        },
      ],
      bindings,
      workspaceRoot: '/app',
    });
    expect(nextPlans).toEqual([expect.objectContaining({
      params: expect.objectContaining({ path: 'src/app/models.py' }),
    })]);
  });

  it('does not retry a rolled-back generic synthesis without new path-specific failure evidence', () => {
    const planner = new AgentToolPlanner();
    const task = [
      'Migrate src/app/router.py to the current runtime.',
      'Run `python -m pytest -q` and verify the result.',
    ].join('\n');
    const calls = [
      {
        toolName: 'fs.read',
        params: { path: 'src/app/router.py' },
        success: true,
        result: {
          path: 'src/app/router.py',
          content: 'from legacy.router import Router\n',
          truncated: false,
        },
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/router.py',
          instructions: 'Migrate the router.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/app/router.py' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          exitCode: 1,
          stderr: '/usr/bin/python: No module named pytest',
          candidateRollback: {
            restored: true,
            path: 'src/app/router.py',
            reason: 'no_verifier_gain',
          },
        },
      },
    ];

    expect(planner.planGroundedImplementationTransition({
      task,
      calls,
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    })).toEqual([]);
  });

  it('reopens a rolled-back source path when a newer verifier failure localizes it', () => {
    const planner = new AgentToolPlanner();
    const task = 'Migrate src/app/router.py to the current runtime and verify it.';
    const calls = [
      {
        toolName: 'fs.read',
        params: { path: 'src/app/router.py' },
        success: true,
        result: {
          path: 'src/app/router.py',
          content: 'from legacy.router import Router\n',
          truncated: false,
        },
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/router.py',
          instructions: 'Migrate the router.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/app/router.py' },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: {
          exitCode: 1,
          stderr: 'src/app/router.py:42: router contract still fails',
          candidateRollback: {
            restored: true,
            path: 'src/app/router.py',
            reason: 'no_verifier_gain',
          },
        },
      },
    ];

    expect(planner.planGroundedImplementationTransition({
      task,
      calls,
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
    })).toEqual([
      expect.objectContaining({
        params: expect.objectContaining({ path: 'src/app/router.py' }),
      }),
    ]);
  });

  it('verifies the latest previously failing acceptance command before broad closure checks', () => {
    const planner = new AgentToolPlanner();
    const task = [
      '## Helpful Commands',
      '```bash',
      'python -m app.cli answer',
      'python -m app.cli replay',
      'python -m pytest -q',
      '```',
    ].join('\n');
    const calls = [
      {
        toolName: 'shell.exec',
        params: { command: 'python -m app.cli answer' },
        success: true,
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python -m pytest -q' },
        success: false,
        result: { exitCode: 1, stderr: 'src/app/router.py:12: assertion failed' },
      },
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/app/router.py',
          instructions: 'Repair the reported failure.',
          strategy: 'patch',
        },
        success: true,
        result: { path: 'src/app/router.py' },
      },
    ];

    expect(planner.planPostMutationVerification({
      task,
      calls,
      bindings: [{ name: 'shell.exec', enabled: true }],
    })).toEqual([
      expect.objectContaining({
        params: { command: 'python -m pytest -q' },
        reason: expect.stringContaining('focused causal feedback'),
      }),
    ]);
  });

  it('does not turn unrelated reads or non-mutation tasks into source synthesis', () => {
    const planner = new AgentToolPlanner();
    const calls = [{
      toolName: 'fs.read',
      params: { path: 'src/app/chain.py' },
      success: true,
      result: {
        path: 'src/app/chain.py',
        content: 'def answer(): return "ok"\n',
        truncated: false,
      },
    }];
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
    ];

    expect(planner.planGroundedImplementationTransition({
      task: 'Summarize the project architecture.',
      calls,
      bindings,
    })).toEqual([]);
    expect(planner.planGroundedImplementationTransition({
      task: 'Modify src/app/router.py to implement the new route contract.',
      calls,
      bindings,
    })).toEqual([]);
  });

  it('does not treat config filenames or generic prose terms as implementation evidence', () => {
    const planner = new AgentToolPlanner();
    const calls = [{
      toolName: 'fs.read',
      params: { path: 'src/app/cli.py' },
      success: true,
      result: {
        path: 'src/app/cli.py',
        content: [
          'def main(argv: list[str] | None = None) -> int:',
          '    runtime = load_runtime("configs/agent.yaml")',
          '    return runtime.inspect()',
        ].join('\n'),
        truncated: false,
      },
    }];

    expect(planner.planGroundedImplementationTransition({
      task: [
        'Migrate the application Runtime. Inspect the installed package.',
        'Read routing policy from `configs/agent.yaml` and preserve None defaults.',
        'Keep the CLI stable.',
      ].join('\n'),
      calls,
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
      ],
    })).toEqual([]);
  });

  it('does not synthesize files from title-case prose or generic acronyms', () => {
    const planner = new AgentToolPlanner();
    const calls = [{
      toolName: 'fs.read',
      params: { path: 'src/app/api.py' },
      success: true,
      result: {
        path: 'src/app/api.py',
        content: [
          'def load_payload(path):',
          '    return JSONDecoder().decode(path.read_text())',
        ].join('\n'),
        truncated: false,
      },
    }];

    expect(planner.planGroundedImplementationTransition({
      task: 'Migrate the project. Load runtime configuration and preserve JSON output.',
      calls,
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
      ],
    })).toEqual([]);
  });

  it('extracts explicitly quoted natural-language commands from a team task', () => {
    const planner = new AgentToolPlanner();
    const plans = planner.plan({
      task: [
        "Run 'cd /app && python -m pip install -e .'.",
        "Confirm with 'python -c \"import app; print(app.__version__)\"'.",
        "If the test runner is absent, install it: 'pip install pytest'.",
      ].join(' '),
      workspacePath: '.',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans.map(plan => plan.params.command)).toEqual([
      'cd /app && python -m pip install -e .',
      'python -c "import app; print(app.__version__)"',
      'pip install pytest',
    ]);
  });

  it('does not keep repairing from stale failures after a newer mutation passes its declared command', () => {
    const planner = new AgentToolPlanner();
    const command = 'python -m dq_audit.cli run --config configs/public_audit.yml --out-dir outputs';
    const plans = planner.planWorkspaceRepairTransition({
      task: [
        'Repair src/dq_audit/audit.py.',
        '## Required Command',
        '```bash',
        command,
        '```',
      ].join('\n'),
      workspaceRoot: '/app',
      bindings: [
        { name: 'fs.read', enabled: true },
        { name: 'fs.synthesize', enabled: true },
        { name: 'shell.exec', enabled: true },
      ],
      calls: [
        {
          toolName: 'shell.exec',
          params: { command },
          success: false,
          result: {
            cwd: '/app',
            exitCode: 1,
            stderr: 'NotImplementedError: implement the pipeline',
          },
        },
        {
          toolName: 'fs.read',
          params: { path: '.roy/official-verifier/test_outputs.py' },
          success: true,
          result: {
            path: '.roy/official-verifier/test_outputs.py',
            content: 'def test_outputs(): ...',
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/dq_audit/audit.py' },
          success: true,
          result: {
            path: 'src/dq_audit/audit.py',
            content: 'def run_audit(): pass',
          },
        },
        {
          toolName: 'fs.synthesize',
          params: { path: 'src/dq_audit/audit.py', instructions: 'Implement it.' },
          success: true,
        },
        {
          toolName: 'shell.exec',
          params: { command },
          success: true,
          result: { cwd: '/app', exitCode: 0, stdout: '' },
        },
      ],
    });

    expect(plans).toEqual([]);
  });

  it('transitions aggregate verifier evidence directly into a preserving repair', () => {
    const planner = new AgentToolPlanner();
    const task = 'Repair src/table_recon/audit.py until the official verifier reward is 1.';
    const bindings = [
      { name: 'fs.read', enabled: true },
      { name: 'fs.synthesize', enabled: true },
      { name: 'shell.exec', enabled: true },
    ];
    const calls = [
        {
          toolName: 'shell.exec',
          params: { command: 'python .roy/official-verifier/grade.py' },
          success: true,
          result: {
            cwd: '/app',
            stdout: '0.020000000000\n',
            exitCode: 0,
            verifierDiagnostics: [{
              path: '/logs/verifier/scorecard.json',
              content: '{"groups":{"G_public_reconstruction":1,"G_hidden_end_to_end_stress":0}}',
            }],
          },
        },
        {
          toolName: 'fs.read',
          params: { path: '.roy/official-verifier/grade.py', maxBytes: 20_000 },
          success: true,
          result: {
            path: '.roy/official-verifier/grade.py',
            content: 'GROUPS = ["G_public_reconstruction", "G_hidden_end_to_end_stress"]',
            truncated: false,
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/table_recon/audit.py', maxBytes: 30_000 },
          success: true,
          result: {
            path: 'src/table_recon/audit.py',
            content: 'def run_audit(manifest, output):\n    reconstruct_public(manifest, output)\n',
            truncated: true,
          },
        },
        {
          toolName: 'fs.read',
          params: { path: 'src/table_recon/cli.py', maxBytes: 30_000 },
          success: true,
          result: {
            path: 'src/table_recon/cli.py',
            content: 'from .audit import run_audit\n',
            truncated: false,
          },
        },
      ];
    const plans = planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls,
    });

    expect(plans).toEqual([expect.objectContaining({
      toolName: 'fs.synthesize',
      params: {
        path: 'src/table_recon/audit.py',
        instructions: expect.stringContaining('preserving behavior already proven'),
        strategy: 'patch',
      },
    })]);
    expect(planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: [
        ...calls,
        {
          toolName: 'fs.synthesize',
          params: { path: 'src/table_recon/audit.py', instructions: 'Preserving repair.' },
          success: true,
          result: { path: 'src/table_recon/audit.py', synthesized: true },
        },
      ],
    })).toEqual([]);

    const rejectedCalls = [
      ...calls,
      {
        toolName: 'fs.synthesize',
        params: {
          path: 'src/table_recon/audit.py',
          instructions: 'Broad preserving repair.',
        },
        success: true,
        result: { path: 'src/table_recon/audit.py', synthesized: true },
      },
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: true,
        result: {
          cwd: '/app',
          stdout: '0.015000000000\n',
          exitCode: 0,
          verifierDiagnostics: [{
            path: '/logs/verifier/scorecard.json',
            content: '{"groups":{"G_public_reconstruction":1,"G_hidden_end_to_end_stress":0},"weights":{"G_public_reconstruction":0.005,"G_hidden_end_to_end_stress":0.89}}',
          }, {
            path: '/logs/verifier/grade.log',
            content: [
              'Traceback (most recent call last):',
              '  File "/app/src/table_recon/audit.py", line 478, in run_audit',
              '    is_cropped = _is_cropped(tokens, all_cells, page_w, page_h)',
              "UnboundLocalError: cannot access local variable 'all_cells'",
            ].join('\n'),
          }],
          candidateRollback: {
            restored: true,
            path: 'src/table_recon/audit.py',
            reason: 'reward_regression',
            baselineReward: 0.02,
            candidateReward: 0.015,
            regressedGroups: [{
              group: 'G_public_reconstruction',
              before: 1,
              after: 0,
            }],
          },
        },
      },
      {
        toolName: 'shell.exec',
        params: {
          command: 'ROY_VERIFIER_PROBE=1 python .roy/runtime/verifier_probe.py',
        },
        success: true,
        result: {
          command: 'ROY_VERIFIER_PROBE=1 python .roy/runtime/verifier_probe.py',
          stdout: [
            'VERIFIER_PROBE_EVIDENCE_VERSION 2',
            'VERIFIER_PROBE_CALL reconstruction_fraction',
            'VERIFIER_PROBE_MISMATCHES {"mismatch_count":1,"mismatches":[{"expected":"Q2","actual":null}]}',
            'VERIFIER_PROBE_SPEC {"artifact":"layout_qc.json","content":"qc.get(\\"cropped_pages_detected\\", 0) >= 1"}',
            `VERIFIER_PROBE_ARTIFACT ${JSON.stringify({
              directory: '/tmp/hidden',
              path: 'hidden_input_manifest.json',
              content: 'x'.repeat(5_000),
            })}`,
            `VERIFIER_PROBE_ARTIFACT ${JSON.stringify({
              directory: '/tmp/hidden',
              path: 'outputs/layout_qc.json',
              content: '{"rotated_or_skewed_pages_detected":["a","b"],"cropped_pages_detected":[]}',
            })}`,
          ].join('\n'),
        },
      },
    ];
    const focused = planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: rejectedCalls,
    });
    expect(focused).toEqual([expect.objectContaining({
      toolName: 'fs.synthesize',
      params: expect.objectContaining({
        path: 'src/table_recon/audit.py',
        instructions: expect.stringMatching(
          /G_hidden_end_to_end_stress[\s\S]*G_public_reconstruction/
        ),
      }),
      reason: expect.stringContaining('repair hypothesis'),
    })]);
    expect(String(focused[0]?.params.instructions)).toContain(
      'VERIFIER_PROBE_MISMATCHES'
    );
    expect(String(focused[0]?.params.instructions)).toContain(
      'G_hidden_end_to_end_stress (score 0.000, weight 0.890)'
    );
    expect(String(focused[0]?.params.instructions)).toContain('layout_qc.json');
    expect(String(focused[0]?.params.instructions).length).toBeLessThanOrEqual(4_000);
    const focusedPlan = focused[0]!;
    expect(planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: [
        ...rejectedCalls,
        {
          toolName: 'shell.exec',
          params: { command: 'python .roy/official-verifier/grade.py' },
          success: true,
          result: {
            cwd: '/app',
            stdout: '0.020000000000\n',
            exitCode: 0,
            verifierDiagnostics: [{
              path: '/logs/verifier/scorecard.json',
              content: '{"groups":{"G_public_reconstruction":1,"G_hidden_end_to_end_stress":0}}',
            }],
          },
        },
      ],
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({ strategy: 'patch' }),
      }),
    ]);
    expect(planner.planWorkspaceRepairTransition({
      task,
      workspaceRoot: '/app',
      bindings,
      calls: [
        ...rejectedCalls,
        {
          toolName: focusedPlan.toolName,
          params: focusedPlan.params,
          success: true,
          result: { path: 'src/table_recon/audit.py', synthesized: true },
        },
        rejectedCalls.at(-1)!,
      ],
    })).toEqual([]);
  });

  it('never turns a verifier traceback into a verifier mutation', () => {
    const planner = new AgentToolPlanner();
    const calls = [
      {
        toolName: 'shell.exec',
        params: { command: 'python .roy/official-verifier/grade.py' },
        success: false,
        result: {
          cwd: '/app',
          exitCode: 1,
          stderr: [
            'Traceback (most recent call last):',
            '  File "/app/.roy/official-verifier/grade.py", line 40, in grade',
            '    raise AssertionError("unexpected output")',
          ].join('\n'),
          verifierDiagnostics: [{
            path: '/logs/verifier/scorecard.json',
            content: '{"groups":{"correctness":0},"weights":{"correctness":1}}',
          }],
        },
      },
      {
        toolName: 'fs.read',
        params: { path: '.roy/official-verifier/grade.py' },
        success: true,
        result: {
          path: '.roy/official-verifier/grade.py',
          content: 'def grade(): raise AssertionError("unexpected output")',
        },
      },
      {
        toolName: 'fs.read',
        params: { path: 'src/dq_audit/audit.py' },
        success: true,
        result: {
          path: 'src/dq_audit/audit.py',
          content: 'def run_audit(): return None',
        },
      },
    ];

    expect(planner.planWorkspaceRepairTransition({
      task: 'Repair src/dq_audit/audit.py until the official verifier passes.',
      workspaceRoot: '/app',
      bindings: [{ name: 'fs.synthesize', enabled: true }],
      calls,
    })).toEqual([
      expect.objectContaining({
        toolName: 'fs.synthesize',
        params: expect.objectContaining({ path: 'src/dq_audit/audit.py' }),
      }),
    ]);
  });

  it('reads the package manifest for an architecture critic', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Identify architectural coupling risks using filesystem evidence.',
      workspacePath: '/workspace',
      archetype: 'critic',
      bindings: [{ name: 'fs.read', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'package.json' }, groundingRequired: true }),
    ]);
  });

  it('plans concrete source reads for runtime API export inspection', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect the exported runtime API surface and identify mismatches.',
      workspacePath: '/workspace',
      archetype: 'custom',
      bindings: [{ name: 'fs.read', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/index.ts' } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/core/runtime/index.ts' } }),
    ]);
  });

  it('merges an explicit manifest target with inferred runtime API source targets', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Read package.json and inspect exported runtime APIs for a consistency mismatch.',
      workspacePath: '/workspace',
      archetype: 'custom',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans.map(plan => plan.params.path)).toEqual([
      'package.json',
      'src/index.ts',
      'src/core/runtime/index.ts',
    ]);
    expect(plans.every(plan => plan.toolName === 'fs.read')).toBe(true);
  });

  it('uses an allowlisted manifest command when a cached critic only exposes shell execution', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Identify architectural coupling risks using filesystem evidence.',
      workspacePath: '/workspace',
      archetype: 'critic',
      bindings: [{ name: 'shell.exec', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'shell.exec', params: { command: 'cat package.json' }, groundingRequired: true }),
    ]);
  });

  it('prioritizes an explicitly requested directory and multiple source files', () => {
    const planner = new AgentToolPlanner();
    const plans = planner.plan({
      task: 'Read src/core/runtime/index.ts and src/server/RuntimeSessionPool.ts. Also read src/core/delegation/index.ts and list tests/ directory.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list', params: { path: 'tests', maxDepth: 3 } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/core/runtime/index.ts' } }),
      expect.objectContaining({ toolName: 'fs.read', params: { path: 'src/server/RuntimeSessionPool.ts' } }),
    ]);
  });

  it('keeps the broad workspace listing fallback for tasks without explicit paths', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Inspect this repository structure using filesystem evidence.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [{ name: 'fs.list', enabled: true }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'fs.list', params: { path: '.', maxDepth: 4 } }),
    ]);
  });

  it('searches the web for current externally verifiable evidence', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Search the web for the latest official Node.js fetch documentation and cite sources.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'fs.read', enabled: true },
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({ toolName: 'web.search', groundingRequired: true }),
    ]);
    expect(plans[0].params.query).toBe('Node.js fetch documentation and cite sources.');
  });

  it('removes imperative web-source scaffolding from the search query', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Use public web sources to compare Node.js AbortSignal.timeout and MDN AbortSignal.timeout. Open at least two sources.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [{ name: 'web.search', enabled: true }],
    });

    expect(plans[0].params.query).toBe('Node.js AbortSignal.timeout and MDN AbortSignal.timeout');
  });

  it('does not treat web tool syntax as a task entity during follow-up alignment', () => {
    const planner = new AgentToolPlanner();

    expect(planner.isWebCandidateAligned(
      'Use web.search and web.fetch to verify the supplied factual claims.',
      'alternate historical scope and counterexample evidence'
    )).toBe(true);
  });

  it('searches each numbered factual question instead of the tool-use scaffolding', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Use web.search and web.fetch to verify every answer against public evidence.',
        '',
        '1. Who created the original Chipmunks characters?',
        '2. Which musical premiered in the United States on December 10, 1993?',
        '3. Who became British prime minister immediately after Arthur Balfour?',
        '4. Who had a 1970s number-one hit with Kiss You All Over?',
        '5. What disease killed Kathleen Ferrier?',
        '',
        'Write one coherent story after resolving the facts.',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'web.search',
        params: {
          query: 'Who created the original Chipmunks characters?',
          maxResults: 5,
        },
      }),
      expect.objectContaining({
        toolName: 'web.search',
        params: {
          query: 'Which musical premiered in the United States on December 10, 1993?',
          maxResults: 5,
        },
      }),
      expect.objectContaining({
        toolName: 'web.search',
        params: {
          query: 'Who became British prime minister immediately after Arthur Balfour?',
          maxResults: 5,
        },
      }),
      expect.objectContaining({
        toolName: 'web.search',
        params: {
          query: 'Who had a 1970s number-one hit with Kiss You All Over?',
          maxResults: 5,
        },
      }),
      expect.objectContaining({
        toolName: 'web.search',
        params: {
          query: 'What disease killed Kathleen Ferrier?',
          maxResults: 5,
        },
      }),
    ]);
  });

  it('extracts independent questions from a compact delegated assignment', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Use web.search and web.fetch to find canonical answers.',
        '1. Who created the original Chipmunks characters?',
        '2. Which musical premiered in the United States on December 10, 1993?',
        '3. Who became British prime minister immediately after Arthur Balfour?',
        '4. Who had a 1970s number-one hit with Kiss You All Over?',
        '5. What disease killed Kathleen Ferrier?',
        'Return answers in a structured format.',
      ].join(' '),
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toHaveLength(5);
    expect(plans.map(plan => plan.params.query)).toEqual([
      'Who created the original Chipmunks characters?',
      'Which musical premiered in the United States on December 10, 1993?',
      'Who became British prime minister immediately after Arthur Balfour?',
      'Who had a 1970s number-one hit with Kiss You All Over?',
      'What disease killed Kathleen Ferrier?',
    ]);
  });

  it('splits a compact question range into independent public web searches', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Use web.search and web.fetch to find precise canonical answers.',
        'Questions 1-3: Who sang the title song? Which state ended prohibition? Which actress was Miss Greenwich Village?',
        'Provide answers with source URLs.',
      ].join(' '),
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans.map(plan => plan.params.query)).toEqual([
      'Who sang the title song?',
      'Which state ended prohibition?',
      'Which actress was Miss Greenwich Village?',
    ]);
  });

  it('fetches an explicitly supplied public URL instead of searching again', () => {
    const plans = new AgentToolPlanner().plan({
      task: 'Read https://nodejs.org/api/globals.html and summarize the fetch section.',
      workspacePath: '.',
      archetype: 'researcher',
      bindings: [
        { name: 'web.search', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/api/globals.html' },
      }),
    ]);
  });

  it('ignores incidental URLs embedded in terminal feedback for workspace tasks', () => {
    const plans = new AgentToolPlanner().plan({
      task: [
        'Repair the current workspace package and rerun its tests.',
        'Latest command output:',
        'WARNING: use a virtual environment: https://pip.pypa.io/warnings/venv',
      ].join('\n'),
      workspacePath: '.',
      archetype: 'coder',
      bindings: [
        { name: 'fs.list', enabled: true },
        { name: 'web.fetch', enabled: true },
      ],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        toolName: 'fs.list',
        params: { path: '.', maxDepth: 4 },
      }),
    ]);
  });

  it('prioritizes official API documentation links over unofficial downloads', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find official Node.js documentation for the global fetch API and AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [
        {
          toolName: 'web.fetch',
          params: { url: 'https://nodejs.org/' },
          success: true,
          result: {
            links: [
              { text: 'Unofficial builds', url: 'https://unofficial-builds.nodejs.org/download/' },
              { text: 'Docs', url: 'https://nodejs.org/docs/latest/api/' },
              { text: 'Global objects', url: 'https://nodejs.org/api/globals.html' },
            ],
          },
        },
      ],
    });

    expect(plans.map(plan => plan.params.url)).toEqual([
      'https://nodejs.org/docs/latest/api/',
      'https://nodejs.org/api/globals.html',
    ]);
  });

  it('treats a task-relevant fragment as focused evidence and hands control back to the agent planner', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find official Node.js documentation for the global fetch API and AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay' },
        success: true,
        result: {
          finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay',
          title: 'Global objects - Static method: AbortSignal.timeout(delay)',
          text: 'AbortSignal.timeout(delay) returns a signal that aborts after delay milliseconds.',
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('does not follow another fragment from an already opened document', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Open and compare at least two official Node.js sources about AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.fetch',
        params: { url: 'https://nodejs.org/docs/latest/api/globals.html' },
        success: true,
        result: {
          finalUrl: 'https://nodejs.org/docs/latest/api/globals.html',
          title: 'Node.js globals introduction',
          text: 'Node.js global API overview.',
          links: [{
            text: 'AbortSignal.timeout',
            url: 'https://nodejs.org/docs/latest/api/globals.html#static-method-abortsignaltimeoutdelay',
          }],
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('does not create false relevance matches across unrelated token boundaries', () => {
    const planner = new AgentToolPlanner();
    const score = planner.webEvidenceScore(
      'Find the Node.js global fetch API and AbortSignal.timeout documentation.',
      {
        toolName: 'web.fetch',
        params: { url: 'https://earth.google.com/web/' },
        success: true,
        result: {
          finalUrl: 'https://earth.google.com/web/',
          title: 'Google Earth',
          text: 'Aw snap. See system requirements for more information.',
        },
      }
    );

    expect(score).toBe(0);
  });

  it('does not fetch unrelated low-relevance search results', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Find the Node.js global fetch API and AbortSignal.timeout documentation.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.search',
        params: { query: 'AbortSignal.timeout documentation' },
        success: true,
        result: {
          results: [
            { title: 'Google Earth', url: 'https://earth.google.com/web/', snippet: 'Explore the world.' },
            { title: 'Node.js Global objects', url: 'https://nodejs.org/docs/latest/api/globals.html#fetch', snippet: 'Global fetch API.' },
          ],
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({ params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' } }),
    ]);
  });

  it('scores multi-question follow-ups against the latest independent question', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: [
        'Verify each answer using public web sources.',
        '1. Who created the original Chipmunks characters?',
        '2. Who became British prime minister immediately after Arthur Balfour?',
      ].join('\n'),
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.search',
        params: { query: 'Who became British prime minister immediately after Arthur Balfour?' },
        success: true,
        result: {
          results: [
            {
              title: 'NEXT Official Site',
              url: 'https://www.next.co.uk/',
              snippet: 'British multinational clothing and homeware retailer.',
            },
            {
              title: 'Arthur Balfour – past Prime Ministers',
              url: 'https://www.gov.uk/government/history/past-prime-ministers/arthur-james-balfour',
              snippet: 'Balfour was succeeded as prime minister by Henry Campbell-Bannerman.',
            },
          ],
        },
      }],
    });

    expect(plans).toEqual([
      expect.objectContaining({
        params: {
          url: 'https://www.gov.uk/government/history/past-prime-ministers/arthur-james-balfour',
        },
      }),
    ]);
  });

  it('requires core API entities before following otherwise keyword-rich search results', () => {
    const plans = new AgentToolPlanner().planWebFollowUps({
      task: 'Use public web sources to compare Node.js AbortSignal.timeout and MDN AbortSignal.timeout.',
      bindings: [{ name: 'web.fetch', enabled: true }],
      maxFetches: 2,
      calls: [{
        toolName: 'web.search',
        params: { query: 'compare Node.js AbortSignal.timeout MDN' },
        success: true,
        result: {
          results: [
            {
              title: 'USE company information',
              url: 'https://www.use-ebisu.co.jp/',
              snippet: 'Public web information, current versions, availability, and comparison sources.',
            },
          ],
        },
      }],
    });

    expect(plans).toEqual([]);
  });

  it('stops web replanning after enough distinct relevant documents were opened', () => {
    const planner = new AgentToolPlanner();
    expect(planner.hasSufficientWebEvidence(
      'Open and compare at least two relevant public sources about Node.js fetch and AbortSignal.timeout.',
      [
        {
          toolName: 'web.fetch', params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' }, success: true,
          result: { finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#fetch', title: 'Node.js fetch API', text: 'Global fetch API.' },
        },
        {
          toolName: 'web.fetch', params: { url: 'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static' }, success: true,
          result: { finalUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static', title: 'AbortSignal timeout API', text: 'AbortSignal.timeout documentation.' },
        },
      ]
    )).toBe(true);
  });

  it('treats "both URLs" as a two-document evidence requirement', () => {
    const planner = new AgentToolPlanner();
    const oneDocument = [{
      toolName: 'web.fetch',
      params: { url: 'https://nodejs.org/docs/latest/api/globals.html#fetch' },
      success: true,
      result: {
        finalUrl: 'https://nodejs.org/docs/latest/api/globals.html#fetch',
        title: 'Node.js fetch API',
        text: 'AbortSignal.timeout and global fetch API documentation.',
      },
    }];

    expect(planner.hasSufficientWebEvidence(
      'Open both URLs and compare Node.js AbortSignal.timeout with MDN.',
      oneDocument
    )).toBe(false);
  });
});
