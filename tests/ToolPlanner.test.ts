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
      'python -m pytest -q',
    ]);
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
