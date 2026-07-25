#!/usr/bin/env node

import { readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';

interface RunOptions {
  task?: string;
  taskFile?: string;
  workspace: string;
  sessionId: string;
  output?: string;
  json: boolean;
  budget?: number;
  wallClockMs?: number;
}

function usage(): string {
  return `Usage: roy-run [options]

Run one Roy task non-interactively and exit.

Options:
  --task <text>          Task text. Reads stdin when omitted.
  --task-file <path>     Read task text from a UTF-8 file.
  --workspace <path>     Workspace exposed to Roy (default: current directory).
  --session-id <id>      Stable session ID (default: generated).
  --budget <tokens>      Optional total token budget.
  --wall-clock-ms <ms>   Bound this invocation to an external wall-clock budget.
  --output <path>        Atomically write the complete JSON run artifact.
  --json                 Print the complete JSON run artifact to stdout.
  -h, --help             Show this help.

Workspace policy remains authoritative. In particular, unrestricted shell execution
must be enabled explicitly in <workspace>/.roy/config.json and should only be used
inside an isolated environment such as a benchmark container.`;
}

function optionValue(args: string[], index: number, name: string): string {
  const value = args[index + 1];
  if (!value || value.startsWith('--')) {
    throw new Error(`${name} requires a value`);
  }
  return value;
}

function parseArgs(args: string[]): RunOptions | null {
  if (args.includes('--help') || args.includes('-h')) return null;
  const options: RunOptions = {
    workspace: process.cwd(),
    sessionId: `run-${new Date().toISOString().replace(/[:.]/g, '-')}`,
    json: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--task') {
      options.task = optionValue(args, index, arg);
      index += 1;
    } else if (arg === '--task-file') {
      options.taskFile = optionValue(args, index, arg);
      index += 1;
    } else if (arg === '--workspace') {
      options.workspace = optionValue(args, index, arg);
      index += 1;
    } else if (arg === '--session-id') {
      options.sessionId = optionValue(args, index, arg);
      index += 1;
    } else if (arg === '--output') {
      options.output = optionValue(args, index, arg);
      index += 1;
    } else if (arg === '--budget') {
      const value = Number(optionValue(args, index, arg));
      if (!Number.isFinite(value) || value <= 0) throw new Error('--budget must be a positive number');
      options.budget = value;
      index += 1;
    } else if (arg === '--wall-clock-ms') {
      const value = Number(optionValue(args, index, arg));
      if (!Number.isSafeInteger(value) || value < 1_000) {
        throw new Error('--wall-clock-ms must be an integer of at least 1000');
      }
      options.wallClockMs = value;
      index += 1;
    } else if (arg === '--json') {
      options.json = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (options.task && options.taskFile) {
    throw new Error('Use only one of --task and --task-file');
  }
  return options;
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function loadTask(options: RunOptions, invocationCwd: string): Promise<string> {
  if (options.task) return options.task.trim();
  if (options.taskFile) {
    const taskPath = path.resolve(invocationCwd, options.taskFile);
    return (await readFile(taskPath, 'utf8')).trim();
  }
  if (process.stdin.isTTY) {
    throw new Error('Provide --task, --task-file, or pipe task text on stdin');
  }
  return (await readStdin()).trim();
}

async function writeJsonAtomically(filePath: string, value: unknown): Promise<void> {
  const target = path.resolve(filePath);
  const temporary = `${target}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await rename(temporary, target);
}

async function main(): Promise<void> {
  const invocationCwd = process.cwd();
  const options = parseArgs(process.argv.slice(2));
  if (!options) {
    process.stdout.write(`${usage()}\n`);
    return;
  }

  const task = await loadTask(options, invocationCwd);
  if (!task) throw new Error('Task must not be empty');
  const workspace = path.resolve(invocationCwd, options.workspace);
  const outputPath = options.output ? path.resolve(invocationCwd, options.output) : undefined;
  process.chdir(workspace);
  if (options.json) process.env.LOG_LEVEL ??= 'error';

  const { Runtime } = await import('../core/runtime/Runtime.js');
  const runtime = new Runtime();
  try {
    await runtime.initialize({
      agentName: 'Roy',
      agentGoal: 'Solve the supplied task using only authorized runtime capabilities.',
      sessionId: options.sessionId,
      workspaceCwd: workspace,
      fsmEnabled: true,
      budget: options.budget,
      wallClockLimitMs: options.wallClockMs,
    });
    try {
      const recovery = await runtime.handleUserTurnWithRecovery(task);
      const result = recovery.result;
      const correlationIds = new Set(recovery.correlationIds);
      const artifact = {
        schemaVersion: 1,
        status: 'completed',
        sessionId: options.sessionId,
        workspace,
        task,
        result,
        recovery: {
          attempts: recovery.attempts,
          recovered: recovery.recovered,
          correlationIds: recovery.correlationIds,
        },
        events: runtime.getEvents().filter(event =>
          (event.correlationId !== undefined && correlationIds.has(event.correlationId))
          || event.type.startsWith('runtime.transient_turn.')
          || event.type === 'runtime.wall_clock_limit.applied'
        ),
        messages: (await Promise.all(
          recovery.correlationIds.map(correlationId =>
            runtime.getMessages({ correlationId, limit: 10_000 })
          )
        )).flat().sort((left, right) => left.createdAt - right.createdAt),
        completedAt: new Date().toISOString(),
      };
      if (outputPath) await writeJsonAtomically(outputPath, artifact);
      process.stdout.write(options.json ? `${JSON.stringify(artifact)}\n` : `${result.finalResponse}\n`);
    } catch (error) {
      const allEvents = runtime.getEvents();
      const transientFailure = [...allEvents].reverse().find(event =>
        event.type === 'runtime.transient_turn.failed'
      );
      const correlationIds = [...new Set(
        allEvents
          .filter(event => event.type === 'root.turn.failed' && event.correlationId)
          .map(event => event.correlationId!)
      )];
      const correlationIdSet = new Set(correlationIds);
      const artifact = {
        schemaVersion: 1,
        status: 'failed',
        sessionId: options.sessionId,
        workspace,
        task,
        error: {
          name: error instanceof Error ? error.name : 'Error',
          message: error instanceof Error ? error.message : String(error),
          retryable: transientFailure?.data?.retryable === true,
          persistedState: transientFailure?.data?.persistedState === true,
        },
        recovery: {
          attempts: Number(transientFailure?.data?.attempt ?? correlationIds.length),
          recovered: false,
          correlationIds,
        },
        events: allEvents.filter(event =>
          (event.correlationId !== undefined && correlationIdSet.has(event.correlationId))
          || event.type.startsWith('runtime.transient_turn.')
          || event.type === 'runtime.wall_clock_limit.applied'
        ),
        messages: (await Promise.all(
          correlationIds.map(correlationId =>
            runtime.getMessages({ correlationId, limit: 10_000 })
          )
        )).flat().sort((left, right) => left.createdAt - right.createdAt),
        failedAt: new Date().toISOString(),
      };
      if (outputPath) await writeJsonAtomically(outputPath, artifact);
      if (options.json) process.stdout.write(`${JSON.stringify(artifact)}\n`);
      throw error;
    }
  } finally {
    await runtime.shutdown();
  }
}

main().catch(error => {
  process.stderr.write(`roy-run: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
