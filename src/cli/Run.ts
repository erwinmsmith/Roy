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
    });
    const result = await runtime.handleUserTurn(task);
    const artifact = {
      schemaVersion: 1,
      sessionId: options.sessionId,
      workspace,
      task,
      result,
      events: runtime.getEvents().filter(event => event.correlationId === result.correlationId),
      messages: await runtime.getMessages({ correlationId: result.correlationId, limit: 10_000 }),
      completedAt: new Date().toISOString(),
    };
    if (outputPath) await writeJsonAtomically(outputPath, artifact);
    process.stdout.write(options.json ? `${JSON.stringify(artifact)}\n` : `${result.finalResponse}\n`);
  } finally {
    await runtime.shutdown();
  }
}

main().catch(error => {
  process.stderr.write(`roy-run: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
