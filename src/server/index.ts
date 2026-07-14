#!/usr/bin/env node
// Server entry point with Socket.IO - Unified via bootstrap

import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { AsyncLocalStorage } from 'node:async_hooks';
import { Server } from 'socket.io';
import { bootstrap, cleanup, type BootstrapContext } from '../bootstrap.js';
import { runtime as defaultRuntime } from '../core/runtime/Runtime.js';
import type Runtime from '../core/runtime/Runtime.js';
import { logger } from '../core/utils/logger.js';
import { RuntimeSessionPool } from './RuntimeSessionPool.js';

// ASCII Banner - compatible with all terminals
const SERVER_BANNER = `
+========================================================+
|                                                        |
|     ██████╗  ██████╗ ██╗   ██╗                     |
|     ██╔══██╗██╔═══██╗╚██╗ ██╔╝                     |
|     ██████╔╝██║   ██║ ╚████╔╝                      |
|     ██╔══██╗██║   ██║  ╚██╔╝                       |
|     ██║  ██║╚██████╔╝   ██║                        |
|     ╚═╝  ╚═╝ ╚═════╝    ╚═╝                        |
|                                                        |
| Theory of Mind based Autonomous Agent System           |
|                                                        |
+========================================================+
`;

// Color utilities for server output
function green(text: string): string {
  return `\x1b[32m${text}\x1b[0m`;
}

function yellow(text: string): string {
  return `\x1b[33m${text}\x1b[0m`;
}

function red(text: string): string {
  return `\x1b[31m${text}\x1b[0m`;
}

function cyan(text: string): string {
  return `\x1b[36m${text}\x1b[0m`;
}

function bold(text: string): string {
  return `\x1b[1m${text}\x1b[0m`;
}

function dim(text: string): string {
  return `\x1b[2m${text}\x1b[0m`;
}

// Graceful port handling
function tryListen(server: ReturnType<typeof createServer>, port: number, retries = 3): Promise<number> {
  return new Promise((resolve, reject) => {
    const attempt = (currentPort: number, remainingRetries: number) => {
      const onError = (err: NodeJS.ErrnoException) => {
        if (err.code === 'EADDRINUSE' && remainingRetries > 0) {
          logger.warn(`Port ${currentPort} in use, trying ${currentPort + 1}...`);
          attempt(currentPort + 1, remainingRetries - 1);
        } else {
          reject(err);
        }
      };
      server.once('error', onError);
      server.listen(currentPort, () => {
        server.off('error', onError);
        resolve(currentPort);
      });
    };
    attempt(port, retries);
  });
}

async function main(): Promise<void> {
  // Bootstrap the system
  let ctx: BootstrapContext;

  try {
    ctx = await bootstrap({
      agentName: 'Roy',
      agentGoal: 'You are Roy, the root agent of a Theory-of-Mind based autonomous agent system.',
      sessionId: 'server-main',
      fsmEnabled: true,
    });

    logger.info('Server Bootstrap complete');
  } catch (error) {
    logger.error('Bootstrap failed:', error);
    console.log(red('\n[ERROR] Failed to initialize server'));
    process.exit(1);
  }

  const PORT = ctx.config.server?.port ?? 3000;
  const runtimeStorage = new AsyncLocalStorage<Runtime>();
  const runtimePool = new RuntimeSessionPool({
    defaultSessionId: ctx.sessionId,
    defaultRuntime,
    defaultContext: ctx,
    workspaceCwd: process.cwd(),
  });
  const sessionSweepTimer = setInterval(() => {
    runtimePool.sweepIdle().then(expired => {
      if (expired.length > 0) logger.info(`Closed ${expired.length} idle runtime session(s)`);
    }).catch(error => logger.error('Runtime session sweep failed:', error));
  }, 60_000);
  sessionSweepTimer.unref();
  const runtime = new Proxy(defaultRuntime, {
    get(target, property) {
      const selected = runtimeStorage.getStore() ?? target;
      const value = Reflect.get(selected, property, selected) as unknown;
      return typeof value === 'function' ? value.bind(selected) : value;
    },
  });

  const app = express();
  const httpServer = createServer(app);
  app.use(express.json());
  app.use('/v1', async (req, res, next) => {
    if (req.method === 'DELETE' && req.path === '/runtime/session') {
      next();
      return;
    }
    const requestedSessionId = req.header('x-roy-session-id')
      ?? (typeof req.query.runtimeSessionId === 'string' ? req.query.runtimeSessionId : undefined)
      ?? req.body?.runtimeSessionId;
    try {
      const sessionId = runtimePool.normalizeSessionId(requestedSessionId);
      const selected = await runtimePool.get(sessionId);
      res.setHeader('X-Roy-Session-Id', sessionId);
      runtimeStorage.run(selected, next);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  // Socket.IO connection handling
  const io = new Server(httpServer, {
    cors: ctx.config.server?.cors ?? {
      origin: '*',
      methods: ['GET', 'POST'],
    },
  });

  io.on('connection', (socket) => {
    const sid = `socket-${socket.id.replace(/[^A-Za-z0-9._-]/g, '_')}`;
    logger.info(`Client connected: ${sid}`);

    socket.on('user_message', async (message: string) => {
      logger.info(`Received message from ${sid}: ${message.substring(0, 100)}...`);

      socket.emit('bot_response_start', 'Bot: Processing your request...');

      try {
        const selected = await runtimePool.get(sid);
        const result = await runtimeStorage.run(selected, () => runtime.handleUserTurn(message));
        socket.emit('bot_response_stream', result.finalResponse);
      } catch (error) {
        logger.error(`Error processing message:`, error);
        socket.emit('bot_response_stream', `Error: ${error}`);
      }
    });

    socket.on('disconnect', async () => {
      logger.info(`Client disconnected: ${sid}`);
      await runtimePool.close(sid);
    });
  });

  // Health check endpoint
  app.get('/health', (req, res) => {
    const state = runtime.getState();
    const fsmInfo = ctx.agent.getFSMInfo();
    res.json({
      status: 'ok',
      name: 'Roy',
      version: '0.1.0',
      mode: 'api',
      rootAgent: state.rootAgent,
      fsm: fsmInfo ? {
        state: fsmInfo.state,
        cost: fsmInfo.cost,
        budget: fsmInfo.budget,
        budgetLabel: fsmInfo.budget === null ? 'unlimited' : String(fsmInfo.budget),
      } : null,
      agents: state.agents,
      sessions: ctx.manager.listSessions(),
      capabilities: ctx.capabilities,
      budget: state.budget,
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/v1/status', (req, res) => {
    res.json(runtime.getState());
  });

  app.get('/v1/runtime/sessions', (_req, res) => {
    res.json({ sessions: runtimePool.list() });
  });

  app.delete('/v1/runtime/session', async (req, res) => {
    const sessionId = req.header('x-roy-session-id') ?? req.body?.runtimeSessionId;
    try {
      const normalized = runtimePool.normalizeSessionId(sessionId);
      const closed = await runtimePool.close(normalized);
      res.status(closed ? 200 : 400).json({ closed, sessionId: normalized });
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.post('/v1/chat', async (req, res) => {
    const input = req.body?.input ?? req.body?.message;
    if (typeof input !== 'string' || input.trim().length === 0) {
      res.status(400).json({ error: 'Expected body { "input": non-empty string }' });
      return;
    }

    try {
      res.json(await runtime.handleUserTurn(input));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/agents', (req, res) => {
    res.json(runtime.getState().agents);
  });

  app.get('/v1/agents/tree', (req, res) => {
    res.json(runtime.getAgentTree());
  });

  app.post('/v1/agents', async (req, res) => {
    try {
      const body = req.body ?? {};
      if (!body.parentId || !body.archetype || !body.description || typeof body.tomLevel !== 'number') {
        res.status(400).json({
          error: 'Expected body { parentId, archetype, tomLevel, description, name?, task?, tools?, budgetTokens?, systemPrompt? }',
        });
        return;
      }

      const agent = await runtime.spawnAgent(body);
      res.status(201).json(agent);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.post('/v1/agents/delegate', async (req, res) => {
    const body = req.body ?? {};
    const parentId = typeof body.parentId === 'string' ? body.parentId : 'root';
    if (typeof body.archetype !== 'string' || typeof body.task !== 'string' || body.task.trim().length === 0) {
      res.status(400).json({
        error: 'Expected body { archetype, task, parentId?, name?, role?, style?, tools?, skills?, budgetTokens?, reuse?, execution?, outputContract? }',
      });
      return;
    }

    try {
      const execution = await runtime.createAgentComputeNode({
        parentId,
        archetype: body.archetype,
        task: body.task,
        name: body.name,
        role: body.role,
        style: body.style,
        description: body.description,
        tools: body.tools,
        skills: body.skills,
        budgetTokens: body.budgetTokens,
        memoryScope: body.memoryScope,
        spawnPolicy: body.spawnPolicy,
        tomProfile: body.tomProfile,
        reuse: body.reuse,
        execution: body.execution,
        outputContract: body.outputContract,
      }, {
        agentId: parentId,
        sessionId: runtime.getState().sessionId,
        source: 'server',
      });
      res.status(201).json(execution);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/agents/:id', (req, res) => {
    const agent = runtime.getState().agents.find(item => item.identity.id === req.params.id);
    if (!agent) {
      res.status(404).json({ error: `Agent "${req.params.id}" not found` });
      return;
    }
    res.json(agent);
  });

  app.post('/v1/agents/:id/run', async (req, res) => {
    const task = req.body?.task;
    if (typeof task !== 'string' || task.trim().length === 0) {
      res.status(400).json({ error: 'Expected body { "task": non-empty string }' });
      return;
    }

    try {
      res.json(await runtime.runAgent(req.params.id, task));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const status = message.includes('not found') ? 404 : 400;
      res.status(status).json({ error: message });
    }
  });

  app.get('/v1/budget', (req, res) => {
    res.json(runtime.getBudgetState());
  });

  app.get('/v1/budget/market', (req, res) => {
    res.json(runtime.getBudgetMarketState());
  });

  app.get('/v1/cache/:kind', async (req, res) => {
    const kind = req.params.kind;
    if (kind === 'evolution') {
      const limit = req.query.limit ? Number(req.query.limit) : 50;
      res.json(await runtime.getEvolutionHistory(Number.isFinite(limit) && limit > 0 ? limit : 50));
      return;
    }
    if (kind !== 'agents' && kind !== 'delegations' && kind !== 'teams') {
      res.status(400).json({ error: 'Cache kind must be agents, delegations, teams, or evolution' });
      return;
    }
    res.json(await runtime.getCachePatterns(kind));
  });

  app.post('/v1/budget', (req, res) => {
    const limitTokens = req.body?.limitTokens;
    if (limitTokens === null) {
      res.json(runtime.setBudget(null));
      return;
    }
    if (typeof limitTokens !== 'number' || !Number.isFinite(limitTokens) || limitTokens <= 0) {
      res.status(400).json({ error: 'Expected body { "limitTokens": positive number | null }' });
      return;
    }
    res.json(runtime.setBudget(limitTokens));
  });

  app.get('/v1/events', (req, res) => {
    const agentId = typeof req.query.agentId === 'string' ? req.query.agentId : undefined;
    const type = typeof req.query.type === 'string' ? req.query.type : undefined;
    const limit = req.query.limit ? Number(req.query.limit) : 50;
    const events = runtime.getEvents()
      .filter(event => !agentId || event.agentId === agentId)
      .filter(event => !type || event.type === type)
      .slice(-(Number.isFinite(limit) && limit > 0 ? limit : 50));
    res.json(events);
  });

  app.get('/v1/teams', (req, res) => {
    res.json(runtime.getTeams());
  });

  app.get('/v1/teams/tree', (req, res) => {
    res.json(runtime.getTeamActorTree());
  });

  app.post('/v1/teams', async (req, res) => {
    const body = req.body ?? {};
    if (typeof body.name !== 'string' || typeof body.description !== 'string') {
      res.status(400).json({ error: 'Expected body { name, description, parentAgentId?, tomLevel?, task?, members?, executionPolicy? }' });
      return;
    }
    try {
      res.status(201).json(await runtime.spawnTeam(body));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/teams/:id', (req, res) => {
    const team = runtime.getTeamTree(req.params.id);
    if (!team) {
      res.status(404).json({ error: `Team "${req.params.id}" not found` });
      return;
    }
    res.json(team);
  });

  app.post('/v1/teams/:id/agents', async (req, res) => {
    const body = req.body ?? {};
    if (typeof body.archetype !== 'string' || typeof body.task !== 'string') {
      res.status(400).json({ error: 'Expected body { archetype, task, name?, role?, tools?, skills?, budgetTokens?, tomLevel?, lead? }' });
      return;
    }
    try {
      res.status(201).json(await runtime.spawnAgentIntoTeam(req.params.id, body));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.post('/v1/teams/:id/run', async (req, res) => {
    const task = req.body?.task;
    if (typeof task !== 'string' || task.trim().length === 0) {
      res.status(400).json({ error: 'Expected body { "task": non-empty string }' });
      return;
    }
    try {
      res.json(await runtime.runTeam(req.params.id, task));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/tools/approvals', (req, res) => {
    const status = typeof req.query.status === 'string' ? req.query.status : undefined;
    const normalized = status === 'pending' || status === 'approved' || status === 'denied' ? status : undefined;
    res.json(runtime.getToolApprovals(normalized));
  });

  app.post('/v1/tools/approvals/:id', async (req, res) => {
    const decision = req.body?.decision;
    if (decision !== 'approved' && decision !== 'denied') {
      res.status(400).json({ error: 'Expected body { "decision": "approved" | "denied" }' });
      return;
    }
    const approval = await runtime.resolveToolApproval(req.params.id, decision);
    if (!approval) {
      res.status(404).json({ error: `Pending tool approval "${req.params.id}" not found` });
      return;
    }
    res.json(approval);
  });

  app.post('/v1/tools/:name/execute', async (req, res) => {
    const agentId = req.body?.agentId ?? 'root';
    try {
      const result = await runtime.executeToolForAgent(agentId, req.params.name, req.body?.params ?? {}, {
        reason: req.body?.reason,
        approvalId: req.body?.approvalId,
        correlationId: req.body?.correlationId,
      });
      res.status(result.metadata?.pendingApproval ? 202 : result.success ? 200 : 400).json(result);
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/messages', async (req, res) => {
    const correlationId = typeof req.query.correlationId === 'string' ? req.query.correlationId : undefined;
    const limit = req.query.limit ? Number(req.query.limit) : 50;
    res.json(await runtime.getMessages({ correlationId, limit: Number.isFinite(limit) && limit > 0 ? limit : 50 }));
  });

  app.get('/v1/queue', async (req, res) => {
    const limit = req.query.limit ? Number(req.query.limit) : 20;
    res.json(await runtime.getQueueState(Number.isFinite(limit) && limit > 0 ? limit : 20));
  });

  app.get('/v1/memory', async (req, res) => {
    res.json(await runtime.getMemoryState());
  });

  app.get('/v1/memory/root', async (req, res) => {
    res.json(await runtime.loadRootMemoryContext());
  });

  app.post('/v1/context/render', async (req, res) => {
    const agentKey = req.body?.agentKey ?? 'roy';
    try {
      res.json(await runtime.renderAgentContext({
        agentKey,
        agentId: req.body?.agentId,
        role: req.body?.role,
        parentId: req.body?.parentId,
        task: req.body?.task,
      }));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  app.get('/v1/traces', async (req, res) => {
    res.json(await runtime.listTraces());
  });

  app.get('/v1/traces/latest', async (req, res) => {
    const limit = req.query.limit ? Number(req.query.limit) : 50;
    res.json(await runtime.readTrace('latest', Number.isFinite(limit) && limit > 0 ? limit : 50));
  });

  app.get('/v1/conversation', async (req, res) => {
    const limit = req.query.limit ? Number(req.query.limit) : 50;
    const sessionId = typeof req.query.sessionId === 'string' ? req.query.sessionId : undefined;
    res.json(await runtime.getConversation(sessionId, Number.isFinite(limit) && limit > 0 ? limit : 50));
  });

  app.get('/v1/conversation/sessions', async (req, res) => {
    res.json(await runtime.listConversationSessions());
  });

  app.post('/v1/conversation/import', async (req, res) => {
    const filePath = req.body?.path;
    const sessionId = req.body?.sessionId;
    if (typeof filePath !== 'string' || filePath.trim().length === 0) {
      res.status(400).json({ error: 'Expected body { "path": string, "sessionId"?: string }' });
      return;
    }
    try {
      res.json(await runtime.importConversation(filePath, typeof sessionId === 'string' ? sessionId : undefined));
    } catch (error) {
      res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
    }
  });

  // Root endpoint
  app.get('/', (req, res) => {
    res.json({
      name: 'Roy',
      description: 'Theory of Mind based autonomous derived MAS',
      version: '0.1.0',
      mode: 'api',
      endpoints: {
        health: '/health',
        websocket: 'Connect via Socket.IO',
      },
    });
  });

  // Start server with graceful port handling
  tryListen(httpServer, PORT)
    .then((actualPort) => {
      console.log(green(SERVER_BANNER));
      console.log(dim('='.repeat(60)));
      console.log(green('[OK]') + ' Server running on port ' + bold(String(actualPort)));
      console.log('  WebSocket: ' + cyan(`ws://localhost:${actualPort}`));
      console.log('  Health:    ' + cyan(`http://localhost:${actualPort}/health`));
      console.log(dim('='.repeat(60)));
      console.log('  Agent: ' + cyan(ctx.agent.name));
      console.log('  FSM: ' + cyan(ctx.fsm.getStateName()));
      console.log(dim('='.repeat(60)));
      console.log('');

      if (actualPort !== PORT) {
        console.log(yellow('[WARN] Port ' + PORT + ' was in use, started on ' + actualPort));
      }
    })
    .catch((err) => {
      logger.error('Failed to start server:', err);
      console.error(red('[ERROR] Failed to start server:') + ' ' + err.message);
      process.exit(1);
  });

  // Graceful shutdown
  let shuttingDown = false;
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    clearInterval(sessionSweepTimer);
    console.log(yellow('\nShutting down...'));
    await Promise.allSettled([runtimePool.shutdown(), cleanup(ctx)]);
    io.close();
    httpServer.close(() => {
      console.log(green('[OK] Server closed'));
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch(err => {
  logger.error('Unhandled error:', err);
  process.exit(1);
});
